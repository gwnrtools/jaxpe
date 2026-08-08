"""Real numerical validation of IMRPhenomD against LALSuite's own IMRPhenomD.

This is the tolerance-calibration reference point for the IMRPhenomT/IMRPhenomTHM
reimplementation (see docs/constants.md and the IMRPhenomT plan): phenomd.py is a careful,
already-correct transcription of Khan et al. 2016 (QNM tables, Erad/FinalSpin fits, the full
coefficient table, PN phase/amplitude, delta-coefficient region matching), but until now nothing
ever checked it numerically against LAL -- test_gw.py only compares jaxpe against itself
(batched vs serial). PhenomD is natively frequency-domain in both LAL and jaxpe, so
SimInspiralChooseFDWaveform is used directly (avoids the TD/FD round-trip artifacts a
TD-generator comparison would introduce).
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from jaxpe.gw import IMRPhenomD, aligo_zdhp_psd, mismatch

lal = pytest.importorskip("lal")
lalsim = pytest.importorskip("lalsimulation")

F_MIN = 20.0
F_MAX = 1024.0
DELTA_F = 0.05
F_REF = 20.0

# (m1, m2) [Msun], (s1z, s2z), distance [Mpc], inclination [rad] -- spans mass ratio 1:1 to
# 5:1, spins including high-spin/high-mass-ratio corners, inclination across [0, pi/2, pi],
# and a range of distances (mismatch should be distance-independent, included for coverage).
PARAM_GRID = [
    (30.0, 20.0, 0.5, -0.3, 100.0, 0.5),
    (30.0, 30.0, 0.0, 0.0, 500.0, 0.0),
    (10.0, 10.0, 0.0, 0.0, 300.0, 1.0),
    (50.0, 10.0, 0.8, 0.8, 800.0, 0.9),
    (60.0, 55.0, -0.5, 0.2, 1000.0, 1.57),
    (15.0, 7.0, 0.3, -0.6, 200.0, 2.5),
]

# Calibrated from this exact grid: worst observed mismatch ~4e-9 (float64). 1e-6 leaves ample
# headroom above real cross-code floating-point/interpolation-grid noise without being a
# hair-trigger threshold -- matches the tolerance already used for internal PSD-weighted
# mismatch checks elsewhere in the repo (tests/test_gw.py::test_batched_matches_serial).
MISMATCH_TOLERANCE = 1e-6


def _lal_phenomd_fd(m1, m2, s1z, s2z, distance_mpc, inclination):
    hp, hc = lalsim.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI,
        m2 * lal.MSUN_SI,
        0.0,
        0.0,
        s1z,
        0.0,
        0.0,
        s2z,
        distance_mpc * 1e6 * lal.PC_SI,
        inclination,
        0.0,  # phiRef
        0.0,  # longAscNodes
        0.0,  # eccentricity
        0.0,  # meanPerAno
        DELTA_F,
        F_MIN,
        F_MAX,
        F_REF,
        lal.CreateDict(),
        lalsim.IMRPhenomD,
    )
    freqs = np.arange(hp.data.length) * hp.deltaF
    return freqs, jnp.asarray(hp.data.data), jnp.asarray(hc.data.data)


def _jax_phenomd_fd(m1, m2, s1z, s2z, distance_mpc, inclination, freqs):
    q = min(m1, m2) / max(m1, m2)
    mtot = m1 + m2
    eta = q / (1.0 + q) ** 2
    mc = mtot * eta**0.6
    params = dict(
        chirp_mass=mc,
        mass_ratio=q,
        spin1z=s1z,
        spin2z=s2z,
        luminosity_distance=distance_mpc,
        geocent_time=0.0,
        phase=0.0,
        inclination=inclination,
    )
    model = IMRPhenomD(f_ref=F_REF)
    return model(params, jnp.asarray(freqs))


@pytest.mark.parametrize("m1, m2, s1z, s2z, distance_mpc, inclination", PARAM_GRID)
def test_phenomd_matches_lalsuite(m1, m2, s1z, s2z, distance_mpc, inclination):
    freqs, hp_lal, hc_lal = _lal_phenomd_fd(m1, m2, s1z, s2z, distance_mpc, inclination)
    hp_jax, hc_jax = _jax_phenomd_fd(m1, m2, s1z, s2z, distance_mpc, inclination, freqs)

    psd = jnp.asarray(aligo_zdhp_psd(freqs, f_low=F_MIN))
    band = jnp.asarray((freqs >= F_MIN) & (freqs <= F_MAX))

    mm_p = float(mismatch(hp_jax, hp_lal, psd, DELTA_F, band=band))
    mm_c = float(mismatch(hc_jax, hc_lal, psd, DELTA_F, band=band))

    assert mm_p < MISMATCH_TOLERANCE, f"h_plus mismatch {mm_p:.3e} vs LAL IMRPhenomD"
    assert mm_c < MISMATCH_TOLERANCE, f"h_cross mismatch {mm_c:.3e} vs LAL IMRPhenomD"


def test_phenomd_matches_lalsuite_across_grid_reports_worst_case():
    """Not a stricter assertion than the parametrized test above -- prints the full grid's
    mismatch table so the calibration this module's MISMATCH_TOLERANCE is based on is visible
    in test output, not just asserted blindly."""
    worst = 0.0
    for m1, m2, s1z, s2z, distance_mpc, inclination in PARAM_GRID:
        freqs, hp_lal, _ = _lal_phenomd_fd(m1, m2, s1z, s2z, distance_mpc, inclination)
        hp_jax, _ = _jax_phenomd_fd(m1, m2, s1z, s2z, distance_mpc, inclination, freqs)
        psd = jnp.asarray(aligo_zdhp_psd(freqs, f_low=F_MIN))
        band = jnp.asarray((freqs >= F_MIN) & (freqs <= F_MAX))
        mm = float(mismatch(hp_jax, hp_lal, psd, DELTA_F, band=band))
        print(
            f"  m1={m1:5.1f} m2={m2:5.1f} s1z={s1z:+.2f} s2z={s2z:+.2f} "
            f"iota={inclination:.2f} -> mismatch {mm:.3e}"
        )
        worst = max(worst, mm)
    assert worst < MISMATCH_TOLERANCE, f"worst-case grid mismatch {worst:.3e}"
