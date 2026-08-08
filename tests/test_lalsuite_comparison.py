import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import lalsimulation as lalsim
import lal
import pytest

from jaxpe.gw import aligo_zdhp_psd, mismatch, rfft_freqs, td_to_fd
from jaxpe.gw.cbc_models.phenomthm import IMRPhenomTHM

F_MIN = 20.0
# Not yet real physics (docs/constants.md: phenomthm.py's t_meco/t_ring/amplitude-envelope
# construction is still a fabricated placeholder). This is expected to fail until the
# IMRPhenomTHM reimplementation (Phase 2 of the IMRPhenomT plan) lands -- strict=True turns a
# surprise pass into a loud error, forcing the xfail marker to be removed once that happens
# rather than the improvement going unnoticed.
@pytest.mark.xfail(
    reason="phenomthm.py is still a placeholder (docs/constants.md); real physics lands in "
    "the IMRPhenomTHM reimplementation's Phase 2.",
    strict=True,
)
def test_compare_with_lalsuite():
    # Define parameters for a reference binary
    m1 = 30.0 * lal.MSUN_SI
    m2 = 20.0 * lal.MSUN_SI
    s1x, s1y, s1z = 0.0, 0.0, 0.5
    s2x, s2y, s2z = 0.0, 0.0, -0.3
    distance = 100.0 * 1e6 * lal.PC_SI
    inclination = 0.5
    phiRef = 0.0
    longAscNodes = 0.0
    eccentricity = 0.0
    meanPerAno = 0.0
    deltaT = 1.0 / 4096.0
    f_min = 20.0
    f_ref = 20.0

    # LALSuite parameters
    params_dict = lal.CreateDict()

    approx = lalsim.IMRPhenomTHM

    # Generate LALSuite waveform
    hp_lal, hc_lal = lalsim.SimInspiralChooseTDWaveform(
        m1,
        m2,
        s1x,
        s1y,
        s1z,
        s2x,
        s2y,
        s2z,
        distance,
        inclination,
        phiRef,
        longAscNodes,
        eccentricity,
        meanPerAno,
        deltaT,
        f_min,
        f_ref,
        params_dict,
        approx,
    )

    # The LALSuite waveform is a TimeSeries. Let's get the time array.
    t_lal = jnp.arange(hp_lal.data.length) * hp_lal.deltaT + float(hp_lal.epoch)

    # Our JAX implementation
    model = IMRPhenomTHM(f_ref=f_ref)

    # JAX parameters dict. mass_ratio follows jaxpe's convention (q = m_smaller/m_larger <= 1,
    # see phenomt.py/phenomthm.py) -- eta/chirp_mass are invariant under q <-> 1/q so this
    # didn't previously affect the shape-only check, but a real physics comparison needs the
    # un-clipped, correctly-ordered value (phenomt.py/phenomthm.py clip mass_ratio to [0.01,1]).
    M_total = (m1 + m2) / lal.MSUN_SI
    q = min(m1, m2) / max(m1, m2)
    mc = M_total * (q / (1.0 + q) ** 2) ** (3.0 / 5.0)

    jax_params = {
        "chirp_mass": mc,
        "mass_ratio": q,
        "spin1z": s1z,
        "spin2z": s2z,
        "luminosity_distance": distance / (1e6 * lal.PC_SI),
        "phase": phiRef,
        "inclination": inclination,
        "geocent_time": 0.0,  # Shift it to align peaks if necessary
    }

    hp_jax, hc_jax = model(jax_params, t_lal)

    assert hp_jax.shape == t_lal.shape
    assert hc_jax.shape == t_lal.shape

    # Real numerical comparison: FFT both (same finite time grid, so edge/leakage effects are
    # shared) to frequency domain and compute the PSD-weighted mismatch (jaxpe.gw.mismatch),
    # the same measure used everywhere else in the repo for "are these the same physical
    # signal" checks -- not just "do the arrays have the same shape."
    dt = float(hp_lal.deltaT)
    n = hp_lal.data.length
    freqs = jnp.asarray(rfft_freqs(n, dt))
    hp_lal_fd = td_to_fd(jnp.asarray(hp_lal.data.data), dt)
    hp_jax_fd = td_to_fd(hp_jax, dt)

    psd = jnp.asarray(aligo_zdhp_psd(freqs, f_low=F_MIN))
    band = jnp.asarray(freqs >= F_MIN)
    mm = float(mismatch(hp_jax_fd, hp_lal_fd, psd, float(freqs[1] - freqs[0]), band=band))
    print(f"IMRPhenomTHM vs LALSuite mismatch: {mm:.3e}")

    assert mm < 1e-6, f"IMRPhenomTHM h_plus mismatch {mm:.3e} vs LALSuite"


if __name__ == "__main__":
    test_compare_with_lalsuite()
