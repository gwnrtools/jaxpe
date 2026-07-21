"""Convention tests for the external-model mode wrappers (design task 3.1).

"This is where silent bugs live": a wrapper can return plausible-looking modes with a
wrong harmonic convention and bias every downstream posterior. Each wrapper is tested
two ways:

* **Contract** -- ``ModesData`` on exactly the analysis grid, coalescence aligned to
  ``t_ref``, negative-m modes consistent with the planar reflection symmetry.
* **Round trip** -- rebuild ``h(iota) = sum_lm h_lm {}_{-2}Y_{lm}(iota, 0)`` from the
  wrapper's modes and compare against the *same engine's own polarizations* at that
  inclination, placed on the same grid. The comparison is done up to one global complex
  phase: Route B marginalizes the overall phase analytically, so the load-bearing
  convention is the inclination weighting of the harmonics (the (1+cos iota)^2/4 vs
  (1-cos iota)^2/4 split between +-m), not the phi <-> phiRef offset, which drops out.
  The taper window is excluded (the direct polarizations are untapered).
"""

import numpy as np
import pytest

from jaxpe.gw.external_models import ModesData
from jaxpe.gw.harmonics import spin_weighted_ylm

# analysis segment shared by all tests
SR = 4096.0
DURATION = 8.0
T0 = 1126259460.0
TIMES = T0 + np.arange(int(DURATION * SR)) / SR
T_REF = T0 + 6.0
PARAMS = dict(mass_1=30.0, mass_2=25.0, spin_1z=0.2, spin_2z=-0.1)
F_LOW = 20.0
D_REF = 500.0


def _reconstruct(md: ModesData, iota: float) -> np.ndarray:
    h = np.zeros(len(md.times), dtype=np.complex128)
    for (l, m), hlm in md.modes.items():
        h = h + hlm * complex(spin_weighted_ylm(iota, 0.0, l, m))
    return h


def _phase_optimized_rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 residual of ``a`` vs ``b`` after the optimal global phase rotation."""
    inner = np.vdot(b, a)
    a_rot = a * np.exp(-1j * np.angle(inner))
    return float(np.linalg.norm(a_rot - b) / np.linalg.norm(b))


def _contract_checks(md: ModesData, times, t_ref):
    np.testing.assert_array_equal(md.times, times)
    assert md.t_ref == t_ref
    # reflection symmetry: h_{l,-m} = (-1)^l conj(h_{lm})
    for (l, m), h in md.modes.items():
        if m > 0:
            assert (l, -m) in md.modes
            np.testing.assert_allclose(
                md.modes[(l, -m)], (-1.0) ** l * np.conj(h), rtol=0, atol=1e-30
            )
    # coalescence (peak of the dominant mode) lands at t_ref to within a few samples
    h22 = md.modes[(2, 2)]
    t_peak = times[int(np.argmax(np.abs(h22)))]
    assert abs(t_peak - t_ref) < 5e-3
    # the placed mode is zero-padded, not wrapped
    assert h22[0] == 0.0 and h22[-1] == 0.0


def _compare_region(md: ModesData, taper_seconds: float):
    """Sample slice where wrapper and direct polarizations must agree: from the end of
    the taper window to the end of the ringdown."""
    h22 = np.abs(md.modes[(2, 2)])
    nz = np.nonzero(h22)[0]
    start = nz[0] + int(round(taper_seconds * SR)) + 1
    stop = nz[-1] + 1
    return slice(start, stop)


# ------------------------------------------------------------------ TEOBResumS via LAL
@pytest.fixture(scope="module")
def teob_model():
    pytest.importorskip("lalsimulation")
    from jaxpe.gw.external_models import TEOBResumSModel

    return TEOBResumSModel(TIMES, T_REF, d_ref_mpc=D_REF, f_low=F_LOW)


def test_teobresums_contract(teob_model):
    md = teob_model(PARAMS)
    assert set(md.modes) == {(2, 2), (2, -2)}
    assert md.d_ref_mpc == D_REF
    _contract_checks(md, TIMES, T_REF)


@pytest.mark.parametrize("iota", [0.7, 1.3])
def test_teobresums_round_trip(teob_model, iota):
    import lal
    import lalsimulation as ls

    md = teob_model(PARAMS)
    h_recon = _reconstruct(md, iota)

    hp, hc = ls.SimInspiralChooseTDWaveform(
        PARAMS["mass_1"] * lal.MSUN_SI,
        PARAMS["mass_2"] * lal.MSUN_SI,
        0.0,
        0.0,
        PARAMS["spin_1z"],
        0.0,
        0.0,
        PARAMS["spin_2z"],
        D_REF * 1e6 * lal.PC_SI,
        iota,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 / SR,
        F_LOW,
        F_LOW,
        None,
        ls.GetApproximantFromString("TEOBResumS"),
    )
    from jaxpe.gw.external_models.base import place_modes_on_grid

    epoch = float(hp.epoch.gpsSeconds) + 1e-9 * float(hp.epoch.gpsNanoSeconds)
    h_direct = hp.data.data - 1j * hc.data.data
    t_rel = epoch + np.arange(len(h_direct)) / SR
    placed = place_modes_on_grid({(0, 0): h_direct}, t_rel, TIMES, T_REF)[(0, 0)]

    sl = _compare_region(md, teob_model.taper_seconds)
    err = _phase_optimized_rel_l2(h_recon[sl], placed[sl])
    assert err < 1e-6, f"inclination-weighting mismatch at iota={iota}: {err:.2e}"


# ------------------------------------------------------------------ pyseobnr (SEOBNRv5)
@pytest.fixture(scope="module")
def seob_model():
    pytest.importorskip("pyseobnr")
    from jaxpe.gw.external_models import PySEOBNRModel

    # dominant-mode config: the round trip is then exact up to one global phase for any
    # azimuth convention, isolating the inclination weighting under test
    return PySEOBNRModel(
        TIMES, T_REF, d_ref_mpc=D_REF, f_low=F_LOW, mode_array=[(2, 2)]
    )


def test_pyseobnr_contract(seob_model):
    md = seob_model(PARAMS)
    assert set(md.modes) == {(2, 2), (2, -2)}
    _contract_checks(md, TIMES, T_REF)


def test_pyseobnr_rejects_precession(seob_model):
    with pytest.raises(ValueError):
        seob_model(dict(PARAMS, spin_1x=0.3))


@pytest.mark.parametrize("iota", [0.7, 1.3])
def test_pyseobnr_round_trip(seob_model, iota):
    from pyseobnr.generate_waveform import GenerateWaveform

    md = seob_model(PARAMS)
    h_recon = _reconstruct(md, iota)

    hp, hc = GenerateWaveform(
        dict(
            mass1=PARAMS["mass_1"],
            mass2=PARAMS["mass_2"],
            spin1x=0.0,
            spin1y=0.0,
            spin1z=PARAMS["spin_1z"],
            spin2x=0.0,
            spin2y=0.0,
            spin2z=PARAMS["spin_2z"],
            deltaT=1.0 / SR,
            f22_start=F_LOW,
            f_ref=F_LOW,
            distance=D_REF,
            inclination=iota,
            phi_ref=0.0,
            approximant="SEOBNRv5HM",
            mode_array=[(2, 2)],
        )
    ).generate_td_polarizations()  # LAL REAL8TimeSeries pair (as used in examples/08)
    from jaxpe.gw.external_models.base import place_modes_on_grid

    epoch = float(hp.epoch.gpsSeconds) + 1e-9 * float(hp.epoch.gpsNanoSeconds)
    h_direct = np.asarray(hp.data.data) - 1j * np.asarray(hc.data.data)
    t_rel = epoch + np.arange(len(h_direct)) / SR
    placed = place_modes_on_grid({(0, 0): h_direct}, t_rel, TIMES, T_REF)[(0, 0)]

    sl = _compare_region(md, seob_model.taper_seconds)
    err = _phase_optimized_rel_l2(h_recon[sl], placed[sl])
    assert err < 1e-6, f"inclination-weighting mismatch at iota={iota}: {err:.2e}"
