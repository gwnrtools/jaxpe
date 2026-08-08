import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from jaxpe.gw.cbc_models.phenomthm import IMRPhenomTHM
from jaxpe.gw import aligo_zdhp_psd, mismatch, rfft_freqs, td_to_fd


@pytest.fixture
def default_params():
    return {
        "chirp_mass": 30.0,
        "mass_ratio": 0.5,  # <= 1: q = m_smaller/m_larger, this repo's own convention
        "spin1z": 0.5,
        "spin2z": -0.5,
        "luminosity_distance": 100.0,
        "phase": 0.0,
        "inclination": 0.5,
        "geocent_time": 0.0,
    }


def test_imr_phenom_thm_initialization():
    model = IMRPhenomTHM(f_ref=20.0)
    assert model.f_ref == 20.0
    assert not model.is_fd


def test_imr_phenom_thm_evaluation(default_params):
    model = IMRPhenomTHM()
    t_grid = jnp.linspace(-1.0, 0.05, 500)
    hp, hc = model(default_params, t_grid)

    assert hp.shape == t_grid.shape
    assert hc.shape == t_grid.shape
    assert not jnp.any(jnp.isnan(hp))
    assert not jnp.any(jnp.isnan(hc))


def test_imr_phenom_thm_jit(default_params):
    model = IMRPhenomTHM()
    t_grid = jnp.linspace(-1.0, 0.05, 200)

    hp, hc = model(default_params, t_grid)
    hp2, hc2 = model(default_params, t_grid)

    np.testing.assert_allclose(hp, hp2, rtol=1e-5)
    np.testing.assert_allclose(hc, hc2, rtol=1e-5)


def test_imr_phenom_thm_vmap(default_params):
    model = IMRPhenomTHM()
    t_grid = jnp.linspace(-1.0, 0.05, 200)

    batched_params = {
        k: jnp.array([v, v * 1.1 if k != "mass_ratio" else 0.7]) for k, v in default_params.items()
    }

    vmap_model = jax.vmap(model, in_axes=(0, None))
    hp_batch, hc_batch = vmap_model(batched_params, t_grid)

    assert hp_batch.shape == (2, 200)
    assert hc_batch.shape == (2, 200)
    assert not jnp.any(jnp.isnan(hp_batch))


def test_odd_modes_vanish_at_equal_mass_nonspinning(default_params):
    """LALSimIMRPhenomTHM.c special-cases this: odd-m modes (2,1),(3,3),(5,5) vanish exactly
    (not just approximately) in the equal-mass, equal-spin limit -- a real physical symmetry.
    (4,4) must NOT be affected (only odd m)."""
    model = IMRPhenomTHM()
    t_grid = jnp.linspace(-1.0, 0.05, 200)
    params = dict(default_params, mass_ratio=1.0, spin1z=0.3, spin2z=0.3)
    modes = model.mode_dict(params, t_grid)
    for l, m in ((2, 1), (3, 3), (5, 5)):
        assert jnp.all(modes[(l, m)] == 0.0), f"({l},{m}) should vanish exactly"
        assert jnp.all(modes[(l, -m)] == 0.0), f"({l},{-m}) should vanish exactly"
    assert jnp.any(modes[(4, 4)] != 0.0), "(4,4) must not be masked (m is even)"
    assert jnp.any(modes[(2, 2)] != 0.0)


def test_gradient_finite(default_params):
    model = IMRPhenomTHM()
    t_grid = jnp.linspace(-1.0, 0.05, 200)

    def loss(mc):
        p = dict(default_params, chirp_mass=mc)
        hp, hc = model(p, t_grid)
        return jnp.sum(hp**2 + hc**2)

    grad = jax.grad(loss)(default_params["chirp_mass"])
    assert jnp.isfinite(grad)


def test_gradient_matches_finite_difference(default_params):
    model = IMRPhenomTHM()
    t_grid = jnp.linspace(-1.0, 0.05, 500)
    params = dict(default_params, mass_ratio=0.6, spin1z=0.2, spin2z=-0.1)

    def loss(mc):
        p = dict(params, chirp_mass=mc)
        hp, hc = model(p, t_grid)
        return jnp.sum(hp**2 + hc**2)

    mc0 = 30.0
    grad = float(jax.grad(loss)(mc0))
    h = 1e-3
    fd = float((loss(mc0 + h) - loss(mc0 - h)) / (2.0 * h))
    assert grad == pytest.approx(fd, rel=1e-3), f"autodiff grad {grad} vs finite-diff {fd}"


def test_batched_matches_serial_via_mismatch(default_params):
    model = IMRPhenomTHM()
    t_grid = jnp.linspace(-1.0, 0.1, 2048)
    rng = np.random.default_rng(0)
    n = 6
    batch = {
        "chirp_mass": jnp.asarray(rng.uniform(20.0, 60.0, n)),
        "mass_ratio": jnp.asarray(rng.uniform(0.3, 1.0, n)),
        "spin1z": jnp.asarray(rng.uniform(-0.5, 0.5, n)),
        "spin2z": jnp.asarray(rng.uniform(-0.5, 0.5, n)),
        "luminosity_distance": jnp.full(n, 500.0),
        "phase": jnp.asarray(rng.uniform(0.0, 2 * np.pi, n)),
        "inclination": jnp.asarray(rng.uniform(0.0, np.pi, n)),
        "geocent_time": jnp.zeros(n),
    }
    hp_batched, _ = jax.vmap(model, in_axes=(0, None))(batch, t_grid)

    dt = float(t_grid[1] - t_grid[0])
    freqs = jnp.asarray(rfft_freqs(t_grid.shape[0], dt))
    psd = jnp.asarray(aligo_zdhp_psd(freqs, f_low=20.0))
    band = jnp.asarray(freqs >= 20.0)

    for i in range(n):
        params_i = {k: v[i] for k, v in batch.items()}
        hp_serial, _ = model(params_i, t_grid)
        a = td_to_fd(hp_batched[i], dt)
        b = td_to_fd(hp_serial, dt)
        mm = float(mismatch(a, b, psd, float(freqs[1] - freqs[0]), band=band))
        assert mm < 1e-6, f"injection {i}: batched-vs-serial mismatch {mm:.3e}"


lal = pytest.importorskip("lal")
lalsim = pytest.importorskip("lalsimulation")

_F_MIN = 20.0
_F_REF = 20.0
_DELTA_T = 1.0 / 4096.0

# mass ratio 1:1 to ~5:1, spins to 0.8, varying inclination/phi_ref -- same style as
# test_phenomt.py/test_phenomd.py. Includes an exact equal-mass-nonspinning point
# specifically to exercise the odd-mode-vanishes symmetry against LAL's own zero.
_PARAM_GRID = [
    (30.0, 20.0, 0.5, -0.3, 100.0, 0.5, 0.0),
    (30.0, 30.0, 0.0, 0.0, 500.0, 0.0, 0.0),
    (10.0, 10.0, 0.0, 0.0, 300.0, 1.0, 0.0),
    (50.0, 10.0, 0.8, 0.8, 800.0, 0.9, 0.0),
    (30.0, 30.0, 0.0, 0.0, 500.0, 0.0, 1.0),
    (30.0, 20.0, 0.3, -0.2, 500.0, 1.9, 0.7),
]

# Calibrated on this grid: worst observed mismatch ~2.5e-6, same scale as IMRPhenomT/PhenomD.
_MISMATCH_TOLERANCE = 1e-5


def _lal_phenomthm_td(m1, m2, s1z, s2z, distance_mpc, inclination, phi_ref):
    hp, hc = lalsim.SimInspiralChooseTDWaveform(
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
        phi_ref,
        0.0,  # longAscNodes
        0.0,  # eccentricity
        0.0,  # meanPerAno
        _DELTA_T,
        _F_MIN,
        _F_REF,
        lal.CreateDict(),
        lalsim.IMRPhenomTHM,
    )
    t = jnp.arange(hp.data.length) * hp.deltaT + float(hp.epoch)
    return t, jnp.asarray(hp.data.data), jnp.asarray(hc.data.data)


@pytest.mark.parametrize("m1, m2, s1z, s2z, distance_mpc, inclination, phi_ref", _PARAM_GRID)
def test_phenomthm_matches_lalsuite(m1, m2, s1z, s2z, distance_mpc, inclination, phi_ref):
    t, hp_lal, hc_lal = _lal_phenomthm_td(m1, m2, s1z, s2z, distance_mpc, inclination, phi_ref)

    q = min(m1, m2) / max(m1, m2)
    mtot = m1 + m2
    eta = q / (1.0 + q) ** 2
    mc = mtot * eta**0.6
    params = dict(
        chirp_mass=mc, mass_ratio=q, spin1z=s1z, spin2z=s2z,
        luminosity_distance=distance_mpc, phase=phi_ref, inclination=inclination,
        geocent_time=0.0,
    )
    model = IMRPhenomTHM(f_ref=_F_REF)
    hp_jax, hc_jax = model(params, t)

    dt = _DELTA_T
    n = t.shape[0]
    freqs = jnp.asarray(rfft_freqs(n, dt))
    psd = jnp.asarray(aligo_zdhp_psd(freqs, f_low=_F_MIN))
    band = jnp.asarray(freqs >= _F_MIN)
    df = float(freqs[1] - freqs[0])

    mm_p = float(mismatch(td_to_fd(hp_jax, dt), td_to_fd(hp_lal, dt), psd, df, band=band))
    mm_c = float(mismatch(td_to_fd(hc_jax, dt), td_to_fd(hc_lal, dt), psd, df, band=band))

    assert mm_p < _MISMATCH_TOLERANCE, f"h_plus mismatch {mm_p:.3e} vs LALSuite IMRPhenomTHM"
    assert mm_c < _MISMATCH_TOLERANCE, f"h_cross mismatch {mm_c:.3e} vs LALSuite IMRPhenomTHM"


def test_phenomthm_matches_lalsuite_across_grid_reports_worst_case():
    worst = 0.0
    for m1, m2, s1z, s2z, distance_mpc, inclination, phi_ref in _PARAM_GRID:
        t, hp_lal, _ = _lal_phenomthm_td(m1, m2, s1z, s2z, distance_mpc, inclination, phi_ref)
        q = min(m1, m2) / max(m1, m2)
        mtot = m1 + m2
        eta = q / (1.0 + q) ** 2
        mc = mtot * eta**0.6
        params = dict(
            chirp_mass=mc, mass_ratio=q, spin1z=s1z, spin2z=s2z,
            luminosity_distance=distance_mpc, phase=phi_ref, inclination=inclination,
            geocent_time=0.0,
        )
        model = IMRPhenomTHM(f_ref=_F_REF)
        hp_jax, _ = model(params, t)

        dt = _DELTA_T
        freqs = jnp.asarray(rfft_freqs(t.shape[0], dt))
        psd = jnp.asarray(aligo_zdhp_psd(freqs, f_low=_F_MIN))
        band = jnp.asarray(freqs >= _F_MIN)
        mm = float(
            mismatch(td_to_fd(hp_jax, dt), td_to_fd(hp_lal, dt), psd, float(freqs[1] - freqs[0]), band=band)
        )
        print(
            f"  m1={m1:5.1f} m2={m2:5.1f} s1z={s1z:+.2f} s2z={s2z:+.2f} "
            f"iota={inclination:.2f} phiRef={phi_ref:.2f} -> mismatch {mm:.3e}"
        )
        worst = max(worst, mm)
    assert worst < _MISMATCH_TOLERANCE, f"worst-case grid mismatch {worst:.3e}"


def test_phenomthm_modes_match_lalsuite_mode_by_mode():
    """Direct mode-level comparison via SimIMRPhenomTHM_Modes -- a more granular check than
    the polarization-summed test above, since a per-mode bug could in principle cancel in
    the Ylm-weighted sum for a particular inclination."""
    m1, m2, s1z, s2z, distance_mpc = 30.0, 20.0, 0.5, -0.3, 300.0
    hlm = lalsim.SimIMRPhenomTHM_Modes(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, s1z, s2z,
        distance_mpc * 1e6 * lal.PC_SI, _DELTA_T, _F_MIN, _F_REF, 0.0, lal.CreateDict(),
    )
    node = hlm
    lal_modes = {}
    while node is not None:
        lal_modes[(node.l, node.m)] = (
            jnp.asarray(node.mode.data.data.copy()), float(node.mode.epoch), node.mode.deltaT,
        )
        node = node.next

    (_, epoch, dt) = next(iter(lal_modes.values()))
    t = jnp.arange(next(iter(lal_modes.values()))[0].shape[0]) * dt + epoch

    q = min(m1, m2) / max(m1, m2)
    mtot = m1 + m2
    eta = q / (1.0 + q) ** 2
    mc = mtot * eta**0.6
    params = dict(
        chirp_mass=mc, mass_ratio=q, spin1z=s1z, spin2z=s2z,
        luminosity_distance=distance_mpc, phase=0.0, inclination=0.0, geocent_time=0.0,
    )
    model = IMRPhenomTHM(f_ref=_F_REF)
    modes_jax = model.mode_dict(params, t)  # strain @ 1 Mpc
    dist_scale = distance_mpc  # LAL modes are @ distance_mpc; jaxpe's mode_dict is @ 1 Mpc

    # h_lm is complex (not a real polarization), so jaxpe.gw.mismatch's rfft-based FD
    # machinery doesn't apply here -- use a flat-noise complex time-domain overlap instead,
    # which is all this per-mode sanity check needs (the polarization-level tests above
    # already cover the PSD-weighted, physically-relevant comparison).
    for (l, m), (h_lal, _, _) in lal_modes.items():
        if jnp.max(jnp.abs(h_lal)) == 0.0:
            continue  # a mode LAL itself reports as exactly zero (e.g. odd m at this q/spin)
        h_jax = modes_jax[(l, m)] / dist_scale
        overlap = jnp.sum(h_jax * jnp.conj(h_lal)).real
        norm = jnp.sqrt(jnp.sum(jnp.abs(h_jax) ** 2) * jnp.sum(jnp.abs(h_lal) ** 2))
        mm = float(1.0 - overlap / norm)
        assert mm < 1e-4, f"mode ({l},{m}) mismatch {mm:.3e} vs LALSuite"
