import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from jaxpe.gw.cbc_models.phenomt import IMRPhenomT
from jaxpe.gw import aligo_zdhp_psd, mismatch, rfft_freqs, td_to_fd


@pytest.fixture
def phenomt_model():
    return IMRPhenomT(f_ref=20.0)


@pytest.fixture
def default_params():
    return {
        "chirp_mass": 30.0,
        "mass_ratio": 1.0,
        "spin1z": 0.0,
        "spin2z": 0.0,
        "luminosity_distance": 100.0,
        "phase": 0.0,
        "inclination": 0.0,
        "geocent_time": 0.0,
    }


def test_initialization(phenomt_model):
    assert not phenomt_model.is_fd
    assert phenomt_model.f_ref == 20.0


def test_evaluate_waveform(phenomt_model, default_params):
    t_grid = jnp.linspace(-2.0, 0.1, 1000)
    hp, hc = phenomt_model(default_params, t_grid)

    assert hp.shape == t_grid.shape
    assert hc.shape == t_grid.shape
    assert not jnp.any(jnp.isnan(hp))
    assert not jnp.any(jnp.isnan(hc))


def test_jit_compilation(phenomt_model, default_params):
    t_grid = jnp.linspace(-2.0, 0.1, 100)
    jitted_model = jax.jit(phenomt_model)

    hp, hc = jitted_model(default_params, t_grid)
    assert hp.shape == t_grid.shape


def test_vmap_vectorization(phenomt_model, default_params):
    # Vectorize over mass_ratio
    t_grid = jnp.linspace(-2.0, 0.1, 100)

    params_batched = {
        "chirp_mass": jnp.array([30.0, 30.0]),
        "mass_ratio": jnp.array([1.0, 0.5]),
        "spin1z": jnp.array([0.0, 0.0]),
        "spin2z": jnp.array([0.0, 0.0]),
        "luminosity_distance": jnp.array([100.0, 100.0]),
        "phase": jnp.array([0.0, 0.0]),
        "inclination": jnp.array([0.0, 0.0]),
        "geocent_time": jnp.array([0.0, 0.0]),
    }

    vmapped_model = jax.vmap(phenomt_model, in_axes=(0, None))
    hp, hc = vmapped_model(params_batched, t_grid)

    assert hp.shape == (2, 100)
    assert hc.shape == (2, 100)
    assert not jnp.any(jnp.isnan(hp))


def test_piecewise_continuity(phenomt_model, default_params):
    """No NaN/inf at the fixed t=0 merger-ringdown boundary, or at an extreme early time."""
    t_grid = jnp.array([-100.0, 0.0])
    hp, hc = phenomt_model(default_params, t_grid)

    assert not jnp.any(jnp.isnan(hp))
    assert not jnp.any(jnp.isnan(hc))


def test_amplitude_and_phase_continuous_at_merger_ringdown_boundary(
    phenomt_model, default_params
):
    """Real continuity check (unlike the placeholder implementation this replaced, per
    docs/constants.md): amplitude and phase must agree closely just below vs. just above t=0,
    the fixed merger/ringdown boundary -- not merely avoid NaN there."""
    eps = 1e-6
    t_grid = jnp.array([-eps, eps])
    hp, hc = phenomt_model(default_params, t_grid)
    h = hp - 1j * hc
    rel_amp_diff = float(jnp.abs(jnp.abs(h[1]) - jnp.abs(h[0])) / jnp.abs(h[0]))
    phase_diff = float(jnp.angle(h[1] / h[0]))
    # Loose tolerances: this checks for a *gross* discontinuity (the placeholder this
    # replaced had none of these guarantees at all), not sub-percent precision -- the
    # LAL-comparison tests below are the real accuracy check, and already show ~1e-6-level
    # mismatch across the whole waveform including through this exact boundary.
    assert rel_amp_diff < 1e-2, f"amplitude jump at t=0 boundary: {rel_amp_diff:.2e}"
    assert abs(phase_diff) < 5e-3, f"phase jump at t=0 boundary: {phase_diff:.2e}"


def test_gradient_finite(phenomt_model, default_params):
    t_grid = jnp.linspace(-1.0, 0.05, 200)

    def loss(mc):
        p = dict(default_params, chirp_mass=mc)
        hp, _ = phenomt_model(p, t_grid)
        return jnp.sum(hp**2)

    grad = jax.grad(loss)(default_params["chirp_mass"])
    assert jnp.isfinite(grad)


def test_gradient_matches_finite_difference(phenomt_model, default_params):
    """Gradient correctness, not just finiteness -- the piecewise inspiral/merger/ringdown
    construction is exactly the kind of thing that can silently break differentiability at a
    region boundary if built carelessly (e.g. a Python-level branch instead of jnp.where).
    """
    t_grid = jnp.linspace(-1.0, 0.05, 500)
    params = dict(default_params, mass_ratio=0.7, spin1z=0.2, spin2z=-0.1)

    def loss(mc):
        p = dict(params, chirp_mass=mc)
        hp, hc = phenomt_model(p, t_grid)
        return jnp.sum(hp**2 + hc**2)

    mc0 = 30.0
    grad = float(jax.grad(loss)(mc0))
    h = 1e-3
    fd = float((loss(mc0 + h) - loss(mc0 - h)) / (2.0 * h))
    assert grad == pytest.approx(
        fd, rel=1e-3
    ), f"autodiff grad {grad} vs finite-diff {fd}"


def test_batched_matches_serial_via_mismatch(phenomt_model):
    """Batched (vmap) evaluation must match a serial loop -- PSD-weighted mismatch, not
    elementwise amplitude, matching tests/test_gw.py::test_batched_matches_serial's own
    rationale (XLA fuses a jitted graph differently per call shape; a raw-amplitude
    comparison would flag harmless ~1e-5-level differences as failures)."""
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
    hp_batched, _ = jax.vmap(phenomt_model, in_axes=(0, None))(batch, t_grid)

    dt = float(t_grid[1] - t_grid[0])
    freqs = jnp.asarray(rfft_freqs(t_grid.shape[0], dt))
    psd = jnp.asarray(aligo_zdhp_psd(freqs, f_low=20.0))
    band = jnp.asarray(freqs >= 20.0)

    for i in range(n):
        params_i = {k: v[i] for k, v in batch.items()}
        hp_serial, _ = phenomt_model(params_i, t_grid)
        a = td_to_fd(hp_batched[i], dt)
        b = td_to_fd(hp_serial, dt)
        mm = float(mismatch(a, b, psd, float(freqs[1] - freqs[0]), band=band))
        assert mm < 1e-6, f"injection {i}: batched-vs-serial mismatch {mm:.3e}"


lal = pytest.importorskip("lal")
lalsim = pytest.importorskip("lalsimulation")

_F_MIN = 20.0
_F_REF = 20.0
_DELTA_T = 1.0 / 4096.0

# Same grid style as test_phenomd.py: mass ratio 1:1 to ~5:1, spins up to 0.8, inclination
# across [0, pi/2, pi], and a nonzero reference phase to exercise the phi_ref/Ylm convention
# (see phenomt.py's __call__ docstring -- phi_ref enters through the Ylm's azimuthal argument,
# not a separate per-mode rotation, and only this convention reproduces LALSuite's own
# phi_ref-dependence away from face-on/edge-on inclination).
_PARAM_GRID = [
    (30.0, 20.0, 0.5, -0.3, 100.0, 0.5, 0.0),
    (30.0, 30.0, 0.0, 0.0, 500.0, 0.0, 0.0),
    (10.0, 10.0, 0.0, 0.0, 300.0, 1.0, 0.0),
    (50.0, 10.0, 0.8, 0.8, 800.0, 0.9, 0.0),
    (30.0, 30.0, 0.0, 0.0, 500.0, 0.0, 1.0),
    # NOT incl=pi/2: for a pure (l,m)=(2,+-2) system (dominant-mode-only, which is exactly
    # what IMRPhenomT is), h_cross is proportional to cos(iota) and vanishes *exactly* at
    # edge-on -- confirmed both LAL and jaxpe agree it's ~1e-37 there regardless of mass/spin
    # (~1e-22 h_plus scale, so that's floating-point noise, not physics). A mismatch computed
    # against a ~0 reference is numerically meaningless. 1.9 rad stays away from both 0 and
    # pi/2 while still exercising a nonzero phi_ref.
    (30.0, 20.0, 0.3, -0.2, 500.0, 1.9, 0.7),
]

# Calibrated on this exact grid: worst observed mismatch ~4.7e-6, same scale as PhenomD's own
# (Phase 0, tests/test_phenomd.py) LAL-comparison tolerance.
_MISMATCH_TOLERANCE = 1e-5


def _lal_phenomt_td(m1, m2, s1z, s2z, distance_mpc, inclination, phi_ref):
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
        lalsim.IMRPhenomT,
    )
    t = jnp.arange(hp.data.length) * hp.deltaT + float(hp.epoch)
    return t, jnp.asarray(hp.data.data), jnp.asarray(hc.data.data)


@pytest.mark.parametrize(
    "m1, m2, s1z, s2z, distance_mpc, inclination, phi_ref", _PARAM_GRID
)
def test_phenomt_matches_lalsuite(m1, m2, s1z, s2z, distance_mpc, inclination, phi_ref):
    t, hp_lal, hc_lal = _lal_phenomt_td(
        m1, m2, s1z, s2z, distance_mpc, inclination, phi_ref
    )

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
        phase=phi_ref,
        inclination=inclination,
        geocent_time=0.0,
    )
    model = IMRPhenomT(f_ref=_F_REF)
    hp_jax, hc_jax = model(params, t)

    dt = _DELTA_T
    n = t.shape[0]
    freqs = jnp.asarray(rfft_freqs(n, dt))
    psd = jnp.asarray(aligo_zdhp_psd(freqs, f_low=_F_MIN))
    band = jnp.asarray(freqs >= _F_MIN)
    df = float(freqs[1] - freqs[0])

    mm_p = float(
        mismatch(td_to_fd(hp_jax, dt), td_to_fd(hp_lal, dt), psd, df, band=band)
    )
    mm_c = float(
        mismatch(td_to_fd(hc_jax, dt), td_to_fd(hc_lal, dt), psd, df, band=band)
    )

    assert (
        mm_p < _MISMATCH_TOLERANCE
    ), f"h_plus mismatch {mm_p:.3e} vs LALSuite IMRPhenomT"
    assert (
        mm_c < _MISMATCH_TOLERANCE
    ), f"h_cross mismatch {mm_c:.3e} vs LALSuite IMRPhenomT"


def test_phenomt_matches_lalsuite_across_grid_reports_worst_case():
    """Prints the full grid's mismatch table (see test_phenomd.py's twin of this test) so the
    calibration behind _MISMATCH_TOLERANCE is visible in test output, not just asserted.
    """
    worst = 0.0
    for m1, m2, s1z, s2z, distance_mpc, inclination, phi_ref in _PARAM_GRID:
        t, hp_lal, _ = _lal_phenomt_td(
            m1, m2, s1z, s2z, distance_mpc, inclination, phi_ref
        )
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
            phase=phi_ref,
            inclination=inclination,
            geocent_time=0.0,
        )
        model = IMRPhenomT(f_ref=_F_REF)
        hp_jax, _ = model(params, t)

        dt = _DELTA_T
        freqs = jnp.asarray(rfft_freqs(t.shape[0], dt))
        psd = jnp.asarray(aligo_zdhp_psd(freqs, f_low=_F_MIN))
        band = jnp.asarray(freqs >= _F_MIN)
        mm = float(
            mismatch(
                td_to_fd(hp_jax, dt),
                td_to_fd(hp_lal, dt),
                psd,
                float(freqs[1] - freqs[0]),
                band=band,
            )
        )
        print(
            f"  m1={m1:5.1f} m2={m2:5.1f} s1z={s1z:+.2f} s2z={s2z:+.2f} "
            f"iota={inclination:.2f} phiRef={phi_ref:.2f} -> mismatch {mm:.3e}"
        )
        worst = max(worst, mm)
    assert worst < _MISMATCH_TOLERANCE, f"worst-case grid mismatch {worst:.3e}"
