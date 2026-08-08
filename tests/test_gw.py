"""GW-layer tests.

The detector geometry is checked against LAL (exact reference); the likelihood is
checked against an independent numpy reimplementation and against the analytic
zero-noise property lnL(true params) = 0.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpe.gw import (
    DETECTORS,
    ToyChirp,
    antenna_pattern,
    bbh_priors,
    gmst_from_gps,
    make_injection,
    mismatch as _mismatch,
    mismatch_f32_f64,
    time_delay_from_geocenter,
    tukey_window,
)

lal = pytest.importorskip("lal")

T_C = 1126259462.4

INJ = dict(
    chirp_mass=30.0,
    mass_ratio=0.8,
    luminosity_distance=800.0,
    inclination=0.6,
    phase=1.2,
    ra=1.95,
    dec=-1.27,
    psi=0.82,
    geocent_time=T_C,
)


def test_detector_geometry_matches_lal():
    lal_dets = {d.frDetector.prefix: d for d in lal.CachedDetectors}
    rng = np.random.default_rng(0)
    gmst = gmst_from_gps(T_C)
    for name, det in DETECTORS.items():
        ld = lal_dets[name]
        np.testing.assert_allclose(det.location, np.array(ld.location), atol=1e-3)
        np.testing.assert_allclose(det.response, np.array(ld.response), atol=1e-6)
        for _ in range(10):
            ra = rng.uniform(0, 2 * np.pi)
            dec = rng.uniform(-np.pi / 2, np.pi / 2)
            psi = rng.uniform(0, np.pi)
            fp_l, fc_l = lal.ComputeDetAMResponse(ld.response, ra, dec, psi, gmst)
            fp, fc = antenna_pattern(det, ra, dec, psi, gmst)
            assert abs(float(fp) - fp_l) < 2e-6 and abs(float(fc) - fc_l) < 2e-6
            dt_l = lal.TimeDelayFromEarthCenter(
                ld.location, ra, dec, lal.LIGOTimeGPS(T_C)
            )
            dt = float(time_delay_from_geocenter(det, ra, dec, gmst))
            assert abs(dt - dt_l) < 1e-9


def test_gmst_matches_lal():
    for gps in [1126259462.4, 1187008882.4, 1264316116.4]:
        assert (
            abs(gmst_from_gps(gps) - lal.GreenwichMeanSiderealTime(gps) % (2 * np.pi))
            < 1e-9
        )


@pytest.fixture(scope="module")
def zero_noise_like():
    return make_injection(ToyChirp(f_start=20.0), INJ, noise_seed=None)


def test_zero_noise_lnl_peaks_at_truth(zero_noise_like):
    like = zero_noise_like
    params = {k: jnp.asarray(v) for k, v in INJ.items()}
    lnl_true = float(like.log_likelihood(params))
    assert abs(lnl_true) < 1e-6, f"lnL(true) = {lnl_true}"

    snrs = like.optimal_snr(params)
    assert all(5.0 < s < 100.0 for s in snrs.values()), snrs

    # perturbations must strictly decrease lnL, by a lot for the chirp mass
    # note: this is a short (~70 Msun total) toy signal with ~30 cycles in band, so
    # chirp mass needs a percent-level shift to dephase it appreciably
    for key, delta, min_drop in [
        ("chirp_mass", 1.0, 5.0),
        ("geocent_time", 0.01, 5.0),
        ("luminosity_distance", 200.0, 1.0),
    ]:
        p = dict(params)
        p[key] = p[key] + delta
        lnl = float(like.log_likelihood(p))
        assert lnl < lnl_true - min_drop, f"{key}: lnL only dropped to {lnl}"


def test_lnl_matches_numpy_reimplementation(zero_noise_like):
    """Independent numpy Whittle sum on a noisy injection."""
    like = make_injection(ToyChirp(f_start=20.0), INJ, noise_seed=7)
    params = {k: jnp.asarray(v) for k, v in INJ.items()}
    lnl_jax = float(like.log_likelihood(params))

    strains = jax.jit(like.detector_strains_fd)(params)
    df = like.freqs[1] - like.freqs[0]
    band = (like.freqs >= like.f_min) & (like.freqs <= like.f_max)
    lnl_np = 0.0
    for det in like.detectors:
        r = np.asarray(like.data_fd[det.name]) - np.asarray(strains[det.name])
        lnl_np += -2.0 * df * np.sum(np.abs(r[band]) ** 2 / like.psds[det.name][band])
    np.testing.assert_allclose(lnl_jax, lnl_np, rtol=1e-10)


def test_snr_scales_inversely_with_distance(zero_noise_like):
    like = zero_noise_like
    p1 = {k: jnp.asarray(v) for k, v in INJ.items()}
    p2 = dict(p1)
    p2["luminosity_distance"] = p1["luminosity_distance"] * 2.0
    s1, s2 = like.optimal_snr(p1), like.optimal_snr(p2)
    for name in s1:
        np.testing.assert_allclose(s1[name] / s2[name], 2.0, rtol=1e-6)


def test_posterior_gradient_finite(zero_noise_like):
    prior = bbh_priors(geocent_time=T_C)
    problem = zero_noise_like.problem(prior)
    y_true = problem.prior.to_unconstrained(
        problem.prior.from_dict({k: jnp.asarray(v) for k, v in INJ.items()})
    )
    val, grad = jax.value_and_grad(problem.log_posterior)(y_true)
    assert jnp.isfinite(val)
    assert jnp.all(jnp.isfinite(grad))
    # chirp mass is measured to ~1e-3: its gradient must dominate distance's
    key = jax.random.PRNGKey(0)
    y = problem.sample_unconstrained(key, 8)
    grads = jax.vmap(jax.grad(problem.log_posterior))(y)
    assert jnp.all(jnp.isfinite(grads))


def test_toychirp_f32_mismatch_small():
    n = int(8.0 * 2048)
    times = T_C - 6.0 + np.arange(n) / 2048.0
    mm = mismatch_f32_f64(ToyChirp(f_start=20.0), INJ, times)
    assert mm < 1e-3, f"float32 mismatch {mm}"


def test_tukey_window_matches_scipy():
    from scipy.signal.windows import tukey

    for n, a in [(256, 0.1), (1024, 0.5)]:
        np.testing.assert_allclose(tukey_window(n, a), tukey(n, a), atol=1e-12)


def test_imrphenomd_matches_ripple():
    pytest.importorskip("ripplegw")
    from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc as ripple_phenomd
    from jaxpe.gw import IMRPhenomD

    # Setup parameters
    q = 1.5
    mtot = 70.0
    m1 = mtot / (1 + 1 / q)
    m2 = mtot / (1 + q)
    eta = (m1 * m2) / mtot**2
    mc = mtot * eta ** (3.0 / 5.0)

    s1z = 0.5
    s2z = -0.3
    dist = 800.0
    tc = 1126259462.4
    phic = 1.2
    iota = 0.6
    f_ref = 20.0

    params = dict(
        chirp_mass=mc,
        mass_ratio=q,
        luminosity_distance=dist,
        inclination=iota,
        phase=phic,
        geocent_time=tc,
        spin1z=s1z,
        spin2z=s2z,
    )

    freqs = jnp.linspace(20.0, 1024.0, 1000)

    # jaxpe evaluation
    model = IMRPhenomD(f_ref=f_ref)
    hp_jaxpe, hc_jaxpe = model(params, freqs)

    # ripple evaluation
    theta_ripple = jnp.array([mc, eta, s1z, s2z, dist, tc, phic, iota])
    hp_ripple, hc_ripple = ripple_phenomd(freqs, theta_ripple, f_ref)

    np.testing.assert_allclose(hp_jaxpe, hp_ripple, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(hc_jaxpe, hc_ripple, rtol=1e-6, atol=1e-10)


def test_unified_waveform_generator_interface():
    from jaxpe.gw.cbc_models.base import (
        WaveformModel,
        TimeDomainModel,
        FrequencyDomainModel,
    )
    from jaxpe.gw import IMRPhenomD

    # Check inheritance
    assert issubclass(IMRPhenomD, WaveformModel)
    assert issubclass(IMRPhenomD, FrequencyDomainModel)

    model = IMRPhenomD()
    assert getattr(model, "is_fd", False)

    # Test __call__ signature
    params = {
        "chirp_mass": 30.0,
        "mass_ratio": 0.5,
        "luminosity_distance": 400.0,
        "inclination": 0.6,
        "phase": 1.2,
        "geocent_time": 0.0,
        "spin1z": 0.0,
        "spin2z": 0.0,
    }
    grid = jnp.linspace(20.0, 1024.0, 100)
    hp, hc = model(params, grid)
    assert hp.shape == grid.shape
    assert hc.shape == grid.shape

    # Check ESIGMAInspiral
    try:
        from jaxpe.gw import ESIGMAInspiral

        if ESIGMAInspiral is not None:
            assert issubclass(ESIGMAInspiral, WaveformModel)
            assert issubclass(ESIGMAInspiral, TimeDomainModel)
            emodel = ESIGMAInspiral()
            assert not getattr(emodel, "is_fd", True)
            egrid = jnp.linspace(-2.0, 0.0, 100)
            ehp, ehc = emodel(params, egrid)
            assert ehp.shape == egrid.shape
            assert ehc.shape == egrid.shape
    except ImportError:
        pass

    # Check NRSur7dq4
    from jaxpe.gw import NRSur7dq4

    assert issubclass(NRSur7dq4, WaveformModel)
    assert issubclass(NRSur7dq4, TimeDomainModel)
    # We do not instantiate it here as it requires a data file path.


def test_lalsim_psd_curves_are_ordered_and_finite():
    """Named LALSimulation curves load, and their sensitivities order as expected.

    CE and ET are third-generation designs, so their strain noise must sit below A+,
    which in turn sits below the analytic aLIGO ZDHP fit. This pins the name -> symbol
    mapping: a typo in LALSIM_PSDS would resolve to a different detector and break the
    ordering rather than failing loudly.
    """
    from jaxpe.gw import LALSIM_PSDS, aligo_zdhp_psd, lalsim_psd

    freqs = np.arange(0.0, 1024.0 + 0.25, 0.25)

    def best_strain(psd):
        finite = psd[np.isfinite(psd)]
        assert finite.size > 0
        return float(np.sqrt(finite.min()))

    ce = best_strain(lalsim_psd("CE", freqs))
    et = best_strain(lalsim_psd("ET", freqs))
    aplus = best_strain(lalsim_psd("aplus", freqs))
    zdhp = best_strain(aligo_zdhp_psd(freqs))

    assert ce < et < aplus < zdhp
    assert set(LALSIM_PSDS) >= {"CE", "ET", "aplus"}


def test_lalsim_psd_rejects_bad_input():
    """Unknown names and non-uniform grids raise instead of silently mis-modelling."""
    from jaxpe.gw import lalsim_psd

    freqs = np.arange(0.0, 64.0 + 0.5, 0.5)
    with pytest.raises(ValueError, match="unknown LALSimulation PSD"):
        lalsim_psd("NotADetector", freqs)
    # the series API is defined by (df, n); an irregular grid cannot be honoured
    with pytest.raises(ValueError, match="uniform frequency grid"):
        lalsim_psd("CE", np.array([0.0, 1.0, 3.0, 7.0]))


# ---------------------------------------------------------------------------
# Noise seeding
#
# Two defects these pin down. (1) make_injection advanced a single numpy Generator
# through the detector loop, so a detector's realisation depended on the position
# it happened to occupy in detector_names. (2) The CLI handed every injection in a
# campaign the same seeds.noise, so a --noise gaussian PP campaign analysed N
# copies of one noise realisation -- which invalidates the test it exists to run.
# ---------------------------------------------------------------------------


def _differ(a, b, frac=1e-3):
    """True if a and b differ appreciably *relative to their own scale*.

    Strain arrays are O(1e-23), so ``np.allclose``'s default ``atol=1e-8`` calls any
    two of them equal and would make these assertions vacuous.
    """
    scale = np.max(np.abs(a))
    return bool(np.max(np.abs(np.asarray(a) - np.asarray(b))) > frac * scale)


def test_noise_is_invariant_to_detector_order():
    """A detector's noise must depend on which detector it is, not on list order."""
    a = make_injection(
        ToyChirp(f_start=20.0), INJ, detector_names=("H1", "L1"), noise_seed=11
    )
    b = make_injection(
        ToyChirp(f_start=20.0), INJ, detector_names=("L1", "H1"), noise_seed=11
    )
    for det in ("H1", "L1"):
        np.testing.assert_array_equal(
            np.asarray(a.data_fd[det]),
            np.asarray(b.data_fd[det]),
            err_msg=f"{det} noise changed when detector_names was reordered",
        )


def test_noise_streams_are_independent_per_detector():
    like = make_injection(
        ToyChirp(f_start=20.0), INJ, detector_names=("H1", "L1"), noise_seed=11
    )
    clean = make_injection(
        ToyChirp(f_start=20.0), INJ, detector_names=("H1", "L1"), noise_seed=None
    )
    n_h1 = np.asarray(like.data_fd["H1"]) - np.asarray(clean.data_fd["H1"])
    n_l1 = np.asarray(like.data_fd["L1"]) - np.asarray(clean.data_fd["L1"])
    # Strain is O(1e-23), so np.allclose's default atol=1e-8 would call any two of
    # these arrays equal. Compare against the array's own scale instead.
    assert _differ(n_h1, n_l1), "detectors share a noise realisation"

    # The residuals must also be *statistically* right, not merely different. With
    # sigma = sqrt(S*T)/2 and n = sigma*(N(0,1) + i N(0,1)), <|n|^2> = 2 sigma^2 =
    # S(f) * T / 2. make_injection's default duration is 8 s.
    band = (like.freqs >= like.f_min) & (like.freqs <= like.f_max)
    for name, n in (("H1", n_h1), ("L1", n_l1)):
        expected = like.psds[name][band] * 8.0 / 2.0
        ratio = float(np.mean(np.abs(n[band]) ** 2 / expected))
        assert 0.9 < ratio < 1.1, f"{name} noise power off by {ratio:.3f}x"


def test_distinct_injections_get_distinct_noise():
    """derive_noise_seed must separate injections drawn from one campaign seed."""
    from jaxpe.gw import derive_noise_seed

    seeds = [derive_noise_seed(42, i) for i in range(8)]
    assert len(set(seeds)) == len(seeds), "campaign seeds collide across injections"
    assert all(isinstance(s, int) for s in seeds), "must be JSON-serialisable ints"

    a = make_injection(
        ToyChirp(f_start=20.0), INJ, detector_names=("H1",), noise_seed=seeds[0]
    )
    b = make_injection(
        ToyChirp(f_start=20.0), INJ, detector_names=("H1",), noise_seed=seeds[1]
    )
    assert _differ(
        np.asarray(a.data_fd["H1"]), np.asarray(b.data_fd["H1"])
    ), "two injections of one campaign share a noise realisation"


def test_derive_noise_seed_is_stable_across_processes():
    """Must not depend on PYTHONHASHSEED: Python's hash() is salted per process."""
    import subprocess
    import sys

    code = "from jaxpe.gw import derive_noise_seed; print(derive_noise_seed(42, 3))"
    out = []
    for salt in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": salt, "JAX_PLATFORMS": "cpu"}
        r = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env
        )
        assert r.returncode == 0, r.stderr
        out.append(r.stdout.strip())
    assert out[0] == out[1], f"seed depends on PYTHONHASHSEED: {out}"


def test_jax_noise_matches_the_numpy_convention():
    """The JAX generator must reproduce the numpy one's *distribution*.

    Not its samples -- different PRNGs -- so this checks the sigma convention that
    would silently shift if sqrt(S*T)/2 were ever mis-transcribed.
    """
    from jaxpe.gw import simulate_noise_fd, simulate_noise_fd_jax

    duration = 8.0
    freqs = np.linspace(20.0, 512.0, 4096)
    psd = 1e-46 * (1.0 + (freqs / 100.0) ** 4)

    n_np = simulate_noise_fd(np.random.default_rng(0), psd, duration)
    n_jx = np.asarray(simulate_noise_fd_jax(jax.random.PRNGKey(0), psd, duration))

    var_np = np.mean(np.abs(n_np) ** 2 / psd)
    var_jx = np.mean(np.abs(n_jx) ** 2 / psd)
    np.testing.assert_allclose(var_jx, var_np, rtol=0.05)
    np.testing.assert_allclose(var_jx, duration / 2.0, rtol=0.05)

    # Non-finite PSD bins (out of band) must carry exactly zero in both.
    psd_inf = np.where(freqs < 30.0, np.inf, psd)
    z = np.asarray(simulate_noise_fd_jax(jax.random.PRNGKey(1), psd_inf, duration))
    assert np.all(z[freqs < 30.0] == 0.0)


def test_analysis_grid_matches_make_injection():
    """The shared grid must be the one make_injection actually builds on.

    Four call sites used to re-derive this by hand to stay in sync with
    make_injection's internals; test_marginalized.py even asserted the copy had not
    drifted. That assertion is now a real check of one definition.
    """
    from jaxpe.gw import analysis_grid, resolve_f_max

    duration, sampling_rate, post_trigger = 8.0, 2048.0, 2.0
    times, freqs = analysis_grid(T_C, duration, sampling_rate, post_trigger)

    like = make_injection(
        ToyChirp(f_start=20.0),
        INJ,
        detector_names=("H1",),
        duration=duration,
        sampling_rate=sampling_rate,
        post_trigger=post_trigger,
        noise_seed=None,
    )
    np.testing.assert_array_equal(like.times, times)
    np.testing.assert_array_equal(like.freqs, freqs)

    # The segment ends post_trigger after the trigger, and f_max defaults to 90% of
    # Nyquist -- the convention that was written out separately in both constructors.
    assert times[0] == pytest.approx(T_C + post_trigger - duration)
    assert times.size == int(duration * sampling_rate)
    assert resolve_f_max(None, sampling_rate) == pytest.approx(0.9 * sampling_rate / 2)
    assert resolve_f_max(512.0, sampling_rate) == 512.0
    assert like.f_max == pytest.approx(resolve_f_max(None, sampling_rate))


def test_snr_targeting_is_exact_in_one_measurement():
    """h ~ 1/D exactly, so the target is hit without iteration."""
    from jaxpe.gw import distance_for_target_snr, network_snr

    like = make_injection(
        ToyChirp(f_start=20.0), INJ, detector_names=("H1", "L1"), noise_seed=None
    )
    # network_snr is the quadrature sum the three call sites used to open-code.
    snrs = like.optimal_snr(INJ)
    expected = float(np.sqrt(sum(float(v) ** 2 for v in snrs.values())))
    assert network_snr(like, INJ) == expected

    for target in (12.0, 25.0, 60.0):
        d = distance_for_target_snr(like, INJ, target)
        params = {**INJ, "luminosity_distance": d}
        rebuilt = make_injection(
            ToyChirp(f_start=20.0), params, detector_names=("H1", "L1"), noise_seed=None
        )
        assert network_snr(rebuilt, params) == pytest.approx(target, rel=1e-9)


def test_snr_targeting_ignores_the_noise_seed():
    """Optimal SNR is a property of the template, not of the data.

    Solving the distance against a noisy build would fit it to one noise draw.
    """
    from jaxpe.gw import distance_for_target_snr

    clean = make_injection(
        ToyChirp(f_start=20.0), INJ, detector_names=("H1",), noise_seed=None
    )
    noisy = make_injection(
        ToyChirp(f_start=20.0), INJ, detector_names=("H1",), noise_seed=3
    )
    assert distance_for_target_snr(clean, INJ, 20.0) == pytest.approx(
        distance_for_target_snr(noisy, INJ, 20.0), rel=1e-12
    )


# ---------------------------------------------------------------------------
# Batched injection creation
# ---------------------------------------------------------------------------


def _psd_weighted_mismatch(like, a, b, det="H1"):
    """Thin wrapper around ``jaxpe.gw.mismatch`` for a jaxpe likelihood object.

    A max-relative amplitude comparison is misleading here. XLA fuses the waveform
    differently between jit graphs, so make_injections and make_injection can differ
    at the 1e-5 level in raw amplitude while being indistinguishable to any inference
    -- the mismatch is ~1e-9 and lnL(truth) stays at ~1e-7.
    """
    band = (like.freqs >= like.f_min) & (like.freqs <= like.f_max)
    df = float(like.freqs[1] - like.freqs[0])
    return float(_mismatch(a, b, like.psds[det], df, band=band))


@pytest.mark.parametrize("model", ["fd", "td"])
def test_batched_matches_serial(model):
    from jaxpe.gw import IMRPhenomD, IMRPhenomT, make_injections

    wf = IMRPhenomD(f_ref=20.0) if model == "fd" else IMRPhenomT(f_ref=20.0)
    kw = dict(
        detector_names=("H1", "L1"), duration=4.0, sampling_rate=1024.0, f_min=30.0
    )
    rng = np.random.default_rng(0)
    n = 4
    batch = {
        "chirp_mass": rng.uniform(25.0, 35.0, n),
        "mass_ratio": rng.uniform(0.5, 1.0, n),
        "spin1z": rng.uniform(-0.3, 0.3, n),
        "spin2z": rng.uniform(-0.3, 0.3, n),
        "luminosity_distance": rng.uniform(500.0, 900.0, n),
        "inclination": rng.uniform(0.0, np.pi, n),
        "phase": rng.uniform(0.0, 2 * np.pi, n),
        "ra": rng.uniform(0.0, 2 * np.pi, n),
        "dec": rng.uniform(-1.4, 1.4, n),
        "psi": rng.uniform(0.0, np.pi, n),
        "geocent_time": T_C,
    }
    out = make_injections(wf, batch, key=jax.random.PRNGKey(0), add_noise=False, **kw)
    assert set(out) == {"H1", "L1"}
    assert out["H1"].shape[0] == n

    for i in range(n):
        params = {k: (v if np.isscalar(v) else float(v[i])) for k, v in batch.items()}
        serial = make_injection(wf, params, noise_seed=None, **kw)
        for det in ("H1", "L1"):
            mm = _psd_weighted_mismatch(
                serial, np.asarray(serial.data_fd[det]), np.asarray(out[det][i]), det
            )
            assert mm < 1e-6, f"{model} {det} injection {i}: mismatch {mm:.3e}"


def test_batched_noise_is_per_injection_and_per_detector():
    from jaxpe.gw import IMRPhenomD, make_injections

    n = 4
    batch = {
        "chirp_mass": np.full(n, 30.0),
        "mass_ratio": np.full(n, 0.8),
        "spin1z": np.zeros(n),
        "spin2z": np.zeros(n),
        "luminosity_distance": np.full(n, 700.0),
        "inclination": np.zeros(n),
        "phase": np.zeros(n),
        "ra": np.zeros(n),
        "dec": np.zeros(n),
        "psi": np.zeros(n),
        "geocent_time": T_C,
    }
    kw = dict(
        detector_names=("H1", "L1"), duration=4.0, sampling_rate=1024.0, f_min=30.0
    )
    wf = IMRPhenomD(f_ref=20.0)
    clean = make_injections(wf, batch, key=jax.random.PRNGKey(0), add_noise=False, **kw)
    noisy = make_injections(wf, batch, key=jax.random.PRNGKey(0), add_noise=True, **kw)

    # Identical parameters, so any difference between rows is the noise alone.
    for det in ("H1", "L1"):
        resid = np.asarray(noisy[det]) - np.asarray(clean[det])
        for i in range(1, n):
            assert _differ(resid[0], resid[i]), f"{det}: injections share noise"
    assert _differ(
        np.asarray(noisy["H1"])[0] - np.asarray(clean["H1"])[0],
        np.asarray(noisy["L1"])[0] - np.asarray(clean["L1"])[0],
    ), "detectors share noise"


def test_batched_rejects_what_it_cannot_do():
    from jaxpe.gw import IMRPhenomD, make_injections

    n = 4
    base = {
        "chirp_mass": np.full(n, 30.0),
        "mass_ratio": np.full(n, 0.8),
        "spin1z": np.zeros(n),
        "spin2z": np.zeros(n),
        "luminosity_distance": np.full(n, 700.0),
        "inclination": np.zeros(n),
        "phase": np.zeros(n),
        "ra": np.zeros(n),
        "dec": np.zeros(n),
        "psi": np.zeros(n),
        "geocent_time": T_C,
    }
    kw = dict(detector_names=("H1",), duration=4.0, sampling_rate=1024.0, f_min=30.0)
    wf = IMRPhenomD(f_ref=20.0)
    key = jax.random.PRNGKey(0)

    # vmap needs one grid, so the trigger time cannot vary across the batch.
    with pytest.raises(ValueError, match="one analysis grid"):
        make_injections(wf, {**base, "geocent_time": T_C + np.arange(n)}, key=key, **kw)
    # ragged batch
    with pytest.raises(ValueError, match="leading axis"):
        make_injections(wf, {**base, "chirp_mass": np.full(2, 30.0)}, key=key, **kw)
    # a long segment must fail with an estimate, not inside XLA
    with pytest.raises(ValueError, match="GiB"):
        make_injections(
            wf,
            base,
            key=key,
            detector_names=("H1",),
            duration=2048.0,
            sampling_rate=4096.0,
            f_min=10.0,
            max_bytes=1 << 28,
        )
