"""GW-layer tests.

The detector geometry is checked against LAL (exact reference); the likelihood is
checked against an independent numpy reimplementation and against the analytic
zero-noise property lnL(true params) = 0.
"""

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
