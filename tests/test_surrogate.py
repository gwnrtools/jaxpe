"""Surrogate-engine seam tests (GPry-fusion Phase 1, docs/gpry_fusion_design.md 9).

Task 1.1: the GPryEngine wrapper must reproduce GPry's own introductory-example
behavior through the jaxpe seam -- active-learning a 2D Gaussian likelihood to
convergence in O(20) truth evaluations and recovering its analytic posterior.

Task 1.3 (gate G1): a pseudo-black-box waveform model driven end-to-end --
theta_int -> modes -> marginalized lnL -> GPry active learning -> surrogate
posterior -- compared against a dense-grid ground truth of the *same* marginal
(exact in 2D; no MCMC error on the reference side).
"""

import numpy as np
import pytest

pytest.importorskip("gpry")

import jax.numpy as jnp

from jaxpe.gw import make_injection, spin_weighted_ylm
from jaxpe.gw.external_models import ModesData, reflect_modes
from jaxpe.gw.marginalized import (
    MarginalizedIntrinsicLikelihood,
    ModesNetworkLikelihood,
)
from jaxpe.surrogate import GPryEngine, SurrogateEngine

MEAN = np.array([3.0, 2.0])
COV = np.array([[0.5, 0.4], [0.4, 1.5]])


@pytest.fixture(scope="module")
def gaussian_engine():
    from scipy.stats import multivariate_normal

    rv = multivariate_normal(MEAN, COV)
    calls = {"n": 0}

    def loglike(x):
        calls["n"] += 1
        return rv.logpdf(x)

    engine = GPryEngine(
        loglike,
        bounds={"a": (-10.0, 10.0), "b": (-10.0, 10.0)},
        options={"seed": 3},
        verbose=0,
    )
    diag = engine.run()
    return engine, diag, calls


def test_engine_protocol_and_run(gaussian_engine):
    engine, diag, calls = gaussian_engine
    assert isinstance(engine, SurrogateEngine)
    assert engine.names == ("a", "b")
    assert diag["has_run"] and diag["has_converged"]
    # the whole point: convergence in O(10) truth calls, not O(10^4)
    assert diag["n_truth_evals"] < 100, diag
    assert calls["n"] >= diag["n_truth_evals"]


def test_engine_posterior_recovery(gaussian_engine):
    engine, _, _ = gaussian_engine
    s = engine.sample()
    assert s.x.shape[1] == 2 and len(s.weights) == len(s.x) == len(s.logpost)
    m = np.average(s.x, weights=s.weights, axis=0)
    c = np.cov(s.x.T, aweights=s.weights)
    np.testing.assert_allclose(m, MEAN, atol=0.2)
    np.testing.assert_allclose(c, COV, atol=0.3)


def test_engine_surrogate_logp_accuracy(gaussian_engine):
    """Near the mode, the GP must match the true log-posterior closely; and
    true_logp must be the exact truth (it re-calls the expensive likelihood)."""
    from scipy.stats import multivariate_normal

    engine, _, _ = gaussian_engine
    rv = multivariate_normal(MEAN, COV)
    rng = np.random.default_rng(0)
    x = rng.multivariate_normal(MEAN, COV, size=20)
    # same arbitrary normalization on both sides: compare *differences* across points
    gp = engine.surrogate_logp(x)
    truth = engine.true_logp(x)
    np.testing.assert_allclose(truth - truth[0], rv.logpdf(x) - rv.logpdf(x[0]),
                               atol=1e-10)
    err = (gp - gp[0]) - (truth - truth[0])
    assert np.max(np.abs(err)) < 0.1, f"GP-truth mismatch near mode: {err}"

# ------------------------------------------------- 1.3 pseudo-black-box (gate G1)

T_C = 1126259462.4
DURATION, SR, POST_TRIGGER, D_REF = 8.0, 2048.0, 2.0, 500.0
TRUTH = dict(f0=37.0, span=55.0)  # injected intrinsic point
BOUNDS = {"f0": (30.0, 45.0), "span": (40.0, 80.0)}
EXTRINSIC = dict(
    inclination=0.6, phase=1.2, luminosity_distance=D_REF,
    ra=1.95, dec=-1.27, psi=0.82, geocent_time=T_C,
)
# identical inner settings on the truth and surrogate sides: only GPry is under test
INNER = dict(n_phi=64, n_dist=48, tc_half_samples=3, dist_min=100.0, dist_max=5000.0)


def _chirp_modes(times, theta):
    """A 2-intrinsic-parameter pseudo-waveform: linear chirp from f0 over a
    fixed-width span (Hz), Hann^2 compact support (as in test_marginalized)."""
    t = times - T_C
    t_on, t_off = -1.5, -0.1
    u = np.clip((t - t_on) / (t_off - t_on), 0.0, 1.0)
    env = np.where((t > t_on) & (t < t_off), np.sin(np.pi * u) ** 2, 0.0)
    tau = t - t_on
    phase = 2.0 * np.pi * (
        theta["f0"] * tau + 0.5 * theta["span"] / (t_off - t_on) * tau**2
    )
    h22 = 1e-22 * env * np.exp(-1j * phase)
    h33 = 0.4e-22 * env * np.exp(-1j * 1.5 * phase)
    return reflect_modes({(2, 2): h22, (3, 3): h33})


class _FixedModesWaveform:
    """Traceable (h+, hx) assembler for the injection only (t_c == T_C)."""

    def __init__(self, modes, times):
        self.modes = {lm: jnp.asarray(h) for lm, h in modes.items()}
        self.n = len(times)

    def __call__(self, params, times):
        h = jnp.zeros((self.n,), dtype=jnp.complex128)
        for (l, m), hlm in self.modes.items():
            h = h + hlm * spin_weighted_ylm(params["inclination"], params["phase"], l, m)
        h = h * (D_REF / params["luminosity_distance"])
        return h.real, -h.imag


@pytest.fixture(scope="module")
def pseudo_blackbox():
    n = int(DURATION * SR)
    times = T_C + POST_TRIGGER - DURATION + np.arange(n) / SR

    def mode_model(theta):
        return ModesData(
            modes=_chirp_modes(times, theta), times=times, d_ref_mpc=D_REF, t_ref=T_C
        )

    md_true = mode_model(TRUTH)
    like_td = make_injection(
        _FixedModesWaveform(md_true.modes, times),
        EXTRINSIC,
        detector_names=("H1", "L1"),
        duration=DURATION,
        sampling_rate=SR,
        post_trigger=POST_TRIGGER,
        noise_seed=None,
    )
    like_modes = ModesNetworkLikelihood.from_likelihood(like_td, md_true)
    lik = MarginalizedIntrinsicLikelihood(
        mode_model,
        like_modes,
        names=tuple(BOUNDS),
        t_center=T_C,
        marginalize_sky=False,  # 3D marginal at fixed sky: G1 tests the GPry layer
        fixed_extrinsic=EXTRINSIC,
        settings=INNER,
    )
    return lik, like_modes


def test_g1_pseudo_blackbox_recovery(pseudo_blackbox):
    """Gate G1 (CI form): GPry-learned surrogate posterior of the marginalized
    intrinsic likelihood must match the dense-grid posterior of the same function."""
    lik, like_modes = pseudo_blackbox

    engine = GPryEngine(lik, bounds=BOUNDS, options={"seed": 11}, verbose=0)
    diag = engine.run()
    assert diag["has_converged"], diag
    assert diag["n_truth_evals"] < 300, diag

    # ground truth: dense grid of the same callable (exact posterior in 2D).
    # Two stages: a coarse scan locates the peak, then a zoomed grid resolves it --
    # a single coarse grid would carry O(cell^2/12) variance bias when the posterior
    # width is comparable to the cell size.
    def grid_moments(f0_lo, f0_hi, sp_lo, sp_hi, n=41):
        f0g = np.linspace(f0_lo, f0_hi, n)
        spg = np.linspace(sp_lo, sp_hi, n)
        lnl = np.array([[lik([a, b]) for b in spg] for a in f0g])
        w = np.exp(lnl - lnl.max())
        w /= w.sum()
        gf, gs = np.meshgrid(f0g, spg, indexing="ij")
        mean = np.array([np.sum(w * gf), np.sum(w * gs)])
        var = np.array(
            [np.sum(w * (gf - mean[0]) ** 2), np.sum(w * (gs - mean[1]) ** 2)]
        )
        return mean, var, (f0g, spg, gf, gs, lnl)

    mean_c, var_c, _ = grid_moments(*BOUNDS["f0"], *BOUNDS["span"])
    lo = mean_c - 6.0 * np.sqrt(var_c)
    hi = mean_c + 6.0 * np.sqrt(var_c)
    lo = np.maximum(lo, [BOUNDS["f0"][0], BOUNDS["span"][0]])
    hi = np.minimum(hi, [BOUNDS["f0"][1], BOUNDS["span"][1]])
    mean_grid, var_grid, (f0g, spg, gf, gs, lnl) = grid_moments(
        lo[0], hi[0], lo[1], hi[1], n=61
    )

    s = engine.sample()
    mean_gp = np.average(s.x, weights=s.weights, axis=0)
    var_gp = np.average((s.x - mean_gp) ** 2, weights=s.weights, axis=0)

    cell = np.array([f0g[1] - f0g[0], spg[1] - spg[0]])
    assert np.all(np.abs(mean_gp - mean_grid) < np.maximum(cell, 0.5 * np.sqrt(var_grid))), (
        mean_gp, mean_grid,
    )
    # width tolerance: GPry's convergence criterion targets few-percent lnL accuracy
    # near the mode (~10-15 percent width error), plus NS sampling noise of the
    # surrogate MC step; 30 percent is the acceptance line, not the expectation
    np.testing.assert_allclose(np.sqrt(var_gp), np.sqrt(var_grid), rtol=0.30)

    # the injected truth must lie inside the recovered 3-sigma region
    assert np.all(
        np.abs(mean_gp - np.array([TRUTH["f0"], TRUTH["span"]])) < 3.0 * np.sqrt(var_gp)
    )

    # GP accuracy where it matters: within a few e-folds of the peak
    top = lnl.max() - lnl < 5.0
    pts = np.stack([gf[top], gs[top]], axis=1)
    gp_vals = engine.surrogate_logp(pts)
    err = (gp_vals - gp_vals[0]) - (lnl[top] - lnl[top][0])
    assert np.max(np.abs(err)) < 0.5, f"GP mismatch near peak: max {np.max(np.abs(err)):.3f}"

    # the compiled marginal evaluator was shared across all evaluations
    n_eval_keys = sum(
        1 for k in like_modes._cache if isinstance(k, tuple) and k[0] == "marginal_eval"
    )
    assert n_eval_keys == 1, "per-point recompilation detected"
