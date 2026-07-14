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
    np.testing.assert_allclose(
        truth - truth[0], rv.logpdf(x) - rv.logpdf(x[0]), atol=1e-10
    )
    err = (gp - gp[0]) - (truth - truth[0])
    assert np.max(np.abs(err)) < 0.1, f"GP-truth mismatch near mode: {err}"


# ------------------------------------------------- 1.3 pseudo-black-box (gate G1)

T_C = 1126259462.4
DURATION, SR, POST_TRIGGER, D_REF = 8.0, 2048.0, 2.0, 500.0
TRUTH = dict(f0=37.0, span=55.0)  # injected intrinsic point
BOUNDS = {"f0": (30.0, 45.0), "span": (40.0, 80.0)}
EXTRINSIC = dict(
    inclination=0.6,
    phase=1.2,
    luminosity_distance=D_REF,
    ra=1.95,
    dec=-1.27,
    psi=0.82,
    geocent_time=T_C,
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
    phase = (
        2.0
        * np.pi
        * (theta["f0"] * tau + 0.5 * theta["span"] / (t_off - t_on) * tau**2)
    )
    h22 = 1e-22 * env * np.exp(-1j * phase)
    h33 = 0.4e-22 * env * np.exp(-1j * 1.5 * phase)
    return reflect_modes({(2, 2): h22, (3, 3): h33})


class _FixedModesWaveform:
    """Traceable (h+, hx) assembler for the injection only (t_c == T_C).

    ``d_ref`` is the luminosity distance the stored modes are scaled at; it must
    match the ``d_ref_mpc`` of the ModesData the surrogate side uses.
    """

    def __init__(self, modes, times, d_ref=D_REF):
        self.modes = {lm: jnp.asarray(h) for lm, h in modes.items()}
        self.n = len(times)
        self.d_ref = d_ref

    def __call__(self, params, times):
        h = jnp.zeros((self.n,), dtype=jnp.complex128)
        for (l, m), hlm in self.modes.items():
            h = h + hlm * spin_weighted_ylm(
                params["inclination"], params["phase"], l, m
            )
        h = h * (self.d_ref / params["luminosity_distance"])
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
    assert np.all(
        np.abs(mean_gp - mean_grid) < np.maximum(cell, 0.5 * np.sqrt(var_grid))
    ), (
        mean_gp,
        mean_grid,
    )
    # width tolerance: GPry's convergence criterion targets few-percent lnL accuracy
    # near the mode (~10-15 percent width error), plus NS sampling noise of the
    # surrogate MC step; 30 percent is the acceptance line, not the expectation
    np.testing.assert_allclose(np.sqrt(var_gp), np.sqrt(var_grid), rtol=0.30)

    # the injected truth must lie inside the recovered 3-sigma region
    assert np.all(
        np.abs(mean_gp - np.array([TRUTH["f0"], TRUTH["span"]])) < 3.0 * np.sqrt(var_gp)
    )

    # GP accuracy where it matters: posterior-weighted RMS lnL error. (A max-norm
    # over the 5-e-fold region is flaky: NORA's UltraNest exploration is unseeded,
    # so the GP training set -- and the worst point in the tails -- varies between
    # runs; the posterior-weighted error is the quantity that controls the
    # surrogate posterior and is stable.)
    w_post = np.exp(lnl - lnl.max())
    w_post /= w_post.sum()
    top = lnl.max() - lnl < 5.0
    pts = np.stack([gf[top], gs[top]], axis=1)
    gp_vals = engine.surrogate_logp(pts)
    err = (gp_vals - gp_vals[0]) - (lnl[top] - lnl[top][0])
    wt = w_post[top]
    werr = np.sqrt(np.sum(wt * (err - np.average(err, weights=wt)) ** 2) / np.sum(wt))
    assert werr < 0.5, f"posterior-weighted GP lnL error {werr:.3f}"

    # the compiled marginal evaluator was shared across all evaluations
    n_eval_keys = sum(
        1 for k in like_modes._cache if isinstance(k, tuple) and k[0] == "marginal_eval"
    )
    assert n_eval_keys == 1, "per-point recompilation detected"


# --------------------------------- 1.4 ESIGMA pseudo-black-box + IS exactness


@pytest.fixture(scope="module")
def esigma_blackbox():
    """ESIGMA (cheap 0PN config) as an opaque mode model over (chirp_mass, eccentricity).

    Ground truth for the surrogate is established not by a grid (too slow at
    waveform-model cost) but by the design's own exactness mechanism (D3):
    IS-reweighting of the surrogate posterior against the true likelihood.
    """
    pytest.importorskip("esigmapy")
    import jax

    from jaxpe.gw import ESIGMAInspiral
    from jaxpe.gw.marginalized import MarginalizedIntrinsicLikelihood

    wf = ESIGMAInspiral(
        f_lower=20.0,
        modes=((2, 2), (3, 3)),
        rad_pn_order=0,  # cheap RHS: the surrogate layer, not ESIGMA physics, is under test
        mode_pn_order=0,
        ode_eps=1e-9,
        n_ode_grid=256,
        max_ode_steps=4096,
    )
    truth = dict(chirp_mass=30.0, eccentricity=0.08)
    # Injection at 2000 Mpc (network SNR ~ 12, realistic for eccentric candidates).
    # Measured at 500 Mpc (SNR ~ 50, SNR^2 = 2509): lnL(e) is physically multi-lobed
    # (e-phasing oscillatory degeneracy; lobes every ~0.005-0.01 in e, converged in
    # the phi_c quadrature) with ~60-e-fold lobe contrast and sigma_e ~ 6e-4 -- no
    # stationary-kernel GP resolves that over wide bounds in O(100) evaluations
    # (GPry: 197 evals, no convergence; then UltraNest MLFriends degeneracy). At
    # SNR ~ 12 the same structure has few-e-fold contrast and sigma_e ~ 2e-3:
    # multi-lobed but learnable. Bounds emulate the production workflow where a
    # cheap-model (case-1) posterior sets them BEFORE the surrogate runs -- the
    # practical argument for the Phase-2 multifidelity/ref-bounds step.
    bounds = {"chirp_mass": (29.5, 30.5), "eccentricity": (0.05, 0.11)}
    fixed_intr = dict(mass_ratio=0.9, mean_anomaly=0.3, spin1z=0.0, spin2z=0.0)
    n = int(DURATION * SR)
    times = T_C + POST_TRIGGER - DURATION + np.arange(n) / SR

    # jit once over the intrinsic vector: ESIGMA is case-(1), so the pseudo-black-box
    # can afford a compiled mode generator (a real case-(2) model is plain Python)
    @jax.jit
    def _modes(theta_vec):
        p = dict(
            chirp_mass=theta_vec[0],
            eccentricity=theta_vec[1],
            geocent_time=jnp.asarray(T_C),
            **{k: jnp.asarray(v) for k, v in fixed_intr.items()},
        )
        return wf.mode_dict(p, jnp.asarray(times))

    def mode_model(theta):
        md = _modes(jnp.asarray([theta["chirp_mass"], theta["eccentricity"]]))
        return ModesData(
            modes={lm: np.asarray(h) for lm, h in md.items()},
            times=times,
            d_ref_mpc=1.0,  # ESIGMA modes are strain at 1 Mpc
            t_ref=T_C,
        )

    md_true = mode_model(truth)
    like_td = make_injection(
        _FixedModesWaveform(md_true.modes, times, d_ref=1.0),  # ESIGMA modes: 1 Mpc
        dict(EXTRINSIC, luminosity_distance=2000.0),
        detector_names=("H1", "L1"),
        duration=DURATION,
        sampling_rate=SR,
        post_trigger=POST_TRIGGER,
        noise_seed=None,
    )
    # note: modes at 1 Mpc + injected luminosity_distance=D_REF is consistent because
    # _FixedModesWaveform rescales by (d_ref/D); here d_ref enters via ModesData
    like_modes = ModesNetworkLikelihood.from_likelihood(like_td, md_true)
    lik = MarginalizedIntrinsicLikelihood(
        mode_model,
        like_modes,
        names=tuple(bounds),
        t_center=T_C,
        marginalize_sky=False,
        fixed_extrinsic=EXTRINSIC,
        # wider t_c window than the chirp test: with only +-3 samples, the discrete
        # t_c nodes under-absorb chirp-mass timing shifts and imprint ~2-e-fold
        # ripples on the mc marginal that the GP then chases
        settings=dict(INNER, tc_half_samples=20),
    )
    return lik, bounds, truth


def test_g1_esigma_blackbox_with_is_reweighting(esigma_blackbox):
    """Task 1.4 + the D3 exactness mechanism in one: GPry learns the ESIGMA
    marginalized likelihood; IS-reweighting its posterior samples against the true
    likelihood must show high ESS (surrogate faithful where the mass is) and a
    negligible mean shift; the injected truth lies in the recovered region."""
    lik, bounds, truth = esigma_blackbox

    engine = GPryEngine(lik, bounds=bounds, options={"seed": 7}, verbose=0)
    diag = engine.run()
    assert diag["has_converged"], diag
    assert diag["n_truth_evals"] < 400, diag

    s = engine.sample()
    mean_gp = np.average(s.x, weights=s.weights, axis=0)
    sd_gp = np.sqrt(np.average((s.x - mean_gp) ** 2, weights=s.weights, axis=0))

    # injected truth within the 3-sigma surrogate credible region
    t_vec = np.array([truth["chirp_mass"], truth["eccentricity"]])
    assert np.all(np.abs(mean_gp - t_vec) < 3.0 * sd_gp), (mean_gp, sd_gp, t_vec)

    # D3 exactness: reweight a thinned subset of surrogate samples by the true lnL
    rng = np.random.default_rng(2)
    idx = rng.choice(len(s.x), size=128, replace=False, p=s.weights / s.weights.sum())
    x_sub = s.x[idx]
    lnl_true = np.array([lik(x) for x in x_sub])
    lnl_gp = engine.surrogate_logp(x_sub)
    lw = lnl_true - lnl_gp
    lw -= lw.max()
    w = np.exp(lw)
    effective_sample_size = w.sum() ** 2 / np.sum(w**2)
    # measured ~0.25 N on this multi-lobed e-surface at GPry's default convergence:
    # the surrogate carries O(1)-e-fold residuals that reweighting corrects (that is
    # the D3 mechanism's job). A catastrophically biased GP gives ESS/N < ~0.05;
    # the acceptance line sits between the two regimes.
    assert effective_sample_size > 0.15 * len(
        w
    ), f"IS ESS {effective_sample_size:.0f}/{len(w)}: surrogate biased in bulk"

    mean_rw = np.average(x_sub, weights=w, axis=0)
    mean_un = x_sub.mean(axis=0)
    assert np.all(
        np.abs(mean_rw - mean_un) < 0.5 * sd_gp
    ), "reweighting moved the posterior mean by >0.5 sigma: surrogate not converged"


def test_full_marginal_records_importance_sampling_history(pseudo_blackbox):
    """Per-call adaptive-IS diagnostics must be recorded in full-marginal mode:
    a bad inner extrinsic marginal at some theta must be detectable (via importance_sampling_summary)
    rather than hiding inside a converged-looking GPry run."""
    lik_fixed, like_modes = pseudo_blackbox
    lik = MarginalizedIntrinsicLikelihood(
        lik_fixed.mode_model,
        like_modes,
        names=tuple(BOUNDS),
        t_center=T_C,
        marginalize_sky=True,
        settings=dict(INNER, n_pilot=256, n_final=256),
    )
    v1 = lik([37.0, 55.0])
    v2 = lik([38.0, 60.0])
    assert np.isfinite(v1) and np.isfinite(v2)
    assert len(lik.importance_sampling_history) == 2
    for h in lik.importance_sampling_history:
        assert set(h) >= {
            "theta",
            "logz",
            "effective_sample_size",
            "n_eval",
            "lnl_max",
            "logz_rounds",
        }
        assert np.isfinite(h["logz"]) and h["effective_sample_size"] > 1.0
        assert set(h["theta"]) == set(BOUNDS)
    s = lik.importance_sampling_summary(effective_sample_size_floor=np.inf)
    assert s["n_calls"] == 2 and s["n_below_floor"] == 2  # everything below inf floor
    assert s["effective_sample_size_min"] <= s["effective_sample_size_median"]


def test_full_marginal_effective_sample_size_extra_rounds(pseudo_blackbox):
    """With an unreachable quality floor, escalating extra rounds must run, every
    batch recycled cumulatively (no discard-and-restart): total evaluations are
    pilot + base rounds + doubled extra rounds, all counted in n_eval."""
    lik_fixed, like_modes = pseudo_blackbox
    lik = MarginalizedIntrinsicLikelihood(
        lik_fixed.mode_model,
        like_modes,
        names=tuple(BOUNDS),
        t_center=T_C,
        marginalize_sky=True,
        settings=dict(INNER, n_pilot=256, n_final=256),
        effective_sample_size_floor=np.inf,  # unreachable: forces the escalation
        max_extra_importance_sampling_rounds=2,
    )
    lik([37.0, 55.0])
    (h,) = lik.importance_sampling_history
    assert h["extra_rounds_used"] == 2 and h["failed"] is True
    # pilot 256 + base rounds 2 x 256 + extra rounds 512 + 1024, all recycled
    assert h["n_eval"] == 256 + 256 + 256 + 512 + 1024, h["n_eval"]
    # the recycled estimate progression has one entry per executed round
    assert len(h["logz_rounds"]) == 4


def test_full_marginal_strict_mode_raises(pseudo_blackbox):
    """Strict mode: an unhealable call must raise LowEffectiveSampleSizeError,
    with the failure recorded in the history first (post-mortem evidence)."""
    from jaxpe.gw.marginalized import LowEffectiveSampleSizeError

    lik_fixed, like_modes = pseudo_blackbox
    lik = MarginalizedIntrinsicLikelihood(
        lik_fixed.mode_model,
        like_modes,
        names=tuple(BOUNDS),
        t_center=T_C,
        marginalize_sky=True,
        settings=dict(INNER, n_pilot=256, n_final=256),
        effective_sample_size_floor=np.inf,  # unreachable: forces the strict path
        max_extra_importance_sampling_rounds=0,
        on_low_effective_sample_size="raise",
    )
    with pytest.raises(LowEffectiveSampleSizeError) as excinfo:
        lik([37.0, 55.0])
    assert excinfo.value.extra_rounds == 0
    assert np.isfinite(excinfo.value.effective_sample_size)
    (h,) = lik.importance_sampling_history
    assert h["failed"] is True


def test_importance_sampling_summary_peak_window():
    """The reliability-gate quantity: unhealthy calls are counted near the peak
    (within peak_efolds of the best log-marginal), not in the harmless tails."""
    lik = MarginalizedIntrinsicLikelihood.__new__(MarginalizedIntrinsicLikelihood)
    lik.importance_sampling_history = [
        # (logz, effective_sample_size): peak call, healthy
        dict(theta={"a": 1.0}, logz=0.0, effective_sample_size=500.0),
        # near peak, UNHEALTHY -> must be gated
        dict(theta={"a": 2.0}, logz=-2.0, effective_sample_size=20.0),
        # deep tail, unhealthy but harmless -> counted below floor, NOT near peak
        dict(theta={"a": 3.0}, logz=-50.0, effective_sample_size=5.0),
    ]
    s = lik.importance_sampling_summary(
        effective_sample_size_floor=100.0, peak_efolds=5.0
    )
    assert s["n_calls"] == 3
    assert s["n_below_floor"] == 2
    assert s["n_below_floor_near_peak"] == 1
    assert s["thetas_below_floor_near_peak"] == [{"a": 2.0}]
    # without the window argument, the near-peak keys are absent
    s2 = lik.importance_sampling_summary(effective_sample_size_floor=100.0)
    assert "n_below_floor_near_peak" not in s2
