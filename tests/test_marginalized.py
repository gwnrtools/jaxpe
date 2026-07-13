"""Phase-0.2 parity tests: ModesNetworkLikelihood vs TDNetworkLikelihood.

Strategy (docs/gpry_fusion_design.md section 9, task 0.2): build synthetic
compact-support modes, wrap them in a WaveformModel that assembles (h+, hx) with the
same mode-sum convention as ESIGMAInspiral, and require the modes-based likelihood to
reproduce the direct time-domain path at fixed extrinsic parameters.

Exactness notes
---------------
* At t_c == t_ref both paths window and FFT literally the same time series, so
  agreement is at float64 round-off.
* For t_c != t_ref the modes path applies an FD phase shift while the TD wrapper
  rolls the series by an integer number of samples: both are exact circular shifts,
  and they commute with the Tukey window because the signal is compactly supported
  in the window's flat region. Non-integer shifts differ by genuine (sub-sample
  periodic-sinc vs re-evaluation) interpolation and are exercised only as a
  smoothness check, not exact parity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpe.gw import make_injection, spin_weighted_ylm
from jaxpe.gw.external_models import ModeCache, ModesData, reflect_modes
from jaxpe.gw.marginalized import ModesNetworkLikelihood

T_C = 1126259462.4
DURATION = 8.0
SR = 2048.0
POST_TRIGGER = 2.0
D_REF = 500.0  # Mpc


def _analysis_times():
    """Replicate make_injection's grid so modes can be built on it."""
    n = int(DURATION * SR)
    t_start = T_C + POST_TRIGGER - DURATION
    return t_start + np.arange(n) * (1.0 / SR)


def _synthetic_modes(times):
    """Compact-support complex chirp modes (strain at D_REF), aligned to t_c = T_C.

    A Hann^2 envelope with hard support [T_C - 1.5 s, T_C - 0.1 s] keeps the signal
    strictly inside the flat region of the alpha=0.1 Tukey window (ramps ~0.4 s at
    the segment edges), which is what makes integer-sample-shift parity exact.
    """
    t = times - T_C
    t_on, t_off = -1.5, -0.1
    u = np.clip((t - t_on) / (t_off - t_on), 0.0, 1.0)
    env = np.where((t > t_on) & (t < t_off), np.sin(np.pi * u) ** 2, 0.0)
    # linear chirp 35 -> 95 Hz across the support: safely inside the 20 Hz--Nyquist band
    f0, f1 = 35.0, 95.0
    tau = t - t_on
    phase = 2.0 * np.pi * (f0 * tau + 0.5 * (f1 - f0) / (t_off - t_on) * tau**2)
    h22 = 1e-22 * env * np.exp(-1j * phase)
    h33 = 0.4e-22 * env * np.exp(-1j * 1.5 * phase)  # 3/2 x the (2,2) phase
    return reflect_modes({(2, 2): h22, (3, 3): h33})


class _ModesWaveform:
    """WaveformModel wrapper assembling (h+, hx) from fixed modes.

    Uses the same mode sum as ESIGMAInspiral.__call__ (all m explicit, hp = Re h,
    hc = -Im h) and realizes geocent_time by rounding to an integer-sample jnp.roll:
    the exact TD counterpart of the modes path's FD phase shift (both are circular
    shifts). Traceable, so make_injection's jitted injection call works; only ever
    evaluate it at integer-sample t_c offsets.
    """

    def __init__(self, modes, times, t_ref, d_ref):
        self.modes = {lm: jnp.asarray(h) for lm, h in modes.items()}
        self.grid_times, self.t_ref, self.d_ref = times, t_ref, d_ref
        self.sr = 1.0 / (times[1] - times[0])

    def __call__(self, params, times):
        iota, phi = params["inclination"], params["phase"]
        h = jnp.zeros(self.grid_times.shape, dtype=jnp.complex128)
        for (l, m), hlm in self.modes.items():
            h = h + hlm * spin_weighted_ylm(iota, phi, l, m)
        h = h * (self.d_ref / params["luminosity_distance"])
        k = jnp.round((params["geocent_time"] - self.t_ref) * self.sr).astype(jnp.int32)
        h = jnp.roll(h, k)
        return h.real, -h.imag


INJ = dict(
    inclination=0.6,
    phase=1.2,
    luminosity_distance=D_REF,
    ra=1.95,
    dec=-1.27,
    psi=0.82,
    geocent_time=T_C,
)


@pytest.fixture(scope="module")
def likelihood_pair():
    times = _analysis_times()
    modes = _synthetic_modes(times)
    wf = _ModesWaveform(modes, times, T_C, D_REF)
    like_td = make_injection(
        wf,
        INJ,
        detector_names=("H1", "L1", "V1"),
        duration=DURATION,
        sampling_rate=SR,
        post_trigger=POST_TRIGGER,
        noise_seed=None,
    )
    np.testing.assert_allclose(like_td.times, times, rtol=0, atol=1e-9)
    md = ModesData(modes=modes, times=times, d_ref_mpc=D_REF, t_ref=T_C)
    like_modes = ModesNetworkLikelihood.from_likelihood(like_td, md)
    return like_td, like_modes


def _params(**overrides):
    p = {k: jnp.asarray(v) for k, v in INJ.items()}
    p.update({k: jnp.asarray(v) for k, v in overrides.items()})
    return p


def test_zero_noise_lnl_and_snr(likelihood_pair):
    """Sanity: injection recovered at truth; SNR in a reasonable range."""
    _, like_modes = likelihood_pair
    assert abs(float(like_modes.log_likelihood(_params()))) < 1e-6
    snrs = like_modes.optimal_snr(_params())
    assert all(3.0 < s < 200.0 for s in snrs.values()), snrs


def test_parity_at_reference_time(likelihood_pair):
    """Identical windowed-FFT path => agreement at float64 round-off, incl. strains."""
    like_td, like_modes = likelihood_pair
    rng = np.random.default_rng(1)
    for _ in range(8):
        p = _params(
            inclination=rng.uniform(0, np.pi),
            phase=rng.uniform(0, 2 * np.pi),
            luminosity_distance=rng.uniform(200.0, 2000.0),
            ra=rng.uniform(0, 2 * np.pi),
            dec=rng.uniform(-np.pi / 2, np.pi / 2),
            psi=rng.uniform(0, np.pi),
        )
        s_td = like_td.detector_strains_fd(p)
        s_m = like_modes.detector_strains_fd(p)
        for name in s_td:
            a, b = np.asarray(s_td[name]), np.asarray(s_m[name])
            scale = np.max(np.abs(a))
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12 * scale)
        lnl_td = float(like_td.log_likelihood(p))
        lnl_m = float(like_modes.log_likelihood(p))
        assert abs(lnl_td - lnl_m) < 1e-8 * max(1.0, abs(lnl_td))


def test_parity_at_shifted_time(likelihood_pair):
    """FD phase shift vs TD integer-sample roll: exact circular-shift parity."""
    like_td, like_modes = likelihood_pair
    dt = 1.0 / SR
    for k in (-7, -1, 3, 40):
        p = _params(geocent_time=T_C + k * dt, phase=0.7, inclination=1.1)
        lnl_td = float(like_td.log_likelihood(p))
        lnl_m = float(like_modes.log_likelihood(p))
        assert abs(lnl_td - lnl_m) < 1e-8 * max(1.0, abs(lnl_td)), f"k={k}"


def test_subsample_time_shift_smooth(likelihood_pair):
    """Non-integer t_c: lnL must interpolate smoothly between integer-sample values."""
    _, like_modes = likelihood_pair
    dt = 1.0 / SR
    lnl = [
        float(like_modes.log_likelihood(_params(geocent_time=T_C + f * dt)))
        for f in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert all(np.isfinite(lnl))
    # lnL at the two integer-sample endpoints brackets a smooth dip/rise; the midpoint
    # must lie within the (loose) envelope of the endpoints' spread
    spread = abs(lnl[4] - lnl[0]) + 1e-3 * max(1.0, abs(lnl[0]))
    assert abs(lnl[2] - 0.5 * (lnl[0] + lnl[4])) < 5.0 * spread


def test_distance_scaling_exact(likelihood_pair):
    """<h|h> scales as 1/D^2 through the modes path."""
    _, like_modes = likelihood_pair
    s1 = like_modes.optimal_snr(_params(luminosity_distance=400.0))
    s2 = like_modes.optimal_snr(_params(luminosity_distance=800.0))
    for name in s1:
        np.testing.assert_allclose(s1[name], 2.0 * s2[name], rtol=1e-10)


def test_extrinsic_gradients_finite(likelihood_pair):
    """The modes -> lnL map must be differentiable in all extrinsic parameters."""
    _, like_modes = likelihood_pair
    grad = jax.grad(
        lambda p: like_modes.log_likelihood(p),
    )(_params(geocent_time=T_C + 0.37 / SR))
    for k, g in grad.items():
        assert np.isfinite(float(g)), k
    # geocent_time gradient vs central finite differences
    eps = 1e-6
    lp = float(like_modes.log_likelihood(_params(geocent_time=T_C + eps)))
    lm = float(like_modes.log_likelihood(_params(geocent_time=T_C - eps)))
    fd = (lp - lm) / (2 * eps)
    ad = float(
        jax.grad(lambda t: like_modes.log_likelihood(_params(geocent_time=t)))(
            jnp.asarray(T_C)
        )
    )
    assert abs(ad - fd) < 1e-4 * max(1.0, abs(fd))


# --------------------------------------------------------------- 0.3 marginalization


def _erf_log_dist_integral(z, s, u_lo, u_hi, log_prior_norm):
    """Closed form of the u-integral for a prior flat in u (dist_power = -2).

    I = e^{z^2/2s} sqrt(2 pi / s) [Phi(b) - Phi(a)] * pi_u, evaluated in log space
    via scipy's log_ndtr so deeply-truncated tails stay finite.
    """
    from scipy.special import log_ndtr

    sq = np.sqrt(s)
    a, b = (u_lo - z / s) * sq, (u_hi - z / s) * sq
    with np.errstate(divide="ignore"):  # log1p(-1) = -inf for zero-mass tails is fine
        log_phi_diff = log_ndtr(b) + np.log1p(-np.exp(log_ndtr(a) - log_ndtr(b)))
    return log_prior_norm + z**2 / (2 * s) + 0.5 * np.log(2 * np.pi / s) + log_phi_diff


D_MIN, D_MAX = 100.0, 5000.0
# dist_power = -2 makes the prior flat in u = d_ref/D: the one case with a closed form
P_FLAT_U = -2.0


def _log_prior_norm_flat_u():
    log_c = -np.log(abs(1.0 / D_MAX - 1.0 / D_MIN))
    return log_c + (P_FLAT_U + 1.0) * np.log(D_REF)


def test_distance_integral_vs_erf(likelihood_pair):
    """Adaptive GL quadrature vs the analytic erf form, per accuracy domain."""
    _, like = likelihood_pair
    u_lo, u_hi = D_REF / D_MAX, D_REF / D_MIN
    lpn = _log_prior_norm_flat_u()

    def gl(z, s):
        return float(
            like._log_distance_integral(
                jnp.asarray(z), jnp.asarray(s), u_lo, u_hi, lpn, P_FLAT_U, 128
            )
        )

    # peak inside the prior range (the regime that matters): near-exact
    for z, s in [(900.0, 900.0), (450.0, 900.0), (0.5, 1.0), (2000.0, 800.0)]:
        assert abs(gl(z, s) - _erf_log_dist_integral(z, s, u_lo, u_hi, lpn)) < 1e-8

    # peak just outside (within a few sigma_u): still near-exact
    s = 900.0
    for z in [s * (u_hi + 2.0 / np.sqrt(s)), s * (u_lo - 2.0 / np.sqrt(s))]:
        assert abs(gl(z, s) - _erf_log_dist_integral(z, s, u_lo, u_hi, lpn)) < 1e-6

    # deeply boundary-truncated (source outside the prior): documented weak regime,
    # only required to be percent-level in log
    z = s * (u_hi + 150.0 / np.sqrt(s))
    assert abs(gl(z, s) - _erf_log_dist_integral(z, s, u_lo, u_hi, lpn)) < 0.05 * abs(
        _erf_log_dist_integral(z, s, u_lo, u_hi, lpn)
    )


class _FixedGmstModes(ModesNetworkLikelihood):
    """Freeze GMST at gmst_ref: makes brute-force t_c sweeps match the marginal
    path's frozen-GMST convention exactly."""

    def _gmst(self, params):
        return jnp.asarray(self.gmst_ref)


def test_log_marginal_vs_semi_brute_force(likelihood_pair):
    """Independent reconstruction of the (phi_c, t_c, D_L) marginal.

    Per (phi_c, t_c) node, (z, sigma^2, <d|d>/2) are solved exactly from three
    fixed-parameter log_likelihood evaluations at u = 1, 2, 4 (linear system), and
    the distance integral is done with the analytic erf form -- no shared code with
    log_marginal_likelihood except the mode cache. Tests the irfft <d|h>(t_c) trick,
    the node weights, and the overall normalization in one shot.
    """
    like_td, like_modes = likelihood_pair
    md = like_modes.modes_data
    like_fix = _FixedGmstModes.from_likelihood(like_td, md)

    n_phi, m_tc = 8, 5
    dt = 1.0 / SR
    lpn = _log_prior_norm_flat_u()
    u_lo, u_hi = D_REF / D_MAX, D_REF / D_MIN

    u_probe = np.array([1.0, 2.0, 4.0])
    coef = np.stack([u_probe, -0.5 * u_probe**2, -np.ones(3)], axis=1)

    log_terms = []
    for j in range(n_phi):
        for k in range(-m_tc, m_tc + 1):
            lnl = np.array(
                [
                    float(
                        like_fix.log_likelihood(
                            _params(
                                phase=2 * np.pi * j / n_phi,
                                geocent_time=T_C + k * dt,
                                luminosity_distance=D_REF / u,
                            )
                        )
                    )
                    for u in u_probe
                ]
            )
            z, s, half_dd = np.linalg.solve(coef, lnl)
            log_terms.append(_erf_log_dist_integral(z, s, u_lo, u_hi, lpn) - half_dd)
    from scipy.special import logsumexp as np_logsumexp

    lbf = np_logsumexp(log_terms) - np.log(n_phi) - np.log(2 * m_tc + 1)

    lmarg = float(
        like_modes.log_marginal_likelihood(
            _params(),
            n_phi=n_phi,
            n_dist=128,
            tc_half_samples=m_tc,
            dist_min=D_MIN,
            dist_max=D_MAX,
            dist_power=P_FLAT_U,
        )
    )
    assert abs(lmarg - lbf) < 1e-6, f"marginal {lmarg} vs brute force {lbf}"


def test_log_marginal_quadrature_convergence(likelihood_pair):
    """Doubling the phi_c and distance node counts must not move the result.

    The phi_c integrand e^{lnL(phi)} carries harmonics up to ~ max|m| SNR^2/2, so the
    trapezoid needs O(SNR^2) nodes -- the per-harmonic decomposition makes dense
    grids cheap, and at this event's SNR 256 -> 512 must be fully converged.
    """
    _, like_modes = likelihood_pair
    kw = dict(tc_half_samples=5, dist_min=D_MIN, dist_max=D_MAX, dist_power=2.0)
    l256 = float(
        like_modes.log_marginal_likelihood(_params(), n_phi=256, n_dist=64, **kw)
    )
    l512 = float(
        like_modes.log_marginal_likelihood(_params(), n_phi=512, n_dist=128, **kw)
    )
    assert abs(l512 - l256) < 1e-6 * max(1.0, abs(l512))


def test_log_marginal_gradients(likelihood_pair):
    """The marginal must be differentiable in the remaining extrinsic parameters."""
    _, like_modes = likelihood_pair
    kw = dict(n_phi=8, n_dist=64, tc_half_samples=5, dist_min=D_MIN, dist_max=D_MAX)

    def f(iota, ra):
        p = _params(inclination=iota, ra=ra)
        return like_modes.log_marginal_likelihood(p, **kw)

    iota0, ra0 = jnp.asarray(0.6), jnp.asarray(1.95)
    g_iota = float(jax.grad(f, argnums=0)(iota0, ra0))
    g_ra = float(jax.grad(f, argnums=1)(iota0, ra0))
    eps = 1e-5
    fd_iota = (float(f(iota0 + eps, ra0)) - float(f(iota0 - eps, ra0))) / (2 * eps)
    fd_ra = (float(f(iota0, ra0 + eps)) - float(f(iota0, ra0 - eps))) / (2 * eps)
    assert abs(g_iota - fd_iota) < 1e-3 * max(1.0, abs(fd_iota))
    assert abs(g_ra - fd_ra) < 1e-3 * max(1.0, abs(fd_ra))


# ------------------------------------------------------- 0.4 sky/psi/iota layer

# cheap inner settings, identical across every comparison so that inner quadrature
# error cancels and only the outer extrinsic layer is under test
INNER = dict(n_phi=64, n_dist=48, tc_half_samples=3, dist_min=D_MIN, dist_max=D_MAX)


def test_mixture_density_normalized():
    """The defensive KDE proposal must be a proper density the sampler follows.

    (a) Each wrapped/reflected 1D kernel integrates to 1 on [0, 1] (deterministic
    trapezoid; this is the exact normalization statement -- a naive uniform-MC
    check of mean(q) is heavy-tailed for narrow kernels and useless).
    (b) E_q[1/q] = 1 with samples drawn from the mixture itself: 1/q <= 1/defense
    is bounded, so this tightly validates sampler <-> density consistency --
    exactly the property importance sampling relies on.
    """
    from jaxpe.gw.marginalized import (
        _EXT_PERIODIC,
        _mixture_log_density,
        _mixture_sample,
    )

    rng = np.random.default_rng(3)
    centers = np.array(
        [[0.02, 0.5, 0.98, 0.5], [0.5, 0.03, 0.5, 0.97], [0.3, 0.7, 0.2, 0.4]]
    )
    widths = np.array([0.05, 0.02, 0.1, 0.03])
    comp_w = np.array([0.5, 0.3, 0.2])

    # (a) exact per-dim kernel normalization
    x = np.linspace(0.0, 1.0, 20001)
    for d in range(4):
        for c in centers[:, d]:
            h = widths[d]
            imgs = (
                [x - c - 1, x - c, x - c + 1]
                if _EXT_PERIODIC[d]
                else [
                    x - c,
                    x + c,
                    x + c - 2,
                ]
            )
            dens = sum(
                np.exp(-0.5 * (im / h) ** 2) / (np.sqrt(2 * np.pi) * h) for im in imgs
            )
            assert abs(np.trapezoid(dens, x) - 1.0) < 1e-6, (d, c)

    # (b) sampler/density consistency via the bounded estimator E_q[1/q] = 1
    s = _mixture_sample(rng, 100_000, centers, widths, comp_w, defense=0.2)
    assert s.min() >= 0.0 and s.max() <= 1.0
    inv_q = np.exp(-_mixture_log_density(s, centers, widths, comp_w, defense=0.2))
    assert abs(inv_q.mean() - 1.0) < 0.02, f"E_q[1/q] = {inv_q.mean():.4f}"


def test_full_marginal_adaptive_importance_sampling(likelihood_pair):
    """Adaptive-IS extrinsic marginal: independent runs agree within their MC error.

    Plain QMC measurably fails here (ESS 1.5/8192; seed spread ~7 in log), which is
    what forced the adaptive scheme. Two runs with disjoint randomness and different
    budgets must now agree within a few sigma of 1/sqrt(ESS), with healthy ESS.
    """
    _, like_modes = likelihood_pair
    la, da = like_modes.log_marginal_likelihood_full(
        _params(),
        n_pilot=2048,
        n_final=2048,
        qmc_seed=7,
        return_diagnostics=True,
        **INNER,
    )
    lb, db = like_modes.log_marginal_likelihood_full(
        _params(),
        n_pilot=1024,
        n_final=4096,
        qmc_seed=123,
        return_diagnostics=True,
        **INNER,
    )
    assert da["effective_sample_size"] > 100.0, f"run A unconverged: {da}"
    assert db["effective_sample_size"] > 100.0, f"run B unconverged: {db}"
    tol = max(0.15, 5.0 / np.sqrt(min(da["effective_sample_size"], db["effective_sample_size"])))
    assert (
        abs(la - lb) < tol
    ), f"{la} (ESS {da['effective_sample_size']:.0f}) vs {lb} (ESS {db['effective_sample_size']:.0f})"
    # the marginal must sit below the extrinsic-optimum lnL (Occam volume factor)
    assert la < da["lnl_max"]


# --------------------------------------- 0.5 extrinsic-conditional sampler prototype


def test_extrinsic_conditional_mala(likelihood_pair):
    """Prototype of the design-note section-5 extrinsic recovery: sample
    p(theta_ext | modes, d) with jaxpe's MALA kernel, gradients flowing through the
    cached-modes likelihood. Checks the machinery (finite grads in unconstrained
    space, healthy acceptance, chains climbing to the high-likelihood region and
    concentrating near the injected extrinsic parameters).
    """
    from jaxpe.core.priors import Cosine, JointPrior, PowerLaw, Sine, Uniform
    from jaxpe.core.problem import InferenceProblem
    from jaxpe.kernels.base import KernelState
    from jaxpe.kernels.mala import MALA

    _, like_modes = likelihood_pair
    prior = JointPrior(
        {
            "luminosity_distance": PowerLaw(alpha=2.0, low=D_MIN, high=2000.0),
            "inclination": Sine(),
            "phase": Uniform(low=0.0, high=2 * np.pi),
            "ra": Uniform(low=0.0, high=2 * np.pi),
            "dec": Cosine(),
            "psi": Uniform(low=0.0, high=np.pi),
            "geocent_time": Uniform(low=T_C - 0.02, high=T_C + 0.02),
        }
    )
    problem = InferenceProblem(prior=prior, log_likelihood=like_modes.log_likelihood)

    n_chains, n_steps = 16, 250
    key = jax.random.PRNGKey(0)
    k_init, k_run = jax.random.split(key)

    # gradients must be finite everywhere, including prior draws far from the peak
    y_prior = problem.sample_unconstrained(k_init, n_chains)
    logp = jax.vmap(jax.value_and_grad(problem.log_posterior))
    lp_p, g_p = logp(y_prior)
    assert np.all(np.isfinite(np.asarray(lp_p))) and np.all(
        np.isfinite(np.asarray(g_p))
    )

    # production workflow (design note section 5): chains start NEAR high-likelihood
    # extrinsics located by the adaptive-IS layer -- the local kernel then explores the
    # conditional posterior. Emulate with a jittered start around the injection.
    x_true = jnp.asarray([INJ[n] for n in prior.names])
    y_true = prior.to_unconstrained(x_true)
    y0 = y_true[None, :] + 0.05 * jax.random.normal(
        jax.random.PRNGKey(5), (n_chains, len(prior.names))
    )
    lp0, g0 = logp(y0)

    # un-preconditioned MALA on a sharp 7D conditional: small step for healthy
    # acceptance (production would use the adapted/preconditioned kernels)
    kernel = MALA(step_size=0.015)

    def one_step(states, k):
        keys = jax.random.split(k, n_chains)
        states, info = jax.vmap(
            lambda kk, s: kernel.step(kk, s, problem.log_posterior)
        )(keys, states)
        return states, info.accepted

    states = KernelState(x=y0, log_prob=lp0, grad=g0)
    states, accepted = jax.lax.scan(one_step, states, jax.random.split(k_run, n_steps))

    acc = float(np.mean(np.asarray(accepted)))
    assert 0.1 < acc < 0.95, f"MALA acceptance {acc:.2f}"
    # chains stay in / converge into the high-likelihood region
    assert float(jnp.mean(states.log_prob)) > float(jnp.mean(lp_p)) + 10.0

    # conditional posterior concentrates near the injected extrinsics (zero noise);
    # loose windows -- this is a machinery prototype, not a coverage test
    x_phys = jax.vmap(prior.to_physical)(states.x)
    post = {n: np.asarray(x_phys[:, i]) for i, n in enumerate(prior.names)}
    assert abs(np.median(post["geocent_time"]) - T_C) < 5e-3
    assert abs(np.median(post["dec"]) - INJ["dec"]) < 0.4
    assert np.median(post["luminosity_distance"]) < 1500.0


def test_mode_cache_roundtrip(tmp_path):
    """Task 0.1: ModeCache save/load preserves ModesData exactly."""
    times = _analysis_times()
    modes = _synthetic_modes(times)
    md = ModesData(modes=modes, times=times, d_ref_mpc=D_REF, t_ref=T_C, f_ref=20.0)
    cache = ModeCache(tmp_path)
    theta = {"chirp_mass": 30.0, "mass_ratio": 0.8, "eccentricity": 0.1}
    assert cache.load(theta) is None
    cache.save(theta, md)
    back = cache.load(theta)
    assert back is not None and back.f_ref == 20.0 and back.d_ref_mpc == D_REF
    assert set(back.modes) == set(modes)
    for lm in modes:
        np.testing.assert_array_equal(back.modes[lm], modes[lm])
    # key stability: insertion order must not matter
    assert cache.key(
        {"mass_ratio": 0.8, "chirp_mass": 30.0, "eccentricity": 0.1}
    ) == cache.key(theta)


# ---------------------------------- balance-heuristic recycling (unit level)


def test_balance_heuristic_accumulator_analytic():
    """The recycled estimator must reproduce a known integral on the unit 4-cube,
    from batches drawn from DIFFERENT proposals, with every batch contributing.

    Target: an unnormalized Gaussian bump; its integral over the cube has a
    closed form via error functions. Batch 1 is uniform (the pilot's role);
    batch 2 comes from a defensive kernel-density mixture centered near the bump
    (the adaptive rounds' role).
    """
    from scipy.special import log_ndtr

    from jaxpe.gw.marginalized import (
        BalanceHeuristicAccumulator,
        _mixture_log_density,
        _mixture_sample,
    )

    rng = np.random.default_rng(7)
    center = np.array([0.4, 0.6, 0.3, 0.7])
    width = 0.05

    def log_target(u):
        return -0.5 * np.sum(((u - center) / width) ** 2, axis=1)

    # exact: prod_d integral of exp(-(x-c)^2 / 2w^2) over [0,1]
    def log_exact_1d(c):
        a, b = (0.0 - c) / width, (1.0 - c) / width
        return (
            np.log(width)
            + 0.5 * np.log(2 * np.pi)
            + log_ndtr(b)
            + np.log1p(-np.exp(log_ndtr(a) - log_ndtr(b)))
        )

    log_exact = sum(log_exact_1d(c) for c in center)

    accumulator = BalanceHeuristicAccumulator()

    # batch 1: uniform proposal (log-density identically zero on the cube)
    u1 = rng.uniform(size=(4096, 4))
    accumulator.add_batch(u1, log_target(u1), lambda pts: np.zeros(len(pts)))
    estimate_uniform_only = accumulator.log_normalization()
    size_uniform_only = accumulator.effective_sample_size()

    # batch 2: defensive mixture concentrated near (but deliberately offset from)
    # the bump, as the adaptive rounds would build
    centers = center[None, :] + 0.02
    widths = np.full(4, 0.08)
    component_weights = np.array([1.0])
    u2 = _mixture_sample(rng, 4096, centers, widths, component_weights, defense=0.2)
    accumulator.add_batch(
        u2,
        log_target(u2),
        lambda pts: _mixture_log_density(
            pts, centers, widths, component_weights, defense=0.2
        ),
    )

    estimate_recycled = accumulator.log_normalization()
    size_recycled = accumulator.effective_sample_size()

    # correctness: within Monte-Carlo error of the closed form
    assert abs(estimate_recycled - log_exact) < 5.0 / np.sqrt(size_recycled), (
        estimate_recycled,
        log_exact,
    )
    # recycling must IMPROVE the quality measure: the focused second batch adds
    # information, and the uniform batch is retained rather than discarded
    assert size_recycled > size_uniform_only
    # the uniform-only estimate is also consistent (sanity of the harness itself)
    assert abs(estimate_uniform_only - log_exact) < 5.0 / np.sqrt(size_uniform_only)


def test_balance_heuristic_matrix_bookkeeping():
    """Every proposal must be evaluated at every point, including points that
    arrived BEFORE the proposal existed (the recycling invariant)."""
    from jaxpe.gw.marginalized import BalanceHeuristicAccumulator

    calls = []

    def make_density(tag, value):
        def log_density(pts):
            calls.append((tag, len(pts)))
            return np.full(len(pts), value)

        return log_density

    accumulator = BalanceHeuristicAccumulator()
    accumulator.add_batch(np.zeros((3, 4)), np.zeros(3), make_density("a", 0.0))
    accumulator.add_batch(np.ones((2, 4)) * 0.5, np.zeros(2), make_density("b", 0.1))
    # proposal a: evaluated on its own batch (3), then extended to b's batch (2);
    # proposal b: evaluated on ALL 5 accumulated points at registration
    assert ("a", 3) in calls and ("a", 2) in calls and ("b", 5) in calls
    assert accumulator.n_points == 5 and accumulator.batch_sizes == [3, 2]
    log_weights = accumulator.log_balance_weights()
    assert log_weights.shape == (5,) and np.all(np.isfinite(log_weights))
