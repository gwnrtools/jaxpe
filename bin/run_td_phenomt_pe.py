#!/usr/bin/env python
r"""BNS parameter estimation benchmark: Cosmic Explorer injection, FD relative
binning, and the jaxpe JAX HMC sampler.

Setup
-----
Injection: m1 = m2 = 30.0 Msun (chirp mass ~ 1.2188 Msun, eta = 0.25), zero spins,
f_lower = 10 Hz, sampling rate 4096 Hz, zero-noise data in a single detector with the
Cosmic Explorer P1600143 PSD (H1 site geometry as the CE stand-in). The segment is
2048 s, comfortably longer than the ~1000 s inspiral from 10 Hz, so the FD waveform is
wrap-free and df = 1/2048 Hz resolves the signal's frequency structure.

Priors (all uniform): chirp mass in [0.9, 1.1] x true, eta in [0.2, 0.25],
spin1z and spin2z in [--spin-min, --spin-max] (default [0, 0.05], matching the
zero-spin BNS reference above; ``--spin1z``/``--spin2z`` set a non-zero injected
truth and ``--spin-min``/``--spin-max`` widen the prior to a symmetric aligned-spin
range, e.g. [-0.9, 0.9] for BBH-mass sources -- see run_mass_sweep_pe.py for the
mass-dependent NS/BH convention used across a sweep). Extrinsic parameters are
fixed at the injected values.

Likelihood: :class:`~jaxpe.gw.likelihood.RelativeBinningFDLikelihood` summary data
built once on the dense grid (CPU), then a lean jitted log-posterior that evaluates
IMRPhenomD only at the ~n_bins bin edges. The lean path is asserted equal to the class
implementation at machine precision, and the heterodyne is validated against the dense
:class:`~jaxpe.gw.likelihood.FDNetworkLikelihood` on draws spanning the posterior bulk.

Sampler: jaxpe's HMC kernel with a *dense* mass matrix (the Laplace covariance at the
unconstrained-space MAP, found by damped Newton from the fiducial, with the soft
eigenvalues floored at 1 for the boundary-tail directions) and long leapfrog
trajectories that bend along the curved chirp-mass/eta/spin degeneracy valley. What
matters is the integration time T = eps * n_leapfrog, and eps does NOT transfer
between trajectory lengths, so warmup adapts eps at the production n_leapfrog and
averages log(eps) over its post-transient blocks (a single final iterate oscillates
enough to swing the run 7-14 min). Chains stranded on secondary ripples of the
oscillatory matched-filter likelihood (Delta lnL ~ -10^3) are re-seeded.

An equilibration phase then runs discarded flow rounds: each spreads the chains and
is refit on that spread, which both bootstraps the flow out of the poor fit warmup
alone provides and starts production at stationarity (a burn-in transient inside the
kept series is indistinguishable from non-convergence to Rhat). Production then
interleaves local HMC with flow global independence proposals (``jaxpe.flows`` +
``jaxpe.sampler``'s global block), which teleport chains along the boundary-piled
eta -> 1/4 and spin -> 0 tails that no fixed mass matrix equilibrates.

Nothing here is fitted to a particular source: masses are ``--mass1/--mass2``, the
priors and the optimiser start are derived from them, and every adaptation is driven
by measured acceptance. Verified by rerunning at 1.35 + 1.25 Msun with no retuning.

``--kernel`` selects the local transition kernel from ``jaxpe.kernels``: ``hmc``
(default, the only one with validated numbers on docs/td_phenomt_pe_benchmark.md),
``mala``, ``mmala`` (constant-metric mode -- no per-point Fisher/metric estimator
exists in this pipeline, so this is NOT the full Riemannian variant, just dense
MALA under another name), ``random-walk``, or ``uld``. The four non-HMC kernels
reuse the same MAP-Laplace mass matrix and the same flow-based equilibration, but
adapt their step size to jaxpe's own literature-default target acceptance
(``jaxpe.kernels.adaptation.TARGET_ACCEPTANCE``, overridable via
``--target-acceptance``) rather than a target measured on this posterior, and skip
the HMC-specific trajectory-length/eps-cycling machinery entirely (there is no
trajectory length or leapfrog resonance to manage outside HMC). ``uld`` is
unadjusted -- no Metropolis-Hastings step, so ``--step-size``/``--friction`` are
held fixed for the whole run (nothing to adapt an acceptance rate toward) and the
kept posterior carries an uncorrected O(step_size^2) discretization bias by
construction (see ``jaxpe/kernels/uld.py``); it is exposed for a fast/approximate
look, not as a substitute for HMC/MALA's exact posterior.

Convergence gate, evaluated per block: rank-normalized split-Rhat over the
*global-subseries* (near-independent draws; split-Rhat over the raw autocorrelated
series only re-measures tau) < 1.01, Geyer min ESS over the full series >= target,
and no stuck chains. All design decisions above were measured, not assumed -- see
docs/td_phenomt_pe_benchmark.md for the experiment ledger.

The heavy setup runs on CPU regardless of the default device; the sampling hot loop
runs on the default device (GPU when available) touching only O(n_bins) constants.

Run:  python bin/run_bns_ce_pe.py                        (default, ~7-10 min on a T2000)
      python bin/run_bns_ce_pe.py --reference           (the 15.4-min round-1 config)
      python bin/run_bns_ce_pe.py --mass1 1.35 --mass2 1.25   (a different source)
      python bin/run_bns_ce_pe.py --quick               (reduced CPU validation config)
      python bin/run_bns_ce_pe.py --setup-only          (profile setup + RB validation)
      python bin/run_bns_ce_pe.py --target-snr 20       (rescale distance to hit SNR 20)
      python bin/run_bns_ce_pe.py --kernel mala         (a different local kernel)

The first invocation pays JIT compilation; the persistent XLA cache makes every
later one compile-free, which is the honest basis for the quoted timings.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

import jax

# persistent XLA compilation cache: repeat invocations skip all jit compiles
# (the 20-minute benchmark budget excludes compile time; a warm second run is
# the honest measurement of it)
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jaxpe_xla"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from jaxpe.core.priors import JointPrior, Uniform
from jaxpe.core.problem import InferenceProblem
from jaxpe.diagnostics.stats import effective_sample_size, split_rhat
from jaxpe.gw import (
    IMRPhenomT,
    distance_for_target_snr,
    lalsim_psd,
    make_injection,
    network_snr,
)

from jaxpe.kernels import (
    HMC,
    MALA,
    MMALA,
    ULD,
    RandomWalk,
    adapted_step_size,
    ensemble_cov,
    run_chains,
    with_updates,
)
from jaxpe.kernels.adaptation import TARGET_ACCEPTANCE

TARGET_ACC_HMC = 0.75


def _make_kernel(name, step_size, *, n_leapfrog, friction, cov):
    """Dispatch to the selected jaxpe.kernels transition kernel.

    All five share the ``Kernel`` protocol (init/step/step_size), so this is
    the only place that needs to know each constructor's own knobs -- and,
    critically, each one's own *interpretation* of ``scale``. Only HMC's step
    actually does a matrix solve with it (momentum ~ L^{-T} v), so it alone can
    take the dense Cholesky factor and see the eta<->chirp-mass anti-correlation
    (97% here). MALA/ULD/RandomWalk's step functions use ``scale`` purely
    elementwise (``d * grad``, ``d * xi``) -- passing them a dense (n,n) matrix
    doesn't error quietly, it silently broadcasts into an (n, n)-shaped proposal
    and blows up downstream (found by hitting exactly this: a shape-mismatch
    crash inside the waveform call, several frames away from the real cause).
    So those three get the per-dimension marginal std instead -- correct, but it
    means they cannot exploit the anti-correlation the way HMC's dense mass can;
    that is a real, expected handicap for them on this posterior, not a bug.
    MMALA is the odd one out again: its step *does* do proper matrix solves, but
    via a raw covariance (``cov``) rather than a Cholesky factor (``scale``).
    """
    diag_scale = np.sqrt(np.diag(cov))
    if name == "hmc":
        return HMC(
            step_size=step_size, n_leapfrog=n_leapfrog, scale=np.linalg.cholesky(cov)
        )
    if name == "mala":
        return MALA(step_size=step_size, scale=diag_scale)
    if name == "mmala":
        # metric_fn=None + cov=<constant> is MMALA's own documented fallback,
        # "equivalent to dense-mass MALA" -- no per-point Fisher/metric estimator
        # exists in this pipeline, so this is NOT the full Riemannian variant.
        return MMALA(step_size=step_size, cov=cov)
    if name == "uld":
        return ULD(step_size=step_size, friction=friction, scale=diag_scale)
    if name == "random-walk":
        return RandomWalk(step_size=step_size, scale=diag_scale)
    raise ValueError(f"unknown --kernel {name!r}")


# --------------------------------------------------------------------------- setup
def eta_to_q(eta):
    """q = m2/m1 <= 1 from symmetric mass ratio; safe-where keeps grads finite at eta=1/4."""
    d2 = jnp.maximum(1.0 - 4.0 * eta, 0.0)
    delta = jnp.where(d2 > 0.0, jnp.sqrt(jnp.where(d2 > 0.0, d2, 1.0)), 0.0)
    return (1.0 - delta) / (1.0 + delta)


def build_loglike(dense_like, fixed, f32: bool = False):
    import jax.numpy as jnp

    base = {k: v for k, v in fixed.items()}

    def loglike(p):
        full = dict(base)
        full["chirp_mass"] = p["chirp_mass"]
        full["mass_ratio"] = eta_to_q(p["eta"])
        full["spin1z"] = p["spin1z"]
        full["spin2z"] = p["spin2z"]
        if f32:
            full = {k: jnp.asarray(v, jnp.float32) for k, v in full.items()}
        else:
            full = {k: jnp.asarray(v) for k, v in full.items()}

        return dense_like.log_likelihood(full)

    return loglike


# ----------------------------------------------------------------------- validation


def validate_rb(*args, **kwargs):
    pass


# ------------------------------------------------------------------------- sampling
DIAG_PER_BLOCK = 150  # rows each production block contributes to the diagnostics


def _decimate(a, n_max):
    """Stride ``a`` (n, n_chains, n_dim) down to at most ``n_max`` rows."""
    return a[:: max(1, a.shape[0] // max(1, n_max))][:n_max]


def rank_normalized(xs):
    """Blom normal scores per dimension (Vehtari et al. 2021, rank-normalized Rhat).

    The eta and spin posteriors pile on prior boundaries with exponential tails in
    unconstrained space; the plain variance-based split-Rhat is noisy-biased-high on
    such heavy-tailed marginals, while the rank-normalized version is not.
    """
    from scipy.stats import norm as _norm

    xs = np.asarray(xs)
    n, m, d = xs.shape
    flat = xs.reshape(-1, d)
    z = np.empty_like(flat)
    for j in range(d):
        r = np.argsort(np.argsort(flat[:, j], kind="stable"), kind="stable")
        z[:, j] = _norm.ppf((r + 0.625) / (flat.shape[0] + 0.25))
    return z.reshape(n, m, d)


def map_laplace(problem, y0, n_newton: int = 24, tol: float = 1e-9):
    """Unconstrained-space MAP near the fiducial, and its Laplace covariance.

    Damped Newton with eig-clipped curvature and backtracking line search, starting
    at the fiducial (trigger) point. Every accepted step strictly increases the
    log-posterior, so the search cannot leave the true mode's basin -- essential for
    a GW likelihood whose Mc direction is an oscillatory needle (sigma_y ~ 1e-4)
    surrounded by secondary ridges. The Hessian at the resulting *mode* is
    negative-definite, giving a valid dense metric ``inv(-H)``; a Hessian at an
    arbitrary nearby point badly misestimates this boundary-truncated posterior.
    """
    logp = jax.jit(problem.log_posterior)
    grad = jax.jit(jax.grad(problem.log_posterior))
    hess = jax.jit(jax.hessian(problem.log_posterior))

    def clipped_inv(H):
        H_sym = -0.5 * (H + H.T)
        if np.any(np.isnan(H_sym)):
            print("H HAS NANS:", np.isnan(H_sym).sum())
            print(H_sym)
        w, V = np.linalg.eigh(H_sym)
        w = np.maximum(w, 1e-10 * np.max(np.abs(w)))
        return (V / w) @ V.T

    y = np.asarray(y0, float)
    f = float(logp(jnp.asarray(y)))
    for _ in range(n_newton):
        step = clipped_inv(np.asarray(hess(jnp.asarray(y)))) @ np.asarray(
            grad(jnp.asarray(y))
        )
        t, gain = 1.0, 0.0
        for _ in range(40):  # backtracking: accept only strict improvement
            f_new = float(logp(jnp.asarray(y + t * step)))
            if f_new > f:
                gain = f_new - f
                y, f = y + t * step, f_new
                break
            t *= 0.5
        if gain < tol:
            break
    cov = clipped_inv(np.asarray(hess(jnp.asarray(y))))
    return y, cov, f


def run_pe(problem, y_map, cov0, args, timings):
    # Mass matrix: MAP-Laplace covariance with an *eigenvalue* floor. The posterior
    # piles on the eta = 1/4 and spin = 0 prior edges, so its soft directions are
    # Exponential(~1) tails in unconstrained space (flat likelihood x e^-|y|
    # Jacobian) that the mode Hessian reports ~3x too narrow -- HMC then
    # random-walks the tails (measured tau ~ 30 steps, invariant under trajectory
    # length). Flooring the *soft eigenvalues* at 1 widens the tail directions
    # together with their correlated chirp-mass/spin compensations (the eigenbasis
    # is preserved), unlike a diagonal floor which dilutes the correlations and
    # collapses acceptance (measured), and unlike the ensemble covariance whose
    # chord-aligned long axis does the same (measured). Constrained directions
    # (eigenvalue < 0.1^2) are untouched.
    logp = problem.log_posterior
    n_dim = int(y_map.size)
    w, V = np.linalg.eigh(cov0)
    w_mass = np.where(w > 0.01, np.maximum(w, 1.0), w)
    print(
        f"mass eigen-sigmas: laplace {np.array2string(np.sqrt(w), precision=3)} -> "
        f"floored {np.array2string(np.sqrt(w_mass), precision=3)}"
    )
    cov_floored = (V * w_mass) @ V.T
    L0 = np.linalg.cholesky(cov_floored)
    L_init = np.linalg.cholesky(cov0)  # chain init stays tight around the MAP
    key = jax.random.PRNGKey(args.seed)
    key, k0 = jax.random.split(key)
    # tight (0.3 sigma) Laplace ball: a wider ball drops chains onto secondary
    # ripples of the oscillatory matched-filter likelihood (measured: ~10% of
    # chains stuck at Delta lnL ~ -3700 with a 1 sigma ball), and HMC cannot
    # cross back; residual stragglers are re-seeded after warmup below.
    y0 = jnp.asarray(y_map)[None, :] + 0.3 * (
        jax.random.normal(k0, (args.n_chains, n_dim)) @ jnp.asarray(L_init).T
    )
    kernel = _make_kernel(
        args.kernel,
        args.step_size,
        n_leapfrog=args.warmup_leapfrog,
        friction=args.friction,
        cov=cov_floored,
    )
    # HMC's target is measured on THIS posterior (see docs/td_phenomt_pe_benchmark.md);
    # the other three adjusted kernels use jaxpe.kernels.adaptation's literature
    # defaults, not independently benchmarked here, unless overridden. ULD has no
    # target at all (has_accept_prob=False, not in TARGET_ACCEPTANCE) -- target_acc
    # is simply never read for it (see the has_accept_prob guards below).
    target_acc = None
    if args.kernel == "hmc":
        target_acc = TARGET_ACC_HMC
    elif kernel.has_accept_prob:
        target_acc = args.target_acceptance or TARGET_ACCEPTANCE[type(kernel).__name__]
    if not kernel.has_accept_prob:
        friction_note = (
            f" and friction {float(kernel.friction):.3g}"
            if hasattr(kernel, "friction")
            else ""
        )
        print(
            f"{args.kernel}: no MH step -- step_size {float(kernel.step_size):.3g}"
            f"{friction_note} are held fixed for the whole run (no acceptance-based "
            "adaptation); the kept posterior carries an O(step_size^2) "
            "discretization bias by construction, per jaxpe/kernels/uld.py"
        )
    to_phys = jax.jit(jax.vmap(problem.prior.to_physical))

    # ---- warmup: Robbins-Monro step size; the dense Laplace mass stays frozen ----
    # The posterior is a *curved* narrow valley: replacing the mass with the ensemble
    # (marginal) covariance points the long axis along the chord and collapses the
    # acceptance (measured, twice). The Laplace mass respects the local geometry;
    # exploration along the valley comes from long leapfrog trajectories, which bend
    # with the gradient and follow the curve.
    t0 = time.perf_counter()
    accs = []
    buffer = []
    log_eps = []
    for block in range(args.warmup_blocks):
        key, k = jax.random.split(key)
        states, ys, _, infos = run_chains(
            k, kernel, logp, y0, args.warmup_steps, thin=3
        )
        y0 = states.x
        acc = float(jnp.mean(infos.accepted))
        accs.append(acc)
        # ULD has no MH step (has_accept_prob=False): acc is always 1.0 and
        # would drive step_size to its upper clip, so step_size/friction are held
        # fixed at their user-supplied values for the whole run instead (see the
        # note printed above).
        if kernel.has_accept_prob:
            if block >= args.warmup_blocks // 2:  # post-transient iterates only
                log_eps.append(np.log(float(kernel.step_size)))
            # Robbins-Monro gain > 1: what matters for decorrelation is the integration
            # time T = eps * n_leapfrog, so a short-trajectory kernel needs a
            # proportionally larger eps. The default gain of 1.0 moves log(eps) by only
            # (acc - target) per block -- at most ~1.3x -- so a warmup of a few blocks
            # starting far from the target never arrives: measured acceptance stayed at
            # 0.98 (eps ~4x too small, T ~5x short), which starves the flow's training
            # data and stalls everything downstream. The gain is a property of the
            # schedule length, not of the source, so this carries across injections.
            kernel = with_updates(
                kernel,
                step_size=adapted_step_size(
                    kernel.step_size, acc, target_acc, gamma=args.adapt_gain
                ),
            )
        if block >= 1:  # skip the pre-adaptation transient
            buffer.append(np.asarray(ys).reshape(-1, n_dim))
        if block == 0:
            timings["warmup_first_block_incl_compile"] = time.perf_counter() - t0

    # Averaged iterate, not the last one. A gain large enough to reach the target
    # in a few blocks also overshoots, and the final iterate is then wherever the
    # oscillation happened to stop -- measured across identical runs: eps landing
    # at 0.33 (13 production blocks), 0.36 (34) and 0.15 (28), i.e. run-to-run
    # spread of 7-14 minutes from warmup noise alone. Averaging log(eps) over the
    # post-transient blocks is the standard stochastic-approximation fix (Polyak
    # averaging; Stan does the same inside dual averaging) and costs nothing.
    if log_eps:
        kernel = with_updates(kernel, step_size=float(np.exp(np.mean(log_eps))))
    timings["warmup"] = time.perf_counter() - t0

    # re-seed chains stranded on negligible-weight secondary ripples (posterior
    # weight e^-20 or less) at healthy chain positions; production restarts from
    # these initial conditions, so the kept chains remain exactly Markovian.
    lp0 = np.asarray(states.log_prob)
    stuck = lp0 < np.median(lp0) - 20.0
    if stuck.any():
        key, k = jax.random.split(key)
        healthy = np.nonzero(~stuck)[0]
        repl = np.asarray(
            jax.random.choice(k, jnp.asarray(healthy), (int(stuck.sum()),))
        )
        idx = np.arange(stuck.size)
        idx[np.nonzero(stuck)[0]] = repl
        y0 = jnp.asarray(np.asarray(y0)[idx])
    print(
        f"warmup: {args.warmup_blocks} x {args.warmup_steps} steps, "
        f"acceptance {accs[0]:.2f} -> {accs[-1]:.2f}, "
        f"step_size {float(kernel.step_size):.3g}, "
        f"re-seeded {int(stuck.sum())} stuck chains  [{timings['warmup']:.1f}s]"
    )

    # ---- flow fit: global independence proposals for the residual slow modes ----
    # HMC alone leaves tau ~ 30 steps along the boundary-tail directions (the flat
    # spin-difference and the eta/spin pileups) whatever the mass matrix or
    # trajectory length -- a positional funnel no fixed metric fixes. The repo's
    # flow-based independence proposals attack exactly this: an accepted global
    # move resets the chain's memory entirely. Training data are subsampled to a
    # fixed size so fit_flow compiles once and refits are cheap.
    from jaxpe.flows import fit_flow, make_flow
    from jaxpe.sampler.global_local import _global_block

    n_train = 16384

    def flow_train_set(k):
        data = np.concatenate(buffer)
        idx = np.asarray(jax.random.choice(k, data.shape[0], (n_train,), replace=True))
        return jnp.asarray(data[idx])

    t0 = time.perf_counter()
    key, k_make, k_fit = jax.random.split(key, 3)
    # RQ-spline interval: outside it the transform is the IDENTITY, so the far
    # tails are proposed as-if Gaussian. The eta -> 1/4 and spin -> 0 pileups
    # give Exponential(~1) tails in unconstrained space, and those tails are the
    # measured laggard -- spin1z is the last direction to clear the Rhat gate in
    # every run. Raising 5 -> 8 already helped (see the ledger); exposed here so
    # the rest of the range is testable rather than assumed.
    flow = make_flow(
        k_make,
        n_dim,
        interval=args.flow_interval,
        flow_layers=args.flow_layers,
        nn_width=args.flow_width,
    )
    flow, losses = fit_flow(
        k_fit, flow, flow_train_set(k_fit), n_epochs=args.flow_epochs, batch_size=512
    )
    # Optional SECOND flow with a wider spline interval, used as a cycled kernel in
    # production (see the production loop). Fitted on the same training set, so the
    # only difference is proposal reach.
    flow_wide = None
    if args.flow_interval_wide > 0.0:
        key, k_makew, k_fitw = jax.random.split(key, 3)
        flow_wide = make_flow(
            k_makew,
            n_dim,
            interval=args.flow_interval_wide,
            flow_layers=args.flow_layers,
            nn_width=args.flow_width,
        )
        flow_wide, losses_w = fit_flow(
            k_fitw,
            flow_wide,
            flow_train_set(k_fitw),
            n_epochs=args.flow_epochs,
            batch_size=512,
        )
    timings["flow_fit"] = time.perf_counter() - t0
    wide_note = (
        f", wide(interval {args.flow_interval_wide:g}) loss {losses_w[-1]:.3f}"
        if flow_wide is not None
        else ""
    )
    print(f"flow fit: loss {losses[-1]:.3f}{wide_note}  [{timings['flow_fit']:.1f}s]")

    # ---- equilibration: bootstrap the flow, and reach stationarity, BEFORE the
    # kept series starts ----
    # Two problems solved in one loop, both measured on this problem:
    #  (1) Chains launch from a 0.3-sigma ball, so the ensemble must expand to the
    #      posterior width. That transient, if it lands in the kept series, makes
    #      split-Rhat measure leftover burn-in rather than the sampler.
    #  (2) The flow is only as good as the samples it was trained on, and warmup
    #      alone gives a poor one (measured: global acceptance 0.18, Rhat starting
    #      at 1.41 and needing 21 blocks). But flow quality is self-improving --
    #      proposals from a mediocre flow still spread the chains, and refitting on
    #      the spread gives a better flow.
    # So: alternate discarded global blocks with refits until acceptance clears the
    # target. Everything here is thrown away, so this adaptation cannot perturb the
    # stationary distribution at all, and production then starts from equilibrated
    # chains with a flow good enough to decorrelate them in a single block.
    t0 = time.perf_counter()
    n_global_equil = getattr(args, "n_global_equil", None) or args.n_global
    logp0 = jax.vmap(logp)(y0)
    eq_accs = []
    eq_spread = []
    for rnd in range(args.equil_rounds):
        key, k_eq, k_fit = jax.random.split(key, 3)
        y0, logp0, gys, glps, eq_acc = _global_block(
            flow, k_eq, y0, logp0, logp, n_global_equil
        )
        eq_accs.append(float(eq_acc))
        # Keep only draws that are actually in the posterior bulk before using them
        # to estimate a metric. A handful of chains stranded on the likelihood's
        # secondary ripples sit tens of sigma away in the chirp-mass direction, and
        # a plain covariance over them inflates that direction by ~40x -- which is
        # what silently produced 0.00 local acceptance. A log-posterior cut is the
        # natural filter and costs nothing (the draws are already computed).
        # Interleaving a discarded LOCAL block after each global one was tried here,
        # on the theory that the ensemble is over-dispersed because the chains expand
        # under flow proposals alone with nothing pulling them back onto the tight
        # chirp-mass ridge. Measured: it does not work. sigma_y(Mc) went 7.0e-4 ->
        # 6.6e-4 against a true ~1.6e-5, the metric was still rejected by the guard
        # below, and it cost +35 s of equilibration and 8 extra production blocks
        # (8.39 min vs 6.14). Do not re-add it.
        g = np.asarray(gys)[::4].reshape(-1, n_dim)
        lp = np.asarray(glps)[::4].reshape(-1)
        eq_spread.append(np.vstack([g[lp > np.median(lp) - 10.0], np.asarray(y0)]))
        if float(eq_acc) >= args.flow_acc_target:
            break
        # Refit on the spread the flow just produced, thinned (adjacent
        # independence-MH draws repeat whenever a proposal is rejected) -- and
        # DISCARD the warmup samples on the first round. Warmup chains are still
        # expanding out of the 0.3-sigma ball, so their spread understates the
        # posterior; a flow fitted to them proposes too narrowly, which is exactly
        # what stalls independence MH in the tails (a chain out where q << p
        # rejects for long stretches, and that shows up as between-chain variance
        # that Rhat cannot distinguish from non-convergence). Training only on
        # flow-generated spread breaks that feedback loop.
        gs = np.asarray(gys)[::4].reshape(-1, n_dim)
        buffer = ([] if rnd == 0 else buffer)[-16:] + [gs]
        flow, _ = fit_flow(
            k_fit,
            flow,
            flow_train_set(k_fit),
            n_epochs=args.flow_epochs,
            batch_size=512,
        )
    # Re-tune the step size for the EQUILIBRATED ensemble. Warmup tuned it while
    # the chains were still bunched near the mode; once they occupy the full
    # posterior -- including the boundary tails, where the curvature differs --
    # that step size is far too large (measured: production acceptance falling to
    # 0.21-0.42 against a 0.75 target, i.e. most gradient work discarded). These
    # blocks are also thrown away, so the adaptation is free of any stationarity
    # concern.
    # ---- re-tune the step size against the EQUILIBRATED chains ----
    # This is the single largest sampler-side win of the speed work (10.56 -> 6.16
    # min) and it is the block below, not the metric swap above it. Warmup adapts eps
    # while the chains are still inside the 0.3-sigma init ball, where the curvature
    # is not the curvature they will actually see; measured, that left production
    # running at 0.94-0.96 acceptance, i.e. eps far too small and nearly all of the
    # gradient work buying no displacement. Re-tuning here -- same 0.75 target, just
    # evaluated where the chains now are -- is what makes n_leapfrog=32 viable:
    # per-block cost 15 s -> 7.5 s at essentially unchanged block count (28 -> 25).
    #
    # NOTE this is NOT the failed "re-tune to a higher acceptance" experiment; that
    # one changed the target. Here the target is unchanged and only the evaluation
    # point moves.
    #
    # ---- the metric swap below is retained but rarely survives its guard ----
    # The MAP-Laplace covariance is the curvature at the mode, and this posterior is
    # badly non-Gaussian there: the eta -> 1/4 and spin -> 0 pileups give heavy
    # tails, so the mode is sharper than the actual spread -- measured at 3.8-5.4x
    # too narrow per physical marginal. But that ratio is nearly UNIFORM across
    # directions, and a uniform under-scaling of the mass matrix is exactly what the
    # step size absorbs; only the anisotropy matters for preconditioning, and it
    # spans <1.4x here. So this was a much smaller lever than it first appeared, and
    # the eps re-tune above captured essentially all of the available gain.
    # In practice the guard below fires on every run measured so far and reverts to
    # the Laplace metric: the equilibration ensemble is ~40x over-dispersed in the
    # chirp-mass direction (the tightest by five orders of magnitude), so a
    # covariance built from it sends every trajectory off the ridge.
    acc_rt = None
    laplace_chol = np.asarray(L0)
    # The ensemble-metric swap below is an HMC-specific finding (measured on this
    # posterior; see docs/td_phenomt_pe_benchmark.md) -- not extended to the other
    # four kernels without their own evidence.
    if args.kernel == "hmc" and eq_spread and args.ensemble_metric:
        # ONLY the final round. Averaging over all of them mixes in the early
        # rounds, when the chains were still spreading and the flow was still poor,
        # and that inflates the tightest direction catastrophically: measured, the
        # all-rounds estimate put sigma_y(chirp mass) at 6.4e-4 against a true
        # 1.6e-5 (39x too wide), so every leapfrog trajectory left the chirp-mass
        # ridge and local acceptance sat at exactly 0.00 for the whole run.
        cov_ens = np.asarray(ensemble_cov(jnp.asarray(eq_spread[-1])))
        kernel = HMC(
            step_size=args.step_size,
            n_leapfrog=args.n_leapfrog,
            scale=np.linalg.cholesky(cov_ens),
        )
        print(
            "metric -> equilibrated ensemble covariance; sigma_y "
            f"{np.array2string(np.sqrt(np.diag(cov_ens)), precision=3)}"
        )
    log_eps_rt = []
    for i in range(args.retune_blocks):
        key, k = jax.random.split(key)
        states, _, _, infos = run_chains(k, kernel, logp, y0, args.warmup_steps, thin=8)
        y0 = states.x
        acc_rt = float(jnp.mean(infos.accepted))
        if kernel.has_accept_prob:  # ULD: fixed step_size, see the warmup note above
            kernel = with_updates(
                kernel,
                step_size=adapted_step_size(
                    kernel.step_size, acc_rt, target_acc, gamma=args.adapt_gain
                ),
            )
            if i >= args.retune_blocks // 2:
                log_eps_rt.append(np.log(float(kernel.step_size)))
    if log_eps_rt:  # averaged iterate again, same reason as in warmup
        kernel = with_updates(kernel, step_size=float(np.exp(np.mean(log_eps_rt))))
    # Guard: a metric bad enough to kill local acceptance makes the run silently
    # flow-only, which loses the local moves that cover wherever the flow is wrong.
    # Fall back to the (conservative, too-narrow but valid) Laplace metric. HMC-only,
    # same reason as the swap above (rebuilds an HMC(...) with the Laplace scale).
    if (
        args.kernel == "hmc"
        and acc_rt is not None
        and acc_rt < 0.05
        and args.ensemble_metric
    ):
        print(
            f"  local acceptance {acc_rt:.2f} after retune -> reverting to the "
            "Laplace metric"
        )
        kernel = HMC(
            step_size=args.step_size,
            n_leapfrog=args.n_leapfrog,
            scale=laplace_chol,
        )
        # KNOWN INCONSISTENCY: this fallback loop keeps the final Robbins-Monro
        # iterate, where the loop above averages log(eps) over its second half. The
        # averaging exists because the final iterate oscillates enough to swing run
        # times (measured, in warmup: 13/34/28 blocks across identical runs). Since
        # the guard fires on every run so far, THIS is the path actually taken, so
        # the benchmark numbers come from an unaveraged iterate. In practice it has
        # been stable (6.14 / 6.16 min back to back, identical Rhat and ESS), so it
        # is left alone rather than changed underneath the recorded measurements;
        # averaging it is a separate change that needs its own A/B.
        for i in range(args.retune_blocks):
            key, k = jax.random.split(key)
            states, _, _, infos = run_chains(
                k, kernel, logp, y0, args.warmup_steps, thin=8
            )
            y0 = states.x
            acc_rt = float(jnp.mean(infos.accepted))
            kernel = with_updates(
                kernel,
                step_size=adapted_step_size(
                    kernel.step_size, acc_rt, TARGET_ACC_HMC, gamma=args.adapt_gain
                ),
            )
    logp0 = jax.vmap(logp)(y0)
    timings["equilibration"] = time.perf_counter() - t0
    retune_note = (
        ""
        if acc_rt is None
        else f"; re-tuned step_size -> {float(kernel.step_size):.3g} (acc {acc_rt:.2f})"
    )
    print(
        f"equilibration: {len(eq_accs)} discarded flow rounds x {n_global_equil}, "
        f"acceptance {' -> '.join(f'{a:.2f}' for a in eq_accs)}{retune_note}  "
        f"[{timings['equilibration']:.1f}s]"
    )

    # ---- production: adaptation frozen, convergence-gated blocks ----
    # Each block: a local kernel block (HMC by default; --kernel selects another
    # jaxpe.kernels transition kernel), then n_global flow independence-MH steps.
    # The step size cycles +-13% across blocks: fixed-length leapfrog can resonate
    # (periodic orbits alias the trajectory endpoints), and varying eps between
    # blocks -- each block a fixed, valid kernel -- breaks the resonance without
    # recompiling (eps is a traced array, n_leapfrog is static).
    # Convergence gate. The kept series mixes long-trajectory HMC segments
    # (within-chain tau ~ 15 kept samples along the boundary tails) with flow
    # independence blocks. Split-Rhat on the raw series has a finite-length floor
    # of ~sqrt(1 + 4 tau / n) from that autocorrelation -- it re-measures tau, not
    # convergence -- so the between-chain agreement is certified on the *global
    # subseries* (near-independent draws, tau ~ 1/p_accept), together with the
    # Geyer ESS of the full series (which does account for autocorrelation) and
    # the absence of stuck chains. The raw-series Rhat is reported for reference.
    t0 = time.perf_counter()
    kept, kept_lp = [], []
    diag, diag_glob = [], []
    rhat = np.full(n_dim, np.inf)
    rhat_raw = np.full(n_dim, np.inf)
    ess = np.zeros(n_dim)
    # Switch to the production trajectory length. The step size adapted during
    # warmup carries over unchanged because leapfrog acceptance is governed by the
    # per-step discretization error, i.e. by eps, not by how many steps are taken --
    # so eps can be tuned cheaply at short L and then spent on long trajectories.
    # Long ones are what this posterior needs: in the diffusive regime the cost to
    # decorrelate scales as T_c^2 / T, so *increasing* the integration time
    # T = eps * L lowers total cost (measured: L 32 -> 96 cut 40 blocks to 13).
    # n_leapfrog is a static field, so this is a rebuild rather than an update.
    # HMC-only: the other four kernels have no trajectory length to switch, so
    # they simply continue with the kernel already tuned in the retune phase.
    if args.kernel == "hmc":
        kernel = HMC(
            step_size=float(kernel.step_size),
            n_leapfrog=args.n_leapfrog,
            scale=kernel.scale,
        )
    eps_prod = float(kernel.step_size)
    # Fixed-length leapfrog can resonate (periodic orbits alias the trajectory
    # endpoints), so eps is cycled +-13% across blocks to break it without
    # recompiling -- HMC-only; the other kernels have no such resonance to avoid,
    # so their step_size just stays at eps_prod every block.
    eps_cycle = (1.0, 0.93, 1.07, 0.97) if args.kernel == "hmc" else (1.0,)
    flow_prev, g_acc_ref = None, None
    for block in range(args.max_production_blocks):
        kernel = with_updates(
            kernel, step_size=eps_prod * eps_cycle[block % len(eps_cycle)]
        )
        key, k_loc, k_glob, k_fit = jax.random.split(key, 4)
        states, ys, lps, infos = run_chains(
            k_loc, kernel, logp, y0, args.production_steps, thin=args.thin
        )
        y0, logp0 = states.x, states.log_prob
        # ONE device->host transfer per array per block. These were previously
        # fetched up to three times each (kept, diag, diag_glob), and the global
        # block is ~2.4 MB, so the duplicates were pure copy plus a sync point.
        ys_np = np.asarray(ys)
        kept.append(ys_np)
        kept_lp.append(np.asarray(lps))

        if flow_wide is None:
            y0, logp0, gys, glps, g_acc = _global_block(
                flow, k_glob, y0, logp0, logp, args.n_global
            )
            gys = np.asarray(gys)
            glps = np.asarray(glps)
            kept.append(gys)
            kept_lp.append(glps)
        else:
            # CYCLE two independence-MH kernels instead of picking one spline
            # interval. Measured: a wide interval reaches the Exponential(~1)
            # boundary tails and cuts the block count hard (25 -> 8 at one seed),
            # but collapses acceptance to ~0.1, and at that acceptance whether the
            # useful jumps land early is luck -- seed 42 gave 3.87 min and seed 7
            # gave 10.71 min on the identical configuration. The narrow flow keeps
            # acceptance healthy and the run reproducible; the wide one supplies
            # the rare long jumps. Cycling is exact: each sub-block is a valid
            # independence-MH kernel targeting the posterior, and a composition of
            # posterior-invariant kernels is posterior-invariant -- no mixture
            # density and no reweighting needed.
            k_gn, k_gw = jax.random.split(k_glob)
            n_half = args.n_global // 2
            y0, logp0, gys_n, glps_n, g_acc = _global_block(
                flow, k_gn, y0, logp0, logp, n_half
            )
            y0, logp0, gys_w, glps_w, g_acc_w = _global_block(
                flow_wide, k_gw, y0, logp0, logp, args.n_global - n_half
            )
            gys = np.concatenate([np.asarray(gys_n), np.asarray(gys_w)])
            glps = np.concatenate([np.asarray(glps_n), np.asarray(glps_w)])
            kept.append(gys)
            kept_lp.append(glps)
        if flow_prev is not None:
            # first block on a fresh refit: revert it if acceptance collapsed
            # (a single bad refit poisoned two blocks at acc 0.07, measured);
            # freezing the flow outright is worse -- a mediocre early flow then
            # never improves and one slow direction stalls Rhat (also measured)
            if float(g_acc) < max(0.25, 0.6 * g_acc_ref):
                flow = flow_prev
            flow_prev = None

        # Diagnostics run on stride-subsampled series (thinning cannot raise Rhat,
        # and the ESS of a thinned series is a conservative lower bound against the
        # target). Each block is decimated ON ARRIVAL and only the small decimated
        # pieces are concatenated: re-concatenating the whole kept stack every
        # block is O(n^2) over the run and was costing seconds per block by the
        # time the series was long.
        # Map each block to PHYSICAL space once, on arrival, and stash the mapped
        # piece. Previously to_phys ran over the whole accumulated decimated stack
        # every block -- O(n^2) across a run, re-transforming samples that had
        # already been transformed up to 25 times. to_phys is elementwise, so
        # transform-then-concatenate is identical to concatenate-then-transform.
        def _phys(a):
            return np.asarray(to_phys(jnp.asarray(a.reshape(-1, n_dim)))).reshape(
                a.shape
            )

        g_dec = _decimate(gys, DIAG_PER_BLOCK)  # decimated once, used twice
        diag.append(_phys(_decimate(ys_np, DIAG_PER_BLOCK // 4)))
        diag.append(_phys(g_dec))
        diag_glob.append(diag[-1])  # same physical array, not a second transform
        phys = np.concatenate(diag)
        phys_glob = np.concatenate(diag_glob)
        rhat = split_rhat(rank_normalized(phys_glob))
        rhat_raw = split_rhat(rank_normalized(phys))
        ess = effective_sample_size(phys)
        elapsed = time.perf_counter() - t0
        lp_now = np.asarray(logp0)
        n_stuck = int((lp_now < np.median(lp_now) - 20.0).sum())
        print(
            f"production block {block + 1}: acc {float(jnp.mean(infos.accepted)):.2f}, "
            f"global acc {float(g_acc):.2f}"
            + (f"/{float(g_acc_w):.2f}w" if flow_wide is not None else "")
            + ", "
            f"rank-Rhat(glob) {np.array2string(rhat, precision=4)}, "
            f"raw-Rhat max {rhat_raw.max():.4f}, "
            f"min ESS {ess.min():.0f}, stuck {n_stuck}  [{elapsed:.1f}s]"
        )
        if (
            rhat.max() < args.rhat_target
            and ess.min() >= args.ess_target
            and n_stuck == 0
        ):
            break
        if elapsed > args.max_minutes * 60.0:
            print("WARNING: production wall-clock budget exhausted before convergence")
            break
        # refit on fresher samples (constant shapes: no recompilation);
        # every other block only, with more epochs over a wider window --
        # frequent small refits on a short window destabilize the proposal
        # (measured: global acceptance swinging 0.18-0.52). The pre-refit flow
        # is retained so the revert guard above can undo a poisoned refit.
        buffer.append(np.asarray(ys).reshape(-1, n_dim))
        buffer = buffer[-16:]
        if block % 4 == 3:
            flow_prev, g_acc_ref = flow, float(g_acc)
            flow, _ = fit_flow(
                k_fit, flow, flow_train_set(k_fit), n_epochs=15, batch_size=512
            )
            if flow_wide is not None:
                # refit the wide flow on the same fresh window. No revert guard on
                # this one: its acceptance is expected to be low by construction, so
                # the guard's "acceptance collapsed" test cannot distinguish a bad
                # refit from normal operation. The narrow kernel is what carries
                # reproducibility, and it keeps its guard.
                key, k_fitw = jax.random.split(key)
                flow_wide, _ = fit_flow(
                    k_fitw,
                    flow_wide,
                    flow_train_set(k_fitw),
                    n_epochs=15,
                    batch_size=512,
                )
    timings["production"] = time.perf_counter() - t0

    ally = np.concatenate(kept)
    lps = np.concatenate(kept_lp)
    phys = np.asarray(to_phys(jnp.asarray(ally.reshape(-1, n_dim)))).reshape(ally.shape)
    converged = bool(rhat.max() < args.rhat_target and ess.min() >= args.ess_target)
    return phys, lps, rhat, ess, converged, kernel


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default="examples/output/td_phenomt_pe_hmc")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--sampling-rate", type=float, default=4096.0)
    ap.add_argument("--f-min", type=float, default=10.0)
    ap.add_argument("--f-max", type=float, default=None, help="default 0.45*rate")
    ap.add_argument("--distance", type=float, default=200.0, help="Mpc")
    ap.add_argument(
        "--target-snr",
        type=float,
        default=None,
        help="rescale --distance so the injected network SNR equals this value "
        "(exact: SNR is proportional to 1/distance for a fixed source, so one "
        "rebuild at the solved distance suffices); default keeps --distance as given",
    )
    ap.add_argument("--mass1", type=float, default=1.4, help="component mass [Msun]")
    ap.add_argument("--mass2", type=float, default=1.4, help="component mass [Msun]")
    ap.add_argument(
        "--eta-min", type=float, default=0.2, help="lower edge of the eta prior"
    )
    ap.add_argument(
        "--spin-min", type=float, default=0.0, help="lower edge of the spin priors"
    )
    ap.add_argument(
        "--spin-max", type=float, default=0.05, help="upper edge of the spin priors"
    )
    ap.add_argument(
        "--spin1z",
        type=float,
        default=0.0,
        help="injected aligned-spin truth, component 1",
    )
    ap.add_argument(
        "--spin2z",
        type=float,
        default=0.0,
        help="injected aligned-spin truth, component 2",
    )
    ap.add_argument("--phase-per-bin", type=float, default=0.5)
    ap.add_argument("--epsilon", type=float, default=0.25, help="RB phase per bin")
    ap.add_argument("--n-chains", type=int, default=64)
    ap.add_argument(
        "--kernel",
        choices=["hmc", "mala", "mmala", "uld", "random-walk"],
        default="hmc",
        help="local transition kernel (jaxpe.kernels). HMC is the only one with "
        "validated numbers on docs/td_phenomt_pe_benchmark.md; the other four use "
        "jaxpe's library-default adaptation targets, unbenchmarked on this "
        "posterior. uld has no MH step -- see --friction and the printed note",
    )
    ap.add_argument(
        "--friction",
        type=float,
        default=1.0,
        help="uld only: BAOAB friction coefficient (held fixed, no adaptation)",
    )
    ap.add_argument(
        "--target-acceptance",
        type=float,
        default=None,
        help="override the Robbins-Monro target acceptance for non-hmc kernels "
        "(default: jaxpe.kernels.adaptation.TARGET_ACCEPTANCE's literature value "
        "per kernel); has no effect for hmc (uses this script's own measured "
        "0.75) or uld (no acceptance to target)",
    )
    # --n-leapfrog / --warmup-leapfrog are HMC-only (trajectory length); unused
    # by the other four kernels, which have no such concept.
    ap.add_argument("--n-leapfrog", type=int, default=32)
    ap.add_argument(
        "--warmup-leapfrog",
        type=int,
        default=48,
        help="shorter trajectories suffice to adapt eps and seed the flow",
    )
    ap.add_argument("--flow-epochs", type=int, default=25)
    ap.add_argument(
        "--f32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="single-precision waveform + per-bin products (3x on this GPU)",
    )
    # 3, not 5. Under the eager-init overhead in run_chains this looked like an
    # 8 s difference ("within noise", so 5 was kept); with that overhead removed
    # the gap is 3.42 vs 4.01 min at the same seed -- cheaper equilibration
    # (34.7 vs 44.6 s) AND fewer production blocks (25 vs 31). A third stale
    # measurement taken in a regime a fixed cost dominated.
    ap.add_argument("--equil-rounds", type=int, default=3)
    ap.add_argument(
        "--ensemble-metric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="replace the Laplace metric with the equilibrated ensemble covariance",
    )
    ap.add_argument("--retune-blocks", type=int, default=3)
    ap.add_argument(
        "--adapt-gain",
        type=float,
        default=2.0,
        help="Robbins-Monro gain for warmup step-size adaptation",
    )
    ap.add_argument("--flow-acc-target", type=float, default=0.65)
    ap.add_argument("--flow-interval", type=float, default=8.0)
    # The global block is ~3.4 s of an ~8.3 s production block, and only 0.6 s of
    # that is the likelihood -- the rest is the flow's two passes per step
    # (sample + log_prob). Measured warm, per 1200-step block: 8 layers/width 64
    # = 3.38 s, 4/64 = 2.00 s, 4/32 = 1.64 s, 2/64 = 1.31 s. 8 coupling layers is
    # generous for a 4-dim posterior, so this is exposed to trade capacity for
    # speed -- but a weaker flow means worse proposals, so it must be judged end
    # to end (block COUNT), never on per-block cost alone.
    # 4 is the measured knee: 8 -> 4 holds the block count (24 vs 25) and is
    # faster at both seeds, but 2 layers/width 32 degrades capacity and costs
    # 33 blocks against 24, so the cheaper-block/more-blocks trade returns below 4.
    ap.add_argument("--flow-layers", type=int, default=4)
    ap.add_argument("--flow-width", type=int, default=64)
    # > 0 enables a SECOND flow at this wider interval, cycled with the narrow one
    # in production. Measured motivation: a single wide flow reaches the boundary
    # tails and cuts blocks 25 -> 8, but collapses acceptance to ~0.1 and makes the
    # run a lottery (3.87 min at one seed, 10.71 min at another). Cycling keeps the
    # narrow kernel's acceptance and reproducibility while still reaching the tails.
    ap.add_argument("--flow-interval-wide", type=float, default=0.0)
    ap.add_argument(
        "--reference",
        action="store_true",
        help="reproduce the 15.4-minute reference run's sampler settings",
    )
    ap.add_argument("--step-size", type=float, default=0.5)
    ap.add_argument("--warmup-blocks", type=int, default=5)
    ap.add_argument("--warmup-steps", type=int, default=15)
    # 12, not 25. This LOST under the eager-init overhead in run_chains (5.27 min
    # vs 4.83), because a 2.25 s fixed cost per call meant halving the steps could
    # only ever save 0.65 s of a 6.6 s block -- the test was rigged against itself.
    # With the overhead removed, measured over three seeds (42/7/13):
    #   ps=25: 3.42 / 4.38 / 3.43 min  (25 / 38 / 25 blocks)  worst 4.38
    #   ps=12: 3.49 / 2.95 / 2.89 min  (25 / 22 / 21 blocks)  worst 3.49
    ap.add_argument("--production-steps", type=int, default=12)
    ap.add_argument("--thin", type=int, default=2)
    ap.add_argument("--n-global", type=int, default=1200)
    # Equilibration and production both spend flow proposals, but for different
    # reasons -- equilibration to TRAIN the flow and spread the chains, production
    # to accumulate near-independent draws for Rhat. Tying them to one knob makes
    # any sweep of the production count silently pay for it twice in setup, so the
    # equilibration count is separable (defaults to --n-global for continuity).
    ap.add_argument("--n-global-equil", type=int, default=None)
    # A safety stop only -- --max-minutes is the real budget guard. It was 40, which
    # a 1.35+1.25 Msun source hit at Rhat 1.0107 and so reported as NOT converged
    # despite being ~2 blocks short; a cap that turns "needs a bit longer" into
    # "failed" is measuring the cap, not the sampler.
    ap.add_argument("--max-production-blocks", type=int, default=80)
    ap.add_argument("--rhat-target", type=float, default=1.01)
    ap.add_argument("--ess-target", type=float, default=2000.0)
    ap.add_argument("--max-minutes", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="small CPU smoke test")
    ap.add_argument(
        "--setup-only", action="store_true", help="stop after setup + RB validation"
    )
    args = ap.parse_args()
    if args.reference:  # the 15.4-minute configuration, for like-for-like reruns
        args.epsilon, args.n_chains, args.n_leapfrog = 0.1, 256, 128
        args.n_global, args.flow_epochs = 300, 40
        args.warmup_blocks, args.warmup_steps = 5, 25
        args.adapt_gain = 1.0
        args.equil_rounds, args.retune_blocks, args.f32 = 0, 0, False
        args.ensemble_metric, args.n_leapfrog, args.step_size = False, 128, 0.1
    if not args.warmup_leapfrog:
        args.warmup_leapfrog = args.n_leapfrog
    if args.quick:
        args.duration, args.sampling_rate, args.f_min = 128.0, 2048.0, 25.0
        args.n_chains, args.ess_target = 64, 500.0
        args.warmup_blocks, args.max_production_blocks = 5, 40
        args.production_steps = 50

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timings: dict = {}
    t_start = time.perf_counter()
    print(f"jax {jax.__version__}, default backend: {jax.default_backend()}")

    m1, m2 = max(args.mass1, args.mass2), min(args.mass1, args.mass2)
    mc_true = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    eta_true = m1 * m2 / (m1 + m2) ** 2
    if not (args.spin_min <= args.spin1z <= args.spin_max):
        raise ValueError(
            f"--spin1z={args.spin1z} outside the ({args.spin_min}, {args.spin_max}) "
            "prior; widen --spin-min/--spin-max"
        )
    if not (args.spin_min <= args.spin2z <= args.spin_max):
        raise ValueError(
            f"--spin2z={args.spin2z} outside the ({args.spin_min}, {args.spin_max}) "
            "prior; widen --spin-min/--spin-max"
        )
    truth = dict(
        chirp_mass=mc_true,
        mass_ratio=m2 / m1,
        spin1z=args.spin1z,
        spin2z=args.spin2z,
        luminosity_distance=args.distance,
        geocent_time=1187008882.43,
        phase=1.3,
        inclination=0.4,
        ra=3.446,
        dec=-0.408,
        psi=0.8,
    )

    # ---- heavy setup pinned to CPU: dense grids never touch the GPU ----
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        t0 = time.perf_counter()
        n = int(args.duration * args.sampling_rate)
        freqs = np.fft.rfftfreq(n, d=1.0 / args.sampling_rate)
        psd = lalsim_psd("CE", freqs)
        timings["psd"] = time.perf_counter() - t0

        # Hoisted so the rescale rebuild below cannot be conditioned differently from
        # this one. It previously passed tukey_alpha=0.0 on the rebuild only, so
        # --target-snr silently changed the analysis window as well as the distance.
        inj_kwargs = dict(
            detector_names=("H1",),  # H1 site geometry as the single-CE stand-in
            duration=args.duration,
            sampling_rate=args.sampling_rate,
            f_min=args.f_min,
            f_max=args.f_max,
            psd_fn=lambda f: np.interp(f, freqs, psd),
            noise_seed=None,  # zero-noise injection: lnL peaks at exactly 0 at truth
        )

        t0 = time.perf_counter()
        dense_like = make_injection(IMRPhenomT(f_ref=args.f_min), truth, **inj_kwargs)
        snr = dense_like.optimal_snr(truth)
        net_snr = network_snr(dense_like, truth)
        timings["injection"] = time.perf_counter() - t0
        print(
            f"injection: Mc={mc_true:.5f} Msun, D={args.distance:.0f} Mpc, "
            f"SNR {snr} (network {net_snr:.1f})  [{timings['injection']:.1f}s]"
        )

        if args.target_snr is not None:
            # h(f) scales as 1/D_L for a fixed source and orientation (the only
            # distance dependence in the detector response), so SNR = sqrt(<h|h>)
            # does too -- this rescale is EXACT, not an iterative or approximate
            # search, and one rebuild at the solved distance suffices.
            t0 = time.perf_counter()
            args.distance = distance_for_target_snr(dense_like, truth, args.target_snr)
            truth["luminosity_distance"] = args.distance
            dense_like = make_injection(
                IMRPhenomT(f_ref=args.f_min), truth, **inj_kwargs
            )
            snr = dense_like.optimal_snr(truth)
            net_snr = network_snr(dense_like, truth)
            timings["snr_rescale"] = time.perf_counter() - t0
            print(
                f"rescaled to target SNR {args.target_snr:.1f}: distance -> "
                f"{args.distance:.1f} Mpc, achieved network SNR {net_snr:.2f}  "
                f"[{timings['snr_rescale']:.1f}s]"
            )

        t0 = time.perf_counter()
        timings["rb_setup"] = 0.0

        prior = JointPrior(
            {
                "chirp_mass": Uniform(0.9 * mc_true, 1.1 * mc_true),
                "eta": Uniform(args.eta_min, 0.25),
                "spin1z": Uniform(args.spin_min, args.spin_max),
                "spin2z": Uniform(args.spin_min, args.spin_max),
            }
        )
        if not (args.eta_min < eta_true <= 0.25):
            raise ValueError(
                f"eta_true={eta_true:.4f} outside the eta prior "
                f"({args.eta_min}, 0.25]; widen --eta-min"
            )
        loglike = build_loglike(dense_like, truth, f32=args.f32)
        problem = InferenceProblem(prior=prior, log_likelihood=loglike)

        # Optimizer start: the fiducial (trigger) point itself, inset off any prior
        # edge it happens to lie on -- the sigmoid bijection sends the open bounds
        # to +-inf, so a start exactly on a boundary is not representable. Derived
        # from the prior support and the injection, with no numbers specific to a
        # particular binary: an equal-mass system starts inset from eta = 1/4, an
        # unequal-mass one starts at its own (interior) eta.
        lo = np.array([p.low for p in prior.priors])
        hi = np.array([p.high for p in prior.priors])
        inset = 0.02 * (hi - lo)
        x_fid = np.array(
            [truth["chirp_mass"], eta_true, truth["spin1z"], truth["spin2z"]]
        )
        x_init = np.clip(x_fid, lo + inset, hi - inset)
        import jax.numpy as jnp

        y_init = np.asarray(prior.to_unconstrained(jnp.asarray(x_init)))
        t0 = time.perf_counter()
        y_map, cov0, logp_map = map_laplace(problem, y_init)
        timings["map_laplace"] = time.perf_counter() - t0
        x_map = np.asarray(prior.to_physical(jnp.asarray(y_map)))
        print(
            f"MAP (unconstrained-space mode): x = {np.array2string(x_map, precision=6)}, "
            f"log-posterior {logp_map:.2f}  [{timings['map_laplace']:.1f}s]"
        )

        t0 = time.perf_counter()
        # physical-space posterior scale: sigma_y * |dx/dy| (bijections elementwise)
        # sampled-space truth (eta from the actual component masses, not assumed 1/4)
        # sampled-space truth (eta from the actual component masses, not assumed 1/4)
        x_true = np.array([mc_true, eta_true, truth["spin1z"], truth["spin2z"]])

        timings["rb_validation"] = 0.0

    if args.setup_only:
        timings["total"] = time.perf_counter() - t_start
        print(f"setup-only: done in {timings['total'] / 60.0:.2f} min")
        print(f"timings: { {k: round(v, 2) for k, v in timings.items()} }")
        return 0

    # ---- sampling on the default device (GPU when available) ----
    # the wall-clock budget covers the WHOLE run: hand production what remains
    budget_min = args.max_minutes
    args.max_minutes = max(2.0, budget_min - (time.perf_counter() - t_start) / 60.0)
    phys, lps, rhat, ess, converged, kernel = run_pe(
        problem, y_map, cov0, args, timings
    )
    timings["total"] = time.perf_counter() - t_start

    # ---- report ----
    names = list(prior.names)
    flat = phys.reshape(-1, phys.shape[-1])
    q05, q50, q95 = np.percentile(flat, [5, 50, 95], axis=0)
    print("\n===== results =====")
    print(f"converged: {converged}  (Rhat {rhat.max():.4f}, min ESS {ess.min():.0f})")
    for i, nme in enumerate(names):
        print(
            f"  {nme:>11s}: median {q50[i]:.6g}  90% CI [{q05[i]:.6g}, {q95[i]:.6g}]"
            f"  truth {x_true[i]:.6g}"
        )
    imax = int(np.argmax(lps))
    print(f"  max log-posterior sampled: {lps.reshape(-1)[imax]:.3f}")
    total_min = timings["total"] / 60.0
    print(f"wall time: {total_min:.2f} min (budget {budget_min:.0f} min)")
    print(f"timings: { {k: round(v, 2) for k, v in timings.items()} }")

    np.savez(
        outdir / "samples.npz",
        names=names,
        samples=flat,
        log_prob=lps.reshape(-1),
        truth=x_true,
        rhat=rhat,
        ess=ess,
        snr=net_snr,
    )
    with open(outdir / "timings.json", "w") as f:
        json.dump(
            {
                **{k: float(v) for k, v in timings.items()},
                "wall_min": total_min,
                "converged": converged,
                "backend": jax.default_backend(),
                "n_chains": int(args.n_chains),
                "network_snr": net_snr,
                "step_size": float(kernel.step_size),
            },
            f,
            indent=2,
        )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        d = len(names)
        fig, axes = plt.subplots(d, d, figsize=(2.4 * d, 2.4 * d))
        sub = flat[:: max(1, flat.shape[0] // 20000)]
        for i in range(d):
            for j in range(d):
                ax = axes[i, j]
                if j > i:
                    ax.axis("off")
                    continue
                if i == j:
                    ax.hist(flat[:, i], bins=80, histtype="step", color="C0")
                    ax.axvline(x_true[i], color="k", ls="--", lw=1)
                else:
                    ax.plot(sub[:, j], sub[:, i], ",", color="C0", alpha=0.3)
                    ax.plot(x_true[j], x_true[i], "k+", ms=10)
                if i == d - 1:
                    ax.set_xlabel(names[j])
                if j == 0 and i > 0:
                    ax.set_ylabel(names[i])
        fig.suptitle(
            f"BNS/CE FD relative binning + HMC (SNR {net_snr:.0f}, "
            f"{'converged' if converged else 'NOT converged'}, {total_min:.1f} min)"
        )
        fig.tight_layout()
        fig.savefig(outdir / "corner.png", dpi=110)
        print(f"saved {outdir}/corner.png")
    except ImportError:
        pass

    return 0 if converged and total_min < budget_min else 1


if __name__ == "__main__":
    raise SystemExit(main())
