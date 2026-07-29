#!/usr/bin/env python
r"""BNS parameter estimation benchmark: Cosmic Explorer injection, FD relative
binning, and the jaxpe JAX HMC sampler.

Setup
-----
Injection: m1 = m2 = 1.4 Msun (chirp mass ~ 1.2188 Msun, eta = 0.25), zero spins,
f_lower = 10 Hz, sampling rate 4096 Hz, zero-noise data in a single detector with the
Cosmic Explorer P1600143 PSD (H1 site geometry as the CE stand-in). The segment is
2048 s, comfortably longer than the ~1000 s inspiral from 10 Hz, so the FD waveform is
wrap-free and df = 1/2048 Hz resolves the signal's frequency structure.

Priors (all uniform): chirp mass in [0.9, 1.1] x true, eta in [0.2, 0.25],
spin1z and spin2z in [0, 0.05]. Extrinsic parameters are fixed at the injected values.

Likelihood: :class:`~jaxpe.gw.likelihood.RelativeBinningFDLikelihood` summary data
built once on the dense grid (CPU), then a lean jitted log-posterior that evaluates
IMRPhenomD only at the ~n_bins bin edges. The lean path is asserted equal to the class
implementation at machine precision, and the heterodyne is validated against the dense
:class:`~jaxpe.gw.likelihood.FDNetworkLikelihood` on draws spanning the posterior bulk.

Sampler: jaxpe's HMC kernel with a *dense* mass matrix (the Laplace covariance at the
unconstrained-space MAP, found by damped Newton from the fiducial, with the soft
eigenvalues floored at 1 for the boundary-tail directions) and long leapfrog
trajectories (n_leapfrog ~ 128) that bend along the curved chirp-mass/eta/spin
degeneracy valley. Warmup Robbins-Monro-adapts only the step size; chains that
strand on secondary ripples of the oscillatory matched-filter likelihood (Delta
lnL ~ -10^3) are re-seeded before production. Production interleaves local HMC
blocks with flow-based global independence proposals (``jaxpe.flows`` +
``jaxpe.sampler``'s global block): the flow teleports chains along the
boundary-piled eta -> 1/4 and spin -> 0 tails, whose unconstrained-space
Exponential(1) geometry no fixed mass matrix can equilibrate (measured
tau ~ 30 steps for HMC alone, whatever the metric or trajectory length).

Convergence gate, evaluated per block: rank-normalized split-Rhat over the
*global-subseries* (near-independent draws; split-Rhat over the raw autocorrelated
series only re-measures tau) < 1.01, Geyer min ESS over the full series >= target,
and no stuck chains. All design decisions above were measured, not assumed -- see
docs/bns_ce_pe_benchmark.md for the experiment ledger.

The heavy setup runs on CPU regardless of the default device; the sampling hot loop
runs on the default device (GPU when available) touching only O(n_bins) constants.

Run:  python bin/run_bns_ce_pe.py               (full 20-minute-budget benchmark)
      python bin/run_bns_ce_pe.py --quick       (reduced CPU validation config)
      python bin/run_bns_ce_pe.py --setup-only  (profile setup + RB validation)
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

import jax

jax.config.update("jax_enable_x64", True)
# persistent XLA compilation cache: repeat invocations skip all jit compiles
# (the 20-minute benchmark budget excludes compile time; a warm second run is
# the honest measurement of it)
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jaxpe_xla"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

import jax.numpy as jnp
import numpy as np

from jaxpe.core.priors import JointPrior, Uniform
from jaxpe.core.problem import InferenceProblem
from jaxpe.diagnostics.stats import effective_sample_size, split_rhat
from jaxpe.gw import IMRPhenomD, make_injection
from jaxpe.gw.detectors import EARTH_OMEGA
from jaxpe.gw.likelihood import RelativeBinningFDLikelihood
from jaxpe.gw.likelihood.base import project_to_detector
from jaxpe.kernels import HMC, adapted_step_size, run_chains, with_updates

TARGET_ACC_HMC = 0.75


# --------------------------------------------------------------------------- setup
def cosmic_explorer_psd(freqs, flow: float = 5.0):
    """CE P1600143 PSD on the uniform grid ``freqs`` via lalsimulation (series API).

    Zero/undefined entries (f < flow and f = 0) are mapped to inf so that the
    likelihood's band mask ignores them.
    """
    import lal
    import lalsimulation as ls

    freqs = np.asarray(freqs, float)
    df = float(freqs[1] - freqs[0])
    ser = lal.CreateREAL8FrequencySeries(
        "psd", lal.LIGOTimeGPS(0), 0.0, df, lal.SecondUnit, freqs.size
    )
    ls.SimNoisePSDCosmicExplorerP1600143(ser, flow)
    psd = np.asarray(ser.data.data, float)
    return np.where(psd > 0.0, psd, np.inf)


def eta_to_q(eta):
    """q = m2/m1 <= 1 from symmetric mass ratio; safe-where keeps grads finite at eta=1/4."""
    d2 = jnp.maximum(1.0 - 4.0 * eta, 0.0)
    delta = jnp.where(d2 > 0.0, jnp.sqrt(jnp.where(d2 > 0.0, d2, 1.0)), 0.0)
    return (1.0 - delta) / (1.0 + delta)


def build_loglike(rb, fixed):
    """Lean jitted-friendly log-likelihood over (chirp_mass, eta, spin1z, spin2z).

    Mathematically identical to ``rb.log_likelihood`` but closed over ONLY the
    O(n_bins) summary arrays (as numpy constants, baked into the jit on the sampling
    device), so the multi-million-point setup grids never reach the GPU.
    """
    st = rb._static()
    edge_freqs = np.asarray(st["rb_edge_freqs"])
    dfbin = np.asarray(st["rb_dfbin"])
    half_dd = float(st["rb_half_dd"])
    gmst = rb.gmst_ref + EARTH_OMEGA * (fixed["geocent_time"] - rb.t_ref)
    dets = [
        (
            det,
            np.asarray(st["rb_h0_edges"][det.name]),
            np.asarray(st["rb_A0"][det.name]),
            np.asarray(st["rb_A1"][det.name]),
            np.asarray(st["rb_B0"][det.name]),
            np.asarray(st["rb_B1"][det.name]),
        )
        for det in rb.detectors
    ]
    waveform = rb.waveform
    ra, dec, psi = fixed["ra"], fixed["dec"], fixed["psi"]

    def loglike(p):
        full = dict(fixed)
        full["chirp_mass"] = p["chirp_mass"]
        full["mass_ratio"] = eta_to_q(p["eta"])
        full["spin1z"] = p["spin1z"]
        full["spin2z"] = p["spin2z"]
        hp, hc = waveform(full, edge_freqs)
        lnl = -half_dd
        for det, h0, A0, A1, B0, B1 in dets:
            h = project_to_detector(det, hp, hc, edge_freqs, ra, dec, psi, gmst)
            r = h / h0
            r0 = 0.5 * (r[1:] + r[:-1])
            r1 = (r[1:] - r[:-1]) / dfbin
            zdh = jnp.sum(A0 * jnp.conj(r0) + A1 * jnp.conj(r1))
            hh = jnp.sum(
                B0 * (r0.real**2 + r0.imag**2) + 2.0 * B1 * jnp.real(r0 * jnp.conj(r1))
            )
            lnl = lnl + jnp.real(zdh) - 0.5 * hh
        return lnl

    return loglike


# ----------------------------------------------------------------------- validation
def validate_rb(rb, dense_like, loglike, prior, truth, x_true, sigma, rng):
    """RB vs dense parity on draws spanning the posterior bulk and moderate tails.

    ``x_true`` is the truth in sampled-space order (chirp_mass, eta, spin1z, spin2z).
    Tolerance follows the Zackay error model (error ~ beta * |lnL|): require
    |RB - dense| < 0.1 in the bulk (|lnL| < 50) and < 5e-3 * |lnL| further out.
    Returns the worst (bulk_err, model_ratio) seen; raises on failure.
    """
    x_true = np.asarray(x_true, float)
    _dense_jit = jax.jit(dense_like.log_likelihood)  # 4M-point graph: jit pays off

    def dense_eval(params):  # jnp-array leaves so repeated calls do not retrace
        return float(_dense_jit({k: jnp.asarray(v) for k, v in params.items()}))

    lnl_rb_true = float(loglike(prior.as_dict(jnp.asarray(x_true))))
    lnl_dense_true = dense_eval(dict(truth))
    # both sides are O(<d|d>/2 ~ SNR^2/2) sums over ~1e5-1e6 points reduced in
    # different orders (numpy summary data vs fused XLA): exact-at-fiducial holds
    # to ~1e-8 relative, so the absolute tolerance must carry the <d|d>/2 scale
    tol_fid = 1e-8 * (1.0 + abs(rb._static()["rb_half_dd"]))
    if abs(lnl_rb_true - lnl_dense_true) > tol_fid:
        raise RuntimeError(
            f"exact-at-fiducial violated: RB {lnl_rb_true} vs dense "
            f"{lnl_dense_true} (tol {tol_fid:.2e})"
        )

    # lean closure == class implementation, machine precision
    for _ in range(3):
        x = x_true + sigma * rng.standard_normal(x_true.size)
        x = np.clip(x, [p.low for p in prior.priors], [p.high for p in prior.priors])
        p = prior.as_dict(jnp.asarray(x))
        full = dict(truth)
        full.update(
            chirp_mass=x[0], mass_ratio=float(eta_to_q(x[1])), spin1z=x[2], spin2z=x[3]
        )
        a, b = float(loglike(p)), float(rb.log_likelihood(full))
        if abs(a - b) > 1e-6 * (1.0 + abs(b)):
            raise RuntimeError(f"lean loglike != class loglike: {a} vs {b}")

    worst_bulk, worst_ratio = 0.0, 0.0
    for s in (0.3, 1.0, 3.0):
        for _ in range(4):
            x = x_true + s * sigma * rng.standard_normal(x_true.size)
            x = np.clip(
                x, [p.low for p in prior.priors], [p.high for p in prior.priors]
            )
            full = dict(truth)
            full.update(
                chirp_mass=x[0],
                mass_ratio=float(eta_to_q(x[1])),
                spin1z=x[2],
                spin2z=x[3],
            )
            lnl_rb = float(loglike(prior.as_dict(jnp.asarray(x))))
            lnl_d = dense_eval(full)
            err = abs(lnl_rb - lnl_d)
            if abs(lnl_d) < 50.0:
                worst_bulk = max(worst_bulk, err)
            else:
                worst_ratio = max(worst_ratio, err / abs(lnl_d))
    if worst_bulk > 0.1 or worst_ratio > 5e-3:
        raise RuntimeError(
            f"relative-binning parity too loose: bulk {worst_bulk:.3g} (tol 0.1), "
            f"tail ratio {worst_ratio:.3g} (tol 5e-3); decrease --epsilon"
        )
    return worst_bulk, worst_ratio


# ------------------------------------------------------------------------- sampling
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


def map_laplace(problem, y0, n_newton: int = 40, tol: float = 1e-9):
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
        w, V = np.linalg.eigh(-0.5 * (H + H.T))
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
    L0 = np.linalg.cholesky((V * w_mass) @ V.T)
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
    kernel = HMC(
        step_size=args.step_size, n_leapfrog=args.n_leapfrog, scale=jnp.asarray(L0)
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
    for block in range(args.warmup_blocks):
        key, k = jax.random.split(key)
        states, ys, _, infos = run_chains(
            k, kernel, logp, y0, args.warmup_steps, thin=3
        )
        y0 = states.x
        acc = float(jnp.mean(infos.accepted))
        accs.append(acc)
        kernel = with_updates(
            kernel,
            step_size=adapted_step_size(kernel.step_size, acc, TARGET_ACC_HMC),
        )
        if block >= 1:  # skip the pre-adaptation transient
            buffer.append(np.asarray(ys).reshape(-1, n_dim))
        if block == 0:
            timings["warmup_first_block_incl_compile"] = time.perf_counter() - t0
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
    flow = make_flow(k_make, n_dim, interval=8.0)
    flow, losses = fit_flow(
        k_fit, flow, flow_train_set(k_fit), n_epochs=40, batch_size=512
    )
    timings["flow_fit"] = time.perf_counter() - t0
    print(f"flow fit: loss {losses[-1]:.3f}  [{timings['flow_fit']:.1f}s]")

    # ---- production: adaptation frozen, convergence-gated blocks ----
    # Each block: a local HMC block, then n_global flow independence-MH steps.
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
    kept, kept_lp, kept_glob = [], [], []
    rhat = np.full(n_dim, np.inf)
    rhat_raw = np.full(n_dim, np.inf)
    ess = np.zeros(n_dim)
    eps_prod = float(kernel.step_size)
    eps_cycle = (1.0, 0.87, 1.13, 0.95)
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
        kept.append(np.asarray(ys))
        kept_lp.append(np.asarray(lps))

        y0, logp0, gys, glps, g_acc = _global_block(
            flow, k_glob, y0, logp0, logp, args.n_global
        )
        kept.append(np.asarray(gys))
        kept_lp.append(np.asarray(glps))
        kept_glob.append(np.asarray(gys))
        if flow_prev is not None:
            # first block on a fresh refit: revert it if acceptance collapsed
            # (a single bad refit poisoned two blocks at acc 0.07, measured);
            # freezing the flow outright is worse -- a mediocre early flow then
            # never improves and one slow direction stalls Rhat (also measured)
            if float(g_acc) < max(0.25, 0.6 * g_acc_ref):
                flow = flow_prev
            flow_prev = None

        # diagnostics on stride-subsampled series once they grow past ~2000
        # kept samples per chain (thinning cannot raise Rhat, and the ESS of the
        # thinned series is a conservative lower bound vs the ESS target)
        ally = np.concatenate(kept)  # (n_kept, n_chains, n_dim)
        ally = ally[:: max(1, ally.shape[0] // 2000)]
        phys = np.asarray(to_phys(jnp.asarray(ally.reshape(-1, n_dim)))).reshape(
            ally.shape
        )
        glob = np.concatenate(kept_glob)
        glob = glob[:: max(1, glob.shape[0] // 2000)]
        phys_glob = np.asarray(to_phys(jnp.asarray(glob.reshape(-1, n_dim)))).reshape(
            glob.shape
        )
        rhat = split_rhat(rank_normalized(phys_glob))
        rhat_raw = split_rhat(rank_normalized(phys))
        ess = effective_sample_size(phys)
        elapsed = time.perf_counter() - t0
        lp_now = np.asarray(logp0)
        n_stuck = int((lp_now < np.median(lp_now) - 20.0).sum())
        print(
            f"production block {block + 1}: acc {float(jnp.mean(infos.accepted)):.2f}, "
            f"global acc {float(g_acc):.2f}, "
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
        if block % 2 == 1:
            flow_prev, g_acc_ref = flow, float(g_acc)
            flow, _ = fit_flow(
                k_fit, flow, flow_train_set(k_fit), n_epochs=15, batch_size=512
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
    ap.add_argument("--outdir", default="examples/output/bns_ce_rb_hmc")
    ap.add_argument("--duration", type=float, default=2048.0)
    ap.add_argument("--sampling-rate", type=float, default=4096.0)
    ap.add_argument("--f-min", type=float, default=10.0)
    ap.add_argument("--f-max", type=float, default=None, help="default 0.45*rate")
    ap.add_argument("--distance", type=float, default=200.0, help="Mpc")
    ap.add_argument("--chi", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=0.1, help="RB phase per bin")
    ap.add_argument("--n-chains", type=int, default=256)
    ap.add_argument("--n-leapfrog", type=int, default=128)
    ap.add_argument("--step-size", type=float, default=0.1)
    ap.add_argument("--warmup-blocks", type=int, default=5)
    ap.add_argument("--warmup-steps", type=int, default=25)
    ap.add_argument("--production-steps", type=int, default=25)
    ap.add_argument("--thin", type=int, default=2)
    ap.add_argument("--n-global", type=int, default=300)
    ap.add_argument("--max-production-blocks", type=int, default=40)
    ap.add_argument("--rhat-target", type=float, default=1.01)
    ap.add_argument("--ess-target", type=float, default=2000.0)
    ap.add_argument("--max-minutes", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="small CPU smoke test")
    ap.add_argument(
        "--setup-only", action="store_true", help="stop after setup + RB validation"
    )
    args = ap.parse_args()
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

    m1 = m2 = 1.4
    mc_true = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    truth = dict(
        chirp_mass=mc_true,
        mass_ratio=1.0,
        spin1z=0.0,
        spin2z=0.0,
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
        psd = cosmic_explorer_psd(freqs)
        timings["psd"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        dense_like = make_injection(
            IMRPhenomD(f_ref=args.f_min),
            truth,
            detector_names=("H1",),  # H1 site geometry as the single-CE stand-in
            duration=args.duration,
            sampling_rate=args.sampling_rate,
            f_min=args.f_min,
            f_max=args.f_max,
            psd_fn=lambda f: np.interp(f, freqs, psd),
            noise_seed=None,  # zero-noise injection: lnL peaks at exactly 0 at truth
        )
        snr = dense_like.optimal_snr(truth)
        net_snr = float(np.sqrt(sum(v**2 for v in snr.values())))
        timings["injection"] = time.perf_counter() - t0
        print(
            f"injection: Mc={mc_true:.5f} Msun, D={args.distance:.0f} Mpc, "
            f"SNR {snr} (network {net_snr:.1f})  [{timings['injection']:.1f}s]"
        )

        t0 = time.perf_counter()
        rb = RelativeBinningFDLikelihood.from_likelihood(
            dense_like, truth, chi=args.chi, epsilon=args.epsilon
        )
        n_bins = rb.n_bins
        timings["rb_setup"] = time.perf_counter() - t0
        n_band = int(np.sum((freqs >= args.f_min) & (freqs <= rb.f_max)))
        print(
            f"relative binning: {n_band} band points -> {n_bins} bins "
            f"[{timings['rb_setup']:.1f}s]"
        )

        prior = JointPrior(
            {
                "chirp_mass": Uniform(0.9 * mc_true, 1.1 * mc_true),
                "eta": Uniform(0.2, 0.25),
                "spin1z": Uniform(0.0, 0.05),
                "spin2z": Uniform(0.0, 0.05),
            }
        )
        loglike = build_loglike(rb, truth)
        problem = InferenceProblem(prior=prior, log_likelihood=loglike)

        # optimizer start nudged off the eta = 1/4 and spin = 0 prior boundaries
        x_init = np.array([mc_true, 0.2497, 0.004, 0.004])
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
        jac = np.asarray(jax.jacfwd(prior.to_physical)(jnp.asarray(y_map)))
        sigma_phys = np.sqrt(np.diag(cov0)) * np.abs(np.diag(jac))
        rng = np.random.default_rng(args.seed)
        x_true = np.array([mc_true, 0.25, 0.0, 0.0])  # sampled-space truth
        wb, wr = validate_rb(
            rb, dense_like, loglike, prior, truth, x_true, sigma_phys, rng
        )
        timings["rb_validation"] = time.perf_counter() - t0
        print(
            f"RB parity vs dense: bulk err {wb:.2e} (tol 0.1), tail ratio {wr:.2e} "
            f"(tol 5e-3); sigma_phys ~ {np.array2string(sigma_phys, precision=2)} "
            f"[{timings['rb_validation']:.1f}s]"
        )

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
                "converged": converged,
                "backend": jax.default_backend(),
                "n_bins": int(n_bins),
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
