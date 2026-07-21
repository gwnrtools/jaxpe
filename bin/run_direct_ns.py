#!/usr/bin/env python
"""Direct Nested Sampling baseline for frequency-domain dominant-mode signals.

This script implements Option 0 of the GPry-fusion design.
It sets up a matched-SNR IMRPhenomD injection (masses 60 and 30), marginalizes
over phase and distance analytically, and runs BlackJAX-NS *directly* on the
marginalized likelihood, completely bypassing the GPry surrogate.

Usage (from the repo root):
    JAX_PLATFORMS=cpu conda run -n lalsuite-dev python bin/run_direct_ns.py --mass-tot 60
"""

import argparse
import time
from pathlib import Path

import jax
import numpy as np

# float64 is REQUIRED: GPS times (~1.1e9 s) and Whittle accumulations are meaningless
# in float32 (lnL silently becomes NaN). Must be set before any jax array is created.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib

ex = importlib.import_module("examples.08_fd_dominant_mode_route_comparison")
make_mass_sweep = ex.make_mass_sweep
build_injection_likelihood = ex.build_injection_likelihood
_chirp_prior = ex._chirp_prior
MASS_RATIO_PRIOR = ex.MASS_RATIO_PRIOR
SPIN_PRIOR = ex.SPIN_PRIOR
REFERENCE_DISTANCE_MPC = ex.REFERENCE_DISTANCE_MPC
DISTANCE_PRIOR_MPC = ex.DISTANCE_PRIOR_MPC


def build_jittable_logp(marginal):
    """A fully JAX-traceable twin of ``PhaseDistanceMarginalLikelihood.__call__``.

    That ``__call__`` returns a Python float: the *overlaps* are jitted JAX
    (``marginal._overlaps``), but the phase+distance marginalization uses host scipy
    (``i0e``/``logsumexp`` over the precomputed distance grid), so it cannot be inlined
    into a jitted nested sampler. We rebuild the marginalization in ``jax.scipy`` from
    the same precomputed arrays, so the whole log-likelihood compiles and runs inside
    the NS step -- the honest apples-to-apples with the JAX acquisition (both fully
    jitted), instead of GPry's ``pure_callback`` point-by-point escape.

    Reads a few "private" attributes of the marginal on purpose (this is a benchmark
    that mirrors its internal math); a parity check against ``__call__`` guards it.
    """
    from jax.scipy.special import i0e, logsumexp

    u = jnp.asarray(marginal._u)
    log_pi = jnp.asarray(marginal._log_pi)
    log_dD = jnp.asarray(marginal._log_dD)
    dd = float(marginal.dd)
    overlaps = marginal._overlaps  # jitted JAX (zr, zi, rho2)

    def logp(x):
        zr, zi, rho2 = overlaps(jnp.asarray(x).ravel())
        abs_z = jnp.hypot(zr, zi)
        log_i0 = jnp.log(i0e(u * abs_z)) + u * abs_z
        integrand = log_pi + log_i0 - 0.5 * u**2 * rho2 + log_dD
        return logsumexp(integrand) - 0.5 * dd

    return logp


def run_direct_blackjax_ns(logp_jax, names, bounds, inj, seed=11):
    """Run the JAX-native BlackJAX nested sampler directly on the marginal likelihood.

    ``logp_jax`` is the jittable twin from ``build_jittable_logp``; it is inlined into
    the jitted NSS step (``JAXInterfaceBlackJAX``), compiled once (one NS run, no growing
    training set). Eval counting is impossible inside the compiled step; the Option-0
    decision quantity is wall-clock vs the Route-B surrogate wall at the same mass.
    """
    from jaxpe.surrogate.jax_acquisition import JAXInterfaceBlackJAX

    print(f"\n=== Direct BlackJAX-NS [{inj.label}] (JAX-native, jitted step) ===")

    ns = JAXInterfaceBlackJAX(bounds, verbosity=3)
    # GPry NORA-standard precision at this dimension
    ns.set_precision(
        nlive=25 * len(names),
        num_repeats=5 * len(names),
        precision_criterion=0.01,
    )

    # Compile/execute split: time one logp call (its own jit) before the NS run; the
    # NS-step compile itself lands inside the run wall (it happens once).
    x0 = jnp.asarray(0.5 * (bounds[:, 0] + bounds[:, 1]))
    t0 = time.perf_counter()
    jax.block_until_ready(logp_jax(x0))
    marginal_compile_seconds = time.perf_counter() - t0

    print(f"[direct:{inj.label}] Starting BlackJAX nested sampling...")
    started = time.time()
    X_MC, y_MC, w_MC, _logZ, _logZstd = ns.run(logp_jax, param_names=names, seed=seed)
    wall_seconds = time.time() - started

    print(
        f"[direct:{inj.label}] Done: {wall_seconds:.0f} s NS wall "
        f"(marginal jit compile {marginal_compile_seconds:.1f} s, "
        f"NS-step compile included in wall), {len(X_MC)} samples."
    )

    return dict(
        samples=np.asarray(X_MC),
        weights=np.asarray(w_MC),
        wall_seconds=wall_seconds,
        compile_seconds=marginal_compile_seconds,
        method="direct-ns",
        device=jax.devices()[0].platform,
        n_samples=len(X_MC),
        duration=inj.duration,
        n_freq=inj.n_freq,
        network_snr=inj.network_snr,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mass-tot",
        type=float,
        default=60.0,
        help="Total mass for the benchmark injection.",
    )
    ap.add_argument("--output", default="output/direct_ns", help="Output directory")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Build the injections (we only take the one matching the requested mass)
    injections = make_mass_sweep("phenomd", total_masses=[args.mass_tot])
    inj = injections[0]

    # 2. Build the exact marginalized likelihood
    from jaxpe.gw import PhaseDistanceMarginalLikelihood

    likelihood, n_freq = build_injection_likelihood("phenomd", inj)
    inj.n_freq = n_freq

    names = list(inj.recover)  # ('chirp_mass', 'mass_ratio', 'spin1z', 'spin2z')

    bounds = {}
    for name in names:
        if name == "chirp_mass":
            bounds[name] = _chirp_prior(inj)
        elif name == "mass_ratio":
            bounds[name] = MASS_RATIO_PRIOR
        elif name in ("spin1z", "spin2z"):
            bounds[name] = SPIN_PRIOR

    # Fixed extrinsics (sky location and inclination are fixed at truth)
    fixed_ext = {k: inj.params[k] for k in ("ra", "dec", "psi", "inclination")}
    for spin in ("spin1z", "spin2z"):
        if spin not in names:
            fixed_ext[spin] = inj.params[spin]

    marginal = PhaseDistanceMarginalLikelihood(
        likelihood,
        names,
        fixed_ext,
        dist_bounds=DISTANCE_PRIOR_MPC,
        dist_power=2.0,
        d_ref=REFERENCE_DISTANCE_MPC,
        check_params=inj.params,
    )

    print(
        f"[direct:{inj.label}] dominant-mode residual "
        f"{marginal.dominant_mode_residual:.2e}"
    )

    # Build the bounds array (d, 2)
    bounds_arr = np.array([bounds[n] for n in names])

    # 3. Build the jittable log-likelihood twin and check it reproduces the host-side
    #    marginal before trusting the NS run.
    logp_jax = build_jittable_logp(marginal)
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(20):
        xr = bounds_arr[:, 0] + rng.random(len(names)) * (
            bounds_arr[:, 1] - bounds_arr[:, 0]
        )
        max_err = max(max_err, abs(float(logp_jax(jnp.asarray(xr))) - marginal(xr)))
    print(f"[direct:{inj.label}] jittable-logp vs host marginal max|Δ| = {max_err:.2e}")
    assert max_err < 1e-6, "jittable logp does not match the host marginal"

    # 4. Run BlackJAX NS directly
    result = run_direct_blackjax_ns(logp_jax, names, bounds_arr, inj, seed=args.seed)

    # 5. Posterior vs truth
    s = result["samples"]
    w = np.asarray(result["weights"], dtype=float)
    w = w / w.sum()
    mean = np.average(s, axis=0, weights=w)
    std = np.sqrt(np.average((s - mean) ** 2, axis=0, weights=w))
    truth = np.array([inj.params[n] for n in names])
    print("\nparameter        truth      mean       std       z")
    for i, n in enumerate(names):
        z = (mean[i] - truth[i]) / std[i] if std[i] > 0 else float("nan")
        print(f"{n:14} {truth[i]:9.4f} {mean[i]:9.4f} {std[i]:9.4f} {z:7.2f}")

    # 6. Option 0 decision readout: same-mass Route-B surrogate wall, if on disk
    for tag in ("surrogate_cpu", "surrogate_jax_cpu"):
        p = (
            Path(__file__).parent.parent
            / "examples"
            / "output"
            / f"phenomd_M{int(args.mass_tot)}_{tag}.npz"
        )
        if p.exists():
            d = np.load(p, allow_pickle=True)
            if "wall_seconds" in d.files:
                ratio = float(d["wall_seconds"]) / result["wall_seconds"]
                verdict = (
                    f"direct is {ratio:.1f}x faster"
                    if ratio > 1
                    else f"surrogate is {1 / ratio:.1f}x faster"
                )
                print(
                    f"[option-0] Route-B {tag} wall {float(d['wall_seconds']):.0f}s vs "
                    f"direct-NS {result['wall_seconds']:.0f}s -> {verdict}"
                )

    # Save results
    np.savez(
        out / f"direct_ns_M{int(args.mass_tot)}.npz",
        samples=result["samples"],
        weights=result["weights"],
        names=names,
        truth=truth,
        wall_seconds=result["wall_seconds"],
        compile_seconds=result["compile_seconds"],
        method=result["method"],
        device=result["device"],
    )
    print(f"Saved results to {out / f'direct_ns_M{int(args.mass_tot)}.npz'}")


if __name__ == "__main__":
    main()
