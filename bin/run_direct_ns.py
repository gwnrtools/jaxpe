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


def run_direct_blackjax_ns(marginal, names, bounds, inj, seed=11):
    """Run the JAX-native BlackJAX nested sampler directly on the marginal likelihood.

    Uses ``JAXInterfaceBlackJAX`` (jaxpe.surrogate.jax_acquisition): the marginal is a
    pure JAX function, so it is inlined into the jitted NSS step -- no ``pure_callback``
    point-by-point escape to Python (the stock GPry interface's path, which would make
    this baseline meaninglessly slow). Closure-based jitting is fine here: unlike the
    in-loop acquisition there is exactly ONE nested-sampling run, so the step is
    compiled once.

    Note on eval counting: the likelihood runs inside the compiled step, so host-side
    call counting is impossible; the decision quantity for Option 0 is wall-clock
    against the Route-B surrogate wall at the same mass (printed by ``main``).
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

    # Compile/execute split: time one marginal call (its own jit) before the NS run;
    # the NS-step compile itself lands inside the run wall and is reported as part of
    # it (it happens once).
    x0 = jnp.asarray(0.5 * (bounds[:, 0] + bounds[:, 1]))
    t0 = time.perf_counter()
    jax.block_until_ready(marginal(x0))
    marginal_compile_seconds = time.perf_counter() - t0

    print(f"[direct:{inj.label}] Starting BlackJAX nested sampling...")
    started = time.time()
    X_MC, y_MC, w_MC, _logZ, _logZstd = ns.run(marginal, param_names=names, seed=seed)
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

    # 3. Run BlackJAX NS directly
    result = run_direct_blackjax_ns(marginal, names, bounds_arr, inj, seed=args.seed)

    # 4. Posterior vs truth
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

    # 5. Option 0 decision readout: same-mass Route-B surrogate wall, if on disk
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
