#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Ensure float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".jax_cache"))

from jaxpe.diagnostics import corner_plot
from jaxpe.gw import ESIGMAInspiral, ebbh_priors, make_injection
from jaxpe.gw.psd import psd_from_file
from jaxpe.kernels import MALA
from jaxpe.sampler import GlobalLocalConfig, Sampler, best_of_prior_init


def main():
    parser = argparse.ArgumentParser(description="Run Production PE on an injection")
    parser.add_argument(
        "--injection-json",
        type=str,
        required=True,
        help="Path to JSON file with injection parameters",
    )
    parser.add_argument(
        "--prior-json",
        type=str,
        required=True,
        help="Path to JSON file with prior bounds",
    )
    parser.add_argument(
        "--psd-file",
        type=str,
        default=None,
        help="Path to 2-column ASCII PSD file (optional)",
    )
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")

    # Sampler configuration
    parser.add_argument("--n-chains", type=int, default=100)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--n-production", type=int, default=2000)
    parser.add_argument("--pn-order", type=int, default=8)

    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.injection_json, "r") as f:
        injection_params = json.load(f)

    with open(args.prior_json, "r") as f:
        prior_kwargs = json.load(f)

    print(f"Loading ESIGMAInspiral (PN={args.pn_order}, forward_sensitivity)...")
    waveform = ESIGMAInspiral(
        f_lower=30.0,
        modes=((2, 2), (3, 3)),
        rad_pn_order=args.pn_order,
        mode_pn_order=args.pn_order,
        ode_eps=1e-9,  # production fidelity
        n_ode_grid=1024,
        max_ode_steps=1024,
        adjoint_mode="forward_sensitivity",
    )

    psd_fn = None
    if args.psd_file:
        print(f"Loading PSD from {args.psd_file}...")
        psd_fn = psd_from_file(args.psd_file)

    print("Generating injection...")
    kwargs = dict(
        waveform=waveform,
        injection_params=injection_params,
        detector_names=("H1", "L1", "V1"),
        duration=4.0,
        sampling_rate=1024.0,
        f_min=30.0,
        noise_seed=None,
    )
    if psd_fn is not None:
        kwargs["psd_fn"] = psd_fn

    like = make_injection(**kwargs)

    print(
        "Injected SNR:",
        like.optimal_snr({k: jnp.asarray(v) for k, v in injection_params.items()}),
    )

    # Prepare prior
    prior = ebbh_priors(**prior_kwargs)
    problem = like.problem(prior)

    # Setup sampler
    per_loop = args.n_chains * (100 // 5)
    cfg = GlobalLocalConfig(
        n_chains=args.n_chains,
        n_prelim_loops=3,
        n_training_loops=30,
        n_production_loops=args.n_production,
        n_local_steps=100,
        n_global_steps=100,
        local_thin=5,
        buffer_size=15 * per_loop,
        flow_layers=8,
        nn_width=64,
        n_epochs=args.n_epochs,
        batch_size=min(1024, 15 * per_loop),
    )

    sampler = Sampler(MALA(step_size=0.05), problem=problem, config=cfg)

    print("Starting init...")
    t0 = time.time()
    key = jax.random.PRNGKey(42)
    x0 = best_of_prior_init(key, problem, cfg.n_chains, n_draws=1_000)
    print(f"init: best-of-prior in {time.time() - t0:.1f} s")

    print(
        f"Starting sampler.run with {args.n_chains} chains... (This will trigger XLA compilation)"
    )
    t0 = time.time()
    res = sampler.run(key, x0=x0)
    dt_run = time.time() - t0
    print("Sampling complete!")

    phys = sampler.to_physical(res.samples)
    flat = phys.reshape(-1, problem.n_dim)
    names = list(problem.names)
    truths = [injection_params.get(n, None) for n in names]

    print(f"\nsampling wall time: {dt_run:.1f} s on {jax.devices()[0].platform}")
    print(f"production samples: {flat.shape[0]}")

    print("\nRecovery Summary (Median [16%, 84%]):")
    for i, n in enumerate(names):
        q16, q50, q84 = np.percentile(flat[:, i], [16, 50, 84])
        t_str = f"{truths[i]:8.3f}" if truths[i] is not None else "N/A"
        print(f"  {n:20s}: {q50:8.3f} [{q16:8.3f}, {q84:8.3f}]  (True: {t_str})")

    np.save(outdir / "posterior_samples.npy", flat)

    # Save corner plot if truths are mostly available
    try:
        fig = corner_plot(flat, names=names, truths=truths)
        fig.savefig(outdir / "corner.png", dpi=120)
        print(f"\nsaved corner plot to {outdir / 'corner.png'}")
    except Exception as e:
        print(f"Warning: Could not generate corner plot: {e}")


if __name__ == "__main__":
    main()
