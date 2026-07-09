"""Script to run 5 production-level BBH injections using IMRPhenomD."""

import time
import json
from pathlib import Path
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
# Enable persistent compilation cache so XLA doesn't recompile on subsequent runs
jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".jax_cache"))

import jax.numpy as jnp

from jaxpe.gw import IMRPhenomD
from jaxpe.diagnostics import corner_plot
from jaxpe.gw import bbh_priors, make_injection
from jaxpe.kernels import MALA
from jaxpe.sampler import GlobalLocalConfig, Sampler, best_of_prior_init

OUT = Path(__file__).resolve().parent.parent / "output" / "production_events"
OUT.mkdir(parents=True, exist_ok=True)

EVENTS = {
    "GW150914": {
        "params": dict(
            chirp_mass=28.1,
            mass_ratio=0.81,
            spin1z=0.0,
            spin2z=0.0,
            luminosity_distance=410.0,
            inclination=0.4,
            phase=1.5,
            geocent_time=1126259462.4,
            ra=1.2,
            dec=0.5,
            psi=0.8,
        ),
        "mc_prior": (20.0, 35.0),
        "dist_prior": (100.0, 1000.0),
    },
    "GW170729": {
        "params": dict(
            chirp_mass=35.7,
            mass_ratio=0.68,
            spin1z=0.0,
            spin2z=0.0,
            luminosity_distance=2840.0,
            inclination=0.4,
            phase=1.5,
            geocent_time=1185389807.3,
            ra=1.2,
            dec=0.5,
            psi=0.8,
        ),
        "mc_prior": (25.0, 50.0),
        "dist_prior": (1000.0, 5000.0),
    },
    "GW170104": {
        "params": dict(
            chirp_mass=21.1,
            mass_ratio=0.62,
            spin1z=0.0,
            spin2z=0.0,
            luminosity_distance=880.0,
            inclination=0.4,
            phase=1.5,
            geocent_time=1167559936.6,
            ra=1.2,
            dec=0.5,
            psi=0.8,
        ),
        "mc_prior": (15.0, 30.0),
        "dist_prior": (300.0, 1500.0),
    },
    "GW190412": {
        "params": dict(
            chirp_mass=13.3,
            mass_ratio=0.28,
            spin1z=0.4,
            spin2z=0.0,
            luminosity_distance=740.0,
            inclination=0.4,
            phase=1.5,
            geocent_time=1239082262.2,
            ra=1.2,
            dec=0.5,
            psi=0.8,
        ),
        "mc_prior": (10.0, 20.0),
        "dist_prior": (200.0, 1500.0),
    },
    "GW190521": {
        "params": dict(
            chirp_mass=64.4,
            mass_ratio=0.78,
            spin1z=0.3,
            spin2z=0.3,
            luminosity_distance=5300.0,
            inclination=0.4,
            phase=1.5,
            geocent_time=1242442967.4,
            ra=1.2,
            dec=0.5,
            psi=0.8,
        ),
        "mc_prior": (50.0, 80.0),
        "dist_prior": (2000.0, 10000.0),
    },
}


def main(n_chains=100, n_epochs=100, n_production=1000):
    f_lower = 20.0
    duration = 4.0

    print("Initializing IMRPhenomD waveform model...")
    waveform = IMRPhenomD(f_ref=f_lower)

    summary = {}

    for name, spec in EVENTS.items():
        print("\n=============================================")
        event_out = OUT / name
        event_out.mkdir(parents=True, exist_ok=True)

        if (event_out / "samples.npy").exists():
            print(f"Skipping {name}, samples.npy already exists (fully complete)")
            continue

        params = spec["params"]
        mc_prior = spec["mc_prior"]
        dist_prior = spec["dist_prior"]

        print("Generating zero-noise injection...")
        like = make_injection(
            waveform,
            params,
            detector_names=("H1", "L1", "V1"),
            duration=duration,
            sampling_rate=2048.0,
            f_min=f_lower,
            noise_seed=None,
        )

        snr = like.optimal_snr({k: jnp.asarray(v) for k, v in params.items()})
        print(f"Injected SNR: {snr}")

        prior = bbh_priors(
            chirp_mass=mc_prior,
            mass_ratio=(0.1, 1.0),
            aligned_spins=(-0.9, 0.9),
            luminosity_distance=dist_prior,
            geocent_time=params["geocent_time"],
            time_width=0.1,
        )
        problem = like.problem(prior)

        per_loop = n_chains * (100 // 5)
        cfg = GlobalLocalConfig(
            n_chains=n_chains,
            n_prelim_loops=3,
            n_training_loops=30,
            n_production_loops=n_production,
            n_local_steps=100,
            n_global_steps=100,
            local_thin=5,
            buffer_size=15 * per_loop,
            flow_layers=8,
            nn_width=64,
            n_epochs=n_epochs,
            batch_size=min(1024, 15 * per_loop),
        )

        sampler = Sampler(MALA(step_size=0.05), problem=problem, config=cfg)

        raw_file = event_out / "raw_samples.npz"
        ckpt_file = event_out / "checkpoint.eqx"

        if raw_file.exists():
            print(
                f"Found {raw_file}, skipping MCMC sampling and jumping to post-processing."
            )
            raw_data = np.load(raw_file)
            samples = raw_data["samples"]
            dt_init = float(raw_data["dt_init"])
            dt_run = float(raw_data["dt_run"])

            # Construct a dummy SamplerResults-like object for to_physical
            class DummyRes:
                pass

            res = DummyRes()
            res.samples = samples
        else:
            print("Starting best-of-prior init...")
            t0 = time.time()
            key = jax.random.PRNGKey(42)
            x0 = best_of_prior_init(key, problem, cfg.n_chains, n_draws=20_000)
            dt_init = time.time() - t0
            print(f"Init finished in {dt_init:.1f} s")

            print(
                f"Starting MCMC run with {n_chains} chains, {n_epochs} epochs, {n_production} production loops..."
            )
            t0 = time.time()
            res = sampler.run(key, x0=x0, checkpoint_file=ckpt_file)
            dt_run = time.time() - t0
            print("Sampling complete!")

            # Save raw samples in case post-processing fails
            np.savez(
                raw_file,
                samples=res.samples,
                log_prob=res.log_prob,
                dt_init=dt_init,
                dt_run=dt_run,
            )

            # Cleanup checkpoint
            if ckpt_file.exists():
                ckpt_file.unlink()
            if Path(str(ckpt_file) + ".flow").exists():
                Path(str(ckpt_file) + ".flow").unlink()
            if Path(str(ckpt_file) + ".kernel").exists():
                Path(str(ckpt_file) + ".kernel").unlink()

        phys = sampler.to_physical(res.samples)
        flat = phys.reshape(-1, problem.n_dim)
        pnames = list(problem.names)
        truths = [params[n] for n in pnames]

        n_samples = flat.shape[0]
        print(f"Sampling wall time: {dt_run:.1f} s")
        print(f"Total production samples: {n_samples}")

        # Recovery stats
        recovery = {}
        for i, pname in enumerate(pnames):
            q16, q50, q84 = np.percentile(flat[:, i], [16, 50, 84])
            recovery[pname] = {
                "median": float(q50),
                "q16": float(q16),
                "q84": float(q84),
                "true": float(truths[i]),
            }

        summary[name] = {
            "snr": {k: float(v) for k, v in snr.items()},
            "timings": {
                "init_time_s": float(dt_init),
                "sampling_time_s": float(dt_run),
            },
            "n_samples": int(n_samples),
            "recovery": recovery,
        }

        # Save artifacts

        np.save(event_out / "samples.npy", flat)
        fig = corner_plot(flat, names=pnames, truths=truths)
        fig.savefig(event_out / "corner.png", dpi=120)

        with open(event_out / "summary.json", "w") as f:
            json.dump(summary[name], f, indent=2)

        print(f"Saved artifacts for {name}")

    with open(OUT / "full_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chains", type=int, default=100)
    ap.add_argument("--n-epochs", type=int, default=100)
    ap.add_argument("--n-production", type=int, default=1000)
    args = ap.parse_args()
    main(
        n_chains=args.n_chains,
        n_epochs=args.n_epochs,
        n_production=args.n_production,
    )
