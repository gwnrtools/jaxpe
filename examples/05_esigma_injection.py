"""Example: Parameter estimation for an eccentric BBH signal using ESIGMA.

This script injects a simulated eccentric BBH waveform (generated using
the ESIGMAInspiral model) into Gaussian noise, and runs the GlobalLocal
sampler to recover the parameters.

Requires:
  - esigmapy (for the waveform)
"""

import time
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)
# Enable persistent compilation cache so XLA doesn't recompile on subsequent runs
jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".jax_cache"))

import jax.numpy as jnp
import numpy as np

try:
    from jaxpe.gw import ESIGMAInspiral
except ImportError:
    raise ImportError("Please install esigmapy to run this example")

from jaxpe.diagnostics import corner_plot
from jaxpe.gw import (
    ebbh_priors,
    make_injection,
)
from jaxpe.kernels import MALA
from jaxpe.sampler import GlobalLocalConfig, Sampler, best_of_prior_init

OUT = Path(__file__).parent / "output"


def main(n_chains: int = 100, n_epochs: int = 100, n_production: int = 2000, pn_order: int = 8):
    f_lower = 30.0
    duration = 4.0
    t_c = 1126259462.4

    print("Initializing ESIGMA waveform model...")
    print(f"  Using PN order {pn_order} with relaxed ode_eps=1e-3 and max_ode_steps=512 to shrink the AD tape.")
    waveform = ESIGMAInspiral(
        f_lower=f_lower,
        modes=((2, 2), (3, 3)),
        rad_pn_order=pn_order,
        mode_pn_order=pn_order,
        ode_eps=1e-3,
        n_ode_grid=512,
        max_ode_steps=512,
    )

    INJECTION = dict(
        chirp_mass=25.0,
        mass_ratio=0.8,
        eccentricity=0.15,
        mean_anomaly=1.0,
        luminosity_distance=400.0,
        inclination=0.4,
        phase=1.5,
        geocent_time=t_c,
        ra=1.2,
        dec=0.5,
        psi=0.8,
    )

    print("Generating injection...")
    like = make_injection(
        waveform,
        INJECTION,
        detector_names=("H1", "L1", "V1"),
        duration=duration,
        sampling_rate=1024.0,
        f_min=f_lower,
        noise_seed=None,
    )

    print("Injected SNR:", like.optimal_snr({k: jnp.asarray(v) for k, v in INJECTION.items()}))

    prior = ebbh_priors(
        chirp_mass=(20.0, 30.0),
        mass_ratio=(0.5, 1.0),
        eccentricity=(0.0, 0.3),
        luminosity_distance=(200.0, 800.0),
        geocent_time=t_c,
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

    print("Starting init...")
    t0 = time.time()
    key = jax.random.PRNGKey(42)
    x0 = best_of_prior_init(key, problem, cfg.n_chains, n_draws=1_000)
    print(f"init: best-of-prior in {time.time() - t0:.1f} s")

    print(f"Starting sampler.run with {n_chains} chains... (This will trigger XLA compilation)")
    t0 = time.time()
    res = sampler.run(key, x0=x0)
    dt_run = time.time() - t0
    print("Sampling complete!")

    phys = sampler.to_physical(res.samples)
    flat = phys.reshape(-1, problem.n_dim)
    names = list(problem.names)
    truths = [INJECTION[n] for n in names]

    print(f"\nsampling wall time: {dt_run:.1f} s on {jax.devices()[0].platform}")
    print(f"production samples: {flat.shape[0]}")

    print("\nRecovery Summary (Median [16%, 84%]):")
    for i, n in enumerate(names):
        q16, q50, q84 = np.percentile(flat[:, i], [16, 50, 84])
        print(f"  {n:20s}: {q50:8.3f} [{q16:8.3f}, {q84:8.3f}]  (True: {truths[i]:8.3f})")

    OUT.mkdir(exist_ok=True)
    np.save(OUT / "esigma_injection_samples.npy", flat)
    fig = corner_plot(flat, names=names, truths=truths)
    fig.savefig(OUT / "esigma_injection_corner.png", dpi=120)
    print(f"\nsaved corner plot to {OUT / 'esigma_injection_corner.png'}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-chains", type=int, default=20)
    ap.add_argument("--n-epochs", type=int, default=6)
    ap.add_argument("--n-production", type=int, default=10)
    ap.add_argument("--pn-order", type=int, default=8)
    args = ap.parse_args()
    main(n_chains=args.n_chains, n_epochs=args.n_epochs, n_production=args.n_production, pn_order=args.pn_order)
