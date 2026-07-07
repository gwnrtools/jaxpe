"""End-to-end GW PE on a toy-BBH injection with the global-local sampler on GPU.

Injects a ToyChirp signal into simulated aLIGO-design noise in H1+L1, samples the
9-parameter posterior, and writes a corner plot + posterior samples to
examples/output/. Runtime knobs are sized for a small (4 GB) GPU; scale n_chains up
on bigger cards.
"""

import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jaxpe.diagnostics import corner_plot, effective_sample_size, split_rhat
from jaxpe.gw import ToyChirp, bbh_priors, make_injection
from jaxpe.kernels import MALA
from jaxpe.sampler import GlobalLocalConfig, Sampler, best_of_prior_init

OUT = Path(__file__).parent / "output"

T_C = 1126259462.4
INJECTION = dict(
    chirp_mass=30.0,
    mass_ratio=0.8,
    luminosity_distance=700.0,
    inclination=0.6,
    phase=1.2,
    ra=1.95,
    dec=-1.27,
    psi=0.82,
    geocent_time=T_C,
)


def main(noise_seed=42, n_chains=80):
    like = make_injection(
        ToyChirp(f_start=20.0),
        INJECTION,
        detector_names=("H1", "L1"),
        duration=8.0,
        sampling_rate=2048.0,
        f_min=20.0,
        noise_seed=noise_seed,
    )
    print(
        "optimal SNRs:",
        like.optimal_snr({k: jnp.asarray(v) for k, v in INJECTION.items()}),
    )

    prior = bbh_priors(
        chirp_mass=(25.0, 35.0),
        mass_ratio=(0.25, 1.0),
        luminosity_distance=(100.0, 2000.0),
        geocent_time=T_C,
        time_width=0.1,
    )
    problem = like.problem(prior)

    # schedule notes for a concentrated 9-dim posterior: local-only warmup keeps
    # burn-in junk out of the flow's training buffer, and the buffer window holds only
    # the ~8 most recent loops so the flow tracks the chains as they concentrate
    per_loop = n_chains * (100 // 5)
    cfg = GlobalLocalConfig(
        n_chains=n_chains,
        n_prelim_loops=3,
        n_training_loops=50,
        n_production_loops=12,
        n_local_steps=100,
        n_global_steps=100,
        local_thin=5,
        buffer_size=15 * per_loop,
        flow_layers=8,
        nn_width=64,
        n_epochs=6,
        batch_size=1024,
    )
    sampler = Sampler(MALA(step_size=0.05), problem=problem, config=cfg)

    t0 = time.time()
    # needle-in-haystack initialization: best of many vmapped prior draws seeds every
    # comparable-likelihood mode (e.g. the two-detector sky reflection)
    key = jax.random.PRNGKey(1)
    x0 = best_of_prior_init(key, problem, cfg.n_chains, n_draws=20_000)
    print(f"init: best-of-prior in {time.time() - t0:.1f} s")
    res = sampler.run(key, x0=x0)
    dt_run = time.time() - t0

    phys = sampler.to_physical(res.samples)
    flat = phys.reshape(-1, problem.n_dim)
    names = list(problem.names)
    truths = [INJECTION[n] for n in names]

    print(f"\nsampling wall time: {dt_run:.1f} s on {jax.devices()[0].platform}")
    print(f"production samples: {flat.shape[0]}")
    print("R-hat:", dict(zip(names, np.round(split_rhat(phys), 3))))
    print("ESS:", dict(zip(names, np.round(effective_sample_size(phys)))))
    print("local acc:", " ".join(f"{a:.2f}" for a in res.local_acceptance))
    print("global acc:", " ".join(f"{a:.2f}" for a in res.global_acceptance))
    print("flow loss:", " ".join(f"{l:.1f}" for l in res.flow_losses))
    print("\nposterior (median [16%, 84%]) vs truth:")
    for i, n in enumerate(names):
        q16, q50, q84 = np.percentile(flat[:, i], [16, 50, 84])
        print(
            f"  {n:22s} {q50:10.4f} [{q16:10.4f}, {q84:10.4f}]   truth {truths[i]:10.4f}"
        )

    OUT.mkdir(exist_ok=True)
    tag = "zero_noise" if noise_seed is None else f"seed{noise_seed}"
    np.save(OUT / f"gw_injection_{tag}_samples.npy", flat)
    fig = corner_plot(flat, names=names, truths=truths)
    fig.savefig(OUT / f"gw_injection_{tag}_corner.png", dpi=120)
    print(f"\nsaved corner -> {OUT / f'gw_injection_{tag}_corner.png'}")
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n-chains",
        type=int,
        default=80,
        help="scale to GPU memory: ~48 for a shared 4 GB card, 512+ on an A40",
    )
    ap.add_argument("--zero-noise", action="store_true")
    args = ap.parse_args()
    main(noise_seed=None if args.zero_noise else 42, n_chains=args.n_chains)
