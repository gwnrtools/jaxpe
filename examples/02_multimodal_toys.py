"""Multimodal targets that defeat purely local MCMC: Gaussian mixture and dual moons.

Demonstrates the flow-driven global moves rebalancing probability mass between modes.
Outputs corner plots to examples/output/.
"""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jaxpe.diagnostics import corner_plot, split_rhat
from jaxpe.kernels import MALA
from jaxpe.sampler import GlobalLocalConfig, Sampler

OUT = Path(__file__).parent / "output"


def mixture_logp(x):
    """Two 2-D Gaussians at (+-2, +-2), std 0.5, weights 0.35/0.65."""
    log_w = jnp.log(jnp.array([0.35, 0.65]))
    centers = jnp.array([[-2.0, -2.0], [2.0, 2.0]])
    sq = jnp.sum((x[None, :] - centers) ** 2, axis=-1)
    return jax.scipy.special.logsumexp(log_w - sq / 0.5 - jnp.log(0.5 * jnp.pi))


def dual_moon_logp(x):
    """Two crescents; the classic flowMC demo target (here in 2-D)."""
    r = jnp.linalg.norm(x)
    term1 = -0.5 * ((r - 2.0) / 0.1) ** 2
    term2 = jax.scipy.special.logsumexp(
        jnp.stack([-0.5 * ((x[0] - 2.0) / 0.6) ** 2, -0.5 * ((x[0] + 2.0) / 0.6) ** 2])
    )
    return term1 + term2


def run(name, logp, n_dim=2):
    cfg = GlobalLocalConfig(
        n_chains=128, n_training_loops=10, n_production_loops=5,
        n_local_steps=100, n_global_steps=50, flow_layers=6, nn_width=48,
    )
    sampler = Sampler(MALA(step_size=0.2), logp_fn=logp, n_dim=n_dim, config=cfg)
    key = jax.random.PRNGKey(3)
    x0 = 3.0 * jax.random.normal(key, (cfg.n_chains, n_dim))
    res = sampler.run(key, x0=x0)

    print(f"[{name}] R-hat: {split_rhat(res.samples)}")
    print(f"[{name}] local acc (last): {res.local_acceptance[-1]:.2f}, "
          f"global acc (last): {res.global_acceptance[-1]:.2f}")
    OUT.mkdir(exist_ok=True)
    fig = corner_plot(res.flat(), names=[f"x{i}" for i in range(n_dim)])
    fig.savefig(OUT / f"{name}.png", dpi=120)
    print(f"[{name}] corner -> {OUT / f'{name}.png'}")


if __name__ == "__main__":
    run("mixture", mixture_logp)
    run("dual_moon", dual_moon_logp)
