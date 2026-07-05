"""Minimal generic-density usage: correlated Gaussian via the raw logp_fn interface."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from jaxpe.diagnostics import effective_sample_size, split_rhat
from jaxpe.kernels import MALA
from jaxpe.sampler import GlobalLocalConfig, Sampler

MEAN = jnp.array([1.0, -2.0, 0.5])
L = jnp.array([[0.5, 0.0, 0.0], [0.6, 0.8, 0.0], [-0.3, 0.4, 3.0]])
COV_INV = jnp.linalg.inv(L @ L.T)


def logp(x):
    d = x - MEAN
    return -0.5 * d @ COV_INV @ d


def main():
    cfg = GlobalLocalConfig(n_chains=128, n_training_loops=6, n_production_loops=4)
    sampler = Sampler(MALA(step_size=0.3), logp_fn=logp, n_dim=3, config=cfg)
    key = jax.random.PRNGKey(0)
    x0 = jax.random.normal(key, (cfg.n_chains, 3))
    res = sampler.run(key, x0=x0)

    flat = res.flat()
    print("mean:", flat.mean(0), " (true:", np.asarray(MEAN), ")")
    print("std :", flat.std(0), " (true:", np.sqrt(np.diag(np.asarray(L @ L.T))), ")")
    print("R-hat:", split_rhat(res.samples))
    print("ESS  :", effective_sample_size(res.samples))
    print("local acc:", [f"{a:.2f}" for a in res.local_acceptance])
    print("global acc:", [f"{a:.2f}" for a in res.global_acceptance])


if __name__ == "__main__":
    main()
