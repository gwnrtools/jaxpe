"""Flow training and global-local sampler tests on multimodal targets.

The acid test: a well-separated Gaussian mixture where a purely local kernel cannot
move between modes, so recovering the correct mode weights requires the flow-driven
global jumps.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jaxpe.core import Gaussian, InferenceProblem, JointPrior, Uniform
from jaxpe.diagnostics import effective_sample_size, split_rhat
from jaxpe.flows import fit_flow, make_flow
from jaxpe.kernels import MALA
from jaxpe.sampler import GlobalLocalConfig, Sampler

SEP = 4.0  # mode separation in units of mode std (modes at +-SEP/2 ... well separated)


def mixture_logp(x):
    """Two isotropic 2-D Gaussians (std 0.5) at (+-2, +-2), weights 0.35 / 0.65."""
    mu = 2.0
    std = 0.5
    log_w = jnp.log(jnp.array([0.35, 0.65]))
    centers = jnp.array([[-mu, -mu], [mu, mu]])
    sq = jnp.sum((x[None, :] - centers) ** 2, axis=-1)
    comps = log_w - sq / (2 * std**2) - 2 * jnp.log(std) - jnp.log(2 * jnp.pi)
    return jax.scipy.special.logsumexp(comps)


def _mode_weight(samples):
    """Fraction of samples in the (+, +) mode."""
    flat = np.asarray(samples).reshape(-1, 2)
    return float(np.mean(flat.sum(axis=1) > 0))


def test_flow_fits_bimodal_data():
    key = jax.random.PRNGKey(0)
    k_data, k_flow, k_fit, k_sample = jax.random.split(key, 4)
    n = 20_000
    comp = jax.random.bernoulli(k_data, 0.65, (n,))
    centers = jnp.where(comp[:, None], 2.0, -2.0) * jnp.ones((n, 2))
    data = centers + 0.5 * jax.random.normal(k_data, (n, 2))

    flow = make_flow(k_flow, 2, flow_layers=6, nn_width=32)
    flow, losses = fit_flow(k_fit, flow, data, n_epochs=12, batch_size=512)
    assert losses[-1] < losses[0], "training did not reduce the loss"

    draws = flow.sample(k_sample, (8000,))
    w = _mode_weight(draws)
    assert 0.5 < w < 0.8, f"flow mode weight {w} (expected ~0.65)"
    # flow density should strongly prefer a mode center over the midpoint
    gap = flow.log_prob(jnp.array([2.0, 2.0])) - flow.log_prob(jnp.array([0.0, 0.0]))
    assert gap > 2.0


def test_global_local_recovers_mixture_weights():
    """MALA alone cannot rebalance chains between modes; the flow moves must."""
    cfg = GlobalLocalConfig(
        n_chains=64,
        n_training_loops=8,
        n_production_loops=6,
        n_local_steps=100,
        n_global_steps=40,
        local_thin=5,
        flow_layers=6,
        nn_width=32,
        n_epochs=6,
        batch_size=512,
    )
    sampler = Sampler(MALA(step_size=0.3), logp_fn=mixture_logp, n_dim=2, config=cfg)
    # start ALL chains in the wrong balance: 90% in the low-weight (-,-) mode
    key = jax.random.PRNGKey(1)
    k0, k1, krun = jax.random.split(key, 3)
    comp = jax.random.bernoulli(k0, 0.1, (cfg.n_chains,))
    x0 = jnp.where(comp[:, None], 2.0, -2.0) + 0.3 * jax.random.normal(
        k1, (cfg.n_chains, 2)
    )

    res = sampler.run(krun, x0=x0)

    w = _mode_weight(res.samples)
    assert abs(w - 0.65) < 0.08, f"mode weight {w}, expected 0.65"
    assert (
        max(res.global_acceptance[-3:]) > 0.2
    ), f"global acceptance too low: {res.global_acceptance}"
    rhat = split_rhat(res.samples)
    assert np.all(rhat < 1.15), f"R-hat {rhat}"
    ess = effective_sample_size(res.samples)
    assert np.all(ess > 200), f"ESS {ess}"

    # moments of the mixture: mean = (0.3*2, 0.3*2) = (0.6, 0.6), var = 0.25 + (1-0.09)*4
    flat = res.flat()
    np.testing.assert_allclose(flat.mean(0), [0.6, 0.6], atol=0.15)
    np.testing.assert_allclose(flat.var(0), 0.25 + 0.91 * 4.0, rtol=0.1)


def test_sampler_with_problem_physical_output():
    """Problem-based path: bounded prior, unimodal likelihood; physical samples in bounds."""
    prior = JointPrior(
        {"a": Uniform(low=0.0, high=10.0), "b": Gaussian(mu=0.0, sigma=2.0)}
    )
    problem = InferenceProblem(
        prior=prior,
        log_likelihood=lambda p: (
            -0.5 * ((p["a"] - 3.0) / 0.2) ** 2 - 0.5 * ((p["b"] - 1.0) / 0.3) ** 2
        ),
    )
    cfg = GlobalLocalConfig(
        n_chains=32,
        n_training_loops=5,
        n_production_loops=3,
        n_local_steps=80,
        n_global_steps=20,
        flow_layers=4,
        nn_width=24,
        n_epochs=5,
        batch_size=512,
    )
    sampler = Sampler(MALA(step_size=0.2), problem=problem, config=cfg)
    res = sampler.run(jax.random.PRNGKey(2))
    phys = sampler.to_physical(res.samples).reshape(-1, 2)

    assert np.all(phys[:, 0] > 0.0) and np.all(phys[:, 0] < 10.0)
    assert abs(phys[:, 0].mean() - 3.0) < 0.1
    assert abs(phys[:, 1].mean() - 1.0) < 0.15
