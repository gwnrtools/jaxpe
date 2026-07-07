"""Core-layer tests: transform round-trips, Jacobians vs autodiff, prior normalization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from jaxpe.core import (
    Affine,
    Cosine,
    Gaussian,
    Identity,
    InferenceProblem,
    Interval,
    JointPrior,
    LogUniform,
    PowerLaw,
    Sine,
    Uniform,
)

BIJECTIONS = [
    Identity(),
    Affine(loc=2.0, scale=3.5),
    Interval(low=-1.0, high=4.0),
]


@pytest.mark.parametrize("bij", BIJECTIONS)
def test_bijection_roundtrip(bij):
    y = jnp.linspace(-5.0, 5.0, 41)
    np.testing.assert_allclose(bij.inverse(bij.forward(y)), y, atol=1e-10)


@pytest.mark.parametrize("bij", BIJECTIONS)
def test_bijection_log_det_matches_autodiff(bij):
    ys = jnp.linspace(-4.0, 4.0, 17)
    autodiff = jnp.log(jnp.abs(jax.vmap(jax.grad(bij.forward))(ys)))
    np.testing.assert_allclose(bij.log_det(ys), autodiff, atol=1e-10)


PRIORS = [
    Uniform(low=-2.0, high=5.0),
    LogUniform(low=0.1, high=30.0),
    PowerLaw(alpha=2.0, low=10.0, high=1000.0),
    PowerLaw(alpha=-3.0, low=1.0, high=8.0),
    Sine(),
    Cosine(),
    Gaussian(mu=1.0, sigma=2.0),
]


@pytest.mark.parametrize("prior", PRIORS)
def test_prior_normalized(prior):
    if isinstance(prior, Gaussian):
        lo, hi = prior.mu - 20 * prior.sigma, prior.mu + 20 * prior.sigma
    else:
        lo, hi = prior.low, prior.high
    integral, err = quad(
        lambda x: float(jnp.exp(prior.log_prob(jnp.asarray(x)))), lo, hi
    )
    assert abs(integral - 1.0) < max(1e-8, 10 * err)


@pytest.mark.parametrize("prior", PRIORS)
def test_prior_sample_moments(prior):
    """Sampled mean matches the mean computed by quadrature from log_prob."""
    key = jax.random.PRNGKey(0)
    samples = prior.sample(key, (200_000,))
    if isinstance(prior, Gaussian):
        lo, hi = prior.mu - 20 * prior.sigma, prior.mu + 20 * prior.sigma
    else:
        lo, hi = prior.low, prior.high
        assert samples.min() >= lo and samples.max() <= hi
    mean_quad, _ = quad(
        lambda x: x * float(jnp.exp(prior.log_prob(jnp.asarray(x)))), lo, hi
    )
    std = float(samples.std())
    assert abs(float(samples.mean()) - mean_quad) < 5 * std / np.sqrt(len(samples))


def _joint():
    return JointPrior(
        {
            "a": Uniform(low=0.0, high=1.0),
            "b": Gaussian(mu=-1.0, sigma=0.5),
            "c": Sine(),
        }
    )


def test_joint_roundtrip_and_dict():
    joint = _joint()
    key = jax.random.PRNGKey(1)
    x = joint.sample(key, 100)
    assert x.shape == (100, 3)
    y = jax.vmap(joint.to_unconstrained)(x)
    x2 = jax.vmap(joint.to_physical)(y)
    np.testing.assert_allclose(x2, x, atol=1e-9)
    d = joint.as_dict(x)
    np.testing.assert_allclose(joint.from_dict(d), x)


def test_joint_unconstrained_density_normalized():
    """log_prob_unconstrained must integrate to 1 over R^n (checked in 1-D)."""
    joint = JointPrior({"a": Uniform(low=2.0, high=3.0)})
    integral, err = quad(
        lambda y: float(jnp.exp(joint.log_prob_unconstrained(jnp.asarray([y])))),
        -30,
        30,
    )
    assert abs(integral - 1.0) < 1e-6


def test_problem_log_posterior_gradient_finite():
    joint = _joint()
    problem = InferenceProblem(
        prior=joint,
        log_likelihood=lambda p: -0.5 * (p["a"] - 0.5) ** 2 - 0.5 * p["b"] ** 2,
    )
    key = jax.random.PRNGKey(2)
    y = problem.sample_unconstrained(key, 7)
    logp = jax.vmap(problem.log_posterior)(y)
    grads = jax.vmap(jax.grad(problem.log_posterior))(y)
    assert jnp.all(jnp.isfinite(logp))
    assert jnp.all(jnp.isfinite(grads))


def test_problem_maps_nan_to_neg_inf():
    joint = JointPrior({"a": Uniform(low=0.0, high=1.0)})
    problem = InferenceProblem(prior=joint, log_likelihood=lambda p: jnp.nan * p["a"])
    assert problem.log_posterior(jnp.asarray([0.3])) == -jnp.inf
