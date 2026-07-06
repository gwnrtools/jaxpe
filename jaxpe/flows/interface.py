"""Normalizing Flow Proposal Distribution (wraps flowjax).

A Normalizing Flow constructs a complex probability distribution by transforming a
simple base distribution (like a standard Normal) through a sequence of invertible,
differentiable mappings (bijections).

Motivation & Math
-----------------
Let $z \sim \mathcal{N}(0, I)$ be a sample from the base distribution. We apply a
bijective function $f_\phi: Z \to X$. The probability density of the transformed
variable $x = f_\phi(z)$ is given by the change of variables formula:
$$ p(x) = p(z) \left| \det \left( \\frac{\partial f_\phi^{-1}(x)}{\partial x} \right) \right| $$

Here we use a Rational-Quadratic Spline (RQS) coupling flow. The spline maps
intervals to intervals monotonically. Because splines act on bounded intervals (e.g., [-5, 5]),
we first standardize the MCMC chain samples (affine whitening) so they fit nicely within
the flow's domain, and then un-standardize on the way out.

Implementation Details
----------------------
``FlowProposal`` bundles the underlying flow with its whitening constants (mean and std).
It exposes only two key methods needed by the MH kernel and the trainer:
1. ``log_prob(y)``: Computes the log-density of a sample $y$.
2. ``sample(key, shape)``: Draws samples from the flow.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import AbstractDistribution, Normal
from flowjax.flows import coupling_flow


def _typed_key(key):
    """flowjax requires new-style typed PRNG keys; accept raw uint32 keys too."""
    if not jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        key = jax.random.wrap_key_data(key)
    return key


class FlowProposal(eqx.Module):
    flow: AbstractDistribution
    mean: jax.Array  # (n_dim,) whitening constants for the training data
    std: jax.Array  # (n_dim,)

    def log_prob(self, y):
        z = (y - self.mean) / self.std
        return self.flow.log_prob(z) - jnp.sum(jnp.log(self.std))

    def sample(self, key, shape=()):
        z = self.flow.sample(_typed_key(key), shape)
        return self.mean + self.std * z


def make_flow(
    key,
    n_dim: int,
    flow_layers: int = 8,
    knots: int = 8,
    interval: float = 5.0,
    nn_width: int = 64,
    nn_depth: int = 1,
) -> FlowProposal:
    """RQ-spline coupling flow over whitened unconstrained coordinates."""
    flow = coupling_flow(
        key,
        base_dist=Normal(jnp.zeros(n_dim)),
        transformer=RationalQuadraticSpline(knots=knots, interval=interval),
        flow_layers=flow_layers,
        nn_width=nn_width,
        nn_depth=nn_depth,
    )
    return FlowProposal(flow=flow, mean=jnp.zeros(n_dim), std=jnp.ones(n_dim))
