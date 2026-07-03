"""Normalizing-flow proposal distribution, wrapping flowjax.

The flow is a rational-quadratic-spline coupling flow whose spline acts on a bounded
interval, so chain samples are standardized (affine whitening) before they reach the
flow and un-standardized on the way out. ``FlowProposal`` bundles the flow with its
whitening constants and exposes only ``log_prob`` / ``sample`` — everything the global
MH kernel and the trainer need.
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
