"""Gaussian random-walk Metropolis: gradient-free baseline and fallback."""

from typing import ClassVar

import jax
import jax.numpy as jnp

from .base import Kernel, KernelState, LogProbFn, mh_accept


class RandomWalk(Kernel):
    """x' = x + step_size * scale * N(0, I)."""

    needs_gradient: ClassVar[bool] = False
    step_size: jax.Array
    scale: jax.Array | None = None  # (n_dim,) per-dimension proposal scale

    def __init__(self, step_size: float, scale=None):
        self.step_size = jnp.asarray(step_size)
        self.scale = None if scale is None else jnp.asarray(scale)

    def step(self, key, state: KernelState, logp_fn: LogProbFn):
        key_prop, key_acc = jax.random.split(key)
        d = self.step_size * (1.0 if self.scale is None else self.scale)
        x_new = state.x + d * jax.random.normal(key_prop, state.x.shape, state.x.dtype)
        proposal = KernelState(x=x_new, log_prob=logp_fn(x_new), grad=state.grad)
        return mh_accept(key_acc, state, proposal, proposal.log_prob - state.log_prob)
