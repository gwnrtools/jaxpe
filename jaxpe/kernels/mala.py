"""Metropolis-adjusted Langevin algorithm (MALA).

One gradient evaluation per step and trivially vmappable, which makes it the default
local kernel for many-chain GPU sampling. An optional per-dimension ``scale`` d acts as
a diagonal preconditioner: the proposal is

    x' = x + (eps d)^2/2 * grad log p(x) + eps d * xi,   xi ~ N(0, I),

with the exact MH correction for the asymmetric proposal density.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp

from .base import Kernel, KernelState, LogProbFn, mh_accept


class MALA(Kernel):
    needs_gradient: ClassVar[bool] = True
    step_size: jax.Array
    scale: jax.Array | None = None  # (n_dim,) diagonal preconditioner

    def __init__(self, step_size: float, scale=None):
        self.step_size = jnp.asarray(step_size)
        self.scale = None if scale is None else jnp.asarray(scale)

    def step(self, key, state: KernelState, logp_fn: LogProbFn):
        key_prop, key_acc = jax.random.split(key)
        d = self.step_size * (1.0 if self.scale is None else self.scale)
        var = d**2

        mean_fwd = state.x + 0.5 * var * state.grad
        x_new = mean_fwd + d * jax.random.normal(key_prop, state.x.shape, state.x.dtype)
        logp_new, grad_new = jax.value_and_grad(logp_fn)(x_new)

        mean_rev = x_new + 0.5 * var * grad_new
        log_q_fwd = -0.5 * jnp.sum(((x_new - mean_fwd) / d) ** 2)
        log_q_rev = -0.5 * jnp.sum(((state.x - mean_rev) / d) ** 2)

        proposal = KernelState(x=x_new, log_prob=logp_new, grad=grad_new)
        log_ratio = logp_new - state.log_prob + log_q_rev - log_q_fwd
        return mh_accept(key_acc, state, proposal, log_ratio)
