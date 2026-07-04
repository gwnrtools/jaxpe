"""Underdamped (kinetic) Langevin dynamics with the BAOAB splitting.

Unadjusted: there is no Metropolis correction, so the invariant density carries an
O(step_size^2) discretization bias — use small steps, and prefer MALA/HMC when exact
stationarity matters (e.g. final production runs). The payoff is non-reversible,
momentum-carrying exploration that mixes quickly through elongated posteriors.

The velocity lives in ``KernelState.aux`` and persists across steps; the O-step
applies the exact Ornstein-Uhlenbeck damping with friction ``gamma``. A per-dimension
``scale`` d preconditions the dynamics (equivalent to simulating in x/d coordinates).
"""

from typing import ClassVar

import jax
import jax.numpy as jnp

from .base import Kernel, KernelState, LogProbFn, StepInfo


class ULD(Kernel):
    needs_gradient: ClassVar[bool] = True
    has_accept_prob: ClassVar[bool] = False  # no MH step: acceptance-based adaptation is meaningless
    step_size: jax.Array
    friction: jax.Array
    scale: jax.Array | None = None

    def __init__(self, step_size: float, friction: float = 1.0, scale=None):
        self.step_size = jnp.asarray(step_size)
        self.friction = jnp.asarray(friction)
        self.scale = None if scale is None else jnp.asarray(scale)

    def init(self, x, logp_fn: LogProbFn) -> KernelState:
        log_prob, grad = jax.value_and_grad(logp_fn)(x)
        return KernelState(x=x, log_prob=log_prob, grad=grad, aux=jnp.zeros_like(x))

    def step(self, key, state: KernelState, logp_fn: LogProbFn):
        eps = self.step_size
        d = 1.0 if self.scale is None else self.scale
        x, v = state.x, state.aux

        v = v + 0.5 * eps * d * state.grad  # B
        x = x + 0.5 * eps * d * v  # A
        c1 = jnp.exp(-self.friction * eps)  # O: exact OU damping
        noise = jax.random.normal(key, v.shape, v.dtype)
        v = c1 * v + jnp.sqrt(1.0 - c1**2) * noise
        x = x + 0.5 * eps * d * v  # A
        log_prob, grad = jax.value_and_grad(logp_fn)(x)
        v = v + 0.5 * eps * d * grad  # B

        new = KernelState(x=x, log_prob=log_prob, grad=grad, aux=v)
        one = jnp.ones((), x.dtype)
        return new, StepInfo(accepted=one, log_accept_ratio=jnp.zeros((), x.dtype))
