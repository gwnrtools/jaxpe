"""Hamiltonian Monte Carlo with a fixed-length leapfrog trajectory.

Fixed trajectory length keeps every chain's step identical in shape, so the kernel
vmaps cleanly (NUTS-style dynamic trees do not). A per-dimension ``scale`` d plays the
role of sqrt(inverse mass): momenta are drawn as p ~ N(0, diag(1/d^2)) and the kinetic
energy is K(p) = ||d * p||^2 / 2.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp

from .base import Kernel, KernelState, LogProbFn, mh_accept


class HMC(Kernel):
    needs_gradient: ClassVar[bool] = True
    step_size: jax.Array
    n_leapfrog: int = 10
    scale: jax.Array | None = None  # (n_dim,) ~ sqrt(inverse mass) diagonal

    def __init__(self, step_size: float, n_leapfrog: int = 10, scale=None):
        self.step_size = jnp.asarray(step_size)
        self.n_leapfrog = n_leapfrog
        self.scale = None if scale is None else jnp.asarray(scale)

    def step(self, key, state: KernelState, logp_fn: LogProbFn):
        key_mom, key_acc = jax.random.split(key)
        d = 1.0 if self.scale is None else self.scale
        eps = self.step_size
        grad_fn = jax.value_and_grad(logp_fn)

        p0 = jax.random.normal(key_mom, state.x.shape, state.x.dtype) / d

        def leapfrog(carry, _):
            x, p, grad = carry
            p = p + 0.5 * eps * grad
            x = x + eps * d**2 * p
            _, grad = grad_fn(x)
            p = p + 0.5 * eps * grad
            return (x, p, grad), None

        (x_new, p_new, grad_new), _ = jax.lax.scan(
            leapfrog, (state.x, p0, state.grad), None, length=self.n_leapfrog
        )
        logp_new = logp_fn(x_new)

        kinetic0 = 0.5 * jnp.sum((d * p0) ** 2)
        kinetic1 = 0.5 * jnp.sum((d * p_new) ** 2)
        proposal = KernelState(x=x_new, log_prob=logp_new, grad=grad_new)
        log_ratio = logp_new - state.log_prob + kinetic0 - kinetic1
        return mh_accept(key_acc, state, proposal, log_ratio)
