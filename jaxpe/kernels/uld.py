"""Underdamped (Kinetic) Langevin Dynamics (ULD).

Unlike Overdamped Langevin (which only has position), Underdamped Langevin Dynamics
simulates a particle with both position and momentum (velocity). This creates smoother,
longer-ranging trajectories.

Motivation & Math
-----------------
The continuous-time system is:
$$ dx_t = v_t dt $$
$$ dv_t = -\\nabla U(x_t) dt - \gamma v_t dt + \sqrt{2\gamma} dW_t $$
where $U(x) = -\log \pi(x)$, $\gamma$ is the friction coefficient, and $W_t$ is Brownian noise.

We simulate this using the BAOAB splitting method.
- **B**: Velocity update by half-step using gradients.
- **A**: Position update by half-step using velocity.
- **O**: Exact Ornstein-Uhlenbeck damping (friction and noise) for a full step.

ULD in this implementation is "unadjusted" (no Metropolis-Hastings correction).
Therefore, it explores extremely fast (momentum carries it through elongated posteriors)
but suffers from an $\mathcal{O}(\epsilon^2)$ discretization bias in the invariant density.
It is excellent for burn-in, but MALA or HMC should be used for exact sampling.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp

from .base import Kernel, KernelState, LogProbFn, StepInfo


class ULD(Kernel):
    """
    Underdamped Langevin Dynamics Kernel.

    Parameters
    ----------
    step_size : float
        The time step size $\epsilon$ for the BAOAB integrator.
    friction : float, default=1.0
        The friction coefficient $\gamma$ damping the velocity.
    scale : jax.Array | None, default=None
        Per-dimension preconditioner. Equivalent to simulating in $x/d$ coordinates.
    """

    needs_gradient: ClassVar[bool] = True
    has_accept_prob: ClassVar[bool] = (
        False  # no MH step: acceptance-based adaptation is meaningless
    )
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
