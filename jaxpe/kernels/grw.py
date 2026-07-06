"""Gaussian Random-Walk Metropolis (GRW).

This is the simplest form of Markov Chain Monte Carlo. It proposes jumps purely
by adding Gaussian noise to the current state, without using any gradient information
to guide the walk.

Motivation & Math
-----------------
The proposal mechanism is symmetric:
$$ q(x' | x) = \mathcal{N}(x, \sigma^2 I) $$
Because the proposal density is symmetric ($q(x'|x) = q(x|x')$), these terms cancel
out in the Metropolis-Hastings acceptance ratio $\alpha$:
$$ \alpha = \min\left(1, \frac{\pi(x')}{\pi(x)}\right) $$

While simple, it scales poorly to high dimensions because the random steps often propose
moves into low-probability regions, leading to low acceptance rates unless the step size
is made tiny (which in turn leads to slow exploration).
"""

from typing import ClassVar

import jax
import jax.numpy as jnp

from .base import Kernel, KernelState, LogProbFn, mh_accept


class RandomWalk(Kernel):
    """
    Gaussian Random Walk Kernel.

    Proposes new states via $x' = x + \\text{step\_size} \times \\text{scale} \times \mathcal{N}(0, I)$.

    Parameters
    ----------
    step_size : float
        The global step size scaling factor.
    scale : jax.Array | None, default=None
        An optional per-dimension proposal scale.
    """

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
