r"""Metropolis-Adjusted Langevin Algorithm (MALA).

MALA is one of the most efficient local MCMC kernels for many-chain GPU sampling
because it requires exactly one gradient evaluation per step and is trivially vmappable
(no inner loops like HMC's leapfrog).

Motivation & Math
-----------------
Imagine a particle floating in a fluid, subject to Brownian motion (random kicks)
and a drift force pulling it toward regions of high probability. This physical process
is described by the Overdamped Langevin Stochastic Differential Equation (SDE):
$$ d\theta_t = \nabla \log \pi(\theta_t) dt + \sqrt{2} dW_t $$
where $W_t$ is a standard Wiener process (Brownian motion).

If we simulate this continuously, the particle's stationary distribution is exactly
the target density $\pi(\theta)$. In discrete time, we use the Euler-Maruyama approximation:
$$ \theta_{t+1} = \theta_t + \frac{\epsilon^2}{2} \nabla \log \pi(\theta_t) + \epsilon \xi $$
where $\xi \sim \mathcal{N}(0, I)$ and $\epsilon$ is the step size.

Because time discretization introduces integration error, the resulting distribution
would be slightly biased. MALA corrects this by adding a Metropolis-Hastings (MH)
accept/reject step.

The proposal density $q(\theta' | \theta)$ is a Gaussian centered at $\theta + \frac{\epsilon^2}{2} \nabla \log \pi(\theta)$,
which makes the proposal asymmetric (i.e., $q(\theta' | \theta) \neq q(\theta | \theta')$).
The exact MH ratio accounts for this asymmetry.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp

from .base import Kernel, KernelState, LogProbFn, mh_accept


class MALA(Kernel):
    """
    Metropolis-Adjusted Langevin Algorithm Kernel.
    Parameters
    ----------
    step_size : float
        The base step size $\epsilon$ for the Langevin diffusion.
    scale : jax.Array | None, default=None
        An optional per-dimension diagonal preconditioner $d$. If provided, the
        proposal becomes:
        $x' = x + \\frac{(\epsilon d)^2}{2} \\nabla \\log p(x) + \epsilon d \\xi$.
    """

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
