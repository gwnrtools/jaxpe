r"""Between-loop Adaptation of Kernel Hyperparameters.

MCMC kernels require careful tuning of hyperparameters (e.g., step size, mass matrix).
If the step size is too large, most proposals are rejected. If too small, the chain
hardly moves.

Motivation & Math
-----------------
The fundamental idea behind step size adaptation is stochastic approximation, primarily
the Robbins-Monro algorithm. We want to find a step size $\epsilon$ such that the
average acceptance probability $\bar{\alpha}$ equals some theoretical optimal target $\alpha^*$.

For example, the optimal target for Random Walk is $\approx 0.234$, and for MALA it is $\approx 0.574$.
We update the log step size at each adaptation epoch:
$$ \log \epsilon_{i+1} = \log \epsilon_i + \gamma (\bar{\alpha}_i - \alpha^*) $$
where $\gamma$ is the learning rate.

Implementation Details
----------------------
In JAXPE, adaptation is deliberately kept *outside* the jitted sampling scans. The orchestrator
runs a block of steps, inspects the mean acceptance rate and the chain ensemble, and
rebuilds the kernel with an updated step size / preconditioner. Freezing adaptation for
production blocks keeps the chains exactly Markovian where it matters.
"""

import equinox as eqx
import jax.numpy as jnp

# Canonical optimal acceptance targets
TARGET_ACCEPTANCE = {"RandomWalk": 0.234, "MALA": 0.574, "MMALA": 0.574, "HMC": 0.65}


def adapted_step_size(
    step_size,
    accept_rate,
    target: float,
    gamma: float = 1.0,
    lo: float = 1e-8,
    hi: float = 1e3,
):
    r"""
    Robbins-Monro update of the step size toward a target acceptance rate.

    Parameters
    ----------
    step_size : float
        The current step size $\epsilon_i$.
    accept_rate : float
        The empirical mean acceptance rate $\bar{\alpha}_i$ from the last block of samples.
    target : float
        The theoretical optimal target acceptance rate $\alpha^*$.
    gamma : float, default=1.0
        The learning rate/step size for the Robbins-Monro update.
    lo : float, default=1e-8
        Minimum allowed step size.
    hi : float, default=1e3
        Maximum allowed step size.

    Returns
    -------
    float
        The new adapted step size.
    """
    new = jnp.exp(jnp.log(step_size) + gamma * (accept_rate - target))
    return jnp.clip(new, lo, hi)


def ensemble_scale(xs, floor: float = 1e-8):
    """Per-dimension std across an ensemble of samples, used as a diagonal preconditioner.

    ``xs`` has shape (..., n_dim); leading axes (steps, chains) are flattened.
    """
    flat = xs.reshape(-1, xs.shape[-1])
    std = jnp.std(flat, axis=0)
    return jnp.maximum(std, floor)


def ensemble_cov(xs, jitter: float = 1e-6):
    """Dense sample covariance across an ensemble, regularized for Cholesky safety."""
    flat = xs.reshape(-1, xs.shape[-1])
    cov = jnp.cov(flat.T)
    cov = jnp.atleast_2d(cov)
    return cov + jitter * jnp.trace(cov) / cov.shape[0] * jnp.eye(
        cov.shape[0], dtype=cov.dtype
    )


def with_updates(kernel, **updates):
    """Return a copy of ``kernel`` with the given array fields replaced."""
    names = list(updates.keys())
    return eqx.tree_at(
        lambda k: tuple(getattr(k, n) for n in names),
        kernel,
        tuple(jnp.asarray(v) for v in updates.values()),
        is_leaf=lambda leaf: leaf is None,
    )
