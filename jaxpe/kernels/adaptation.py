"""Between-loop adaptation of kernel hyperparameters.

Adaptation is deliberately kept *outside* the jitted sampling scans: the orchestrator
runs a block of steps, inspects the mean acceptance rate and the chain ensemble, and
rebuilds the kernel with an updated step size / preconditioner. Freezing adaptation for
production blocks keeps the chains exactly Markovian where it matters.
"""

import equinox as eqx
import jax.numpy as jnp

# Canonical optimal acceptance targets
TARGET_ACCEPTANCE = {"RandomWalk": 0.234, "MALA": 0.574, "MMALA": 0.574, "HMC": 0.65}


def adapted_step_size(step_size, accept_rate, target: float, gamma: float = 1.0,
                      lo: float = 1e-8, hi: float = 1e3):
    """Robbins-Monro update of the step size toward a target acceptance rate."""
    new = jnp.exp(jnp.log(step_size) + gamma * (accept_rate - target))
    return jnp.clip(new, lo, hi)


def ensemble_scale(xs, floor: float = 1e-8):
    """Per-dimension std across an ensemble of samples, used as a diagonal preconditioner.

    ``xs`` has shape (..., n_dim); leading axes (steps, chains) are flattened.
    """
    flat = xs.reshape(-1, xs.shape[-1])
    std = jnp.std(flat, axis=0)
    return jnp.maximum(std, floor)


def with_updates(kernel, **updates):
    """Return a copy of ``kernel`` with the given array fields replaced."""
    names = list(updates.keys())
    return eqx.tree_at(
        lambda k: tuple(getattr(k, n) for n in names),
        kernel,
        tuple(jnp.asarray(v) for v in updates.values()),
        is_leaf=lambda leaf: leaf is None,
    )
