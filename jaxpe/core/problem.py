"""The problem interface consumed by the sampling engine.

An ``InferenceProblem`` couples a ``JointPrior`` with a log-likelihood over named
physical parameters, and exposes the unconstrained-space log-posterior
``log_posterior(y)`` (scalar in, scalar out, differentiable) that every kernel and the
flow sampler operate on. Non-finite likelihoods are mapped to -inf so that a bad
waveform evaluation is rejected rather than propagated.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .priors import JointPrior


@dataclass(frozen=True)
class InferenceProblem:
    """A prior plus a log-likelihood over named physical parameters.

    Parameters
    ----------
    prior
        Joint prior defining parameter names, order, and support.
    log_likelihood
        Maps ``{name: scalar}`` to a scalar log-likelihood. Must be JAX-traceable;
        differentiability is required only for gradient-based kernels.
    """

    prior: JointPrior
    log_likelihood: Callable[[dict], jnp.ndarray]

    @property
    def n_dim(self) -> int:
        return self.prior.n_dim

    @property
    def names(self) -> tuple[str, ...]:
        return self.prior.names

    def log_likelihood_vec(self, x):
        """Log-likelihood of a (n_dim,) physical vector."""
        return self.log_likelihood(self.prior.as_dict(x))

    def log_posterior(self, y):
        """Unnormalized log-posterior density in unconstrained coordinates, shape (n_dim,) -> scalar."""
        x = self.prior.to_physical(y)
        log_p = self.log_likelihood_vec(x) + self.prior.log_prob(x) + self.prior.log_det(y)
        return jnp.where(jnp.isfinite(log_p), log_p, -jnp.inf)

    def sample_unconstrained(self, key, n: int):
        """Draw (n, n_dim) prior samples mapped to unconstrained space (chain initialization)."""
        x = self.prior.sample(key, n)
        return jax.vmap(self.prior.to_unconstrained)(x)
