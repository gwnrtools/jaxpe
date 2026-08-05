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


def _no_likelihood(params):
    """Placeholder for a problem built only to carry the prior's coordinate maps."""
    raise TypeError(
        "this InferenceProblem was built prior-only (for coordinate transforms); "
        "it has no likelihood to evaluate. Construct it with a log_likelihood, or "
        "use `prior` directly if you only need to_physical/to_unconstrained."
    )


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

        Defaults to a stub that raises. Post-processing only ever needs the prior's
        bijections (``PostProcessor`` touches ``problem.prior`` and nothing else), and
        forcing those callers to build a likelihood cost them a full waveform
        generation and FFT per samples file. Omitting it fails loudly at use rather
        than silently returning a wrong density.
    """

    prior: JointPrior
    log_likelihood: Callable[[dict], jnp.ndarray] = _no_likelihood

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
        log_p = (
            self.log_likelihood_vec(x) + self.prior.log_prob(x) + self.prior.log_det(y)
        )
        return jnp.where(jnp.isfinite(log_p), log_p, -jnp.inf)

    def sample_unconstrained(self, key, n: int):
        """Draw (n, n_dim) prior samples mapped to unconstrained space (chain initialization)."""
        x = self.prior.sample(key, n)
        return jax.vmap(self.prior.to_unconstrained)(x)
