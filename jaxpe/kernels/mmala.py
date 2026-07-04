"""Simplified Riemannian-manifold MALA.

The Langevin proposal is preconditioned by a position-dependent metric G(x)
(e.g. a Fisher information matrix supplied by the problem):

    x' = x + eps^2/2 * G(x)^{-1} grad log p(x) + eps * G(x)^{-1/2} xi.

This is the "simplified" mMALA of Girolami & Calderhead (2011): the metric enters the
drift and the proposal covariance, but the Christoffel/curvature drift terms are
dropped. The MH correction below uses the *exact* proposal densities (with G evaluated
at x for the forward move and at x' for the reverse), so detailed balance is exact and
the invariant density is unaffected by the simplification.

With ``metric_fn=None`` a constant dense metric ``cov`` (the proposal covariance,
typically the ensemble posterior covariance) is used instead — i.e. dense-mass MALA;
the log-det and quadratic terms then cancel symmetrically.
"""

from collections.abc import Callable
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cholesky, solve_triangular

from .base import Kernel, KernelState, LogProbFn, mh_accept


class MMALA(Kernel):
    needs_gradient: ClassVar[bool] = True
    step_size: jax.Array
    metric_fn: Callable | None = eqx.field(static=True, default=None)  # x -> (n, n) PSD G(x)
    cov: jax.Array | None = None  # constant proposal covariance if metric_fn is None

    def __init__(self, step_size: float, metric_fn=None, cov=None):
        self.step_size = jnp.asarray(step_size)
        self.metric_fn = metric_fn
        self.cov = None if cov is None else jnp.asarray(cov)

    def _metric(self, x):
        """Return G(x); falls back to inv(cov) or identity."""
        if self.metric_fn is not None:
            return self.metric_fn(x)
        if self.cov is not None:
            return jnp.linalg.inv(self.cov)
        return jnp.eye(x.shape[0], dtype=x.dtype)

    def _log_q(self, x_to, mean, chol_G):
        """log N(x_to; mean, eps^2 G^{-1}) up to the dimension-independent constant."""
        r = x_to - mean
        # r^T G r = ||L^T r||^2 with G = L L^T; (r @ L)_j = (L^T r)_j
        quad = jnp.sum((r @ chol_G) ** 2) / self.step_size**2
        log_det_G = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_G)))
        return -0.5 * quad + 0.5 * log_det_G - x_to.shape[0] * jnp.log(self.step_size)

    def _propose_mean(self, x, grad, chol_G):
        sigma_grad = cho_solve((chol_G, True), grad)  # G^{-1} grad
        return x + 0.5 * self.step_size**2 * sigma_grad

    def step(self, key, state: KernelState, logp_fn: LogProbFn):
        key_prop, key_acc = jax.random.split(key)
        eps = self.step_size

        G = self._metric(state.x)
        chol_G = cholesky(G, lower=True)
        mean_fwd = self._propose_mean(state.x, state.grad, chol_G)
        xi = jax.random.normal(key_prop, state.x.shape, state.x.dtype)
        # G^{-1/2} xi with G = L L^T: solve L^T u = xi
        x_new = mean_fwd + eps * solve_triangular(chol_G.T, xi, lower=False)

        logp_new, grad_new = jax.value_and_grad(logp_fn)(x_new)
        G_rev = self._metric(x_new)
        chol_G_rev = cholesky(G_rev, lower=True)
        mean_rev = self._propose_mean(x_new, grad_new, chol_G_rev)

        log_q_fwd = self._log_q(x_new, mean_fwd, chol_G)
        log_q_rev = self._log_q(state.x, mean_rev, chol_G_rev)

        proposal = KernelState(x=x_new, log_prob=logp_new, grad=grad_new)
        log_ratio = logp_new - state.log_prob + log_q_rev - log_q_fwd
        return mh_accept(key_acc, state, proposal, log_ratio)
