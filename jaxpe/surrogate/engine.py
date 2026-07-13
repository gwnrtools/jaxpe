"""Surrogate-engine interface for expensive, non-differentiable likelihoods.

The GPry-fusion design (``docs/gpry_fusion_design.md``, D4) mandates a *thin* seam
between jaxpe and any active-learning surrogate backend: exactly the operations the
pipeline calls, nothing speculative. GPry is the first (and so far only) backend
(:class:`~jaxpe.surrogate.gpry_backend.GPryEngine`); a component would only ever be
replaced after the Phase-1 profiling checkpoint shows it dominating wall-clock.

Everything here is host-side numpy by construction: the expensive likelihood is an
opaque Python callable that must never enter a JAX trace.
"""

from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np


class SurrogateSamples(NamedTuple):
    """MC samples drawn from a surrogate posterior.

    Attributes
    ----------
    x
        (n, d) sample positions, columns ordered as ``names``.
    weights
        (n,) non-negative sample weights (all ones for unweighted chains).
    logpost
        (n,) surrogate log-posterior at the samples (the proposal density that
        importance-sampling reweighting against the true likelihood divides by).
    names
        Parameter names, one per column of ``x``.
    """

    x: np.ndarray
    weights: np.ndarray
    logpost: np.ndarray
    names: tuple


@runtime_checkable
class SurrogateEngine(Protocol):
    """The four operations the jaxpe pipeline needs from a surrogate backend."""

    def run(self) -> dict:
        """Drive the active-learning loop to convergence; return diagnostics."""

    def surrogate_logp(self, x: np.ndarray) -> np.ndarray:
        """Surrogate log-posterior at (n, d) points (for IS reweighting)."""

    def sample(self) -> SurrogateSamples:
        """Draw MC samples from the current surrogate posterior."""

    def diagnostics(self) -> dict:
        """Current state: truth-evaluation count, convergence flag, etc."""
