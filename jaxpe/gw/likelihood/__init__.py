"""Likelihoods for gravitational-wave parameter estimation.

Two abstract roots (``base.py``):

* :class:`NetworkLikelihood` -- the Whittle likelihood of a detector network
  (``params dict -> JAX scalar``, differentiable). Concrete implementations:
  :class:`TDNetworkLikelihood` (time-domain waveform), :class:`FDNetworkLikelihood`
  (frequency-domain waveform), :class:`ModesNetworkLikelihood` (from precomputed
  spherical-harmonic modes, with extrinsic marginalization).
* :class:`IntrinsicLikelihood` -- the GPry-facing scalar likelihood over an
  intrinsic-parameter vector (``x -> float``, host-side, not differentiable).
  Concrete implementations: :class:`PhaseDistanceMarginalLikelihood` (closed-form
  phase+distance marginal for dominant-mode FD models) and
  :class:`MarginalizedIntrinsicLikelihood` (full extrinsic marginal via adaptive IS).
"""

from .base import IntrinsicLikelihood, NetworkLikelihood, project_to_detector
from .fd import FDNetworkLikelihood
from .fd_marginal import PhaseDistanceMarginalLikelihood
from .importance_sampling import BalanceHeuristicAccumulator
from .marginalized_intrinsic import (
    LowEffectiveSampleSizeError,
    MarginalizedIntrinsicLikelihood,
)
from .modes import ModesNetworkLikelihood
from .relative_binning_fd import RelativeBinningFDLikelihood, frequency_bin_edges
from .td import TDNetworkLikelihood

__all__ = [
    "NetworkLikelihood",
    "IntrinsicLikelihood",
    "project_to_detector",
    "TDNetworkLikelihood",
    "FDNetworkLikelihood",
    "ModesNetworkLikelihood",
    "RelativeBinningFDLikelihood",
    "frequency_bin_edges",
    "BalanceHeuristicAccumulator",
    "MarginalizedIntrinsicLikelihood",
    "LowEffectiveSampleSizeError",
    "PhaseDistanceMarginalLikelihood",
]
