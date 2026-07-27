"""Time-domain network likelihood.

Concrete :class:`~jaxpe.gw.likelihood.base.NetworkLikelihood` for time-domain
waveform models: the geocenter polarizations are evaluated on the TD grid, Tukey
windowed and FFT'd to the frequency domain. Everything else -- detector
projection, PSD banding, the Whittle sum -- is inherited from the base.
"""

from dataclasses import dataclass

from ..conditioning import td_to_fd
from .base import NetworkLikelihood


@dataclass(frozen=True)
class TDNetworkLikelihood(NetworkLikelihood):
    """Whittle likelihood for a network of detectors evaluated from a TD waveform."""

    def polarizations_fd(self, params: dict):
        st = self._static()
        hp, hc = self.waveform(params, st["times"])
        return td_to_fd(hp, st["dt"], st["window"]), td_to_fd(
            hc, st["dt"], st["window"]
        )
