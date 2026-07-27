"""Frequency-domain network likelihood.

Concrete :class:`~jaxpe.gw.likelihood.base.NetworkLikelihood` for frequency-domain
waveform models: the model returns the geocenter polarizations directly on
``self.freqs``, bypassing the time-domain to frequency-domain FFT of the
time-domain path.
"""

from dataclasses import dataclass

from .base import NetworkLikelihood


@dataclass(frozen=True)
class FDNetworkLikelihood(NetworkLikelihood):
    """Network likelihood for Frequency Domain waveform models."""

    def polarizations_fd(self, params: dict):
        st = self._static()
        hp_fd, hc_fd = self.waveform(params, st["freqs"])
        return hp_fd, hc_fd
