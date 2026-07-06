"""Frequency-Domain Network Likelihood.

This module computes the likelihood of observing the gravitational-wave strain data
given a set of proposed waveform parameters.

Motivation & Math
-----------------
Assuming stationary Gaussian noise, the probability of observing noise $\tilde{n}(f)$ is:
$$ p(n) \propto \exp\left( -2 \int_{f_{\min}}^{f_{\max}} \frac{|\tilde{n}(f)|^2}{S_n(f)} df \right) $$

In a GW detection, the data $d$ is the sum of the true signal $h$ and noise $n$: $d = h + n$.
Therefore, $n = d - h$. The log-likelihood of observing data $d$ given parameters $\theta$ is:
$$ \ln \mathcal{L}(d | \theta) = -\frac{1}{2} (d - h(\theta) | d - h(\theta)) $$
where the inner product is defined as:
$$ (A | B) = 4 \Re \int_{f_{\min}}^{f_{\max}} \frac{\tilde{A}(f) \tilde{B}^*(f)}{S_n(f)} df $$

Expanding the inner product gives:
$$ \ln \mathcal{L} = (d | h(\theta)) - \frac{1}{2}(h(\theta) | h(\theta)) - \frac{1}{2}(d | d) $$
The term $-\frac{1}{2}(d | d)$ is independent of $\theta$ and cancels out in MCMC acceptance ratios.
JAXPE computes the full $(d-h | d-h)$ form up to the constant term.

Implementation Details
----------------------
The jitted path evaluates the waveform, applies a time-domain Tukey window, takes the FFT,
projects it onto the detectors (antenna patterns and time delays), and computes the
Whittle log-likelihood sum over frequency bins.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from ..core.problem import InferenceProblem
from ..core.priors import JointPrior
from .conditioning import td_to_fd, time_shift, tukey_window
from .detectors import EARTH_OMEGA, Detector, antenna_pattern, time_delay_from_geocenter
from .waveform import WaveformModel


def project_to_detector(det: Detector, hp_fd, hc_fd, freqs, ra, dec, psi, gmst):
    """Detector-frame FD strain from geocenter FD polarizations."""
    f_plus, f_cross = antenna_pattern(det, ra, dec, psi, gmst)
    delay = time_delay_from_geocenter(det, ra, dec, gmst)
    return time_shift(f_plus * hp_fd + f_cross * hc_fd, freqs, delay)


@dataclass(frozen=True)
class NetworkLikelihood:
    """Whittle likelihood for a network of detectors sharing one time/frequency grid.

    Parameters
    ----------
    waveform
        ``(params, times) -> (h_plus, h_cross)`` at the geocenter.
    detectors, data_fd, psds
        Per-detector geometry, FD data (continuum convention) and one-sided PSDs
        on ``freqs``; PSDs may be np.inf outside the analysis band.
    times
        Geocentric GPS times of the TD grid the waveform is evaluated on.
    gmst_ref, t_ref
        GMST is linearized as gmst_ref + EARTH_OMEGA (t_c - t_ref); exact to
        microradians over sub-second coalescence-time priors.
    """

    waveform: WaveformModel
    detectors: tuple[Detector, ...]
    data_fd: dict  # name -> complex (n_f,)
    psds: dict  # name -> (n_f,)
    freqs: np.ndarray
    times: np.ndarray
    f_min: float
    f_max: float
    gmst_ref: float
    t_ref: float
    tukey_alpha: float = 0.1
    accumulate_f64: bool = True
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # build the constant cache eagerly, OUTSIDE any jit trace: constants created
        # during tracing are tracers, and caching those leaks them into later traces
        self._static()

    def _static(self):
        """Precomputed jnp constants (built once, reused across traces)."""
        if not self._cache:
            dt = float(self.times[1] - self.times[0])
            df = float(self.freqs[1] - self.freqs[0])
            band = (self.freqs >= self.f_min) & (self.freqs <= self.f_max)
            self._cache.update(
                dt=dt,
                df=df,
                window=jnp.asarray(tukey_window(len(self.times), self.tukey_alpha)),
                freqs=jnp.asarray(self.freqs),
                times=jnp.asarray(self.times),
                inv_psd_banded={
                    name: jnp.asarray(np.where(band, 1.0 / np.asarray(psd), 0.0))
                    for name, psd in self.psds.items()
                },
                data={name: jnp.asarray(d) for name, d in self.data_fd.items()},
            )
        return self._cache

    def polarizations_fd(self, params: dict):
        st = self._static()
        hp, hc = self.waveform(params, st["times"])
        return td_to_fd(hp, st["dt"], st["window"]), td_to_fd(hc, st["dt"], st["window"])

    def _gmst(self, params):
        return self.gmst_ref + EARTH_OMEGA * (params["geocent_time"] - self.t_ref)

    def detector_strains_fd(self, params: dict):
        st = self._static()
        hp_fd, hc_fd = self.polarizations_fd(params)
        gmst = self._gmst(params)
        return {
            det.name: project_to_detector(
                det,
                hp_fd,
                hc_fd,
                st["freqs"],
                params["ra"],
                params["dec"],
                params["psi"],
                gmst,
            )
            for det in self.detectors
        }

    def _accumulate(self, x):
        if self.accumulate_f64:
            # canonicalize: float64 when x64 is enabled, harmless no-op otherwise
            x = x.astype(jax.dtypes.canonicalize_dtype(jnp.float64))
        return jnp.sum(x)

    def log_likelihood(self, params: dict):
        """Whittle lnL up to the <d|d> constant; -inf-safe via InferenceProblem."""
        st = self._static()
        strains = self.detector_strains_fd(params)
        lnl = 0.0
        for det in self.detectors:
            r = st["data"][det.name] - strains[det.name]
            integrand = (r.real**2 + r.imag**2) * st["inv_psd_banded"][det.name]
            lnl = lnl - 2.0 * st["df"] * self._accumulate(integrand)
        return lnl

    __call__ = log_likelihood

    def optimal_snr(self, params: dict) -> dict:
        """Per-detector optimal SNR sqrt(<h|h>) of the template at ``params``."""
        st = self._static()
        strains = self.detector_strains_fd(params)
        out = {}
        for det in self.detectors:
            h = strains[det.name]
            hh = 4.0 * st["df"] * jnp.sum((h.real**2 + h.imag**2) * st["inv_psd_banded"][det.name])
            out[det.name] = float(jnp.sqrt(hh))
        return out

    def problem(self, prior: JointPrior) -> InferenceProblem:
        """Bundle with a prior into the engine's InferenceProblem."""
        return InferenceProblem(prior=prior, log_likelihood=self.log_likelihood)
