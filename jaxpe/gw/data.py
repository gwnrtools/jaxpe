"""Data handling: simulated noise, injections, and (optional) GWOSC strain.

Gravitational Wave Data Analysis fundamentally relies on comparing theoretical waveforms
to observed strain data, which contains both a potential signal and instrumental noise.

Motivation & Math
-----------------
The noise in a GW detector is generally modeled as stationary, Gaussian colored noise.
This means the noise $n(t)$ in the time domain is entirely characterized by its
Power Spectral Density (PSD) $S_n(f)$ in the frequency domain.

For a finite duration $T$, the Fourier transform of the noise $\tilde{n}(f)$ satisfies:
$$ \langle \tilde{n}(f) \tilde{n}^*(f') \rangle = \frac{1}{2} S_n(f) \delta(f - f') $$
In discrete frequency bins of width $\Delta f = 1/T$, the variance of the complex noise
is $\sigma^2 = S_n(f) / (4 \Delta f)$.

This module provides tools to:
1. Simulate this noise in the frequency domain.
2. Inject a known signal ("injection") into simulated noise to test PE pipelines.
3. Fetch real open data from GWOSC and construct likelihoods.
"""

import numpy as np

from .conditioning import rfft_freqs
from .detectors import DETECTORS, gmst_from_gps
from .likelihood import NetworkLikelihood
from .psd import aligo_zdhp_psd, welch_psd
from .waveform import WaveformModel


def simulate_noise_fd(rng: np.random.Generator, psd, duration: float):
    """
    Simulate stationary Gaussian colored noise in the frequency domain.

    Motivation & Math
    -----------------
    Since the noise is stationary and Gaussian, its Fourier coefficients are independent
    Gaussian random variables. For a one-sided PSD $S(f)$, the real and imaginary parts
    of the noise at frequency $f$ are drawn from:
    $$ \tilde{n}(f) = \sigma (\mathcal{N}(0, 1) + i \mathcal{N}(0, 1)) $$
    where the standard deviation $\sigma = \sqrt{\frac{S(f) \times \text{duration}}{4}}$.

    Parameters
    ----------
    rng : np.random.Generator
        A numpy random number generator instance.
    psd : np.ndarray
        The one-sided Power Spectral Density evaluated at the frequency bins.
    duration : float
        The duration $T$ of the segment in seconds.

    Returns
    -------
    np.ndarray
        The complex frequency-domain noise.
    """
    psd = np.asarray(psd, float)
    sigma = np.sqrt(psd * duration) / 2.0
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    return sigma * (rng.standard_normal(psd.shape) + 1j * rng.standard_normal(psd.shape))


def make_injection(
    waveform: WaveformModel,
    injection_params: dict,
    detector_names=("H1", "L1"),
    duration: float = 8.0,
    sampling_rate: float = 2048.0,
    f_min: float = 20.0,
    f_max: float | None = None,
    psd_fn=aligo_zdhp_psd,
    noise_seed: int | None = None,
    post_trigger: float = 2.0,
    tukey_alpha: float = 0.1,
) -> NetworkLikelihood:
    """
    Inject a simulated gravitational wave signal into simulated noise.

    "Injections" are software-simulated signals used to test and calibrate Parameter
    Estimation pipelines. We generate a clean waveform using `injection_params`,
    project it onto the requested detectors, and add simulated Gaussian noise.

    If `noise_seed=None`, no noise is added (a "zero-noise" injection). In a zero-noise
    injection, the likelihood perfectly peaks at exactly 0.0 at the true parameters.

    Parameters
    ----------
    waveform : WaveformModel
        The waveform model to generate the signal.
    injection_params : dict
        A dictionary containing the true parameters of the injected signal.
    detector_names : tuple, default=("H1", "L1")
        The network of detectors (e.g., LIGO Hanford, LIGO Livingston).
    duration : float, default=8.0
        Duration of the data segment in seconds.
    sampling_rate : float, default=2048.0
        Sampling rate in Hz.
    f_min : float, default=20.0
        Lower frequency cutoff for the likelihood integration.
    f_max : float | None, default=None
        Upper frequency cutoff. If None, defaults to the Nyquist frequency (0.9 * sampling_rate / 2).
    psd_fn : Callable, default=aligo_zdhp_psd
        A function mapping frequencies to PSD values.
    noise_seed : int | None, default=None
        Seed for the noise realization. If None, zero noise is added.
    post_trigger : float, default=2.0
        How many seconds of data to keep after the trigger time.
    tukey_alpha : float, default=0.1
        The shape parameter for the Tukey window used to taper the time-domain signal.

    Returns
    -------
    NetworkLikelihood
        A constructed likelihood object holding the injection data.
    """
    import jax
    import jax.numpy as jnp

    t_c = float(injection_params["geocent_time"])
    n = int(duration * sampling_rate)
    dt = 1.0 / sampling_rate
    t_start = t_c + post_trigger - duration
    times = t_start + np.arange(n) * dt
    freqs = rfft_freqs(n, dt)
    f_max = f_max if f_max is not None else 0.9 * (sampling_rate / 2.0)

    detectors = tuple(DETECTORS[name] for name in detector_names)
    psds = {name: np.asarray(psd_fn(freqs)) for name in detector_names}

    gmst_ref = gmst_from_gps(t_c)
    like = NetworkLikelihood(
        waveform=waveform,
        detectors=detectors,
        data_fd={name: np.zeros(len(freqs), complex) for name in detector_names},
        psds=psds,
        freqs=freqs,
        times=times,
        f_min=f_min,
        f_max=f_max,
        gmst_ref=gmst_ref,
        t_ref=t_c,
        tukey_alpha=tukey_alpha,
    )

    # signal via the likelihood's own projection machinery
    params_j = {k: jnp.asarray(v) for k, v in injection_params.items()}
    signal_fd = jax.jit(like.detector_strains_fd)(params_j)

    rng = None if noise_seed is None else np.random.default_rng(noise_seed)
    data_fd = {}
    for name in detector_names:
        d = np.asarray(signal_fd[name])
        if rng is not None:
            d = d + simulate_noise_fd(rng, psds[name], duration)
        data_fd[name] = d
    like.data_fd.update(data_fd)
    like._cache.clear()
    like._static()  # eager rebuild with the injected data (never inside a trace)
    return like


def fetch_open_strain(detector: str, gps_start: float, gps_end: float):
    """Download open strain via gwpy (requires the ``jaxpe[gwdata]`` extra).

    Returns (strain array, sampling_rate).
    """
    try:
        from gwpy.timeseries import TimeSeries
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install gwpy/gwosc: pip install 'jaxpe[gwdata]'") from exc
    ts = TimeSeries.fetch_open_data(detector, gps_start, gps_end, cache=True)
    return np.asarray(ts.value), float(ts.sample_rate.value)


def likelihood_from_strain(
    waveform: WaveformModel,
    strain: dict,
    strain_start: float,
    sampling_rate: float,
    trigger_time: float,
    duration: float = 8.0,
    psd_strain: dict | None = None,
    f_min: float = 20.0,
    f_max: float | None = None,
    post_trigger: float = 2.0,
    tukey_alpha: float = 0.1,
) -> NetworkLikelihood:
    """Build a likelihood from real strain around ``trigger_time``.

    ``strain`` maps detector name -> downloaded strain array whose first sample is at
    GPS time ``strain_start`` and which covers the analysis segment
    [trigger_time + post_trigger - duration, trigger_time + post_trigger]. PSDs are
    Welch-estimated from ``psd_strain`` (e.g. minutes of off-source data), defaulting
    to the full ``strain`` arrays themselves.
    """
    n = int(duration * sampling_rate)
    dt = 1.0 / sampling_rate
    t_start = trigger_time + post_trigger - duration
    i0 = int(round((t_start - strain_start) * sampling_rate))
    freqs = rfft_freqs(n, dt)
    f_max = f_max if f_max is not None else 0.9 * (sampling_rate / 2.0)

    from .conditioning import tukey_window

    window = tukey_window(n, tukey_alpha)
    data_fd, psds = {}, {}
    for name, s in strain.items():
        s = np.asarray(s, float)
        if i0 < 0 or i0 + n > len(s):
            raise ValueError(f"{name}: strain does not cover the analysis segment")
        seg = s[i0 : i0 + n]
        data_fd[name] = np.fft.rfft(seg * window) * dt
        src = (psd_strain or strain)[name]
        psds[name] = welch_psd(src, sampling_rate, seg_duration=duration, freqs=freqs)

    return NetworkLikelihood(
        waveform=waveform,
        detectors=tuple(DETECTORS[name] for name in strain),
        data_fd=data_fd,
        psds=psds,
        freqs=freqs,
        times=t_start + np.arange(n) * dt,
        f_min=f_min,
        f_max=f_max,
        gmst_ref=gmst_from_gps(trigger_time),
        t_ref=trigger_time,
        tukey_alpha=tukey_alpha,
    )
