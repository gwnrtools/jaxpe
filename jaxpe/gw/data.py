r"""Data handling: simulated noise, injections, and (optional) GWOSC strain.

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

import hashlib

import numpy as np

from .conditioning import analysis_grid, resolve_f_max
from .detectors import DETECTORS, gmst_from_gps
from .likelihood import FDNetworkLikelihood, TDNetworkLikelihood
from .psd import aligo_zdhp_psd, welch_psd
from .waveform import WaveformModel

# fold_in takes a uint32; blake2b gives a stable 4-byte digest, masked to stay in
# range. Deliberately NOT Python's hash(), which is salted per process by
# PYTHONHASHSEED and would make a "seeded" noise realisation irreproducible between
# runs of the same script.
_LABEL_MASK = (1 << 31) - 1


def _label_int(label: str) -> int:
    """Stable, process-independent integer for a string label."""
    return (
        int.from_bytes(hashlib.blake2b(label.encode(), digest_size=4).digest(), "big")
        & _LABEL_MASK
    )


def derive_noise_seed(seed: int, index: int) -> int:
    """Noise seed for injection ``index`` of a campaign seeded by ``seed``.

    Returned as an integer so it can be recorded in the injection artifact: the
    realisation then survives even if this derivation is ever changed. Keyed by
    *which* injection it is, not by draw order, so re-analysing injection 7 on its
    own reproduces what the full campaign gave it.
    """
    payload = f"{int(seed)}:{int(index)}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def noise_key(seed: int, detector: str = ""):
    """PRNG key for one detector's noise stream under ``seed``.

    Folding the detector *label* rather than its position is what makes the
    realisation independent of the order ``detector_names`` happens to be given in.
    """
    import jax

    key = jax.random.PRNGKey(int(seed) & _LABEL_MASK)
    if detector:
        key = jax.random.fold_in(key, _label_int(detector))
    return key


def simulate_noise_fd_jax(key, psd, duration: float):
    r"""Stationary Gaussian coloured noise in the frequency domain, in JAX.

    The JAX twin of :func:`simulate_noise_fd`, with the same convention
    $\sigma = \sqrt{S(f)\,T}/2$ and the same zeroing of non-finite bins (the PSD is
    ``inf`` out of band, so those bins carry exactly zero). Being traceable, it
    ``vmap``s over a batch of injections and runs on the accelerator alongside the
    signal, with no host round-trip.

    Reproducibility differs from the numpy version in one respect worth knowing.
    JAX's underlying random stream is bitwise identical across CPU and GPU, but the
    Gaussian transform is not: ``jax.random.normal`` was measured to differ by up to
    3e-15 between backends, because XLA compiles ``erf_inv`` differently for each.
    Realisations are therefore bitwise reproducible *on a given platform* and equal
    to ~1e-15 across platforms. Use :func:`simulate_noise_fd` where bitwise
    cross-platform equality is required.
    """
    import jax
    import jax.numpy as jnp

    psd = jnp.asarray(psd, dtype=float)
    sigma = jnp.sqrt(psd * duration) / 2.0
    sigma = jnp.where(jnp.isfinite(sigma), sigma, 0.0)
    kr, ki = jax.random.split(key)
    return sigma * (
        jax.random.normal(kr, psd.shape) + 1j * jax.random.normal(ki, psd.shape)
    )


def simulate_noise_fd(rng: np.random.Generator, psd, duration: float):
    r"""
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
    return sigma * (
        rng.standard_normal(psd.shape) + 1j * rng.standard_normal(psd.shape)
    )


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
) -> TDNetworkLikelihood | FDNetworkLikelihood:
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
    times, freqs = analysis_grid(t_c, duration, sampling_rate, post_trigger)
    f_max = resolve_f_max(f_max, sampling_rate)

    detectors = tuple(DETECTORS[name] for name in detector_names)
    psds = {name: np.asarray(psd_fn(freqs)) for name in detector_names}

    gmst_ref = gmst_from_gps(t_c)
    LikelihoodClass = (
        FDNetworkLikelihood
        if getattr(waveform, "is_fd", False)
        else TDNetworkLikelihood
    )
    like = LikelihoodClass(
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

    # One independent stream per detector, keyed by the detector's *name*. A single
    # generator advanced through this loop -- which is what this used to do -- makes
    # every detector's realisation depend on the order detector_names was given in.
    data_fd = {}
    for name in detector_names:
        d = np.asarray(signal_fd[name])
        if noise_seed is not None:
            noise = simulate_noise_fd_jax(
                noise_key(noise_seed, name), psds[name], duration
            )
            # Back to numpy at the boundary: data_fd is a numpy field of a frozen
            # dataclass whose jnp constants are materialized once, eagerly, in
            # _static(). Storing a device array here would put a live buffer in a
            # place the cache contract assumes is host memory.
            d = d + np.asarray(noise)
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
) -> TDNetworkLikelihood | FDNetworkLikelihood:
    """Build a likelihood from real strain around ``trigger_time``.

    ``strain`` maps detector name -> downloaded strain array whose first sample is at
    GPS time ``strain_start`` and which covers the analysis segment
    [trigger_time + post_trigger - duration, trigger_time + post_trigger]. PSDs are
    Welch-estimated from ``psd_strain`` (e.g. minutes of off-source data), defaulting
    to the full ``strain`` arrays themselves.
    """
    times, freqs = analysis_grid(trigger_time, duration, sampling_rate, post_trigger)
    n = times.size
    dt = 1.0 / sampling_rate
    i0 = int(round((times[0] - strain_start) * sampling_rate))
    f_max = resolve_f_max(f_max, sampling_rate)

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

    LikelihoodClass = (
        FDNetworkLikelihood
        if getattr(waveform, "is_fd", False)
        else TDNetworkLikelihood
    )
    return LikelihoodClass(
        waveform=waveform,
        detectors=tuple(DETECTORS[name] for name in strain),
        data_fd=data_fd,
        psds=psds,
        freqs=freqs,
        times=times,
        f_min=f_min,
        f_max=f_max,
        gmst_ref=gmst_from_gps(trigger_time),
        t_ref=trigger_time,
        tukey_alpha=tukey_alpha,
    )
