"""Data handling: simulated noise, injections, and (optional) GWOSC strain.

Everything here is host-side preparation (numpy / gwpy); the output is a
``NetworkLikelihood`` whose jitted path consumes only static arrays. Injections reuse
the likelihood's own projection, so a zero-noise injection has lnL(true params) = 0
identically — the strongest cheap self-consistency check the pipeline has.
"""

import numpy as np

from .conditioning import rfft_freqs
from .detectors import DETECTORS, gmst_from_gps
from .likelihood import NetworkLikelihood
from .psd import aligo_zdhp_psd, welch_psd
from .waveform import WaveformModel


def simulate_noise_fd(rng: np.random.Generator, psd, duration: float):
    """Draw one-sided FD Gaussian noise with <|n(f)|^2> = S(f) T / 2 (inf-PSD bins -> 0)."""
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
    """Inject ``waveform(injection_params)`` into simulated noise; return the likelihood.

    ``noise_seed=None`` gives a zero-noise injection (lnL peaks at exactly 0).
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
