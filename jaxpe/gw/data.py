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

    like = _empty_likelihood(
        waveform,
        detector_names,
        times,
        freqs,
        f_min,
        f_max,
        psd_fn,
        t_c,
        tukey_alpha,
    )
    psds = like.psds

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


# Refuse a batch larger than this many bytes of strain rather than letting XLA
# fail somewhere unhelpful. Deliberately conservative: the signal, the noise and the
# padded FFT working set all coexist during the vmap.
MAX_BATCH_BYTES = 2 << 30  # 2 GiB


def make_injections(
    waveform: WaveformModel,
    params_batch: dict,
    *,
    key,
    detector_names=("H1", "L1"),
    duration: float = 8.0,
    sampling_rate: float = 2048.0,
    f_min: float = 20.0,
    f_max: float | None = None,
    psd_fn=aligo_zdhp_psd,
    post_trigger: float = 2.0,
    tukey_alpha: float = 0.1,
    add_noise: bool = True,
    max_bytes: int = MAX_BATCH_BYTES,
) -> dict:
    """Strain for a batch of injections sharing one analysis grid.

    Builds all ``n`` injections in a single ``vmap``, keeping signal generation,
    projection and noise on the accelerator with no per-injection host round-trip.

    Two different speedups, both measured on this repo's 4 s @ 1024 Hz H1+L1
    configuration, and worth separating because they have different causes:

    * **~4x CPU / ~6x GPU** for the vectorised arithmetic alone, comparing a vmapped
      call against a serial loop over one already-jitted function (n = 256).
    * **8x at n = 16 and 28x at n = 64 on GPU** against a loop over
      :func:`make_injection` -- 173.5 s to 6.2 s at n = 64. Most of that margin is
      not parallelism: ``make_injection`` builds a fresh ``jax.jit`` on every call,
      so a serial loop pays one compilation per injection (~2.7 s each here) while
      the batch pays one for the whole set.

    The second figure is what a caller actually experiences, so it is the one to plan
    around; the first is what the hardware is doing.

    Use this when a large suite of injections is the product -- a training set, a
    waveform bank, a systematics study. It is *not* a way to speed up a validation
    campaign: injection creation is a fraction of a percent of a PP run's wall time,
    which is dominated by the parameter estimation that follows.

    Agreement with :func:`make_injection` is exact to within XLA's fusion of the
    waveform: the PSD-weighted mismatch is ~1e-9, though raw amplitudes can differ at
    the 1e-5 level because the two jit graphs fuse ``IMRPhenomD`` differently. Compare
    injections by mismatch, not by elementwise amplitude.

    Parameters
    ----------
    params_batch : dict
        ``{name: (n,) array}``. Every parameter must carry the same leading axis;
        ``geocent_time`` is the exception and must be a scalar, since it fixes the
        analysis grid, which vmap requires to be identical across the batch.
    key : jax.Array
        PRNG key, split internally per injection and per detector. Unused when
        ``add_noise`` is False.
    add_noise : bool, default True
        False gives the pure projected signal (the batched zero-noise injection).
    max_bytes : int
        Refuse batches whose strain would exceed this. A 2048 s BNS segment is 67 MB
        per injection per detector, so batching that configuration is infeasible on a
        small card and should fail with an estimate rather than inside XLA.

    Returns
    -------
    dict
        ``{detector: (n, n_freq) complex}``. Arrays, not ``NetworkLikelihood``
        objects: building ``n`` frozen dataclasses, each with its own eagerly
        materialized constant cache, would give back everything the batching won.
    """
    import jax
    import jax.numpy as jnp

    names = list(params_batch)
    if "geocent_time" not in params_batch:
        raise ValueError("params_batch must include geocent_time")
    t_c = np.asarray(params_batch["geocent_time"])
    if t_c.ndim != 0 and np.unique(t_c).size != 1:
        raise ValueError(
            "make_injections requires one analysis grid for the whole batch, so "
            "geocent_time must be a scalar (or identical across the batch); got "
            f"{np.unique(t_c).size} distinct values. Vary the trigger time with "
            "separate make_injection calls."
        )
    t_c = float(t_c.reshape(-1)[0])

    batched = {k: np.atleast_1d(np.asarray(v)) for k, v in params_batch.items()}
    sizes = {k: v.shape[0] for k, v in batched.items() if k != "geocent_time"}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"every parameter must share one leading axis; got {sizes}")
    n_inj = next(iter(sizes.values()))

    times, freqs = analysis_grid(t_c, duration, sampling_rate, post_trigger)
    f_max = resolve_f_max(f_max, sampling_rate)

    n_bytes = n_inj * freqs.size * 16 * len(detector_names)
    if n_bytes > max_bytes:
        raise ValueError(
            f"batch would need {n_bytes / 2**30:.2f} GiB of strain "
            f"({n_inj} injections x {freqs.size} bins x {len(detector_names)} "
            f"detectors, complex128), above the {max_bytes / 2**30:.2f} GiB limit. "
            "Reduce the batch, shorten the segment, or use make_injection in a loop "
            "-- long segments (a 2048 s BNS is 67 MB per injection per detector) "
            "cannot be batched meaningfully."
        )

    # One template likelihood carries the grid, PSDs and projection; vmapping its
    # detector_strains_fd reuses the whole conditioned path without rebuilding it per
    # injection. Zero data: this object is only ever asked for signal.
    template = _empty_likelihood(
        waveform,
        detector_names,
        times,
        freqs,
        f_min,
        f_max,
        psd_fn,
        t_c,
        tukey_alpha,
    )

    order = [k for k in names if k != "geocent_time"]
    stacked = jnp.stack([jnp.asarray(batched[k]) for k in order], axis=-1)

    def one(row):
        p = {k: row[i] for i, k in enumerate(order)}
        p["geocent_time"] = jnp.asarray(t_c)
        return template.detector_strains_fd(p)

    signal = jax.jit(jax.vmap(one))(stacked)
    if not add_noise:
        return {name: signal[name] for name in detector_names}

    psds = {name: jnp.asarray(template.psds[name]) for name in detector_names}
    keys = jax.random.split(key, n_inj)

    def noise_for(k, name):
        det_key = jax.random.fold_in(k, _label_int(name))
        return simulate_noise_fd_jax(det_key, psds[name], duration)

    out = {}
    for name in detector_names:
        noise = jax.jit(jax.vmap(lambda k, n=name: noise_for(k, n)))(keys)
        out[name] = signal[name] + noise
    return out


def _empty_likelihood(
    waveform, detector_names, times, freqs, f_min, f_max, psd_fn, t_c, tukey_alpha
):
    """A likelihood carrying the conditioning but no data, for signal generation."""
    LikelihoodClass = (
        FDNetworkLikelihood
        if getattr(waveform, "is_fd", False)
        else TDNetworkLikelihood
    )
    return LikelihoodClass(
        waveform=waveform,
        detectors=tuple(DETECTORS[name] for name in detector_names),
        data_fd={name: np.zeros(len(freqs), complex) for name in detector_names},
        psds={name: np.asarray(psd_fn(freqs)) for name in detector_names},
        freqs=freqs,
        times=times,
        f_min=f_min,
        f_max=f_max,
        gmst_ref=gmst_from_gps(t_c),
        t_ref=t_c,
        tukey_alpha=tukey_alpha,
    )


def network_snr(like, params: dict) -> float:
    """Quadrature sum of the per-detector optimal SNRs at ``params``."""
    snrs = like.optimal_snr(params)
    return float(np.sqrt(sum(float(v) ** 2 for v in snrs.values())))


def distance_for_target_snr(like, params: dict, target_snr: float) -> float:
    r"""Distance at which ``params`` would have network optimal SNR ``target_snr``.

    ``luminosity_distance`` enters the detector response only as an overall $1/D$
    amplitude, so $\rho = \sqrt{\langle h|h\rangle} \propto 1/D$ and the target is hit
    in a single measurement:

    $$ D_{\rm new} = D_{\rm old} \, \rho_{\rm old} / \rho_{\rm target}. $$

    No iteration and no search. ``like`` must be a zero-noise injection: optimal SNR
    is a property of the template, and solving against noise-contaminated data would
    not give the distance the caller asked for.
    """
    return (
        float(params["luminosity_distance"])
        * network_snr(like, params)
        / float(target_snr)
    )


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
