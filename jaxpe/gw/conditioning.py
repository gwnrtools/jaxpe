"""Time-domain -> frequency-domain conditioning, jittable and differentiable.

Conventions (match the continuum limit of the discrete Fourier transform):

    h(f_k) = dt * sum_j h(t_j) exp(-2 pi i f_k t_j)   (jnp.fft.rfft * dt)
    one-sided PSD S(f) with  <|n(f)|^2> = S(f) T / 2.
"""

import jax.numpy as jnp
import numpy as np


def tukey_window(n: int, alpha: float = 0.1) -> np.ndarray:
    """Tukey (tapered cosine) window; host-side constant, applied inside the jitted path."""
    if alpha <= 0:
        return np.ones(n)
    if alpha >= 1:
        alpha = 1.0
    w = np.ones(n)
    edge = int(np.floor(alpha * (n - 1) / 2.0)) + 1
    t = np.arange(edge)
    ramp = 0.5 * (1 + np.cos(np.pi * (2.0 * t / (alpha * (n - 1)) - 1)))
    w[:edge] = ramp
    w[-edge:] = ramp[::-1]
    return w


def rfft_freqs(n: int, dt: float) -> np.ndarray:
    return np.fft.rfftfreq(n, dt)


def analysis_grid(
    trigger_time: float,
    duration: float,
    sampling_rate: float,
    post_trigger: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Geocentric sample times and rfft frequencies of one analysis segment.

    The segment ends ``post_trigger`` seconds after ``trigger_time`` and runs for
    ``duration``, so ``times[0] = trigger_time + post_trigger - duration``.

    This is the single definition of the grid that ``make_injection`` and
    ``likelihood_from_strain`` build on. Callers that need to evaluate something on
    exactly that grid -- external mode models, relative-binning fiducials, tests
    checking a likelihood's ``times`` -- should call this rather than re-deriving it,
    which is what several of them used to do.

    Returns
    -------
    (times, freqs)
        ``times`` has ``n = int(duration * sampling_rate)`` samples; ``freqs`` is the
        corresponding one-sided rfft grid of ``n // 2 + 1`` bins.
    """
    n = int(duration * sampling_rate)
    dt = 1.0 / sampling_rate
    t_start = trigger_time + post_trigger - duration
    return t_start + np.arange(n) * dt, rfft_freqs(n, dt)


def resolve_f_max(f_max: float | None, sampling_rate: float) -> float:
    """Upper analysis frequency, defaulting to 90% of Nyquist.

    The margin keeps the band clear of the anti-alias roll-off at the very top of the
    rfft grid. Defined once so the factor cannot drift between the two constructors.
    """
    return f_max if f_max is not None else 0.9 * (sampling_rate / 2.0)


def td_to_fd(h_td, dt: float, window=None):
    """FFT a (possibly windowed) time series to the continuum-normalized FD."""
    if window is not None:
        h_td = h_td * window
    return jnp.fft.rfft(h_td) * dt


def time_shift(h_fd, freqs, delta_t):
    """Apply h(f) -> h(f) exp(-2 pi i f delta_t), i.e. delay the signal by delta_t."""
    return h_fd * jnp.exp(-2j * jnp.pi * freqs * delta_t)
