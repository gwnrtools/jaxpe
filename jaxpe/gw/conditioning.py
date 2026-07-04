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


def td_to_fd(h_td, dt: float, window=None):
    """FFT a (possibly windowed) time series to the continuum-normalized FD."""
    if window is not None:
        h_td = h_td * window
    return jnp.fft.rfft(h_td) * dt


def time_shift(h_fd, freqs, delta_t):
    """Apply h(f) -> h(f) exp(-2 pi i f delta_t), i.e. delay the signal by delta_t."""
    return h_fd * jnp.exp(-2j * jnp.pi * freqs * delta_t)
