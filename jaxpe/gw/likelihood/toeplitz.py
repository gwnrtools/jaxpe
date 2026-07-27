r"""Symmetric Toeplitz covariance utilities for time-domain relative binning.

Phase RB-3 of ``docs/relative_binning_design.md``. Time-domain Gaussian inference uses
the *non-diagonal* noise covariance ``C_ij = rho(|i-j|)`` (``rho`` = noise
autocorrelation), a symmetric positive-definite Toeplitz matrix. Explicitly inverting
it is ``O(N^3)`` time / ``O(N^2)`` memory -- infeasible for long segments. This module
provides the two operations the heterodyned likelihood needs:

* ``toeplitz_matvec(col, v)``  -- ``C v``           in ``O(N log N)`` (FFT).
* ``inverse_matvec(x, v)``     -- ``C^{-1} v``       in ``O(N log N)`` via the
  **Gohberg-Semencul** representation (arXiv:2601.11239 App. B; classical result of
  Gohberg & Semencul), given the generator ``x = inverse_generator(col)`` obtained
  once by a Levinson-Durbin solve (``scipy.linalg.solve_toeplitz``, ``O(N^2)``, host).

Both matvecs are pure JAX (``jax.numpy.fft``), so they are jittable, differentiable,
and ``vmap``-able over the many right-hand sides (per-bin-masked fiducial modes) that
the summary-data precompute forms.

Gohberg-Semencul (symmetric case)
---------------------------------
For a symmetric positive-definite Toeplitz ``C`` with first column ``c``, let ``x``
solve ``C x = e_0`` (``x_0 > 0``). Then

    C^{-1} = (1 / x_0) [ L(x) L(x)^T - L(z) L(z)^T ],

where ``L(a)`` is the lower-triangular Toeplitz matrix with first column ``a`` and
``z = (0, x_{N-1}, x_{N-2}, ..., x_1)``. Each triangular-Toeplitz matvec is a linear
convolution/correlation evaluated by FFT, so ``C^{-1} v`` costs four FFT-convolutions.
The identity is verified against a dense solve in ``tests/test_toeplitz.py``.
"""

import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_toeplitz


def autocorrelation_from_psd(one_sided_psd, dt: float) -> np.ndarray:
    r"""Noise autocorrelation ``rho[m] = C_{i,i+m}`` from a one-sided PSD.

    ``one_sided_psd`` is sampled at the ``rfft`` frequencies of a length-``N`` segment
    (so its length is ``N // 2 + 1``); ``dt`` is the sampling interval. Returns the
    length-``N`` ACF whose ``m = 0`` entry is the noise variance
    ``rho[0] = sum_k S(f_k) df`` (``df = 1 / (N dt)``). Host-side (numpy).
    """
    S = np.asarray(one_sided_psd, dtype=float)
    n_freq = S.shape[0]
    n = 2 * (n_freq - 1)
    # rho(tau) = inverse FT of the two-sided PSD; irfft folds the one-sided spectrum,
    # and the 1/(2 dt) makes rho[0] = sum_k S_k df (the continuum variance).
    rho = np.fft.irfft(S, n=n) / (2.0 * dt)
    return rho


def _lin_conv(a, b, n_out: int):
    """First ``n_out`` samples of the linear convolution ``a * b`` (FFT, JAX)."""
    length = a.shape[0] + b.shape[0]
    fa = jnp.fft.rfft(a, n=length)
    fb = jnp.fft.rfft(b, n=length)
    return jnp.fft.irfft(fa * fb, n=length)[:n_out]


def ltri_matvec(col, v):
    """Lower-triangular Toeplitz with first column ``col``, applied to ``v``.

    ``(L(col) v)[i] = sum_{k<=i} col[i-k] v[k]`` -- a causal convolution.
    """
    return _lin_conv(col, v, col.shape[0])


def utri_matvec(col, v):
    """Transpose of :func:`ltri_matvec`: upper-triangular Toeplitz applied to ``v``.

    ``(L(col)^T v)[i] = sum_{m>=0} col[m] v[i+m]`` -- a correlation.
    """
    n = col.shape[0]
    return jnp.flip(_lin_conv(col, jnp.flip(v), n))


def toeplitz_matvec(col, v):
    """Symmetric Toeplitz ``C`` (first column ``col``) times ``v``, in ``O(N log N)``.

    Uses ``C = L + L^T - diag(col[0])`` for a symmetric Toeplitz.
    """
    return ltri_matvec(col, v) + utri_matvec(col, v) - col[0] * v


def inverse_generator(col) -> np.ndarray:
    """Generator ``x`` (first column of ``C^{-1}``) for the Gohberg-Semencul matvec.

    Solves ``C x = e_0`` by Levinson-Durbin (``scipy.linalg.solve_toeplitz``) for the
    symmetric positive-definite Toeplitz ``C`` with first column ``col``. Host-side,
    ``O(N^2)`` time / ``O(N)`` memory -- run once per PSD, never in the hot path.
    """
    col = np.asarray(col, dtype=float)
    e0 = np.zeros_like(col)
    e0[0] = 1.0
    return solve_toeplitz(col, e0)


def inverse_matvec(x, v):
    """``C^{-1} v`` via Gohberg-Semencul, given ``x = inverse_generator(col)``. JAX/FFT.

    ``C^{-1} v = (1/x_0)(L(x) L(x)^T v - L(z) L(z)^T v)`` with
    ``z = (0, x_{N-1}, ..., x_1)``.
    """
    z = jnp.concatenate([jnp.zeros((1,), x.dtype), jnp.flip(x[1:])])
    t1 = ltri_matvec(x, utri_matvec(x, v))
    t2 = ltri_matvec(z, utri_matvec(z, v))
    return (t1 - t2) / x[0]
