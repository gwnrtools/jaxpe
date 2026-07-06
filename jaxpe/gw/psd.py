"""Power Spectral Density (PSD) Estimation and Models.

The PSD $S_n(f)$ quantifies how the power of a random noise process is distributed across frequency.
In Gravitational Wave data analysis, we strictly use the **one-sided** PSD, defined for positive
frequencies only.

Motivation & Math
-----------------
The Whittle likelihood relies on knowing the noise variance in every frequency bin.
$$ \sigma_f^2 = \frac{S_n(f)}{4 \Delta f} $$
If the PSD is large at a specific frequency (e.g., due to instrumental resonances like
the 60Hz US power grid lines), that frequency bin is heavily down-weighted in the likelihood.

This module provides:
1. Analytic fits to theoretical design sensitivities (e.g., Advanced LIGO).
2. Utilities to load PSDs from ASCII files.
3. Welch's method to estimate the PSD empirically from real off-source strain data.
"""

import numpy as np
from scipy.signal import welch as _welch


def aligo_zdhp_psd(freqs, f_low: float = 10.0):
    """
    Analytic fit to the Advanced LIGO Zero-Detuning High-Power (ZDHP) design PSD.

    This is a theoretical model of how sensitive aLIGO is expected to be under optimal
    conditions (see arXiv:0903.0338).

    Returns `np.inf` below the cutoff frequency ``f_low``. When the likelihood divides
    by the PSD ($1 / S_n(f)$), these frequencies will naturally be zeroed out and ignored.

    Parameters
    ----------
    freqs : np.ndarray
        Array of frequencies to evaluate the PSD at.
    f_low : float, default=10.0
        The lower cutoff frequency.

    Returns
    -------
    np.ndarray
        The PSD values.
    """
    freqs = np.asarray(freqs, float)
    x = np.where(freqs > 0, freqs / 215.0, 1.0)
    psd = 1e-49 * (x**-4.14 - 5.0 * x**-2 + 111.0 * (1.0 - x**2 + 0.5 * x**4) / (1.0 + 0.5 * x**2))
    return np.where(freqs >= f_low, psd, np.inf)


def psd_from_file(path, freqs):
    """Load a two-column (f, S) ASCII PSD and interpolate onto ``freqs`` (inf outside)."""
    f_in, s_in = np.loadtxt(path, unpack=True)[:2]
    return np.interp(np.asarray(freqs, float), f_in, s_in, left=np.inf, right=np.inf)


def welch_psd(strain, sampling_rate: float, seg_duration: float = 4.0, freqs=None):
    """Median-averaged Welch PSD from off-source strain.

    Returns (frequencies, psd), or the PSD interpolated onto ``freqs`` if given.
    """
    nperseg = int(seg_duration * sampling_rate)
    f, s = _welch(
        np.asarray(strain, float),
        fs=sampling_rate,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        average="median",
    )
    if freqs is None:
        return f, s
    return np.interp(np.asarray(freqs, float), f, s, left=np.inf, right=np.inf)
