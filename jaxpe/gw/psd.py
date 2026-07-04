"""One-sided noise power spectral densities: analytic fits, files, and Welch estimates."""

import numpy as np
from scipy.signal import welch as _welch


def aligo_zdhp_psd(freqs, f_low: float = 10.0):
    """Analytic fit to the aLIGO zero-detuning high-power design PSD (arXiv:0903.0338).

    Returns np.inf below ``f_low`` so masked-band likelihoods can divide safely.
    """
    freqs = np.asarray(freqs, float)
    x = np.where(freqs > 0, freqs / 215.0, 1.0)
    psd = 1e-49 * (
        x**-4.14 - 5.0 * x**-2 + 111.0 * (1.0 - x**2 + 0.5 * x**4) / (1.0 + 0.5 * x**2)
    )
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
