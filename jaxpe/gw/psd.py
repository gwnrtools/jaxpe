r"""Power Spectral Density (PSD) Estimation and Models.

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
    psd = 1e-49 * (
        x**-4.14 - 5.0 * x**-2 + 111.0 * (1.0 - x**2 + 0.5 * x**4) / (1.0 + 0.5 * x**2)
    )
    return np.where(freqs >= f_low, psd, np.inf)


def psd_from_file(path, freqs):
    """Load a two-column (f, S) ASCII PSD and interpolate onto ``freqs`` (inf outside)."""
    f_in, s_in = np.loadtxt(path, unpack=True)[:2]
    return np.interp(np.asarray(freqs, float), f_in, s_in, left=np.inf, right=np.inf)


#: Short names for the LALSimulation design curves worth reaching for by name.
#: Everything else remains available through ``lalsim_psd(..., name=<full symbol>)``.
LALSIM_PSDS = {
    "CE": "SimNoisePSDCosmicExplorerP1600143",
    "CE-wideband": "SimNoisePSDCosmicExplorerWidebandP1600143",
    "CE-pessimistic": "SimNoisePSDCosmicExplorerPessimisticP1600143",
    "ET": "SimNoisePSDEinsteinTelescopeP1600143",
    "aplus": "SimNoisePSDaLIGOAPlusDesignSensitivityT1800042",
    "aligo-design": "SimNoisePSDaLIGODesignSensitivityP1200087",
    "advirgo-O4": "SimNoisePSDaLIGOAdVO4T1800545",
}


def lalsim_psd(name, freqs, f_low: float = 5.0):
    r"""Evaluate a LALSimulation design PSD on a uniform grid via the series API.

    Third-generation sensitivity curves (Cosmic Explorer, Einstein Telescope) have no
    closed-form fit of the :func:`aligo_zdhp_psd` kind, so they come from LALSimulation
    rather than from an analytic expression here.

    Parameters
    ----------
    name : str
        A key of :data:`LALSIM_PSDS` (``"CE"``, ``"ET"``, ...) or the full
        ``SimNoisePSD*`` symbol name.
    freqs : np.ndarray
        **Uniform** frequency grid starting at 0, as produced by
        :func:`jaxpe.gw.conditioning.rfft_freqs`. The series API is defined by
        ``(df, n)`` rather than by arbitrary sample points, so a non-uniform grid
        cannot be honoured and is rejected.
    f_low : float, default=5.0
        Low-frequency cutoff handed to LALSimulation.

    Returns
    -------
    np.ndarray
        One-sided PSD, with zero/undefined entries (including ``f = 0`` and
        ``f < f_low``) mapped to ``inf`` so the likelihood's band mask drops them.
    """
    try:
        import lal
        import lalsimulation as ls
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"PSD {name!r} comes from LALSimulation, which is not installed. Install "
            "lalsuite, or supply the curve as a two-column ASCII file instead."
        ) from exc

    symbol = LALSIM_PSDS.get(name, name)
    fn = getattr(ls, symbol, None)
    if fn is None:
        raise ValueError(
            f"unknown LALSimulation PSD {name!r} (resolved to {symbol!r}). Known short "
            f"names: {', '.join(sorted(LALSIM_PSDS))}."
        )

    freqs = np.asarray(freqs, float)
    if freqs.size < 2:
        raise ValueError("freqs must contain at least two samples")
    df = float(freqs[1] - freqs[0])
    if not np.allclose(np.diff(freqs), df, rtol=1e-9, atol=0.0):
        raise ValueError(
            "lalsim_psd needs a uniform frequency grid: the series API is specified by "
            "(df, n), not by arbitrary sample points. Interpolate from a file instead."
        )

    series = lal.CreateREAL8FrequencySeries(
        "psd", lal.LIGOTimeGPS(0), 0.0, df, lal.SecondUnit, freqs.size
    )
    fn(series, f_low)
    psd = np.asarray(series.data.data, float)
    return np.where(psd > 0.0, psd, np.inf)


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
