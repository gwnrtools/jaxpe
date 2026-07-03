"""Convergence diagnostics: split R-hat and effective sample size.

Inputs follow the engine convention ``xs`` of shape (n_steps, n_chains, n_dim).
These run on the host in numpy — they are read-out tools, not part of the jitted loop.
"""

import numpy as np


def split_rhat(xs) -> np.ndarray:
    """Gelman-Rubin split-R-hat per dimension; values near 1 indicate convergence."""
    xs = np.asarray(xs)
    n, m, d = xs.shape
    half = n // 2
    chains = np.concatenate([xs[:half], xs[half : 2 * half]], axis=1)  # (half, 2m, d)
    n, m = chains.shape[:2]

    chain_means = chains.mean(axis=0)  # (2m, d)
    within = chains.var(axis=0, ddof=1).mean(axis=0)  # (d,)
    between = n * chain_means.var(axis=0, ddof=1)  # (d,)
    var_hat = (n - 1) / n * within + between / n
    return np.sqrt(var_hat / within)


def effective_sample_size(xs) -> np.ndarray:
    """ESS per dimension across all chains (Geyer initial-monotone estimator)."""
    xs = np.asarray(xs)
    n, m, d = xs.shape
    ess = np.empty(d)
    for j in range(d):
        x = xs[:, :, j] - xs[:, :, j].mean(axis=0)
        # FFT autocovariance per chain, averaged
        nfft = int(2 ** np.ceil(np.log2(2 * n)))
        f = np.fft.rfft(x, n=nfft, axis=0)
        acov = np.fft.irfft(f * np.conj(f), n=nfft, axis=0)[:n].real
        acov = (acov / np.arange(n, 0, -1)[:, None]).mean(axis=1)
        rho = acov / acov[0]
        # Geyer: sum consecutive pairs while positive and monotone
        even = rho[0::2][: (n // 2)]
        odd = rho[1::2][: len(even)]
        pair = even + odd[: len(even)] if len(odd) == len(even) else even[:-1] + odd
        pair = np.minimum.accumulate(pair)
        keep = pair > 0
        if keep.any():
            tau = -1.0 + 2.0 * pair[keep].sum()
        else:
            tau = 1.0
        ess[j] = m * n / max(tau, 1.0)
    return ess
