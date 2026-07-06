"""Convergence Diagnostics: Split $\hat{R}$ and Effective Sample Size (ESS).

In MCMC, we need to know whether our chains have forgotten their starting positions
(convergence) and how many independent samples we have actually drawn (since MCMC samples
are correlated).

Motivation & Math
-----------------
- **Split $\hat{R}$ (Gelman-Rubin statistic)**: Compares the variance *between* multiple chains
  to the variance *within* the chains. If the chains have converged to the same stationary
  distribution, these variances should be roughly equal, yielding $\hat{R} \approx 1$.
  We split each chain in half to also test for non-stationarity within a single chain.
- **Effective Sample Size (ESS)**: Correlated samples contain less information than independent
  ones. The ESS estimates the number of independent samples our chains are worth by
  analyzing the autocorrelation function $\rho_t$:
  $$ \text{ESS} = \frac{N}{1 + 2 \sum_{t=1}^\infty \rho_t} $$
  Higher ESS means better, more robust posterior estimates.

These functions run on the host in pure NumPy as post-processing read-out tools.
"""

import numpy as np


def split_rhat(xs) -> np.ndarray:
    """
    Compute the Gelman-Rubin split $\hat{R}$ statistic for each parameter dimension.

    Values near 1.0 indicate convergence. A common rule of thumb is to require
    $\hat{R} < 1.05$ (or $1.01$ for stricter bounds) before trusting the samples.

    Parameters
    ----------
    xs : np.ndarray
        Array of MCMC samples with shape (n_steps, n_chains, n_dim).

    Returns
    -------
    np.ndarray
        The $\hat{R}$ value for each of the `n_dim` parameters.
    """
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
    """
    Compute the Effective Sample Size (ESS) for each parameter dimension.

    Uses the Geyer initial-monotone sequence estimator. It calculates the autocorrelation
    using Fast Fourier Transforms (FFTs) for efficiency, and then sums the autocorrelation
    values until the sum of adjacent pairs ceases to be positive and monotonically decreasing.

    Parameters
    ----------
    xs : np.ndarray
        Array of MCMC samples with shape (n_steps, n_chains, n_dim).

    Returns
    -------
    np.ndarray
        The ESS value for each of the `n_dim` parameters.
    """
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
