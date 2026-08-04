"""Metrics for evaluating posterior samples."""

import numpy as np


def js_divergence(a, b, bins=120):
    """Jensen-Shannon divergence (base 2) between two 1-D sample sets.

    Args:
        a: Array of samples from distribution P.
        b: Array of samples from distribution Q.
        bins: Number of histogram bins.

    Returns:
        The JS divergence (base 2). 0 = identical, 1 = disjoint.
    """
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(a, bins=edges, density=True)
    q, _ = np.histogram(b, bins=edges, density=True)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(x, y):
        ok = x > 0
        return float(np.sum(x[ok] * np.log2(x[ok] / y[ok])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
