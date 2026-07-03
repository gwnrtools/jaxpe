"""Plotting helpers: corner and trace plots for engine-convention sample arrays."""

import matplotlib

matplotlib.use("Agg")
import corner as corner_module
import matplotlib.pyplot as plt
import numpy as np


def corner_plot(samples, names=None, truths=None, **kwargs):
    """Corner plot of (n_samples, n_dim) samples (or (n_steps, n_chains, n_dim))."""
    samples = np.asarray(samples)
    if samples.ndim == 3:
        samples = samples.reshape(-1, samples.shape[-1])
    defaults = dict(
        labels=list(names) if names is not None else None,
        truths=None if truths is None else list(np.asarray(truths)),
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
        bins=40,
        smooth=0.9,
    )
    defaults.update(kwargs)
    return corner_module.corner(samples, **defaults)


def trace_plot(xs, names=None, max_chains: int = 8):
    """Trace plot of (n_steps, n_chains, n_dim) positions, a few chains per panel."""
    xs = np.asarray(xs)
    n_dim = xs.shape[-1]
    fig, axes = plt.subplots(n_dim, 1, figsize=(8, 1.8 * n_dim), sharex=True, squeeze=False)
    for j in range(n_dim):
        ax = axes[j, 0]
        ax.plot(xs[:, :max_chains, j], alpha=0.6, lw=0.5)
        ax.set_ylabel(names[j] if names is not None else f"x{j}")
    axes[-1, 0].set_xlabel("step")
    fig.tight_layout()
    return fig
