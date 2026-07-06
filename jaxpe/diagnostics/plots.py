"""Plotting helpers: corner and trace plots for engine-convention sample arrays."""

import matplotlib

matplotlib.use("Agg")
import corner as corner_module
import matplotlib.pyplot as plt
import numpy as np


def corner_plot(samples, names=None, truths=None, **kwargs):
    """
    Generate a corner plot (pair grid) of the 1D and 2D marginal posteriors.

    Corner plots are the standard way to visualize MCMC results in high dimensions.
    They show the 1D marginalized probability density function (PDF) for each parameter
    along the diagonal, and the 2D correlations between parameter pairs in the off-diagonals.

    Parameters
    ----------
    samples : np.ndarray
        Array of samples. Can be 2D `(n_samples, n_dim)` or 3D `(n_steps, n_chains, n_dim)`.
    names : list of str, optional
        Labels for each parameter dimension.
    truths : list of float, optional
        True values of the parameters (if known, e.g., from an injection) to overlay
        as lines on the plots.
    **kwargs
        Additional keyword arguments passed to `corner.corner`.

    Returns
    -------
    matplotlib.figure.Figure
        The generated corner plot figure.
    """
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
    """
    Generate a trace plot of the MCMC chains over time.

    Trace plots show the parameter value (y-axis) as a function of the MCMC step number
    (x-axis) for several independent chains. They are excellent for visually diagnosing:
    - **Burn-in**: Do the chains start far away and take a while to reach the bulk?
    - **Mixing**: Do the chains rapidly oscillate around the mean, or do they wander slowly?
    - **Convergence**: Do all plotted chains overlap in the same stationary distribution?

    Parameters
    ----------
    xs : np.ndarray
        Array of positions of shape `(n_steps, n_chains, n_dim)`.
    names : list of str, optional
        Labels for each parameter dimension.
    max_chains : int, default=8
        Maximum number of individual chains to plot (to avoid visual clutter).

    Returns
    -------
    matplotlib.figure.Figure
        The generated trace plot figure.
    """
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
