"""Plotting helpers: corner and trace plots for engine-convention sample arrays."""

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
    try:
        import corner as corner_module
    except ImportError:
        raise ImportError(
            "corner is required for corner_plot. Install with `pip install jaxpe[plot]`."
        )

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
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for trace_plot. Install with `pip install jaxpe[plot]`."
        )

    xs = np.asarray(xs)
    n_dim = xs.shape[-1]
    fig, axes = plt.subplots(
        n_dim, 1, figsize=(8, 1.8 * n_dim), sharex=True, squeeze=False
    )
    for j in range(n_dim):
        ax = axes[j, 0]
        ax.plot(xs[:, :max_chains, j], alpha=0.6, lw=0.5)
        ax.set_ylabel(names[j] if names is not None else f"x{j}")
    axes[-1, 0].set_xlabel("step")
    fig.tight_layout()
    return fig


def _derived_m1_m2_chieff(s, truth, i):
    """(m1, m2, chi_eff) per sample and at the truth, from (chirp_mass, eta, spins).

    m1, m2, chi_eff are not themselves sampled -- they are the standard invertible
    map from (chirp_mass, eta) to component masses (m1 >= m2, same convention as
    ``eta_to_q`` in run_bns_ce_pe.py) plus the mass-weighted spin combination
    chi_eff = (m1 spin1z + m2 spin2z)/(m1+m2). Computed identically for every
    sample and for the truth so the two are directly comparable.
    """

    def convert(mc, eta, s1, s2):
        mtot = mc * eta ** (-0.6)
        delta = np.sqrt(np.clip(1.0 - 4.0 * eta, 0.0, None))
        m1 = mtot * (1.0 + delta) / 2.0
        m2 = mtot * (1.0 - delta) / 2.0
        chi_eff = (m1 * s1 + m2 * s2) / mtot
        return m1, m2, chi_eff

    m1, m2, chi_eff = convert(
        s[:, i["chirp_mass"]], s[:, i["eta"]], s[:, i["spin1z"]], s[:, i["spin2z"]]
    )
    m1_t, m2_t, chieff_t = convert(
        truth[i["chirp_mass"]], truth[i["eta"]], truth[i["spin1z"]], truth[i["spin2z"]]
    )
    return m1, m2, chi_eff, m1_t, m2_t, chieff_t


def _chirp_mass_offset_scale(spread):
    """Power-of-ten multiplier putting an O(spread)-wide offset into O(1-10) units.

    The chirp-mass posterior is ~1e-6 Msun wide for a BNS but ~0.1-1 Msun wide for
    an 80 Msun BBH (same relative SNR, very different absolute precision) -- a
    fixed 1e6 multiplier that reads well for one is unreadable for the other, so
    the scale is derived from the actual spread each time rather than hardcoded.
    """
    if spread <= 0:
        return 1.0, 0
    exp = int(np.floor(np.log10(spread)))
    return 10.0**-exp, exp
