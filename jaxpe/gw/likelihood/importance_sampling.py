"""Defensive adaptive importance-sampling toolkit for the extrinsic marginal.

The mode-based full marginal (:meth:`ModesNetworkLikelihood.log_marginal_likelihood_full`)
integrates the extrinsic likelihood over the unit 4-cube (ra, dec, psi, iota) by
defensive adaptive importance sampling. This module holds the pieces that are
independent of the likelihood itself: the flat-measure cube parametrization, the
defensive Gaussian-KDE mixture proposal (wrapped on periodic dims, reflected on
bounded dims), and the balance-heuristic accumulator that recycles every batch.
"""

import numpy as np
from scipy.special import logsumexp as logsumexp_np

# unit-cube dims of the extrinsic (ra, dec, psi, iota) parametrization: ra and psi are
# periodic on their intervals; dec and iota are reflected at their boundaries
_EXT_PERIODIC = (True, False, True, False)


def _ext_cube_to_angles(u):
    """Map the unit 4-cube (flat prior measure) to (ra, dec, psi, iota)."""
    return np.stack(
        [
            2.0 * np.pi * u[:, 0],
            np.arcsin(2.0 * u[:, 1] - 1.0),
            np.pi * u[:, 2],
            np.arccos(2.0 * u[:, 3] - 1.0),
        ],
        axis=1,
    )


def _mixture_log_density(u, centers, widths, comp_w, defense):
    """log q(u) of the defensive mixture on the unit 4-cube.

    q = defense * 1 + (1 - defense) * sum_k comp_w[k] * prod_d phi_d(u_d; c_kd, h_d),
    with per-dim Gaussian kernels wrapped (periodic dims) or reflected (bounded dims)
    back into [0, 1] via their +-1-cell images, so q integrates to 1 on the cube.
    """
    dens = np.ones((u.shape[0], centers.shape[0]))
    for d in range(4):
        x = u[:, None, d] - centers[None, :, d]
        h = widths[d]
        if _EXT_PERIODIC[d]:
            imgs = (x - 1.0, x, x + 1.0)
        else:
            c = centers[None, :, d]
            imgs = (x, u[:, None, d] + c, u[:, None, d] + c - 2.0)
        dens_d = sum(
            np.exp(-0.5 * (im / h) ** 2) / (np.sqrt(2.0 * np.pi) * h) for im in imgs
        )
        dens *= dens_d
    return np.log(defense + (1.0 - defense) * dens @ comp_w)


def _mixture_sample(rng, n, centers, widths, comp_w, defense):
    """Draw n points from the defensive mixture, folded back into the unit cube."""
    out = rng.uniform(size=(n, 4))
    kde = rng.uniform(size=n) >= defense
    n_kde = int(kde.sum())
    if n_kde and len(centers):
        idx = rng.choice(len(centers), p=comp_w, size=n_kde)
        x = centers[idx] + rng.normal(size=(n_kde, 4)) * widths[None, :]
        for d in range(4):
            if _EXT_PERIODIC[d]:
                x[:, d] = np.mod(x[:, d], 1.0)
            else:
                y = np.abs(np.mod(x[:, d], 2.0))
                x[:, d] = np.where(y > 1.0, 2.0 - y, y)
        out[kde] = x
    return out


class BalanceHeuristicAccumulator:
    """Recycles importance-sampling batches drawn from *different* proposals.

    Motivation: the adaptive extrinsic marginalization draws a pilot batch from
    the uniform prior and further batches from successively refined defensive
    kernel-density proposals. Estimating the integral from the last batch alone
    (the original implementation) discards every earlier evaluation. The balance
    heuristic (Veach & Guibas 1995) instead weights EVERY point as if it had been
    drawn from the sample-count-weighted mixture of all proposals actually used:

        log w_i = log_likelihood_i - log q_bar(u_i),
        q_bar(u) = sum_j (n_j / N) q_j(u),   N = sum_j n_j,

    which is a valid importance-sampling estimator for the whole collection and
    provably close to the best possible combination for fixed proposals. All
    evaluations then contribute to both the integral estimate and the effective
    sample size, so a quality target is reached with roughly half the evaluations
    the discard-and-double retry strategy needed.

    Honest caveat: our proposals are built adaptively from earlier batches, so
    later proposals are not independent of earlier points (the adaptive multiple
    importance sampling regime). Strict unbiasedness is lost; consistency holds,
    and the finite-sample bias is far below the variance of discarding batches.

    The accumulator stores each proposal's log-density function and maintains the
    (n_batches, n_points) matrix of every proposal evaluated at every point,
    extended incrementally as batches arrive -- pure Gaussian algebra, negligible
    next to the likelihood evaluations being recycled.
    """

    def __init__(self):
        self.points: np.ndarray | None = None  # (N, d) accumulated positions
        self.log_likelihoods: np.ndarray | None = None  # (N,)
        self.batch_sizes: list[int] = []
        self._log_density_functions: list = []
        # row j = proposal j's log-density at ALL accumulated points
        self._log_density_rows: list[np.ndarray] = []

    @property
    def n_points(self) -> int:
        return 0 if self.points is None else len(self.points)

    def add_batch(self, points, log_likelihood_values, log_density_function):
        """Add one batch: positions, their log-likelihoods, and the log-density
        of the proposal they were drawn from (callable ``(n, d) -> (n,)``)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        log_likelihood_values = np.asarray(log_likelihood_values, dtype=float)
        # extend every EXISTING proposal's row with its density at the NEW points
        for j, density_fn in enumerate(self._log_density_functions):
            self._log_density_rows[j] = np.concatenate(
                [self._log_density_rows[j], density_fn(points)]
            )
        if self.points is None:
            self.points = points
            self.log_likelihoods = log_likelihood_values
        else:
            self.points = np.concatenate([self.points, points])
            self.log_likelihoods = np.concatenate(
                [self.log_likelihoods, log_likelihood_values]
            )
        # ... and the NEW proposal's row over ALL points (old + new)
        self._log_density_functions.append(log_density_function)
        self._log_density_rows.append(log_density_function(self.points))
        self.batch_sizes.append(len(points))

    def log_balance_weights(self) -> np.ndarray:
        """log w_i = log_likelihood_i - log q_bar(u_i) for every accumulated point."""
        counts = np.asarray(self.batch_sizes, dtype=float)
        log_batch_fractions = np.log(counts / counts.sum())
        # mixture density: logsumexp over proposals of log(n_j/N) + log q_j(u_i)
        log_mixture = logsumexp_np(
            log_batch_fractions[:, None] + np.stack(self._log_density_rows), axis=0
        )
        return self.log_likelihoods - log_mixture

    def log_normalization(self) -> float:
        """The recycled estimate of log integral(likelihood x prior)."""
        log_weights = self.log_balance_weights()
        return float(logsumexp_np(log_weights) - np.log(self.n_points))

    def effective_sample_size(self) -> float:
        """(sum w)^2 / sum w^2 over ALL accumulated points."""
        log_weights = self.log_balance_weights()
        return float(
            np.exp(2.0 * logsumexp_np(log_weights) - logsumexp_np(2.0 * log_weights))
        )
