"""Unit tests for the JAX-native acquisition support (jaxpe.surrogate.jax_acquisition).

The load-bearing correctness property is that ``build_jax_predictive_mean`` reproduces
GPry's own GP posterior *mean* -- because NORA's MC-sampling step draws from the mean
posterior (``surrogate.predict(..., return_std=False)``; the exploration std enters
only later, in the numpy ranking step). If the JAX mean drifts from GPry's, the
acquisition samples the wrong distribution and silently biases the surrogate posterior.

We therefore build a *real* fitted ``gpry`` ``SurrogateModel`` -- the exact object the
pipeline hands to ``build_jax_predictive_mean`` -- via a tiny ``Runner``, and assert the
JAX predictive mean matches GPry's to a tight tolerance on random query points, for both
kernel families the JAX builder special-cases (RBF and Matern). Parity does not require a
deterministic fit: both sides read the *same* fitted GP.
"""

import numpy as np
import jax
import pytest

jax.config.update("jax_enable_x64", True)

pytest.importorskip("gpry")
from jaxpe.surrogate.jax_acquisition import build_jax_predictive_mean  # noqa: E402


def _fit_surrogate(kernel_spec, dim=2, seed=0):
    """Fit a tiny real GPry SurrogateModel and return it (``runner.gpr``).

    ``kernel_spec`` is whatever ``Runner(surrogate=...)`` accepts: the string ``"RBF"``
    or a ``{"regressor": {"kernel": {...}}}`` dict. The evaluation budget is kept just
    above ``n_initial`` so the run is a few seconds: a fit on a handful of points is all
    the predictive-mean parity check needs.
    """
    from gpry import Runner

    def loglike(x):
        x = np.asarray(x, dtype=float)
        return float(-0.5 * np.sum(x**2))  # isotropic Gaussian, finite everywhere

    bounds = np.array([[-3.0, 3.0]] * dim)
    runner = Runner(
        loglike,
        bounds=bounds,
        verbose=0,
        surrogate=kernel_spec,
        options={
            "seed": seed,
            "n_initial": 8,
            "max_initial": 14,
            "max_total": 16,
            "max_finite": 16,
        },
    )
    runner.run()
    return runner.gpr  # a gpry.surrogate.SurrogateModel


def _max_abs_mean_error(surrogate, dim=2, n_query=40, seed=1):
    """max |GPry mean - JAX mean| over random query points inside the prior box."""
    predict_mean = build_jax_predictive_mean(surrogate)
    rng = np.random.default_rng(seed)
    x_query = rng.uniform(-3.0, 3.0, size=(n_query, dim))
    y_gpry = np.ravel(surrogate.predict(x_query, return_std=False))
    y_jax = np.array([float(predict_mean(x_query[i])) for i in range(n_query)])
    return float(np.max(np.abs(y_gpry - y_jax)))


def test_jax_predictive_mean_matches_gpry_rbf():
    surrogate = _fit_surrogate("RBF")
    # sanity: we really did fit a GP with an RBF-family kernel
    assert surrogate.gpr.kernel_ is not None
    assert "RBF" in str(surrogate.gpr.kernel_)
    assert _max_abs_mean_error(surrogate) < 1e-6


def test_jax_predictive_mean_matches_gpry_matern():
    surrogate = _fit_surrogate({"regressor": {"kernel": {"Matern": {"nu": 2.5}}}})
    assert surrogate.gpr.kernel_ is not None
    assert "Matern" in str(surrogate.gpr.kernel_)
    assert _max_abs_mean_error(surrogate) < 1e-6
