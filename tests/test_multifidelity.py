"""Unit tests for MultifidelityGaussianProcessRegressor (jaxpe.surrogate.multifidelity).

The class is a thin wrapper around gpry's GaussianProcessRegressor that (a) fits the GP
to residuals ``y - mean_func(X)`` and (b) adds ``mean_func`` (and its gradient) back on
``predict``. GPry's GPR cannot be fit standalone (its ``kernel_`` is initialised only by
the driving SurrogateModel), so we test the two behaviours the way they can each be
isolated faithfully:

* ``fit`` -- patch the parent ``fit`` to capture its arguments, and assert the wrapper
  passes it the residuals and forwards valid kwargs (this also pins the regression where
  ``fit`` used to forward an unsupported ``y_std=`` and raised ``TypeError``).
* ``predict`` -- reclass a *real* Runner-fitted GPR to the multifidelity subclass and
  assert the mean/gradient additions and output-tuple handling match the parent plus the
  mean function exactly.
"""

import numpy as np
import jax
import pytest
from unittest.mock import patch

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

pytest.importorskip("gpry")
from gpry.gpr import GaussianProcessRegressor  # noqa: E402
from jaxpe.surrogate.multifidelity import (  # noqa: E402
    MultifidelityGaussianProcessRegressor,
)


def _mean_func(X):
    """A non-linear, JAX-differentiable prior mean: (N, d) -> (N,)."""
    return jnp.sin(X[:, 0]) + 2.0 * jnp.cos(X[:, 1])


def test_fit_subtracts_mean_and_forwards_kwargs():
    length_scale_prior = np.array([[1e-3, 1e1]] * 2)
    gpr = MultifidelityGaussianProcessRegressor(
        mean_func=_mean_func,
        kernel="Matern",
        length_scale_prior=length_scale_prior,
        optimizer=None,
        noise_level=1e-4,
    )
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 5, (10, 2))
    y = np.asarray(_mean_func(X)) + rng.normal(0, 0.1, 10)

    captured = {}

    def fake_parent_fit(self, X_fit, y_fit, **kwargs):
        captured["X"], captured["y"], captured["kwargs"] = X_fit, y_fit, kwargs
        return self

    # super().fit(...) resolves to the parent class method, which we replace.
    with patch.object(GaussianProcessRegressor, "fit", fake_parent_fit):
        gpr.fit(X, y, validate=False, fit_hyperparameters=False)

    # (1) the GP is fit on residuals, not the raw target
    np.testing.assert_allclose(captured["y"], y - np.asarray(_mean_func(X)), rtol=1e-12)
    np.testing.assert_allclose(captured["X"], X, rtol=1e-12)
    # (2) valid kwargs are forwarded verbatim (and no unsupported y_std= is injected)
    assert captured["kwargs"] == {"validate": False, "fit_hyperparameters": False}


def _fit_real_gpr(dim=2, seed=0):
    """Return a real fitted gpry GPR (runner.gpr.gpr) via a tiny Runner."""
    from gpry import Runner

    def loglike(x):
        x = np.asarray(x, dtype=float)
        return float(-0.5 * np.sum(x**2))

    bounds = np.array([[-3.0, 3.0]] * dim)
    runner = Runner(
        loglike,
        bounds=bounds,
        verbose=0,
        surrogate={"regressor": {"kernel": {"Matern": {"nu": 2.5}}}},
        options={
            "seed": seed,
            "n_initial": 8,
            "max_initial": 14,
            "max_total": 16,
            "max_finite": 16,
        },
    )
    runner.run()
    return runner.gpr.gpr  # the underlying fitted GaussianProcessRegressor


def test_predict_adds_mean_and_grad():
    gpr = _fit_real_gpr()
    rng = np.random.default_rng(1)
    x_query = rng.uniform(-3.0, 3.0, size=(20, 2))

    # parent predictions, captured before reclassing
    base_mean = np.ravel(gpr.predict(x_query, return_std=False))
    # GPry supports return_mean_grad only for a single (1, d) point
    base_mean1, base_grad = gpr.predict(x_query[:1], return_mean_grad=True)

    # reclass the fitted parent into the multifidelity subclass, attaching a mean func:
    # its predict override now calls super().predict (identical state) and adds mean_func.
    gpr.__class__ = MultifidelityGaussianProcessRegressor
    gpr.mean_func = jax.jit(_mean_func)
    gpr.mean_func_grad = jax.jit(jax.grad(lambda x: _mean_func(x[None, :])[0]))

    # mean: super().predict + mean_func(X)
    mf_mean = np.ravel(gpr.predict(x_query, return_std=False))
    np.testing.assert_allclose(
        mf_mean, base_mean + np.asarray(_mean_func(x_query)), rtol=1e-6, atol=1e-8
    )

    # mean-grad: super() grad + grad(mean_func) at the query point
    mf_mean1, mf_grad = gpr.predict(x_query[:1], return_mean_grad=True)
    expected_grad = np.asarray(base_grad) + np.asarray(gpr.mean_func_grad(x_query[0]))
    np.testing.assert_allclose(mf_grad, expected_grad, rtol=1e-6, atol=1e-8)
    # the mean returned alongside the gradient is consistent with the mean-only call
    np.testing.assert_allclose(np.ravel(mf_mean1), mf_mean[:1], rtol=1e-6, atol=1e-8)
