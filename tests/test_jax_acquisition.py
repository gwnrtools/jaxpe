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
import jax.numpy as jnp  # noqa: E402

pytest.importorskip("gpry")
from jaxpe.surrogate.jax_acquisition import (  # noqa: E402
    JAXInterfaceBlackJAX,
    build_jax_predictive_mean,
)


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


def _predict_jax(predict_mean, x_query):
    return np.array([float(predict_mean(x_query[i])) for i in range(len(x_query))])


def _max_abs_mean_error(surrogate, pad_to=None, dim=2, n_query=40, seed=1):
    """max |GPry mean - JAX mean| over random query points inside the prior box."""
    predict_mean = build_jax_predictive_mean(surrogate, pad_to=pad_to)
    rng = np.random.default_rng(seed)
    x_query = rng.uniform(-3.0, 3.0, size=(n_query, dim))
    y_gpry = np.ravel(surrogate.predict(x_query, return_std=False))
    return float(np.max(np.abs(y_gpry - _predict_jax(predict_mean, x_query))))


@pytest.fixture(scope="module")
def rbf_surrogate():
    return _fit_surrogate("RBF")


@pytest.fixture(scope="module")
def matern_surrogate():
    return _fit_surrogate({"regressor": {"kernel": {"Matern": {"nu": 2.5}}}})


def test_jax_predictive_mean_matches_gpry_rbf(rbf_surrogate):
    # sanity: we really did fit a GP with an RBF-family kernel
    assert rbf_surrogate.gpr.kernel_ is not None
    assert "RBF" in str(rbf_surrogate.gpr.kernel_)
    assert _max_abs_mean_error(rbf_surrogate) < 1e-6


def test_jax_predictive_mean_matches_gpry_matern(matern_surrogate):
    assert matern_surrogate.gpr.kernel_ is not None
    assert "Matern" in str(matern_surrogate.gpr.kernel_)
    assert _max_abs_mean_error(matern_surrogate) < 1e-6


def test_fixed_capacity_padding_is_exact(rbf_surrogate):
    """Zero-alpha padding to a fixed training capacity must not change the mean.

    This is what lets a jitted predictive keep a static shape as the GPry training set
    grows (killing the per-iteration recompile) without perturbing the result.
    """
    n_train = rbf_surrogate.gpr.X_train_.shape[0]
    f_unpadded = build_jax_predictive_mean(rbf_surrogate)
    f_padded = build_jax_predictive_mean(rbf_surrogate, pad_to=n_train + 50)
    rng = np.random.default_rng(2)
    x_query = rng.uniform(-3.0, 3.0, size=(30, 2))
    y_unpadded = _predict_jax(f_unpadded, x_query)
    y_padded = _predict_jax(f_padded, x_query)
    # Exact in real arithmetic (padded rows carry alpha=0, so they drop out of the sum);
    # the only difference is float64 reduction-order noise from summing more terms, which
    # is ~1e-10 here -- four orders below the 1e-6 GPry-parity scale asserted below.
    np.testing.assert_allclose(y_padded, y_unpadded, rtol=1e-8, atol=1e-9)
    # and padding must not break agreement with GPry
    assert _max_abs_mean_error(rbf_surrogate, pad_to=n_train + 50) < 1e-6


def _fake_predictive_params(n_pad=64, d=2, seed=0, scale=1.0):
    """Fabricate a valid fixed-shape predictive-params dict (a small GP mean surface).

    Values are arbitrary but the *shapes* are what the jitted step keys on, so this is
    enough to exercise compile reuse without fitting a real GP.
    """
    rng = np.random.default_rng(seed)
    n = 10
    X = np.zeros((n_pad, d))
    X[:n] = rng.uniform(-2.0, 2.0, (n, d))
    alpha = np.zeros(n_pad)
    alpha[:n] = scale * rng.normal(size=n)
    return dict(
        X_train=jnp.asarray(X),
        alpha=jnp.asarray(alpha),
        W=jnp.eye(d),
        b=jnp.zeros(d),
        length_scale=jnp.ones(d),
        constant=jnp.asarray(1.0),
        y_mean=jnp.asarray(0.0),
        y_std=jnp.asarray(1.0),
    )


def test_run_predictive_reuses_compiled_step_across_values():
    """The recompile fix: the jitted NS step is keyed on shape/precision, not on the GP
    values, so successive acquisitions at the same bucket reuse one compiled artifact;
    a larger training bucket compiles a second one."""
    pytest.importorskip("blackjax")
    bounds = np.array([[-3.0, 3.0], [-3.0, 3.0]])
    itf = JAXInterfaceBlackJAX(bounds, verbosity=0)
    itf.set_precision(
        nlive=20, num_inner_steps=5, num_delete=4, precision_criterion=0.01
    )
    itf.precision_settings["max_steps"] = 25

    # two acquisitions, same bucket (N_pad=64), different GP values -> compile once
    itf.run_predictive("RBF", None, _fake_predictive_params(seed=0, scale=1.0), seed=1)
    assert len(itf._compiled_runtime_cache) == 1
    itf.run_predictive("RBF", None, _fake_predictive_params(seed=99, scale=3.0), seed=2)
    assert len(itf._compiled_runtime_cache) == 1  # reused, not recompiled

    # a larger training bucket is a different shape -> a second compiled artifact
    itf.run_predictive("RBF", None, _fake_predictive_params(n_pad=128, seed=0), seed=3)
    assert len(itf._compiled_runtime_cache) == 2
