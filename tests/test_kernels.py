"""Kernel correctness: each kernel must recover the moments of a correlated Gaussian.

These are the load-bearing tests of the sampling engine: a kernel with a wrong MH
correction or integrator will bias the variance at the several-percent level, which
these tolerances catch with the sample sizes used.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpe.kernels import (
    HMC,
    MALA,
    MMALA,
    ULD,
    RandomWalk,
    adapted_step_size,
    ensemble_scale,
    run_chains,
    with_updates,
)

N_DIM = 3
MEAN = jnp.array([1.0, -2.0, 0.5])
# Correlated covariance with strongly unequal scales (stds ~ 0.5, 1.0, 3.0)
L = jnp.array([[0.5, 0.0, 0.0], [0.6, 0.8, 0.0], [-0.3, 0.4, 3.0]])
COV = L @ L.T
COV_INV = jnp.linalg.inv(COV)
STD = jnp.sqrt(jnp.diag(COV))


def logp(x):
    d = x - MEAN
    return -0.5 * d @ COV_INV @ d


KERNELS = [
    RandomWalk(step_size=0.35),
    MALA(step_size=0.5),
    HMC(step_size=0.35, n_leapfrog=8),
    # exact constant metric: near-ideal preconditioning, large steps possible
    MMALA(step_size=1.2, metric_fn=lambda x: COV_INV),
    # constant dense proposal covariance (dense-mass MALA path)
    MMALA(step_size=1.2, cov=COV),
]


@pytest.mark.parametrize(
    "kernel",
    KERNELS,
    ids=["RandomWalk", "MALA", "HMC", "MMALA-metric", "MMALA-cov"],
)
def test_kernel_recovers_gaussian_moments(kernel):
    key = jax.random.PRNGKey(0)
    key_init, key_run = jax.random.split(key)
    n_chains, n_steps, burn = 64, 4000, 1000

    x0 = jax.random.normal(key_init, (n_chains, N_DIM)) + MEAN
    _, xs, _, infos = run_chains(key_run, kernel, logp, x0, n_steps)
    samples = xs[burn:].reshape(-1, N_DIM)

    acc = float(jnp.mean(infos.accepted))
    assert 0.1 < acc < 0.995, f"acceptance {acc} out of range"

    std = np.asarray(STD)
    mean_err = np.abs(np.asarray(samples.mean(0) - MEAN))
    assert np.all(mean_err < 0.2 * std), f"mean error {mean_err}"
    cov_est = np.cov(np.asarray(samples).T)
    cov_tol = 0.25 * np.outer(std, std)
    assert np.all(np.abs(cov_est - np.asarray(COV)) < cov_tol), (
        f"cov error\n{cov_est - np.asarray(COV)}"
    )


def test_mala_preconditioner_improves_acceptance():
    """With the true per-dimension scales, MALA at fixed eps accepts more than without."""
    key = jax.random.PRNGKey(3)
    x0 = jax.random.normal(key, (32, N_DIM)) + MEAN
    _, _, _, info_plain = run_chains(key, MALA(step_size=1.0), logp, x0, 500)
    _, _, _, info_scaled = run_chains(key, MALA(step_size=1.0, scale=STD), logp, x0, 500)
    assert jnp.mean(info_scaled.accepted) > jnp.mean(info_plain.accepted) + 0.1


def test_adaptation_reaches_target():
    key = jax.random.PRNGKey(1)
    x0 = jax.random.normal(key, (64, N_DIM)) + MEAN
    kernel = MALA(step_size=5.0)  # deliberately far too large
    for i in range(15):
        key, sub = jax.random.split(key)
        states, xs, _, infos = run_chains(sub, kernel, logp, x0, 100)
        x0 = states.x
        acc = jnp.mean(infos.accepted)
        kernel = with_updates(
            kernel, step_size=adapted_step_size(kernel.step_size, acc, target=0.574)
        )
    assert 0.4 < float(acc) < 0.75, f"final acceptance {float(acc)}"
    scale = ensemble_scale(xs)
    np.testing.assert_allclose(scale, STD, rtol=0.4)


def test_uld_recovers_gaussian_moments():
    """Unadjusted kinetic Langevin: correct to O(eps^2); test with loose tolerances."""
    key = jax.random.PRNGKey(5)
    key_init, key_run = jax.random.split(key)
    n_chains, n_steps, burn = 64, 4000, 1000

    kernel = ULD(step_size=0.08, friction=1.5, scale=STD)
    x0 = jax.random.normal(key_init, (n_chains, N_DIM)) + MEAN
    _, xs, _, infos = run_chains(key_run, kernel, logp, x0, n_steps)
    assert float(jnp.mean(infos.accepted)) == 1.0  # no MH step

    samples = xs[burn:].reshape(-1, N_DIM)
    std = np.asarray(STD)
    mean_err = np.abs(np.asarray(samples.mean(0) - MEAN))
    assert np.all(mean_err < 0.25 * std), f"mean error {mean_err}"
    cov_est = np.cov(np.asarray(samples).T)
    cov_tol = 0.3 * np.outer(std, std)
    assert np.all(np.abs(cov_est - np.asarray(COV)) < cov_tol), (
        f"cov error\n{cov_est - np.asarray(COV)}"
    )


def test_run_chains_thinning_shape():
    key = jax.random.PRNGKey(2)
    x0 = jax.random.normal(key, (8, N_DIM))
    _, xs, logps, infos = run_chains(key, RandomWalk(step_size=0.5), logp, x0, 100, thin=10)
    assert xs.shape == (10, 8, N_DIM)
    assert logps.shape == (10, 8)


def test_kernel_rejects_neg_inf_region():
    """A proposal into log_prob = -inf must never be accepted."""

    def logp_bounded(x):
        inside = jnp.all(jnp.abs(x) < 1.0)
        return jnp.where(inside, -0.5 * jnp.sum(x**2), -jnp.inf)

    key = jax.random.PRNGKey(4)
    x0 = jnp.zeros((16, N_DIM))
    for kernel in [RandomWalk(step_size=2.0), MALA(step_size=1.0)]:
        _, xs, logps, _ = run_chains(key, kernel, logp_bounded, x0, 200)
        assert jnp.all(jnp.isfinite(logps))
        assert jnp.all(jnp.abs(xs) < 1.0)
