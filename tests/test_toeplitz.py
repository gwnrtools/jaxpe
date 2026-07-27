"""Tests for the symmetric-Toeplitz covariance utilities (toeplitz.py, Phase RB-3).

Every fast operation is checked against an explicit dense reference: ``toeplitz_matvec``
against ``C @ v`` and ``inverse_matvec`` (Gohberg-Semencul) against ``np.linalg.solve``.
The performance test confirms the Gohberg-Semencul ``C^-1 v`` is much faster than a dense
solve once the generator is precomputed (JAX compile time excluded).
"""

import time

import numpy as np

import jax
import jax.numpy as jnp

from jaxpe.gw.likelihood.toeplitz import (
    autocorrelation_from_psd,
    inverse_generator,
    inverse_matvec,
    ltri_matvec,
    toeplitz_matvec,
    utri_matvec,
)


def _ar1_col(n, r=0.6, sigma2=1.3):
    """First column of an AR(1) autocovariance -- a symmetric positive-definite Toeplitz."""
    return sigma2 * r ** np.arange(n)


def _dense(col):
    """Dense symmetric Toeplitz matrix with first column/row ``col``."""
    n = len(col)
    idx = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return np.asarray(col)[idx]


# --------------------------------------------------------------------- matvecs


def test_triangular_matvecs_vs_dense():
    n = 40
    col = _ar1_col(n)
    v = np.random.default_rng(0).standard_normal(n)
    C = _dense(col)
    L = np.tril(C)
    got_l = np.asarray(ltri_matvec(jnp.asarray(col), jnp.asarray(v)))
    got_u = np.asarray(utri_matvec(jnp.asarray(col), jnp.asarray(v)))
    assert np.allclose(got_l, L @ v, atol=1e-11)
    assert np.allclose(got_u, L.T @ v, atol=1e-11)


def test_toeplitz_matvec_vs_dense():
    n = 64
    col = _ar1_col(n)
    rng = np.random.default_rng(1)
    C = _dense(col)
    for _ in range(5):
        v = rng.standard_normal(n)
        got = np.asarray(toeplitz_matvec(jnp.asarray(col), jnp.asarray(v)))
        assert np.allclose(got, C @ v, atol=1e-11), np.max(np.abs(got - C @ v))


def test_inverse_matvec_vs_dense_solve():
    n = 96
    col = _ar1_col(n)
    C = _dense(col)
    x = inverse_generator(col)
    rng = np.random.default_rng(2)
    for _ in range(5):
        v = rng.standard_normal(n)
        got = np.asarray(inverse_matvec(jnp.asarray(x), jnp.asarray(v)))
        want = np.linalg.solve(C, v)
        assert np.allclose(got, want, atol=1e-9, rtol=1e-9), np.max(np.abs(got - want))


def test_inverse_generator_is_first_column_of_inverse():
    n = 50
    col = _ar1_col(n)
    x = inverse_generator(col)
    assert np.allclose(x, np.linalg.inv(_dense(col))[:, 0], atol=1e-10)


def test_inverse_round_trip():
    """C^-1 (C v) == v to high precision."""
    n = 128
    col = _ar1_col(n)
    x = inverse_generator(col)
    v = np.random.default_rng(3).standard_normal(n)
    cv = toeplitz_matvec(jnp.asarray(col), jnp.asarray(v))
    back = np.asarray(inverse_matvec(jnp.asarray(x), cv))
    assert np.allclose(back, v, atol=1e-9)


def test_inverse_matvec_jittable_and_vmapped():
    n = 64
    col = _ar1_col(n)
    x = jnp.asarray(inverse_generator(col))
    C = _dense(col)
    V = np.random.default_rng(4).standard_normal((7, n))  # 7 right-hand sides
    batched = jax.jit(jax.vmap(inverse_matvec, in_axes=(None, 0)))
    got = np.asarray(batched(x, jnp.asarray(V)))
    want = np.linalg.solve(C, V.T).T
    assert np.allclose(got, want, atol=1e-9)


# --------------------------------------------------------------------- ACF from PSD


def test_autocorrelation_white_noise_is_diagonal():
    """A flat PSD gives an autocorrelation concentrated at lag 0 (white noise)."""
    n = 256
    dt = 1.0 / 2048.0
    S = np.full(n // 2 + 1, 3.0)
    rho = autocorrelation_from_psd(S, dt)
    df = 1.0 / (n * dt)
    assert rho.shape == (n,)
    # rho[0] == variance == sum_k S_k df (up to the endpoint-weight approximation)
    assert abs(rho[0] - S.sum() * df) < 0.02 * rho[0]
    # off-diagonal lags are negligible compared to lag 0
    assert np.max(np.abs(rho[1:])) < 1e-9 * rho[0]


def test_autocorrelation_from_colored_psd_is_positive_definite():
    """A smooth colored PSD yields a valid (PD) Toeplitz covariance."""
    n = 512
    dt = 1.0 / 2048.0
    f = np.arange(n // 2 + 1) * (1.0 / (n * dt))
    f_safe = np.clip(f, f[1], None)
    S = 1.0 + (30.0 / f_safe) ** 4 + (f_safe / 300.0) ** 2  # red + blue tilt
    rho = autocorrelation_from_psd(S, dt)
    C = _dense(rho)
    eigmin = np.linalg.eigvalsh(C).min()
    assert eigmin > 0, eigmin
    # and the fast inverse round-trips on this covariance
    x = inverse_generator(rho)
    v = np.random.default_rng(5).standard_normal(n)
    back = np.asarray(
        inverse_matvec(
            jnp.asarray(x), toeplitz_matvec(jnp.asarray(rho), jnp.asarray(v))
        )
    )
    assert np.allclose(back, v, atol=1e-6)


# --------------------------------------------------------------------- performance


def _median_time(fn, *args, n=30):
    jax.block_until_ready(fn(*args))  # warm up: compile + first exec (EXCLUDED)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def test_gohberg_semencul_faster_than_dense_solve():
    """Applying C^-1 via Gohberg-Semencul (O(N log N), generator precomputed) is much
    faster than a dense solve (O(N^3)), and gives the same answer."""
    n = 2048
    col = _ar1_col(n, r=0.7)
    C = _dense(col)
    x = jnp.asarray(inverse_generator(col))
    v = np.random.default_rng(6).standard_normal(n)
    vj = jnp.asarray(v)

    f = jax.jit(inverse_matvec)
    got = np.asarray(f(x, vj))
    assert np.allclose(got, np.linalg.solve(C, v), atol=1e-8, rtol=1e-8)

    t_gs = _median_time(f, x, vj)
    # time a dense solve (from scratch, as the full time-domain likelihood would do)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        np.linalg.solve(C, v)
        ts.append(time.perf_counter() - t0)
    t_dense = float(np.median(ts))
    speedup = t_dense / t_gs
    print(
        f"\n[toeplitz] N={n}: Gohberg-Semencul C^-1 v={t_gs * 1e3:.3f} ms, "
        f"dense solve={t_dense * 1e3:.3f} ms, speedup={speedup:.1f}x"
    )
    assert speedup > 5.0, f"expected G-S >> dense, got {speedup:.2f}x"
