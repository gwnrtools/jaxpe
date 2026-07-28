"""Tests for time-domain relative binning (relative_binning_td.py, Phase RB-4).

The heterodyned time-domain likelihood is a controlled approximation of the exact dense
likelihood -1/2 (d - s)^T C^-1 (d - s) with a non-diagonal Toeplitz covariance. Every
accuracy test compares it against ``td_dense_loglikelihood`` built from the same modes
and covariance; the strongest check is the identity at the fiducial (ratio == 1). A
synthetic complex chirp stands in for the (2,2) waveform mode so the whole chain is
exactly controllable. The performance test confirms the binned evaluation is much faster
than the dense solve (compile time excluded).
"""

import time

import numpy as np

import jax
import jax.numpy as jnp

from jaxpe.gw.likelihood.relative_binning_td import (
    RelativeBinningTDLikelihood,
    td_dense_loglikelihood,
    time_bin_edges,
)


def _chirp_mode(times, mc, k=90.0):
    """Synthetic complex (2,2)-like mode: smooth envelope x chirp phase.

    The envelope cancels in the ratio u(mc)/u(mc0), leaving a smooth phase difference --
    exactly the heterodyning assumption. ``mc`` plays the role of the intrinsic parameter.
    """
    n = times.shape[0]
    x = np.linspace(0.0, 1.0, n)
    env = np.sin(np.pi * x) ** 2  # smooth, zero at both ends
    tau = (times[-1] + 0.1) - times  # positive, decreasing toward the end
    phase = -k * tau**0.625 * mc ** (-0.625)
    return env * np.exp(1j * phase)


def _ar1_acf(n, r=0.35, sigma2=1.0):
    return sigma2 * r ** np.arange(n)


P0 = 1.3 - 0.4j  # fiducial extrinsic coefficient
MC0 = 1.0


def _setup(n=1024, fs=1024.0, noise_seed=None):
    times = np.arange(n) / fs
    u0 = _chirp_mode(times, MC0)
    acf = _ar1_acf(n, sigma2=2.5e-3)  # scale sets the SNR
    data = np.real(P0 * u0)
    if noise_seed is not None:
        # colored noise consistent with the covariance C = chol L L^T
        rng = np.random.default_rng(noise_seed)
        C = acf[np.abs(np.subtract.outer(np.arange(n), np.arange(n)))]
        L = np.linalg.cholesky(C)
        data = data + L @ rng.standard_normal(n)
    return times, u0, acf, data


def test_time_bin_edges_are_increasing_and_bounded():
    times, u0, acf, data = _setup()
    edges = time_bin_edges(u0, phase_per_bin=1.0)
    assert np.all(np.diff(edges) > 0)
    assert edges[0] >= 0 and edges[-1] <= len(u0) - 1
    assert (edges.size - 1) < 0.5 * len(u0)  # genuine downsampling


def test_exact_at_fiducial():
    """Ratio == 1 at the fiducial: heterodyned == dense, and ~0 for a zero-noise signal."""
    times, u0, acf, data = _setup()
    like = RelativeBinningTDLikelihood(u0, times, data, acf, phase_per_bin=1.0)
    edges = like.edge_indices
    binned = float(like.log_likelihood(jnp.asarray(u0[edges]), P0))
    dense = td_dense_loglikelihood(u0, P0, data, acf)
    assert abs(binned - dense) < 1e-6, (binned, dense)
    assert abs(dense) < 1e-4  # zero-noise: dense lnL is 0 at the fiducial


def _beta_tol(exact, beta=0.05):
    return beta * (1.0 + abs(exact))


def test_parity_vs_dense_intrinsic():
    """Vary the intrinsic parameter (chirp) near the fiducial; binned must track dense."""
    times, u0, acf, data = _setup()
    like = RelativeBinningTDLikelihood(u0, times, data, acf, phase_per_bin=1.0)
    edges = like.edge_indices
    worst = -np.inf
    for dmc in (-0.03, -0.01, 0.0, 0.01, 0.03):
        mc = MC0 + dmc
        u = _chirp_mode(times, mc)
        binned = float(like.log_likelihood(jnp.asarray(u[edges]), P0))
        dense = td_dense_loglikelihood(u, P0, data, acf)
        worst = max(worst, abs(binned - dense) - _beta_tol(dense))
    assert worst < 0.0, worst


def test_parity_vs_dense_extrinsic():
    """Vary the extrinsic coefficient p (intrinsic fixed); binned must match dense."""
    times, u0, acf, data = _setup()
    like = RelativeBinningTDLikelihood(u0, times, data, acf, phase_per_bin=1.0)
    edges = like.edge_indices
    for p in (P0, 1.0 + 0.0j, 0.7 - 0.9j, 1.6 + 0.2j):
        binned = float(like.log_likelihood(jnp.asarray(u0[edges]), p))
        dense = td_dense_loglikelihood(u0, p, data, acf)
        assert abs(binned - dense) < _beta_tol(dense), (p, binned, dense)


def test_parity_with_noise():
    """With a genuine colored-noise realization (nonzero residual) the heterodyned and
    dense likelihoods still agree near the fiducial."""
    times, u0, acf, data = _setup(noise_seed=7)
    like = RelativeBinningTDLikelihood(u0, times, data, acf, phase_per_bin=1.0)
    edges = like.edge_indices
    for dmc in (-0.02, 0.0, 0.02):
        u = _chirp_mode(times, MC0 + dmc)
        binned = float(like.log_likelihood(jnp.asarray(u[edges]), P0))
        dense = td_dense_loglikelihood(u, P0, data, acf)
        assert abs(binned - dense) < _beta_tol(dense) + 0.05, (dmc, binned, dense)


def test_finer_bins_reduce_error():
    times, u0, acf, data = _setup()
    u = _chirp_mode(times, MC0 + 0.06)  # a bit further out, so binning error is visible
    dense = td_dense_loglikelihood(u, P0, data, acf)
    coarse = RelativeBinningTDLikelihood(u0, times, data, acf, phase_per_bin=3.0)
    fine = RelativeBinningTDLikelihood(u0, times, data, acf, phase_per_bin=0.3)
    err_coarse = abs(float(coarse.log_likelihood(jnp.asarray(u[coarse.edge_indices]), P0)) - dense)
    err_fine = abs(float(fine.log_likelihood(jnp.asarray(u[fine.edge_indices]), P0)) - dense)
    assert fine.n_bins > coarse.n_bins
    assert err_fine <= err_coarse + 1e-9


def test_jittable():
    times, u0, acf, data = _setup()
    like = RelativeBinningTDLikelihood(u0, times, data, acf, phase_per_bin=1.0)
    u = _chirp_mode(times, MC0 + 0.01)
    ue = jnp.asarray(u[like.edge_indices])
    f = jax.jit(like.log_likelihood)
    assert abs(float(f(ue, P0)) - float(like.log_likelihood(ue, P0))) < 1e-9


# --------------------------------------------------------------------- performance

def test_td_relative_binning_faster_than_dense():
    """Heterodyned lnL (bin-edge modes + precomputed summary data) is much faster than
    the dense time-domain likelihood, which re-solves C^-1 every evaluation."""
    n = 4096
    times = np.arange(n) / 2048.0
    u0 = _chirp_mode(times, MC0)
    acf = _ar1_acf(n, r=0.4, sigma2=2.5e-3)
    data = np.real(P0 * u0)
    like = RelativeBinningTDLikelihood(u0, times, data, acf, phase_per_bin=1.0)
    edges = like.edge_indices
    assert like.n_bins < 0.2 * n

    u = _chirp_mode(times, MC0 + 0.01)
    ue = jnp.asarray(u[edges])
    f = jax.jit(like.log_likelihood)

    v_bin = float(f(ue, P0))
    v_dense = td_dense_loglikelihood(u, P0, data, acf)
    assert abs(v_bin - v_dense) < _beta_tol(v_dense), (v_bin, v_dense)

    jax.block_until_ready(f(ue, P0))  # warm up (compile EXCLUDED)
    ts = []
    for _ in range(40):
        t0 = time.perf_counter()
        jax.block_until_ready(f(ue, P0))
        ts.append(time.perf_counter() - t0)
    t_bin = float(np.median(ts))
    td = []
    for _ in range(5):
        t0 = time.perf_counter()
        td_dense_loglikelihood(u, P0, data, acf)
        td.append(time.perf_counter() - t0)
    t_dense = float(np.median(td))
    speedup = t_dense / t_bin
    print(
        f"\n[TD relative binning] N={n}, bins={like.n_bins}: "
        f"dense={t_dense * 1e3:.3f} ms, binned={t_bin * 1e3:.3f} ms, speedup={speedup:.1f}x"
    )
    assert speedup > 2.0, f"expected binned >> dense, got {speedup:.2f}x"
