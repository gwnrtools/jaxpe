"""Tests for Fourier-domain relative binning (relative_binning_fd.py).

Correctness strategy: the relative-binning likelihood is a controlled approximation of
the *exact* full-resolution FD Whittle likelihood already in jaxpe
(``FDNetworkLikelihood``), so every accuracy test compares the two directly on the same
data. The strongest check is the identity at the fiducial point, where the trial/fiducial
ratio is exactly 1 and the two likelihoods must agree to machine precision.

Performance: ``test_relative_binning_much_faster`` verifies both that the binned and full
likelihoods agree numerically AND that the binned evaluation is much faster once compiled
(JAX compile time is excluded by warming up before timing).
"""

import time

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jaxpe.gw import IMRPhenomD, make_injection
from jaxpe.gw.likelihood import RelativeBinningFDLikelihood, frequency_bin_edges

F_LOWER = 20.0
T_C = 1126259462.4

INJECTION = dict(
    chirp_mass=25.0,
    mass_ratio=0.8,
    spin1z=0.2,
    spin2z=-0.1,
    luminosity_distance=2000.0,  # moderate network SNR (~25)
    inclination=0.4,
    phase=1.5,
    geocent_time=T_C,
    ra=1.2,
    dec=0.5,
    psi=0.8,
)


def _make_like(duration=4.0, sampling_rate=1024.0, f_max=None, noise_seed=None):
    waveform = IMRPhenomD(f_ref=F_LOWER)
    return make_injection(
        waveform,
        INJECTION,
        detector_names=("H1", "L1"),
        duration=duration,
        sampling_rate=sampling_rate,
        f_min=F_LOWER,
        f_max=f_max,
        noise_seed=noise_seed,
    )


@pytest.fixture(scope="module")
def full_like():
    """Exact full-resolution FD likelihood (the reference), zero-noise injection."""
    return _make_like()


@pytest.fixture(scope="module")
def rb_like(full_like):
    """Relative-binning likelihood with the fiducial at the injection."""
    return RelativeBinningFDLikelihood.from_likelihood(full_like, INJECTION)


def _params(**overrides):
    p = {**INJECTION, **overrides}
    return {k: jnp.asarray(float(v)) for k, v in p.items()}


# --------------------------------------------------------------------- bin scheme


def test_bin_edges_tile_band_monotonically():
    like = _make_like()
    freqs = np.asarray(like.freqs)
    edges = frequency_bin_edges(freqs, like.f_min, like.f_max)
    # strictly increasing indices
    assert np.all(np.diff(edges) > 0)
    # confined to the band and spanning it end to end (so bins tile [f_min, f_max])
    assert freqs[edges[0]] >= like.f_min
    assert freqs[edges[-1]] <= like.f_max
    band = np.nonzero((freqs >= like.f_min) & (freqs <= like.f_max))[0]
    assert edges[0] == band[0] and edges[-1] == band[-1]
    # far fewer bins than full-resolution band points -- the whole point
    assert (edges.size - 1) < 0.5 * band.size


def test_bin_count_increases_with_resolution():
    like = _make_like()
    freqs = np.asarray(like.freqs)
    n_coarse = frequency_bin_edges(freqs, like.f_min, like.f_max, epsilon=1.0).size
    n_fine = frequency_bin_edges(freqs, like.f_min, like.f_max, epsilon=0.25).size
    n_high_chi = frequency_bin_edges(freqs, like.f_min, like.f_max, chi=2.0).size
    assert n_fine > n_coarse  # smaller epsilon -> more bins
    assert n_high_chi > frequency_bin_edges(freqs, like.f_min, like.f_max).size


def test_summary_data_shapes(rb_like):
    n = rb_like.n_bins
    assert n >= 2
    st = rb_like._static()
    assert st["rb_edge_freqs"].shape == (n + 1,)
    for det in rb_like.detectors:
        for key in ("rb_A0", "rb_A1", "rb_B0", "rb_B1"):
            assert st[key][det.name].shape == (n,)
        assert st["rb_h0_edges"][det.name].shape == (n + 1,)


# --------------------------------------------------------------------- correctness


def test_exact_at_fiducial(full_like, rb_like):
    """At the fiducial the ratio is identically 1, so the heterodyned likelihood must
    equal the exact one to machine precision (and be ~0 for a zero-noise injection)."""
    p = _params()
    exact = float(full_like.log_likelihood(p))
    binned = float(rb_like.log_likelihood(p))
    assert abs(binned - exact) < 1e-6, (binned, exact)
    assert abs(exact) < 1e-4  # zero-noise: full lnL is 0 at the injected truth


# Relative binning's error scales as ~ beta * |lnL - lnL_max| (Zackay et al.): the
# discrepancy grows with distance from the fiducial in *likelihood* units, so the
# principled parity tolerance is proportional to (1 + |lnL|), not a fixed absolute.
def _beta_tol(exact, beta=0.05):
    return beta * (1.0 + abs(exact))


@pytest.mark.parametrize("dmc", [-0.2, -0.05, 0.0, 0.05, 0.2])
def test_parity_vs_full_near_fiducial(full_like, rb_like, dmc):
    """Binned lnL tracks the exact lnL across a chirp-mass slice around the fiducial,
    to the relative-binning error model beta*(1+|lnL|)."""
    p = _params(chirp_mass=INJECTION["chirp_mass"] + dmc)
    exact = float(full_like.log_likelihood(p))
    binned = float(rb_like.log_likelihood(p))
    assert abs(binned - exact) < _beta_tol(exact), (dmc, binned, exact)


def test_parity_over_random_draws(full_like, rb_like):
    """Perturb several parameters jointly (intrinsic + extrinsic) near the fiducial and
    require the binned and exact likelihoods to agree to the beta error model."""
    rng = np.random.default_rng(0)
    worst = -np.inf  # tracks the largest (|error| - tol) margin; must stay negative
    for _ in range(25):
        p = _params(
            chirp_mass=INJECTION["chirp_mass"] + rng.uniform(-0.1, 0.1),
            mass_ratio=np.clip(
                INJECTION["mass_ratio"] + rng.uniform(-0.03, 0.03), 0.1, 1.0
            ),
            spin1z=INJECTION["spin1z"] + rng.uniform(-0.05, 0.05),
            spin2z=INJECTION["spin2z"] + rng.uniform(-0.05, 0.05),
            luminosity_distance=INJECTION["luminosity_distance"]
            * (1.0 + rng.uniform(-0.1, 0.1)),
            geocent_time=INJECTION["geocent_time"] + rng.uniform(-5e-4, 5e-4),
        )
        exact = float(full_like.log_likelihood(p))
        binned = float(rb_like.log_likelihood(p))
        worst = max(worst, abs(binned - exact) - _beta_tol(exact))
    assert worst < 0.0, worst


def test_binned_likelihood_is_jittable(full_like, rb_like):
    p = _params(chirp_mass=25.05)
    f = jax.jit(rb_like.log_likelihood)
    jitted = float(f(p))
    eager = float(rb_like.log_likelihood(p))
    assert abs(jitted - eager) < 1e-9
    assert abs(jitted - float(full_like.log_likelihood(p))) < _beta_tol(eager)


def test_finer_bins_reduce_error(full_like):
    """More bins -> smaller heterodyning error at a fixed off-fiducial point."""
    p = _params(chirp_mass=25.4)
    exact = float(full_like.log_likelihood(p))
    coarse = RelativeBinningFDLikelihood.from_likelihood(
        full_like, INJECTION, epsilon=1.0
    )
    fine = RelativeBinningFDLikelihood.from_likelihood(
        full_like, INJECTION, epsilon=0.1
    )
    err_coarse = abs(float(coarse.log_likelihood(p)) - exact)
    err_fine = abs(float(fine.log_likelihood(p)) - exact)
    assert fine.n_bins > coarse.n_bins
    assert err_fine <= err_coarse + 1e-9


# --------------------------------------------------------------------- guards


def test_missing_fiducial_raises(full_like):
    with pytest.raises(ValueError, match="fiducial_params is required"):
        RelativeBinningFDLikelihood.from_likelihood(full_like, None)


def test_non_fd_waveform_raises(full_like):
    class NotFD:
        is_fd = False

    with pytest.raises(ValueError, match="frequency-domain"):
        RelativeBinningFDLikelihood(
            waveform=NotFD(),
            detectors=full_like.detectors,
            data_fd=full_like.data_fd,
            psds=full_like.psds,
            freqs=full_like.freqs,
            times=full_like.times,
            f_min=full_like.f_min,
            f_max=full_like.f_max,
            gmst_ref=full_like.gmst_ref,
            t_ref=full_like.t_ref,
            fiducial_params=INJECTION,
        )


# --------------------------------------------------------------------- performance


def _median_eval_time(fn, p, n=60):
    """Median wall-clock per evaluation, after compilation, forcing device sync."""
    jax.block_until_ready(fn(p))  # warm up: compile + first exec (EXCLUDED from timing)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(p))
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def test_relative_binning_much_faster():
    """Identical likelihood, computed with and without relative binning: the two must
    match numerically, and the binned version must be much faster once compiled."""
    # long, finely sampled segment (band kept within IMRPhenomD's support): many
    # frequency points, so heterodyning's downsampling advantage is unambiguous.
    full = _make_like(duration=8.0, sampling_rate=2048.0, f_max=700.0)
    rb = RelativeBinningFDLikelihood.from_likelihood(full, INJECTION)

    n_band = int(
        np.sum(
            (np.asarray(full.freqs) >= full.f_min)
            & (np.asarray(full.freqs) <= full.f_max)
        )
    )
    assert rb.n_bins < 0.1 * n_band  # binning genuinely downsamples the band

    f_full = jax.jit(full.log_likelihood)
    f_rb = jax.jit(rb.log_likelihood)

    p = _params(chirp_mass=25.05)  # slightly off the fiducial: a realistic trial point
    v_full, v_rb = float(f_full(p)), float(f_rb(p))
    assert abs(v_full - v_rb) < _beta_tol(v_full), (v_full, v_rb)

    t_full = _median_eval_time(f_full, p)
    t_rb = _median_eval_time(f_rb, p)
    speedup = t_full / t_rb
    print(
        f"\n[relative binning] band points={n_band}, bins={rb.n_bins}: "
        f"full={t_full * 1e3:.3f} ms, binned={t_rb * 1e3:.3f} ms, speedup={speedup:.1f}x"
    )
    assert speedup > 2.0, f"expected binned >> full, got {speedup:.2f}x"


def test_support_restriction_warns_and_stays_exact():
    """When the analysis band runs past the fiducial's ringdown cutoff, construction
    warns, yet the likelihood is still exact at the fiducial (<d|d> spans the full band
    while bins cover only the supported sub-band)."""
    full = _make_like(duration=8.0, sampling_rate=2048.0)  # f_max ~ 922 Hz > cutoff
    with pytest.warns(UserWarning, match="supported only"):
        rb = RelativeBinningFDLikelihood.from_likelihood(full, INJECTION)
    p = _params()
    assert abs(float(rb.log_likelihood(p)) - float(full.log_likelihood(p))) < 1e-6


# ----------------------------------------------------- higher-mode FD (synthetic modes)

from jaxpe.gw.likelihood.relative_binning_fd import (  # noqa: E402
    RelativeBinningFDLikelihoodHM,
    fd_dense_loglikelihood_modes,
)

_FD_FREQS = np.fft.rfftfreq(2048, d=1.0 / 1024.0)  # df = 0.5 Hz, up to 512 Hz
_FD_FMIN, _FD_FMAX = 20.0, 460.0
_FD_PSD = 1.0 + (30.0 / np.clip(_FD_FREQS, _FD_FREQS[1], None)) ** 4  # red-tilted
_C_HM = np.array([1.2 - 0.3j, 0.5 + 0.4j])


def _fd_modes(freqs, mc, k=7000.0):
    fs = np.clip(freqs, freqs[1], None)
    amp = fs ** (-7.0 / 6.0)
    amp = amp / amp.max()
    psi = -k * mc ** (-5.0 / 3.0) * fs ** (-5.0 / 3.0)
    return {(2, 2): amp * np.exp(1j * psi), (3, 3): 0.3 * amp * np.exp(1.5j * psi)}


def _fd_setup():
    modes0 = _fd_modes(_FD_FREQS, 25.0)
    stack0 = np.stack([modes0[k] for k in modes0])
    data = (_C_HM[:, None] * stack0).sum(0)
    return modes0, data


def _fd_edges_stack(modes, keys, edges):
    return np.stack([modes[k][edges] for k in keys])


def test_fd_hm_exact_at_fiducial():
    modes0, data = _fd_setup()
    like = RelativeBinningFDLikelihoodHM(
        modes0, _FD_FREQS, data, _FD_PSD, _FD_FMIN, _FD_FMAX, phase_per_bin=1.0
    )
    trial = _fd_edges_stack(modes0, like.mode_keys, like.edge_indices)
    binned = float(like.log_likelihood(jnp.asarray(trial), _C_HM))
    dense = fd_dense_loglikelihood_modes(
        np.stack([modes0[k] for k in like.mode_keys]),
        _C_HM,
        data,
        _FD_PSD,
        _FD_FREQS,
        _FD_FMIN,
        _FD_FMAX,
    )
    assert abs(binned - dense) < 1e-6, (binned, dense)
    assert abs(dense) < 1e-4


def test_fd_hm_parity_intrinsic():
    modes0, data = _fd_setup()
    like = RelativeBinningFDLikelihoodHM(
        modes0, _FD_FREQS, data, _FD_PSD, _FD_FMIN, _FD_FMAX, phase_per_bin=1.0
    )
    worst = -np.inf
    for dmc in (-0.3, -0.1, 0.0, 0.1, 0.3):
        modes = _fd_modes(_FD_FREQS, 25.0 + dmc)
        trial = _fd_edges_stack(modes, like.mode_keys, like.edge_indices)
        binned = float(like.log_likelihood(jnp.asarray(trial), _C_HM))
        dense = fd_dense_loglikelihood_modes(
            np.stack([modes[k] for k in like.mode_keys]),
            _C_HM,
            data,
            _FD_PSD,
            _FD_FREQS,
            _FD_FMIN,
            _FD_FMAX,
        )
        worst = max(worst, abs(binned - dense) - _beta_tol(dense))
    assert worst < 0.0, worst


def test_fd_hm_parity_extrinsic():
    modes0, data = _fd_setup()
    like = RelativeBinningFDLikelihoodHM(
        modes0, _FD_FREQS, data, _FD_PSD, _FD_FMIN, _FD_FMAX, phase_per_bin=1.0
    )
    trial = _fd_edges_stack(modes0, like.mode_keys, like.edge_indices)
    stack0 = np.stack([modes0[k] for k in like.mode_keys])
    for c in (
        _C_HM,
        np.array([1.0 + 0j, 0.0 + 0j]),
        np.array([0.8 - 0.5j, 1.1 + 0.6j]),
    ):
        binned = float(like.log_likelihood(jnp.asarray(trial), c))
        dense = fd_dense_loglikelihood_modes(
            stack0, c, data, _FD_PSD, _FD_FREQS, _FD_FMIN, _FD_FMAX
        )
        assert abs(binned - dense) < _beta_tol(dense), (c, binned, dense)
