"""Unit tests for jaxpe.gw.match -- the standalone PSD-weighted match/mismatch used by the
LALSuite-comparison tests (test_phenomd.py, test_phenomt.py, ...)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpe.gw import inner_product, match, mismatch


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _random_fd(rng, n=64):
    return jnp.asarray(rng.normal(size=n) + 1j * rng.normal(size=n))


def test_mismatch_identical_is_zero(rng):
    a = _random_fd(rng)
    psd = jnp.ones(a.shape[0])
    assert float(mismatch(a, a, psd, df=1.0)) == pytest.approx(0.0, abs=1e-12)
    assert float(match(a, a, psd, df=1.0)) == pytest.approx(1.0, abs=1e-12)


def test_mismatch_invariant_under_positive_real_scaling(rng):
    """match(a, b) is unaffected by an overall positive real amplitude rescaling -- the
    normalization is exactly what makes it insensitive to the raw-amplitude differences
    that XLA's per-jit-graph fusion introduces (see module docstring)."""
    a = _random_fd(rng)
    psd = jnp.ones(a.shape[0])
    for scale in (2.0, 0.1, 1e-3):
        b = scale * a
        assert float(mismatch(a, b, psd, df=1.0)) == pytest.approx(0.0, abs=1e-10)


def test_mismatch_is_phase_sensitive(rng):
    """Unlike a phase-maximized match, this uses Re[<a|b>] (matching fd_marginal.py's own
    inner-product convention) -- a relative phase rotation must show up as nonzero
    mismatch, since two independent implementations should agree on absolute phase, not
    just waveform shape up to an arbitrary phase."""
    a = _random_fd(rng)
    psd = jnp.ones(a.shape[0])
    b = jnp.exp(1j * 0.7) * a
    assert float(mismatch(a, b, psd, df=1.0)) > 1e-3


def test_mismatch_of_orthogonal_signals_is_one():
    n = 4
    psd = jnp.ones(n)
    a = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.complex64)
    b = jnp.asarray([0.0, 1.0, 0.0, 0.0], dtype=jnp.complex64)
    assert float(mismatch(a, b, psd, df=1.0)) == pytest.approx(1.0, abs=1e-6)


def test_band_mask_restricts_the_sum(rng):
    """A band that excludes everywhere two signals differ must report zero mismatch."""
    a = _random_fd(rng, n=32)
    b = a.at[16:].set(_random_fd(rng, n=16))  # differs only in the second half
    psd = jnp.ones(a.shape[0])
    band = jnp.arange(32) < 16
    assert float(mismatch(a, b, psd, df=1.0, band=band)) == pytest.approx(
        0.0, abs=1e-10
    )
    # without the band restriction, the differing half must show up
    assert float(mismatch(a, b, psd, df=1.0)) > 1e-3


def test_inner_product_matches_manual_computation(rng):
    a, b = _random_fd(rng), _random_fd(rng)
    psd = jnp.asarray(rng.uniform(0.5, 2.0, size=a.shape[0]))
    df = 0.25
    got = float(inner_product(a, b, psd, df))
    want = float(
        4.0
        * df
        * np.sum((np.asarray(a) * np.conj(np.asarray(b)) / np.asarray(psd)).real)
    )
    assert got == pytest.approx(want, rel=1e-10)


def test_jit_and_vmap_compatible(rng):
    a = _random_fd(rng, n=16)
    psd = jnp.ones(16)

    jitted = jax.jit(lambda x, y: mismatch(x, y, psd, df=1.0))
    assert float(jitted(a, a)) == pytest.approx(0.0, abs=1e-10)

    batch = jnp.stack([a, 2.0 * a, a + 0.01 * _random_fd(rng, n=16)])
    out = jax.vmap(lambda x: mismatch(x, a, psd, df=1.0))(batch)
    assert out.shape == (3,)
    assert float(out[0]) == pytest.approx(0.0, abs=1e-10)
    assert float(out[1]) == pytest.approx(0.0, abs=1e-10)
