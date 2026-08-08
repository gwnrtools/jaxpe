r"""Standalone, PSD-weighted match/mismatch between two frequency-domain waveforms.

The noise-weighted inner product, matching the normalization already used throughout
``jaxpe.gw.likelihood`` (e.g. ``fd_marginal.py``'s ``Z = 4 df sum(conj(data) h / psd)``)::

    <a|b> = 4 df sum_f Re[ a(f) conj(b(f)) / psd(f) ]

``match`` is the normalized overlap ``<a|b> / sqrt(<a|a><b|b>)`` (1.0 = identical up to
overall amplitude/phase within the summed band); ``mismatch`` is ``1 - match``. Unlike an
elementwise amplitude/phase comparison, this is the measure that actually matters for two
waveforms meant to represent the same physical signal -- XLA fuses a given model differently
across jit graphs (see ``tests/test_gw.py::test_batched_matches_serial``'s docstring), so raw
values can differ at the 1e-5 level while being fully indistinguishable to inference.

Pure JAX throughout (no numpy, no host round-trips), so this composes under ``jax.jit``/
``jax.vmap`` itself -- e.g. a batched calibration sweep over many parameter points, not just a
one-off test assertion. ``band`` (if given) must be a boolean/float array the same shape as
``a``/``b`` -- a static-shape mask multiplied into the integrand, not dynamic fancy-indexing,
so the whole thing stays traceable.
"""

import jax.numpy as jnp


def inner_product(
    a: jnp.ndarray, b: jnp.ndarray, psd: jnp.ndarray, df: float, band=None
):
    """``4 df sum_f Re[a(f) conj(b(f)) / psd(f)]``, optionally restricted to ``band``."""
    integrand = (a * jnp.conj(b) / psd).real
    if band is not None:
        integrand = jnp.where(band, integrand, 0.0)
    return 4.0 * df * jnp.sum(integrand)


def match(a: jnp.ndarray, b: jnp.ndarray, psd: jnp.ndarray, df: float, band=None):
    """Normalized noise-weighted overlap ``<a|b> / sqrt(<a|a><b|b>)``."""
    aa = inner_product(a, a, psd, df, band)
    bb = inner_product(b, b, psd, df, band)
    ab = inner_product(a, b, psd, df, band)
    return ab / jnp.sqrt(aa * bb)


def mismatch(a: jnp.ndarray, b: jnp.ndarray, psd: jnp.ndarray, df: float, band=None):
    """``1 - match(a, b, psd, df, band)``."""
    return 1.0 - match(a, b, psd, df, band)
