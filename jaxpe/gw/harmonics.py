"""Spin-weighted spherical harmonics, JAX-traceable in the angles.

Implemented via the Wigner-d sum with (l, m, s) static, so each harmonic reduces to a
short polynomial in cos(iota/2), sin(iota/2) — differentiable and vmappable, unlike
lal.SpinWeightedSphericalHarmonic. Validated against LAL in the test suite.
"""

from math import factorial, pi, sqrt

import jax.numpy as jnp


def spin_weighted_ylm(iota, phi, l: int, m: int, s: int = -2):
    """sYlm(iota, phi) with static (l, m, s); traceable/differentiable in the angles.

    Convention matches lal.SpinWeightedSphericalHarmonic:
        sYlm = (-1)^s sqrt((2l+1)/4pi) d^l_{m,-s}(iota) e^{i m phi}.
    """
    mp = -s  # second index of the Wigner-d function
    k_min = max(0, m - mp)
    k_max = min(l + m, l - mp)
    pref = sqrt(factorial(l + m) * factorial(l - m) * factorial(l + mp) * factorial(l - mp))
    c = jnp.cos(iota / 2.0)
    si = jnp.sin(iota / 2.0)
    d = 0.0
    for k in range(k_min, k_max + 1):
        coef = (
            (-1.0) ** k
            * pref
            / (factorial(l + m - k) * factorial(l - mp - k) * factorial(k) * factorial(k - m + mp))
        )
        d = d + coef * c ** (2 * l + m - mp - 2 * k) * si ** (2 * k - m + mp)
    return (-1.0) ** s * sqrt((2 * l + 1) / (4.0 * pi)) * d * jnp.exp(1j * m * phi)
