"""Ground-based detector geometry: antenna patterns and geocentric time delays.

Locations and response tensors are the LAL cached values (validated against
``lal.CachedDetectors`` in the test suite). All functions are JAX-traceable and
differentiable in (ra, dec, psi, gmst); GMST itself is handled linearly around a
reference epoch by the likelihood layer, which is accurate to microradians over the
sub-second coalescence-time priors used in PE.
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

C_SI = 299792458.0
EARTH_OMEGA = 7.292115855377074e-5  # rad/s, sidereal rotation rate


class Detector(NamedTuple):
    name: str
    location: np.ndarray  # (3,) meters, Earth-fixed frame
    response: np.ndarray  # (3, 3) detector tensor D = (xx^T - yy^T)/2


# LAL cached detector geometry (meters / dimensionless)
H1 = Detector(
    "H1",
    np.array([-2.16141492636e06, -3.83469517889e06, 4.60035022664e06]),
    np.array(
        [
            [-0.3926141, -0.0776130, -0.2473886],
            [-0.0776130, 0.3195244, 0.2279981],
            [-0.2473886, 0.2279981, 0.0730903],
        ]
    ),
)
L1 = Detector(
    "L1",
    np.array([-7.42760447238e04, -5.49628371971e06, 3.22425701744e06]),
    np.array(
        [
            [0.4112809, 0.1402097, 0.2472943],
            [0.1402097, -0.1090056, -0.1816157],
            [0.2472943, -0.1816157, -0.3022755],
        ]
    ),
)
V1 = Detector(
    "V1",
    np.array([4.54637409900e06, 8.42989697626e05, 4.37857696241e06]),
    np.array(
        [
            [0.2438740, -0.0990838, -0.2325762],
            [-0.0990838, -0.4478258, 0.1878331],
            [-0.2325762, 0.1878331, 0.2039518],
        ]
    ),
)

DETECTORS = {"H1": H1, "L1": L1, "V1": V1}


def _wave_frame(ra, dec, psi, gmst):
    """Polarization basis vectors (x, y) of the wave frame in Earth-fixed coordinates."""
    gha = gmst - ra  # Greenwich hour angle
    cosgha, singha = jnp.cos(gha), jnp.sin(gha)
    cosdec, sindec = jnp.cos(dec), jnp.sin(dec)
    cospsi, sinpsi = jnp.cos(psi), jnp.sin(psi)

    x = jnp.stack(
        [
            -cospsi * singha - sinpsi * cosgha * sindec,
            -cospsi * cosgha + sinpsi * singha * sindec,
            sinpsi * cosdec,
        ]
    )
    y = jnp.stack(
        [
            sinpsi * singha - cospsi * cosgha * sindec,
            sinpsi * cosgha + cospsi * singha * sindec,
            cospsi * cosdec,
        ]
    )
    return x, y


def antenna_pattern(det: Detector, ra, dec, psi, gmst):
    """(F+, Fx) response of ``det`` for a source at (ra, dec) with polarization angle psi."""
    D = jnp.asarray(det.response)
    x, y = _wave_frame(ra, dec, psi, gmst)
    f_plus = x @ D @ x - y @ D @ y
    f_cross = x @ D @ y + y @ D @ x
    return f_plus, f_cross


def time_delay_from_geocenter(det: Detector, ra, dec, gmst):
    """Arrival-time delay (seconds) at ``det`` relative to the geocenter."""
    gha = gmst - ra
    e_src = jnp.stack(
        [
            jnp.cos(dec) * jnp.cos(gha),
            -jnp.cos(dec) * jnp.sin(gha),
            jnp.sin(dec),
        ]
    )
    return -jnp.dot(jnp.asarray(det.location), e_src) / C_SI


def gmst_from_gps(gps_time: float) -> float:
    """Greenwich mean sidereal time (radians) — host-side reference computation.

    Uses lal when available (exact), otherwise a linear approximation good to ~1e-4 rad
    over years around 2020.
    """
    try:
        import lal

        return lal.GreenwichMeanSiderealTime(float(gps_time)) % (2 * np.pi)
    except ImportError:
        gps_2000 = 630720013.0  # 2000-01-01 00:00:00 UTC in GPS seconds
        gmst_2000 = 1.74476716333061  # rad
        return (gmst_2000 + EARTH_OMEGA * (float(gps_time) - gps_2000)) % (2 * np.pi)
