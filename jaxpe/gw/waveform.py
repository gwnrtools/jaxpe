"""Waveform interface and a built-in PN-inspired time-domain toy chirp.

The engine-facing contract is ``WaveformModel``: a callable

    (params: dict[str, scalar], times: (n,) array of geocentric GPS seconds)
        -> (h_plus, h_cross) each of shape (n,)

that is JAX-traceable and differentiable in the parameters.

Motivation & Math
-----------------
In the linear regime of General Relativity, the metric perturbation $h_{\mu\nu}$ far from 
the source can be decomposed into two transverse-traceless (TT) polarization states, 
$h_+$ and $h_\\times$. For a binary system in the inspiral phase, the dominant radiation 
arises from the time-varying mass quadrupole moment $I_{ij}$. To leading order, the 
strain at a distance $d$ is:
$$ h_{ij}^{\\text{TT}} = \\frac{2G}{c^4 d} \\ddot{I}_{ij}^{\\text{TT}}(t - d/c) $$

For a binary of component masses $m_1$ and $m_2$ (total mass $M=m_1+m_2$, symmetric 
mass ratio $\\eta = m_1 m_2 / M^2$), the orbital phase evolution is governed by the 
loss of energy and angular momentum to gravitational radiation. The Post-Newtonian (PN) 
expansion characterizes this evolution as an asymptotic series in $v/c$.

The ``ToyChirp`` model provided here implements a truncated PN frequency and phase 
evolution up to the Innermost Stable Circular Orbit (ISCO). While not a production-level 
LALSimulation approximant, it rigorously exposes the non-linear relationship between the 
masses $(\\mathcal{M}_c, q)$, the phase $\\phi_c$, and the resulting wave morphology, 
serving as a robust pedagogical sandbox for parameter estimation.

Parameter names follow standard conventions:
``chirp_mass`` [Msun], ``mass_ratio`` ($q \le 1$), ``luminosity_distance`` [Mpc],
``inclination`` ($\\iota$), ``phase`` ($\\phi_c$), ``geocent_time`` ($t_c$).
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

MTSUN_SI = 4.925491025543576e-06  # solar mass in seconds (GMsun/c^3)
MPC_SI = 3.0856775814913673e22  # Mpc in meters
C_SI = 299792458.0
MRSUN_SI = 1.4766250385053816e3  # solar mass in meters (GMsun/c^2)

WaveformModel = Callable[[dict, jax.Array], tuple[jax.Array, jax.Array]]


class ToyChirp:
    """PN-inspired inspiral-only quadrupole chirp, terminated near ISCO.

    Parameters
    ----------
    f_start
        Frequency below which the signal is smoothly tapered on (avoids a hard
        turn-on edge in the FFT).
    """

    def __init__(self, f_start: float = 20.0):
        self.f_start = f_start

    def __call__(self, params: dict, times: jax.Array):
        mc = params["chirp_mass"] * MTSUN_SI
        q = params["mass_ratio"]
        eta = q / (1.0 + q) ** 2
        m_total = mc / eta**0.6
        d = params["luminosity_distance"] * MPC_SI / C_SI  # seconds
        iota = params["inclination"]
        phi_c = params["phase"]
        t_c = params["geocent_time"]

        tau = t_c - times
        # dimensionless PN time variable; clip for evaluation safety where tau <= 0
        theta_raw = eta * tau / (5.0 * m_total)
        theta = jnp.maximum(theta_raw, 1e-9)

        # PN-flavoured frequency evolution x(theta) ~ (v/c)^2 and phase (truncated series)
        th_m14 = theta**-0.25
        x = 0.25 * th_m14**2 * (1.0 + (743.0 / 4032.0 + 11.0 * eta / 48.0) * th_m14**2)
        phase_orb = phi_c - (1.0 / eta) * (
            theta**0.625
            + (3715.0 / 8064.0 + 55.0 * eta / 96.0) * theta**0.375
            - (3.0 * jnp.pi / 4.0) * theta**0.25
        )

        f_gw = x**1.5 / (jnp.pi * m_total)  # dominant-mode GW frequency

        # amplitude: leading-order quadrupole
        amp = 4.0 * eta * m_total * x / d
        c_i = jnp.cos(iota)
        h_plus = amp * 0.5 * (1.0 + c_i**2) * jnp.cos(2.0 * phase_orb)
        h_cross = amp * c_i * jnp.sin(2.0 * phase_orb)

        # validity mask: after t_c or beyond ISCO -> zero; smooth turn-on below f_start
        x_isco = 1.0 / 6.0
        alive = (theta_raw > 1e-9) & (x < x_isco)
        turn_on = 0.5 * (1.0 + jnp.tanh(4.0 * (f_gw - self.f_start) / self.f_start))
        w = jnp.where(alive, turn_on, 0.0)
        return w * h_plus, w * h_cross


def mismatch_f32_f64(waveform: WaveformModel, params: dict, times) -> float:
    """Flat-noise mismatch between float32 and float64 evaluations of ``waveform``.

    A quick certification that the float32 fast path is safe for a given model and
    parameter point: values well below ~1e-4 mean float32 waveform error is far inside
    the statistical uncertainty of typical CBC posteriors.

    Times are re-referenced to the segment start before the float32 evaluation —
    absolute GPS epochs (~1e9 s) are unrepresentable at float32 resolution (~64 s), so
    any float32 fast path must work in segment-relative time exactly like this.
    """
    times64 = jnp.asarray(times, jnp.float64)
    p64 = {k: jnp.asarray(v, jnp.float64) for k, v in params.items()}
    hp64, _ = waveform(p64, times64)

    epoch = times64[0]
    p32 = {k: jnp.asarray(v, jnp.float32) for k, v in params.items()}
    p32["geocent_time"] = jnp.asarray(p64["geocent_time"] - epoch, jnp.float32)
    hp32, _ = waveform(p32, jnp.asarray(times64 - epoch, jnp.float32))

    a = hp64
    b = jnp.asarray(hp32, jnp.float64)
    overlap = jnp.sum(a * b) / jnp.sqrt(jnp.sum(a * a) * jnp.sum(b * b))
    return float(1.0 - overlap)
