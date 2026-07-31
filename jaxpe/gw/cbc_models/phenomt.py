import jax
import jax.numpy as jnp
from typing import Tuple, Dict

from .base import TimeDomainModel
from ..waveform import MTSUN_SI as MTSUN, MPC_SI as MPC

# Constants
PI = jnp.pi
EULERGAMMA = 0.577215664901532860606512090082402431
C_SI = 299792458.0

def _safe_sqrt(x):
    safe_x = jnp.where(x > 0, x, 1e-12)
    return jnp.where(x > 0, jnp.sqrt(safe_x), 0.0)

class IMRPhenomT(TimeDomainModel):
    is_fd = False

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref

    def mode_dict(self, params: dict, grid: jax.Array) -> Dict[Tuple[int, int], jax.Array]:
        """
        Returns tapered spherical-harmonic modes {(l, +-m): h_lm(t)}, strain at 1 Mpc.
        Used by the relative binning likelihood.
        """
        mc = jnp.maximum(params["chirp_mass"], 1.0)
        q = jnp.clip(params["mass_ratio"], 0.01, 1.0)
        eta = q / (1.0 + q) ** 2
        eta = jnp.clip(eta, 1e-4, 0.25)
        M = mc / (eta ** (3.0 / 5.0))
        
        s1z = jnp.clip(params.get("spin1z", jnp.zeros(())), -0.99, 0.99)
        s2z = jnp.clip(params.get("spin2z", jnp.zeros(())), -0.99, 0.99)

        t_geom = grid / (M * MTSUN)
        tc = params.get("geocent_time", 0.0)
        t_geom = t_geom - (tc / (M * MTSUN))

        coeffs = self._compute_phenom_coefficients(eta, s1z, s2z)
        
        omega_22 = self._evaluate_frequency(t_geom, eta, s1z, s2z, coeffs)
        amp_22 = self._evaluate_amplitude(t_geom, omega_22, eta, s1z, s2z, coeffs)
        phase_22 = self._evaluate_phase(t_geom, eta, s1z, s2z, coeffs)

        # Amplitude is scaled to 1 Mpc for mode_dict
        amp_factor = (M * MTSUN * C_SI) / MPC
        
        h22 = amp_factor * amp_22 * jnp.exp(-1j * phase_22)
        
        return {
            (2, 2): h22,
            (2, -2): jnp.conj(h22)
        }

    def __call__(self, params: dict, grid: jax.Array) -> Tuple[jax.Array, jax.Array]:
        modes = self.mode_dict(params, grid)
        
        dist = params.get("luminosity_distance", 100.0)
        phi_ref = params.get("phase", 0.0)
        iota = params.get("inclination", 0.0)
        
        dist_scale = 1.0 / dist
        
        # Apply phi_ref and project
        h22 = modes[(2, 2)] * dist_scale * jnp.exp(1j * 2 * phi_ref)
        h2m2 = modes[(2, -2)] * dist_scale * jnp.exp(-1j * 2 * phi_ref)
        
        Y22 = jnp.sqrt(5.0 / (64.0 * PI)) * (1.0 + jnp.cos(iota))**2
        Y2m2 = jnp.sqrt(5.0 / (64.0 * PI)) * (1.0 - jnp.cos(iota))**2
        
        h_comp = h22 * Y22 + h2m2 * Y2m2
        hp = jnp.real(h_comp)
        hc = -jnp.imag(h_comp)
        
        return hp, hc

    def _compute_phenom_coefficients(self, eta, chi1, chi2) -> dict:
        """
        Placeholder for the phenomenological fits which are in the supplementary Mathematica notebook.
        Without these exact polynomial fits (which interpolate numerical relativity data),
        the waveform will compile and run, but will not be physically correct.
        """
        return {
            "t_meco": -100.0,
            "c8": 0.0, "c9": 0.0, "c10": 0.0, "c11": 0.0, "c12": 0.0,
            "d8": 0.0, "d9": 0.0, "d10": 0.0,
            "a0": 0.1, "a1": 0.0, "a2": 0.0, "a3": 0.0, "a4": 0.0,
            "b0": 0.1, "b1": 0.0, "b2": 0.0, "b3": 0.0,
            "c1": 0.0, "c2": 0.01, "c3": 1.0, "c4": 0.0,
            "d1": 0.1, "d2": 0.01, "d3": 0.0, "d4": 0.1,
            "omega_rd": 0.5,
            "alpha_1": 0.05
        }

    def _pn_omega(self, eta, chi1, chi2):
        dm = _safe_sqrt(1.0 - 4.0 * eta)
        m1 = (1.0 + dm) / 2.0
        m2 = (1.0 - dm) / 2.0
        
        w = jnp.zeros(8)
        w = w.at[0].set(1.0)
        w = w.at[1].set(0.0)
        w = w.at[2].set(743.0/2688.0 + 11.0/32.0 * eta)
        w = w.at[3].set(-3.0*PI/10.0 + 113.0/160.0 * (m1*chi1 + m2*chi2) - 19.0/80.0 * eta * (chi1 + chi2))
        
        w4 = (1855099.0/14450688.0 - 243.0*(m1*chi1**2 + m2*chi2**2)/1024.0 
              + 56975.0*eta/258048.0 
              + 3.0*eta*(81.0*chi1**2 - 158.0*chi1*chi2 + 81.0*chi2**2)/1024.0 
              + 371.0*eta**2/2048.0)
        w = w.at[4].set(w4)
        
        w5 = (-7729.0*PI/21504.0 + 146597.0/(64512.0) * (m1*chi1 + m2*chi2)
              + 13.0*PI*eta/256.0 - 1213.0*eta*(chi1+chi2)/1152.0
              + 7.0*eta/128.0 * dm*(chi1-chi2) - 17.0/128.0 * eta**2 * (chi1+chi2))
        w = w.at[5].set(w5)
        
        w6 = 0.0 # Truncated for simplicity in placeholder
        w = w.at[6].set(w6)
        
        w7 = 0.0 # Truncated for simplicity in placeholder
        w = w.at[7].set(w7)
        return w

    def _pn_amp(self, eta, chi1, chi2):
        dm = _safe_sqrt(1.0 - 4.0 * eta)
        m1 = (1.0 + dm) / 2.0
        m2 = (1.0 - dm) / 2.0
        
        h = jnp.zeros(8, dtype=jnp.complex128)
        h = h.at[0].set(1.0)
        h = h.at[1].set(0.0)
        h = h.at[2].set(-107.0/42.0 + 55.0/42.0 * eta)
        h = h.at[3].set(2.0*PI - 2.0*(chi1+chi2)/3.0 + 2.0*dm*(chi1-chi2)/(3.0*(m1+m2)) + 2.0/3.0 * eta * (chi1+chi2))
        h = h.at[4].set(-2173.0/1512.0 - 1069.0*eta/216.0 + 2047.0*eta**2/1512.0 + (m1*chi1**2 + m2*chi2**2) - eta*(chi1-chi2)**2)
        h = h.at[5].set(-107.0*PI/21.0 + 34.0*PI*eta/21.0 - 24.0j*eta)
        
        h6 = 0.0 # Truncated for simplicity in placeholder
        h = h.at[6].set(h6)
        
        h7 = 0.0 # Truncated for simplicity in placeholder
        h = h.at[7].set(h7)
        return h

    def _evaluate_frequency(self, t, eta, chi1, chi2, c):
        t_insp = jnp.minimum(t, c["t_meco"])
        theta_insp = jnp.power(eta * jnp.maximum(-t_insp, 1e-10) / 5.0, -1.0/8.0)
        w_pn = self._pn_omega(eta, chi1, chi2)
        
        omega_0 = theta_insp**3 / 8.0
        
        w_insp = omega_0 * (
            w_pn[0] + w_pn[1]*theta_insp + w_pn[2]*theta_insp**2 + w_pn[3]*theta_insp**3 +
            w_pn[4]*theta_insp**4 + w_pn[5]*theta_insp**5 + w_pn[6]*theta_insp**6 + w_pn[7]*theta_insp**7 +
            c["c8"]*theta_insp**8 + c["c9"]*theta_insp**9 + c["c10"]*theta_insp**10 + c["c11"]*theta_insp**11 + c["c12"]*theta_insp**12
        )
        
        t_merg = jnp.clip(t, c["t_meco"], 0.0)
        w_merg = c["a0"] + c["a1"]*jnp.arcsinh(c["alpha_1"]*t_merg) + c["a2"]*jnp.arcsinh(c["alpha_1"]*t_merg)**2 + c["a3"]*jnp.arcsinh(c["alpha_1"]*t_merg)**3 + c["a4"]*jnp.arcsinh(c["alpha_1"]*t_merg)**4
        
        t_rd = jnp.maximum(t, 0.0)
        denom = 1.0 + c["c3"]*jnp.exp(-c["c2"]*t_rd) + c["c4"]*jnp.exp(-2.0*c["c2"]*t_rd)
        num = c["c2"] * (c["c3"]*jnp.exp(-c["c2"]*t_rd) + 2.0*c["c4"]*jnp.exp(-2.0*c["c2"]*t_rd))
        w_rd = c["omega_rd"] + c["c1"] * (num / denom)
        
        return jnp.where(t <= c["t_meco"], w_insp, jnp.where(t < 0.0, w_merg, w_rd))

    def _evaluate_amplitude(self, t, omega, eta, chi1, chi2, c):
        omega = jnp.maximum(omega, 1e-10)
        x = (omega / 2.0)**(2.0/3.0)
        x = jnp.maximum(x, 1e-10)
        h_pn = self._pn_amp(eta, chi1, chi2)
        
        amp_prefactor = 2.0 * eta * jnp.sqrt(16.0/5.0) * x
        amp_insp = amp_prefactor * (
            h_pn[0] + h_pn[1]*x**0.5 + h_pn[2]*x**1.0 + h_pn[3]*x**1.5 +
            h_pn[4]*x**2.0 + h_pn[5]*x**2.5 + h_pn[6]*x**3.0 + h_pn[7]*x**3.5 +
            c["d8"]*x**4.0 + c["d9"]*x**4.5 + c["d10"]*x**5.0
        )
        amp_insp = jnp.abs(amp_insp)
        
        t_merg = jnp.clip(t, c["t_meco"], 0.0)
        amp_merg = c["b0"] + c["b1"]*t_merg**2 + c["b2"]*(1.0/jnp.cosh(c["alpha_1"]*t_merg))**(1.0/7.0) + c["b3"]*(1.0/jnp.cosh(c["alpha_1"]*t_merg))
        
        t_rd = jnp.maximum(t, 0.0)
        amp_rd = jnp.exp(-c["alpha_1"]*t_rd) * (c["d1"]*jnp.tanh(c["d2"]*t_rd + c["d3"]) + c["d4"])
        
        return jnp.where(t <= c["t_meco"], amp_insp, jnp.where(t < 0.0, amp_merg, amp_rd))

    def _evaluate_phase(self, t, eta, chi1, chi2, c):
        omega = self._evaluate_frequency(t, eta, chi1, chi2, c)
        dt = jnp.diff(t, prepend=t[0] - (t[1] - t[0]))
        
        # Use lax.scan to compute cumsum without XLA loop unrolling
        def scan_fn(carry, xi):
            new_carry = carry + xi
            return new_carry, new_carry
        _, phase = jax.lax.scan(scan_fn, 0.0, omega * dt, unroll=1)
        
        return phase
