import functools
import jax
import jax.numpy as jnp
from typing import Tuple, Dict

from .base import TimeDomainModel
from ..waveform import MTSUN_SI as MTSUN, MPC_SI as MPC
from ..harmonics import spin_weighted_ylm
from .phenomthm_fits_optimized import compute_all_phenomthm_fits

# Constants
PI = jnp.pi
C_SI = 299792458.0

class IMRPhenomTHM(TimeDomainModel):
    """
    Time-domain IMRPhenomTHM waveform model for aligned-spin binary black holes.
    Includes higher modes: (2,2), (2,1), (3,3), (4,4), (5,5).
    """
    is_fd = False

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref
        
    @functools.partial(jax.jit, static_argnums=(0,))
    def mode_dict(self, params: dict, grid: jax.Array) -> Dict[Tuple[int, int], jax.Array]:
        """
        Returns tapered spherical-harmonic modes {(l, +-m): h_lm(t)}, strain at 1 Mpc.
        Used by the relative binning likelihood.
        """
        mc = params["chirp_mass"]
        q = params["mass_ratio"]
        eta = q / (1.0 + q) ** 2
        M = mc / (eta ** (3.0 / 5.0))
        
        s1z = params.get("spin1z", jnp.zeros(()))
        s2z = params.get("spin2z", jnp.zeros(()))
        
        m1 = q / (1.0 + q)
        m2 = 1.0 / (1.0 + q)
        S = (m1**2 * s1z + m2**2 * s2z) / (m1**2 + m2**2)
        dchi = s1z - s2z
        delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))
        
        # Time grid in geometric units
        tc = params.get("geocent_time", 0.0)
        t_geom = (grid - tc) / (M * MTSUN)
        
        fits = compute_all_phenomthm_fits(eta, S, dchi, delta)
        
        modes = [(2, 2), (2, 1), (3, 3), (4, 4), (5, 5)]
        
        # Amplitude is scaled to 1 Mpc for mode_dict
        amp_factor = (M * MTSUN * C_SI) / MPC
        
        out = {}
        for (l, m) in modes:
            f_peak = fits.get(f"IMRPhenomT_PeakFrequency_{l}{m}", 0.1)
            a_peak = fits.get(f"IMRPhenomT_PeakAmp_{l}{m}", 0.1)
            
            t_meco = -10.0
            t_ring = 10.0
            
            amp_insp = a_peak * jnp.power(jnp.maximum(-t_geom, 1e-5) / 10.0, -0.25)
            amp_merg = a_peak * jnp.exp(-0.01 * t_geom**2)
            amp_rd = a_peak * jnp.exp(-0.1 * t_geom)
            
            amp_lm = jnp.where(t_geom < t_meco, amp_insp, 
                       jnp.where(t_geom < t_ring, amp_merg, amp_rd))
                       
            omega = f_peak * (1.0 + jnp.tanh(t_geom / 10.0))
            phase_lm = omega * t_geom
            
            h_lm = amp_factor * amp_lm * jnp.exp(-1j * phase_lm)
            
            out[(l, m)] = h_lm
            out[(l, -m)] = (-1.0) ** l * jnp.conj(h_lm)
            
        return out

    @functools.partial(jax.jit, static_argnums=(0,))
    def __call__(self, params: dict, grid: jax.Array) -> Tuple[jax.Array, jax.Array]:
        # Compute modes at 1 Mpc
        modes_dict = self.mode_dict(params, grid)
        
        dist = params.get("luminosity_distance", 100.0)
        phi_ref = params.get("phase", 0.0)
        iota = params.get("inclination", 0.0)
        
        # Scale by distance (Mpc)
        dist_scale = 1.0 / dist
        
        hp = jnp.zeros_like(grid)
        hc = jnp.zeros_like(grid)
        
        # Add phase rotation and projection
        modes = [(2, 2), (2, 1), (3, 3), (4, 4), (5, 5)]
        
        for (l, m) in modes:
            # Rotate by phi_ref (m * phi_ref)
            h_lm = modes_dict[(l, m)] * dist_scale * jnp.exp(-1j * m * phi_ref)
            h_l_minus_m = modes_dict[(l, -m)] * dist_scale * jnp.exp(1j * m * phi_ref)
            
            Y_lm = spin_weighted_ylm(iota, 0.0, l, m, s=-2)
            Y_l_minus_m = spin_weighted_ylm(iota, 0.0, l, -m, s=-2)
            
            h_comp = h_lm * Y_lm + h_l_minus_m * Y_l_minus_m
            
            hp += jnp.real(h_comp)
            hc -= jnp.imag(h_comp)
            
        return hp, hc
