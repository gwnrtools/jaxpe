r"""ESIGMA inspiral waveform adapter: esigmapy's JAX backend as a jaxpe WaveformModel.

Motivation & Math
-----------------
To extract the underlying astrophysics of coalescing compact binaries, we construct highly 
accurate waveform templates. In the framework of black hole perturbation theory, the 
gravitational radiation is encoded in the Newman-Penrose scalar $\Psi_4$, which obeys the 
Teukolsky equation. Asymptotic evaluation of $\Psi_4$ yields the two polarization states 
$h_+$ and $h_\times$. For eccentric binaries, the dynamics are solved via a coupled set of 
Post-Newtonian (PN) ordinary differential equations (ODEs).

The ESIGMA model implemented here evolves the binary through an eccentric inspiral. The 
orbital dynamics—characterized by the semi-latus rectum $p$ (or inverse radius $x$), 
eccentricity $e$, and mean anomaly $l$—are integrated utilizing the Tsit5 solver. The 
radiation field is then constructed by decomposing the strain into spin-weighted spherical 
harmonics ${}_{-2}Y_{lm}(\iota, \phi)$:
$$ h_+ - i h_\times = \sum_{l=2}^{\infty} \sum_{m=-l}^{l} h_{lm}(t) {}_{-2}Y_{lm}(\iota, \phi) $$

esigmapy's high-level API returns non-traceable numpy arrays. This adapter meticulously 
rebuilds the pipeline from JAX-differentiable primitives so the map 
$\boldsymbol{\theta} \to (h_+, h_\times)$ is fully compatible with our gradient-based 
MCMC samplers.

Implementation details:
  1. diffrax Tsit5 integration of the eccentric ODEs on a fixed-length time grid.
  2. Sub-grid ISCO-crossing localization via linear interpolation.
  3. Kepler's equation solved over the mapped detector time grid.
  4. Spherical harmonic modes $h_{lm}$ built from esigmapy's kernels, applying the 
     non-precessing symmetry $h_{l,-m} = (-1)^l h_{lm}^*$.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .waveform import MTSUN_SI


class ESIGMAInspiral:
    """End-to-end traceable ESIGMA inspiral polarizations. See module docstring.

    Parameters
    ----------
    f_lower
        GW frequency (Hz) at which the dynamics start. Choose a few Hz below the
        likelihood's f_min so the turn-on taper sits out of band.
    modes
        (l, |m|) pairs; negative m added via the nonprecessing symmetry.
    n_ode_grid
        Static number of points the ODE solution is saved on (spanning the
        parameter-dependent [0, T_max]); 4096 resolves the orbital timescale for
        stellar-mass binaries from ~20 Hz.
    taper_on_seconds, taper_off_seconds
        Smooth cosine ramps at the start of the signal and before the ISCO cutoff.
    """

    def __init__(
        self,
        f_lower: float = 20.0,
        modes=((2, 2), (3, 3), (4, 4)),
        rad_pn_order: int = 8,
        mode_pn_order: int = 8,
        ode_eps: float = 1e-8,
        inspiral_end_radius: float = 4.0,
        n_ode_grid: int = 4096,
        max_ode_steps: int = 65536,
        taper_on_seconds: float = 0.05,
        taper_off_seconds: float = 0.02,
        s1z_table_points: int = 2049,
        adjoint_mode: str = "forward_sensitivity",
    ):
        import diffrax
        from esigmapy.inspiral.jax_backend.generator import LAL_MRSUN_SI, LAL_PC_SI
        from esigmapy.inspiral.jax_backend.go_terms import hlmGOresult_jax
        from esigmapy.inspiral.jax_backend.inspiral import (
            dphi_dt_jax,
            eccentric_x_model_odes_jax,
        )
        from esigmapy.inspiral.jax_backend.kepler import (
            separation_jax,
            solve_kepler_jax,
        )
        from esigmapy.inspiral.numba_backend.pn_inspiral import x_dot_4pn_SF

        self.f_lower = float(f_lower)
        self.modes = tuple((int(l), abs(int(m))) for l, m in modes)
        self.rad_pn_order = int(rad_pn_order)
        self.mode_pn_order = int(mode_pn_order)
        self.ode_eps = float(ode_eps)
        self.x_final = 1.0 / float(inspiral_end_radius)
        self.n_ode_grid = int(n_ode_grid)
        self.max_ode_steps = int(max_ode_steps)
        self.taper_on_seconds = float(taper_on_seconds)
        self.taper_off_seconds = float(taper_off_seconds)
        self.adjoint_mode = adjoint_mode

        self._diffrax = diffrax
        self._odes = eccentric_x_model_odes_jax
        self._kepler = solve_kepler_jax
        self._separation = separation_jax
        self._dphi_dt = dphi_dt_jax
        self._hlm = hlmGOresult_jax
        self._mrsun = float(LAL_MRSUN_SI)
        self._mpc_m = 1.0e6 * float(LAL_PC_SI)

        # 4PN SF horizon-flux term: x_dot_4pn_SF(e, eta, S1z) = eta * g(S1z);
        # g uses complex polygamma (host-side), so tabulate it once
        grid = np.linspace(-0.995, 0.995, int(s1z_table_points))
        vals = np.array([x_dot_4pn_SF(0.0, 1.0, s) for s in grid])
        self._sf_grid = jnp.asarray(grid)
        self._sf_vals = jnp.asarray(vals)

    # ------------------------------------------------------------------ dynamics

    def _rhs(self, sf_val, t_scale):
        """diffrax RHS in the fixed s=t/t_scale in [0,1] domain: dy/ds = t_scale * dy/dt.

        Reparametrizing onto a *parameter-independent* domain is required for correct
        gradients: differentiating an ODE solve whose ``SaveAt`` times are themselves a
        function of the parameter being differentiated (as ``linspace(0, t_max(mc), n)``
        would be) hits a moving-output-time adjoint case that diffrax's adjoints do not
        handle via plain reverse-mode AD (verified empirically: autodiff disagreed with
        finite differences by orders of magnitude before this fix). With ``s`` fixed,
        ``t_scale`` enters only as an ordinary multiplicative RHS parameter — standard,
        correctly-differentiable sensitivity.
        """
        rad_pn, vpn, x_final = self.rad_pn_order, self.mode_pn_order, self.x_final
        odes = self._odes

        def rhs(s, y, args):
            eta, m1, m2, s1z, s2z = args
            t = s * t_scale
            past_isco = y[0] >= x_final
            y_capped = y.at[0].set(jnp.where(past_isco, x_final, y[0]))
            dydt = odes(t, y_capped, (eta, m1, m2, s1z, s2z, rad_pn, vpn, sf_val))
            dydt = jnp.where(past_isco, jnp.zeros_like(dydt), dydt)
            return dydt * t_scale

        return rhs

    def _solve_ys(self, theta, adjoint):
        x_init, e0, l0, m1, m2, s1z, s2z = theta
        eta = m1 * m2 / (m1 + m2) ** 2
        dfx = self._diffrax
        t_max = 1.5 * (5.0 / 256.0) / eta * (x_init**-4 - self.x_final**-4)
        sf_val = eta * jnp.interp(s1z, self._sf_grid, self._sf_vals)
        s_grid = jnp.linspace(0.0, 1.0, self.n_ode_grid)

        sol = dfx.diffeqsolve(
            terms=dfx.ODETerm(self._rhs(sf_val, t_max)),
            solver=dfx.Tsit5(),
            t0=0.0,
            t1=1.0,
            dt0=s_grid[1] - s_grid[0],
            y0=jnp.stack([x_init, e0, l0, jnp.zeros_like(x_init)]),
            args=(eta, m1, m2, s1z, s2z),
            saveat=dfx.SaveAt(ts=s_grid),
            stepsize_controller=dfx.PIDController(rtol=self.ode_eps, atol=self.ode_eps),
            max_steps=self.max_ode_steps,
            adjoint=adjoint,
            throw=False,
        )
        return sol.ys

    def _integrate(self, x_init, e0, l0, eta, m1, m2, s1z, s2z):
        """Solve the x-model ODEs."""
        dfx = self._diffrax
        theta = jnp.stack([x_init, e0, l0, m1, m2, s1z, s2z])

        if self.adjoint_mode == "recursive_checkpoint":
            # Baseline reverse-mode (leads to compile OOM for large PN orders)
            ys = self._solve_ys(
                theta, adjoint=dfx.RecursiveCheckpointAdjoint(checkpoints=16)
            )
        elif self.adjoint_mode == "forward_sensitivity":
            # Forward-mode pushes directly through the ODE steps via DirectAdjoint
            ys = self._solve_ys(theta, adjoint=dfx.DirectAdjoint())
        else:
            raise ValueError(f"Unknown adjoint_mode: {self.adjoint_mode}")

        # Reconstruct ts explicitly
        t_max = 1.5 * (5.0 / 256.0) / eta * (x_init**-4 - self.x_final**-4)
        s_grid = jnp.linspace(0.0, 1.0, self.n_ode_grid)
        return s_grid * t_max, ys

    @staticmethod
    def _isco_time(ts, x_arr, x_final, t_max):
        """First crossing of x_final, sub-grid by linear interpolation (smooth in params)."""
        crossed = x_arr >= x_final
        has_crossed = jnp.any(crossed)
        i1 = jnp.clip(jnp.argmax(crossed), 1, x_arr.shape[0] - 1)
        x_lo, x_hi = x_arr[i1 - 1], x_arr[i1]
        frac = jnp.clip((x_final - x_lo) / jnp.maximum(x_hi - x_lo, 1e-300), 0.0, 1.0)
        t_cross = ts[i1 - 1] + frac * (ts[i1] - ts[i1 - 1])
        return jnp.where(has_crossed, t_cross, t_max)

    # ------------------------------------------------------------------ waveform

    def __call__(self, params: dict, times: jax.Array):
        @jax.remat
        def _compute(params, times):
            def _heavy_math(theta, times):
                mc, q, e0, l0, s1z, s2z, t_c = theta
                eta = q / (1.0 + q) ** 2
                m_total = mc / eta**0.6
                m1 = m_total / (1.0 + q)
                m2 = m_total * q / (1.0 + q)

                m_sec = m_total * MTSUN_SI
                x_init = (m_sec * jnp.pi * self.f_lower) ** (2.0 / 3.0)

                ts, ys = self._integrate(x_init, e0, l0, eta, m1, m2, s1z, s2z)
                x_a, e_a, l_a, phi_a = ys[:, 0], ys[:, 1], ys[:, 2], ys[:, 3]
                t_isco = self._isco_time(ts, x_a, self.x_final, ts[-1])

                t_geo = (times - t_c) / m_sec + t_isco
                valid = (t_geo >= 0.0) & (t_geo <= t_isco)
                tq = jnp.clip(t_geo, 0.0, t_isco)

                x_t = jnp.interp(tq, ts, x_a)
                e_t = jnp.interp(tq, ts, e_a)
                l_t = jnp.interp(tq, ts, l_a)
                phi_t = jnp.interp(tq, ts, phi_a)

                u_t = jax.vmap(self._kepler, in_axes=(0, 0))(l_t, e_t)
                r_t = jax.vmap(
                    lambda u, x, e: self._separation(u, eta, x, e, m1, m2, s1z, s2z)
                )(u_t, x_t, e_t)
                phidot_t = jax.vmap(
                    lambda u, x, e: self._dphi_dt(
                        u, eta, m1, m2, s1z, s2z, x, e, self.mode_pn_order
                    )
                )(u_t, x_t, e_t)
                dt_geo = (times[1] - times[0]) / m_sec
                rdot_t = jnp.gradient(r_t) / dt_geo

                hlm_batch = jax.vmap(
                    self._hlm,
                    in_axes=(
                        None,
                        None,
                        None,
                        None,
                        0,
                        0,
                        0,
                        0,
                        None,
                        None,
                        None,
                        None,
                        0,
                    ),
                )

                hlms = []
                for l, m in self.modes:
                    hlm = (
                        hlm_batch(
                            l,
                            m,
                            m_total,
                            eta,
                            r_t * m_total,
                            rdot_t,
                            phi_t,
                            phidot_t / m_total,
                            self._mpc_m,
                            self.mode_pn_order,
                            s1z,
                            s2z,
                            x_t,
                        )
                        * self._mrsun
                    )
                    hlms.append(hlm)
                hlms = jnp.stack(hlms)

                on_geo = jnp.maximum(self.taper_on_seconds / m_sec, 1e-12)
                off_geo = jnp.maximum(self.taper_off_seconds / m_sec, 1e-12)
                w_on = 0.5 - 0.5 * jnp.cos(jnp.pi * jnp.clip(t_geo / on_geo, 0.0, 1.0))
                w_off = 0.5 - 0.5 * jnp.cos(
                    jnp.pi * jnp.clip((t_isco - t_geo) / off_geo, 0.0, 1.0)
                )
                w = jnp.where(valid, w_on * w_off, 0.0)
                return hlms, w

            mc = params["chirp_mass"]
            q = params["mass_ratio"]
            s1z = params.get("spin1z", jnp.zeros(()))
            s2z = params.get("spin2z", jnp.zeros(()))
            e0 = params.get("eccentricity", jnp.zeros(()))
            l0 = params.get("mean_anomaly", jnp.zeros(()))
            iota = params["inclination"]
            beta = params["phase"]
            t_c = params["geocent_time"]
            r_si = params["luminosity_distance"] * self._mpc_m

            theta = jnp.stack([mc, q, e0, l0, s1z, s2z, t_c])

            if self.adjoint_mode == "recursive_checkpoint":
                hlms, w = _heavy_math(theta, times)
            elif self.adjoint_mode == "forward_sensitivity":
                heavy_math = jax.custom_vjp(
                    lambda th, tms: _heavy_math(th, tms)
                )
                    
                def heavy_fwd(th, tms):
                    out = _heavy_math(th, tms)
                    jac = jax.jacfwd(lambda t: _heavy_math(t, tms))(th)
                    return out, jac
                    
                def heavy_bwd(jac, g):
                    j_hlms, j_w = jac
                    g_hlms, g_w = g
                    term1 = jnp.real(jnp.tensordot(g_hlms, j_hlms, axes=([0, 1], [0, 1])))
                    term2 = jnp.tensordot(g_w, j_w, axes=([0], [0]))
                    return (term1 + term2, None)
                    
                heavy_math.defvjp(heavy_fwd, heavy_bwd)
                hlms, w = heavy_math(theta, times)
            else:
                raise ValueError(f"Unknown adjoint_mode: {self.adjoint_mode}")

            from .harmonics import spin_weighted_ylm

            h = jnp.zeros(times.shape, dtype=jnp.complex128)
            for i, (l, m) in enumerate(self.modes):
                hlm = hlms[i] * (self._mpc_m / r_si)
                h = h + hlm * spin_weighted_ylm(iota, beta, l, m)
                h = h + (-1.0) ** l * jnp.conj(hlm) * spin_weighted_ylm(
                    iota, beta, l, -m
                )

            return w * h.real, -w * h.imag

        return _compute(params, times)
