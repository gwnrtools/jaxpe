"""Closed-form phase-and-distance-marginalized intrinsic likelihood for FD models.

This is the "Route B" inner likelihood specialized to frequency-domain, dominant
(2,2)-mode waveform models (IMRPhenomD today; IMRPhenomXAS, TaylorF2, ... later).
Sky position, inclination and coalescence time are held fixed; the coalescence
phase and the luminosity distance are marginalized analytically, leaving a
likelihood over the intrinsic parameters alone:

    L(theta_int) = log integral_D  pi(D) I0(u |Z|) exp(-1/2 u^2 rho^2) dD  -  1/2 <d|d>

with u = d_ref / D, Z = <d|h_ref> (complex), rho^2 = <h_ref|h_ref>, and h_ref the
detector strain at a reference distance d_ref and zero phase. The phase integral is
the closed form ln I0(u|Z|) (a Bessel function); the distance integral is a 1-D
quadrature over the distance prior.

Why the closed form is exact for dominant-mode models
-----------------------------------------------------
For a model whose only mode is (2,2), the two polarizations share a single complex
factor h0: h_+ = h0 (1 + cos^2 iota)/2 and h_x = -i h0 cos iota, and a shift of the
coalescence phase acts as h0(phi_c) = h0(0) e^{-2i phi_c} EXACTLY. The detector
strain therefore factorizes as

    h_det(phi_c) = e^{s 2i phi_c} h_det(0),   s = +-1,

so ln L(phi_c, D) = Re[e^{s 2i phi_c} u Z] - 1/2 u^2 rho^2 - 1/2 <d|d>. Marginalizing
phi_c against a uniform prior gives ln I0(u|Z|); the result depends only on |Z|,
rho^2 and <d|d> -- never on the sign s. This needs no spherical-harmonic-mode
decomposition and no FFT, which is why it is cheaper and numerically cleaner than the
general mode-based marginalizer (:class:`~jaxpe.gw.marginalized.ModesNetworkLikelihood`)
whenever the model is genuinely dominant-mode with a fixed sky.

Guardrail: the factorization is only APPROXIMATE for a higher-mode model. The
constructor MEASURES the deviation on a representative parameter point
(``check_params``) and warns, so plugging in an unsuitable model fails loudly rather
than silently biasing the posterior. Use the mode-based marginalizer in that case.

See ``docs/gpry_fusion_design.md`` for how this fits into the surrogate route, and
``examples/08_fd_dominant_mode_route_comparison.py`` for an end-to-end cross-validation
against gradient-based direct sampling.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import i0e, logsumexp


class PhaseDistanceMarginalLikelihood:
    """L(theta_int) with coalescence phase (Bessel I0) and distance (quadrature)
    marginalized, for any dominant-(2,2)-mode frequency-domain model.

    The only waveform touchpoint is ``like.detector_strains_fd`` -- a method of every
    jaxpe network likelihood -- so any model wrapped by
    :func:`~jaxpe.gw.make_injection` can be plugged in unchanged.

    Parameters
    ----------
    like
        A jaxpe network likelihood built by :func:`~jaxpe.gw.make_injection` on a
        frequency-domain model.
    names
        Ordered intrinsic-parameter names; this fixes the vector order expected by
        :meth:`__call__` (e.g. ``("chirp_mass", "mass_ratio", "spin1z", "spin2z")``).
    fixed_ext
        The parameters held fixed at their true values: the sky (``ra``, ``dec``,
        ``psi``), ``inclination``, and any intrinsic parameter pinned in a given run
        (e.g. the spins in a non-spinning analysis).
    dist_bounds, dist_power, d_ref, n_dist
        Distance-prior quadrature settings: bounds (Mpc), the ``distance^power`` prior
        exponent (2.0 for a Euclidean/volume prior), the reference distance at which
        ``h_ref`` is evaluated, and the number of quadrature nodes.
    check_params
        A representative FULL parameter dict used ONCE to verify the dominant-mode
        factorization for the plugged-in model. ``None`` skips the check and warns.
    dominant_mode_tol
        Residual above which the constructor warns that the closed form is only
        approximate for this model.
    t_ref
        Fixed coalescence GPS time. Defaults to ``check_params["geocent_time"]`` when
        available, else 0.0.

    Attributes
    ----------
    dominant_mode_residual : float or None
        The measured deviation of ``h_det(phi_c)/h_det(0)`` from a pure phase; ~0 for a
        dominant-mode model (closed form exact), O(1) if higher modes are present.
        ``None`` if ``check_params`` was not supplied.
    """

    def __init__(
        self,
        like,
        names,
        fixed_ext,
        *,
        dist_bounds=(1000.0, 8000.0),
        dist_power=2.0,
        d_ref=1000.0,
        n_dist=400,
        check_params=None,
        dominant_mode_tol=1e-2,
        t_ref=None,
    ):
        self.names = tuple(names)
        self.d_ref = float(d_ref)
        if t_ref is not None:
            self.t_ref = float(t_ref)
        elif check_params is not None:
            self.t_ref = float(check_params["geocent_time"])
        else:
            self.t_ref = 0.0

        st = like._static()
        self.df = float(st["df"])
        dets = list(like.detectors)
        data = {d.name: st["data"][d.name] for d in dets}
        invpsd = {d.name: st["inv_psd_banded"][d.name] for d in dets}
        # <d|d> is theta-independent: precompute once.
        self.dd = float(
            sum(
                4.0
                * self.df
                * jnp.sum(
                    (data[d.name].real ** 2 + data[d.name].imag ** 2) * invpsd[d.name]
                )
                for d in dets
            )
        )
        fixed = {k: float(v) for k, v in fixed_ext.items()}
        nm, d_ref_, t_ref_ = self.names, self.d_ref, self.t_ref

        def overlaps(theta_vec):
            """(Re Z, Im Z, rho^2) for h_ref at phase 0, distance d_ref, time t_ref."""
            theta = {n: theta_vec[i] for i, n in enumerate(nm)}
            p = {
                **theta,
                **{k: jnp.asarray(v) for k, v in fixed.items()},
                "phase": jnp.asarray(0.0),
                "luminosity_distance": jnp.asarray(d_ref_),
                "geocent_time": jnp.asarray(t_ref_),
            }
            strains = like.detector_strains_fd(p)
            Z, rho2 = 0.0 + 0.0j, 0.0
            for d in dets:
                h = strains[d.name]
                Z = Z + 4.0 * self.df * jnp.sum(
                    jnp.conj(data[d.name]) * h * invpsd[d.name]
                )
                rho2 = rho2 + 4.0 * self.df * jnp.sum(
                    (h.real**2 + h.imag**2) * invpsd[d.name]
                )
            return jnp.real(Z), jnp.imag(Z), rho2

        self._overlaps = jax.jit(overlaps)

        # Fixed distance-quadrature grid, normalized in log space.
        self._D = np.linspace(dist_bounds[0], dist_bounds[1], n_dist)
        self._u = d_ref / self._D
        log_prior = dist_power * np.log(self._D)
        self._log_dD = np.log(np.gradient(self._D))
        self._log_pi = log_prior - logsumexp(log_prior + self._log_dD)

        self.dominant_mode_residual = self._check_dominant_mode(
            like, fixed, check_params
        )
        if self.dominant_mode_residual is None:
            warnings.warn(
                "PhaseDistanceMarginalLikelihood: no check_params given; the "
                "dominant-mode factorization underpinning the I0 phase marginal was "
                "NOT verified for this model."
            )
        elif self.dominant_mode_residual > dominant_mode_tol:
            warnings.warn(
                f"PhaseDistanceMarginalLikelihood: dominant-mode factorization residual "
                f"{self.dominant_mode_residual:.2e} > tol {dominant_mode_tol:.1e}. The "
                f"plugged-in model appears to carry sub-dominant modes; the closed-form "
                f"I0 phase marginal is only APPROXIMATE. Use the mode-based marginalizer."
            )

    def _check_dominant_mode(self, like, fixed, check_params):
        """Residual of h_det(phi_c=delta)/h_det(0) from a constant phase e^{+-2i delta}.

        ~0 for a dominant-mode model (closed form exact); O(1) if higher modes present.
        """
        if check_params is None:
            return None
        delta = 0.3

        def strains(phi):
            p = {
                **{k: jnp.asarray(v) for k, v in check_params.items()},
                **{k: jnp.asarray(v) for k, v in fixed.items()},
                "phase": jnp.asarray(phi),
                "luminosity_distance": jnp.asarray(self.d_ref),
            }
            return like.detector_strains_fd(p)

        h0, hd = strains(0.0), strains(delta)
        worst = 0.0
        for d in like.detectors:
            a, b = np.asarray(h0[d.name]), np.asarray(hd[d.name])
            band = np.abs(a) > 1e-3 * np.max(np.abs(a))
            if not band.any():
                continue
            ratio = b[band] / a[band]  # should be a constant e^{+-2i delta}
            center = np.median(ratio.real) + 1j * np.median(ratio.imag)
            worst = max(
                worst, float(np.max(np.abs(ratio - center)) / (abs(center) + 1e-30))
            )
        return worst

    def __call__(self, x):
        """Log marginal likelihood at intrinsic vector ``x`` (order given by ``names``)."""
        zr, zi, rho2 = self._overlaps(jnp.asarray(np.asarray(x, float).ravel()))
        abs_z = float(np.hypot(float(zr), float(zi)))
        rho2 = float(rho2)
        # log I0(u |Z|) via the exponentially-scaled i0e to stay finite at large argument
        log_i0 = np.log(i0e(self._u * abs_z)) + self._u * abs_z
        integrand = self._log_pi + log_i0 - 0.5 * self._u**2 * rho2 + self._log_dD
        return float(logsumexp(integrand) - 0.5 * self.dd)
