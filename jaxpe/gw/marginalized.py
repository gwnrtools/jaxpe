"""Mode-based network likelihood: fixed-extrinsic core of the marginalized-likelihood path.

This module implements stage one of the extrinsic-marginalized intrinsic likelihood
``L(theta_int)`` of the GPry-fusion design (``docs/gpry_fusion_design.md`` section 3):
a Whittle network likelihood evaluated from precomputed spherical-harmonic modes
``h_lm(t)`` (one expensive external-model call) instead of from a traceable waveform.

Why per-mode FFTs
-----------------
With the complex strain ``h = h_+ - i h_x = sum_lm c_lm h_lm(t)``,
``c_lm = {}_{-2}Y_{lm}(iota, phi)``, linearity of the (windowed) FFT gives

    h_+(f) = sum_lm [ Re(c_lm) A_lm(f) - Im(c_lm) B_lm(f) ]
    h_x(f) = -sum_lm [ Re(c_lm) B_lm(f) + Im(c_lm) A_lm(f) ]

where ``A_lm = FFT(window * Re h_lm) dt`` and ``B_lm = FFT(window * Im h_lm) dt`` are
cached once per external-model call. Every extrinsic parameter then acts on cached FD
arrays: (iota, phi) as scalar coefficients, distance as ``d_ref/D_L``, coalescence time
and sky position as phase factors. This is what makes the extrinsic marginalization
(quadratures and FFT over t_c; later stages of the design) cheap and ``vmap``-able, and
the whole map modes -> lnL differentiable in the extrinsic parameters.

The Whittle sum, detector projection, PSD banding and GMST linearization are inherited
unchanged from ``TDNetworkLikelihood``, so agreement with the direct time-domain path
is structural rather than reimplemented (verified in ``tests/test_marginalized.py``).
"""

import functools
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from scipy.special import logsumexp as logsumexp_np

from .conditioning import td_to_fd, time_shift
from .detectors import antenna_pattern, time_delay_from_geocenter
from .external_models import ModesData
from .harmonics import spin_weighted_ylm
from .likelihood import TDNetworkLikelihood


# unit-cube dims of the extrinsic (ra, dec, psi, iota) parametrization: ra and psi are
# periodic on their intervals; dec and iota are reflected at their boundaries
_EXT_PERIODIC = (True, False, True, False)


def _ext_cube_to_angles(u):
    """Map the unit 4-cube (flat prior measure) to (ra, dec, psi, iota)."""
    return np.stack(
        [
            2.0 * np.pi * u[:, 0],
            np.arcsin(2.0 * u[:, 1] - 1.0),
            np.pi * u[:, 2],
            np.arccos(2.0 * u[:, 3] - 1.0),
        ],
        axis=1,
    )


def _mixture_log_density(u, centers, widths, comp_w, defense):
    """log q(u) of the defensive mixture on the unit 4-cube.

    q = defense * 1 + (1 - defense) * sum_k comp_w[k] * prod_d phi_d(u_d; c_kd, h_d),
    with per-dim Gaussian kernels wrapped (periodic dims) or reflected (bounded dims)
    back into [0, 1] via their +-1-cell images, so q integrates to 1 on the cube.
    """
    dens = np.ones((u.shape[0], centers.shape[0]))
    for d in range(4):
        x = u[:, None, d] - centers[None, :, d]
        h = widths[d]
        if _EXT_PERIODIC[d]:
            imgs = (x - 1.0, x, x + 1.0)
        else:
            c = centers[None, :, d]
            imgs = (x, u[:, None, d] + c, u[:, None, d] + c - 2.0)
        dens_d = sum(
            np.exp(-0.5 * (im / h) ** 2) / (np.sqrt(2.0 * np.pi) * h) for im in imgs
        )
        dens *= dens_d
    return np.log(defense + (1.0 - defense) * dens @ comp_w)


def _mixture_sample(rng, n, centers, widths, comp_w, defense):
    """Draw n points from the defensive mixture, folded back into the unit cube."""
    out = rng.uniform(size=(n, 4))
    kde = rng.uniform(size=n) >= defense
    n_kde = int(kde.sum())
    if n_kde and len(centers):
        idx = rng.choice(len(centers), p=comp_w, size=n_kde)
        x = centers[idx] + rng.normal(size=(n_kde, 4)) * widths[None, :]
        for d in range(4):
            if _EXT_PERIODIC[d]:
                x[:, d] = np.mod(x[:, d], 1.0)
            else:
                y = np.abs(np.mod(x[:, d], 2.0))
                x[:, d] = np.where(y > 1.0, 2.0 - y, y)
        out[kde] = x
    return out


@functools.lru_cache(maxsize=8)
def _leggauss(n: int):
    # host-side numpy ONLY: this may first be hit inside a trace, and caching jnp
    # arrays created there would leak tracers into later traces (see the eager-cache
    # rule in likelihood.py); numpy operands are converted per-use instead
    return np.polynomial.legendre.leggauss(n)


@dataclass(frozen=True)
class ModesNetworkLikelihood(TDNetworkLikelihood):
    """Network likelihood evaluated from fixed spherical-harmonic modes.

    The modes (one intrinsic-parameter point, one external-model call) are supplied as
    ``ModesData`` on the same uniform time grid as the analysis segment; extrinsic
    parameters (``inclination, phase, luminosity_distance, geocent_time, ra, dec,
    psi``) remain free, cheap, and differentiable.

    ``waveform`` is unused (pass None; the modes replace it); construct via
    ``from_likelihood`` to share grids/data/PSDs with an existing
    ``TDNetworkLikelihood``.
    """

    modes_data: ModesData = field(default=None, kw_only=True)

    def __post_init__(self):
        md = self.modes_data
        if md is None:
            raise ValueError("modes_data is required")
        if md.times.shape != self.times.shape or not np.allclose(
            md.times, self.times, rtol=0.0, atol=1e-9
        ):
            raise ValueError(
                "ModesData.times must equal the analysis time grid "
                "(resampling of external-model output happens upstream, "
                "in the external_models conditioning layer)."
            )
        super().__post_init__()

    def _static(self):
        if not self._cache:
            super()._static()
            dt, window = self._cache["dt"], self._cache["window"]
            lms = tuple(sorted(self.modes_data.modes))
            mode_a = jnp.stack(
                [
                    td_to_fd(jnp.asarray(np.real(self.modes_data.modes[lm])), dt, window)
                    for lm in lms
                ]
            )
            mode_b = jnp.stack(
                [
                    td_to_fd(jnp.asarray(np.imag(self.modes_data.modes[lm])), dt, window)
                    for lm in lms
                ]
            )
            # 0.5 <d|d>: eagerly, so no tracer can ever be cached (see likelihood.py)
            half_dd = 0.0
            for det in self.detectors:
                d = self._cache["data"][det.name]
                half_dd += (
                    2.0
                    * self._cache["df"]
                    * jnp.sum(
                        (d.real**2 + d.imag**2)
                        * self._cache["inv_psd_banded"][det.name]
                    )
                )
            self._cache.update(
                mode_lms=lms, mode_a=mode_a, mode_b=mode_b, half_dd=half_dd
            )
        return self._cache

    def _reference_polarizations_fd(self, iota, phi):
        """(h+, hx)(f) at the reference distance and coalescence time of the modes."""
        st = self._static()
        c = jnp.stack(
            [spin_weighted_ylm(iota, phi, l, m) for (l, m) in st["mode_lms"]]
        )
        cr, ci = jnp.real(c)[:, None], jnp.imag(c)[:, None]
        hp_fd = jnp.sum(cr * st["mode_a"] - ci * st["mode_b"], axis=0)
        hc_fd = -jnp.sum(cr * st["mode_b"] + ci * st["mode_a"], axis=0)
        return hp_fd, hc_fd

    def polarizations_fd(self, params: dict):
        st = self._static()
        hp_fd, hc_fd = self._reference_polarizations_fd(
            params["inclination"], params["phase"]
        )
        scale = self.modes_data.d_ref_mpc / params["luminosity_distance"]
        # place the (t_ref-aligned) modes at the requested coalescence time
        dtc = params["geocent_time"] - self.modes_data.t_ref
        return (
            time_shift(scale * hp_fd, st["freqs"], dtc),
            time_shift(scale * hc_fd, st["freqs"], dtc),
        )

    # ------------------------------------------------------------- marginalization

    def _log_distance_integral(
        self, z, sig2, u_lo, u_hi, log_prior_norm, dist_power, n_dist
    ):
        r"""log \int_{u_lo}^{u_hi} pi_u(u) exp(u z - u^2 sig2 / 2) du,  u = d_ref / D_L.

        The integrand is a Gaussian in u centered at u* = z/sig2 with width 1/sqrt(sig2),
        typically much narrower than the prior range at realistic SNR, so fixed nodes on
        [u_lo, u_hi] would under-resolve it. Gauss-Legendre nodes are instead placed on
        [u* - 12/sigma, u* + 12/sigma] clipped to the prior range, with a floor of
        24/sigma of coverage against the nearest boundary so the window never collapses
        when the peak lies outside the range (there the integrand is a monotone Gaussian
        tail dominated by the boundary). The window edges move smoothly with (z, sig2);
        the induced non-smoothness of the quadrature *error* is far below its magnitude.

        Accuracy domain: ~1e-10 relative when the peak u* lies inside or within a few
        1/sigma of the prior range; for deeply boundary-truncated tails (source outside
        the distance prior, |u* - boundary| >> 10/sigma) the boundary layer is thinner
        than the node spacing and accuracy degrades to ~percent -- those contributions
        are exponentially subdominant in the full marginal whenever the prior actually
        contains the source (verified in tests/test_marginalized.py).

        ``pi_u`` is the power-law distance prior pi(D) \propto D^p mapped to u:
        log pi_u(u) = log_prior_norm - (p + 2) log u.
        """
        sig = jnp.sqrt(sig2)
        su = 1.0 / sig
        u_star = z / sig2
        lo = jnp.clip(jnp.minimum(u_star - 12.0 * su, u_hi - 24.0 * su), u_lo, u_hi)
        hi = jnp.clip(jnp.maximum(u_star + 12.0 * su, u_lo + 24.0 * su), u_lo, u_hi)
        x, w = _leggauss(n_dist)
        u = lo + 0.5 * (hi - lo) * (x + 1.0)
        log_w = jnp.log(0.5 * (hi - lo) * w)
        log_f = (
            u * z
            - 0.5 * u**2 * sig2
            + log_prior_norm
            - (dist_power + 2.0) * jnp.log(u)
        )
        return logsumexp(log_f + log_w)

    def _phase_decomposition(self, iota, ra, dec, psi, gmst):
        r"""Exact phi_c decomposition of the network filter.

        The detector strain is a trig polynomial in phi_c: with
        c_lm = {}_{-2}Y_{lm}(iota, phi_c) = y_lm(iota) e^{i m phi_c} the projected
        template is  h_det(f; phi_c) = sum_M e^{i M phi_c} G_M(f)  over the distinct
        azimuthal numbers M of the stored modes. This yields, exactly in phi_c,

            <d|h>(phi_c, t_c = k dt) = Re sum_M e^{-i M phi_c} Z_M[k]
            <h|h>(phi_c)             = Re e(phi_c)^T Gamma conj(e(phi_c))

        with one length-n complex FFT per M (NOT per phi_c node) for
        Z_M[k] = sum_{f>0} 4 df d(f) conj(G_M(f)) e^{2 pi i f k dt} / S(f), summed
        over detectors, and the Hermitian Gram matrix
        Gamma_MM' = sum_det 4 df sum_f G_M conj(G_M') / S. Dense phi_c grids are then
        essentially free -- which matters because the phi_c integrand e^{lnL} carries
        harmonics up to |M| ~ max|m| SNR^2 / 2 and needs O(SNR^2) quadrature nodes.

        Returns (m_values (n_M,), Z (n_M, n) complex, Gamma (n_M, n_M) complex).
        """
        st = self._static()
        n = len(self.times)
        n_f = len(self.freqs)
        lms = st["mode_lms"]
        m_values = tuple(sorted({m for (_, m) in lms}))
        m_index = {m: i for i, m in enumerate(m_values)}

        # y_lm(iota): the phi=0 harmonic (complex in general)
        y = [spin_weighted_ylm(iota, jnp.zeros(()), l, m) for (l, m) in lms]

        z_kernel = jnp.zeros((len(m_values), n_f), dtype=jnp.complex128)
        gamma = jnp.zeros((len(m_values), len(m_values)), dtype=jnp.complex128)
        for det in self.detectors:
            f_plus, f_cross = antenna_pattern(det, ra, dec, psi, gmst)
            delay = time_delay_from_geocenter(det, ra, dec, gmst)
            phasor = jnp.exp(-2j * jnp.pi * st["freqs"] * delay)
            g = jnp.zeros((len(m_values), n_f), dtype=jnp.complex128)
            for i, (l, m) in enumerate(lms):
                # strain response to mode i: cr*P + ci*Q with c = y e^{i m phi};
                # regroup into e^{+-i m phi} coefficients
                p_i = (f_plus * st["mode_a"][i] - f_cross * st["mode_b"][i]) * phasor
                q_i = (-f_plus * st["mode_b"][i] - f_cross * st["mode_a"][i]) * phasor
                g = g.at[m_index[m]].add(0.5 * y[i] * (p_i - 1j * q_i))
                g = g.at[m_index[-m]].add(0.5 * jnp.conj(y[i]) * (p_i + 1j * q_i))
            inv_psd = st["inv_psd_banded"][det.name]
            z_kernel = z_kernel + 4.0 * st["df"] * st["data"][det.name] * jnp.conj(
                g
            ) * inv_psd
            gamma = gamma + 4.0 * st["df"] * jnp.einsum(
                "af,bf,f->ab", g, jnp.conj(g), inv_psd
            )

        # one-sided complex sum over f>0 at every integer-sample shift k: n * ifft of
        # the zero-padded kernel (banding has already zeroed DC and Nyquist)
        z_series = n * jnp.fft.ifft(
            jnp.pad(z_kernel, ((0, 0), (0, n - n_f))), axis=-1
        )
        return jnp.asarray(m_values), z_series, gamma

    def log_marginal_likelihood(
        self,
        params: dict,
        *,
        n_phi: int = 512,
        n_dist: int = 128,
        tc_half_samples: int = 205,
        dist_min: float = 100.0,
        dist_max: float = 5000.0,
        dist_power: float = 2.0,
        phi_batch: int = 32,
    ):
        r"""lnL marginalized over (phi_c, t_c, D_L) at fixed (ra, dec, psi, inclination).

        Priors: phi_c uniform on [0, 2pi); t_c uniform over the ``2*tc_half_samples+1``
        sample-grid nodes centered on ``params['geocent_time']`` (at 2048 Hz the
        default 205 samples is a +-0.1 s window); D_L with pi(D) \propto
        D^{dist_power} on [dist_min, dist_max] Mpc.

        Structure (design note section 3): the phi_c dependence is decomposed exactly
        into azimuthal harmonics (one FFT per distinct |m|, see
        ``_phase_decomposition``), <d|h> at every integer-sample t_c comes from those
        same FFTs, and the D_L integral is Gaussian in u = d_ref/D_L, done by adaptive
        Gauss-Legendre. phi_c uses a dense trapezoid grid: the integrand carries
        harmonics up to ~ max|m| SNR^2/2, so ``n_phi`` must exceed twice that --
        the default 512 covers network SNR ~ 20; double it (cheap) for louder events
        and convergence-check by comparing doubled settings. GMST is frozen at the
        window center (microradian-exact over sub-second windows). ``phi_batch``
        chunks the phi grid to bound peak memory at
        ``phi_batch * n_tc * n_dist`` floats.

        Returns the same normalization as ``log_likelihood`` (includes -0.5 <d|d>),
        so narrow priors reproduce the fixed-parameter value.
        """
        st = self._static()
        n = len(self.times)
        t_center = params["geocent_time"]
        gmst = self._gmst({"geocent_time": t_center})

        d_ref = self.modes_data.d_ref_mpc
        u_lo, u_hi = d_ref / dist_max, d_ref / dist_min
        p = float(dist_power)
        # normalization of pi_u(u) du = pi_D(D) dD, pi_D = C D^p:
        # pi_u(u) = C d_ref^(p+1) u^(-(p+2))
        if p == -1.0:
            log_c = -np.log(np.log(dist_max / dist_min))
        else:
            log_c = np.log(abs(p + 1.0)) - np.log(
                abs(dist_max ** (p + 1.0) - dist_min ** (p + 1.0))
            )
        log_prior_norm = log_c + (p + 1.0) * np.log(d_ref)

        m_values, z_series, gamma = self._phase_decomposition(
            params["inclination"], params["ra"], params["dec"], params["psi"], gmst
        )

        dt = st["dt"]
        k0 = jnp.round((t_center - self.modes_data.t_ref) / dt).astype(jnp.int64)
        k_idx = jnp.mod(k0 + jnp.arange(-tc_half_samples, tc_half_samples + 1), n)
        z_win = z_series[:, k_idx]  # (n_M, n_tc) complex

        phi_nodes = jnp.arange(n_phi) * (2.0 * jnp.pi / n_phi)

        def per_phi(phi):
            e = jnp.exp(1j * m_values * phi)  # (n_M,)
            z_t = jnp.real(jnp.conj(e) @ z_win)  # (n_tc,)
            sig2 = jnp.real(e @ gamma @ jnp.conj(e))
            return jax.vmap(
                lambda z: self._log_distance_integral(
                    z, sig2, u_lo, u_hi, log_prior_norm, p, n_dist
                )
            )(z_t)

        log_i = jax.lax.map(per_phi, phi_nodes, batch_size=phi_batch)

        n_tc = 2 * tc_half_samples + 1
        return (
            logsumexp(log_i)
            - jnp.log(float(n_phi))
            - jnp.log(float(n_tc))
            - st["half_dd"]
        )

    def log_marginal_likelihood_full(
        self,
        params: dict,
        *,
        n_pilot: int = 4096,
        n_final: int = 4096,
        rounds: int = 2,
        defense: float = 0.2,
        max_centers: int = 256,
        qmc_seed: int = 7,
        return_diagnostics: bool = False,
        n_phi: int = 512,
        n_dist: int = 128,
        tc_half_samples: int = 205,
        dist_min: float = 100.0,
        dist_max: float = 5000.0,
        dist_power: float = 2.0,
        phi_batch: int = 32,
        ext_batch: int = 8,
    ):
        r"""The fully extrinsic-marginalized intrinsic likelihood L(theta_int).

        Marginalizes ``log_marginal_likelihood`` (phi_c, t_c, D_L -- exact/adaptive)
        over (ra, dec, psi, iota) with isotropic priors -- uniform in
        (alpha, sin(delta), psi, cos(iota)) -- by **defensive adaptive importance
        sampling**. Plain QMC is hopeless here: e^{lnL} occupies ~1e-4 of the
        extrinsic space already at network SNR ~ 10 (measured ESS 1.5/8192 on the
        test event), concentrated at the sky/orientation of the source. The scheme:

        1. Pilot: ``n_pilot`` scrambled-Sobol nodes on the flat-measure unit cube.
        2. Proposal: a Gaussian KDE (wrapped/reflected onto the cube) centered on the
           importance-weighted high-lnL nodes, mixed with a ``defense`` fraction of
           the uniform prior -- so a mode the KDE missed is still sampled and the
           estimator stays consistent; repeat ``rounds`` times.
        3. Estimate from the last batch: logsumexp(lnL - log q) - log n, with the
           effective sample size ESS = (sum w)^2 / sum w^2 as the built-in
           diagnostic; **treat the result as unconverged if ESS is low** (the IS
           log-estimate is biased low by ~ var/2, i.e. ~ 1/(2 ESS)).

        This runs host-side by construction (data-dependent sampling cannot live
        inside a trace); each evaluation batch is a compiled ``lax.map``. It is the
        GPry-facing scalar L(theta_int) of the fusion design (section 3) and is not
        differentiable end-to-end -- by design, nothing needs gradients through it.

        ``params`` supplies only ``geocent_time`` (the t_c prior-window center).
        With ``return_diagnostics=True`` also returns
        dict(ess, n_eval, lnl_max, logz_rounds).
        """
        from scipy.stats import qmc

        inner = dict(
            n_phi=n_phi,
            n_dist=n_dist,
            tc_half_samples=tc_half_samples,
            dist_min=dist_min,
            dist_max=dist_max,
            dist_power=dist_power,
            phi_batch=phi_batch,
        )
        t_center = params["geocent_time"]

        def evaluate(u):
            nodes = jnp.asarray(_ext_cube_to_angles(u))

            def per_node(node):
                p = {
                    "ra": node[0],
                    "dec": node[1],
                    "psi": node[2],
                    "inclination": node[3],
                    "geocent_time": t_center,
                }
                return self.log_marginal_likelihood(p, **inner)

            return np.asarray(jax.lax.map(per_node, nodes, batch_size=ext_batch))

        rng = np.random.default_rng(qmc_seed)
        u_all = qmc.Sobol(d=4, scramble=True, seed=qmc_seed).random(n_pilot)
        lnl_all = evaluate(u_all)
        logq_all = np.zeros(n_pilot)  # pilot proposal: the uniform prior measure

        logz_rounds, lw_last, n_last = [], None, n_pilot
        for _ in range(rounds):
            # centers: importance-corrected top nodes within 15 e-folds of the peak
            lw = lnl_all - logq_all
            keep = np.flatnonzero(lnl_all > lnl_all.max() - 15.0)
            keep = keep[np.argsort(lnl_all[keep])[-max_centers:]]
            centers = u_all[keep]
            comp_w = np.exp(lw[keep] - logsumexp_np(lw[keep]))
            k_eff = float(np.exp(-np.sum(comp_w * np.log(comp_w + 1e-300))))
            mu = comp_w @ centers
            var = comp_w @ (centers - mu) ** 2
            widths = np.clip(1.5 * np.sqrt(var) * k_eff ** (-1.0 / 6.0), 0.01, 0.25)

            u_new = _mixture_sample(rng, n_final, centers, widths, comp_w, defense)
            logq_new = _mixture_log_density(u_new, centers, widths, comp_w, defense)
            lnl_new = evaluate(u_new)

            lw_last, n_last = lnl_new - logq_new, n_final
            logz_rounds.append(logsumexp_np(lw_last) - np.log(n_final))
            u_all = np.concatenate([u_all, u_new])
            lnl_all = np.concatenate([lnl_all, lnl_new])
            logq_all = np.concatenate([logq_all, logq_new])

        log_z = logsumexp_np(lw_last) - np.log(n_last)
        ess = float(np.exp(2.0 * logsumexp_np(lw_last) - logsumexp_np(2.0 * lw_last)))
        if return_diagnostics:
            return log_z, dict(
                ess=ess,
                n_eval=len(lnl_all),
                lnl_max=float(lnl_all.max()),
                logz_rounds=logz_rounds,
            )
        return log_z

    def _log_marginal_over_nodes(
        self, nodes, log_w, t_center, *, ext_batch: int = 8, **inner
    ):
        """logsumexp of the 3D marginal over (ra, dec, psi, iota) quadrature nodes.

        ``nodes`` is (n, 4) columns (ra, dec, psi, iota); ``log_w`` the log quadrature
        weights of a normalized measure. Shared by the QMC path and by independent
        product-quadrature cross-checks in the tests.
        """

        def per_node(node):
            p = {
                "ra": node[0],
                "dec": node[1],
                "psi": node[2],
                "inclination": node[3],
                "geocent_time": t_center,
            }
            return self.log_marginal_likelihood(p, **inner)

        log_l = jax.lax.map(per_node, nodes, batch_size=ext_batch)
        return logsumexp(log_l + log_w)

    @classmethod
    def from_likelihood(
        cls, like: TDNetworkLikelihood, modes_data: ModesData
    ) -> "ModesNetworkLikelihood":
        """Share grids, data, PSDs and conventions with an existing likelihood."""
        return cls(
            waveform=None,
            detectors=like.detectors,
            data_fd=like.data_fd,
            psds=like.psds,
            freqs=like.freqs,
            times=like.times,
            f_min=like.f_min,
            f_max=like.f_max,
            gmst_ref=like.gmst_ref,
            t_ref=like.t_ref,
            tukey_alpha=like.tukey_alpha,
            accumulate_f64=like.accumulate_f64,
            modes_data=modes_data,
        )
