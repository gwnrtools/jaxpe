r"""Time-domain relative binning (heterodyned) likelihood.

Phase RB-4 of ``docs/relative_binning_design.md``, implementing the dominant-mode core
of the time-domain heterodyned likelihood (Sharma, Vijaykumar & Kumar 2026,
arXiv:2601.11239, Sec. II.C, Eqs. 13-22). Unlike the frequency-domain path
(:mod:`~jaxpe.gw.likelihood.relative_binning_fd`), the noise covariance here is the
*non-diagonal* symmetric Toeplitz ``C_ij = rho(|i-j|)``; the linear algebra is done in
``O(N log N)`` by :mod:`~jaxpe.gw.likelihood.toeplitz` (Gohberg-Semencul).

Model (single detector, dominant (2, +-2) mode, fixed sky)
---------------------------------------------------------
The detector strain is ``s(t) = Re[p * u(t)]`` with ``u(t)`` the complex (2,2) waveform
mode (carrying the intrinsic dependence) and ``p`` a complex extrinsic coefficient
(antenna response, inclination/phase harmonic, distance); ``p = c_{2,2} +
conj(c_{2,-2})`` with ``c_{2,m} = (F_+ + i F_x) {}_{-2}Y_{2,m}(iota, phi) / D_L`` (using
``h_{2,-2} = conj(h_{2,2})``). Heterodyning approximates the mode ratio
``r(t) = u(t) / u_0(t)`` as linear in time bins,

    r(t) ~= r_0(b) + r_1(b) (t - t_c(b)),

so the trial mode is needed only at the bin edges. The log-likelihood,

    ln L = -1/2 L(d, d) + L(d, s) - 1/2 L(s, s),   L(a, b) = a^T C^{-1} b,

reduces (dominant mode) to

    L(d, s)  = Re[p Z],                    Z = sum_b [r_0 A_0(b) + r_1 A_1(b)],
    L(s, s)  = 1/2 Re[p^2 G] + 1/2 |p|^2 G',
    G  = r_0^T B_0 r_0 + 2 r_0^T B_1 r_1,
    G' = r_0^T B_2 conj(r_0) + r_0^T B_3 conj(r_1) + r_1^T B_3b conj(r_0),

with summary data (``w = C^{-1} d``; ``dt_j = t_j - t_c(bin j)``)

    A_0(b) = sum_{j in b} w_j u_{0,j},              A_1(b) = sum_{j in b} w_j u_{0,j} dt_j,
    B_0(b1,b2) = sum_{i in b1} u_{0,i} v_0(b2)_i,    v_0(b2) = C^{-1}(u_0 1_{b2}),
    B_1(b1,b2) = sum_{i in b1} u_{0,i} v_1(b2)_i,    v_1(b2) = C^{-1}(u_0 dt 1_{b2}),
    B_2 = sum_{i in b1} u_{0,i} conj(v_0)_i,   B_3 = sum u_{0,i} conj(v_1)_i,
    B_3b = sum_{i in b1} u_{0,i} dt_i conj(v_0)_i.

Because ``C^{-1}`` is real, ``C^{-1} conj(x) = conj(C^{-1} x)``, so the conjugate-mode
tensors reuse ``v_0, v_1`` -- no extra solves. At the fiducial (``r_0 = 1, r_1 = 0``)
this is *exactly* the dense likelihood; away from it the error is the linear-ratio
truncation, controlled by the bin resolution. Verified against
:func:`td_dense_loglikelihood` in ``tests/test_relative_binning_td.py``.

Scope: single detector, dominant mode, fixed sky and coalescence time. Higher modes
(per-mode bins + the nine-tensor cross-mode summary data of Appendix A), the detector
network, and t_c marginalization are documented extensions.
"""

import numpy as np

import jax
import jax.numpy as jnp

from .toeplitz import inverse_generator, inverse_matvec


def time_bin_edges(reference_mode, phase_per_bin: float = 0.5, amp_floor: float = 1e-6):
    r"""Adaptive time-bin edges from the fiducial mode's phase evolution.

    Bins are placed so the unwrapped phase of ``reference_mode`` advances by about
    ``phase_per_bin`` radians per bin (so the mode ratio is well approximated as linear
    within a bin), over the support where ``|reference_mode| > amp_floor * max``.
    Returns the integer sample indices of the bin edges; there are ``len(edges) - 1``
    bins tiling the support.
    """
    u = np.asarray(reference_mode)
    amp = np.abs(u)
    support = np.nonzero(amp > amp_floor * (amp.max() + 1e-300))[0]
    lo, hi = int(support[0]), int(support[-1])
    phase = np.unwrap(np.angle(u[lo : hi + 1]))
    # cumulative absolute phase advance (monotone; handles chirp up or down)
    cum = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(phase)))])
    total = float(cum[-1])
    n_bins = max(1, int(np.ceil(total / phase_per_bin)))
    levels = np.linspace(0.0, total, n_bins + 1)
    local = np.searchsorted(cum, levels, side="left")
    local = np.unique(np.concatenate([[0], local, [hi - lo]]))
    local = np.clip(local, 0, hi - lo)
    return lo + np.unique(local)


class RelativeBinningTDLikelihood:
    """Dominant-mode time-domain heterodyned likelihood for one detector.

    Parameters
    ----------
    fiducial_mode
        Complex fiducial (2,2) mode ``u_0(t)`` on the analysis time grid.
    times
        Sample times of the grid (uniform).
    data
        Real time-domain strain ``d(t)`` of the detector.
    acf
        First column of the symmetric Toeplitz noise covariance ``C`` (the noise
        autocorrelation; see :func:`~jaxpe.gw.likelihood.toeplitz.autocorrelation_from_psd`).
    phase_per_bin
        Bin-resolution control (radians of fiducial phase per bin).

    Notes
    -----
    Construct, then call :meth:`log_likelihood` with the trial mode evaluated at the bin
    edges (:attr:`edge_indices`) and the complex extrinsic coefficient ``p``. All the
    heavy ``C^{-1}`` work happens once, here.
    """

    def __init__(self, fiducial_mode, times, data, acf, *, phase_per_bin=0.5):
        u0 = np.asarray(fiducial_mode, dtype=complex)
        t = np.asarray(times, dtype=float)
        d = np.asarray(data, dtype=float)
        acf = np.asarray(acf, dtype=float)
        n = u0.shape[0]

        self.times = t
        self._x = jnp.asarray(inverse_generator(acf))  # C^{-1} generator (Levinson, once)

        # w = C^{-1} d and the theta-independent 1/2 <d|d>
        w = np.asarray(inverse_matvec(self._x, jnp.asarray(d)))
        self.half_dd = 0.5 * float(d @ w)

        edges = time_bin_edges(u0, phase_per_bin=phase_per_bin)
        self.edge_indices = edges
        self.n_bins = int(edges.size - 1)
        edge_t = t[edges]
        self.edge_times = jnp.asarray(edge_t)
        self.dt_bin = jnp.asarray(np.diff(edge_t))
        t_c = 0.5 * (edge_t[:-1] + edge_t[1:])  # bin reference (center) times

        pts = np.arange(edges[0], edges[-1] + 1)  # supported samples
        bin_id = np.clip(np.searchsorted(edges, pts, side="right") - 1, 0, self.n_bins - 1)
        dt = t[pts] - t_c[bin_id]  # (t_j - t_c(bin j)) over the support
        u0_s = u0[pts]

        # --- A summary data (complex, per bin)
        self.A0 = jnp.asarray(_bin_reduce(w[pts] * u0_s, bin_id, self.n_bins))
        self.A1 = jnp.asarray(_bin_reduce(w[pts] * u0_s * dt, bin_id, self.n_bins))
        self.u0_edges = jnp.asarray(u0[edges])

        # --- B summary data (complex, n_bins x n_bins). For each bin b2 apply C^{-1} to
        # the fiducial mode masked to b2 (and to u0*dt masked to b2); C^{-1} is real, so
        # the conjugate-mode tensors reuse these solves.
        pts_j = jnp.asarray(pts)
        onehot = (bin_id[None, :] == np.arange(self.n_bins)[:, None]).astype(float)
        v0 = self._inverse_of_masked(onehot * u0_s[None, :], pts_j, n)[:, pts_j]  # (nb, npts)
        v1 = self._inverse_of_masked(onehot * (u0_s * dt)[None, :], pts_j, n)[:, pts_j]

        u0p = jnp.asarray(u0_s)  # (npts,)
        dtp = jnp.asarray(dt)  # (npts,)
        idx = jnp.asarray(bin_id)  # (npts,)
        nb = self.n_bins
        red = jax.vmap(lambda row: jax.ops.segment_sum(row, idx, num_segments=nb))
        # red(X)[b2] = bin-sum over b1 of X[b2]; transpose to index [b1, b2]
        self.B0 = red(u0p[None, :] * v0).T
        self.B1 = red(u0p[None, :] * v1).T
        self.B2 = red(u0p[None, :] * jnp.conj(v0)).T
        self.B3 = red(u0p[None, :] * jnp.conj(v1)).T
        self.B3b = red((u0p * dtp)[None, :] * jnp.conj(v0)).T

    def _inverse_of_masked(self, rows_over_support, pts_j, n):
        """C^{-1} of each support-masked complex row, returned on the full grid (nb, N)."""
        full = jnp.zeros((rows_over_support.shape[0], n), dtype=complex)
        full = full.at[:, pts_j].set(jnp.asarray(rows_over_support))
        inv = jax.vmap(
            lambda v: inverse_matvec(self._x, jnp.real(v))
            + 1j * inverse_matvec(self._x, jnp.imag(v))
        )
        return inv(full)

    def _ratio_coeffs(self, trial_mode_edges):
        r = jnp.asarray(trial_mode_edges) / self.u0_edges  # (n_bins+1,)
        r0 = 0.5 * (r[1:] + r[:-1])
        r1 = (r[1:] - r[:-1]) / self.dt_bin
        return r0, r1

    def log_likelihood(self, trial_mode_edges, p):
        """Heterodyned lnL given the trial (2,2) mode at the bin edges and coefficient ``p``.

        ``trial_mode_edges`` has shape ``(n_bins + 1,)``; ``p`` is the complex extrinsic
        coefficient ``c_{2,2} + conj(c_{2,-2})``.
        """
        r0, r1 = self._ratio_coeffs(trial_mode_edges)
        p = jnp.asarray(p) + 0.0j

        z = jnp.sum(r0 * self.A0 + r1 * self.A1)  # complex
        g = r0 @ (self.B0 @ r0) + 2.0 * r0 @ (self.B1 @ r1)
        gp = (
            r0 @ (self.B2 @ jnp.conj(r0))
            + r0 @ (self.B3 @ jnp.conj(r1))
            + r1 @ (self.B3b @ jnp.conj(r0))
        )
        l_ds = jnp.real(p * z)
        l_ss = 0.5 * jnp.real(p * p * g) + 0.5 * (jnp.abs(p) ** 2) * jnp.real(gp)
        return -self.half_dd + l_ds - 0.5 * l_ss

    __call__ = log_likelihood


def _bin_reduce(values, bin_id, n_bins):
    """Sum ``values`` (length N) into ``n_bins`` bins given each sample's ``bin_id``."""
    out = np.zeros(n_bins, dtype=values.dtype)
    np.add.at(out, bin_id, values)
    return out


def td_dense_loglikelihood(trial_mode_full, p, data, acf):
    """Exact dense time-domain log-likelihood ``-1/2 (d - s)^T C^{-1} (d - s)``.

    Reference for the heterodyned likelihood: ``s = Re[p * trial_mode_full]`` at full
    time resolution, ``C`` the symmetric Toeplitz covariance with first column ``acf``.
    """
    u = np.asarray(trial_mode_full, dtype=complex)
    s = np.real(np.asarray(p) * u)
    r = np.asarray(data, dtype=float) - s
    x = inverse_generator(np.asarray(acf, dtype=float))
    cinv_r = np.asarray(inverse_matvec(jnp.asarray(x), jnp.asarray(r)))
    return -0.5 * float(r @ cinv_r)


def extrinsic_coefficient(f_plus, f_cross, iota, phi, distance, ylm22, ylm2m2):
    """Dominant-mode extrinsic coefficient ``p = c_{2,2} + conj(c_{2,-2})``.

    ``ylm22 = {}_{-2}Y_{2,2}(iota, phi)`` and ``ylm2m2 = {}_{-2}Y_{2,-2}(iota, phi)``.
    """
    fpc = f_plus + 1j * f_cross
    c22 = fpc * ylm22 / distance
    c2m2 = fpc * ylm2m2 / distance
    return c22 + np.conj(c2m2)
