r"""Fourier-domain relative binning (heterodyned) likelihood.

Phase RB-1 of ``docs/relative_binning_design.md``: an accelerated inner-product engine
for dominant-mode frequency-domain waveform models (IMRPhenomD today). It computes the
Zackay, Dai & Venumadhav (2018, arXiv:1806.08792) heterodyned approximation to the
exact FD Whittle likelihood of :class:`~jaxpe.gw.likelihood.fd.FDNetworkLikelihood`,
evaluating the trial waveform only at ``~N_bins`` bin-edge frequencies instead of at
the full frequency grid.

Method
------
Pick a **fiducial** waveform ``h0`` at a nearby parameter point. Any nearby trial
waveform's ratio ``r(f) = h(f)/h0(f)`` is slow in ``f``, so it is well approximated as
piecewise-linear on coarse bins ``b = [f_lo, f_hi]``:

    r(f) ~= r0(b) + r1(b) (f - f_c(b)),
    r0(b) = (r(f_hi) + r(f_lo)) / 2,   r1(b) = (r(f_hi) - r(f_lo)) / (f_hi - f_lo),

with ``f_c(b)`` the bin center. Precompute, per detector and bin, the **summary data**
(from data ``d``, fiducial ``h0`` and one-sided PSD ``S``, at full resolution)

    A0(b) = 4 df sum_{f in b} d h0* / S,     A1(b) = 4 df sum_{f in b} d h0* (f-f_c) / S,
    B0(b) = 4 df sum_{f in b} |h0|^2 / S,    B1(b) = 4 df sum_{f in b} |h0|^2 (f-f_c) / S,

so the noise-weighted overlaps become ``O(N_bins)`` sums (Zackay Eq. 7)

    <d|h> = sum_b [A0 r0* + A1 r1*],
    <h|h> = sum_b [B0 |r0|^2 + 2 B1 Re(r0 r1*)].

The full log-likelihood is reproduced with the same normalization as
``FDNetworkLikelihood`` (i.e. including the ``-1/2 <d|d>`` constant):

    ln L = -1/2 <d|d> + Re<d|h> - 1/2 <h|h>.

Because the bins tile the analysis band exactly once, at the fiducial parameters
(``r == 1``: ``r0 == 1``, ``r1 == 0``) this is **exactly** the full likelihood; away
from the fiducial the error is the ``O((f-f_c)^2)`` truncation of the linear ratio,
controlled by the bin resolution ``(chi, epsilon)`` (Zackay Eqs. 8-10).

Scope: valid for **dominant-(2,2)-mode** FD models, where the summed detector-strain
ratio is smooth. Higher-mode FD models need per-mode binning (see the design note,
§4.4) and are out of scope for this class.
"""

import warnings
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from .base import NetworkLikelihood, project_to_detector
from .relative_binning_td import _bin_reduce
from .relative_binning_td import time_bin_edges as _phase_bin_edges

# leading post-Newtonian frequency powers of the phase (bilby's set): chirp mass
# (-5/3), spin (-2/3), merger-time shift (1), tidal (5/3), and a higher term (7/3).
# The bin scheme only needs these to be a representative basis of the phase drift.
_DEFAULT_GAMMA = (-5.0 / 3.0, -2.0 / 3.0, 1.0, 5.0 / 3.0, 7.0 / 3.0)


def frequency_bin_edges(
    freqs, f_min, f_max, chi=1.0, epsilon=0.5, gamma=_DEFAULT_GAMMA
):
    r"""Relative-binning frequency bin edges (Zackay et al. 2018, Eqs. 8-10).

    Returns the integer indices into ``freqs`` marking bin boundaries; there are
    ``len(edges) - 1`` bins, and (because the band endpoints are always included) the
    bins tile the analysis band ``[f_min, f_max]`` exactly once.

    The worst-case differential phase drift ``dPsi_max(f) = 2 pi chi sum_i
    sgn(g_i) (f/f_*i)^{g_i}`` (with ``f_*i = f_min`` if ``g_i < 0`` else ``f_max``) is
    monotone increasing in ``f``; bin edges are placed at uniform increments of
    ``epsilon`` in ``dPsi_max`` and snapped to the frequency grid.
    """
    freqs = np.asarray(freqs, dtype=float)
    band = np.nonzero((freqs >= f_min) & (freqs <= f_max))[0]
    if band.size < 3:
        raise ValueError("need >= 3 frequency samples in [f_min, f_max] to bin")
    fb = freqs[band]
    gamma = np.asarray(gamma, dtype=float)
    f_star = np.where(gamma < 0.0, f_min, f_max)
    # dPsi_max(f): each term is increasing in f, so the sum is monotone increasing.
    dpsi = (
        2.0
        * np.pi
        * chi
        * np.sum(
            np.sign(gamma)[:, None] * (fb[None, :] / f_star[:, None]) ** gamma[:, None],
            axis=0,
        )
    )
    dpsi = dpsi - dpsi[0]
    total = float(dpsi[-1])
    n_bins = max(1, int(np.ceil(total / epsilon)))
    levels = np.linspace(0.0, total, n_bins + 1)
    local = np.searchsorted(dpsi, levels, side="left")
    # dedupe (coarse levels can collapse onto one grid point) and pin the band ends
    local = np.unique(np.concatenate([[0], local, [fb.size - 1]]))
    local = np.clip(local, 0, fb.size - 1)
    return band[np.unique(local)]


@dataclass(frozen=True)
class RelativeBinningFDLikelihood(NetworkLikelihood):
    """Relative-binning Whittle likelihood for a dominant-mode FD waveform model.

    Drop-in for :class:`~jaxpe.gw.likelihood.base.NetworkLikelihood`:
    ``log_likelihood(params)`` returns the heterodyned approximation to the exact FD
    likelihood, evaluating the trial waveform only at the bin edges. Build it from an
    existing :class:`~jaxpe.gw.likelihood.fd.FDNetworkLikelihood` (or any FD network
    likelihood) via :meth:`from_likelihood`, supplying the fiducial parameters.

    Parameters (in addition to the base fields)
    -------------------------------------------
    fiducial_params
        Full parameter dict of the reference waveform ``h0``; should lie in the
        posterior-dominant region (e.g. the injection/trigger point).
    chi, epsilon, gamma
        Bin-scheme controls (see :func:`frequency_bin_edges`). Smaller ``epsilon``
        (or larger ``chi``) means more bins and higher accuracy.
    """

    fiducial_params: dict = field(default=None, kw_only=True)
    chi: float = field(default=1.0, kw_only=True)
    epsilon: float = field(default=0.5, kw_only=True)
    gamma: tuple = field(default=_DEFAULT_GAMMA, kw_only=True)

    def __post_init__(self):
        if self.fiducial_params is None:
            raise ValueError("fiducial_params is required")
        if not getattr(self.waveform, "is_fd", False):
            raise ValueError(
                "RelativeBinningFDLikelihood requires a frequency-domain waveform "
                "(waveform.is_fd is True)"
            )
        super().__post_init__()

    def polarizations_fd(self, params: dict):
        st = self._static()
        return self.waveform(params, st["freqs"])

    def _strains_at(self, params: dict, freqs):
        """Detector-frame FD strains at an arbitrary frequency array (FD model)."""
        hp_fd, hc_fd = self.waveform(params, freqs)
        gmst = self._gmst(params)
        return {
            det.name: project_to_detector(
                det,
                hp_fd,
                hc_fd,
                freqs,
                params["ra"],
                params["dec"],
                params["psi"],
                gmst,
            )
            for det in self.detectors
        }

    def _static(self):
        if not self._cache:
            super()._static()
            freqs = np.asarray(self._cache["freqs"])
            df = self._cache["df"]
            band = np.nonzero((freqs >= self.f_min) & (freqs <= self.f_max))[0]

            # fiducial strains over the full grid, evaluated eagerly OUTSIDE any trace
            fp = {k: jnp.asarray(v) for k, v in self.fiducial_params.items()}
            h0_grid = {
                n: np.asarray(v)
                for n, v in self._strains_at(fp, jnp.asarray(freqs)).items()
            }

            # Fiducial support: relative binning needs |h0| > 0 (r = h/h0). IMRPhenomD
            # vanishes above the ringdown cutoff, so bins are placed only where the
            # fiducial has power. <d|d> is still taken over the FULL band, which keeps
            # the likelihood exact at the fiducial even when the band exceeds support
            # (only trial power *above* the fiducial cutoff -- high f, low sensitivity --
            # goes unmodelled; the standard relative-binning limitation).
            amp = np.max(
                [np.abs(h0_grid[det.name][band]) for det in self.detectors], axis=0
            )
            # Boundary at the TRUE support edge (IMRPhenomD hard-zeros above ringdown):
            # only the exactly-zero region may be excluded, else the fiducial's own
            # power there would be dropped and the exact-at-fiducial identity broken.
            # The near-cliff denominators (|h0| ~ 1e-6 max) are non-zero because they are
            # read from h0_grid, so the ratio stays finite; those bins carry negligible
            # weight (A, B ~ |h0|).
            sup = np.nonzero(amp > 1e-12 * (amp.max() + 1e-300))[0]
            lo, hi = int(band[sup[0]]), int(band[sup[-1]])
            if lo != band[0] or hi != band[-1]:
                warnings.warn(
                    "RelativeBinningFDLikelihood: fiducial waveform supported only on "
                    f"[{freqs[lo]:.1f}, {freqs[hi]:.1f}] Hz within the analysis band "
                    f"[{self.f_min:.1f}, {self.f_max:.1f}] Hz; trial power outside that "
                    "sub-band is not heterodyned (standard relative-binning limitation)."
                )

            edges = frequency_bin_edges(
                freqs, freqs[lo], freqs[hi], self.chi, self.epsilon, self.gamma
            )
            edge_f = freqs[edges]
            f_c = 0.5 * (edge_f[:-1] + edge_f[1:])  # (n_bins,)
            n_bins = f_c.size
            pts = np.arange(edges[0], edges[-1] + 1)  # every supported band point
            bin_id = np.clip(
                np.searchsorted(edges, pts, side="right") - 1, 0, n_bins - 1
            )
            f_rel = freqs[pts] - f_c[bin_id]

            # ratio denominators: index the fiducial grid at the edges (NOT a fresh
            # waveform call), so the summary data and the r=h/h0 denominator use the
            # identical h0 values and stay consistent to machine precision.
            h0_edge = {
                name: h0_grid[name][edges] for name in (d.name for d in self.detectors)
            }

            A0, A1, B0, B1, h0_edges = {}, {}, {}, {}, {}
            half_dd = 0.0
            for det in self.detectors:
                name = det.name
                invb = np.asarray(self._cache["inv_psd_banded"][name])
                db = np.asarray(self._cache["data"][name])
                # 1/2 <d|d> over the full band (theta-independent; inv_psd is 0 off-band)
                half_dd += 2.0 * df * float(np.sum((db.real**2 + db.imag**2) * invb))
                # summary data over the fiducial support
                d, h0, inv = db[pts], h0_grid[name][pts], invb[pts]
                w = 4.0 * df * inv
                dh = w * d * np.conj(h0)
                h0sq = w * (h0.real**2 + h0.imag**2)
                a0 = np.zeros(n_bins, dtype=complex)
                a1 = np.zeros(n_bins, dtype=complex)
                b0 = np.zeros(n_bins, dtype=float)
                b1 = np.zeros(n_bins, dtype=float)
                np.add.at(a0, bin_id, dh)
                np.add.at(a1, bin_id, dh * f_rel)
                np.add.at(b0, bin_id, h0sq)
                np.add.at(b1, bin_id, h0sq * f_rel)
                A0[name], A1[name] = jnp.asarray(a0), jnp.asarray(a1)
                B0[name], B1[name] = jnp.asarray(b0), jnp.asarray(b1)
                h0_edges[name] = jnp.asarray(h0_edge[name])

            self._cache.update(
                rb_edges=np.asarray(edges),
                rb_n_bins=n_bins,
                rb_edge_freqs=jnp.asarray(edge_f),
                rb_dfbin=jnp.asarray(np.diff(edge_f)),
                rb_A0=A0,
                rb_A1=A1,
                rb_B0=B0,
                rb_B1=B1,
                rb_h0_edges=h0_edges,
                rb_half_dd=half_dd,
            )
        return self._cache

    @property
    def n_bins(self) -> int:
        return self._static()["rb_n_bins"]

    def log_likelihood(self, params: dict):
        """Heterodyned lnL; matches ``FDNetworkLikelihood`` up to the linear-ratio error."""
        st = self._static()
        strains = self._strains_at(params, st["rb_edge_freqs"])
        lnl = -st["rb_half_dd"]
        for det in self.detectors:
            name = det.name
            r = strains[name] / st["rb_h0_edges"][name]  # (n_bins+1,)
            r0 = 0.5 * (r[1:] + r[:-1])
            r1 = (r[1:] - r[:-1]) / st["rb_dfbin"]
            zdh = jnp.sum(
                st["rb_A0"][name] * jnp.conj(r0) + st["rb_A1"][name] * jnp.conj(r1)
            )
            hh = jnp.sum(
                st["rb_B0"][name] * (r0.real**2 + r0.imag**2)
                + 2.0 * st["rb_B1"][name] * jnp.real(r0 * jnp.conj(r1))
            )
            lnl = lnl + jnp.real(zdh) - 0.5 * hh
        return lnl

    __call__ = log_likelihood

    @classmethod
    def from_likelihood(
        cls,
        like: NetworkLikelihood,
        fiducial_params: dict,
        *,
        chi: float = 1.0,
        epsilon: float = 0.5,
        gamma: tuple = _DEFAULT_GAMMA,
    ) -> "RelativeBinningFDLikelihood":
        """Share grids, data, PSDs and conventions with an existing FD likelihood."""
        return cls(
            waveform=like.waveform,
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
            fiducial_params=fiducial_params,
            chi=chi,
            epsilon=epsilon,
            gamma=gamma,
        )


class RelativeBinningFDLikelihoodHM:
    r"""Higher-mode Fourier-domain relative binning, working on FD waveform modes.

    The FD analogue of :class:`~jaxpe.gw.likelihood.relative_binning_td.RelativeBinningTDLikelihoodHM`.
    Because the frequency-domain noise covariance is **diagonal** (the one-sided PSD),
    the summary data are **per bin** (single index), with no cross-bin ``B(b1, b2)`` and
    no Gohberg-Semencul -- just PSD-weighted sums. The strain is ``h(f) = sum_a c_a
    h_a(f)``; each mode is heterodyned by its own ratio ``r_a = h_a / h_{0,a}`` on shared
    frequency bins. With the jaxpe inner product ``(x|y) = 4 df Re sum_f x y* / S``,

        ln L = -1/2 (d|d) + Re sum_a conj(c_a) Z_a - 1/2 sum_{a,b} c_a conj(c_b) M_{ab},
        Z_a = sum_bin [ conj(r0_a) A0_a + conj(r1_a) A1_a ],
        M_{ab} = sum_bin [ r0_a conj(r0_b) B0_{ab} + (r0_a conj(r1_b) + r1_a conj(r0_b)) B1_{ab} ],

    with ``A0_a(bin) = 4 df sum_{f in bin} d conj(h_{0,a}) / S`` (``A1`` weighted by
    ``f - f_c``) and ``B0_{ab}(bin) = 4 df sum_{f in bin} h_{0,a} conj(h_{0,b}) / S``
    (``B1`` weighted by ``f - f_c``). Exact at the fiducial; verified against
    :func:`fd_dense_loglikelihood_modes`.
    """

    def __init__(
        self, fiducial_modes, freqs, data_fd, psd, f_min, f_max, *, phase_per_bin=0.5
    ):
        keys = tuple(fiducial_modes.keys())
        h0 = {k: np.asarray(fiducial_modes[k], dtype=complex) for k in keys}
        freqs = np.asarray(freqs, dtype=float)
        d = np.asarray(data_fd, dtype=complex)
        psd = np.asarray(psd, dtype=float)
        df = float(freqs[1] - freqs[0])

        self.mode_keys = keys
        band = np.nonzero((freqs >= f_min) & (freqs <= f_max))[0]
        inv = np.where((freqs >= f_min) & (freqs <= f_max), 1.0 / psd, 0.0)
        # 1/2 (d|d) over the full band (theta-independent)
        self.half_dd = 2.0 * df * float(np.sum((d.real**2 + d.imag**2) * inv))

        # shared bins from the fastest mode's phase advance, within the band+support
        fb = band
        ref = max(
            keys,
            key=lambda k: float(
                np.sum(np.abs(np.diff(np.unwrap(np.angle(h0[k][fb])))))
            ),
        )
        loc = _phase_bin_edges(h0[ref][fb], phase_per_bin=phase_per_bin)
        edges = fb[loc]  # global frequency indices of the bin edges
        self.edge_indices = edges
        self.n_bins = int(edges.size - 1)
        fe = freqs[edges]
        self.edge_freqs = jnp.asarray(fe)
        f_c = 0.5 * (fe[:-1] + fe[1:])

        pts = np.arange(edges[0], edges[-1] + 1)
        bin_id = np.clip(
            np.searchsorted(edges, pts, side="right") - 1, 0, self.n_bins - 1
        )
        f_rel = freqs[pts] - f_c[bin_id]
        w = 4.0 * df * inv[pts]  # noise weight over the support

        self.h0_edges = jnp.stack(
            [jnp.asarray(h0[k][edges]) for k in keys]
        )  # (M, nb+1)

        A0, A1 = [], []
        for k in keys:
            g = w * d[pts] * np.conj(h0[k][pts])
            A0.append(_bin_reduce(g, bin_id, self.n_bins))
            A1.append(_bin_reduce(g * f_rel, bin_id, self.n_bins))
        self.A0 = jnp.asarray(np.stack(A0))
        self.A1 = jnp.asarray(np.stack(A1))

        M = len(keys)
        B0 = np.zeros((M, M, self.n_bins), dtype=complex)
        B1 = np.zeros((M, M, self.n_bins), dtype=complex)
        for a, ka in enumerate(keys):
            for b, kb in enumerate(keys):
                g = w * h0[ka][pts] * np.conj(h0[kb][pts])
                B0[a, b] = _bin_reduce(g, bin_id, self.n_bins)
                B1[a, b] = _bin_reduce(g * f_rel, bin_id, self.n_bins)
        self.B0 = jnp.asarray(B0)
        self.B1 = jnp.asarray(B1)

    def log_likelihood(self, trial_mode_edges, c):
        """Heterodyned lnL. ``trial_mode_edges`` (M, n_bins+1); ``c`` (M,) coefficients."""
        r = jnp.asarray(trial_mode_edges) / self.h0_edges  # (M, nb+1)
        r0 = 0.5 * (r[:, 1:] + r[:, :-1])
        r1 = (r[:, 1:] - r[:, :-1]) / (self.edge_freqs[1:] - self.edge_freqs[:-1])[
            None, :
        ]
        c = jnp.asarray(c) + 0.0j

        z = jnp.sum(jnp.conj(r0) * self.A0 + jnp.conj(r1) * self.A1, axis=1)  # (M,)
        l_dh = jnp.real(jnp.sum(jnp.conj(c) * z))

        s0 = jnp.einsum("ai,bi,abi->ab", r0, jnp.conj(r0), self.B0)
        s1 = jnp.einsum("ai,bi,abi->ab", r0, jnp.conj(r1), self.B1) + jnp.einsum(
            "ai,bi,abi->ab", r1, jnp.conj(r0), self.B1
        )
        m = s0 + s1
        l_hh = jnp.real(jnp.einsum("a,b,ab->", c, jnp.conj(c), m))
        return -self.half_dd + l_dh - 0.5 * l_hh

    __call__ = log_likelihood


def fd_dense_loglikelihood_modes(
    trial_modes_full, c, data_fd, psd, freqs, f_min, f_max
):
    """Exact dense FD log-likelihood from modes: ``-1/2 (d - h | d - h)``, ``h = sum_a c_a h_a``."""
    U = np.asarray(trial_modes_full, dtype=complex)
    c = np.asarray(c, dtype=complex)
    h = (c[:, None] * U).sum(axis=0)
    freqs = np.asarray(freqs, dtype=float)
    df = float(freqs[1] - freqs[0])
    inv = np.where(
        (freqs >= f_min) & (freqs <= f_max), 1.0 / np.asarray(psd, float), 0.0
    )
    r = np.asarray(data_fd, dtype=complex) - h
    return -2.0 * df * float(np.sum((r.real**2 + r.imag**2) * inv))
