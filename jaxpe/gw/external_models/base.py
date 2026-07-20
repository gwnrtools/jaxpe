"""Interface for external (non-JAX) waveform models that expose spherical-harmonic modes.

Motivation
----------
Expensive time-domain models (TEOBResumS, SEOBNRv6EHM, ...) cannot enter a JAX trace:
they are opaque, minutes-per-call black boxes. The GPry-fusion design
(``docs/gpry_fusion_design.md``) therefore splits the likelihood at the mode level:

    theta_int  --(external model, plain Python)-->  {h_lm(t)}  --(JAX)-->  lnL

Everything downstream of the modes (detector projection, extrinsic handling,
marginalization) is differentiable JAX code in ``jaxpe.gw.marginalized``; everything
upstream lives here and must NEVER be called inside ``jit``/``grad``.

Conventions
-----------
The complex strain at inclination ``iota`` and reference phase ``phi`` is

    h(t) = h_+ - i h_x = (d_ref / D_L) * sum_{l,m} h_lm(t) * {}_{-2}Y_{lm}(iota, phi)

with the sum over ALL stored (l, m), positive and negative m alike (matching the mode
sum in ``ESIGMAInspiral.__call__``). For non-precessing systems the negative-m modes
follow from the reflection symmetry ``h_{l,-m} = (-1)^l conj(h_{lm})``; use
``reflect_modes`` to fill them in. Any end-of-waveform tapering must already be applied
to the stored modes (the taper is part of the waveform model, not of the likelihood).
"""

import abc
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np


class ModesData(NamedTuple):
    """Spherical-harmonic modes of one waveform evaluation, on a uniform time grid.

    Attributes
    ----------
    modes
        ``{(l, m): complex128 (n,) array}`` — strain modes at ``d_ref_mpc``, including
        negative m (see module docstring).
    times
        (n,) geocentric GPS times, uniformly spaced; must match the analysis grid of
        the likelihood that consumes these modes.
    d_ref_mpc
        Luminosity distance [Mpc] at which the stored modes are scaled.
    t_ref
        The ``geocent_time`` the stored modes are aligned to; the likelihood realizes
        other coalescence times by frequency-domain time shifts relative to this.
    f_ref
        Reference frequency [Hz] defining spin (and eccentricity) conventions, if the
        generating model has one.
    """

    modes: dict
    times: np.ndarray
    d_ref_mpc: float
    t_ref: float
    f_ref: float | None = None


def reflect_modes(modes: dict) -> dict:
    """Fill in missing negative-m modes via ``h_{l,-m} = (-1)^l conj(h_{lm})``.

    Valid for non-precessing (planar) systems only; precessing models must supply
    all modes explicitly.
    """
    out = dict(modes)
    for (l, m), h in modes.items():
        if m != 0 and (l, -m) not in out:
            out[(l, -m)] = (-1.0) ** l * np.conj(h)
    return out


def taper_start(h: np.ndarray, dt: float, taper_seconds: float) -> np.ndarray:
    """Half-cosine on-ramp over the first ``taper_seconds`` of a complex mode.

    TD models switched on at ``f_low`` carry a turn-on transient; the base-class
    contract makes tapering the wrapper's job ("the taper is part of the waveform
    model, not of the likelihood"). A half-cosine in amplitude over a few orbits at
    ``f_low`` is the standard choice; it multiplies the complex strain, so the phase
    is untouched.
    """
    n_taper = int(round(taper_seconds / dt))
    if n_taper <= 1 or n_taper >= len(h):
        return h
    out = np.array(h)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_taper) / n_taper))
    out[:n_taper] = out[:n_taper] * ramp
    return out


def place_modes_on_grid(
    modes: dict, t_rel: np.ndarray, times: np.ndarray, t_ref: float
) -> dict:
    """Place raw model modes (own grid, coalescence at ``t_rel=0``) onto the analysis
    grid ``times`` with coalescence at the GPS time ``t_ref``, zero-padded/truncated.

    ``t_rel`` are the model's sample times relative to its coalescence. The model must
    have been generated at the analysis sampling rate (checked); alignment is to the
    nearest grid sample (sub-sample residual < dt/2 -- the marginalized likelihood
    realizes other coalescence times by FD shifts *relative to t_ref*, so a fixed
    sub-sample epoch offset only shifts the reported t_c, never the intrinsic
    parameters). Samples falling outside the grid are dropped (the early inspiral of a
    signal longer than the segment), which is the same truncation any segmented
    analysis applies.
    """
    times = np.asarray(times, dtype=float)
    t_rel = np.asarray(t_rel, dtype=float)
    dt = times[1] - times[0]
    dt_model = t_rel[1] - t_rel[0]
    if abs(dt_model - dt) > 1e-9 * dt:
        raise ValueError(
            f"model sampling step {dt_model} != analysis grid step {dt}; generate the "
            "waveform at the analysis sampling rate"
        )
    n = len(times)
    j0 = int(round((t_ref + t_rel[0] - times[0]) / dt))
    src_lo = max(0, -j0)
    dst_lo = max(0, j0)
    length = min(len(t_rel) - src_lo, n - dst_lo)
    if length <= 0:
        raise ValueError(
            "waveform does not overlap the analysis grid: check t_ref/duration"
        )
    out = {}
    for lm, h in modes.items():
        placed = np.zeros(n, dtype=np.complex128)
        placed[dst_lo : dst_lo + length] = np.asarray(h, dtype=np.complex128)[
            src_lo : src_lo + length
        ]
        out[lm] = placed
    return out


class ExternalModeModel(abc.ABC):
    """A non-JAX waveform model returning modes for intrinsic parameters.

    Deliberately NOT a ``WaveformModel``: that contract promises JAX traceability,
    which implementations of this class must not pretend to have.
    """

    @abc.abstractmethod
    def __call__(self, params_intrinsic: dict) -> ModesData:
        """Generate modes for one intrinsic-parameter point (plain Python, host-side)."""


class ModeCache:
    """Disk cache of ModesData keyed by the intrinsic-parameter dict.

    Every expensive evaluation is worth keeping: the cache enables the
    importance-sampling reweighting and extrinsic-recovery steps of the GPry fusion
    (design note section 5), and doubles as posterior-concentrated training data for a
    possible later ROM. Storage is one ``.npz`` per evaluation, ~MB each.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(params_intrinsic: dict) -> str:
        blob = json.dumps(
            {k: float(v) for k, v in sorted(params_intrinsic.items())},
            sort_keys=True,
        )
        return hashlib.sha1(blob.encode()).hexdigest()

    def path(self, params_intrinsic: dict) -> Path:
        return self.directory / f"{self.key(params_intrinsic)}.npz"

    def save(self, params_intrinsic: dict, data: ModesData) -> Path:
        lms = sorted(data.modes)
        payload = {
            "lms": np.array(lms, dtype=np.int64),
            "times": np.asarray(data.times),
            "d_ref_mpc": np.float64(data.d_ref_mpc),
            "t_ref": np.float64(data.t_ref),
            "f_ref": np.float64(np.nan if data.f_ref is None else data.f_ref),
            "params_json": np.bytes_(
                json.dumps(
                    {k: float(v) for k, v in sorted(params_intrinsic.items())}
                ).encode()
            ),
        }
        for i, lm in enumerate(lms):
            payload[f"mode_{i}"] = np.asarray(data.modes[lm], dtype=np.complex128)
        path = self.path(params_intrinsic)
        np.savez(path, **payload)
        return path

    def load(self, params_intrinsic: dict) -> ModesData | None:
        path = self.path(params_intrinsic)
        if not path.exists():
            return None
        with np.load(path) as f:
            lms = [tuple(int(x) for x in lm) for lm in f["lms"]]
            f_ref = float(f["f_ref"])
            return ModesData(
                modes={lm: f[f"mode_{i}"] for i, lm in enumerate(lms)},
                times=f["times"],
                d_ref_mpc=float(f["d_ref_mpc"]),
                t_ref=float(f["t_ref"]),
                f_ref=None if np.isnan(f_ref) else f_ref,
            )
