"""pyseobnr (SEOBNRv5 family) wrapper satisfying the ``ExternalModeModel`` contract.

``pyseobnr.generate_waveform.GenerateWaveform.generate_td_modes`` returns inertial-frame
modes on the model's own grid with ``t=0`` at the waveform peak; this wrapper generates
at the analysis sampling rate, tapers the turn-on, and places the modes onto the fixed
analysis grid with the peak at ``geocent_time`` (the pipeline pattern: grid and
reference time are fixed at construction, ``__call__`` maps intrinsic parameters to
``ModesData`` on that grid).

Aligned-spin only for now: negative-m modes are filled by planar reflection, which is
wrong for precessing systems -- the wrapper raises if in-plane spins are passed.
"""

import numpy as np

from jaxpe.gw.external_models.base import (
    ExternalModeModel,
    ModesData,
    place_modes_on_grid,
    reflect_modes,
    taper_start,
)

# SEOBNRv5HM's native mode content capped at (l, m) <= (4, 4), matching the timing
# methodology in examples/08 (`_MODE_ARRAY_44` plus the (3,2)/(4,3) modes v5HM carries).
_DEFAULT_MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]


class PySEOBNRModel(ExternalModeModel):
    """SEOBNR (pyseobnr) modes on a fixed analysis grid.

    Parameters
    ----------
    times, geocent_time, d_ref_mpc, f_low, f_ref, taper_seconds
        As in ``TEOBResumSModel``: the analysis grid, the GPS time the waveform peak is
        aligned to, the stored-mode reference distance, start/reference frequencies,
        and the turn-on taper (default four orbits at ``f_low``).
    approximant
        Any TD pyseobnr approximant ("SEOBNRv5HM", "SEOBNRv5EHM", ...).
    mode_array
        ``[(l, m), ...]`` with m > 0 to request from pyseobnr (negative m by
        reflection); ``None`` uses the (4,4)-capped v5HM set.
    """

    def __init__(
        self,
        times,
        geocent_time: float,
        d_ref_mpc: float = 1000.0,
        f_low: float = 20.0,
        f_ref: float | None = None,
        taper_seconds: float | None = None,
        approximant: str = "SEOBNRv5HM",
        mode_array: list | None = None,
    ):
        from pyseobnr.generate_waveform import GenerateWaveform  # noqa: F401

        self.times = np.asarray(times, dtype=float)
        self.t_ref = float(geocent_time)
        self.d_ref_mpc = float(d_ref_mpc)
        self.f_low = float(f_low)
        self.f_ref = float(f_ref) if f_ref is not None else float(f_low)
        self.dt = float(self.times[1] - self.times[0])
        self.taper_seconds = (
            float(taper_seconds) if taper_seconds is not None else 4.0 / self.f_low
        )
        self.approximant = approximant
        self.mode_array = (
            list(mode_array) if mode_array is not None else list(_DEFAULT_MODES)
        )

    def __call__(self, params_intrinsic: dict) -> ModesData:
        from pyseobnr.generate_waveform import GenerateWaveform

        m1 = float(params_intrinsic["mass_1"])
        m2 = float(params_intrinsic["mass_2"])
        s1z = float(params_intrinsic.get("spin_1z", 0.0))
        s2z = float(params_intrinsic.get("spin_2z", 0.0))
        for k in ("spin_1x", "spin_1y", "spin_2x", "spin_2y"):
            if float(params_intrinsic.get(k, 0.0)) != 0.0:
                raise ValueError(
                    "PySEOBNRModel is aligned-spin only (negative-m modes are filled "
                    f"by planar reflection); got nonzero {k}"
                )

        params = dict(
            mass1=m1,
            mass2=m2,
            spin1x=0.0,
            spin1y=0.0,
            spin1z=s1z,
            spin2x=0.0,
            spin2y=0.0,
            spin2z=s2z,
            deltaT=self.dt,
            f22_start=self.f_low,
            f_ref=self.f_ref,
            distance=self.d_ref_mpc,
            inclination=0.0,
            phi_ref=0.0,
            approximant=self.approximant,
            # pyseobnr validates requested modes against a set of (l, m) *tuples*
            # (SEOBNRv5Base), so pass tuples, not lists.
            mode_array=[tuple(lm) for lm in self.mode_array],
        )
        if "eccentricity" in params_intrinsic:
            params["eccentricity"] = float(params_intrinsic["eccentricity"])

        t_rel, hlm = GenerateWaveform(params).generate_td_modes()

        modes = {}
        for key, h in hlm.items():
            lm = (
                tuple(int(x) for x in key.split(","))
                if isinstance(key, str)
                else (int(key[0]), int(key[1]))
            )
            if lm[1] <= 0:  # keep m > 0 only; reflection restores negative m
                continue
            modes[lm] = taper_start(
                np.asarray(h, dtype=np.complex128), self.dt, self.taper_seconds
            )
        modes = reflect_modes(modes)
        modes = place_modes_on_grid(
            modes, np.asarray(t_rel, dtype=float), self.times, self.t_ref
        )
        return ModesData(
            modes=modes,
            times=self.times,
            d_ref_mpc=self.d_ref_mpc,
            t_ref=self.t_ref,
            f_ref=self.f_ref,
        )
