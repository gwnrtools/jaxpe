"""TEOBResumS wrapper (via lalsimulation) satisfying the ``ExternalModeModel`` contract.

Engine choice: this environment provides TEOBResumS through **lalsimulation** (that is
the engine `examples/08 --eob-timing` measured); the native ``EOBRun_module`` python
package is not installed, and LAL's mode interface (``SimInspiralChooseTDModes``) does
not implement TEOBResumS ("generator does not provide a method to generate time-domain
modes", verified). LAL therefore only exposes *polarizations* -- but for a
dominant-(2,2) aligned-spin model that is enough to recover the mode exactly:

    at iota=0, phi=0:  h_+ - i h_x = h_22 * {}_{-2}Y_{22}(0, 0)
    (the (2,-2) harmonic vanishes face-on: {}_{-2}Y_{2,-2}(0, phi) = 0)

so ``h_22 = (h_+ - i h_x)|_faceon / {}_{-2}Y_{22}(0,0)`` and ``h_{2,-2}`` follows from
the planar reflection symmetry. This matches how ``examples/08`` treats LAL TEOBResumS
(registry entry ``highest_m=2``). Higher modes require the native package; if that is
installed later, extend this wrapper rather than trusting LAL polarizations at inclined
angles.

Following the pipeline pattern (see the pseudo-black-box models in
``tests/test_surrogate.py``), the analysis grid and reference time are fixed at
construction; ``__call__`` maps intrinsic parameters to ``ModesData`` on that grid.
"""

import numpy as np

from jaxpe.gw.external_models.base import (
    ExternalModeModel,
    ModesData,
    place_modes_on_grid,
    reflect_modes,
    taper_start,
)


class TEOBResumSModel(ExternalModeModel):
    """Aligned-spin TEOBResumS (2, +-2) modes on a fixed analysis grid, via LAL.

    Parameters
    ----------
    times
        Analysis time grid (uniform, geocentric GPS) the likelihood consumes;
        ``ModesData.times`` will be exactly this array.
    geocent_time
        GPS coalescence time the modes are aligned to (``ModesData.t_ref``).
    d_ref_mpc
        Luminosity distance [Mpc] the stored modes are scaled to.
    f_low, f_ref
        Start and (spin-)reference frequencies [Hz]; ``f_ref=None`` uses ``f_low``.
    taper_seconds
        Half-cosine on-ramp applied to the mode start (turn-on transient control);
        default is four orbits at ``f_low`` (~2 GW cycles of the (2,2)).
    """

    def __init__(
        self,
        times,
        geocent_time: float,
        d_ref_mpc: float = 1000.0,
        f_low: float = 20.0,
        f_ref: float | None = None,
        taper_seconds: float | None = None,
    ):
        import lalsimulation  # noqa: F401  (fail at construction, not first call)

        self.times = np.asarray(times, dtype=float)
        self.t_ref = float(geocent_time)
        self.d_ref_mpc = float(d_ref_mpc)
        self.f_low = float(f_low)
        self.f_ref = float(f_ref) if f_ref is not None else float(f_low)
        self.dt = float(self.times[1] - self.times[0])
        self.taper_seconds = (
            float(taper_seconds) if taper_seconds is not None else 4.0 / self.f_low
        )

    def __call__(self, params_intrinsic: dict) -> ModesData:
        import lal
        import lalsimulation as ls

        m1 = float(params_intrinsic["mass_1"])
        m2 = float(params_intrinsic["mass_2"])
        s1z = float(params_intrinsic.get("spin_1z", 0.0))
        s2z = float(params_intrinsic.get("spin_2z", 0.0))

        hp, hc = ls.SimInspiralChooseTDWaveform(
            m1 * lal.MSUN_SI,
            m2 * lal.MSUN_SI,
            0.0,
            0.0,
            s1z,
            0.0,
            0.0,
            s2z,
            self.d_ref_mpc * 1e6 * lal.PC_SI,
            0.0,  # face-on
            0.0,  # phiRef = 0
            0.0,
            0.0,
            0.0,
            self.dt,
            self.f_low,
            self.f_ref,
            None,
            ls.GetApproximantFromString("TEOBResumS"),
        )
        h_faceon = hp.data.data - 1j * hc.data.data
        # {}_{-2}Y_{22}(0, 0) = sqrt(5/(64 pi)) (1 + cos 0)^2 = sqrt(5/(4 pi)); use the
        # same harmonic implementation the likelihood reconstructs with.
        from jaxpe.gw import spin_weighted_ylm

        y22_faceon = complex(spin_weighted_ylm(0.0, 0.0, 2, 2))
        h22 = taper_start(h_faceon / y22_faceon, self.dt, self.taper_seconds)

        # LAL epoch: sample 0 sits at (epoch) seconds relative to the coalescence.
        epoch = float(hp.epoch.gpsSeconds) + 1e-9 * float(hp.epoch.gpsNanoSeconds)
        t_rel = epoch + self.dt * np.arange(len(h22))

        modes = reflect_modes({(2, 2): h22})
        modes = place_modes_on_grid(modes, t_rel, self.times, self.t_ref)
        return ModesData(
            modes=modes,
            times=self.times,
            d_ref_mpc=self.d_ref_mpc,
            t_ref=self.t_ref,
            f_ref=self.f_ref,
        )
