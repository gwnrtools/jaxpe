"""Time-domain IMRPhenomTHM waveform model: modes (2,2),(2,1),(3,3),(4,4),(5,5).

Faithfully reimplemented against LALSuite's LALSimIMRPhenomTHM.c/_internals.c/_fits.c
(Estelles et al. 2020/2021, arXiv:2004.08302, arXiv:2012.11923). Extends
jaxpe.gw.cbc_models.phenomt's IMRPhenomT (the (2,2)-only reconstruction) to the 4 additional
higher modes -- see that module's docstring for the shared (2,2) construction and general
architecture notes (closed-form ansatz, no LALSuite array-bookkeeping needed, the f_ref
phase-reference Newton solve, the accelerator-performance/differentiability discipline). This
module adds only what IMRPhenomT doesn't already cover:

- **Amplitude** for (2,1)/(3,3)/(4,4)/(5,5) reuses the exact same generic construction
  IMRPhenomT already built for (2,2) (``phenomt._amp_hm``: inspiral PN + NR-fitted
  collocation points -> merger sech ansatz -> ringdown tanh*exp envelope) -- only the PN
  amplitude coefficients and NR fits differ per mode.
- **Frequency/phase** for the higher modes reuses the (2,2) mode's phase/frequency
  (``(m/2) * phase22`` in the inspiral region) plus mode-specific merger/ringdown ansätze that
  are *structurally identical* to the (2,2) ones (confirmed by direct comparison of
  ``IMRPhenomTMergerOmegaAnsatzHM``/``RDOmegaAnsatzHM``/``MergerPhaseAnsatzHM``/
  ``RDPhaseAnsatzHM`` against their (2,2)-specific counterparts in LALSuite -- same functional
  form, different per-mode coefficients) -- so ``phenomt._merger_omega_bar_ansatz22`` etc. are
  reused directly rather than reimplemented.
- A **mode-dependent constant phase offset** (0, +pi/2, -pi/2, +pi, +pi/2 for
  (2,2),(2,1),(3,3),(4,4),(5,5)) that rotates each mode's PN amplitude to be mostly real
  (LALSimIMRPhenomTHM.c's ``phoff``), and a **complex-amplitude phase correction**
  (``omegaCutPNAMP``/``phiCutPNAMP``) needed because higher-mode amplitude is complex during
  inspiral (unlike (2,2), whose complex inspiral amplitude phase is folded into the dedicated
  22 phase construction instead and taken in absolute value).
- Odd-``m`` modes vanish *exactly* in the equal-mass, equal-spin limit (a real physical
  symmetry, not a numerical artifact) -- LALSuite special-cases this with a hard zero rather
  than letting the fits extrapolate through the degenerate point; replicated here the same way.
"""

import functools
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from .base import TimeDomainModel
from ..harmonics import spin_weighted_ylm
from ..waveform import MPC_SI as MPC
from ..waveform import MTSUN_SI as MTSUN
from .phenomthm_fits_optimized import (
    compute_all_phenomthm_fits,
    evaluate_QNMfit_fdamp21,
    evaluate_QNMfit_fdamp21n2,
    evaluate_QNMfit_fdamp22,
    evaluate_QNMfit_fdamp22n2,
    evaluate_QNMfit_fdamp33,
    evaluate_QNMfit_fdamp33n2,
    evaluate_QNMfit_fdamp44,
    evaluate_QNMfit_fdamp44n2,
    evaluate_QNMfit_fdamp55,
    evaluate_QNMfit_fdamp55n2,
    evaluate_QNMfit_fring21,
    evaluate_QNMfit_fring33,
    evaluate_QNMfit_fring44,
    evaluate_QNMfit_fring55,
)
from .phenomt import (
    C_SI,
    EULERGAMMA,
    PI,
    _EPS,
    _EXP_ARG_CLIP,
    _FINITE_DIFF_H,
    _T_CUT_AMP,
    _T_MERGER_CP_AMP,
    _amp_hm,
    _inspiral_amp_ansatz_hm,
    _merger_omega_bar_ansatz22,
    _merger_phase_ansatz22,
    _omega22,
    _phase22,
    _rd_omega_ansatz22,
    _rd_phase_ansatz22,
    _solve_22_amplitude_coefficients,
    _solve_22_phase_coefficients,
    _t_ref_from_f_ref,
)

_HM_MODES = ((2, 1), (3, 3), (4, 4), (5, 5))
_ALL_MODES = ((2, 2),) + _HM_MODES
# LALSimIMRPhenomTHM.c: constant phase offset rotating each higher mode's PN amplitude to be
# mostly real -- eq. 13 of the THM paper. (2,2) needs none (dedicated phase construction).
_PHOFF = {(2, 1): 0.5 * PI, (3, 3): -0.5 * PI, (4, 4): PI, (5, 5): 0.5 * PI}
# tCUT_Freq, LALSimIMRPhenomTHM_internals.h -- numerically equal to tCUT_Amp (-150.0), but a
# logically distinct boundary (frequency/phase inspiral-merger cut for the generic HM ansatz,
# vs. the amplitude inspiral-merger cut every mode including (2,2) uses).
_T_CUT_FREQ = -150.0

_QNM_FRING = {
    (2, 1): evaluate_QNMfit_fring21,
    (3, 3): evaluate_QNMfit_fring33,
    (4, 4): evaluate_QNMfit_fring44,
    (5, 5): evaluate_QNMfit_fring55,
}
_QNM_FDAMP = {
    (2, 2): evaluate_QNMfit_fdamp22,
    (2, 1): evaluate_QNMfit_fdamp21,
    (3, 3): evaluate_QNMfit_fdamp33,
    (4, 4): evaluate_QNMfit_fdamp44,
    (5, 5): evaluate_QNMfit_fdamp55,
}
_QNM_FDAMP_N2 = {
    (2, 2): evaluate_QNMfit_fdamp22n2,
    (2, 1): evaluate_QNMfit_fdamp21n2,
    (3, 3): evaluate_QNMfit_fdamp33n2,
    (4, 4): evaluate_QNMfit_fdamp44n2,
    (5, 5): evaluate_QNMfit_fdamp55n2,
}


def _pn_amp_coefficients(l, m, eta, chi1z, chi2z, delta):
    """PN amplitude coefficients for mode (l,m), Appendix A of the THM paper (eq. A1-A5),
    before the global phoff rotation. l,m are static Python ints (never traced) -- this is a
    plain Python dispatch, matching LALSuite's own if/elif structure, not a jnp.where branch.
    Returns the same 17-element tuple layout phenomt.py's (2,2)-only amp_pn uses.
    """
    chi1, chi2 = chi1z, chi2z
    zero = jnp.zeros_like(eta)
    if (l, m) == (2, 2):
        ampN = jnp.ones_like(eta)
        amp0h_re, amp0h_im = zero, zero
        amp1_re = -2.5476190476190474 + (55.0 * eta) / 42.0
        amp1_im = zero
        amp1h_re = (
            (-2.0 * chi1) / 3.0
            - (2.0 * chi2) / 3.0
            - (2.0 * chi1 * delta) / 3.0
            + (2.0 * chi2 * delta) / 3.0
            + (2.0 * chi1 * eta) / 3.0
            + (2.0 * chi2 * eta) / 3.0
            + 2.0 * PI
        )
        amp1h_im = zero
        amp2_re = (
            -1.437169312169312
            + chi1**2 / 2.0
            + chi2**2 / 2.0
            + (chi1**2 * delta) / 2.0
            - (chi2**2 * delta) / 2.0
            - (1069.0 * eta) / 216.0
            - chi1**2 * eta
            + 2.0 * chi1 * chi2 * eta
            - chi2**2 * eta
            + (2047.0 * eta**2) / 1512.0
        )
        amp2_im = zero
        amp2h_re = -(107.0 * PI) / 21.0 + (34.0 * eta * PI) / 21.0
        amp2h_im = -24.0 * eta
        amp3_re = (
            41.78634662956092
            - (278185.0 * eta) / 33264.0
            - (20261.0 * eta**2) / 2772.0
            + (114635.0 * eta**3) / 99792.0
            - (856.0 * EULERGAMMA) / 105.0
            + (2.0 * PI**2) / 3.0
            + (41.0 * eta * PI**2) / 96.0
        )
        amp3_im = (428.0 / 105.0) * PI
        amp3h_re = (-2173.0 * PI) / 756.0 - (2495.0 * eta * PI) / 378.0 + (40.0 * eta**2 * PI) / 27.0
        amp3h_im = (14333.0 * eta) / 162.0 - (4066.0 * eta**2) / 945.0
        amplog = -428.0 / 105.0
    elif (l, m) == (2, 1):
        ampN = zero
        amp0h_re, amp0h_im = delta / 3.0, zero
        amp1_re = -chi1 / 4.0 + chi2 / 4.0 - (chi1 * delta) / 4.0 - (chi2 * delta) / 4.0
        amp1_im = zero
        amp1h_re = -17.0 * delta / 84.0 + (5.0 * delta * eta) / 21.0
        amp1h_im = zero
        amp2_re = (
            (79.0 * chi1) / 84.0
            - (79.0 * chi2) / 84.0
            + (79.0 * chi1 * delta) / 84.0
            + (79.0 * chi2 * delta) / 84.0
            - (43.0 * chi1) / 42.0
            + (43.0 * chi2) / 42.0
            - (43.0 * chi1 * delta) / 42.0
            - (43.0 * chi2 * delta) / 42.0
            - (139.0 * chi1 * eta) / 84.0
            + (139.0 * chi2 * eta) / 84.0
            - (139.0 * chi1 * delta * eta) / 84.0
            - (139.0 * chi2 * delta * eta) / 84.0
            + (86.0 * chi1 * eta) / 21.0
            - (86.0 * chi2 * eta) / 21.0
            + (43.0 * chi1 * delta * eta) / 21.0
            + (43.0 * chi2 * delta * eta) / 21.0
            + (delta * PI) / 3.0
        )
        amp2_im = -delta / 6.0 - (2.0 * delta * jnp.log(2.0)) / 3.0
        amp2h_re = (-43.0 * delta) / 378.0 - (509.0 * delta * eta) / 378.0 + (79.0 * delta * eta**2) / 504.0
        amp2h_im = zero
        amp3_re = (-17.0 * delta * PI) / 84.0 + (delta * eta * PI) / 14.0
        amp3_im = -(
            (-17.0 * delta) / 168.0
            + (353.0 * delta * eta) / 84.0
            - (17.0 * delta * jnp.log(2.0)) / 42.0
            + (delta * eta * jnp.log(2.0)) / 7.0
        )
        amp3h_re, amp3h_im = zero, zero
        amplog = zero
    elif (l, m) == (3, 3):
        ampN = zero
        amp0h_re, amp0h_im = 0.7763237542601484 * delta, zero
        amp1_re, amp1_im = zero, zero
        amp1h_re = -3.1052950170405937 * delta + 1.5526475085202969 * delta * eta
        amp1h_im = zero
        amp2_re = -(
            -0.5822428156951114 * chi1
            + 0.5822428156951114 * chi2
            - 7.316679009572791 * delta
            - 0.5822428156951114 * chi1 * delta
            - 0.5822428156951114 * chi2 * delta
            + 1.3585665699552598 * chi1
            - 1.3585665699552598 * chi2
            + 1.3585665699552598 * chi1 * delta
            + 1.3585665699552598 * chi2 * delta
            + 1.7467284470853341 * chi1 * eta
            - 1.7467284470853341 * chi2 * eta
            + 1.7467284470853341 * chi1 * delta * eta
            + 1.7467284470853341 * chi2 * delta * eta
            - 5.434266279821039 * chi1 * eta
            + 5.434266279821039 * chi2 * eta
            - 2.7171331399105196 * chi1 * delta * eta
            - 2.7171331399105196 * chi2 * delta * eta
        )
        amp2_im = -1.371926598204461 * delta
        amp2h_re = -(
            -0.08680711070363478 * delta
            + 8.647776123213047 * delta * eta
            - 2.0866641516022777 * delta * eta**2
        )
        amp2h_im = zero
        amp3_re, amp3_im = zero, zero
        amp3h_re, amp3h_im = zero, zero
        amplog = zero
    elif (l, m) == (4, 4):
        ampN = zero
        amp0h_re, amp0h_im = zero, zero
        amp1_re = 0.751248226425348 * (1.0 - 3.0 * eta)
        amp1_im = zero
        amp1h_re, amp1h_im = zero, zero
        amp2_re = -4.049910893365739 + 14.489984730901032 * eta - 5.9758381647470875 * eta**2
        amp2_im = zero
        amp2h_re = 0.751248226425348 * (4.0 * PI - 12.0 * eta * PI)
        amp2h_im = 0.751248226425348 * (-2.854822555520438 + 13.189467666561313 * eta)
        amp3_re = -(
            -8.0
            * jnp.power(0.7142857142857143, 0.5)
            * (
                5.338016983016983
                - (1088119.0 * eta) / 28600.0
                + (146879.0 * eta**2) / 2340.0
                - (226097.0 * eta**3) / 17160.0
            )
        ) / 9.0
        amp3_im = zero
        amp3h_re, amp3h_im = zero, zero
        amplog = zero
    elif (l, m) == (5, 5):
        ampN = zero
        amp0h_re, amp0h_im = zero, zero
        amp1_re, amp1_im = zero, zero
        amp1h_re = 0.8013768943966973 * delta * (1.0 - 2.0 * eta)
        amp1h_im = zero
        amp2_re, amp2_im = zero, zero
        amp2h_re = 0.8013768943966973 * delta * (-6.743589743589744 + (688.0 * eta) / 39.0 - (256.0 * eta**2) / 39.0)
        amp2h_im = zero
        amp3_re = 12.58799882096634 * delta - 25.175997641932675 * delta * eta
        amp3_im = -3.0177162096765713 * delta + 12.454250695829877 * delta * eta
        amp3h_re, amp3h_im = zero, zero
        amplog = zero
    else:
        raise ValueError(f"IMRPhenomTHM: unsupported mode {(l, m)}")

    fac0 = 2.0 * eta * jnp.sqrt(16.0 * PI / 5.0)
    return (
        fac0, ampN, amp0h_re, amp0h_im, amp1_re, amp1_im, amp1h_re, amp1h_im,
        amp2_re, amp2_im, amp2h_re, amp2h_im, amp3_re, amp3_im, amp3h_re, amp3h_im, amplog,
    )


def _solve_hm_amplitude_coefficients(l, m, eta, chi1z, chi2z, delta, fits, Mfinal, afinal, phase_c):
    """Mirrors the (l,m) branch (any of the 5 modes) of IMRPhenomTSetHMAmplitudeCoefficients,
    generalizing phenomt.py's (2,2)-only _solve_22_amplitude_coefficients. Also computes the
    complex-amplitude phase-continuity correction (omegaCutPNAMP/phiCutPNAMP) every mode's
    struct carries -- read only by the higher modes' phase (IMRPhenomTHMPhase), but harmless
    (and simpler) to compute uniformly here rather than special-casing (2,2).
    """
    key = f"{l}{m}"
    dtype = eta.dtype if hasattr(eta, "dtype") else jnp.float64
    amp_pn = _pn_amp_coefficients(l, m, eta, chi1z, chi2z, delta)
    amp_insp_cp = jnp.stack(
        [
            fits[f"IMRPhenomT_Inspiral_Amp_CP1_{key}"],
            fits[f"IMRPhenomT_Inspiral_Amp_CP2_{key}"],
            fits[f"IMRPhenomT_Inspiral_Amp_CP3_{key}"],
        ]
    )
    amp_merger_cp1 = fits[f"IMRPhenomT_Merger_Amp_CP1_{key}"]
    amp_peak = fits[f"IMRPhenomT_PeakAmp_{key}"]
    amp_rd_c3 = fits[f"IMRPhenomT_RD_Amp_C3_{key}"]
    tpeak = jnp.zeros_like(eta) if (l, m) == (2, 2) else fits[f"IMRPhenomT_tshift_{key}"]

    fDAMP = _QNM_FDAMP[(l, m)](afinal) / Mfinal
    fDAMPn2 = _QNM_FDAMP_N2[(l, m)](afinal) / Mfinal
    alpha1RD = 2.0 * PI * fDAMP
    alpha2RD = 2.0 * PI * fDAMPn2

    rd_c2 = 0.5 * (alpha2RD - alpha1RD)
    coshc3 = jnp.cosh(amp_rd_c3)
    tanhc3 = jnp.tanh(amp_rd_c3)
    rd_c2 = jnp.where(jnp.abs(rd_c2) > jnp.abs(0.5 * alpha1RD / tanhc3), -0.5 * alpha1RD / tanhc3, rd_c2)
    rd_c1 = amp_peak * alpha1RD * coshc3**2 / rd_c2
    rd_c4 = amp_peak - rd_c1 * tanhc3
    rd_c = (rd_c1, rd_c2, amp_rd_c3, rd_c4)

    def _omega22_at(t):
        return _omega22(
            t, eta, phase_c["tcut22"], phase_c["dtM"], phase_c["pn"], phase_c["insp_c"],
            phase_c["alpha1RD"], phase_c["omega_peak"], phase_c["omega_ring"],
            phase_c["domega_peak"], phase_c["merger_c"], phase_c["rd_c"],
        )

    t_insp_pts = jnp.asarray((-2000.0, -250.0, -150.0), dtype=dtype)
    omega_insp = _omega22_at(t_insp_pts)
    x_insp = jnp.power(0.5 * omega_insp, 2.0 / 3.0)
    zero = jnp.zeros_like(eta)
    zero_c = (zero, zero, zero)
    amp_offset = jnp.real(_inspiral_amp_ansatz_hm(x_insp, amp_pn, zero_c))
    b_insp = (1.0 / amp_pn[0] / x_insp) * (amp_insp_cp - amp_offset)
    A_insp = jnp.stack([x_insp**4, x_insp**4 * jnp.sqrt(x_insp), x_insp**5], axis=1)
    insp_amp_c = jnp.linalg.solve(A_insp, b_insp)

    t_cut_amp = jnp.asarray(_T_CUT_AMP, dtype=dtype)
    omega_at_tcut = _omega22_at(t_cut_amp)
    x_at_tcut = jnp.power(0.5 * omega_at_tcut, 2.0 / 3.0)
    amp_insp_at_cut = _inspiral_amp_ansatz_hm(x_at_tcut, amp_pn, insp_amp_c)
    ampinsp = jnp.sign(jnp.real(amp_insp_at_cut)) * jnp.abs(amp_insp_at_cut)

    tau_cut = t_cut_amp - tpeak
    sech1_cut = 1.0 / jnp.cosh(jnp.clip(alpha1RD * tau_cut, -_EXP_ARG_CLIP, _EXP_ARG_CLIP))
    sech2_cut = 1.0 / jnp.cosh(jnp.clip(2.0 * alpha1RD * tau_cut, -_EXP_ARG_CLIP, _EXP_ARG_CLIP))
    tau_cp = _T_MERGER_CP_AMP - tpeak
    sech1_cp = 1.0 / jnp.cosh(jnp.clip(alpha1RD * tau_cp, -_EXP_ARG_CLIP, _EXP_ARG_CLIP))
    sech2_cp = 1.0 / jnp.cosh(jnp.clip(2.0 * alpha1RD * tau_cp, -_EXP_ARG_CLIP, _EXP_ARG_CLIP))

    omega_at_tcut_m1 = _omega22_at(t_cut_amp - _FINITE_DIFF_H)
    x_at_tcut_m1 = jnp.power(0.5 * omega_at_tcut_m1, 2.0 / 3.0)
    amp_insp_at_cut_m1 = _inspiral_amp_ansatz_hm(x_at_tcut_m1, amp_pn, insp_amp_c)
    amp_insp_signed_m1 = jnp.sign(jnp.real(amp_insp_at_cut_m1)) * jnp.abs(amp_insp_at_cut_m1)
    damp_meco = (ampinsp - amp_insp_signed_m1) / _FINITE_DIFF_H

    tanh_cut = jnp.tanh(jnp.clip(alpha1RD * tau_cut, -_EXP_ARG_CLIP, _EXP_ARG_CLIP))
    sinh2_cut = jnp.sinh(jnp.clip(2.0 * alpha1RD * tau_cut, -_EXP_ARG_CLIP, _EXP_ARG_CLIP))
    aux1 = -alpha1RD * sech1_cut * tanh_cut
    aux2 = (-2.0 / 7.0) * alpha1RD * sinh2_cut * jnp.power(sech2_cut, 8.0 / 7.0)
    aux3 = 2.0 * tau_cut

    one = jnp.ones_like(eta)
    A_merger_amp = jnp.stack(
        [
            jnp.stack([one, sech1_cut, jnp.power(sech2_cut, 1.0 / 7.0), tau_cut**2]),
            jnp.stack([one, sech1_cp, jnp.power(sech2_cp, 1.0 / 7.0), tau_cp**2]),
            jnp.stack([one, one, one, zero]),
            jnp.stack([zero, aux1, aux2, aux3]),
        ]
    )
    b_merger_amp = jnp.stack([ampinsp, amp_merger_cp1, amp_peak, damp_meco])
    merger_c = jnp.linalg.solve(A_merger_amp, b_merger_amp)

    # Complex-amplitude phase-continuity correction (ComplexAmpOrientation, atan2 branch):
    # only read by higher modes' phase, computed here for every mode for one code path.
    phi_x2 = jnp.angle(_inspiral_amp_ansatz_hm(x_at_tcut, amp_pn, insp_amp_c))
    phi_x1 = jnp.angle(_inspiral_amp_ansatz_hm(x_at_tcut_m1, amp_pn, insp_amp_c))
    omega_cut_pn_amp = -(phi_x2 - phi_x1) / _FINITE_DIFF_H
    phi_cut_pn_amp = jnp.arctan2(jnp.imag(amp_insp_at_cut), jnp.real(amp_insp_at_cut))
    phi_cut_pn_amp = jnp.where(jnp.real(amp_insp_at_cut) < 0.0, phi_cut_pn_amp + PI, phi_cut_pn_amp)

    return dict(
        tpeak=tpeak, alpha1RD=alpha1RD, amp_pn=amp_pn, insp_amp_c=insp_amp_c,
        merger_c=merger_c, rd_c=rd_c,
        omega_cut_pn_amp=omega_cut_pn_amp, phi_cut_pn_amp=phi_cut_pn_amp,
    )


def _solve_hm_phase_coefficients(l, m, eta, fits, Mfinal, afinal, amp_c_lm, omega_cut_22, domega_cut_22_raw):
    """Mirrors the (l,m) != (2,2) branch of IMRPhenomTSetHMPhaseCoefficients."""
    key = f"{l}{m}"
    dtype = eta.dtype if hasattr(eta, "dtype") else jnp.float64
    fRING = _QNM_FRING[(l, m)](afinal) / Mfinal
    fDAMP = _QNM_FDAMP[(l, m)](afinal) / Mfinal
    omega_ring = 2.0 * PI * fRING
    alpha1RD = 2.0 * PI * fDAMP

    omega_cut_lm = (m / 2.0) * omega_cut_22
    omega_cut_bar = 1.0 - (omega_cut_lm + amp_c_lm["omega_cut_pn_amp"]) / omega_ring
    omega_merger_cp = 1.0 - fits[f"IMRPhenomT_Merger_Freq_CP1_{key}"] / omega_ring
    omega_peak = fits[f"IMRPhenomT_PeakFrequency_{key}"]

    rd_c3 = fits[f"IMRPhenomT_RD_Freq_D3_{key}"]
    rd_c2 = fits[f"IMRPhenomT_RD_Freq_D2_{key}"]
    rd_c4 = jnp.zeros_like(eta)
    rd_c1 = (1.0 + rd_c3 + rd_c4) * (omega_ring - omega_peak) / rd_c2 / (rd_c3 + 2.0 * rd_c4)
    rd_c = (rd_c1, rd_c2, rd_c3, rd_c4)

    domega_cut_lm = -(m / 2.0) * domega_cut_22_raw / omega_ring
    domega_peak = (
        -(_rd_omega_ansatz22(_FINITE_DIFF_H, rd_c, omega_ring) - _rd_omega_ansatz22(0.0, rd_c, omega_ring))
        / _FINITE_DIFF_H
        / omega_ring
    )

    t_cut = jnp.asarray(_T_CUT_FREQ, dtype=dtype)
    t_merger_cp = jnp.asarray(_T_MERGER_CP_AMP, dtype=dtype)  # tcpMerger: shared by freq & amp
    ascut = jnp.arcsinh(alpha1RD * t_cut)
    as_cp = jnp.arcsinh(alpha1RD * t_merger_cp)
    dencut = jnp.sqrt(1.0 + t_cut**2 * alpha1RD**2)
    A_merger = jnp.stack(
        [
            jnp.stack([ascut**2, ascut**3, ascut**4]),
            jnp.stack([as_cp**2, as_cp**3, as_cp**4]),
            jnp.stack([2.0 * alpha1RD * ascut, 3.0 * alpha1RD * ascut**2, 4.0 * alpha1RD * ascut**3]) / dencut,
        ]
    )
    b_merger = jnp.stack(
        [
            omega_cut_bar - (1.0 - omega_peak / omega_ring) - (domega_peak / alpha1RD) * ascut,
            omega_merger_cp - (1.0 - omega_peak / omega_ring) - (domega_peak / alpha1RD) * as_cp,
            domega_cut_lm - domega_peak / dencut,
        ]
    )
    merger_c = jnp.linalg.solve(A_merger, b_merger)

    return dict(omega_ring=omega_ring, alpha1RD=alpha1RD, omega_peak=omega_peak, domega_peak=domega_peak, merger_c=merger_c, rd_c=rd_c)


def _hm_phase_offsets(phase22_at_tcut, m, phase_c_lm):
    """Phase continuity offsets (phOffMerger, phOffRD), the last piece of
    IMRPhenomTSetHMPhaseCoefficients (needs phase22_at_tcut, i.e. (2,2)'s own phase at
    t=tCUT_Freq, computed once by the caller and shared across all 4 higher modes)."""
    t_cut = jnp.asarray(_T_CUT_FREQ, dtype=phase22_at_tcut.dtype if hasattr(phase22_at_tcut, "dtype") else jnp.float64)
    ph_meco_insp = (m / 2.0) * phase22_at_tcut
    ph_meco_merger = _merger_phase_ansatz22(
        t_cut, phase_c_lm["alpha1RD"], phase_c_lm["omega_peak"], phase_c_lm["omega_ring"],
        phase_c_lm["domega_peak"], phase_c_lm["merger_c"], 0.0,
    )
    ph_off_merger = ph_meco_insp - ph_meco_merger
    ph_off_rd = _merger_phase_ansatz22(
        0.0, phase_c_lm["alpha1RD"], phase_c_lm["omega_peak"], phase_c_lm["omega_ring"],
        phase_c_lm["domega_peak"], phase_c_lm["merger_c"], ph_off_merger,
    )
    return ph_off_merger, ph_off_rd


def _hm_phase(t, m, phase22, phi_cut_pn_amp, c, ph_off_merger, ph_off_rd):
    """IMRPhenomTHMPhase: (m/2)*phase22 in the inspiral region (t < tCUT_Freq), else the
    mode's own merger/ringdown ansatz (reusing phenomt.py's (2,2) closed forms -- same
    functional shape, mode-specific coefficients) minus phiCutPNAMP."""
    ph_insp = (m / 2.0) * phase22
    ph_rd = _rd_phase_ansatz22(t, c["rd_c"], c["omega_ring"], ph_off_rd) - phi_cut_pn_amp
    ph_merger = (
        _merger_phase_ansatz22(t, c["alpha1RD"], c["omega_peak"], c["omega_ring"], c["domega_peak"], c["merger_c"], ph_off_merger)
        - phi_cut_pn_amp
    )
    return jnp.where(t < _T_CUT_FREQ, ph_insp, jnp.where(t > 0.0, ph_rd, ph_merger))


class IMRPhenomTHM(TimeDomainModel):
    """Time-domain IMRPhenomTHM: aligned-spin BBH, modes (2,2),(2,1),(3,3),(4,4),(5,5)."""

    is_fd = False

    def __init__(self, f_ref: float = 20.0):
        self.f_ref = f_ref

    @functools.partial(jax.jit, static_argnums=(0,))
    def mode_dict(self, params: dict, grid: jax.Array) -> Dict[Tuple[int, int], jax.Array]:
        """Returns {(l,+-m): h_lm(t)} for all 5 modes, strain at 1 Mpc."""
        mc = jnp.maximum(params["chirp_mass"], 1.0)
        q = jnp.clip(params["mass_ratio"], 0.01, 1.0)
        eta = jnp.clip(q / (1.0 + q) ** 2, 1e-4, 0.25)
        M = mc / (eta ** (3.0 / 5.0))

        s1z = jnp.clip(params.get("spin1z", jnp.zeros(())), -0.99, 0.99)
        s2z = jnp.clip(params.get("spin2z", jnp.zeros(())), -0.99, 0.99)
        delta = jnp.sqrt(jnp.maximum(1.0 - 4.0 * eta, 0.0))
        m1 = 0.5 * (1.0 + delta)
        m2 = 0.5 * (1.0 - delta)
        S = (m1 * m1 * s1z + m2 * m2 * s2z) / (m1 * m1 + m2 * m2)
        dchi = s1z - s2z

        tc = params.get("geocent_time", 0.0)
        t_geom = (grid - tc) / (M * MTSUN)
        dtM = jnp.min(jnp.diff(t_geom)) if t_geom.shape[0] > 1 else 0.0

        fits = compute_all_phenomthm_fits(eta, S, dchi, delta)
        Mfinal = fits["IMRPhenomX_FinalMass2017"]
        afinal = fits["IMRPhenomX_FinalSpin2017"]

        # --- (2,2): the dedicated construction from phenomt.py, reused directly ---
        phase_c = _solve_22_phase_coefficients(eta, s1z, s2z, delta, S, dchi, fits, Mfinal, afinal, dtM)
        amp_c22 = _solve_22_amplitude_coefficients(eta, s1z, s2z, delta, fits, Mfinal, afinal, phase_c)
        pn, insp_c = phase_c["pn"], phase_c["insp_c"]

        def _omega22_at(t):
            return _omega22(
                t, eta, phase_c["tcut22"], phase_c["dtM"], pn, insp_c, phase_c["alpha1RD"],
                phase_c["omega_peak"], phase_c["omega_ring"], phase_c["domega_peak"],
                phase_c["merger_c"], phase_c["rd_c"],
            )

        def _phase22_at(t):
            return _phase22(
                t, eta, phase_c["tcut22"], phase_c["dtM"], pn, insp_c, phase_c["alpha1RD"],
                phase_c["omega_peak"], phase_c["omega_ring"], phase_c["domega_peak"],
                phase_c["merger_c"], phase_c["rd_c"], phase_c["ph_off_merger"], phase_c["ph_off_rd"],
            )

        omega22 = _omega22_at(t_geom)
        phase22 = _phase22_at(t_geom)
        x = jnp.power(jnp.maximum(0.5 * omega22, _EPS), 2.0 / 3.0)
        amp_factor = (M * MTSUN * C_SI) / MPC

        omega_f_ref = 2.0 * PI * self.f_ref * (M * MTSUN)
        t_ref = _t_ref_from_f_ref(eta, pn, insp_c, omega_f_ref)
        phiref0 = _phase22_at(t_ref)

        out = {}
        amp22 = jnp.abs(
            _amp_hm(t_geom, x, amp_c22["tpeak"], amp_c22["alpha1RD"], amp_c22["amp_pn"], amp_c22["insp_amp_c"], amp_c22["merger_c"], amp_c22["rd_c"])
        )
        h22 = amp_factor * amp22 * jnp.exp(-1j * (phase22 - phiref0))
        out[(2, 2)] = h22
        out[(2, -2)] = jnp.conj(h22)

        # --- Higher modes: shared (2,2)-derived boundary quantities, computed once ---
        dtype = eta.dtype if hasattr(eta, "dtype") else jnp.float64
        t_cut_freq = jnp.asarray(_T_CUT_FREQ, dtype=dtype)
        omega_cut_22 = _omega22_at(t_cut_freq)
        domega_cut_22_raw = (omega_cut_22 - _omega22_at(t_cut_freq - _FINITE_DIFF_H)) / _FINITE_DIFF_H
        phase22_at_tcut = _phase22_at(t_cut_freq)

        # Odd-m modes vanish exactly in the equal-mass, equal-spin limit (LALSimIMRPhenomTHM.c:
        # a real physical symmetry, replicated as a hard zero rather than letting the
        # NR fits/PN coefficients extrapolate through the degenerate point).
        is_degenerate_odd = (delta < 1e-10) & (jnp.abs(s1z - s2z) < 1e-10)

        for l, m in _HM_MODES:
            amp_c_lm = _solve_hm_amplitude_coefficients(l, m, eta, s1z, s2z, delta, fits, Mfinal, afinal, phase_c)
            phase_c_lm = _solve_hm_phase_coefficients(l, m, eta, fits, Mfinal, afinal, amp_c_lm, omega_cut_22, domega_cut_22_raw)
            ph_off_merger, ph_off_rd = _hm_phase_offsets(phase22_at_tcut, m, phase_c_lm)

            amp_lm = _amp_hm(
                t_geom, x, amp_c_lm["tpeak"], amp_c_lm["alpha1RD"], amp_c_lm["amp_pn"],
                amp_c_lm["insp_amp_c"], amp_c_lm["merger_c"], amp_c_lm["rd_c"],
            )  # complex -- unlike (2,2), NOT taken in absolute value (its phase is physical)
            philm = (
                _hm_phase(t_geom, m, phase22, amp_c_lm["phi_cut_pn_amp"], phase_c_lm, ph_off_merger, ph_off_rd)
                - (m / 2.0) * phiref0
                - _PHOFF[(l, m)]
            )
            h_lm = amp_factor * amp_lm * jnp.exp(-1j * philm)
            if m % 2 != 0:
                # Only odd-m modes vanish at this symmetry point (m even, e.g. (4,4), does
                # not) -- m is a static Python int here, so this is a plain Python
                # conditional over which jnp.where to build, not a traced branch.
                h_lm = jnp.where(is_degenerate_odd, 0.0 + 0.0j, h_lm)

            out[(l, m)] = h_lm
            out[(l, -m)] = ((-1.0) ** l) * jnp.conj(h_lm)

        return out

    @functools.partial(jax.jit, static_argnums=(0,))
    def __call__(self, params: dict, grid: jax.Array) -> Tuple[jax.Array, jax.Array]:
        modes = self.mode_dict(params, grid)

        dist = params.get("luminosity_distance", 100.0)
        phi_ref = params.get("phase", 0.0)
        iota = params.get("inclination", 0.0)
        dist_scale = 1.0 / dist

        # phi_ref enters through each Ylm's own azimuthal argument (pi/2 - phiRef, LALSuite's
        # XLALSimAddMode convention -- see phenomt.py's __call__ docstring), not a separate
        # per-mode exp(i*m*phi_ref) rotation.
        phi_ylm = PI / 2.0 - phi_ref

        hp = jnp.zeros_like(grid)
        hc = jnp.zeros_like(grid)
        for l, m in _ALL_MODES:
            for mm in (m, -m):
                h_lm = modes[(l, mm)] * dist_scale
                Y_lm = spin_weighted_ylm(iota, phi_ylm, l, mm, s=-2)
                h_comp = h_lm * Y_lm
                hp = hp + jnp.real(h_comp)
                hc = hc - jnp.imag(h_comp)

        return hp, hc
