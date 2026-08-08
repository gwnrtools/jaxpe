"""Time-domain IMRPhenomT (dominant (2,2) mode) waveform model.

Faithfully reimplemented against LALSuite's LALSimIMRPhenomTHM_internals.c (Estelles et al.
2020, arXiv:2004.08302; the (2,2)-only reconstruction IMRPhenomT shares with IMRPhenomTHM,
selected there via an ``only22`` flag). See docs/constants.md for why this replaced a
placeholder (``_compute_phenom_coefficients`` used to ignore its eta/spin arguments entirely),
and the IMRPhenomT reimplementation plan for the port's scope and verification.

Structure -- all closed-form algebra, plus one small Newton solve (see below), no numerical
integration anywhere:

- **Inspiral**: 3.5PN TaylorT3 (pure PN theory, not a fit) plus 6 higher-order correction
  terms solved from 5 NR-fitted frequency collocation points via a 6x6 linear system
  (``jnp.linalg.solve``, replacing LALSuite's GSL LU decomposition).
- **Merger**: an ``arcsinh(alpha_damp * t)``-parametrized frequency ansatz with 3 free
  coefficients solved via a 3x3 system (continuity with the inspiral boundary + 1 NR-fitted
  collocation point + differentiability with the ringdown ansatz).
- **Ringdown**: Damour & Nagar 2014's tanh/log ansatz (arXiv:1406.0401 eq. 5, 9), closed-form
  given the final-BH QNM ringdown/damping frequency.
- **Phase** is the *exact analytic integral* of each region's frequency ansatz (verified
  against LALSuite's own closed forms, not re-derived) -- so evaluating it never needs
  ``cumsum``/``lax.scan`` phase-integration the way the old placeholder did.
- **Amplitude** reuses the same generic mode-amplitude construction IMRPhenomTHM uses for
  every mode (inspiral PN + 3 NR-fitted collocation points -> merger sech-based ansatz ->
  ringdown ``tanh * exp(-t/tau)`` QNM envelope), specialized here to (l,m)=(2,2) -- LALSuite
  does not special-case (2,2) amplitude either.

Deliberately out of scope (see the plan): LALSuite's optional 4-region inspiral split
(``inspVersion != 0``, an early/late TaylorT3 boundary) is off by default upstream and would
add a second data-dependent branch for no accuracy benefit in the default-used mode -- only
the default single-region (``t0=0``) reconstruction is implemented. LALSuite's own GSL
root-finding for the waveform's start time/array length is not needed: jaxpe's
``mode_dict(params, grid)`` convention already takes a caller-supplied time grid
(``geocent_time``-aligned to t=0, LALSuite's own convention for the (2,2) peak-amplitude
time), so there is nothing to solve for there. A *different*, small root-find is still needed
for the phase: LALSuite anchors the reference phase (``params["phase"]``) at the moment the
GW frequency crosses ``f_ref``, not at t=0/merger -- ``_t_ref_from_f_ref`` finds that time via
a fixed-iteration-count Newton solve on the (smooth, monotonic) inspiral frequency ansatz, so
it stays jit/vmap-safe (no data-dependent stopping condition). Skipping this reference-phase
alignment was tried first and measured wrong: frequency/amplitude alone already matched
LALSuite to ~1e-6 mismatch, but the *phase* was off by a mass/spin-dependent constant (the
accumulated phase between f_ref and merger) without it.

Final-BH mass/spin use ``XLALSimIMRPhenomXFinalMass2017``/``FinalSpin2017`` (Jimenez-Forteza
et al 2017, arXiv:1611.00332) -- IMRPhenomTHM's own choice, a different fit family from this
package's ``phenomd.py`` sibling's ``EradRational0815``/``FinalSpin0815`` (IMRPhenomD's own
final-state fit).
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
    evaluate_QNMfit_fdamp22,
    evaluate_QNMfit_fdamp22n2,
    evaluate_QNMfit_fring22,
)

PI = jnp.pi
EULERGAMMA = 0.577215664901532860606512090082402431
C_SI = 299792458.0

_EPS = 1e-12
# LALSimIMRPhenomTHM_internals.h: fixed collocation-point/boundary locations (theta
# parametrization for frequency/phase; geometric-mass-units time for amplitude/merger).
_THETA_POINTS = (0.33, 0.45, 0.55, 0.65, 0.75, 0.82)
_THETA_CUT = 0.81  # 22 inspiral-merger boundary (frequency/phase)
_THETA_MERGER_CP = 0.95  # 22 merger frequency/phase collocation point
_T_CUT_AMP = -150.0  # tCUT_Amp: every mode's inspiral-merger amplitude boundary
_T_INSP_AMP_CP = (-2000.0, -250.0, -150.0)  # amplitude inspiral collocation-point times
_T_MERGER_CP_AMP = -25.0  # tcpMerger: merger amplitude collocation-point time
_FINITE_DIFF_H = 1e-7  # boundary-derivative step, matching LALSuite's 0.0000001


_EXP_ARG_CLIP = 500.0  # safely under float64's ~709 overflow threshold, with margin


def _safe_exp(x):
    """exp(clip(x)) -- every ringdown/merger closed form below is only physically meaningful
    in its own region, but jnp.where evaluates *every* branch everywhere (that's what makes it
    vmap/jit-safe): at deep-inspiral t, the ringdown/merger branches' exp/cosh arguments would
    otherwise overflow to inf. The forward *value* there is discarded by jnp.where regardless,
    but jnp.where's VJP still computes a gradient contribution for the unselected branch
    (weighted by 0) -- 0 * inf is NaN, not 0, so an unclipped inf here poisons the gradient of
    every parameter even though it never affects the forward pass. Clipping first keeps the
    unselected branch's gradient at a large-but-finite (and irrelevant) value instead of NaN.
    """
    return jnp.exp(jnp.clip(x, -_EXP_ARG_CLIP, _EXP_ARG_CLIP))


def _t_of_theta(theta, eta):
    return -5.0 / (eta * jnp.power(theta, 8))


def _theta_of_t(t, eta):
    """theta = (eta*(-t)/5)^(-1/8), t<0 (guarded for t>=0, where dispatch never reads it)."""
    return jnp.power(eta * jnp.maximum(-t, _EPS) / 5.0, -1.0 / 8.0)


def _thetabar_of_t(t, eta):
    """thetabar = (eta*(-t))^(-1/8) -- the parametrization the inspiral PHASE ansatz uses."""
    return jnp.power(eta * jnp.maximum(-t, _EPS), -1.0 / 8.0)


def _pn_omega_coefficients(eta, chi1L, chi2L, delta):
    """3.5PN TaylorT3 coefficients (Appendix A1, Estelles et al 2020, arXiv:2004.08302.pdf).

    Exact PN theory, not an NR-calibrated fit -- transcribed verbatim from
    IMRPhenomTSetPhase22Coefficients in LALSimIMRPhenomTHM_internals.c.
    """
    omega1PN = 0.27641369047619047 + (11.0 * eta) / 32.0
    omega1halfPN = (-19.0 * (chi1L + chi2L) * eta) / 80.0 + (
        -113.0 * (chi2L * (-1.0 + delta) - chi1L * (1.0 + delta)) - 96.0 * PI
    ) / 320.0
    omega2PN = (
        (
            1855099.0
            + 1714608.0 * chi2L**2 * (-1.0 + delta)
            - 1714608.0 * chi1L**2 * (1.0 + delta)
        )
        / 1.4450688e7
        + (
            (
                56975.0
                + 61236.0 * chi1L**2
                - 119448.0 * chi1L * chi2L
                + 61236.0 * chi2L**2
            )
            * eta
        )
        / 258048.0
        + (371.0 * eta**2) / 2048.0
    )
    omega2halfPN = (
        (-17.0 * (chi1L + chi2L) * eta**2) / 128.0
        + (-146597.0 * (chi2L * (-1.0 + delta) - chi1L * (1.0 + delta)) - 46374.0 * PI)
        / 129024.0
        + (
            eta
            * (
                -2.0
                * (chi1L * (1213.0 - 63.0 * delta) + chi2L * (1213.0 + 63.0 * delta))
                + 117.0 * PI
            )
        )
        / 2304.0
    )
    omega3PN = (
        -2.499258364444952
        - (16928263.0 * chi1L**2) / 1.376256e8
        - (16928263.0 * chi2L**2) / 1.376256e8
        - (16928263.0 * chi1L**2 * delta) / 1.376256e8
        + (16928263.0 * chi2L**2 * delta) / 1.376256e8
        + (
            (
                -2318475.0
                + 18767224.0 * chi1L**2
                - 54663952.0 * chi1L * chi2L
                + 18767224.0 * chi2L**2
            )
            * eta**2
        )
        / 1.376256e8
        + (235925.0 * eta**3) / 1.769472e6
        + (107.0 * EULERGAMMA) / 280.0
        - (6127.0 * chi1L * PI) / 12800.0
        - (6127.0 * chi2L * PI) / 12800.0
        - (6127.0 * chi1L * delta * PI) / 12800.0
        + (6127.0 * chi2L * delta * PI) / 12800.0
        + (53.0 * PI**2) / 200.0
        + (
            eta
            * (
                632550449425.0
                + 35200873512.0 * chi1L**2
                - 28527282000.0 * chi1L * chi2L
                + 9605339856.0 * chi1L**2 * delta
                - 1512.0 * chi2L**2 * (-23281001.0 + 6352738.0 * delta)
                + 34172264448.0 * (chi1L + chi2L) * PI
                - 22912243200.0 * PI**2
            )
        )
        / 1.040449536e11
        + (107.0 * jnp.log(2.0)) / 280.0
    )
    omega3halfPN = (
        (-12029.0 * (chi1L + chi2L) * eta**3) / 92160.0
        + (
            eta**2
            * (
                507654.0 * chi1L * chi2L**2
                - 838782.0 * chi2L**3
                + chi2L * (-840149.0 + 507654.0 * chi1L**2 - 870576.0 * delta)
                + chi1L * (-840149.0 - 838782.0 * chi1L**2 + 870576.0 * delta)
                + 1701228.0 * PI
            )
        )
        / 1.548288e7
        + (
            eta
            * (
                218532006.0 * chi1L * chi2L**2 * (-1.0 + delta)
                - 1134.0 * chi2L**3 * (-206917.0 + 71931.0 * delta)
                - chi2L
                * (
                    1496368361.0
                    - 429508815.0 * delta
                    + 218532006.0 * chi1L**2 * (1.0 + delta)
                )
                + chi1L
                * (
                    -1496368361.0
                    - 429508815.0 * delta
                    + 1134.0 * chi1L**2 * (206917.0 + 71931.0 * delta)
                )
                - 144.0
                * (
                    488825.0
                    + 923076.0 * chi1L**2
                    - 1782648.0 * chi1L * chi2L
                    + 923076.0 * chi2L**2
                )
                * PI
            )
        )
        / 1.8579456e8
        + (
            -6579635551.0 * chi2L * (-1.0 + delta)
            + 535759434.0 * chi2L**3 * (-1.0 + delta)
            - chi1L * (-6579635551.0 + 535759434.0 * chi1L**2) * (1.0 + delta)
            + (
                -565550067.0
                - 465230304.0 * chi2L**2 * (-1.0 + delta)
                + 465230304.0 * chi1L**2 * (1.0 + delta)
            )
            * PI
        )
        / 1.30056192e9
    )
    return (omega1PN, omega1halfPN, omega2PN, omega2halfPN, omega3PN, omega3halfPN)


def _taylor_t3_omega(theta, pn):
    """IMRPhenomTTaylorT3: leading-order + 3.5PN TaylorT3 frequency, in the PN parameter theta."""
    omega1PN, omega1halfPN, omega2PN, omega2halfPN, omega3PN, omega3halfPN = pn
    theta2 = theta * theta
    theta3 = theta2 * theta
    theta4 = theta2 * theta2
    theta5 = theta3 * theta2
    theta6 = theta3 * theta3
    theta7 = theta4 * theta3
    fac = theta3 / 8.0
    logterm = (107.0 * jnp.log(theta)) / 280.0
    out = (
        1.0
        + omega1PN * theta2
        + omega1halfPN * theta3
        + omega2PN * theta4
        + omega2halfPN * theta5
        + omega3PN * theta6
        + logterm * theta6
        + omega3halfPN * theta7
    )
    return 2.0 * fac * out


def _inspiral_omega_ansatz22(theta, pn, insp_c):
    """IMRPhenomTInspiralOmegaAnsatz22: TaylorT3 + 6 NR-calibrated higher-order corrections."""
    c1, c2, c3, c4, c5, c6 = insp_c
    theta8 = jnp.power(theta, 8)
    theta9 = theta8 * theta
    theta10 = theta9 * theta
    theta11 = theta10 * theta
    theta12 = theta11 * theta
    theta13 = theta12 * theta
    fac = theta * theta * theta / 8.0
    taylort3 = _taylor_t3_omega(theta, pn)
    out = (
        c1 * theta8
        + c2 * theta9
        + c3 * theta10
        + c4 * theta11
        + c5 * theta12
        + c6 * theta13
    )
    return taylort3 + 2.0 * fac * out


def _inspiral_phase_ansatz22(t, thetabar, eta, pn, insp_c, ph_off_insp):
    """IMRPhenomTInspiralPhaseAnsatz22: exact analytic integral of the inspiral frequency ansatz."""
    omega1PN, omega1halfPN, omega2PN, omega2halfPN, omega3PN, omega3halfPN = pn
    c1, c2, c3, c4, c5, c6 = insp_c
    aux = (
        -(
            jnp.power(5.0, -0.625)
            * jnp.power(eta, -2)
            * jnp.power(t, -1)
            * jnp.power(thetabar, -7)
            * (
                3.0 * (-107.0 + 280.0 * omega3PN) * jnp.power(5.0, 0.75)
                + 321.0
                * jnp.log(thetabar * jnp.power(5.0, 0.125))
                * jnp.power(5.0, 0.75)
                + 420.0 * omega3halfPN * thetabar * jnp.power(5.0, 0.875)
                + 56.0 * (25.0 * c1 + 3.0 * eta * t) * jnp.power(thetabar, 2)
                + 1050.0 * c2 * jnp.power(5.0, 0.125) * jnp.power(thetabar, 3)
                + 280.0
                * (3.0 * c3 + eta * omega1PN * t)
                * jnp.power(5.0, 0.25)
                * jnp.power(thetabar, 4)
                + 140.0
                * (5.0 * c4 + 3.0 * eta * omega1halfPN * t)
                * jnp.power(5.0, 0.375)
                * jnp.power(thetabar, 5)
                + 120.0
                * (5.0 * c5 + 7.0 * eta * omega2PN * t)
                * jnp.power(5.0, 0.5)
                * jnp.power(thetabar, 6)
                + 525.0 * c6 * jnp.power(5.0, 0.625) * jnp.power(thetabar, 7)
                + 105.0
                * eta
                * omega2halfPN
                * t
                * jnp.log(-t)
                * jnp.power(5.0, 0.625)
                * jnp.power(thetabar, 7)
            )
        )
        / 84.0
    )
    return aux + ph_off_insp


def _rd_omega_ansatz22(t, rd_c, omega_ring):
    """IMRPhenomTRDOmegaAnsatz22."""
    c1, c2, c3, c4 = rd_c
    # expC2 = _safe_exp(-2*c2*t), clipped independently -- squaring an already-clipped expC
    # (exp(500)**2) still overflows float64, since the clip bound alone doesn't account for
    # the later squaring.
    expC = _safe_exp(-c2 * t)
    expC2 = _safe_exp(-2.0 * c2 * t)
    num = c1 * (-2.0 * c2 * c4 * expC2 - c2 * c3 * expC)
    den = 1.0 + c4 * expC2 + c3 * expC
    return num / den + omega_ring


def _rd_phase_ansatz22(t, rd_c, omega_ring, ph_off_rd):
    """IMRPhenomTRDPhaseAnsatz22 (Damour & Nagar 2014, eq. 5)."""
    c1, c2, c3, c4 = rd_c
    expC = _safe_exp(-c2 * t)
    expC2 = _safe_exp(-2.0 * c2 * t)  # independent clip -- see _rd_omega_ansatz22
    num = 1.0 + c3 * expC + c4 * expC2
    den = 1.0 + c3 + c4
    return c1 * jnp.log(num / den) + omega_ring * t + ph_off_rd


def _merger_omega_bar_ansatz22(
    t, alpha1RD, omega_peak, omega_ring, domega_peak, merger_c
):
    """IMRPhenomTMergerOmegaAnsatz22: rescaled frequency bar-omega = 1 - omega/omega_ring."""
    c1, c2, c3 = merger_c
    x = jnp.arcsinh(alpha1RD * t)
    x2, x3, x4 = x * x, x * x * x, x * x * x * x
    return (
        1.0
        - omega_peak / omega_ring
        + (domega_peak / alpha1RD) * x
        + c1 * x2
        + c2 * x3
        + c3 * x4
    )


def _merger_phase_ansatz22(
    t, alpha1RD, omega_peak, omega_ring, domega_peak, merger_c, ph_off_merger
):
    """IMRPhenomTMergerPhaseAnsatz22: exact analytic integral of the merger frequency ansatz."""
    cc, dd, ee = merger_c
    x = jnp.arcsinh(alpha1RD * t)
    sq = jnp.sqrt(1.0 + alpha1RD**2 * t**2)
    aux = omega_ring * t - omega_ring * (
        2.0 * cc * t
        + 24.0 * ee * t
        + 6.0 * dd * t * x
        + domega_peak * t * x / alpha1RD
        + t * (1.0 - omega_peak / omega_ring)
        + cc * t * x**2
        + 12.0 * ee * t * x**2
        + dd * t * x**3
        + ee * t * x**4
        - domega_peak / alpha1RD**2 * sq
        - 6.0 * dd / alpha1RD * sq
        - 2.0 * cc * x / alpha1RD * sq
        - 24.0 * ee * x / alpha1RD * sq
        - 3.0 * dd / alpha1RD * x**2 * sq
        - 4.0 * ee / alpha1RD * x**3 * sq
    )
    return aux + ph_off_merger


def _omega22(
    t,
    eta,
    tcut22,
    dtM,
    pn,
    insp_c,
    alpha1RD,
    omega_peak,
    omega_ring,
    domega_peak,
    merger_c,
    rd_c,
):
    theta = _theta_of_t(t, eta)
    w_insp = _inspiral_omega_ansatz22(theta, pn, insp_c)
    w_rd = _rd_omega_ansatz22(t, rd_c, omega_ring)
    w_merger_bar = _merger_omega_bar_ansatz22(
        t, alpha1RD, omega_peak, omega_ring, domega_peak, merger_c
    )
    w_merger = omega_ring * (1.0 - w_merger_bar)
    return jnp.where(t < tcut22 - dtM, w_insp, jnp.where(t > 0.0, w_rd, w_merger))


def _phase22(
    t,
    eta,
    tcut22,
    dtM,
    pn,
    insp_c,
    alpha1RD,
    omega_peak,
    omega_ring,
    domega_peak,
    merger_c,
    rd_c,
    ph_off_merger,
    ph_off_rd,
):
    thetabar = _thetabar_of_t(t, eta)
    # _inspiral_phase_ansatz22 has a log(-t) term and a power(t, -1) term -- both undefined
    # (log) or singular (power) for t>=0. jnp.where evaluates this branch everywhere (that's
    # what makes it vmap/jit-safe), so even though it's discarded for t>=0 by the dispatch
    # below, an unguarded log(-t) there produces an actual NaN *value* (not just a large
    # finite one, unlike the exp-overflow case _safe_exp guards against) -- and 0 * NaN is
    # still NaN in jnp.where's gradient rule. Clamp t to stay negative for this branch only;
    # the result is discarded for t>=0 regardless, so any finite value here is fine.
    t_insp_safe = jnp.minimum(t, -_EPS)
    ph_insp = _inspiral_phase_ansatz22(t_insp_safe, thetabar, eta, pn, insp_c, 0.0)
    ph_rd = _rd_phase_ansatz22(t, rd_c, omega_ring, ph_off_rd)
    ph_merger = _merger_phase_ansatz22(
        t, alpha1RD, omega_peak, omega_ring, domega_peak, merger_c, ph_off_merger
    )
    return jnp.where(t < tcut22 - dtM, ph_insp, jnp.where(t > 0.0, ph_rd, ph_merger))


def _t_ref_from_f_ref(eta, pn, insp_c, omega_ref):
    """Geometric time at which the (2,2) GW angular frequency equals ``omega_ref``.

    LALSuite's own reference-phase convention (see IMRPhenomTSetPhase22Coefficients's GSL
    root-find for ``tRef``) is defined relative to this time, not to merger: the orbital phase
    equals ``phiRef`` *at f_ref*, not at t=0. ``f_ref`` is always in the inspiral region for
    any physically sensible choice (f_ref <= the analysis f_min, deep inspiral), so this only
    needs to invert the smooth inspiral frequency ansatz.

    Solved by Newton's method, but gradients are **not** naively backpropagated through the
    unrolled iteration (measured: doing so produces NaN gradients -- differentiating through
    ~30 chained Newton steps is numerically fragile even though the converged value itself is
    fine). Instead: converge ``theta*`` under ``stop_gradient`` (a fixed iteration count, no
    data-dependent stopping condition, so this stays jit/vmap-safe), then take one further
    Newton-corrector step *with* gradients tracked -- by the implicit function theorem this
    one extra step from an already-converged root gives the exact sensitivity of theta* to
    (eta, pn, insp_c, omega_ref), without differentiating through the solver's own iteration.
    """

    def _residual(theta):
        return _inspiral_omega_ansatz22(theta, pn, insp_c) - omega_ref

    theta0 = jnp.clip(
        jnp.power(jnp.maximum(4.0 * omega_ref, _EPS), 1.0 / 3.0), 0.02, 0.9
    )

    def _newton_step(theta, _):
        f, df = jax.value_and_grad(_residual)(theta)
        theta_new = theta - f / df
        return jnp.clip(theta_new, 0.005, 0.98), None

    theta_star, _ = jax.lax.scan(
        _newton_step, jax.lax.stop_gradient(theta0), xs=None, length=30
    )
    theta_star = jax.lax.stop_gradient(theta_star)
    f_star, df_star = jax.value_and_grad(_residual)(theta_star)
    theta_final = theta_star - f_star / df_star
    return _t_of_theta(theta_final, eta)


def _inspiral_amp_ansatz_hm(x, amp_pn, insp_amp_c):
    """IMRPhenomTInspiralAmpAnsatzHM: PN amplitude (complex) + 3 NR-calibrated corrections."""
    (
        fac0,
        ampN,
        amp0h_re,
        amp0h_im,
        amp1_re,
        amp1_im,
        amp1h_re,
        amp1h_im,
        amp2_re,
        amp2_im,
        amp2h_re,
        amp2h_im,
        amp3_re,
        amp3_im,
        amp3h_re,
        amp3h_im,
        amplog,
    ) = amp_pn
    c1, c2, c3 = insp_amp_c
    xhalf = jnp.sqrt(x)
    x1half = x * xhalf
    x2 = x * x
    x2half = x2 * xhalf
    x3 = x2 * x
    x3half = x3 * xhalf
    x4 = x2 * x2
    x4half = x4 * xhalf
    x5 = x3 * x2

    ampreal = (
        ampN
        + amp0h_re * xhalf
        + amp1_re * x
        + amp1h_re * x1half
        + amp2_re * x2
        + amp2h_re * x2half
        + amp3_re * x3
        + amp3h_re * x3half
        + amplog * jnp.log(16.0 * x) * x3
    )
    ampimag = (
        amp0h_im * xhalf
        + amp1_im * x
        + amp1h_im * x1half
        + amp2_im * x2
        + amp2h_im * x2half
        + amp3_im * x3
        + amp3h_im * x3half
    )
    amp_complex = (ampreal + c1 * x4 + c2 * x4half + c3 * x5) + 1j * ampimag
    return fac0 * x * amp_complex


def _merger_amp_ansatz_hm(t, tpeak, alpha1RD, merger_c):
    """IMRPhenomTMergerAmpAnsatzHM."""
    c1, c2, c3, c4 = merger_c
    tau = t - tpeak
    # cosh(x) ~ exp(|x|)/2 for large |x| -- clip its argument for the same reason _safe_exp
    # clips: this branch is evaluated (and its gradient computed) even at deep-inspiral t,
    # far outside the merger region it's actually meant for.
    sech1 = 1.0 / jnp.cosh(jnp.clip(alpha1RD * tau, -_EXP_ARG_CLIP, _EXP_ARG_CLIP))
    sech2 = 1.0 / jnp.cosh(
        jnp.clip(2.0 * alpha1RD * tau, -_EXP_ARG_CLIP, _EXP_ARG_CLIP)
    )
    return c1 + c2 * sech1 + c3 * jnp.power(sech2, 1.0 / 7.0) + c4 * tau * tau


def _rd_amp_ansatz_hm(t, tpeak, alpha1RD, rd_c):
    """IMRPhenomTRDAmpAnsatzHM (Damour & Nagar 2014, eq. 4)."""
    c1, c2, c3, c4 = rd_c
    tau = t - tpeak
    return _safe_exp(-alpha1RD * tau) * (c1 * jnp.tanh(c2 * tau + c3) + c4)


def _amp_hm(t, x, tpeak, alpha1RD, amp_pn, insp_amp_c, merger_c, rd_c):
    insp = _inspiral_amp_ansatz_hm(x, amp_pn, insp_amp_c)
    merger = _merger_amp_ansatz_hm(t, tpeak, alpha1RD, merger_c) + 0j
    rd = _rd_amp_ansatz_hm(t, tpeak, alpha1RD, rd_c) + 0j
    return jnp.where(t < _T_CUT_AMP, insp, jnp.where(t > tpeak, rd, merger))


def _solve_22_phase_coefficients(
    eta, chi1z, chi2z, delta, S, dchi, fits, Mfinal, afinal, dtM
):
    """Mirrors IMRPhenomTSetPhase22Coefficients: everything the (2,2) frequency/phase ansatz needs."""
    fRING = evaluate_QNMfit_fring22(afinal) / Mfinal
    fDAMP = evaluate_QNMfit_fdamp22(afinal) / Mfinal
    omega_ring = 2.0 * PI * fRING
    alpha1RD = 2.0 * PI * fDAMP

    pn = _pn_omega_coefficients(eta, chi1z, chi2z, delta)

    # --- Ringdown coefficients: closed form (Damour & Nagar 2014 eq. 9) ---
    omega_peak = fits["IMRPhenomT_PeakFrequency_22"]
    rd_c3 = fits["IMRPhenomT_RD_Freq_D3_22"]
    rd_c2 = fits["IMRPhenomT_RD_Freq_D2_22"]
    rd_c4 = jnp.zeros_like(rd_c2)
    rd_c1 = (
        (1.0 + rd_c3 + rd_c4)
        * (omega_ring - omega_peak)
        / rd_c2
        / (rd_c3 + 2.0 * rd_c4)
    )
    rd_c = (rd_c1, rd_c2, rd_c3, rd_c4)

    # --- Inspiral coefficients: 6x6 solve (5 collocation-point fits + 1 TaylorT3-t0 point) ---
    tt0 = fits["IMRPhenomT_Inspiral_TaylorT3_t0"]
    t_early = _t_of_theta(_THETA_POINTS[0], eta)
    thetaini = jnp.power(eta * (tt0 - t_early) / 5.0, -1.0 / 8.0)
    omega_cp = jnp.stack(
        [
            _taylor_t3_omega(thetaini, pn),
            fits["IMRPhenomT_Inspiral_Freq_CP1_22"],
            fits["IMRPhenomT_Inspiral_Freq_CP2_22"],
            fits["IMRPhenomT_Inspiral_Freq_CP3_22"],
            fits["IMRPhenomT_Inspiral_Freq_CP4_22"],
            fits["IMRPhenomT_Inspiral_Freq_CP5_22"],
        ]
    )
    theta_pts = jnp.asarray(
        _THETA_POINTS, dtype=eta.dtype if hasattr(eta, "dtype") else jnp.float64
    )
    t3_offset = _taylor_t3_omega(theta_pts, pn)
    b_insp = (4.0 / theta_pts**3) * (omega_cp - t3_offset)
    A_insp = jnp.stack([theta_pts ** (8 + j) for j in range(6)], axis=1)
    insp_c = jnp.linalg.solve(A_insp, b_insp)

    t_cut = _t_of_theta(_THETA_CUT, eta)
    tcut22 = t_cut
    omega_cut = _inspiral_omega_ansatz22(jnp.asarray(_THETA_CUT), pn, insp_c)

    # --- Merger coefficients: 3x3 solve (continuity + 1 collocation point + differentiability) ---
    t_merger_cp = _t_of_theta(_THETA_MERGER_CP, eta)
    omega_merger_cp = 1.0 - fits["IMRPhenomT_Merger_Freq_CP1_22"] / omega_ring
    omega_cut_bar = 1.0 - omega_cut / omega_ring

    theta2 = _theta_of_t(t_cut, eta)
    theta1 = _theta_of_t(t_cut - _FINITE_DIFF_H, eta)
    domega_cut = (
        -(
            _inspiral_omega_ansatz22(theta2, pn, insp_c)
            - _inspiral_omega_ansatz22(theta1, pn, insp_c)
        )
        / _FINITE_DIFF_H
        / omega_ring
    )
    domega_peak = (
        -(
            _rd_omega_ansatz22(_FINITE_DIFF_H, rd_c, omega_ring)
            - _rd_omega_ansatz22(0.0, rd_c, omega_ring)
        )
        / _FINITE_DIFF_H
        / omega_ring
    )

    ascut = jnp.arcsinh(alpha1RD * t_cut)
    as_cp = jnp.arcsinh(alpha1RD * t_merger_cp)
    dencut = jnp.sqrt(1.0 + t_cut**2 * alpha1RD**2)
    A_merger = jnp.stack(
        [
            jnp.stack([ascut**2, ascut**3, ascut**4]),
            jnp.stack([as_cp**2, as_cp**3, as_cp**4]),
            jnp.stack(
                [
                    2.0 * alpha1RD * ascut,
                    3.0 * alpha1RD * ascut**2,
                    4.0 * alpha1RD * ascut**3,
                ]
            )
            / dencut,
        ]
    )
    b_merger = jnp.stack(
        [
            omega_cut_bar
            - (1.0 - omega_peak / omega_ring)
            - (domega_peak / alpha1RD) * ascut,
            omega_merger_cp
            - (1.0 - omega_peak / omega_ring)
            - (domega_peak / alpha1RD) * as_cp,
            domega_cut - domega_peak / dencut,
        ]
    )
    merger_c = jnp.linalg.solve(A_merger, b_merger)

    # --- Phase continuity offsets (default 3-region reconstruction: ph_off_insp = 0) ---
    thetabar_cut = _thetabar_of_t(t_cut, eta)
    ph_meco_insp = _inspiral_phase_ansatz22(t_cut, thetabar_cut, eta, pn, insp_c, 0.0)
    ph_meco_merger = _merger_phase_ansatz22(
        t_cut, alpha1RD, omega_peak, omega_ring, domega_peak, merger_c, 0.0
    )
    ph_off_merger = ph_meco_insp - ph_meco_merger
    ph_off_rd = _merger_phase_ansatz22(
        0.0, alpha1RD, omega_peak, omega_ring, domega_peak, merger_c, ph_off_merger
    )

    return dict(
        omega_ring=omega_ring,
        alpha1RD=alpha1RD,
        pn=pn,
        insp_c=insp_c,
        tcut22=tcut22,
        omega_peak=omega_peak,
        domega_peak=domega_peak,
        merger_c=merger_c,
        rd_c=rd_c,
        ph_off_merger=ph_off_merger,
        ph_off_rd=ph_off_rd,
        dtM=dtM,
    )


def _solve_22_amplitude_coefficients(
    eta, chi1z, chi2z, delta, fits, Mfinal, afinal, phase_c
):
    """Mirrors the l==2,m==2 branch of IMRPhenomTSetHMAmplitudeCoefficients."""
    fac0 = 2.0 * eta * jnp.sqrt(16.0 * PI / 5.0)
    amp1_re = -2.5476190476190474 + (55.0 * eta) / 42.0
    amp1h_re = (
        (-2.0 * chi1z) / 3.0
        - (2.0 * chi2z) / 3.0
        - (2.0 * chi1z * delta) / 3.0
        + (2.0 * chi2z * delta) / 3.0
        + (2.0 * chi1z * eta) / 3.0
        + (2.0 * chi2z * eta) / 3.0
        + 2.0 * PI
    )
    amp2_re = (
        -1.437169312169312
        + chi1z**2 / 2.0
        + chi2z**2 / 2.0
        + (chi1z**2 * delta) / 2.0
        - (chi2z**2 * delta) / 2.0
        - (1069.0 * eta) / 216.0
        - chi1z**2 * eta
        + 2.0 * chi1z * chi2z * eta
        - chi2z**2 * eta
        + (2047.0 * eta**2) / 1512.0
    )
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
    amp3h_re = (
        (-2173.0 * PI) / 756.0
        - (2495.0 * eta * PI) / 378.0
        + (40.0 * eta**2 * PI) / 27.0
    )
    amp3h_im = (14333.0 * eta) / 162.0 - (4066.0 * eta**2) / 945.0
    amplog = -428.0 / 105.0
    zero = jnp.zeros_like(eta)
    amp_pn = (
        fac0,
        jnp.ones_like(eta),
        zero,
        zero,
        amp1_re,
        zero,
        amp1h_re,
        zero,
        amp2_re,
        zero,
        amp2h_re,
        amp2h_im,
        amp3_re,
        amp3_im,
        amp3h_re,
        amp3h_im,
        amplog,
    )

    fDAMP = evaluate_QNMfit_fdamp22(afinal) / Mfinal
    fDAMPn2 = evaluate_QNMfit_fdamp22n2(afinal) / Mfinal
    alpha1RD = 2.0 * PI * fDAMP
    alpha2RD = 2.0 * PI * fDAMPn2
    tpeak = jnp.zeros_like(eta)  # (2,2) peak-amplitude time, by construction

    amp_peak = fits["IMRPhenomT_PeakAmp_22"]
    amp_rd_c3 = fits["IMRPhenomT_RD_Amp_C3_22"]
    rd_c2 = 0.5 * (alpha2RD - alpha1RD)
    coshc3 = jnp.cosh(amp_rd_c3)
    tanhc3 = jnp.tanh(amp_rd_c3)
    rd_c2 = jnp.where(
        jnp.abs(rd_c2) > jnp.abs(0.5 * alpha1RD / tanhc3),
        -0.5 * alpha1RD / tanhc3,
        rd_c2,
    )
    rd_c1 = amp_peak * alpha1RD * coshc3**2 / rd_c2
    rd_c4 = amp_peak - rd_c1 * tanhc3
    rd_c = (rd_c1, rd_c2, amp_rd_c3, rd_c4)

    # --- Inspiral amplitude coefficients: 3x3 solve against 3 NR-fitted collocation points ---
    t_insp_pts = jnp.asarray(
        _T_INSP_AMP_CP, dtype=eta.dtype if hasattr(eta, "dtype") else jnp.float64
    )
    omega_insp = _inspiral_omega_ansatz22(
        _theta_of_t(t_insp_pts, eta), phase_c["pn"], phase_c["insp_c"]
    )
    x_insp = jnp.power(0.5 * omega_insp, 2.0 / 3.0)
    zero_c = (zero, zero, zero)
    amp_offset = jnp.real(_inspiral_amp_ansatz_hm(x_insp, amp_pn, zero_c))
    amp_insp_cp = jnp.stack(
        [
            fits["IMRPhenomT_Inspiral_Amp_CP1_22"],
            fits["IMRPhenomT_Inspiral_Amp_CP2_22"],
            fits["IMRPhenomT_Inspiral_Amp_CP3_22"],
        ]
    )
    b_insp = (1.0 / fac0 / x_insp) * (amp_insp_cp - amp_offset)
    A_insp = jnp.stack([x_insp**4, x_insp**4 * jnp.sqrt(x_insp), x_insp**5], axis=1)
    insp_amp_c = jnp.linalg.solve(A_insp, b_insp)

    # --- Merger amplitude coefficients: 4x4 solve ---
    omega_at_tcut = _inspiral_omega_ansatz22(
        _theta_of_t(phase_c["tcut22"], eta), phase_c["pn"], phase_c["insp_c"]
    )
    x_at_tcut = jnp.power(0.5 * omega_at_tcut, 2.0 / 3.0)
    amp_insp_at_cut = _inspiral_amp_ansatz_hm(x_at_tcut, amp_pn, insp_amp_c)
    ampinsp = jnp.sign(jnp.real(amp_insp_at_cut)) * jnp.abs(amp_insp_at_cut)

    tau_cut = phase_c["tcut22"] - tpeak
    sech1_cut = 1.0 / jnp.cosh(alpha1RD * tau_cut)
    sech2_cut = 1.0 / jnp.cosh(2.0 * alpha1RD * tau_cut)
    tau_cp = _T_MERGER_CP_AMP - tpeak
    sech1_cp = 1.0 / jnp.cosh(alpha1RD * tau_cp)
    sech2_cp = 1.0 / jnp.cosh(2.0 * alpha1RD * tau_cp)

    theta_at_tcut_m1 = _theta_of_t(phase_c["tcut22"] - _FINITE_DIFF_H, eta)
    omega_at_tcut_m1 = _inspiral_omega_ansatz22(
        theta_at_tcut_m1, phase_c["pn"], phase_c["insp_c"]
    )
    x_at_tcut_m1 = jnp.power(0.5 * omega_at_tcut_m1, 2.0 / 3.0)
    amp_insp_at_cut_m1 = _inspiral_amp_ansatz_hm(x_at_tcut_m1, amp_pn, insp_amp_c)
    amp_insp_signed_m1 = jnp.sign(jnp.real(amp_insp_at_cut_m1)) * jnp.abs(
        amp_insp_at_cut_m1
    )
    damp_meco = (ampinsp - amp_insp_signed_m1) / _FINITE_DIFF_H

    tanh_cut = jnp.tanh(alpha1RD * tau_cut)
    sinh2_cut = jnp.sinh(2.0 * alpha1RD * tau_cut)
    aux1 = -alpha1RD * sech1_cut * tanh_cut
    aux2 = (-2.0 / 7.0) * alpha1RD * sinh2_cut * jnp.power(sech2_cut, 8.0 / 7.0)
    aux3 = 2.0 * tau_cut

    A_merger_amp = jnp.stack(
        [
            jnp.stack(
                [
                    jnp.ones_like(eta),
                    sech1_cut,
                    jnp.power(sech2_cut, 1.0 / 7.0),
                    tau_cut**2,
                ]
            ),
            jnp.stack(
                [
                    jnp.ones_like(eta),
                    sech1_cp,
                    jnp.power(sech2_cp, 1.0 / 7.0),
                    tau_cp**2,
                ]
            ),
            jnp.stack(
                [jnp.ones_like(eta), jnp.ones_like(eta), jnp.ones_like(eta), zero]
            ),
            jnp.stack([zero, aux1, aux2, aux3]),
        ]
    )
    amp_merger_cp1 = fits["IMRPhenomT_Merger_Amp_CP1_22"]
    b_merger_amp = jnp.stack([ampinsp, amp_merger_cp1, amp_peak, damp_meco])
    merger_c = jnp.linalg.solve(A_merger_amp, b_merger_amp)

    return dict(
        tpeak=tpeak,
        alpha1RD=alpha1RD,
        amp_pn=amp_pn,
        insp_amp_c=insp_amp_c,
        merger_c=merger_c,
        rd_c=rd_c,
    )


class IMRPhenomT(TimeDomainModel):
    is_fd = False

    def __init__(self, f_ref: float = 20.0):
        # t=0 is always the (2,2) peak-amplitude time (matched to geocent_time by the caller),
        # matching LALSuite's own convention -- but the *phase* reference is anchored at f_ref,
        # not at t=0: the orbital phase equals params["phase"] at the moment the GW frequency
        # crosses f_ref, matching LALSuite's tRef-based reference-phase convention (found from
        # a genuine cross-check against LAL's IMRPhenomT: without this, the phase is shifted by
        # a mass/spin-dependent constant -- the accumulated phase between f_ref and merger --
        # relative to LALSuite, even though the frequency/amplitude evolution already agreed to
        # ~5-6 significant figures. See _t_ref_from_f_ref.
        self.f_ref = f_ref

    @functools.partial(jax.jit, static_argnums=(0,))
    def mode_dict(
        self, params: dict, grid: jax.Array
    ) -> Dict[Tuple[int, int], jax.Array]:
        """Returns {(2,2), (2,-2)}: strain at 1 Mpc, tapered by nothing (caller's own window)."""
        mc = jnp.maximum(params["chirp_mass"], 1.0)
        q = jnp.clip(params["mass_ratio"], 0.01, 1.0)
        eta = q / (1.0 + q) ** 2
        eta = jnp.clip(eta, 1e-4, 0.25)
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

        phase_c = _solve_22_phase_coefficients(
            eta, s1z, s2z, delta, S, dchi, fits, Mfinal, afinal, dtM
        )
        amp_c = _solve_22_amplitude_coefficients(
            eta, s1z, s2z, delta, fits, Mfinal, afinal, phase_c
        )

        omega = _omega22(
            t_geom,
            eta,
            phase_c["tcut22"],
            phase_c["dtM"],
            phase_c["pn"],
            phase_c["insp_c"],
            phase_c["alpha1RD"],
            phase_c["omega_peak"],
            phase_c["omega_ring"],
            phase_c["domega_peak"],
            phase_c["merger_c"],
            phase_c["rd_c"],
        )
        phase = _phase22(
            t_geom,
            eta,
            phase_c["tcut22"],
            phase_c["dtM"],
            phase_c["pn"],
            phase_c["insp_c"],
            phase_c["alpha1RD"],
            phase_c["omega_peak"],
            phase_c["omega_ring"],
            phase_c["domega_peak"],
            phase_c["merger_c"],
            phase_c["rd_c"],
            phase_c["ph_off_merger"],
            phase_c["ph_off_rd"],
        )
        x = jnp.power(jnp.maximum(0.5 * omega, _EPS), 2.0 / 3.0)
        amp = jnp.abs(
            _amp_hm(
                t_geom,
                x,
                amp_c["tpeak"],
                amp_c["alpha1RD"],
                amp_c["amp_pn"],
                amp_c["insp_amp_c"],
                amp_c["merger_c"],
                amp_c["rd_c"],
            )
        )

        # Reference the phase at f_ref (not at t=0/merger) -- see __init__.
        omega_f_ref = 2.0 * PI * self.f_ref * (M * MTSUN)
        t_ref = _t_ref_from_f_ref(eta, phase_c["pn"], phase_c["insp_c"], omega_f_ref)
        phase_ref = _phase22(
            t_ref,
            eta,
            phase_c["tcut22"],
            phase_c["dtM"],
            phase_c["pn"],
            phase_c["insp_c"],
            phase_c["alpha1RD"],
            phase_c["omega_peak"],
            phase_c["omega_ring"],
            phase_c["domega_peak"],
            phase_c["merger_c"],
            phase_c["rd_c"],
            phase_c["ph_off_merger"],
            phase_c["ph_off_rd"],
        )

        amp_factor = (M * MTSUN * C_SI) / MPC
        h22 = amp_factor * amp * jnp.exp(-1j * (phase - phase_ref))

        return {(2, 2): h22, (2, -2): jnp.conj(h22)}

    @functools.partial(jax.jit, static_argnums=(0,))
    def __call__(self, params: dict, grid: jax.Array) -> Tuple[jax.Array, jax.Array]:
        modes = self.mode_dict(params, grid)

        dist = params.get("luminosity_distance", 100.0)
        phi_ref = params.get("phase", 0.0)
        iota = params.get("inclination", 0.0)

        dist_scale = 1.0 / dist

        # phi_ref enters through the Ylm's own azimuthal argument (LALSuite's XLALSimAddMode
        # convention: phi = pi/2 - phiRef), not as a separate exp(i*m*phi_ref) mode rotation --
        # the two are NOT interchangeable away from face-on/edge-on inclination (confirmed by
        # cross-checking against LALSuite's IMRPhenomT: only this convention reproduces LAL's
        # phi_ref-dependence at general inclination, not just at iota in {0, pi/2}).
        phi_ylm = PI / 2.0 - phi_ref

        h22 = modes[(2, 2)] * dist_scale
        h2m2 = modes[(2, -2)] * dist_scale

        Y22 = spin_weighted_ylm(iota, phi_ylm, 2, 2, s=-2)
        Y2m2 = spin_weighted_ylm(iota, phi_ylm, 2, -2, s=-2)

        h_comp = h22 * Y22 + h2m2 * Y2m2
        hp = jnp.real(h_comp)
        hc = -jnp.imag(h_comp)

        return hp, hc
