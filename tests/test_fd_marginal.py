"""Correctness tests for the closed-form phase+distance marginal (fd_marginal.py).

The key claim under test is the phase reduction: for a dominant-(2,2)-mode
frequency-domain model the coalescence-phase integral of the full likelihood equals
the closed form ln I0(u|Z|). We check this by reconstructing the marginal a second,
independent way -- a brute-force 2-D quadrature of the *full* network likelihood over
(phase, distance) -- and requiring the two to agree.

To isolate the phase closed form (the novel, risky part) from trivial distance-
quadrature discretization, the brute force reuses the marginal object's own distance
grid, so the distance trapezoid cancels exactly and only the phase reduction and the
overlap bookkeeping are exercised. A separate test checks the dominant-mode self-check
fires (residual ~ 0 for PhenomD) and a third checks the marginal peaks at the injected
truth in a zero-noise injection.
"""

import numpy as np
import pytest
from scipy.special import logsumexp

import jax
import jax.numpy as jnp

from jaxpe.gw import IMRPhenomD, PhaseDistanceMarginalLikelihood, make_injection

F_LOWER = 20.0
DURATION = 4.0
SAMPLING_RATE = 1024.0  # test is a self-consistency check, so the band choice is free
T_C = 1126259462.4

# Aligned-spin injection at a moderate network SNR (~10): loud enough that the phase
# integrand has real structure, quiet enough that a few-hundred-node phase trapezoid
# resolves it spectrally.
INJECTION = dict(
    chirp_mass=25.0,
    mass_ratio=0.8,
    spin1z=0.2,
    spin2z=-0.1,
    luminosity_distance=5000.0,
    inclination=0.4,
    phase=1.5,
    geocent_time=T_C,
    ra=1.2,
    dec=0.5,
    psi=0.8,
)
INTRINSIC_NAMES = ("chirp_mass", "mass_ratio", "spin1z", "spin2z")
FIXED_EXT = {k: INJECTION[k] for k in ("ra", "dec", "psi", "inclination")}
DIST_BOUNDS = (1000.0, 8000.0)


@pytest.fixture(scope="module")
def phenomd_likelihood():
    """Zero-noise IMRPhenomD injection into a two-detector network."""
    waveform = IMRPhenomD(f_ref=F_LOWER)
    like = make_injection(
        waveform,
        INJECTION,
        detector_names=("H1", "L1"),
        duration=DURATION,
        sampling_rate=SAMPLING_RATE,
        f_min=F_LOWER,
        noise_seed=None,
    )
    return like


@pytest.fixture(scope="module")
def marginal(phenomd_likelihood):
    return PhaseDistanceMarginalLikelihood(
        phenomd_likelihood,
        INTRINSIC_NAMES,
        FIXED_EXT,
        dist_bounds=DIST_BOUNDS,
        dist_power=2.0,
        d_ref=1000.0,
        n_dist=400,
        check_params=INJECTION,
    )


def test_dominant_mode_residual_is_zero(marginal):
    """IMRPhenomD is dominant-(2,2): the factorization self-check must be ~exact."""
    assert marginal.dominant_mode_residual is not None
    assert marginal.dominant_mode_residual < 1e-6, marginal.dominant_mode_residual


def test_closed_form_marginal_vs_brute_force(phenomd_likelihood, marginal):
    """Closed-form marginal == independent 2-D (phase, distance) quadrature of the
    full likelihood, to phase-quadrature precision.

    The brute force integrates exp(full log-likelihood) over a dense uniform phase
    grid (periodic trapezoid; the phase prior is uniform on [0, 2*pi)) at each distance
    node, then over the distance prior. Reusing the marginal's own distance grid and
    prior makes the distance quadrature identical on both sides, so any disagreement is
    the phase reduction ln I0(u|Z|) -- exactly what we want to test.
    """
    like = phenomd_likelihood
    truth_vec = [INJECTION[n] for n in INTRINSIC_NAMES]
    closed_form = marginal(truth_vec)

    base = {
        **{n: jnp.asarray(INJECTION[n]) for n in INTRINSIC_NAMES},
        **{k: jnp.asarray(v) for k, v in FIXED_EXT.items()},
        "geocent_time": jnp.asarray(T_C),
    }
    n_phi = 384
    phases = jnp.asarray(np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False))

    @jax.jit
    def lnl_over_phase(distance):
        def one(phase):
            return like.log_likelihood(
                {**base, "phase": phase, "luminosity_distance": distance}
            )

        return jax.vmap(one)(phases)

    # phase marginal at each distance node: log of (1/2pi) * integral exp(lnL) dphi
    dist_grid = marginal._D
    phase_marginal = np.array(
        [
            logsumexp(np.asarray(lnl_over_phase(jnp.asarray(D)))) - np.log(n_phi)
            for D in dist_grid
        ]
    )
    # then the distance integral against the SAME prior/trapezoid the marginal uses
    brute_force = logsumexp(marginal._log_pi + phase_marginal + marginal._log_dD)

    assert abs(closed_form - brute_force) < 1e-3, (closed_form, brute_force)


def test_marginal_peaks_at_injected_chirp_mass(marginal):
    """Zero-noise injection: the phase+distance marginal must peak at the true
    chirp mass when the other intrinsics are held at truth."""
    chirp_grid = np.linspace(24.0, 26.0, 21)
    values = np.array(
        [
            marginal(
                [mc, INJECTION["mass_ratio"], INJECTION["spin1z"], INJECTION["spin2z"]]
            )
            for mc in chirp_grid
        ]
    )
    assert np.all(np.isfinite(values))
    assert abs(chirp_grid[int(np.argmax(values))] - INJECTION["chirp_mass"]) < 0.2


def test_missing_check_params_warns(phenomd_likelihood):
    """Without check_params the factorization cannot be verified -- warn loudly."""
    with pytest.warns(UserWarning, match="NOT verified"):
        PhaseDistanceMarginalLikelihood(
            phenomd_likelihood,
            INTRINSIC_NAMES,
            FIXED_EXT,
            dist_bounds=DIST_BOUNDS,
            check_params=None,
        )


def test_higher_mode_model_trips_self_check():
    """A model carrying sub-dominant modes at non-zero inclination must fail the
    dominant-mode self-check and warn that the I0 form is only approximate."""
    esigmapy = pytest.importorskip("esigmapy")  # noqa: F841
    from jaxpe.gw import ESIGMAInspiral

    waveform = ESIGMAInspiral(
        f_lower=F_LOWER,
        modes=((2, 2), (3, 3)),
        rad_pn_order=0,
        mode_pn_order=0,
        ode_eps=1e-6,
        n_ode_grid=256,
        max_ode_steps=16384,
    )
    injection = dict(
        chirp_mass=25.0,
        mass_ratio=0.7,  # unequal masses so the (3,3) mode is non-negligible
        eccentricity=0.0,
        mean_anomaly=0.0,
        spin1z=0.0,
        spin2z=0.0,
        luminosity_distance=1000.0,
        inclination=1.1,  # away from face-on so higher modes contribute
        phase=1.5,
        geocent_time=T_C,
        ra=1.2,
        dec=0.5,
        psi=0.8,
    )
    like = make_injection(
        waveform,
        injection,
        detector_names=("H1", "L1"),
        duration=DURATION,
        sampling_rate=SAMPLING_RATE,
        f_min=F_LOWER,
        noise_seed=None,
    )
    with pytest.warns(UserWarning, match="only APPROXIMATE"):
        marginal = PhaseDistanceMarginalLikelihood(
            like,
            ("chirp_mass", "mass_ratio"),
            {
                k: injection[k]
                for k in (
                    "ra",
                    "dec",
                    "psi",
                    "inclination",
                    "eccentricity",
                    "mean_anomaly",
                    "spin1z",
                    "spin2z",
                )
            },
            dist_bounds=DIST_BOUNDS,
            check_params=injection,
        )
    assert marginal.dominant_mode_residual > 1e-2
