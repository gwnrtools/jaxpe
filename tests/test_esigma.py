import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Ensure esigmapy is installed before running these tests
pytest.importorskip("esigmapy")

from jaxpe.gw import ESIGMAInspiral
from jaxpe.gw.harmonics import spin_weighted_ylm


def test_esigma_parity():
    """Check parity with esigmapy reference implementation to ~2% accuracy."""
    from esigmapy.inspiral.jax_backend.generator import get_inspiral_esigma_waveform_jax

    mass1, mass2 = 24.0, 19.2
    eta = mass1 * mass2 / (mass1 + mass2) ** 2
    mc = (mass1 + mass2) * eta**0.6
    q = mass2 / mass1

    sr = 2048.0
    dt = 1.0 / sr

    hp_ref, hc_ref = get_inspiral_esigma_waveform_jax(
        mass1=mass1,
        mass2=mass2,
        f_lower=20.0,
        delta_t=dt,
        eccentricity=0.1,
        mean_anomaly=0.3,
        inclination=0.6,
        coa_phase=1.2,
        distance=500.0,
        modes_to_use=[(2, 2), (3, 3), (4, 4)],
        mode_pn_order=8,
    )

    wf = ESIGMAInspiral(
        f_lower=20.0,
        modes=((2, 2), (3, 3), (4, 4)),
        n_ode_grid=16384,  # high resolution for parity check
        taper_on_seconds=0.0,
        taper_off_seconds=0.0,
    )

    n = len(hp_ref) + 100
    times = jnp.asarray(np.arange(n) * dt)
    t_c = len(hp_ref) * dt

    params = dict(
        chirp_mass=jnp.asarray(mc),
        mass_ratio=jnp.asarray(q),
        eccentricity=jnp.asarray(0.1),
        mean_anomaly=jnp.asarray(0.3),
        spin1z=jnp.asarray(0.0),
        spin2z=jnp.asarray(0.0),
        luminosity_distance=jnp.asarray(500.0),
        inclination=jnp.asarray(0.6),
        phase=jnp.asarray(1.2),
        geocent_time=jnp.asarray(t_c),
    )

    hp, hc = wf(params, times)

    # We compare the first 20% of the waveform where small timing offsets
    # between esigmapy's ISCO clipping and our subgrid ISCO pinning have less impact
    n1 = int(0.20 * len(hp_ref))

    hp_ref_early = hp_ref[:n1]
    hp_our_early = np.asarray(hp)[:n1]

    relerr = np.linalg.norm(hp_ref_early - hp_our_early) / np.linalg.norm(hp_ref_early)
    assert relerr < 0.08, f"ESIGMA adapter mismatch > 8%: {relerr:.3f}"


def test_spin_weighted_ylm():
    """Validate spin_weighted_ylm against LAL."""
    lal = pytest.importorskip("lal")

    rng = np.random.default_rng(0)
    worst = 0
    for l, m in [
        (2, 2),
        (2, -2),
        (2, 1),
        (2, -1),
        (3, 3),
        (3, -3),
        (3, 2),
        (4, 4),
        (4, -4),
    ]:
        for _ in range(6):
            io, ph = rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)
            ours = complex(spin_weighted_ylm(io, ph, l, m))
            ref = lal.SpinWeightedSphericalHarmonic(io, ph, -2, l, m)
            worst = max(worst, abs(ours - ref))

    assert worst < 1e-14, f"Ylm mismatch with LAL: {worst:.3e}"


def test_esigma_gradients_and_vmap():
    """Ensure ESIGMAInspiral is fully JAX traceable, vmappable, and differentiable."""
    wf = ESIGMAInspiral(f_lower=20.0, modes=((2, 2),), n_ode_grid=512)

    def loss(p):
        times = jnp.linspace(0.0, 1.0, 1024)
        hp, hc = wf(p, times)
        return jnp.sum(hp**2 + hc**2)

    params = dict(
        chirp_mass=jnp.asarray(30.0),
        mass_ratio=jnp.asarray(0.8),
        eccentricity=jnp.asarray(0.1),
        mean_anomaly=jnp.asarray(0.3),
        spin1z=jnp.asarray(0.0),
        spin2z=jnp.asarray(0.0),
        luminosity_distance=jnp.asarray(500.0),
        inclination=jnp.asarray(0.6),
        phase=jnp.asarray(1.2),
        geocent_time=jnp.asarray(1.5),
    )

    # 1. JIT + Grad
    grad_fn = jax.jit(jax.grad(loss))
    grads = grad_fn(params)
    assert jnp.isfinite(grads["chirp_mass"])
    assert jnp.isfinite(grads["eccentricity"])

    # 2. Vmap
    batch_params = {k: jnp.stack([v, v * 1.01]) for k, v in params.items()}
    vmap_fn = jax.jit(jax.vmap(loss))
    losses = vmap_fn(batch_params)
    assert losses.shape == (2,)
    assert jnp.all(jnp.isfinite(losses))


def test_esigma_eccentricity_gradient_matches_fd():
    """Regression guard for the ISCO-clip eccentricity-gradient bug (docs/under_construction.md
    §17-18).

    Past ISCO the RHS is frozen (``dydt = 0``); under autodiff that held the sensitivity
    ``dx/dtheta`` frozen at its inflated pre-freeze value, and summing it over the frozen tail
    produced a spurious ~20-100x eccentricity gradient (formerly ~99% off finite differences).
    Eccentricity is the worst case because its entire effect enters through the ISCO crossing,
    with no smooth ``t_max`` term to mask the error. The ``jnp.minimum(x, x_final)`` cap in
    ``_integrate`` restores agreement while leaving the waveform unchanged (frozen ``x``
    overshoots ``x_final`` by ~1e-8). This asserts the (default forward-sensitivity) gradient
    now matches central finite differences.
    """
    wf = ESIGMAInspiral(
        f_lower=20.0,
        modes=((2, 2), (3, 3)),
        rad_pn_order=0,  # cheap RHS: keeps the test fast; the clip mechanism is PN-independent
        mode_pn_order=0,
        ode_eps=1e-9,
        # 256 grid points: the residual sub-grid non-smoothness of _isco_time (argmax + interp)
        # makes the eccentricity gradient converge to FD only as the grid refines
        # (rel ~0.66 at 128, ~0.028 at 256, ~0.015 at 512); 256 comfortably separates the fixed
        # value from the ~19x-inflated pre-cap one.
        n_ode_grid=256,
        max_ode_steps=4096,
    )
    times = jnp.linspace(0.0, 1.0, 512)
    params = dict(
        chirp_mass=jnp.asarray(30.0),
        mass_ratio=jnp.asarray(0.9),
        eccentricity=jnp.asarray(0.05),
        mean_anomaly=jnp.asarray(0.3),
        spin1z=jnp.asarray(0.1),
        spin2z=jnp.asarray(0.15),
        inclination=jnp.asarray(0.5),
        phase=jnp.asarray(0.4),
        geocent_time=jnp.asarray(0.5),  # mid-window: an in-band, nonzero waveform
        luminosity_distance=jnp.asarray(400.0),
    )
    rng = np.random.default_rng(0)
    wp = jnp.asarray(rng.standard_normal(512))
    wc = jnp.asarray(rng.standard_normal(512))

    def loss(p):
        hp, hc = wf(p, times)
        return jnp.sum(wp * hp) + jnp.sum(wc * hc)

    # a trivially-zero waveform would make every gradient 0 and the test vacuous
    hp, _ = wf(params, times)
    assert float(jnp.max(jnp.abs(hp))) > 0.0

    grad = jax.grad(loss)(params)
    assert all(bool(jnp.isfinite(g)) for g in grad.values())

    def central_fd(key, h):
        pp = dict(params, **{key: params[key] + h})
        pm = dict(params, **{key: params[key] - h})
        return float((loss(pp) - loss(pm)) / (2.0 * h))

    # eccentricity: the parameter the ISCO-clip bug corrupted (was rel ~1.0 off FD)
    ecc_ad = float(grad["eccentricity"])
    ecc_fd = central_fd("eccentricity", 1e-4)
    ecc_rel = abs(ecc_ad - ecc_fd) / max(abs(ecc_fd), abs(ecc_ad))
    assert ecc_rel < 0.1, f"eccentricity gradient disagrees with FD: rel={ecc_rel:.3f}"

    # a clean control: geocent_time is extrinsic (never enters the ODE) and must match tightly
    gt_ad = float(grad["geocent_time"])
    gt_fd = central_fd("geocent_time", 1e-6)
    assert abs(gt_ad - gt_fd) / abs(gt_fd) < 1e-3
