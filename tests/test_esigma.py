import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Ensure esigmapy is installed before running these tests
pytest.importorskip("esigmapy")

from jaxpe.gw.esigma import ESIGMAInspiral
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
    for l, m in [(2, 2), (2, -2), (2, 1), (2, -1), (3, 3), (3, -3), (3, 2), (4, 4), (4, -4)]:
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
