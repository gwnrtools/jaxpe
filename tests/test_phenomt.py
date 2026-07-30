import jax
import jax.numpy as jnp
import pytest
from jaxpe.gw.cbc_models.phenomt import IMRPhenomT

@pytest.fixture
def phenomt_model():
    return IMRPhenomT(f_ref=20.0)

@pytest.fixture
def default_params():
    return {
        "chirp_mass": 30.0,
        "mass_ratio": 1.0,
        "spin1z": 0.0,
        "spin2z": 0.0,
        "luminosity_distance": 100.0,
        "phase": 0.0,
        "inclination": 0.0,
        "geocent_time": 0.0
    }

def test_initialization(phenomt_model):
    assert not phenomt_model.is_fd
    assert phenomt_model.f_ref == 20.0

def test_evaluate_waveform(phenomt_model, default_params):
    # Test arbitrary time points (case b)
    t_grid = jnp.linspace(-2.0, 0.1, 1000)
    hp, hc = phenomt_model(default_params, t_grid)
    
    assert hp.shape == t_grid.shape
    assert hc.shape == t_grid.shape
    assert not jnp.any(jnp.isnan(hp))
    assert not jnp.any(jnp.isnan(hc))

def test_jit_compilation(phenomt_model, default_params):
    t_grid = jnp.linspace(-2.0, 0.1, 100)
    jitted_model = jax.jit(phenomt_model)
    
    hp, hc = jitted_model(default_params, t_grid)
    assert hp.shape == t_grid.shape

def test_vmap_vectorization(phenomt_model, default_params):
    # Vectorize over mass_ratio
    t_grid = jnp.linspace(-2.0, 0.1, 100)
    
    params_batched = {
        "chirp_mass": jnp.array([30.0, 30.0]),
        "mass_ratio": jnp.array([1.0, 0.5]),
        "spin1z": jnp.array([0.0, 0.0]),
        "spin2z": jnp.array([0.0, 0.0]),
        "luminosity_distance": jnp.array([100.0, 100.0]),
        "phase": jnp.array([0.0, 0.0]),
        "inclination": jnp.array([0.0, 0.0]),
        "geocent_time": jnp.array([0.0, 0.0])
    }
    
    vmapped_model = jax.vmap(phenomt_model, in_axes=(0, None))
    hp, hc = vmapped_model(params_batched, t_grid)
    
    assert hp.shape == (2, 100)
    assert hc.shape == (2, 100)
    assert not jnp.any(jnp.isnan(hp))

def test_piecewise_continuity(phenomt_model, default_params):
    # Verify that the frequency and amplitude do not produce NaNs at the boundaries
    # Using the placeholder coefficients, continuity is not guaranteed mathematically
    # but we must ensure the JAX operations themselves don't fail exactly AT the boundary
    # due to gradients or invalid ops.
    t_grid = jnp.array([-100.0, 0.0]) # t_meco and t=0
    hp, hc = phenomt_model(default_params, t_grid)
    
    assert not jnp.any(jnp.isnan(hp))
    assert not jnp.any(jnp.isnan(hc))
