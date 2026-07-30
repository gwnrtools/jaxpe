import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
import jax.numpy as jnp
from jaxpe.gw.cbc_models.phenomthm import IMRPhenomTHM

import pytest

@pytest.fixture
def default_params():
    return {
        "chirp_mass": 30.0,
        "mass_ratio": 2.0,
        "spin1z": 0.5,
        "spin2z": -0.5,
        "luminosity_distance": 100.0,
        "phase": 0.0,
        "inclination": 0.5,
        "geocent_time": 0.0
    }

def test_imr_phenom_thm_initialization():
    model = IMRPhenomTHM(f_ref=20.0)
    assert model.f_ref == 20.0
    assert not model.is_fd
    print("Initialization test passed.")

def test_imr_phenom_thm_evaluation(default_params):
    model = IMRPhenomTHM()
    t_grid = jnp.linspace(-0.1, 0.02, 100)
    hp, hc = model(default_params, t_grid)
    
    assert hp.shape == t_grid.shape
    assert hc.shape == t_grid.shape
    assert not jnp.any(jnp.isnan(hp))
    assert not jnp.any(jnp.isnan(hc))
    print("Evaluation test passed.")

def test_imr_phenom_thm_jit(default_params):
    model = IMRPhenomTHM()
    t_grid = jnp.linspace(-0.1, 0.02, 100)
    
    hp, hc = model(default_params, t_grid)
    hp2, hc2 = model(default_params, t_grid)
    
    import numpy as np
    np.testing.assert_allclose(hp, hp2, rtol=1e-5)
    np.testing.assert_allclose(hc, hc2, rtol=1e-5)
    print("JIT test passed.")

def test_imr_phenom_thm_vmap(default_params):
    model = IMRPhenomTHM()
    t_grid = jnp.linspace(-0.1, 0.02, 100)
    
    batched_params = {
        k: jnp.array([v, v * 1.1]) for k, v in default_params.items()
    }
    
    def batched_call(p, g):
        return model(p, g)
        
    vmap_model = jax.vmap(batched_call, in_axes=(0, None))
    hp_batch, hc_batch = vmap_model(batched_params, t_grid)
    
    assert hp_batch.shape == (2, 100)
    assert hc_batch.shape == (2, 100)
    print("VMAP test passed.")

if __name__ == "__main__":
    params = default_params()
    test_imr_phenom_thm_initialization()
    test_imr_phenom_thm_evaluation(params)
    test_imr_phenom_thm_jit(params)
    test_imr_phenom_thm_vmap(params)
    print("All tests passed.")
