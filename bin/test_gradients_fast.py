import jax
import jax.numpy as jnp
from jaxpe.gw.esigma import ESIGMAInspiral

def test_gradient_correctness():
    times = jnp.linspace(0.0, 1.0, 1024)
    params = {
        "chirp_mass": jnp.array(30.0),
        "mass_ratio": jnp.array(0.9),
        "eccentricity": jnp.array(0.01),
        "mean_anomaly": jnp.array(0.0),
        "spin1z": jnp.array(0.1),
        "spin2z": jnp.array(0.1),
        "inclination": jnp.array(0.5),
        "phase": jnp.array(0.0),
        "geocent_time": jnp.array(0.8),
        "luminosity_distance": jnp.array(400.0)
    }

    wf_rev = ESIGMAInspiral(f_lower=20.0, rad_pn_order=8, mode_pn_order=8, adjoint_mode="recursive_checkpoint")
    wf_fwd = ESIGMAInspiral(f_lower=20.0, rad_pn_order=8, mode_pn_order=8, adjoint_mode="forward_sensitivity")

    def loss_rev(p):
        hp, hc = wf_rev(p, times)
        return jnp.sum(hp) + jnp.sum(hc)
        
    def loss_fwd(p):
        hp, hc = wf_fwd(p, times)
        return jnp.sum(hp) + jnp.sum(hc)

    print("Computing rev gradients...")
    g_rev = jax.grad(loss_rev)(params)
    print("Computing fwd gradients...")
    g_fwd = jax.grad(loss_fwd)(params)

    for k in params:
        diff = jnp.abs(g_rev[k] - g_fwd[k])
        print(f"{k}: max diff = {diff}")
        assert diff < 1e-6, f"Gradient mismatch for {k}"

    print("Gradient check PASSED!")

if __name__ == "__main__":
    test_gradient_correctness()
