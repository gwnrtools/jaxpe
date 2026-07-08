import time
import jax
import jax.numpy as jnp
from jaxpe.gw.esigma import ESIGMAInspiral


def run_benchmark():
    # Setup inputs for GW150914-like source
    # duration = 2s, f_lower = 20Hz, sample_rate = 4096Hz
    # So times grid is 2 * 4096 = 8192 points
    times = jnp.linspace(0.0, 2.0, 8192)

    params = {
        "chirp_mass": jnp.array(30.0),
        "mass_ratio": jnp.array(0.9),
        "eccentricity": jnp.array(0.01),
        "mean_anomaly": jnp.array(0.0),
        "spin1z": jnp.array(0.0),
        "spin2z": jnp.array(0.0),
        "inclination": jnp.array(0.5),
        "phase": jnp.array(0.0),
        "geocent_time": jnp.array(1.8),  # coalescence near end of 2s window
        "luminosity_distance": jnp.array(400.0),
    }

    # ODE grid needs to be larger for production
    # Tighter ode_eps (e.g. 1e-9)
    wf_prod = ESIGMAInspiral(
        f_lower=20.0,
        rad_pn_order=8,
        mode_pn_order=8,
        n_ode_grid=1024,
        max_ode_steps=1024,
        ode_eps=1e-9,
        adjoint_mode="forward_sensitivity",
    )

    def loss(p):
        hp, hc = wf_prod(p, times)
        return jnp.sum(hp) + jnp.sum(hc)

    val_and_grad = jax.jit(jax.value_and_grad(loss))

    print(
        "Compiling for production settings (duration=2s, ode_eps=1e-9, n_ode_grid=1024)..."
    )
    t0 = time.time()
    val_and_grad.lower(params).compile()
    print(f"Compiled in {time.time() - t0:.2f}s")

    # Warmup
    val_and_grad(params)

    print("Benchmarking execution time (50 iterations)...")
    t0 = time.time()
    for _ in range(50):
        val, grad = val_and_grad(params)
        jax.block_until_ready(grad)
    t_exec = (time.time() - t0) / 50.0
    print(f"Time per gradient evaluation: {t_exec*1000:.2f} ms")


if __name__ == "__main__":
    run_benchmark()
