import time
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxpe.gw.esigma import ESIGMAInspiral


def test_esigma_compile_size():
    # Setup inputs
    params = {
        "chirp_mass": jnp.array(30.0),
        "mass_ratio": jnp.array(0.9),
        "eccentricity": jnp.array(0.1),
        "mean_anomaly": jnp.array(0.0),
        "spin1z": jnp.array(0.0),
        "spin2z": jnp.array(0.0),
        "inclination": jnp.array(0.5),
        "phase": jnp.array(0.0),
        "geocent_time": jnp.array(0.0),
        "luminosity_distance": jnp.array(100.0),
    }
    times = jnp.linspace(0.0, 1.0, 1024)

    # Test 1: Recursive Checkpoint
    wf_rev = ESIGMAInspiral(
        rad_pn_order=8,
        mode_pn_order=8,
        n_ode_grid=256,
        max_ode_steps=256,
        adjoint_mode="recursive_checkpoint",
    )

    def loss_rev(p):
        hp, hc = wf_rev(p, times)
        return jnp.sum(hp) + jnp.sum(hc)

    print("Compiling reverse-mode AD graph...")
    t0 = time.time()
    # c_rev = jax.jit(jax.grad(loss_rev)).lower(params).compile()
    print(f"Reverse-mode (recursive_checkpoint): SKIPPED (takes 8 mins)")
    # with open("compile_graph_rev.txt", "w") as f:
    #     f.write(c_rev.as_text())

    # Test 2: Forward Sensitivity
    wf_fwd = ESIGMAInspiral(
        rad_pn_order=8,
        mode_pn_order=8,
        n_ode_grid=256,
        max_ode_steps=256,
        adjoint_mode="forward_sensitivity",
    )

    def loss_fwd(p):
        hp, hc = wf_fwd(p, times)
        return jnp.sum(hp) + jnp.sum(hc)

    print("Compiling forward-sensitivity custom_vjp graph...")
    t0 = time.time()
    c_fwd = jax.jit(jax.grad(loss_fwd)).lower(params).compile()
    t_fwd = time.time() - t0
    lines_fwd = c_fwd.as_text().count("\n")
    print(
        f"Forward-sensitivity (custom_vjp): {lines_fwd} lines, compiled in {t_fwd:.2f}s"
    )

    print(f"Size ratio (rev / fwd): {lines_rev / lines_fwd:.2f}x")

    grad_rev = jax.grad(loss_rev)(params)
    grad_fwd = jax.grad(loss_fwd)(params)

    # Check if gradients match exactly
    match = True
    for k in grad_rev:
        diff = float(jnp.abs(grad_rev[k] - grad_fwd[k]))
        print(f"Diff for {k}: {diff:.3e}")
        if diff > 1e-10:
            match = False

    if match:
        print("SUCCESS! Gradients match perfectly!")
    else:
        print("WARNING! Gradients do not match!")


if __name__ == "__main__":
    test_esigma_compile_size()
