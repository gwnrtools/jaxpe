import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax as dfx
import numpy as np


def make_rhs(cost):
    def rhs(t, y, args):
        a, b = args
        acc0 = -a * y[0] + jnp.sin(y[1])
        acc1 = -b * y[1] + jnp.cos(y[0])
        p = y[0]
        for k in range(cost):
            p = p * y[1] + jnp.sin(p) * 0.001 + jnp.cos(y[0] * k * 1e-3)
        return jnp.stack([acc0 + 1e-9 * p, acc1 + 1e-9 * p])

    return rhs


def build_primal(max_steps, n_save, cost):
    ts = jnp.linspace(0.0, 1.0, n_save)
    rhs = make_rhs(cost)

    def solve(args):
        sol = dfx.diffeqsolve(
            terms=dfx.ODETerm(rhs),
            solver=dfx.Tsit5(),
            t0=0.0,
            t1=1.0,
            dt0=1.0 / n_save,
            y0=jnp.array([1.0, 0.5]),
            args=args,
            saveat=dfx.SaveAt(ts=ts),
            stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-6),
            max_steps=max_steps,
            adjoint=dfx.DirectAdjoint(),
            throw=False,
        )
        return sol.ys

    return solve


def build_custom_vjp_sum(max_steps, n_save, cost):
    # original ODE returning raw states
    solve_raw = build_primal(max_steps, n_save, cost)

    @jax.custom_vjp
    def solve_wrapper(a, b):
        return solve_raw((a, b))

    def fwd(a, b):
        args = (a, b)
        ys = solve_raw(args)
        # compute full jacobian via forward mode
        # jacfwd differentiates wrt args!
        J_a, J_b = jax.jacfwd(solve_raw)(args)
        return ys, (J_a, J_b)

    def bwd(res, g_ys):
        J_a, J_b = res
        # contract cotangents with Jacobian
        # g_ys has shape (n_save, 2), J_a has shape (n_save, 2)
        grad_a = jnp.sum(g_ys * J_a)
        grad_b = jnp.sum(g_ys * J_b)
        return (grad_a, grad_b)

    solve_wrapper.defvjp(fwd, bwd)

    def loss(args):
        a, b = args
        ys = solve_wrapper(a, b)
        return jnp.sum(ys)

    return loss


def test_derivatives():
    args0 = (jnp.array(2.0), jnp.array(3.0))
    cost = 20
    max_steps = 1024
    n_save = 256

    # 1. Standard reverse-mode (backprop through the loop)
    ts = jnp.linspace(0.0, 1.0, n_save)
    rhs = make_rhs(cost)

    def solve_standard(args):
        sol = dfx.diffeqsolve(
            terms=dfx.ODETerm(rhs),
            solver=dfx.Tsit5(),
            t0=0.0,
            t1=1.0,
            dt0=1.0 / n_save,
            y0=jnp.array([1.0, 0.5]),
            args=args,
            saveat=dfx.SaveAt(ts=ts),
            stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-6),
            max_steps=max_steps,
            adjoint=dfx.RecursiveCheckpointAdjoint(checkpoints=16),
            throw=False,
        )
        return jnp.sum(sol.ys)

    grad_standard = jax.grad(solve_standard)(args0)

    # 2. Custom VJP (forward sensitivity equations)
    loss_custom = build_custom_vjp_sum(max_steps, n_save, cost)
    grad_custom = jax.grad(loss_custom)(args0)

    # 3. Finite Differences (Ground Truth)
    eps = 1e-6
    v0 = solve_standard(args0)
    v_a = solve_standard((args0[0] + eps, args0[1]))
    v_b = solve_standard((args0[0], args0[1] + eps))
    grad_fd = ((v_a - v0) / eps, (v_b - v0) / eps)

    print("Standard Reverse-Mode (Backprop):", [float(x) for x in grad_standard])
    print("Custom VJP (Forward Sensitivities):", [float(x) for x in grad_custom])
    print("Finite Differences (eps=1e-6):   ", [float(x) for x in grad_fd])

    diff_a = abs(grad_standard[0] - grad_custom[0])
    diff_b = abs(grad_standard[1] - grad_custom[1])
    print(f"Absolute Difference: ({diff_a:.3e}, {diff_b:.3e})")

    if np.allclose(grad_standard, grad_custom, rtol=1e-12, atol=1e-12):
        print("SUCCESS! The derivatives match perfectly.")
    else:
        print("WARNING! The derivatives differ.")


if __name__ == "__main__":
    test_derivatives()
