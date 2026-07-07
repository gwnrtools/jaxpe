"""Compile-graph size (HLO lines) vs per-step RHS complexity, and vs max_steps."""

import time, json
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax as dfx


def make_rhs(cost):
    # cost = degree of unrolled work per RHS call: a proxy for PN expression size
    def rhs(t, y, args):
        a, b = args
        acc0 = -a * y[0] + jnp.sin(y[1])
        acc1 = -b * y[1] + jnp.cos(y[0])
        p = y[0]
        for k in range(cost):
            p = p * y[1] + jnp.sin(p) * 0.001 + jnp.cos(y[0] * k * 1e-3)
        return jnp.stack([acc0 + 1e-9 * p, acc1 + 1e-9 * p])

    return rhs


def build(max_steps, n_save, cost):
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
            adjoint=dfx.RecursiveCheckpointAdjoint(checkpoints=16),
            throw=False,
        )
        return jnp.sum(sol.ys)

    return solve


args0 = (jnp.array(2.0), jnp.array(3.0))


def size(fn):  # compile and count optimized-HLO lines
    t0 = time.time()
    c = jax.jit(fn).lower(args0).compile()
    dt = time.time() - t0
    return dt, c.as_text().count("\n")


out = {"cost": [], "max_steps": []}
for cost in [5, 10, 20, 40, 80, 160]:  # driver axis: per-step RHS complexity
    loss = build(1024, 256, cost)
    dtf, nf = size(loss)  # forward-only
    dtr, nr = size(jax.grad(loss))  # reverse-mode gradient (what MALA needs)
    out["cost"].append({"cost": cost, "fwd_lines": nf, "rev_lines": nr})
    print(f"cost={cost:4d}  fwd={nf:>8}  rev={nr:>9}")

for ms in [256, 1024, 4096, 16384]:  # doc's suspected driver -> expect FLAT
    _, nr = size(jax.grad(build(ms, 256, 40)))
    out["max_steps"].append({"max_steps": ms, "rev_lines": nr})
    print(f"max_steps={ms:6d}  rev={nr:>9}")

json.dump(out, open("scaling_data.json", "w"), indent=2)
