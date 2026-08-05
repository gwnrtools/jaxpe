#!/usr/bin/env python
r"""Fixed-vs-marginal cost model for the jaxpe sampling loops.

Why this exists
---------------
Point measurements ("a production block is 6.6 s, of which the local HMC is 3.6 s")
are not enough to optimise against: a sum that balances at one configuration does
not predict what happens when you change the configuration. During the BNS/CE
benchmark work a point-sum "closed" the block budget and was used to argue no
overhead remained -- while ``run_chains`` was in fact carrying **2.25 s of fixed
cost per call**, because it initialised chains with a bare ``jax.vmap(kernel.init)``
*outside* the jit and dispatched the target's ~3600-instruction gradient graph
eagerly. Fitting ``T = fixed + marginal x work`` across several sizes exposed it
immediately (see ``docs/bns_ce_pe_benchmark.md``). This tool is that fit, made
repeatable -- most usefully when moving to different hardware, where the question
is *which* costs move.

What it measures
----------------
``grad``     likelihood-gradient wall time vs chain count. Sublinear scaling means
             the device is latency/occupancy-bound rather than FLOP-bound, which is
             what decides whether a bigger GPU will help at all.
``local``    ``run_chains`` timed at several ``n_steps``, fitted to fixed + per-gradient.
             A large fixed term means "fewer, bigger blocks", not "cheaper blocks".
``global``   ``_global_block`` timed at several ``n_global``, fitted the same way.
             Splits flow-proposal cost from likelihood cost.
``flow``     global-block cost vs flow capacity (layers x width). The flow runs *two*
             passes per proposal, so it is easy to have it dominate unnoticed.
``init``     regression guard: eager vs jitted ``kernel.init``. These should now be
             within ~2x of each other. A 100x+ gap means the eager-dispatch bug is
             back.

Methodology, learned the hard way
---------------------------------
* **Warm at the exact size you time.** ``n_steps`` / ``n_global`` are *static*
  arguments, so a 10-step warmup does not warm the 1200-step trace and the timing
  silently includes ~1.3 s of compilation.
* **Profile on the backend the run uses.** Diagnostics timed under
  ``JAX_PLATFORMS=cpu`` do not bound the same code running on the GPU.
* **Time the component in the context that executes it.** A standalone
  ``vmap(grad(loglike))`` and the same gradient inside a ``lax.scan`` over an HMC
  trajectory differ measurably (3.2 vs 4.5 ms here) -- and the standalone figure is
  the one that misled.

Run:
    python bin/profile_sampler_scaling.py                    # all sections
    python bin/profile_sampler_scaling.py --only grad local  # a subset
    python bin/profile_sampler_scaling.py --quick            # smaller/faster setup
"""

import argparse
import importlib.util
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_DRIVER = Path(__file__).with_name("run_bns_ce_pe.py")

SECTIONS = ("grad", "local", "global", "flow", "init")


def _load_driver():
    """Import the benchmark driver for its lean-likelihood builder.

    The PSD now comes from ``jaxpe.gw.lalsim_psd``; only ``build_loglike`` is still
    reached this way.
    """
    spec = importlib.util.spec_from_file_location("_bns_driver", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def warm_time(fn, reps: int = 3):
    """Minimum wall time over ``reps`` calls, after one warm-up call.

    The warm-up must run at the *same* shapes and static arguments as the timed
    calls, otherwise the first timed call pays for compilation.
    """
    jax.block_until_ready(fn())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        jax.block_until_ready(r)
        ts.append(time.perf_counter() - t0)
    return min(ts)


def build_problem(args):
    """The benchmark's own relative-binning log-posterior, on the sampling device.

    Heavy setup is CPU-pinned exactly as the benchmark does it, so this measures
    the sampling hot loop rather than the dense-grid construction.
    """
    from jaxpe.core.priors import JointPrior, Uniform
    from jaxpe.core.problem import InferenceProblem
    from jaxpe.gw import IMRPhenomD, lalsim_psd, make_injection
    from jaxpe.gw.likelihood import RelativeBinningFDLikelihood

    drv = _load_driver()
    m1, m2 = args.mass1, args.mass2
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    eta = m1 * m2 / (m1 + m2) ** 2
    truth = dict(
        chirp_mass=mc,
        mass_ratio=m2 / m1,
        spin1z=0.0,
        spin2z=0.0,
        luminosity_distance=args.distance,
        geocent_time=1187008882.43,
        phase=1.3,
        inclination=0.4,
        ra=3.446,
        dec=-0.408,
        psi=0.8,
    )
    with jax.default_device(jax.devices("cpu")[0]):
        n = int(args.duration * args.sampling_rate)
        freqs = np.fft.rfftfreq(n, d=1.0 / args.sampling_rate)
        psd = lalsim_psd("CE", freqs)
        dense = make_injection(
            IMRPhenomD(f_ref=args.f_min),
            truth,
            detector_names=("H1",),
            duration=args.duration,
            sampling_rate=args.sampling_rate,
            f_min=args.f_min,
            f_max=None,
            psd_fn=lambda f: np.interp(f, freqs, psd),
            noise_seed=None,
        )
        rb = RelativeBinningFDLikelihood.from_likelihood(
            dense, truth, chi=1.0, epsilon=args.epsilon
        )
        loglike = drv.build_loglike(rb, truth)
        prior = JointPrior(
            {
                "chirp_mass": Uniform(0.9 * mc, 1.1 * mc),
                "eta": Uniform(args.eta_min, 0.25),
                "spin1z": Uniform(0.0, args.spin_max),
                "spin2z": Uniform(0.0, args.spin_max),
            }
        )
        problem = InferenceProblem(prior=prior, log_likelihood=loglike)
        y_c = np.asarray(
            prior.to_unconstrained(
                jnp.asarray([mc, min(eta, 0.25 - 1e-4), 0.002, 0.002])
            )
        )
    print(f"  problem: {rb.n_bins} bins, Mc={mc:.5f} Msun", flush=True)
    return problem, jnp.asarray(y_c)


def _start(y_c, n_chains, key=0):
    return y_c[None, :] + 1e-3 * jax.random.normal(
        jax.random.key(key), (n_chains, y_c.size)
    )


def section_grad(problem, y_c, args):
    """Gradient cost vs chain count: is the device occupancy-bound?"""
    print("\n== likelihood gradient vs chain count ==")
    print(f"{'chains':>8} {'grad ms':>10} {'per chain us':>14}")
    logp = problem.log_posterior
    g = jax.jit(jax.vmap(jax.grad(logp)))
    rows = []
    for nc in args.chains:
        x = _start(y_c, nc)
        dt = warm_time(lambda x=x: g(x))
        rows.append((nc, dt))
        print(f"{nc:>8} {dt * 1e3:>10.3f} {dt / nc * 1e6:>14.1f}")
    lo, hi = rows[0], rows[-1]
    work = hi[0] / lo[0]
    cost = hi[1] / lo[1]
    print(
        f"  {work:.0f}x the chains for {cost:.1f}x the time -> "
        + (
            "SUBLINEAR: latency/occupancy-bound, more FLOPs will not help much"
            if cost < 0.6 * work
            else "near-linear: throughput-bound, a faster device should help"
        )
    )


def section_local(problem, y_c, args):
    """run_chains fitted to fixed-per-call + marginal-per-gradient."""
    from jaxpe.kernels import HMC, run_chains

    print("\n== run_chains vs n_steps ==")
    logp = problem.log_posterior
    x0 = _start(y_c, args.n_chains)
    scale = np.linalg.cholesky(np.diag(np.full(y_c.size, 1e-2) ** 2))
    kern = HMC(step_size=0.25, n_leapfrog=args.n_leapfrog, scale=scale)
    xs, ys = [], []
    print(f"{'steps':>7} {'grads':>8} {'time s':>9}")
    for s in args.steps:
        dt = warm_time(
            lambda s=s: run_chains(jax.random.key(1), kern, logp, x0, s, thin=2)
        )
        xs.append(s * args.n_leapfrog)
        ys.append(dt)
        print(f"{s:>7} {s * args.n_leapfrog:>8} {dt:>9.3f}")
    slope, fixed = np.polyfit(xs, ys, 1)
    print(f"  fit: {slope * 1e3:.4f} ms/gradient + {fixed:.3f} s FIXED per call")
    if fixed > 0.25 * (ys[-1]):
        print(
            "  WARNING: the fixed term dominates. Prefer FEWER, BIGGER blocks -- and\n"
            "           check that chain init is inside the jit (see the 'init' section)."
        )
    return slope, fixed


def section_global(problem, y_c, args):
    """_global_block fitted to fixed-per-call + marginal-per-proposal."""
    from jaxpe.flows import fit_flow, make_flow
    from jaxpe.sampler.global_local import _global_block

    print("\n== _global_block vs n_global ==")
    logp = problem.log_posterior
    x0 = _start(y_c, args.n_chains)
    lp0 = jax.vmap(logp)(x0)
    train = y_c[None, :] + 0.5 * jax.random.normal(jax.random.key(9), (4096, y_c.size))
    flow = make_flow(
        jax.random.key(1),
        y_c.size,
        flow_layers=args.flow_layers,
        nn_width=args.flow_width,
    )
    flow, _ = fit_flow(jax.random.key(2), flow, train, n_epochs=2, batch_size=512)
    xs, ys = [], []
    print(f"{'globals':>9} {'time s':>9}")
    for gsz in args.globals_:
        dt = warm_time(
            lambda gsz=gsz: _global_block(flow, jax.random.key(3), x0, lp0, logp, gsz)
        )
        xs.append(gsz)
        ys.append(dt)
        print(f"{gsz:>9} {dt:>9.3f}")
    slope, fixed = np.polyfit(xs, ys, 1)
    print(f"  fit: {slope * 1e3:.4f} ms/proposal + {fixed:.3f} s FIXED per call")

    v = jax.jit(jax.vmap(logp))
    n = xs[-1]
    t = warm_time(lambda: v(x0))
    print(
        f"  likelihood alone: {t * 1e3:.3f} ms/call -> of the {slope * 1e3:.3f} ms per\n"
        f"  proposal, ~{t * 1e3:.3f} ms is the target and the rest is the flow's two\n"
        f"  passes (sample + log_prob). n={n} used for the fit."
    )
    return slope, fixed


def section_flow(problem, y_c, args):
    """Global-block cost vs flow capacity -- the flow runs two passes per proposal."""
    from jaxpe.flows import fit_flow, make_flow
    from jaxpe.sampler.global_local import _global_block

    print("\n== global block vs flow capacity ==")
    logp = problem.log_posterior
    x0 = _start(y_c, args.n_chains)
    lp0 = jax.vmap(logp)(x0)
    train = y_c[None, :] + 0.5 * jax.random.normal(jax.random.key(9), (4096, y_c.size))
    g = args.globals_[-1]
    print(f"{'layers':>7} {'width':>6} {'block s':>9} {'per step ms':>12}")
    for layers, width in args.flow_grid:
        fl = make_flow(jax.random.key(1), y_c.size, flow_layers=layers, nn_width=width)
        fl, _ = fit_flow(jax.random.key(2), fl, train, n_epochs=2, batch_size=512)
        dt = warm_time(
            lambda fl=fl: _global_block(fl, jax.random.key(3), x0, lp0, logp, g)
        )
        print(f"{layers:>7} {width:>6} {dt:>9.3f} {dt / g * 1e3:>12.3f}")
    print(
        "  A cheaper flow is only a win if the BLOCK COUNT holds -- judge it end to\n"
        "  end, never on per-block cost alone."
    )


def section_init(problem, y_c, args):
    """Regression guard for the eager-dispatch chain-init bug."""
    from jaxpe.kernels import HMC

    print("\n== chain init: eager vs jitted dispatch ==")
    logp = problem.log_posterior
    x0 = _start(y_c, args.n_chains)
    scale = np.linalg.cholesky(np.diag(np.full(y_c.size, 1e-2) ** 2))
    kern = HMC(step_size=0.25, n_leapfrog=args.n_leapfrog, scale=scale)
    eager = warm_time(lambda: jax.vmap(lambda x: kern.init(x, logp))(x0))
    jitted_fn = jax.jit(jax.vmap(lambda x: kern.init(x, logp)))
    jitted = warm_time(lambda: jitted_fn(x0))
    print(f"  eager  vmap(kernel.init): {eager:7.3f} s")
    print(
        f"  jitted vmap(kernel.init): {jitted:7.3f} s   ({eager / max(jitted, 1e-9):.0f}x)"
    )
    print(
        "  run_chains must initialise INSIDE the jit. It did not until this was found:\n"
        "  the eager path dispatched the target's ~3600-instruction gradient graph one\n"
        "  op at a time, costing ~2.2 s on EVERY call regardless of n_steps."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=SECTIONS, default=list(SECTIONS))
    ap.add_argument("--n-chains", type=int, default=64)
    ap.add_argument("--n-leapfrog", type=int, default=32)
    ap.add_argument("--chains", type=int, nargs="+", default=[32, 64, 256, 1024])
    ap.add_argument("--steps", type=int, nargs="+", default=[6, 12, 25, 50])
    ap.add_argument(
        "--globals",
        dest="globals_",
        type=int,
        nargs="+",
        default=[300, 600, 1200, 2400],
    )
    ap.add_argument("--flow-layers", type=int, default=4)
    ap.add_argument("--flow-width", type=int, default=64)
    ap.add_argument("--mass1", type=float, default=1.4)
    ap.add_argument("--mass2", type=float, default=1.4)
    ap.add_argument("--distance", type=float, default=200.0)
    ap.add_argument("--duration", type=float, default=2048.0)
    ap.add_argument("--sampling-rate", type=float, default=4096.0)
    ap.add_argument("--f-min", type=float, default=10.0)
    ap.add_argument("--epsilon", type=float, default=0.25)
    ap.add_argument("--eta-min", type=float, default=0.2)
    ap.add_argument("--spin-max", type=float, default=0.05)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="short segment / coarse bins: seconds of setup, for smoke-testing",
    )
    args = ap.parse_args()
    if args.quick:
        args.duration, args.sampling_rate, args.epsilon = 128.0, 1024.0, 0.5
        args.chains = [32, 64, 256]
        args.steps = [6, 12, 25]
        args.globals_ = [150, 300, 600]
    args.flow_grid = [(8, 64), (4, 64), (4, 32), (2, 64)]

    print(f"jax {jax.__version__}, backend: {jax.default_backend()}")
    print("building the problem (heavy setup is CPU-pinned, as in the benchmark)...")
    problem, y_c = build_problem(args)

    model = {}
    if "grad" in args.only:
        section_grad(problem, y_c, args)
    if "local" in args.only:
        model["local"] = section_local(problem, y_c, args)
    if "global" in args.only:
        model["global"] = section_global(problem, y_c, args)
    if "flow" in args.only:
        section_flow(problem, y_c, args)
    if "init" in args.only:
        section_init(problem, y_c, args)

    if "local" in model and "global" in model:
        (ls, lf), (gs, gf) = model["local"], model["global"]
        fixed = lf + gf
        print(
            f"\n== cost model ==\n  T_block = {fixed:.2f} s FIXED"
            f" + {ls * 1e3:.3f} ms x grads + {gs * 1e3:.4f} ms x globals"
        )
        for s, g in ((25, 1200), (12, 1200)):
            grads = s * args.n_leapfrog
            print(
                f"  steps={s:3d} globals={g:5d}: "
                f"{fixed + ls * grads + gs * g:.2f} s/block"
            )
        print(
            "  Compare against the per-block times the benchmark prints. A gap means a\n"
            "  cost outside these two loops (diagnostics, refits, host transfers)."
        )


if __name__ == "__main__":
    raise SystemExit(main())
