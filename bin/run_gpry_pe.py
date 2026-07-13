#!/usr/bin/env python
"""Surrogate (GPry) PE driver for expensive mode-based waveform models.

Phase-1 deliverable of the GPry-fusion design (docs/gpry_fusion_design.md, tasks
1.2 + 1.5): drives ``MarginalizedIntrinsicLikelihood -> GPryEngine`` to a surrogate
posterior over the intrinsic parameters, and reports the wall-clock split
(waveform / marginalization / GP fit / acquisition / convergence / final MC) that
feeds the design's D4 profiling checkpoint.

Only the self-contained ``--demo`` problem (2D synthetic chirp, zero-noise
injection) is wired up until the Phase-3 external-model wrappers (TEOBResumS,
SEOBNRv6EHM) land; the driver structure is model-agnostic.

Usage (from the repo root, CPU):
    JAX_PLATFORMS=cpu conda run -n lalsuite-dev python bin/run_gpry_pe.py \
        --demo --output output/gpry_demo [--full-marginal] [--seed 11]
"""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

# float64 is REQUIRED: GPS times (~1.1e9 s) and Whittle accumulations are meaningless
# in float32 (lnL silently becomes NaN). Must be set before any jax array is created.
jax.config.update("jax_enable_x64", True)


def build_demo_problem(full_marginal: bool):
    """2D synthetic-chirp pseudo-black-box (same construction as tests/test_surrogate.py)."""
    import jax.numpy as jnp

    from jaxpe.gw import make_injection, spin_weighted_ylm
    from jaxpe.gw.external_models import ModesData, reflect_modes
    from jaxpe.gw.marginalized import (
        MarginalizedIntrinsicLikelihood,
        ModesNetworkLikelihood,
    )

    t_c, duration, sr, post_trigger, d_ref = 1126259462.4, 8.0, 2048.0, 2.0, 500.0
    truth = dict(f0=37.0, span=55.0)
    bounds = {"f0": (30.0, 45.0), "span": (40.0, 80.0)}
    extrinsic = dict(
        inclination=0.6, phase=1.2, luminosity_distance=d_ref,
        ra=1.95, dec=-1.27, psi=0.82, geocent_time=t_c,
    )
    n = int(duration * sr)
    times = t_c + post_trigger - duration + np.arange(n) / sr

    def chirp_modes(theta):
        t = times - t_c
        t_on, t_off = -1.5, -0.1
        u = np.clip((t - t_on) / (t_off - t_on), 0.0, 1.0)
        env = np.where((t > t_on) & (t < t_off), np.sin(np.pi * u) ** 2, 0.0)
        tau = t - t_on
        ph = 2 * np.pi * (theta["f0"] * tau + 0.5 * theta["span"] / (t_off - t_on) * tau**2)
        h22 = 1e-22 * env * np.exp(-1j * ph)
        return reflect_modes({(2, 2): h22, (3, 3): 0.4e-22 * env * np.exp(-1.5j * ph)})

    def mode_model(theta):
        return ModesData(modes=chirp_modes(theta), times=times, d_ref_mpc=d_ref, t_ref=t_c)

    md_true = mode_model(truth)

    class _Wf:  # injection-only traceable assembler
        def __call__(self, params, _):
            h = jnp.zeros((n,), dtype=jnp.complex128)
            for (l, m), hlm in md_true.modes.items():
                h = h + jnp.asarray(hlm) * spin_weighted_ylm(
                    params["inclination"], params["phase"], l, m
                )
            h = h * (d_ref / params["luminosity_distance"])
            return h.real, -h.imag

    like_td = make_injection(
        _Wf(), extrinsic, detector_names=("H1", "L1"), duration=duration,
        sampling_rate=sr, post_trigger=post_trigger, noise_seed=None,
    )
    like_modes = ModesNetworkLikelihood.from_likelihood(like_td, md_true)
    settings = dict(n_phi=128, n_dist=64, tc_half_samples=10)
    if full_marginal:
        settings.update(n_pilot=1024, n_final=1024)
    lik = MarginalizedIntrinsicLikelihood(
        mode_model, like_modes, names=tuple(bounds), t_center=t_c,
        marginalize_sky=full_marginal,
        fixed_extrinsic=None if full_marginal else extrinsic,
        settings=settings,
    )
    return lik, bounds, truth


class TimedLoglike:
    """Wraps the loglike, splitting waveform-generation from marginalization time."""

    def __init__(self, lik):
        self.lik = lik
        self.t_waveform = 0.0
        self.t_total = 0.0
        self.n_calls = 0
        inner_model = lik.mode_model

        def timed_model(theta):
            t0 = time.perf_counter()
            out = inner_model(theta)
            self.t_waveform += time.perf_counter() - t0
            return out

        lik.mode_model = timed_model

    def __call__(self, x):
        t0 = time.perf_counter()
        out = self.lik(x)
        self.t_total += time.perf_counter() - t0
        self.n_calls += 1
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo", action="store_true", help="run the built-in 2D demo")
    ap.add_argument("--full-marginal", action="store_true",
                    help="marginalize sky/psi/iota too (adaptive IS) instead of fixing them")
    ap.add_argument("--output", default="output/gpry_pe", help="output directory")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--verbose", type=int, default=2)
    args = ap.parse_args()

    if not args.demo:
        raise SystemExit(
            "Only --demo is wired up until the Phase-3 external-model wrappers land."
        )

    from jaxpe.surrogate import GPryEngine

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    lik, bounds, truth = build_demo_problem(args.full_marginal)
    timed = TimedLoglike(lik)
    engine = GPryEngine(
        timed, bounds=bounds, options={"seed": args.seed}, verbose=args.verbose
    )

    t0 = time.perf_counter()
    diag = engine.run()
    t_run = time.perf_counter() - t0
    t0 = time.perf_counter()
    samples = engine.sample()
    t_mc_final = time.perf_counter() - t0

    # ---- profiling split (design note D4 checkpoint) ----
    prog = engine.runner.progress.data
    gpry_t = {
        k: float(np.nansum(prog[f"time_{k}"]))
        for k in ("acquire", "truth", "fit", "convergence", "mc")
        if f"time_{k}" in prog
    }
    t_marginal = timed.t_total - timed.t_waveform
    non_truth = sum(v for k, v in gpry_t.items() if k != "truth") + t_mc_final
    total = t_run + t_mc_final
    profile = dict(
        n_truth_calls=timed.n_calls,
        truth_waveform_s=round(timed.t_waveform, 2),
        truth_marginalization_s=round(t_marginal, 2),
        gp_fit_s=round(gpry_t.get("fit", 0.0), 2),
        acquisition_s=round(gpry_t.get("acquire", 0.0), 2),
        convergence_s=round(gpry_t.get("convergence", 0.0), 2),
        final_mc_s=round(gpry_t.get("mc", 0.0) + t_mc_final, 2),
        total_s=round(total, 2),
        non_truth_share=round(non_truth / max(total, 1e-9), 3),
    )

    np.savez(
        out / "posterior.npz",
        x=samples.x, weights=samples.weights, logpost=samples.logpost,
        names=np.array(samples.names),
    )
    (out / "diagnostics.json").write_text(
        json.dumps(dict(diag=diag, profile=profile, truth=truth), indent=2)
    )

    m = np.average(samples.x, weights=samples.weights, axis=0)
    sd = np.sqrt(np.average((samples.x - m) ** 2, weights=samples.weights, axis=0))
    print("\n=== GPry PE summary ===")
    print(f"converged: {diag['has_converged']}   truth evals: {diag['n_truth_evals']}")
    for i, nm in enumerate(samples.names):
        print(f"  {nm}: {m[i]:.3f} +- {sd[i]:.3f}   (truth {truth[nm]})")
    print("=== wall-clock profile ===")
    for k, v in profile.items():
        print(f"  {k}: {v}")
    print(
        "D4 checkpoint: revisit JAX-porting a GPry component only if non_truth_share"
        f" > 0.30 with a *production* (minutes-per-call) waveform; here it is"
        f" {profile['non_truth_share']} against a millisecond demo waveform, so a"
        " large share is expected and NOT actionable."
    )
    print(f"outputs in {out}/")


if __name__ == "__main__":
    main()
