#!/usr/bin/env python
r"""Mass-sweep PE suite: N injections from BNS to BBH masses, each run by run_bns_ce_pe.py.

This is a *driver*, not a new sampler pipeline: every injection is run by the
already-validated HMC + FD relative-binning engine in ``run_bns_ce_pe.py``
(dense-Laplace-mass HMC, flow global proposals, rank-normalized split-Rhat /
Geyer-ESS convergence gate -- see that script's docstring and
``docs/bns_ce_pe_benchmark.md`` for what "rigorous" means here and why). This
script only decides, per injection: what total mass, what segment duration, and
what distance -- then shells out and collects the results. Nothing about the
sampler, the likelihood, or the waveform is touched.

Injections run **strictly sequentially, one subprocess at a time**. This machine's
GPU is a 4 GB part shared with the desktop session (see
``XLA_PYTHON_CLIENT_MEM_FRACTION`` below) -- concurrent JAX processes on it is a
measured way to OOM or corrupt an unrelated session, not a speed lever. A
subprocess-per-injection design is also what makes that safe *and* simple: each
injection gets a clean process, so its distinct static shapes (duration ->
n_bins is different at every mass) never accumulate compiled executables or
fragment device memory across the sweep, and a crash on one injection cannot take
the rest of the suite down with it.

What varies across the sweep
-----------------------------
- **Total mass**: log-spaced from ``--total-mass-min`` (BNS-like, default 2.8
  Msun) to ``--total-mass-max`` (default 80 Msun). Component masses are an equal
  split by default (``--mass-ratio 1.0``); a fixed non-unit ratio is supported but
  NOT swept -- sweeping mass *and* mass ratio at once conflates two independent
  questions ("does the sampler generalize across total mass" vs "... across
  asymmetry"), and the latter is already covered by the 1.35+1.25 Msun generality
  check recorded in ``docs/bns_ce_pe_benchmark.md``.
- **Segment duration**: sized per injection from the LALSimulation chirp/merger/
  ringdown time bounds (``SimInspiralChirpTimeBound`` etc.) rather than reusing
  the BNS default of 2048 s for every mass. A 40+40 Msun signal is in band for
  ~5 s from 10 Hz; running it in a 2048 s segment would waste the setup cost on
  ~400x more frequency bins than the signal needs for no accuracy gain, while a
  fixed *short* duration would truncate the BNS end of the sweep and alias the
  inspiral. Sampling rate stays fixed at 4096 Hz across the whole sweep: it is set
  by the highest merger frequency in the sweep, which is the *lowest*-mass system
  (BNS), already the validated case in ``run_bns_ce_pe.py``; every higher-mass
  system merges at a lower frequency and is easier, not harder, to resolve.
- **Distance**, solved so the network SNR lands near ``--target-snr`` with a
  per-injection jitter (``--snr-jitter``, default +-15%) so no two injections in
  the suite carry an identical SNR -- forwarded as ``--target-snr`` to
  ``run_bns_ce_pe.py`` itself (added there alongside this script; see its
  docstring), which does the actual rescale exactly (SNR ~ 1/D for a fixed
  source, so this needs no root-finding).

What is assumed and NOT swept (stated explicitly; change if you need otherwise)
--------------------------------------------------------------------------------
- Zero spin truth at every mass (matches ``run_bns_ce_pe.py``'s own convention);
  the recovered ``spin1z``/``spin2z`` prior width is the same ``--spin-max`` at
  every mass.
- Equal mass (``eta_true = 0.25`` exactly) at every point unless ``--mass-ratio``
  is set, so the eta-boundary-pileup degeneracy structure that the sampler's
  defaults (flow spline interval, eigenvalue-floored Laplace mass) were tuned
  against is the SAME shape at every mass in the sweep, isolating the effect of
  total mass and SNR. This is very much NOT validated to hold at 80 Msun the way
  it is at BNS masses -- that is exactly what running this suite will tell you.
- Sky location, orientation, polarization, and reference phase are held fixed at
  ``run_bns_ce_pe.py``'s own defaults for every injection, so SNR variation across
  the suite comes only from the deliberate distance/jitter above, not from
  orientation lottery.
- All other sampler knobs (n_leapfrog, flow capacity, equilibration rounds, ...)
  are left at ``run_bns_ce_pe.py``'s measured defaults and forwarded unchanged
  (or via ``--extra-args``) -- they are DERIVED per injection already (the
  Laplace mass matrix and step size are refit at each source's own MAP), so no
  further per-mass tuning is threaded through this script.

Convergence, per injection: exactly the gate already in ``run_bns_ce_pe.py`` --
rank-normalized split-Rhat over the global-subseries < ``--rhat-target``, Geyer
min ESS >= ``--ess-target``, and no stuck chains, evaluated at every production
block up to a ``--max-minutes`` wall-clock budget. A run that exhausts the budget
is reported as NOT converged, not silently truncated. This script adds no second
convergence criterion of its own; it aggregates the validated one across the
suite and additionally reports, per parameter, |median - truth| / posterior sigma
as a recovery sanity check (not a coverage/calibration test -- that needs
repeated draws at fixed parameters, a different experiment design).

Profiling: every injection's full component-wise ``timings.json`` (psd,
injection, snr_rescale, rb_setup, map_laplace, rb_validation, warmup, flow_fit,
equilibration, production, total) is copied into the sweep summary verbatim, so
per-component scaling with mass is visible without re-deriving it. The sweep
summary also records the ``default backend: ...`` line each injection actually
printed (column ``backend``), so "did this run on the GPU" is read from the log,
not assumed from having launched it in the right conda env.

GPU and compile-time caveats
----------------------------
Backend selection is jax's own auto-detection (unchanged from
``run_bns_ce_pe.py``): whichever conda env this process runs in decides it, most
likely ``lalsuite-dev`` for a CUDA-enabled jaxlib here. ``--require-gpu`` sets
``JAX_PLATFORMS=cuda`` for every subprocess so a missing/invisible GPU raises
immediately instead of silently completing the entire sweep on CPU.

Every injection has a DIFFERENT duration -> a different n_bins -> different
compiled shapes, so unlike a single fixed-config run of ``run_bns_ce_pe.py``,
there is no "later injection" in the same sweep that reuses an earlier one's
JIT compile. ``--warm-cache`` runs one throwaway
``--max-production-blocks 1`` pass per injection first (into
``<outdir>_warmup``, discarded) to populate the persistent XLA cache
(``~/.cache/jaxpe_xla``, shared with ``run_bns_ce_pe.py``) before the timed run,
so its ``timings.json`` is the honest, compile-free basis this codebase's other
benchmarks already use -- at the cost of roughly doubling the setup + warmup +
equilibration time per injection (production is unaffected: it is not part of
the priming pass beyond its first block, and needs no priming since only its
FIRST block per injection pays a compile either way).

Run:
    python bin/run_mass_sweep_pe.py --n-injections 6 --dry-run   # preview only
    python bin/run_mass_sweep_pe.py --n-injections 6 --require-gpu --warm-cache
"""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Shared with the desktop session; see the module docstring. Set here too (not
# just relied upon via run_bns_ce_pe.py's own os.environ.setdefault) so it is
# visible in this process's env dump / any subprocess inherits it explicitly.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

TIMING_KEYS = (
    "psd",
    "injection",
    "snr_rescale",
    "rb_setup",
    "map_laplace",
    "rb_validation",
    "warmup",
    "flow_fit",
    "equilibration",
    "production",
    "total",
)


# --------------------------------------------------------------------------- grid
def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def symmetric_mass_ratio(m1, m2):
    return m1 * m2 / (m1 + m2) ** 2


def component_masses(total_mass, mass_ratio):
    """m1 >= m2 from total mass and q = m2/m1 <= 1."""
    m1 = total_mass / (1.0 + mass_ratio)
    m2 = mass_ratio * m1
    return m1, m2


def estimate_duration(m1, m2, f_min, safety, dur_min, dur_max):
    """Segment length from LALSimulation's own chirp/merger/ringdown time bounds.

    ``SimInspiralChirpTimeBound`` (and the merger/ringdown bounds alongside it)
    are the standard LALSimulation utilities for exactly this: a conservative
    upper bound on how long a signal from ``f_min`` to ringdown occupies, used
    throughout the field for segment-length planning. Using them here rather than
    a hand-rolled PN formula means the estimate is exactly what the rest of the
    LIGO/Virgo/CE tooling would use for the same source, not a bespoke
    approximation. Zero spin, matching the zero-spin truth injected everywhere in
    this sweep.

    The result is padded by ``safety``, clamped to ``[dur_min, dur_max]``, and
    rounded up to the next power of two (FFT-friendly; matches the 2048 s
    convention already used for the BNS reference configuration).
    """
    import lal
    import lalsimulation as ls

    m1_si, m2_si = m1 * lal.MSUN_SI, m2 * lal.MSUN_SI
    t_insp = ls.SimInspiralChirpTimeBound(f_min, m1_si, m2_si, 0.0, 0.0)
    t_merge = ls.SimInspiralMergeTimeBound(m1_si, m2_si)
    t_ring = ls.SimInspiralRingdownTimeBound(m1_si + m2_si, 0.0)
    t_est = safety * (t_insp + t_merge + t_ring)
    t_est = float(np.clip(t_est, dur_min, dur_max))
    return float(2.0 ** np.ceil(np.log2(max(t_est, 1.0))))


def build_grid(args):
    """Per-injection (m1, m2, duration, target_snr, seed) table.

    Total mass is log-spaced (a sweep from 2.8 to 80 Msun spans more than an
    order of magnitude; a linear grid would waste most of its points above
    ~30 Msun). SNR jitter and the mass grid itself are both drawn/spaced so no
    two injections coincide; see the module docstring for what is and is not
    varied.
    """
    n = args.n_injections
    total_masses = np.geomspace(args.total_mass_min, args.total_mass_max, n)
    rng = np.random.default_rng(args.seed)
    jitter = rng.uniform(-1.0, 1.0, size=n)
    target_snrs = args.target_snr * (1.0 + args.snr_jitter * jitter)

    rows = []
    for i, (mtot, snr_i) in enumerate(zip(total_masses, target_snrs)):
        m1, m2 = component_masses(float(mtot), args.mass_ratio)
        duration = estimate_duration(
            m1, m2, args.f_min, args.duration_safety, args.duration_min,
            args.duration_max,
        )
        rows.append(
            dict(
                index=i,
                total_mass=float(mtot),
                m1=m1,
                m2=m2,
                mc=chirp_mass(m1, m2),
                eta=symmetric_mass_ratio(m1, m2),
                duration=duration,
                target_snr=float(snr_i),
                seed=args.seed + i,
            )
        )
    return rows


# ------------------------------------------------------------------------- driver
def _build_cmd(row, args, outdir):
    return [
        sys.executable,
        str(args.run_script),
        "--outdir", str(outdir),
        "--mass1", f"{row['m1']:.6f}",
        "--mass2", f"{row['m2']:.6f}",
        "--distance", f"{args.reference_distance:.3f}",
        "--target-snr", f"{row['target_snr']:.4f}",
        "--duration", f"{row['duration']:.3f}",
        "--sampling-rate", f"{args.sampling_rate:.1f}",
        "--f-min", f"{args.f_min:.3f}",
        "--eta-min", f"{args.eta_min:.4f}",
        "--seed", str(row["seed"]),
        "--max-minutes", f"{args.max_minutes:.2f}",
        "--rhat-target", f"{args.rhat_target:.4f}",
        "--ess-target", f"{args.ess_target:.1f}",
    ] + shlex.split(args.extra_args)


def _subprocess_env(args):
    """Child env: inherits this process's env, optionally hardening the backend.

    ``--require-gpu`` sets ``JAX_PLATFORMS=cuda,cpu`` rather than leaving
    backend selection to jax's default auto-detection. Without it, a
    misconfigured environment (wrong conda env, GPU busy/invisible) silently
    falls back to CPU and you only find out after an N x 20 min sweep finishes
    slow -- with it, jax raises immediately if no CUDA device is visible, so
    the mistake costs seconds, not hours. ``cpu`` MUST stay in the list:
    ``run_bns_ce_pe.py`` explicitly pins its heavy setup to
    ``jax.devices("cpu")[0]``, and ``JAX_PLATFORMS=cuda`` alone makes that call
    raise ("Unknown backend cpu") instead of only excluding cpu from the
    *default*-backend choice -- both backends must be initialized, with cuda
    listed first so it wins the default.
    """
    env = os.environ.copy()
    if args.require_gpu:
        env["JAX_PLATFORMS"] = "cuda,cpu"
    return env


def _parse_backend(log_path):
    """Pull the ``default backend: <x>`` line run_bns_ce_pe.py prints at startup.

    This is the ground truth for "did it actually use the GPU", independent of
    what was requested -- read from the log rather than assumed.
    """
    try:
        with open(log_path) as f:
            for line in f:
                if "default backend:" in line:
                    return line.rsplit("default backend:", 1)[1].strip()
    except FileNotFoundError:
        pass
    return None


def _run_with_timeout(cmd, log_path, args, timeout_s):
    """subprocess.run with a hard wall-clock cap and one retry after a cooldown.

    Observed in practice: back-to-back subprocesses sharing one small consumer
    GPU can hit a multi-minute hang with ZERO output (not even the first
    ``print`` in ``main()``) -- most likely jax's lazy CUDA initialization
    stalling while the previous process's device context is still being torn
    down by the driver. Nothing in this script can fix that race, but it can
    refuse to let it silently kill the whole sweep: without a timeout, a hang
    like that either wedges the sweep forever or gets cleaned up by some
    OUTSIDE mechanism (observed: the child dies with zero flushed output and a
    truncated/corrupt ``samples.npz``, then ``np.load`` on that file raises and
    crashes the entire sweep with everything after it unrun). This function
    bounds the damage to one wasted ``cooldown_s + timeout_s`` per injection.

    Returns (returncode, timed_out: bool). A timeout is reported via
    returncode = -1 after the retry is also exhausted.
    """
    for attempt in range(2):
        with open(log_path, "w") as logf:
            try:
                proc = subprocess.run(
                    cmd, stdout=logf, stderr=subprocess.STDOUT,
                    env=_subprocess_env(args), timeout=timeout_s,
                )
                return proc.returncode, False
            except subprocess.TimeoutExpired:
                print(
                    f"  TIMEOUT after {timeout_s:.0f}s (attempt {attempt + 1}/2) "
                    f"-- subprocess killed, see {log_path}"
                )
                if attempt == 0:
                    time.sleep(args.cooldown_seconds)
    return -1, True


def _prime_cache(row, args, outdir):
    """Throwaway pass that forces every JIT compile site in the pipeline once.

    ``run_bns_ce_pe.py`` and this script share one persistent XLA cache
    (``~/.cache/jaxpe_xla``), keyed on device + compiled program + shapes, not
    on process identity -- so a subprocess that ran once already leaves the
    cache warm for the NEXT subprocess with the same shapes, exactly as if it
    were the same long-lived process (see that script's own docstring: "the
    persistent XLA cache makes every later one compile-free"). Every injection
    in this sweep has a distinct duration -> distinct n_bins -> distinct
    compiled shapes, so each one needs its OWN priming pass; there is no
    shortcut that reuses another injection's cache entries.

    ``--max-production-blocks 1`` (appended last, so it overrides anything in
    ``--extra-args``) is what keeps this cheap: it still runs setup, one
    warmup block, one retune block, the full equilibration loop (which already
    exercises fit_flow and the global block), and exactly one production
    block -- enough to touch every distinct static shape in the pipeline --
    without paying for a full run to convergence twice.
    """
    warm_outdir = outdir.parent / f"{outdir.name}_warmup"
    warm_outdir.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd(row, args, warm_outdir) + ["--max-production-blocks", "1"]
    log_path = warm_outdir / "run.log"
    t0 = time.perf_counter()
    _rc, timed_out = _run_with_timeout(cmd, log_path, args, args.injection_timeout * 60.0)
    wall = time.perf_counter() - t0
    backend = _parse_backend(log_path)
    note = " (TIMED OUT)" if timed_out else ""
    print(f"  priming cache: {wall:.1f}s, backend={backend}  [{log_path}]  (discarded){note}")


def run_injection(row, args):
    """One blocking subprocess call into run_bns_ce_pe.py; returns the result row."""
    outdir = Path(args.outdir) / f"inj_{row['index']:02d}_M{row['total_mass']:.1f}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n[{row['index'] + 1}/{args.n_injections}] M_tot={row['total_mass']:.2f} "
        f"Msun (m1={row['m1']:.2f}, m2={row['m2']:.2f}), duration={row['duration']:.0f}s, "
        f"target SNR={row['target_snr']:.1f}  ->  {outdir}"
    )

    if args.warm_cache:
        _prime_cache(row, args, outdir)

    cmd = _build_cmd(row, args, outdir)
    print("  " + " ".join(cmd))

    t0 = time.perf_counter()
    log_path = outdir / "run.log"
    returncode, timed_out = _run_with_timeout(
        cmd, log_path, args, args.injection_timeout * 60.0
    )
    wall = time.perf_counter() - t0

    result = dict(row)
    result["returncode"] = returncode
    result["timed_out"] = timed_out
    result["wall_clock_driver"] = wall
    result["backend"] = _parse_backend(log_path)
    result["log"] = str(log_path)
    for k in TIMING_KEYS:
        result[k] = None
    result["converged"] = False
    result["rhat_max"] = None
    result["ess_min"] = None
    result["achieved_snr"] = None
    result["worst_recovery_sigma"] = None
    result["parse_error"] = None

    # Best-effort: a corrupt/partial timings.json or samples.npz (observed after
    # a killed/timed-out subprocess, mid-write) must never crash the sweep --
    # one bad injection's leftovers should cost that injection's diagnostics,
    # not the rest of the grid.
    try:
        timings_path = outdir / "timings.json"
        if timings_path.exists():
            with open(timings_path) as f:
                t = json.load(f)
            for k in TIMING_KEYS:
                result[k] = t.get(k)
            result["converged"] = bool(t.get("converged", False))
            result["achieved_snr"] = t.get("network_snr")

        samples_path = outdir / "samples.npz"
        if samples_path.exists():
            npz = np.load(samples_path)
            result["rhat_max"] = float(np.max(npz["rhat"]))
            result["ess_min"] = float(np.min(npz["ess"]))
            flat = npz["samples"]
            truth = npz["truth"]
            std = flat.std(axis=0)
            med = np.median(flat, axis=0)
            result["worst_recovery_sigma"] = float(
                np.max(np.abs(med - truth) / np.where(std > 0, std, 1.0))
            )
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        result["parse_error"] = f"{type(exc).__name__}: {exc}"
        print(f"  WARNING: could not parse outputs for this injection: {result['parse_error']}")

    status = "OK" if returncode == 0 else f"FAILED (exit {returncode})"
    conv = "converged" if result["converged"] else "NOT converged"
    print(
        f"  -> {status}, {conv}, backend={result['backend']}, "
        f"{wall / 60.0:.2f} min driver wall time  [{log_path}]"
    )
    return result


def write_summary(rows, outdir):
    csv_path = Path(outdir) / "sweep_summary.csv"
    json_path = Path(outdir) / "sweep_summary.json"
    # Union of keys across all rows, not just rows[0]: a resumed sweep can mix
    # rows loaded from an older sweep_summary.json (different script version,
    # e.g. missing 'timed_out'/'parse_error') with freshly computed ones, and
    # DictWriter raises on a row with keys outside a rows[0]-only fieldnames
    # list rather than just filling the gap with restval.
    fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    return csv_path, json_path


def print_report(rows):
    n_ok = sum(1 for r in rows if r["returncode"] == 0)
    n_conv = sum(1 for r in rows if r["converged"])
    backends = sorted({r["backend"] for r in rows if r["backend"]})
    print("\n===== sweep summary =====")
    header = (
        f"{'idx':>3} {'M_tot':>7} {'SNR(tgt/ach)':>14} {'dur[s]':>8} "
        f"{'total[min]':>10} {'converged':>10} {'Rhat_max':>9} {'ESS_min':>8} "
        f"{'backend':>8}"
    )
    print(header)
    for r in rows:
        tot_min = r["total"] / 60.0 if r["total"] else float("nan")
        ach = r["achieved_snr"] if r["achieved_snr"] is not None else float("nan")
        rhat = r["rhat_max"] if r["rhat_max"] is not None else float("nan")
        ess = r["ess_min"] if r["ess_min"] is not None else float("nan")
        print(
            f"{r['index']:>3} {r['total_mass']:>7.2f} "
            f"{r['target_snr']:>6.1f}/{ach:<6.1f} {r['duration']:>8.0f} "
            f"{tot_min:>10.2f} {str(r['converged']):>10} {rhat:>9.4f} {ess:>8.0f} "
            f"{str(r['backend']):>8}"
        )
    print(f"\n{n_ok}/{len(rows)} runs exited cleanly, {n_conv}/{len(rows)} converged")
    if backends and backends != ["gpu"]:
        print(f"WARNING: mixed/non-GPU backends seen across the sweep: {backends}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--n-injections", type=int, required=True, help="N: number of injections"
    )
    ap.add_argument("--total-mass-min", type=float, default=2.8, help="Msun, BNS-like")
    ap.add_argument("--total-mass-max", type=float, default=80.0, help="Msun")
    ap.add_argument(
        "--mass-ratio", type=float, default=1.0, help="fixed q = m2/m1 <= 1 for all"
    )
    ap.add_argument("--f-min", type=float, default=10.0)
    ap.add_argument("--sampling-rate", type=float, default=4096.0)
    ap.add_argument(
        "--eta-min",
        type=float,
        default=0.2,
        help="recovery-prior lower edge on eta, forwarded to run_bns_ce_pe.py",
    )
    ap.add_argument("--target-snr", type=float, default=20.0, help="baseline network SNR")
    ap.add_argument(
        "--snr-jitter",
        type=float,
        default=0.15,
        help="fractional half-width of the per-injection SNR jitter (0 = identical)",
    )
    ap.add_argument(
        "--reference-distance",
        type=float,
        default=100.0,
        help="placeholder distance passed through before run_bns_ce_pe.py's own "
        "--target-snr rescale",
    )
    ap.add_argument(
        "--duration-safety",
        type=float,
        default=1.5,
        help="multiplicative pad on the LAL chirp+merge+ringdown time bound",
    )
    ap.add_argument("--duration-min", type=float, default=4.0, help="seconds")
    ap.add_argument("--duration-max", type=float, default=2048.0, help="seconds")
    ap.add_argument("--outdir", default="examples/output/mass_sweep_pe")
    ap.add_argument(
        "--run-script", default=str(HERE / "run_bns_ce_pe.py"), help="single-injection engine"
    )
    ap.add_argument("--max-minutes", type=float, default=20.0, help="per-injection budget")
    ap.add_argument("--rhat-target", type=float, default=1.01)
    ap.add_argument("--ess-target", type=float, default=2000.0)
    ap.add_argument("--seed", type=int, default=42, help="master seed; injection i uses seed+i")
    ap.add_argument(
        "--extra-args",
        default="",
        help="raw string appended to every run_bns_ce_pe.py call, e.g. '--f32 --flow-layers 4'",
    )
    ap.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="abort the sweep on the first non-zero exit instead of continuing",
    )
    ap.add_argument(
        "--require-gpu",
        action="store_true",
        help="set JAX_PLATFORMS=cuda,cpu for every subprocess, so a missing/invisible "
        "GPU fails immediately instead of silently falling back to CPU for the "
        "whole sweep",
    )
    ap.add_argument(
        "--injection-timeout",
        type=float,
        default=None,
        help="minutes: hard wall-clock cap per subprocess (priming and real), one "
        "retry after --cooldown-seconds before giving up on that injection. "
        "Default: 2 x --max-minutes + 15, generous enough for setup",
    )
    ap.add_argument(
        "--cooldown-seconds",
        type=float,
        default=10.0,
        help="pause before retrying a timed-out subprocess, to let the GPU driver "
        "settle (observed: back-to-back subprocesses on a small shared GPU can "
        "hang for minutes with zero output, most likely a CUDA context "
        "teardown race with the previous subprocess)",
    )
    ap.add_argument(
        "--warm-cache",
        action="store_true",
        help="run a throwaway --max-production-blocks=1 pass per injection first, "
        "to prime the persistent XLA cache so the timed run's timings.json "
        "excludes JIT compilation (see the module docstring); roughly doubles "
        "the setup+warmup+equilibration cost per injection but not production",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned grid and commands; run nothing",
    )
    ap.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="skip injections with index < this (resume a sweep); indices < "
        "start-index are pulled from an existing sweep_summary.json in --outdir "
        "if present, so the report still covers the full grid",
    )
    args = ap.parse_args()

    if not (0.0 < args.mass_ratio <= 1.0):
        raise ValueError(f"--mass-ratio must be in (0, 1], got {args.mass_ratio}")
    eta_fixed = args.mass_ratio / (1.0 + args.mass_ratio) ** 2
    if not (args.eta_min < eta_fixed <= 0.25):
        raise ValueError(
            f"--mass-ratio {args.mass_ratio} gives eta={eta_fixed:.4f}, outside the "
            f"({args.eta_min}, 0.25] recovery prior for EVERY injection in the sweep "
            "(this would fail identically on all of them) -- lower --eta-min or raise "
            "--mass-ratio"
        )
    args.run_script = Path(args.run_script)
    if not args.run_script.exists():
        raise FileNotFoundError(args.run_script)
    if args.injection_timeout is None:
        args.injection_timeout = 2.0 * args.max_minutes + 15.0

    rows = build_grid(args)

    print(f"mass sweep: {args.n_injections} injections, "
          f"M_tot in [{args.total_mass_min:.1f}, {args.total_mass_max:.1f}] Msun, "
          f"mass ratio {args.mass_ratio:.3f} (eta={eta_fixed:.4f}), "
          f"target SNR {args.target_snr:.1f} +- {100 * args.snr_jitter:.0f}%")
    print(f"worst-case wall time: {args.n_injections} x {args.max_minutes:.0f} min "
          f"= {args.n_injections * args.max_minutes:.0f} min, run strictly sequentially")
    for r in rows:
        print(
            f"  [{r['index']}] M_tot={r['total_mass']:7.2f}  m1={r['m1']:6.2f}  "
            f"m2={r['m2']:6.2f}  Mc={r['mc']:7.3f}  duration={r['duration']:7.0f}s  "
            f"target SNR={r['target_snr']:5.1f}  seed={r['seed']}"
        )

    if args.dry_run:
        print("\n--dry-run: no injections were executed")
        return 0

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    results = []
    if args.start_index > 0:
        prior_json = Path(args.outdir) / "sweep_summary.json"
        if prior_json.exists():
            with open(prior_json) as f:
                results = [r for r in json.load(f) if r["index"] < args.start_index]
            print(f"resuming from index {args.start_index}: kept {len(results)} prior result(s)")

    ran_one = False
    for row in rows:
        if row["index"] < args.start_index:
            continue
        if ran_one:
            # Let the GPU driver fully release the previous subprocess's CUDA
            # context first -- see _run_with_timeout's docstring for why.
            time.sleep(args.cooldown_seconds)
        ran_one = True
        try:
            result = run_injection(row, args)
        except Exception as exc:  # noqa: BLE001 -- one bad injection must not
            # take the rest of the sweep down with it (see run_injection's own
            # internal guard for the common case; this is the outer net for
            # anything unexpected, e.g. failing to even create the outdir).
            print(f"  UNHANDLED ERROR on injection {row['index']}: {type(exc).__name__}: {exc}")
            result = dict(row)
            result.update(
                {k: None for k in TIMING_KEYS}
                | dict(
                    returncode=-1, timed_out=False, wall_clock_driver=None,
                    backend=None, log=None, converged=False, rhat_max=None,
                    ess_min=None, achieved_snr=None, worst_recovery_sigma=None,
                    parse_error=f"{type(exc).__name__}: {exc}",
                )
            )
        results.append(result)
        write_summary(results, args.outdir)  # incremental: safe to interrupt
        if args.stop_on_failure and result["returncode"] != 0:
            print(f"--stop-on-failure: aborting after injection {row['index']}")
            break

    csv_path, json_path = write_summary(results, args.outdir)
    print_report(results)
    print(f"\nwrote {csv_path} and {json_path}")

    n_conv = sum(1 for r in results if r["converged"])
    return 0 if n_conv == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
