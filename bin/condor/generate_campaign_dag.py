#!/usr/bin/env python3
"""Generate the HTCondor submit files + DAG for the 300-run PE method-comparison
campaign (PhenomD+HMC, PhenomD+Gpry, ESIGMA+Gpry(marginalized_intrinsic)), all on
the same shared set of injections.

Injection generation is *not* a DAG node -- it is fast (<0.1% of PE wall time) and
deterministic, so run it once yourself first:

    jaxpe generate-injections --config <campaign config> --n-injections 100 \\
        --network H1,L1 --noise zero --psd aligo --outdir <injections dir>

Then generate the DAG:

    python bin/condor/generate_campaign_dag.py \\
        --config examples/configs/campaign_10to80_aligned.json \\
        --injections-dir campaign/injections --results-dir campaign/results \\
        --n-injections 100 --condor-dir campaign/condor

This writes three submit-file templates (one per PE variant, each job = 1 PE run),
a `smoke.dag` with just injection 0 of each variant (for the required single-job
debug pass), and the full `campaign.dag` with all 300 PE jobs -- mutually
independent (no PARENT/CHILD among themselves, since each only reads a
pre-generated injection file) -- plus one final `POSTPROCESS` node in each DAG that
runs `bin/postprocess_campaign.py` (corner-plot gallery, PP plots, HTML report)
as a child of every PE job in that DAG, so it only runs once every PE run has
finished (or failed).
"""

import argparse
from pathlib import Path

# All three variants request the same core count. Sized from the cluster's real
# inventory and JAX's real (not requested) behavior, both measured directly on
# this pool during the smoke test:
#   - condor_status shows ~25 nodes x 32 cores each; all but one or two are
#     static whole-node slots, where a job gets the *entire* node regardless of
#     request_cpus (so request_cpus mostly matters for the couple of
#     partitionable nodes, and for honest HTCondor accounting).
#   - JAX/XLA's CPU thread pool is sized from the process's CPU affinity mask
#     (sched_getaffinity), NOT from OMP_NUM_THREADS/MKL_NUM_THREADS/XLA_FLAGS --
#     those left every job spawning ~106 threads regardless of what was
#     requested. `taskset -c 0-N` (applied in run_pe_job.sh) is what actually
#     works: measured 106 -> 41 threads at an 8-core affinity mask. A 4-core
#     affinity mask made even a tiny compile+matmul smoke test time out --
#     compilation itself appears to want more headroom -- so 8 is the floor
#     that was actually tested to work, not just a round number.
PE_REQUEST_CPUS = 8

VARIANTS = {
    "phenomd_hmc": {
        "flags": "--sampler hmc --domain fd --likelihood full",
        "request_cpus": PE_REQUEST_CPUS,
        # Measured ~9.8GB resident on the smoke-test run (128 chains, 40s@2048Hz
        # data) -- above an original 8GB guess; sized here with real headroom.
        "request_memory": "16GB",
    },
    "phenomd_gpry": {
        # smoke-test injection 0 (PhenomD SNR ~73, near the population's high end --
        # see plan.md amendment 13) crashed NORA with GPAcquisitionError under GPry's
        # defaults ("training points are very close to each other"). ref_bounds
        # tightening/widening, BatchOptimizer, and n_initial up to 512 all failed
        # identically (amendment 15) -- not a sampling-density problem. The actual
        # fix (amendment 16), following the GPry LISA paper's own high-SNR SMBHB
        # settings: raise the GP regressor's noise floor 1e-2 -> 1.0 (the default
        # underfits the GP once the log-likelihood's dynamic range gets large) and
        # loosen both infinities classifiers to "20s". Verified on this exact
        # injection at its full SNR ~73 -- converged (174 truth evals) with a
        # correctly-recovered posterior (credible levels 0.44-0.54, near-median, on
        # all 4 parameters) where every prior attempt crashed. Applied to all 100
        # phenomd_gpry jobs, not just this one, since SNR varies population-wide
        # (16.99-124.27) and nothing about the fix is injection-0-specific.
        "flags": (
            "--sampler gpry --domain fd --likelihood marginalized_phase_distance "
            "--gpry-noise-level 1.0 --gpry-svm-threshold 20s "
            "--gpry-trust-region-threshold 20s"
        ),
        "request_cpus": PE_REQUEST_CPUS,
        "request_memory": "4GB",
    },
    "esigma_gpry": {
        # marginalized_intrinsic's adaptive-IS extrinsic marginal measured ~20 s per
        # L(theta_int) call at this campaign's data length (benchmarked directly,
        # not guessed) -- the config's gpry.max_total=2000 (sized for the cheap
        # closed-form marginalized_phase_distance likelihood PhenomD+Gpry uses) would
        # be ~11 h worst case here, so cap this variant's budget on the command line.
        # --gpry-n-initial raised from GPry's own default (3*n_dim=12) to 24: the
        # smoke-test run logged "Some of the initial training points are very close
        # to each other. This may lead to numerical instability in the GP" with 12,
        # and its spin1z posterior came back implausibly broad/offset (median sign
        # flipped from the injected truth) -- a real GP-conditioning problem, not
        # just slow convergence. 24 initial points is still cheap at ~20s/call
        # (+~8 min) against the 300-call budget.
        "flags": (
            "--sampler gpry --waveform esigma --likelihood marginalized_intrinsic "
            "--gpry-n-initial 24 --gpry-max-total 300 --gpry-max-initial 100"
        ),
        "request_cpus": PE_REQUEST_CPUS,
        "request_memory": "4GB",
    },
}

SUBMIT_TEMPLATE = """universe = vanilla
executable = {executable}
arguments = "{ncpus} {inj_placeholder} {outdir_placeholder} --config {config} {flags}"
accounting_group = {accounting_group}
getenv = True
request_cpus = {request_cpus}
request_memory = {request_memory}
request_disk = {request_disk}
should_transfer_files = IF_NEEDED
when_to_transfer_output = ON_EXIT
log = {logs_dir}/pe_$(inj_id).log
output = {logs_dir}/pe_$(inj_id).out
error = {logs_dir}/pe_$(inj_id).err
queue
"""

POSTPROCESS_SUBMIT_TEMPLATE = """universe = vanilla
executable = {executable}
arguments = "--results-dir {results_dir} --outdir {report_dir}"
accounting_group = {accounting_group}
getenv = True
request_cpus = {request_cpus}
request_memory = {request_memory}
request_disk = {request_disk}
should_transfer_files = IF_NEEDED
when_to_transfer_output = ON_EXIT
log = {logs_dir}/postprocess.log
output = {logs_dir}/postprocess.out
error = {logs_dir}/postprocess.err
queue
"""


def write_submit_files(args, condor_dir: Path, repo_root: Path, results_dir: Path) -> dict:
    executable = repo_root / "bin" / "condor" / "run_pe_job.sh"
    sub_paths = {}
    for variant, spec in VARIANTS.items():
        logs_dir = results_dir / variant / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        sub_text = SUBMIT_TEMPLATE.format(
            executable=executable,
            ncpus=spec["request_cpus"],
            inj_placeholder="$(inj)",
            outdir_placeholder="$(outdir)",
            config=Path(args.config).resolve(),
            flags=spec["flags"],
            accounting_group=args.accounting_group,
            request_cpus=spec["request_cpus"],
            request_memory=spec["request_memory"],
            request_disk=args.request_disk,
            logs_dir=logs_dir,
        )
        sub_path = condor_dir / f"{variant}.sub"
        sub_path.write_text(sub_text)
        sub_paths[variant] = sub_path
    return sub_paths


def write_postprocess_submit(args, condor_dir: Path, repo_root: Path, results_dir: Path, report_dir: Path) -> Path:
    executable = repo_root / "bin" / "condor" / "postprocess_job.sh"
    logs_dir = condor_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    sub_text = POSTPROCESS_SUBMIT_TEMPLATE.format(
        executable=executable,
        results_dir=results_dir,
        report_dir=report_dir,
        accounting_group=args.accounting_group,
        request_cpus=2,
        request_memory="4GB",
        request_disk=args.request_disk,
        logs_dir=logs_dir,
    )
    sub_path = condor_dir / "postprocess.sub"
    sub_path.write_text(sub_text)
    return sub_path


def dag_lines(sub_paths: dict, injections_dir: Path, results_dir: Path, indices):
    """Returns (lines, job_names): job_names is every PE job's DAG node name, so
    the caller can make them all PARENT of the final POSTPROCESS node."""
    lines, job_names = [], []
    for variant, sub_path in sub_paths.items():
        for i in indices:
            job = f"{variant}_{i}"
            job_names.append(job)
            inj = injections_dir / f"inj_{i}.json"
            outdir = results_dir / variant / str(i)
            lines.append(f"JOB {job} {sub_path}")
            lines.append(
                f'VARS {job} inj_id="{i}" inj="{inj}" outdir="{outdir}"'
            )
            lines.append(f"RETRY {job} 1")
    return lines, job_names


def _write_dagman_config(path: Path, maxjobs: int) -> Path:
    """Each DAG gets its own DAGMAN_MAX_JOBS_SUBMITTED config file, named after the DAG
    itself (not a single shared 'dagman.config') -- required once multiple DAGs in the same
    --condor-dir are meant to run concurrently with independently-tuned throttles (see
    write_variant_dag)."""
    config_path = path.with_suffix(".dagman.config")
    config_path.write_text(f"DAGMAN_MAX_JOBS_SUBMITTED = {maxjobs}\n")
    return config_path


def write_dag(
    path: Path, sub_paths, injections_dir, results_dir, indices, maxjobs, postprocess_sub: Path
):
    lines = [f"# Auto-generated by {Path(__file__).name}; do not edit by hand."]
    if maxjobs:
        lines.append(f"CONFIG {_write_dagman_config(path, maxjobs)}")
    pe_lines, job_names = dag_lines(sub_paths, injections_dir, results_dir, indices)
    lines += pe_lines
    # One aggregation job at the end: corner-plot gallery + PP plots + HTML report,
    # over every PE run in this DAG. It always runs once all its parents reach a
    # terminal state (succeeded or exhausted retries) -- postprocess_campaign.py
    # itself is tolerant of missing/failed runs, so a partially-completed campaign
    # still gets a report of whatever did finish.
    lines.append(f"JOB POSTPROCESS {postprocess_sub}")
    lines.append("PARENT " + " ".join(job_names) + " CHILD POSTPROCESS")
    path.write_text("\n".join(lines) + "\n")


def write_variant_dag(path: Path, variant: str, sub_path: Path, injections_dir, results_dir, indices, maxjobs):
    """A single-variant DAG with no POSTPROCESS node -- lets one track (e.g. phenomd_gpry)
    be submitted and run to completion independently of the other two, instead of every
    track being bundled into one campaign.dag where a single slow track (phenomd_hmc, the
    one PE variant that isn't self-terminating via GPry's own convergence check) blocks
    submission of the other two entirely. Postprocessing stays a separate, manually-invoked
    step (bin/postprocess_campaign.py already auto-discovers whatever variants/runs exist
    under --results-dir, partial or complete) rather than being wired per-track here, since
    the point of splitting is exactly to decouple each track's completion time from the
    others' -- a per-track POSTPROCESS node would still need the other tracks' results to
    produce the combined report this campaign wants."""
    lines = [f"# Auto-generated by {Path(__file__).name}; do not edit by hand."]
    if maxjobs:
        lines.append(f"CONFIG {_write_dagman_config(path, maxjobs)}")
    pe_lines, _ = dag_lines({variant: sub_path}, injections_dir, results_dir, indices)
    lines += pe_lines
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Campaign run-configuration JSON")
    p.add_argument("--injections-dir", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--condor-dir", required=True)
    p.add_argument(
        "--report-dir",
        default=None,
        help="Where the final POSTPROCESS job writes its HTML report "
        "(default: <results-dir>/../report)",
    )
    p.add_argument("--n-injections", type=int, default=100)
    p.add_argument(
        "--accounting-group", default="ligo.dev.o4.cbc.pe.jaxpe"
    )
    p.add_argument("--request-disk", default="2GB")
    p.add_argument(
        "--maxjobs",
        type=int,
        default=0,
        help="DAGMAN_MAX_JOBS_SUBMITTED throttle for campaign.dag (all 3 tracks bundled). "
        "Default 0: no self-imposed cap -- the condor negotiator already does fair-share "
        "scheduling across users on this shared pool, so an additional DAGMan-side throttle "
        "just leaves nodes idle rather than protecting anything. Set explicitly (e.g. to "
        "leave headroom for other users during a specific run) if ever needed.",
    )
    p.add_argument(
        "--track-maxjobs",
        type=int,
        default=0,
        help="DAGMAN_MAX_JOBS_SUBMITTED throttle for each of the 3 per-track "
        "campaign_<variant>.dag files. Default 0: no cap, same reasoning as --maxjobs -- "
        "let the condor scheduler's own fair-share allocation decide how many of this "
        "pool's nodes each track actually gets, rather than pre-guessing a split.",
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    injections_dir = Path(args.injections_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    condor_dir = Path(args.condor_dir).resolve()
    condor_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir).resolve() if args.report_dir else results_dir.parent / "report"

    sub_paths = write_submit_files(args, condor_dir, repo_root, results_dir)
    postprocess_sub = write_postprocess_submit(args, condor_dir, repo_root, results_dir, report_dir)

    smoke_path = condor_dir / "smoke.dag"
    write_dag(smoke_path, sub_paths, injections_dir, results_dir, [0], maxjobs=0, postprocess_sub=postprocess_sub)
    print(f"Wrote {smoke_path} (3 PE jobs + 1 POSTPROCESS job)")

    full_path = condor_dir / "campaign.dag"
    write_dag(
        full_path,
        sub_paths,
        injections_dir,
        results_dir,
        range(args.n_injections),
        maxjobs=args.maxjobs,
        postprocess_sub=postprocess_sub,
    )
    print(f"Wrote {full_path} ({3 * args.n_injections} PE jobs + 1 POSTPROCESS job)")

    track_paths = {}
    for variant, sub_path in sub_paths.items():
        track_path = condor_dir / f"campaign_{variant}.dag"
        write_variant_dag(
            track_path, variant, sub_path, injections_dir, results_dir,
            range(args.n_injections), maxjobs=args.track_maxjobs,
        )
        track_paths[variant] = track_path
        print(f"Wrote {track_path} ({args.n_injections} PE jobs, no POSTPROCESS -- run "
              f"bin/postprocess_campaign.py manually once you want a report)")

    print("\nSubmit files:")
    for variant, sub_path in sub_paths.items():
        print(f"  {variant}: {sub_path}")
    print(f"  postprocess: {postprocess_sub}  (report -> {report_dir})")
    print(f"\nSmoke test:  condor_submit_dag {smoke_path}")
    print(f"Full campaign, all 3 tracks bundled (after smoke test succeeds): "
          f"condor_submit_dag {full_path}")
    print("Full campaign, one track at a time (e.g. to start the fast GPry tracks without "
          "waiting on the slow HMC track):")
    for variant, track_path in track_paths.items():
        print(f"  condor_submit_dag {track_path}")


if __name__ == "__main__":
    main()
