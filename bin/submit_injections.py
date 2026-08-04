#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path


from jaxpe.gw.injections import parse_sim_inspiral_table


def generate_condor_submit(job_name, cmd_args, log_dir):
    sub = f"""universe = vanilla
executable = {cmd_args[0]}
arguments = {" ".join(cmd_args[1:])}
output = {log_dir}/{job_name}.out
error = {log_dir}/{job_name}.err
log = {log_dir}/{job_name}.log
request_cpus = 16
request_memory = 32GB
getenv = True
queue 1
"""
    return sub


def generate_slurm_submit(job_name, cmd_args, log_dir):
    sbatch = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}.out
#SBATCH --error={log_dir}/{job_name}.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00

{" ".join(cmd_args)}
"""
    return sbatch


def main():
    parser = argparse.ArgumentParser(
        description="Submit PE runs for injections from an XML file."
    )
    parser.add_argument(
        "--xml-file",
        type=str,
        required=True,
        help="LIGO_LW XML file containing sim_inspiral table",
    )
    parser.add_argument(
        "--prior-json", type=str, required=True, help="JSON file with prior bounds"
    )
    parser.add_argument("--psd-file", type=str, default=None, help="Path to PSD file")
    parser.add_argument(
        "--outdir", type=str, default="runs", help="Base output directory"
    )
    parser.add_argument(
        "--scheduler",
        choices=["local", "slurm", "condor"],
        default="local",
        help="How to dispatch jobs",
    )
    parser.add_argument(
        "--run-script",
        type=str,
        default="bin/run_pe.py",
        help="Path to run_pe.py script",
    )

    args = parser.parse_args()

    base_outdir = Path(args.outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)

    log_dir = base_outdir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Read prior file just to validate it exists and copy it
    with open(args.prior_json, "r") as f:
        prior_data = json.load(f)

    injections = parse_sim_inspiral_table(args.xml_file)
    print(f"Found {len(injections)} injections in {args.xml_file}")

    # Get absolute path to the run script so scheduler doesn't get confused
    run_script = Path(args.run_script).resolve()

    for i, inj in enumerate(injections):
        job_name = f"inj_{i}"
        job_outdir = base_outdir / job_name
        job_outdir.mkdir(parents=True, exist_ok=True)

        # We need to map sim_inspiral column names to jaxpe kwargs (chirp_mass, mass_ratio, etc.)
        # Typical sim_inspiral columns: mchirp, q (or mass1, mass2), distance, inclination, coa_phase, etc.
        # This mapping assumes standard ligolw naming conventions, though we can just map what we can.
        # A full mapping would depend on exact XML output. For now, we will map standard fields and pass the rest.
        mapped_inj = {}
        if "mchirp" in inj:
            mapped_inj["chirp_mass"] = inj["mchirp"]
        elif "mass1" in inj and "mass2" in inj:
            m1, m2 = max(inj["mass1"], inj["mass2"]), min(inj["mass1"], inj["mass2"])
            mc = ((m1 * m2) ** 0.6) / ((m1 + m2) ** 0.2)
            mapped_inj["chirp_mass"] = mc
            mapped_inj["mass_ratio"] = m2 / m1

        if "distance" in inj:
            mapped_inj["luminosity_distance"] = inj["distance"]
        if "inclination" in inj:
            mapped_inj["inclination"] = inj["inclination"]
        if "coa_phase" in inj:
            mapped_inj["phase"] = inj["coa_phase"]
        if "geocent_end_time" in inj:
            mapped_inj["geocent_time"] = inj["geocent_end_time"]
            if "geocent_end_time_ns" in inj:
                mapped_inj["geocent_time"] += inj["geocent_end_time_ns"] * 1e-9

        # Eccentricity is sometimes not in sim_inspiral standard, but might be added by custom pipelines
        if "eccentricity" in inj:
            mapped_inj["eccentricity"] = inj["eccentricity"]
        else:
            mapped_inj["eccentricity"] = 0.0  # Default fallback

        if "spin1z" in inj:
            mapped_inj["spin1z"] = inj["spin1z"]
        if "spin2z" in inj:
            mapped_inj["spin2z"] = inj["spin2z"]
        if "mean_anomaly" in inj:
            mapped_inj["mean_anomaly"] = inj["mean_anomaly"]
        else:
            mapped_inj["mean_anomaly"] = 0.0

        inj_json_path = job_outdir / "injection.json"
        with open(inj_json_path, "w") as f:
            json.dump(mapped_inj, f, indent=4)

        prior_json_path = job_outdir / "prior.json"
        with open(prior_json_path, "w") as f:
            json.dump(prior_data, f, indent=4)

        cmd_args = [
            "python",
            str(run_script),
            "--injection-json",
            str(inj_json_path),
            "--prior-json",
            str(prior_json_path),
            "--outdir",
            str(job_outdir),
        ]
        if args.psd_file:
            cmd_args.extend(["--psd-file", str(Path(args.psd_file).resolve())])

        print(f"[{job_name}] Submitting...")

        if args.scheduler == "local":
            # For local, we could use subprocess.Popen if we want concurrent,
            # or just os.system for sequential. Let's run concurrently in background.
            env = os.environ.copy()
            env["XLA_FLAGS"] = (
                "--xla_force_host_platform_device_count=16"  # Enable multicore
            )
            with (
                open(log_dir / f"{job_name}.out", "w") as fout,
                open(log_dir / f"{job_name}.err", "w") as ferr,
            ):
                subprocess.Popen(cmd_args, stdout=fout, stderr=ferr, env=env)
            print(f"  Launched local subprocess. Logs in {log_dir}")

        elif args.scheduler == "slurm":
            sbatch_str = generate_slurm_submit(job_name, cmd_args, log_dir)
            sbatch_path = job_outdir / "submit.sh"
            with open(sbatch_path, "w") as f:
                f.write(sbatch_str)
            subprocess.run(["sbatch", str(sbatch_path)])
            print(f"  Submitted sbatch {sbatch_path}")

        elif args.scheduler == "condor":
            condor_str = generate_condor_submit(job_name, cmd_args, log_dir)
            condor_path = job_outdir / "submit.sub"
            with open(condor_path, "w") as f:
                f.write(condor_str)
            subprocess.run(["condor_submit", str(condor_path)])
            print(f"  Submitted condor_submit {condor_path}")


if __name__ == "__main__":
    main()
