#!/bin/bash
# Condor job executable: activate the jaxpe conda env, run PE on one injection, then
# post-process its samples. Every campaign submit file points its `arguments` at this
# script; the sampler/likelihood/waveform choice is carried entirely in the extra
# jaxpe run-pe flags appended after --outdir (see generate_campaign_dag.py).
set -eo pipefail
# NOT `set -u`: conda's own activate/deactivate hooks reference unbound variables
# internally (e.g. CONDA_BACKUP_LALSIMULATION_DATADIR) and break under nounset.

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate jaxpe

# Cap actual CPU usage to NCPUS via taskset. This matters because JAX/XLA's CPU
# thread pool is sized from the process's CPU affinity mask (sched_getaffinity),
# NOT from OMP_NUM_THREADS/MKL_NUM_THREADS/XLA_FLAGS -- measured directly: those
# env vars left the process spawning ~106 threads and using several cores
# regardless of request_cpus, while `taskset -c 0-N` measurably shrank the XLA
# thread pool (106 -> 41 threads at an 8-core affinity mask). Most nodes in this
# pool are static whole-node slots (a job gets the whole node regardless of
# request_cpus, so this doesn't cost parallelism there), but the pool also has a
# few partitionable nodes where several jobs can land on one physical machine --
# there, an uncapped job silently using more cores than it requested is a real
# contention risk against its node-mates.
NCPUS="$1"
INJ="$2"
OUTDIR="$3"
shift 3

taskset -c "0-$((NCPUS - 1))" jaxpe run-pe --injection "$INJ" --outdir "$OUTDIR" "$@"
taskset -c "0-$((NCPUS - 1))" jaxpe process-samples "$OUTDIR/raw_samples.npz"
