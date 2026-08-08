#!/bin/bash
# Condor job executable for the DAG's final aggregation node: activate the jaxpe
# conda env and run the general-purpose campaign post-processor.
set -eo pipefail
# NOT `set -u`: conda's own activate/deactivate hooks reference unbound variables
# internally and break under nounset (see run_pe_job.sh).

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate jaxpe

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "${REPO_ROOT}/bin/postprocess_campaign.py" "$@"
