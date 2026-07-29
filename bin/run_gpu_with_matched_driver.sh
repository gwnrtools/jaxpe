#!/bin/bash
# Run the BNS/CE PE benchmark on the GPU with a version-matched NVIDIA userspace.
#
# This machine's NVIDIA packages were upgraded to 580.173.02 while the loaded
# kernel module is still 580.159.03 (reboot pending), so CUDA cannot initialize
# against the system libraries. As a no-root, reversible workaround, the exact
# 580.159.03 userspace (libcuda, libnvidia-ml, nvvm, ptxjitcompiler) was
# downloaded from the official Ubuntu Launchpad archive
#   https://launchpad.net/ubuntu/+archive/primary/+files/libnvidia-compute-580_580.159.03-0ubuntu0.22.04.1_amd64.deb
# and extracted (dpkg-deb -x) into the directory below; LD_LIBRARY_PATH points
# this process (and only this process) at it, restoring an exact userspace <->
# kernel-module version match. After the next reboot this script is unnecessary:
# run bin/run_bns_ce_pe.py directly.
set -euo pipefail

NVLIBS="/tmp/claude-1000/-home-prayush-src-jaxpe/21fae678-1c47-41e1-a2d6-952a65f6933e/scratchpad/nvidia159/usr/lib/x86_64-linux-gnu"

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate lalsuite-dev

export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec python -u bin/run_bns_ce_pe.py "$@"
