#!/bin/bash

# We iterate from PN order 8 (4PN) down to 0 (Newtonian).
# JAXPE translates rad_pn_order=8 to 4PN, etc.
# esigmapy supports integer values for PN order (e.g., 8, 7, 6...)
for PN in 8 7 6 5 4 3 2 1 0; do
    echo "=========================================================="
    echo "Running PN=$PN"
    echo "=========================================================="

    time XLA_FLAGS="--xla_cpu_parallel_codegen_split_count=1" MALLOC_ARENA_MAX=1 JAX_PLATFORMS=cpu \
    conda run -n lalsuite-dev python examples/05_esigma_injection.py \
    --n-chains 20 --n-epochs 10 --n-production 100 --pn-order $PN

    # Check if the python script exited successfully
    if [ $? -eq 0 ]; then
        echo "=========================================================="
        echo "SUCCESS! The script successfully compiled and ran at PN=$PN."
        echo "=========================================================="
        break
    else
        echo "FAILED at PN=$PN. Trying the next lower PN order..."
    fi
done
