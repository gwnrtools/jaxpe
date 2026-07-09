#!/bin/bash
while true; do
  conda run -n lalsuite-dev python bin/run_phenomd_events.py
  EXIT_CODE=$?
  if [ $EXIT_CODE -eq 0 ]; then
    echo "All done!"
    break
  else
    echo "Crashed with $EXIT_CODE, restarting in 5s..."
    sleep 5
  fi
done
