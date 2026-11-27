#!/usr/bin/env bash
# Run one WL111A python command in the frozen-stack container on the LabServer70 RTX 3090 (WL111's ctr_r111.sh, no CPU
# cap). usage: run_r111a.sh NAME SCRIPT.py ARGS...   (env RECON111A_OUT optional; logs go to the caller)
set -eu
name=$1; shift
R=$HOME/rest_sss101a
py=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
cd "$R/experiments/ecro_recon111a"
exec "$R/experiments/ecro_recon111a/lab70/ctr_r111a.sh" "recon111a_$name" env RECON111A_OUT="${RECON111A_OUT:-$R/results/ecro_recon111a}" \
  RECON111A_SMOKE="${RECON111A_SMOKE:-0}" RECON111A_ITERS="${RECON111A_ITERS:-20000}" "$py" "$@"
