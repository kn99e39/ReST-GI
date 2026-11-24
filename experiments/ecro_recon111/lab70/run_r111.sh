#!/usr/bin/env bash
# WL111 chain launcher on LabServer70. usage: run_r111.sh OUT_NAME COMMIT [--smoke]
# ONE guard (guard_r111.py = guard109c + WL111 execution fix 1, --gpu-uuid RTX 3090) waits once for 120 s without a
# compute process on that GPU, then starts ONE container (ctr_r111.sh) that runs every stage (inner_r111.sh) back to back.
set -eu
R=$HOME/rest_sss101a
O=$R/results/ecro_recon111/$1; commit=$2; SM=${3:-}
mkdir -p "$O"
py=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
[[ -f "$R/experiments/ecro_recon111/guard_r111.py" ]]
printf '%s %s\n' "$(date -Iseconds)" "WL111 CHAIN LAUNCH commit=$commit host=$(hostname) $SM" >> "$O/chain.state"
cd "$R/experiments/ecro_recon111"
exec "$py" "$R/experiments/ecro_recon111/guard_r111.py" --name "recon111_$1" --log "$O/guard.jsonl" --free 120 --poll 2 \
  --container "recon111_$1" --gpu-uuid GPU-ab3bc073-096c-6d0b-a304-13cef177a843 -- \
  "$R/experiments/ecro_recon111/lab70/ctr_r111.sh" "recon111_$1" bash "$R/experiments/ecro_recon111/lab70/inner_r111.sh" "$O" $SM
