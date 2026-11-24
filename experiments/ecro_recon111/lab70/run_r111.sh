#!/usr/bin/env bash
# WL111 chain launcher on LabServer70. usage: run_r111.sh OUT_NAME COMMIT [--smoke]
# ONE guard (experiments/ecro_estimator109b/guard109c.py, unchanged, --gpu-uuid RTX 3090) waits once for 120 s without a
# compute process on that GPU, then starts ONE container (ctr_r111.sh) that runs every stage (inner_r111.sh) back to back.
set -eu
R=$HOME/rest_sss101a
O=$R/results/ecro_recon111/$1; commit=$2; SM=${3:-}
mkdir -p "$O"
py=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
[[ $(sha256sum "$R/experiments/ecro_estimator109b/guard109c.py" | cut -c1-16) == f7d6072bbb45f925 ]]
printf '%s %s\n' "$(date -Iseconds)" "WL111 CHAIN LAUNCH commit=$commit host=$(hostname) $SM" >> "$O/chain.state"
cd "$R/experiments/ecro_recon111"
exec "$py" "$R/experiments/ecro_estimator109b/guard109c.py" --name "recon111_$1" --log "$O/guard.jsonl" --free 120 --poll 2 \
  --container "recon111_$1" --gpu-uuid GPU-ab3bc073-096c-6d0b-a304-13cef177a843 -- \
  "$R/experiments/ecro_recon111/lab70/ctr_r111.sh" "recon111_$1" bash "$R/experiments/ecro_recon111/lab70/inner_r111.sh" "$O" $SM
