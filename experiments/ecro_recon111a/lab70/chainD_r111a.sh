#!/usr/bin/env bash
# WL111A Stage D: final S111A training (Stage C declaration; seeds 17950021 / 17950022; RTX 3080 Ti, one process at a time).
# RTX 3080 Ti only (owner), ONE process at a time, each under guard_r111.py --gpu-uuid <3080 Ti>; CUDA OOM retried after
# 120 s up to 3 times. Smoke first (separate OUT, 200 steps, 2 views x 4 frames); the evidence-free run follows only if the
# smoke completes. Steps: train S111A r0, r1 (FullResNet, S111 contract, declared seeds).
set -eu -o pipefail
R=$HOME/rest_sss101a
L=$R/experiments/ecro_recon111a/lab70
GTI=GPU-746555dc-08d4-ae98-4ba9-cc74672b34e1
py=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
body() {   # uses RECON111A_OUT, RECON111A_SMOKE, RECON111A_ITERS from the environment
  local O=$RECON111A_OUT
  mkdir -p "$O/logs"
  st() { printf '%s %s\n' "$(date -Iseconds)" "$*" >> "$O/chainD.state"; }
  oom() { grep -qE "OutOfMemoryError|CUDA out of memory" "$O/logs/$1.log" 2>/dev/null; }
  step() { local name=$1 k; shift
    for k in 1 2 3 4; do
      echo "$name rtx3080ti" >> "$O/devices.txt"
      if "$py" "$R/experiments/ecro_recon111/guard_r111.py" --name "recon111a_$name" --log "$O/logs/guard_$name.jsonl" \
           --free 120 --poll 2 --container "recon111a_$name" --gpu-uuid "$GTI" -- \
           env WL111A_GPU=$GTI "$L/run_r111a.sh" "$name" "$@" > "$O/logs/$name.log" 2>&1; then return 0; fi
      if oom "$name" && (( k < 4 )); then cp "$O/logs/$name.log" "$O/logs/${name}_oom$k.log"; st "OOM in $name ($k); retry in 120 s"; sleep 120
      else st "FAILED $name"; return 1; fi
    done; }
  st "START"
  for ri in 0 1; do
    [[ -f $O/models/S111A_r$ri.pt ]] || { st "RUNNING train S111A r$ri"; step "train_S111A_r$ri" train_r111a.py S111A mixed 4 "$ri" --net fullres --seed $((17950021 + ri)); st "DONE train S111A r$ri"; }
  done
  st "COMPLETE"; }
export -f body
export R L GTI py
RECON111A_OUT=$R/results/ecro_recon111a_smoke RECON111A_SMOKE=1 RECON111A_ITERS=200 bash -c body || { echo "SMOKE FAILED"; exit 6; }
RECON111A_OUT=$R/results/ecro_recon111a RECON111A_SMOKE=0 RECON111A_ITERS=20000 bash -c body
