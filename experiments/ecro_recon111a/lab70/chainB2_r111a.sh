#!/usr/bin/env bash
# WL111A Stage B chain, version 2 (owner request 2026-11-27: move to the RTX 3080 Ti when it is free; mind its 12 GB).
# Replaces chainB_r111a.sh mid-run without touching its running training containers. Per step it picks the device:
#   OUT/use_ti absent  -> RTX 3090, up to 3 concurrent trainings (6.6 GB each), no guard (as before)
#   OUT/use_ti present -> RTX 3080 Ti, ONE process at a time (two trainings would need 13.2 GB of 12 GB), each wrapped in
#                         guard_r111.py --gpu-uuid <3080 Ti> (waits for 120 s idle; yields to any other process and retries)
# Trainings still running from version 1 are waited for, never duplicated. A training that ends without its weights is
# retried once; a second failure stops the chain.
set -eu -o pipefail
R=$HOME/rest_sss101a
O=${RECON111A_OUT:-$R/results/ecro_recon111a}
export RECON111A_OUT=$O
L=$R/experiments/ecro_recon111a/lab70
G3090=GPU-ab3bc073-096c-6d0b-a304-13cef177a843
GTI=GPU-746555dc-08d4-ae98-4ba9-cc74672b34e1
py=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
st() { printf '%s %s\n' "$(date -Iseconds)" "$*" >> "$O/chainB.state"; }
mkdir -p "$O/logs"
trap 'rc=$?; (( rc )) && st "STOP exit=$rc"' EXIT
use_ti() { [[ -f $O/use_ti ]]; }
declare -A lpid ldev
busy() {   # NAME: a version-2 launch of it is alive (guard possibly still waiting), or a version-1 container of it runs
  if [[ -n ${lpid[$1]:-} ]] && kill -0 "${lpid[$1]}" 2>/dev/null; then return 0; fi
  docker ps --format '{{.Names}}' | grep -q "^recon111a_$1\$"; }
running_on() {   # DEVICE -> busy trainings on it (version-1 containers ran on the 3090)
  local c=0 n
  for spec in $SPECS; do
    IFS=: read -r arm age sub ri <<< "$spec"; n="train_${arm}_r$ri"
    busy "$n" && [[ ${ldev[$n]:-rtx3090} == "$1" ]] && c=$((c + 1))
  done
  echo $c; }
launch() {   # NAME (blocking) script args... on the device chosen now
  local name=$1; shift
  if use_ti; then
    echo "$name rtx3080ti" >> "$O/devices.txt"
    WL111A_GPU=$GTI "$py" "$R/experiments/ecro_recon111/guard_r111.py" --name "recon111a_$name" --log "$O/logs/guard_$name.jsonl" \
      --free 120 --poll 2 --container "recon111a_$name" --gpu-uuid "$GTI" -- \
      env WL111A_GPU=$GTI "$L/run_r111a.sh" "$name" "$@" > "$O/logs/$name.log" 2>&1
  else
    echo "$name rtx3090" >> "$O/devices.txt"
    WL111A_GPU=$G3090 "$L/run_r111a.sh" "$name" "$@" > "$O/logs/$name.log" 2>&1
  fi; }
st "CHAIN v2 START (device per step; use_ti=$(use_ti && echo yes || echo no))"
[[ -f $O/.doneB_guides8 ]] || { st "RUNNING guides8"; launch guides8 guides8_r111a.py; touch "$O/.doneB_guides8"; st "DONE guides8"; }
SPECS="C4:cold:4:0 C4:cold:4:1 M8:mixed:8:0 M8:mixed:8:1 C8:cold:8:0 C8:cold:8:1"
declare -A tries
st "RUNNING trainings"
while :; do
  pending=0
  for spec in $SPECS; do
    IFS=: read -r arm age sub ri <<< "$spec"; n="train_${arm}_r$ri"
    [[ -f $O/models/${arm}_r$ri.pt ]] && continue
    pending=1
    busy "$n" && continue
    t=${tries[$n]:-0}
    if (( t >= 2 )); then st "training $n failed twice"; exit 4; fi
    if use_ti; then
      (( $(running_on rtx3080ti) == 0 )) || continue     # 3080 Ti: one at a time (12 GB)
    else
      (( $(running_on rtx3090) < 3 )) || continue
    fi
    tries[$n]=$((t + 1)); st "LAUNCH $n try $((t + 1)) on $(use_ti && echo 3080Ti || echo 3090)"
    ldev[$n]=$(use_ti && echo rtx3080ti || echo rtx3090)
    launch "$n" train_r111a.py "$arm" "$age" "$sub" "$ri" &
    lpid[$n]=$!
    sleep 20
  done
  (( pending )) || break
  sleep 30
done
wait || true
st "DONE trainings"
[[ -f $O/.doneB_evalB ]] || { st "RUNNING evalB"; launch evalB evalB_r111a.py C4:4 M8:8 C8:8; touch "$O/.doneB_evalB"; st "DONE evalB"; }
[[ -f $O/.doneB_oracle ]] || { st "RUNNING oracle"; launch oracle oracle_r111a.py; touch "$O/.doneB_oracle"; st "DONE oracle"; }
st "COMPLETE"
