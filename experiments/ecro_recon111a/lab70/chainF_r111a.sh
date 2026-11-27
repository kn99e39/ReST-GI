#!/usr/bin/env bash
# WL111A final evaluation chain (protocol/recon111a.json, committed before this chain runs).
# RTX 3080 Ti only (owner), ONE process at a time, each step under guard_r111a.py --gpu-uuid <3080 Ti> --idle-file
# (fast start when gpumon_r111a already proves 120 s without a foreign compute process); CUDA OOM retried after 120 s up to
# 3 times. Smoke first (OUT/final_smoke: 2 new views at 2 x 64 spp, V0 + E_cfg1 at 1 / 4 spp x 2 replicates, 3 timing rounds,
# analysis); the evidence run follows only if the smoke completes.
# Evidence steps: seal (new sealed views N1-N6 + 32 x 8192-spp references) -> timing (+ Stage C cost futility: exit 3 =
# STOP before quality evidence) -> quality -> analysis.
set -eu -o pipefail
R=$HOME/rest_sss101a
L=$R/experiments/ecro_recon111a/lab70
O=$R/results/ecro_recon111a
GTI=GPU-746555dc-08d4-ae98-4ba9-cc74672b34e1
py=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
mkdir -p "$O/logs"
st() { printf '%s %s\n' "$(date -Iseconds)" "$*" >> "$O/chainF.state"; }
oom() { grep -qE "OutOfMemoryError|CUDA out of memory" "$O/logs/$1.log" 2>/dev/null; }
step() { local name=$1 k rc; shift
  for k in 1 2 3 4; do
    echo "$name rtx3080ti" >> "$O/devices.txt"
    rc=0
    "$py" "$R/experiments/ecro_recon111a/guard_r111a.py" --name "recon111a_$name" --log "$O/logs/guard_$name.jsonl" \
      --free 120 --poll 2 --container "recon111a_$name" --gpu-uuid "$GTI" --idle-file "$O/gpumon_ti.json" -- \
      env WL111A_GPU=$GTI "$L/run_r111a.sh" "$name" "$@" > "$O/logs/$name.log" 2>&1 || rc=$?
    (( rc == 0 )) && return 0
    if grep -q FUTILITY_STOP "$O/logs/$name.log"; then st "STOP cost futility ($name)"; return 3; fi
    if oom "$name" && (( k < 4 )); then cp "$O/logs/$name.log" "$O/logs/${name}_oom$k.log"; st "OOM in $name ($k); retry in 120 s"; sleep 120
    else st "FAILED $name rc=$rc"; return 1; fi
  done; }
run() { local rc=0; st "RUNNING $1"; step "$@" || rc=$?; (( rc == 0 )) && st "DONE $1"; return $rc; }
st "START"
# smoke
if [[ ! -f $O/final_smoke/analysis.json ]]; then
  run F_smoke_seal seal_r111a.py --smoke
  run F_smoke_timing timingF_r111a.py --smoke
  run F_smoke_quality evalF_r111a.py --smoke
  run F_smoke_analysis analyzeF_r111a.py "$O/final_smoke"
  st "SMOKE COMPLETE"
fi
# evidence
run F_seal seal_r111a.py
if [[ ! -f $O/final/timing/futility.json ]]; then rc=0; run F_timing timingF_r111a.py || rc=$?; (( rc == 0 || rc == 3 )) || exit $rc; fi
stop=$("$py" -c "import json; print(json.load(open('$O/final/timing/futility.json'))['stop'])")
[[ $stop == False ]] || { st "STOPPED (cost futility, futility.json stop=$stop)"; exit 0; }
run F_quality evalF_r111a.py
run F_analysis analyzeF_r111a.py "$O/final"
st "COMPLETE"
