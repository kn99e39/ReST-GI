#!/usr/bin/env bash
# WL109A chain on LabServer69.  usage: run109a.sh TAG COMMIT
# CPU stages (tests, Stage A reproduction, Stage B historical splits and synthetic families, null-pool analysis, Stage C
# analysis, Stage E) run with CUDA_VISIBLE_DEVICES= and never create a CUDA context. Every GPU stage runs under the unchanged
# WL109 guard (guard109.py): it waits without a GPU library until no tmux session exists and the GPU has had no compute
# process for 120 s, and kills only its own child's process group the moment a foreign compute process appears.
set -eu -o pipefail
cd "$HOME/rest_sss101a"
tag=$1; commit=$2
src=experiments/ecro_estimator109a
out="$PWD/results/ecro_estimator109a/$tag"
ev9="$PWD/results/ecro_estimator109/evidence"
py="$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python"
guardpy="$PWD/experiments/ecro_estimator109/guard109.py"
mkdir -p "$out/tmp" "$out/mpl"
export DYN101_MI_DIR="$HOME/rest_dyn101/external/mi_3.8.0" DYN101_MI_VERSION=3.8.0
export LD_LIBRARY_PATH="$PWD/results/ecro_denoise108/runtime/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 WL109A_THREADS=6
export MPLCONFIGDIR="$out/mpl" TMPDIR="$out/tmp" PYTHONDONTWRITEBYTECODE=1
guard() { local name=$1 cap=$2; shift 2; "$py" "$guardpy" --name "$name" --log "$out/guard109a.jsonl" --free 120 --poll 2 --attempt-timeout "$cap" "$@"; }
cpu() { CUDA_VISIBLE_DEVICES= nice -n 10 "$py" "$@"; }
stage=start
state() { printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "$out/chain.state"; }
trap 'rc=$?; if (( rc )); then state "STOP stage=$stage exit=$rc"; fi' EXIT
state "START commit=$commit"
cd "$src"
jget() { "$py" -c 'import json,sys; d=json.load(open(sys.argv[1]));
for k in sys.argv[2].split("."): d=d[k]
print(d)' "$1" "$2"; }

stage=tests; state "RUNNING $stage"
cpu tests109a.py > "$out/tests.log" 2>&1

stage=stageA; state "RUNNING $stage"
[[ -f $out/stageA/stageA.json ]] || cpu repro109a.py "$ev9" "$out/stageA" > "$out/stageA.log" 2>&1

# Stage B CPU analyses run in the background while the GPU stages wait for an exclusive window
stage=calib_cpu_start; state "RUNNING hist and synth in background"
mkdir -p "$out/calib"
( [[ -f $out/calib/hist.json ]] || cpu calib109a.py hist "$ev9" "$out/calib" > "$out/hist.log" 2>&1 ) &
pid_hist=$!
( [[ -f $out/calib/synth.json ]] || cpu calib109a.py synth "$out/calib" --workers 3 > "$out/synth.log" 2>&1 ) &
pid_synth=$!

stage=smoke_gpu; state "RUNNING $stage"
for m in null paired grid timing; do
  guard "smoke_$m" 3600 -- "$py" capture109a.py "$out/smoke" "$m" --smoke >> "$out/smoke.log" 2>&1
done
printf '{"cells": {}}\n' > "$out/smoke/synth_dummy.json"
cpu calib109a.py pool "$out/smoke" "$out/smoke/calib" "$out/smoke/synth_dummy.json" --smoke >> "$out/smoke.log" 2>&1
printf '{"decision": {"R": 32, "outcome": "SMOKE"}}\n' > "$out/smoke/budget_smoke.json"
cpu analyze109a.py "$out/smoke" "$out/smoke/budget_smoke.json" "$out/smoke/metrics" --smoke >> "$out/smoke.log" 2>&1
printf '{"final_static_verdict": "SMOKE"}\n' > "$out/smoke/static_smoke.json"
if [[ ! -f $out/smoke/dyn/dyn109a.json ]]; then
  guard smoke_dynamic 3600 --before "if [ -d '$out/smoke/dyn' ]; then mv '$out/smoke/dyn' '$out/smoke/dyn_preempted_'\$(date +%s); fi" -- \
    "$py" dynamic109a.py "$out/smoke/dyn" "$out/smoke/static_smoke.json" "$out/smoke/budget_smoke.json" --smoke >> "$out/smoke.log" 2>&1
fi
state "SMOKE PASS"

stage=null; state "RUNNING $stage"
guard null 21600 -- "$py" capture109a.py "$out/cap" null > "$out/null_stage.log" 2>&1
stage=calib_cpu_wait; state "WAITING hist / synth"
wait "$pid_hist"; wait "$pid_synth"
[[ -f $out/calib/hist.json && -f $out/calib/synth.json ]]
stage=pool; state "RUNNING $stage"
cpu calib109a.py pool "$out/cap" "$out/calib" "$out/calib/synth.json" > "$out/pool.log" 2>&1
outcome=$(jget "$out/calib/budget109a.json" decision.outcome)
state "BUDGET outcome=$outcome"
if [[ $outcome != PROCEED ]]; then stage=done; state "COMPLETE verdict=INCONCLUSIVE (Stage B)"; exit 0; fi
R=$(jget "$out/calib/budget109a.json" decision.R)

stage=paired; state "RUNNING $stage R=$R"
guard paired 21600 -- "$py" capture109a.py "$out/cap" paired --R "$R" > "$out/paired_stage.log" 2>&1
stage=grid; state "RUNNING $stage"
guard grid 10800 -- "$py" capture109a.py "$out/cap" grid > "$out/grid_stage.log" 2>&1
stage=timing; state "RUNNING $stage"
guard timing 10800 -- "$py" capture109a.py "$out/cap" timing --R "$R" > "$out/timing_stage.log" 2>&1

stage=analysis; state "RUNNING $stage"
cpu analyze109a.py "$out/cap" "$out/calib/budget109a.json" "$out/metrics" > "$out/analysis.log" 2>&1
cpu decide109a.py "$out/metrics/metrics109a.json" "$out/metrics/static109a.json" >> "$out/analysis.log" 2>&1
sv=$(jget "$out/metrics/static109a.json" final_static_verdict)
state "STATIC verdict=$sv"
if [[ $sv != STATIC_PASS ]]; then stage=done; state "COMPLETE static=$sv (Stage D / final report decide next)"; exit 0; fi

stage=dynamic; state "RUNNING $stage"
guard dynamic 21600 --before "if [ -d '$out/dyn' ]; then mv '$out/dyn' '$out/dyn_preempted_'\$(date +%s); fi" -- \
  "$py" dynamic109a.py "$out/dyn" "$out/metrics/static109a.json" "$out/calib/budget109a.json" > "$out/dyn_stage.log" 2>&1
dv=$(jget "$out/dyn/dyn109a.json" verdict)
stage=done; state "COMPLETE verdict=$dv"
