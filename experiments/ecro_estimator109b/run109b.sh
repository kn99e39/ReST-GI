#!/usr/bin/env bash
# WL109B chain on LabServer69.  usage: run109b.sh TAG COMMIT
# CPU stages run with CUDA_VISIBLE_DEVICES= and never create a CUDA context. Every GPU stage runs under the unchanged WL109
# guard (guard109.py): it waits without a GPU library until no tmux session exists and the GPU has had no compute process for
# 120 s, and terminates only its own child's process group the moment a foreign compute process appears.
set -eu -o pipefail
cd "$HOME/rest_sss101a"
tag=$1; commit=$2
src=experiments/ecro_estimator109b
out="$PWD/results/ecro_estimator109b/$tag"
evA="$PWD/results/ecro_estimator109a/evidence"
ev9="$PWD/results/ecro_estimator109/evidence"
evalA="$PWD/evaluation/109A"
py="$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python"
guardpy="$PWD/experiments/ecro_estimator109/guard109.py"
mkdir -p "$out/tmp" "$out/mpl"
export DYN101_MI_DIR="$HOME/rest_dyn101/external/mi_3.8.0" DYN101_MI_VERSION=3.8.0
export LD_LIBRARY_PATH="$PWD/results/ecro_denoise108/runtime/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="$out/mpl" TMPDIR="$out/tmp" PYTHONDONTWRITEBYTECODE=1
guard() { local name=$1 cap=$2; shift 2; "$py" "$guardpy" --name "$name" --log "$out/guard109b.jsonl" --free 120 --poll 2 --attempt-timeout "$cap" "$@"; }
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
cpu tests109b.py > "$out/tests.log" 2>&1

# Stage A (WL109A reproduction) and Stage C synthetic stage 1 run on the CPU while the GPU stages wait for a window
stage=cpu_background; state "RUNNING stageA and synth stage 1 in background"
mkdir -p "$out/calib"
( [[ -f $out/stageA/stageA109b.json ]] || cpu repro109b.py "$evA" "$evalA" "$ev9" "$out/stageA" > "$out/stageA.log" 2>&1 ) &
pid_a=$!
( [[ -f $out/calib/synth_stage1.json ]] || cpu calib109b.py synth "$out/calib" --stage 1 --workers 8 > "$out/synth1.log" 2>&1 ) &
pid_s=$!

stage=smoke_gpu; state "RUNNING $stage"
for m in null paired grid timing; do
  guard "smoke_$m" 3600 -- "$py" capture109b.py "$out/smoke" "$m" --smoke >> "$out/smoke.log" 2>&1
done
cpu calib109b.py synth "$out/smoke/calib" --smoke --workers 4 >> "$out/smoke.log" 2>&1
cpu calib109b.py pool "$out/smoke" "$out/smoke/calib" --smoke --workers 2 >> "$out/smoke.log" 2>&1
cpu calib109b.py decide "$out/smoke/calib" --smoke >> "$out/smoke.log" 2>&1
cpu analyze109b.py "$out/smoke" "$out/smoke/calib/budget109b.json" "$out/smoke/metrics" --smoke --phase bias >> "$out/smoke.log" 2>&1
cpu analyze109b.py "$out/smoke" "$out/smoke/calib/budget109b.json" "$out/smoke/metrics" --smoke --phase static >> "$out/smoke.log" 2>&1
printf '{"final_static_verdict": "SMOKE"}\n' > "$out/smoke/static_smoke.json"
if [[ ! -f $out/smoke/dyn/dyn109b.json ]]; then
  guard smoke_dynamic 3600 --before "if [ -d '$out/smoke/dyn' ]; then mv '$out/smoke/dyn' '$out/smoke/dyn_preempted_'\$(date +%s); fi" -- \
    "$py" dynamic109b.py "$out/smoke/dyn" "$out/smoke/static_smoke.json" "$out/smoke/calib/budget109b.json" --smoke >> "$out/smoke.log" 2>&1
fi
state "SMOKE PASS"

stage=null; state "RUNNING $stage"
guard null 28800 -- "$py" capture109b.py "$out/cap" null > "$out/null_stage.log" 2>&1

stage=stageA_wait; state "WAITING stageA"
wait "$pid_a"
[[ $(jget "$out/stageA/stageA109b.json" pass) == True ]] || { state "STOP Stage A reproduction failed"; exit 3; }
state "STAGE A PASS"
stage=synth_wait; state "WAITING synth stage 1"
wait "$pid_s"
stage=pool; state "RUNNING $stage"
[[ -f $out/calib/pool109b.json ]] || cpu calib109b.py pool "$out/cap" "$out/calib" --workers 6 > "$out/pool.log" 2>&1
stage=decide; state "RUNNING $stage"
cpu calib109b.py decide "$out/calib" > "$out/decide.log" 2>&1
outcome=$(jget "$out/calib/budget109b.json" decision.outcome)
if [[ $outcome == ESCALATE ]]; then
  state "ESCALATE synthetic stage 2"
  stage=synth2; cpu calib109b.py synth "$out/calib" --stage 2 --workers 12 > "$out/synth2.log" 2>&1
  stage=decide2; cpu calib109b.py decide "$out/calib" >> "$out/decide.log" 2>&1
  outcome=$(jget "$out/calib/budget109b.json" decision.outcome)
fi
state "BUDGET outcome=$outcome"
if [[ $outcome != PROCEED ]]; then stage=done; state "COMPLETE verdict=INCONCLUSIVE (Stage C qualification)"; exit 0; fi
R=$(jget "$out/calib/budget109b.json" decision.R)

stage=paired; state "RUNNING $stage R=$R"
guard paired 28800 -- "$py" capture109b.py "$out/cap" paired --R "$R" > "$out/paired_stage.log" 2>&1
stage=bias; state "RUNNING $stage"
cpu analyze109b.py "$out/cap" "$out/calib/budget109b.json" "$out/metrics" --phase bias > "$out/bias.log" 2>&1
bv=$(jget "$out/metrics/bias109b.json" bias_verdict)
state "BIAS verdict=$bv"
if [[ $bv != EQUIVALENT_ALL ]]; then stage=done; state "COMPLETE bias=$bv (Stage E / final report decide next)"; exit 0; fi

stage=grid; state "RUNNING $stage"
guard grid 10800 -- "$py" capture109b.py "$out/cap" grid > "$out/grid_stage.log" 2>&1
stage=timing; state "RUNNING $stage"
guard timing 10800 -- "$py" capture109b.py "$out/cap" timing --R "$R" > "$out/timing_stage.log" 2>&1
stage=static; state "RUNNING $stage"
cpu analyze109b.py "$out/cap" "$out/calib/budget109b.json" "$out/metrics" --phase static > "$out/static.log" 2>&1
cpu decide109b.py "$out/metrics/metrics109b.json" "$out/metrics/static109b.json" >> "$out/static.log" 2>&1
sv=$(jget "$out/metrics/static109b.json" final_static_verdict)
state "STATIC verdict=$sv"
if [[ $sv != STATIC_PASS ]]; then stage=done; state "COMPLETE static=$sv"; exit 0; fi

stage=dynamic; state "RUNNING $stage"
guard dynamic 28800 --before "if [ -d '$out/dyn' ]; then mv '$out/dyn' '$out/dyn_preempted_'\$(date +%s); fi" -- \
  "$py" dynamic109b.py "$out/dyn" "$out/metrics/static109b.json" "$out/calib/budget109b.json" > "$out/dyn_stage.log" 2>&1
dv=$(jget "$out/dyn/dyn109b.json" verdict)
stage=done; state "COMPLETE verdict=$dv"
