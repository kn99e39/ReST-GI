#!/usr/bin/env bash
# WL109 Amendment 1 (owner-approved 2026-11-22): resume the frozen chain from the development stage after the foreign GPU
# chain ends. Same code, seeds, sizes and thresholds as run109.sh; the frozen synthetic result is reused (same commit).
# Changes: (1) wait for the watched foreign chain PID(s) to exit and for 120 s of uncontended GPU before starting;
#          (2) the GPU stages' elapsed-time caps no longer bound exclusive-GPU waiting (24 h caps).
# usage: run109_amend1.sh TAG COMMIT WATCH_PID [WATCH_PID ...]
set -eu -o pipefail
cd "$HOME/rest_sss101a"
tag=$1; commit=$2; shift 2
src=experiments/ecro_estimator109
out="$PWD/results/ecro_estimator109/$tag"
py="$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python"
export DYN101_MI_DIR="$HOME/rest_dyn101/external/mi_3.8.0" DYN101_MI_VERSION=3.8.0
export LD_LIBRARY_PATH="$PWD/results/ecro_denoise108/runtime/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=4 MPLCONFIGDIR="$out/mpl" TMPDIR="$out/tmp" PYTHONDONTWRITEBYTECODE=1
stage=amend1_watch
state() { printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "$out/chain.state"; }
trap 'rc=$?; if (( rc )); then state "STOP stage=$stage exit=$rc"; fi' EXIT
state "AMENDMENT1 START commit=$commit watching=$*"
for pid in "$@"; do
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
done
state "AMENDMENT1 watched chain ended"
free=0
while (( free < 120 )); do
  if [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]') ]]; then free=$((free + 10)); else free=0; fi
  sleep 10
done
state "AMENDMENT1 GPU uncontended for 120 s"
"$py" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r["complete"] and r["pass"] and not r["smoke"] else 3)' "$out/synth/synth.json"
if [[ -d $out/dev ]]; then mv "$out/dev" "$out/dev_stopped_cap"; mv "$out/dev_stage.log" "$out/dev_stage_stopped_cap.log"; fi
cd "$src"
stage=development; state "RUNNING $stage"; timeout 86400 "$py" dev109.py "$out/dev" > "$out/dev_stage.log" 2>&1
if "$py" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r["futile"] else 1)' "$out/dev/dev.json"; then
  state "CLOSED futility (NOT_SUPPORTED)"; exit 0
fi
stage=capture;    state "RUNNING $stage"; timeout 86400 "$py" run109.py "$out/run" "$out/dev/dev.json" "$out/synth/synth.json" > "$out/run_stage.log" 2>&1
stage=analysis;   state "RUNNING $stage"; timeout 7200 "$py" analyze109.py "$out/run" "$out/dev/dev.json" "$out/synth/synth.json" "$out/metrics" > "$out/analysis.log" 2>&1
verdict=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$out/metrics/metrics109.json")
state "ANALYSIS verdict=$verdict"
if [[ $verdict == PENDING_ESCALATION ]]; then
  stage=escalation; state "RUNNING $stage"
  timeout 86400 "$py" run109.py "$out/run" "$out/dev/dev.json" "$out/synth/synth.json" --escalate > "$out/escalate_stage.log" 2>&1
  stage=analysis_escalated; state "RUNNING $stage"
  mv "$out/metrics" "$out/metrics_pre_escalation"
  timeout 7200 "$py" analyze109.py "$out/run" "$out/dev/dev.json" "$out/synth/synth.json" "$out/metrics" > "$out/analysis_escalated.log" 2>&1
  verdict=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$out/metrics/metrics109.json")
  state "ANALYSIS verdict=$verdict"
fi
if [[ $verdict == STATIC_PASS_PENDING_DYNAMIC ]]; then
  stage=dynamic; state "RUNNING $stage"
  timeout 86400 "$py" dynamic109.py "$out/dyn" "$out/run" "$out/dev/dev.json" "$out/metrics/metrics109.json" > "$out/dyn_stage.log" 2>&1
  stage=analysis_dynamic; state "RUNNING $stage"
  mv "$out/metrics" "$out/metrics_static"
  timeout 7200 "$py" analyze109.py "$out/run" "$out/dev/dev.json" "$out/synth/synth.json" "$out/metrics" --dynamic "$out/dyn/dyn.json" > "$out/analysis_dynamic.log" 2>&1
  verdict=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$out/metrics/metrics109.json")
  state "ANALYSIS verdict=$verdict"
fi
stage=done
state "COMPLETE verdict=$verdict"
