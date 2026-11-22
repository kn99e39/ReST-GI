#!/usr/bin/env bash
# WL109 frozen chain on LabServer69.  usage: run109.sh TAG COMMIT [--smoke]
# Stages: tests -> synthetic contract -> development (M*, futility) -> evaluation capture -> analysis
#         [-> predeclared bias escalation -> analysis] [-> dynamic (static pass only) -> analysis]
set -eu -o pipefail
cd "$HOME/rest_sss101a"
tag=$1; commit=$2; smoke=${3:-}
src=experiments/ecro_estimator109
out="$PWD/results/ecro_estimator109/$tag"
py="$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python"
mkdir -p "$out" "$out/tmp" "$out/mpl"
export DYN101_MI_DIR="$HOME/rest_dyn101/external/mi_3.8.0" DYN101_MI_VERSION=3.8.0
export LD_LIBRARY_PATH="$PWD/results/ecro_denoise108/runtime/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=4 MPLCONFIGDIR="$out/mpl" TMPDIR="$out/tmp" PYTHONDONTWRITEBYTECODE=1
stage=start
state() { printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "$out/chain.state"; }
trap 'rc=$?; if (( rc )); then state "STOP stage=$stage exit=$rc"; fi' EXIT
state "START commit=$commit smoke=${smoke:-no}"
cd "$src"
stage=tests;      state "RUNNING $stage"; timeout 900  "$py" tests109.py > "$out/tests.log" 2>&1
stage=synthetic;  state "RUNNING $stage"; timeout 5400 "$py" synth109.py "$out/synth" $smoke > "$out/synth_stage.log" 2>&1
"$py" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r["complete"] and (r["pass"] or r["smoke"]) else 3)' "$out/synth/synth.json"
stage=development; state "RUNNING $stage"; timeout 3600 "$py" dev109.py "$out/dev" $smoke > "$out/dev_stage.log" 2>&1
if "$py" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r["futile"] and not r["smoke"] else 1)' "$out/dev/dev.json"; then
  state "CLOSED futility (NOT_SUPPORTED)"; exit 0
fi
stage=capture;    state "RUNNING $stage"; timeout 10800 "$py" run109.py "$out/run" "$out/dev/dev.json" "$out/synth/synth.json" $smoke > "$out/run_stage.log" 2>&1
stage=analysis;   state "RUNNING $stage"; timeout 7200 "$py" analyze109.py "$out/run" "$out/dev/dev.json" "$out/synth/synth.json" "$out/metrics" > "$out/analysis.log" 2>&1
verdict=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$out/metrics/metrics109.json")
state "ANALYSIS verdict=$verdict"
if [[ $verdict == PENDING_ESCALATION ]]; then
  stage=escalation; state "RUNNING $stage"
  timeout 7200 "$py" run109.py "$out/run" "$out/dev/dev.json" "$out/synth/synth.json" $smoke --escalate > "$out/escalate_stage.log" 2>&1
  stage=analysis_escalated; state "RUNNING $stage"
  mv "$out/metrics" "$out/metrics_pre_escalation"
  timeout 7200 "$py" analyze109.py "$out/run" "$out/dev/dev.json" "$out/synth/synth.json" "$out/metrics" > "$out/analysis_escalated.log" 2>&1
  verdict=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$out/metrics/metrics109.json")
  state "ANALYSIS verdict=$verdict"
fi
if [[ $verdict == STATIC_PASS_PENDING_DYNAMIC ]]; then
  stage=dynamic; state "RUNNING $stage"
  timeout 7200 "$py" dynamic109.py "$out/dyn" "$out/run" "$out/dev/dev.json" "$out/metrics/metrics109.json" $smoke > "$out/dyn_stage.log" 2>&1
  stage=analysis_dynamic; state "RUNNING $stage"
  mv "$out/metrics" "$out/metrics_static"
  timeout 7200 "$py" analyze109.py "$out/run" "$out/dev/dev.json" "$out/synth/synth.json" "$out/metrics" --dynamic "$out/dyn/dyn.json" > "$out/analysis_dynamic.log" 2>&1
  verdict=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$out/metrics/metrics109.json")
  state "ANALYSIS verdict=$verdict"
fi
stage=done
state "COMPLETE verdict=$verdict"
