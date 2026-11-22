#!/usr/bin/env bash
# WL109 Amendment 2 (owner-approved 2026-11-23): resume the frozen chain at Stage 3 timing without holding GPU memory while
# foreign work runs or is queued. Same estimator, seeds, sizes, thresholds and decision rules; capture data are reused.
# Every GPU-touching command (timing per view, analysis, escalation, dynamic) runs under guard109.py:
#   start only when no tmux session exists and the GPU has had no compute process for 120 s;
#   kill the command's process group the moment a foreign compute process appears, then wait again and restart it.
# Waiting happens only in the guard (no GPU library loaded). Analysis keeps its 7200 s cap per attempt.
# usage: run109_amend2.sh TAG COMMIT
set -eu -o pipefail
cd "$HOME/rest_sss101a"
tag=$1; commit=$2
src=experiments/ecro_estimator109
out="$PWD/results/ecro_estimator109/$tag"
py="$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python"
export DYN101_MI_DIR="$HOME/rest_dyn101/external/mi_3.8.0" DYN101_MI_VERSION=3.8.0
export LD_LIBRARY_PATH="$PWD/results/ecro_denoise108/runtime/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=4 MPLCONFIGDIR="$out/mpl" TMPDIR="$out/tmp" PYTHONDONTWRITEBYTECODE=1
guard() { local name=$1 cap=$2; shift 2; "$py" "$PWD/guard109.py" --name "$name" --log "$out/guard109.jsonl" --free 120 --poll 2 --attempt-timeout "$cap" "$@"; }
stage=amend2_start
state() { printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "$out/chain.state"; }
trap 'rc=$?; if (( rc )); then state "STOP stage=$stage exit=$rc"; fi' EXIT
state "AMENDMENT2 START commit=$commit"
"$py" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r["complete"] and r["pass"] and not r["smoke"] else 3)' "$out/synth/synth.json"
"$py" -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r["complete"] and not r["futile"] and not r["smoke"] else 3)' "$out/dev/dev.json"
cd "$src"
for v in $("$py" -c 'from common109 import EVAL_VIEWS; print(" ".join(EVAL_VIEWS))'); do
  if [[ -f $out/run/timing_parts/$v.json ]]; then state "timing $v exists, kept"; continue; fi
  stage=timing_$v; state "RUNNING $stage"
  guard "timing_$v" 7200 -- "$py" timing109.py "$out/run" "$out/dev/dev.json" "$v" >> "$out/timing_stage.log" 2>&1
done
stage=timing_merge; state "RUNNING $stage"
"$py" timing109.py "$out/run" "$out/dev/dev.json" --merge >> "$out/timing_stage.log" 2>&1
stage=analysis; state "RUNNING $stage"
guard analysis 7200 -- "$py" analyze109.py "$out/run" "$out/dev/dev.json" "$out/synth/synth.json" "$out/metrics" > "$out/analysis.log" 2>&1
verdict=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$out/metrics/metrics109.json")
state "ANALYSIS verdict=$verdict"
if [[ $verdict == PENDING_ESCALATION ]]; then
  stage=escalation; state "RUNNING $stage"
  guard escalation 86400 -- "$py" run109.py "$out/run" "$out/dev/dev.json" "$out/synth/synth.json" --escalate > "$out/escalate_stage.log" 2>&1
  stage=analysis_escalated; state "RUNNING $stage"
  mv "$out/metrics" "$out/metrics_pre_escalation"
  guard analysis_escalated 7200 -- "$py" analyze109.py "$out/run" "$out/dev/dev.json" "$out/synth/synth.json" "$out/metrics" > "$out/analysis_escalated.log" 2>&1
  verdict=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$out/metrics/metrics109.json")
  state "ANALYSIS verdict=$verdict"
fi
if [[ $verdict == STATIC_PASS_PENDING_DYNAMIC ]]; then
  stage=dynamic; state "RUNNING $stage"
  guard dynamic 86400 --before "if [ -d '$out/dyn' ]; then mv '$out/dyn' '$out/dyn_preempted_'\$(date +%s); fi" -- \
    "$py" dynamic109.py "$out/dyn" "$out/run" "$out/dev/dev.json" "$out/metrics/metrics109.json" > "$out/dyn_stage.log" 2>&1
  stage=analysis_dynamic; state "RUNNING $stage"
  mv "$out/metrics" "$out/metrics_static"
  guard analysis_dynamic 7200 -- "$py" analyze109.py "$out/run" "$out/dev/dev.json" "$out/synth/synth.json" "$out/metrics" --dynamic "$out/dyn/dyn.json" > "$out/analysis_dynamic.log" 2>&1
  verdict=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$out/metrics/metrics109.json")
  state "ANALYSIS verdict=$verdict"
fi
stage=done
state "COMPLETE verdict=$verdict"
