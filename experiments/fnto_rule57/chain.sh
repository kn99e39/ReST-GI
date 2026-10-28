#!/usr/bin/env bash
# Launched by detached server-side session; no dependency on a Codex background task.
set -euo pipefail
R="$HOME/rest_wl57"
mkdir -p "$R/out/tests" "$R/tmp" "$R/logs"
cd "$R/repo/experiments/fnto_rule57"
stage=initializing
status() { printf '%s stage=%s status=%s\n' "$(date --iso-8601=seconds)" "$stage" "$1" | tee "$R/status.txt"; }
trap 'rc=$?; status "FAILED rc=$rc"; exit "$rc"' ERR
run() { stage="$1"; shift; status RUNNING; bash run.sh "$@" > "$R/logs/$stage.log" 2>&1; status DONE; }
run environment r57_environment.py
run focused tests/test_r57.py "$R/out/tests/focused.json"
run regressions r57_regressions.py
run supervision r57_data.py build
stage=data_gate;status RUNNING
if bash run.sh r57_data.py gate > "$R/logs/data_gate.log" 2>&1; then
  status DONE
else
  status STOP_GATE1
  run report r57_report.py
  stage=batch;status STOPPED_GATE1
  exit 0
fi
# Balanced arm order; identical seed budget, fixed final checkpoints.
run train_D_5701 r57_train.py D 5701
run train_R_5701 r57_train.py R 5701
run train_R_5702 r57_train.py R 5702
run train_D_5702 r57_train.py D 5702
run heldout r57_eval.py evaluate
run cost r57_eval.py cost
run composition r57_compose.py
run report r57_report.py
stage=batch;status COMPLETE
