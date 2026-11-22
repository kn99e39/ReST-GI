#!/usr/bin/env bash
# WL104K-A chain (detached). Usage: bash chain.sh <run_root>
set -u
ROOT="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
K="$HERE/../ecro_wl84_104k"
PY="$HERE/../../external/8dna26/.venv-cu128/Scripts/python.exe"
export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$ROOT/logs"
STATUS="$ROOT/status.txt"
say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }
fail() { say "FAILED $*"; exit 1; }
cd "$HERE" || exit 1
say "START commit=$(git rev-parse HEAD)"
"$PY" -B stageA104ka.py "$ROOT/stageA.json" > "$ROOT/logs/stageA.log" 2>&1 || fail stageA
say "stage A reproduced"
"$PY" -B extend104ka.py fit "$ROOT/ext" > "$ROOT/logs/ext_fit.log" 2>&1 || fail ext_fit
say "extension done"
( cd "$K" && "$PY" -B tests104k.py "$ROOT/tests_fitted_ext.json" --fitted "$ROOT/ext" ) > "$ROOT/logs/tests_fitted_ext.log" 2>&1 || fail tests_fitted_ext
"$PY" -B extend104ka.py eval "$ROOT/ext" "$ROOT/ext_angular.json" > "$ROOT/logs/ext_eval.log" 2>&1 || fail ext_eval
say "extended states qualified and evaluated"
pids=()
for s in test83_iid_1_L2 test83_hard_1_L1 test83_iid_159_L2; do
  ( "$PY" -B lanes104ka.py "$s" "$ROOT/ext" "$ROOT/lanes/$s" > "$ROOT/logs/lanes_$s.log" 2>&1 && echo ok > "$ROOT/logs/lanes_$s.ok" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
for s in test83_iid_1_L2 test83_hard_1_L1 test83_iid_159_L2; do [ -f "$ROOT/logs/lanes_$s.ok" ] || fail "lanes $s"; done
say "lane attribution done"
pids=()
for s in test83_iid_1_L2 test83_hard_1_L1 test83_iid_159_L2; do
  ( cd "$K" && "$PY" -B scenes104k.py fixed "$s" "$ROOT/ext" "$ROOT/scenes_ext/$s" && "$PY" -B scenes104k.py analyze "$s" "$ROOT/ext" "$ROOT/scenes_ext/$s" ) \
      > "$ROOT/logs/scene_ext_$s.log" 2>&1 && echo ok > "$ROOT/logs/scene_ext_$s.ok" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
for s in test83_iid_1_L2 test83_hard_1_L1 test83_iid_159_L2; do [ -f "$ROOT/logs/scene_ext_$s.ok" ] || fail "scene_ext $s"; done
say "extended scene replay done"
"$PY" -B decision104ka.py "$ROOT" > "$ROOT/logs/decision.log" 2>&1 || fail decision
say "DONE"
