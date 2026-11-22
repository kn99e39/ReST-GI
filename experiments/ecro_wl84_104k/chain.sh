#!/usr/bin/env bash
# WL104K end-to-end chain (detached). Usage: bash chain.sh <run_root> [steps=12000]
# contracts (gate) -> {primary fits || synthetic controls (concurrent) -> extended} -> fitted contracts -> angular evaluation
# -> scenes (only if the frozen angular gate passes) -> decision.
set -u
ROOT="$1"; STEPS="${2:-12000}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../../external/8dna26/.venv-cu128/Scripts/python.exe"
export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$ROOT/logs"
STATUS="$ROOT/status.txt"
say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }
fail() { say "FAILED $*"; exit 1; }
cd "$HERE" || exit 1
say "START steps=$STEPS commit=$(git rev-parse HEAD)"
"$PY" -B tests104k.py "$ROOT/tests.json" > "$ROOT/logs/tests.log" 2>&1 || fail tests
say "contracts passed"
( "$PY" -B fit104k.py primary "$ROOT/fit_primary" --steps "$STEPS" > "$ROOT/logs/fit_primary.log" 2>&1 && echo ok > "$ROOT/logs/primary.ok" ) &
P1=$!
( pids=()
  for c in S0 ST SP; do
    ( "$PY" -B fit104k.py synthetic "$ROOT/synth_$c" --control "$c" --steps "$STEPS" > "$ROOT/logs/synth_$c.log" 2>&1 && echo ok > "$ROOT/logs/synth_$c.ok" ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  "$PY" -B fit104k.py extended "$ROOT/fit_extended" --steps "$STEPS" > "$ROOT/logs/fit_extended.log" 2>&1 && echo ok > "$ROOT/logs/fit_extended.ok" ) &
P2=$!
wait $P1; wait $P2
for f in primary synth_S0 synth_ST synth_SP fit_extended; do [ -f "$ROOT/logs/$f.ok" ] || fail "$f"; done
say "fits done"
"$PY" -B tests104k.py "$ROOT/tests_fitted.json" --fitted "$ROOT/fit_primary" > "$ROOT/logs/tests_fitted.log" 2>&1 || fail tests_fitted
say "fitted contracts done"
"$PY" -B eval104k.py "$ROOT" "$ROOT/eval" > "$ROOT/logs/eval.log" 2>&1 || fail eval
say "eval done"
ELIG=$("$PY" -c "import json;print(json.load(open('$ROOT/eval/eval.json'))['labels']['scene_stage_eligible'])")
if [ "$ELIG" = "True" ]; then
  say "scene stage eligible"
  spids=()
  for s in test83_iid_1_L2 test83_hard_1_L1 test83_iid_159_L2; do
    ( "$PY" -B scenes104k.py fixed "$s" "$ROOT/fit_primary" "$ROOT/scenes/$s" > "$ROOT/logs/scene_$s.log" 2>&1 \
        && "$PY" -B scenes104k.py analyze "$s" "$ROOT/fit_primary" "$ROOT/scenes/$s" >> "$ROOT/logs/scene_$s.log" 2>&1 \
        && echo ok > "$ROOT/logs/scene_$s.ok" ) &
    spids+=($!)
  done
  for p in "${spids[@]}"; do wait "$p"; done
  say "scenes done"
else
  say "scene stage NOT RUN (angular gate not passed)"
fi
"$PY" -B decision104k.py "$ROOT" > "$ROOT/logs/decision.log" 2>&1 || fail decision
say "DONE"
