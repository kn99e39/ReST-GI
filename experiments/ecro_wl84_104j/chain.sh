#!/usr/bin/env bash
# WL104J end-to-end chain (detached). Usage: bash chain.sh <run_root> [steps=12000] [existing_data_dir]
# Stages: data -> focused tests (gate) -> {primary fits || synthetic controls + extended check} -> evaluation -> decision.
set -u
ROOT="$1"; STEPS="${2:-12000}"; DATA_SRC="${3:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../../external/8dna26/.venv-cu128/Scripts/python.exe"
export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$ROOT/logs"
STATUS="$ROOT/status.txt"
say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }
fail() { say "FAILED $*"; exit 1; }
cd "$HERE" || exit 1
say "START steps=$STEPS commit=$(git rev-parse HEAD)"
if [ -n "$DATA_SRC" ]; then cp -r "$DATA_SRC" "$ROOT/data" || fail data_copy; say "data copied from $DATA_SRC";
else "$PY" -B data104j.py "$ROOT/data" > "$ROOT/logs/data.log" 2>&1 || fail data; say "data done"; fi
"$PY" -B tests104j.py "$ROOT/data" "$ROOT/tests.json" > "$ROOT/logs/tests.log" 2>&1 || fail tests
say "tests passed"
( "$PY" -B fit104j.py primary "$ROOT/data" "$ROOT/fit_primary" --steps "$STEPS" > "$ROOT/logs/fit_primary.log" 2>&1 \
    && echo ok > "$ROOT/logs/primary.ok" ) &
P1=$!
( for c in S0a S0b S1 S2; do
    "$PY" -B fit104j.py synthetic "$ROOT/data" "$ROOT/synth_$c" --control "$c" --steps "$STEPS" > "$ROOT/logs/synth_$c.log" 2>&1 || exit 1
  done
  "$PY" -B fit104j.py extended "$ROOT/data" "$ROOT/fit_extended" --steps "$STEPS" > "$ROOT/logs/fit_extended.log" 2>&1 || exit 1
  echo ok > "$ROOT/logs/secondary.ok" ) &
P2=$!
wait $P1; wait $P2
[ -f "$ROOT/logs/primary.ok" ] || fail fit_primary
[ -f "$ROOT/logs/secondary.ok" ] || fail synthetic_or_extended
say "fits done"
"$PY" -B eval104j.py "$ROOT" "$ROOT/eval" > "$ROOT/logs/eval.log" 2>&1 || fail eval
say "eval done"
"$PY" -B decision104j.py "$ROOT/eval" > "$ROOT/logs/decision.log" 2>&1 || fail decision
say "DONE"
