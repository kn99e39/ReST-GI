#!/usr/bin/env bash
# WL104J execution amendment 1 (scheduling only): resume the frozen chain after the primary fits with the remaining
# synthetic controls and the extended check run concurrently instead of sequentially, because another project's job
# occupies most of the shared GPU. Same commands, arguments, seeds and steps as chain.sh; no scientific change.
# Usage: bash chain_resume.sh <run_root> [steps=12000]
set -u
ROOT="$1"; STEPS="${2:-12000}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../../external/8dna26/.venv-cu128/Scripts/python.exe"
export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
STATUS="$ROOT/status.txt"
say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }
fail() { say "FAILED $*"; exit 1; }
cd "$HERE" || exit 1
say "RESUME (amendment 1, concurrent secondary fits) commit=$(git rev-parse HEAD)"
[ -f "$ROOT/logs/primary.ok" ] || fail primary_not_done
pids=()
for c in S1 S2; do
  [ -d "$ROOT/synth_$c" ] && fail "synth_$c exists"
  ( "$PY" -B fit104j.py synthetic "$ROOT/data" "$ROOT/synth_$c" --control "$c" --steps "$STEPS" > "$ROOT/logs/synth_$c.log" 2>&1 \
      && echo ok > "$ROOT/logs/synth_$c.ok" ) & pids+=($!)
done
( "$PY" -B fit104j.py extended "$ROOT/data" "$ROOT/fit_extended" --steps "$STEPS" > "$ROOT/logs/fit_extended.log" 2>&1 \
    && echo ok > "$ROOT/logs/fit_extended.ok" ) & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
# S0b was started by the original chain and keeps running; wait for its result
until [ -f "$ROOT/synth_S0b/fit.json" ] || grep -q Traceback "$ROOT/logs/synth_S0b.log" 2>/dev/null; do sleep 20; done
[ -f "$ROOT/synth_S0b/fit.json" ] || fail synth_S0b
for c in synth_S1 synth_S2 fit_extended; do [ -f "$ROOT/logs/$c.ok" ] || fail "$c"; done
say "fits done"
"$PY" -B eval104j.py "$ROOT" "$ROOT/eval" > "$ROOT/logs/eval.log" 2>&1 || fail eval
say "eval done"
"$PY" -B decision104j.py "$ROOT/eval" > "$ROOT/logs/decision.log" 2>&1 || fail decision
say "DONE"
