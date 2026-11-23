#!/usr/bin/env bash
# WL104K-B second resume (after the ~17:53 2026-11-23 crash): block_186_L2 replay from replicate 0 (alone on the GPU), then the
# descriptive determinism check and both decisions. Usage: bash resume_b186.sh <run_root>
set -u
mkdir -p "$1" && ROOT="$(cd "$1" && (pwd -W 2>/dev/null || pwd))"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../../external/8dna26/.venv-cu128/Scripts/python.exe"
export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
STATUS="$ROOT/status.txt"
say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }
fail() { say "FAILED $*"; exit 1; }
cd "$HERE" || exit 1
say "RESUME 2 (block_186 replay alone; iid_1 analysis re-run after the zero-division execution fix) commit=$(git rev-parse HEAD)"
s=test83_block_186_L2
[ -n "$(ls -A "$ROOT/scenes/$s" 2>/dev/null)" ] && fail "scenes/$s not empty"
mv "$ROOT/logs/scene_$s.log" "$ROOT/logs/scene_${s}_crashed_empty.log" 2>/dev/null
"$PY" -B scenes104kb.py fixed "$s" "$ROOT" "$ROOT/scenes/$s" > "$ROOT/logs/scene_$s.log" 2>&1 || fail "scene fixed $s"
"$PY" -B scenes104kb.py analyze "$s" "$ROOT" "$ROOT/scenes/$s" >> "$ROOT/logs/scene_$s.log" 2>&1 || fail "scene analyze $s"
echo ok > "$ROOT/logs/scene_$s.ok"; say "scene $s done"
"$PY" -B amend104kb.py determinism "$ROOT" > "$ROOT/logs/amend_determinism.log" 2>&1 || say "determinism check FAILED (descriptive)"
"$PY" -B decision104kb.py "$ROOT" > "$ROOT/logs/decision.log" 2>&1 || fail decision
"$PY" -B decision104kb.py "$ROOT" --test "$ROOT/amendment1/test_amended.json" --out "$ROOT/amendment1/decision_amendment1.json" \
    > "$ROOT/logs/decision_amendment1.log" 2>&1 || fail decision_amendment1
say "DONE"
