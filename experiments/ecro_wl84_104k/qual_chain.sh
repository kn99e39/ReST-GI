#!/usr/bin/env bash
# WL104K POST HOC (owner request 2026-11-22): qualitative scene comparison. Runs the frozen, never-executed scene replay
# (scenes104k.py, unchanged) for the three existing scenes as presentation material; the frozen verdict is not changed.
# Usage: bash qual_chain.sh <run_root>
set -u
ROOT="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../../external/8dna26/.venv-cu128/Scripts/python.exe"
export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT="$ROOT/posthoc_scenes"; mkdir -p "$OUT/logs"
STATUS="$OUT/status.txt"
say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }
cd "$HERE" || exit 1
say "START posthoc qualitative scenes commit=$(git rev-parse HEAD)"
pids=()
for s in test83_iid_1_L2 test83_hard_1_L1 test83_iid_159_L2; do
  ( "$PY" -B scenes104k.py fixed "$s" "$ROOT/fit_primary" "$OUT/$s" > "$OUT/logs/$s.log" 2>&1 \
      && "$PY" -B scenes104k.py analyze "$s" "$ROOT/fit_primary" "$OUT/$s" >> "$OUT/logs/$s.log" 2>&1 \
      && echo ok > "$OUT/logs/$s.ok" ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
for s in test83_iid_1_L2 test83_hard_1_L1 test83_iid_159_L2; do [ -f "$OUT/logs/$s.ok" ] || { say "FAILED $s"; exit 1; }; done
say "DONE"
