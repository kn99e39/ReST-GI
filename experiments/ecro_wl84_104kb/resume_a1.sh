#!/usr/bin/env bash
# WL104K-B resume after the 2026-11-23 06:06 reboot + Amendment 1 (amendment1.json). Usage: bash resume_a1.sh <run_root>
# 1) preserve the interrupted partial scene replicates; 2) Amendment 1 reruns R4a, R4b, R8 (sequential, GPU-saturating) and evaluation;
# 3) scene replays + analyses for all five scenes from replicate 0 (identical seeds / code); 4) determinism check of the partial files;
# 5) original-rule decision (decision.json) and amended decision (amendment1/decision_amendment1.json).
set -u
mkdir -p "$1" && ROOT="$(cd "$1" && (pwd -W 2>/dev/null || pwd))"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../../external/8dna26/.venv-cu128/Scripts/python.exe"
export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
STATUS="$ROOT/status.txt"
say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }
fail() { say "FAILED $*"; exit 1; }
cd "$HERE" || exit 1
say "RESUME (Amendment 1 + scene replays) commit=$(git rev-parse HEAD)"

mkdir -p "$ROOT/scenes_interrupted" "$ROOT/amendment1/fits"
for s in "$ROOT"/scenes/*/; do
  [ -d "$s" ] || continue
  name=$(basename "$s")
  [ -e "$ROOT/scenes_interrupted/$name" ] && fail "scenes_interrupted/$name exists"
  mv "$s" "$ROOT/scenes_interrupted/$name" || fail "move $name"
done
say "interrupted partial scene replicates preserved in scenes_interrupted/"

for spec in "R4a 4" "R4b 4" "R8 8"; do
  set -- $spec
  "$PY" -B fit104kb.py "$ROOT/amendment1/fits/$1" --method adam --data original --objective original --starts "$2" --budget 12000 \
      --data-dir "$ROOT/data" > "$ROOT/logs/amend_fit_$1.log" 2>&1 || fail "amend_fit_$1"
  say "amendment rerun $1 done"
done
"$PY" -B amend104kb.py evaluate "$ROOT" > "$ROOT/logs/amend_eval.log" 2>&1 || fail amend_eval
say "amendment 1: $(grep '^AMENDMENT1' "$ROOT/logs/amend_eval.log" | tail -1)"

SCENES="test83_iid_1_L2 test83_hard_1_L1 test83_iid_159_L2 $(tr '\n' ' ' < "$ROOT/scenes_new_qualified.txt")"
say "scene set: $SCENES"
spids=()
for s in $SCENES; do
  ( "$PY" -B scenes104kb.py fixed "$s" "$ROOT" "$ROOT/scenes/$s" > "$ROOT/logs/scene_$s.log" 2>&1 \
      && "$PY" -B scenes104kb.py analyze "$s" "$ROOT" "$ROOT/scenes/$s" >> "$ROOT/logs/scene_$s.log" 2>&1 \
      && echo ok > "$ROOT/logs/scene_$s.ok" && echo "$(date '+%F %T') scene $s done" >> "$STATUS" ) &
  spids+=($!)
  if [ "${#spids[@]}" -ge 2 ]; then wait "${spids[0]}"; spids=("${spids[@]:1}"); fi
done
for p in "${spids[@]}"; do wait "$p"; done
for s in $SCENES; do [ -f "$ROOT/logs/scene_$s.ok" ] || say "scene $s FAILED (see log)"; done
say "scenes done"
"$PY" -B amend104kb.py determinism "$ROOT" > "$ROOT/logs/amend_determinism.log" 2>&1 || say "determinism check FAILED (descriptive)"
"$PY" -B decision104kb.py "$ROOT" > "$ROOT/logs/decision.log" 2>&1 || fail decision
"$PY" -B decision104kb.py "$ROOT" --test "$ROOT/amendment1/test_amended.json" --out "$ROOT/amendment1/decision_amendment1.json" \
    > "$ROOT/logs/decision_amendment1.log" 2>&1 || fail decision_amendment1
say "DONE"
