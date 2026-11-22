#!/usr/bin/env bash
# WL104K-B end-to-end chain (detached). Usage: bash chain.sh <run_root> [budget=12000]
# pre-fit contracts -> data (train_coverage, dev, test; qualified) -> new scene references (N1, N2, fallbacks)
# -> physical fits O0, O1, D, E2 (8 starts) and synthetic controls S_O0, S_O1, S_D, S_E2 (4 starts) -> fitted contracts
# -> dev evaluation (E1 selections, F eligibility; hashed) -> conditional F fit -> leak check -> test evaluation
# -> scene replays + analyses (archived 3 + qualified new) -> decision.
set -u
mkdir -p "$1" && ROOT="$(cd "$1" && (pwd -W 2>/dev/null || pwd))"; BUDGET="${2:-12000}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/../../external/8dna26/.venv-cu128/Scripts/python.exe"
export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$ROOT/logs" "$ROOT/contracts" "$ROOT/fits" "$ROOT/scenes"
STATUS="$ROOT/status.txt"
say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }
fail() { say "FAILED $*"; exit 1; }
cd "$HERE" || exit 1
say "START budget=$BUDGET commit=$(git rev-parse HEAD)"

"$PY" -B tests104kb.py "$ROOT/contracts/pre_fit.json" > "$ROOT/logs/tests_pre.log" 2>&1 || fail tests_pre
say "pre-fit contracts passed"
"$PY" -B data104kb.py "$ROOT/data" > "$ROOT/logs/data.log" 2>&1 || fail data
"$PY" -B tests104kb.py "$ROOT/contracts/data.json" --data "$ROOT" > "$ROOT/logs/tests_data.log" 2>&1 || fail tests_data
say "data qualified"

# new scene references (predeclared order; at most two qualified are used), serialized before the fits (GPU memory)
q=0
for s in test83_hard_53_L2 test83_block_186_L2 test83_block_161_L1 test83_hard_76_L1; do
  [ "$q" -ge 2 ] && break
  "$PY" -B scenes104kb.py reference "$s" "$ROOT" > "$ROOT/logs/reference_$s.log" 2>&1 || echo "reference error $s" >> "$ROOT/logs/references.txt"
  if grep -q "^QUALIFIED $s" "$ROOT/logs/reference_$s.log"; then q=$((q+1)); echo "$s" >> "$ROOT/scenes_new_qualified.txt"; fi
done
say "new scene references done: $( [ -f "$ROOT/scenes_new_qualified.txt" ] && tr '\n' ' ' < "$ROOT/scenes_new_qualified.txt" || echo NONE)"

fit() {  # name method data objective starts [--synthetic]
  local name="$1"; shift
  "$PY" -B fit104kb.py "$ROOT/fits/$name" --method "$1" --data "$2" --objective "$3" --starts "$4" ${5:-} --budget "$BUDGET" --data-dir "$ROOT/data" \
      > "$ROOT/logs/fit_$name.log" 2>&1 && echo ok > "$ROOT/logs/fit_$name.ok"
}
# fits run sequentially: one 8-start fit saturates the GPU (probe: 101.6 ms/step, 98 % utilization, 4.5 GB)
for spec in "O0 adam original original 8" "O1 lbfgs original original 8" "D adam coverage original 8" "E2 adam original e2 8"             "S_O0 adam original original 4 --synthetic" "S_O1 lbfgs original original 4 --synthetic"             "S_D adam coverage original 4 --synthetic" "S_E2 adam original e2 4 --synthetic"; do
  fit $spec
  say "fit ${spec%% *} $( [ -f "$ROOT/logs/fit_${spec%% *}.ok" ] && echo done || echo FAILED)"
done
for f in O0 O1 D E2 S_O0 S_O1 S_D S_E2; do [ -f "$ROOT/logs/fit_$f.ok" ] || fail "fit_$f"; done
say "fits done"
for f in O0 O1 D E2 S_O0 S_O1 S_D S_E2; do
  "$PY" -B tests104kb.py "$ROOT/contracts/fitted_$f.json" --fitted "$ROOT/fits/$f" > "$ROOT/logs/tests_fitted_$f.log" 2>&1 || fail "tests_fitted_$f"
done
say "fitted contracts passed"

"$PY" -B eval104kb.py dev "$ROOT" > "$ROOT/logs/eval_dev.log" 2>&1 || fail eval_dev
SPEC=$(grep "^F_SPEC" "$ROOT/logs/eval_dev.log" | tail -1)
say "dev evaluation done: $(grep '^ELIGIBLE' "$ROOT/logs/eval_dev.log" | tail -1) | $SPEC"
if echo "$SPEC" | grep -q "NOT_RUN"; then
  say "F NOT_RUN"
else
  M=$(echo "$SPEC" | sed -n 's/.*method=\([a-z]*\).*/\1/p'); D=$(echo "$SPEC" | sed -n 's/.*data=\([a-z]*\).*/\1/p'); O=$(echo "$SPEC" | sed -n 's/.*objective=\([a-z0-9]*\).*/\1/p')
  fit F "$M" "$D" "$O" 8
  [ -f "$ROOT/logs/fit_F.ok" ] || fail fit_F
  "$PY" -B tests104kb.py "$ROOT/contracts/fitted_F.json" --fitted "$ROOT/fits/F" > "$ROOT/logs/tests_fitted_F.log" 2>&1 || fail tests_fitted_F
  say "F fitted ($M $D $O)"
fi
"$PY" -B tests104kb.py "$ROOT/contracts/leak.json" --leak "$ROOT" > "$ROOT/logs/tests_leak.log" 2>&1 || fail tests_leak
"$PY" -B eval104kb.py test "$ROOT" > "$ROOT/logs/eval_test.log" 2>&1 || fail eval_test
say "test evaluation done"

SCENES="test83_iid_1_L2 test83_hard_1_L1 test83_iid_159_L2"
[ -f "$ROOT/scenes_new_qualified.txt" ] && SCENES="$SCENES $(tr '\n' ' ' < "$ROOT/scenes_new_qualified.txt")"
say "scene set: $SCENES"
spids=()
for s in $SCENES; do
  ( "$PY" -B scenes104kb.py fixed "$s" "$ROOT" "$ROOT/scenes/$s" > "$ROOT/logs/scene_$s.log" 2>&1 \
      && "$PY" -B scenes104kb.py analyze "$s" "$ROOT" "$ROOT/scenes/$s" >> "$ROOT/logs/scene_$s.log" 2>&1 \
      && echo ok > "$ROOT/logs/scene_$s.ok" ) &
  spids+=($!)
  if [ "${#spids[@]}" -ge 2 ]; then wait "${spids[0]}"; spids=("${spids[@]:1}"); fi
done
for p in "${spids[@]}"; do wait "$p"; done
for s in $SCENES; do [ -f "$ROOT/logs/scene_$s.ok" ] || say "scene $s FAILED (see log)"; done
say "scenes done"
"$PY" -B decision104kb.py "$ROOT" > "$ROOT/logs/decision.log" 2>&1 || fail decision
say "DONE"
