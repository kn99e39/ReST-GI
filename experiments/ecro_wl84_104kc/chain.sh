#!/usr/bin/env bash
# WL104K-C main chain (protocol.json). Usage: bash chain.sh <run_root>   (resumable: completed steps are skipped)
# Every python call uses ABSOLUTE script paths so that gpu_duty.ps1 (match 'ecro_wl84_104kc') applies the 50 % GPU cap.
# GPU work is strictly sequential (one process at a time).
set -u
mkdir -p "$1" && ROOT="$(cd "$1" && (pwd -W 2>/dev/null || pwd))"
HERE="$(cd "$(dirname "$0")" && (pwd -W 2>/dev/null || pwd))"
PY="$HERE/../../external/8dna26/.venv-cu128/Scripts/python.exe"
export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
STATUS="$ROOT/status.txt"; mkdir -p "$ROOT/logs" "$ROOT/fits" "$ROOT/contracts"
say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }
fail() { say "FAILED $*"; exit 1; }
say "MAIN START commit=$(git -C "$HERE" rev-parse HEAD)"
grep -q '"verdict": "BASELINE_QUALIFIED"' "$ROOT/stageA/stageA.json" || fail "STAGE_A_NOT_QUALIFIED (main experiment not run)"
[ -f "$ROOT/contracts/prefit.json" ] || { "$PY" -B "$HERE/tests104kc.py" prefit "$ROOT/contracts/prefit.json" "$ROOT" \
    > "$ROOT/logs/tests_prefit.log" 2>&1 || fail prefit_contracts; say "pre-fit contracts passed"; }
[ -f "$ROOT/data/data.json" ] || { "$PY" -B "$HERE/data104kc.py" "$ROOT/data" > "$ROOT/logs/data.log" 2>&1 || fail data; say "test_c data qualified"; }
# new independent scene references (protocol order; fallbacks only after a precision failure; at most two qualified)
for s in test83_hard_76_L1 test83_block_161_L1 test83_iid_191_L1 test83_iid_0_L2; do
  q=$(grep -c '"precision_passed": true' "$ROOT/scenes_new/manifest.json" 2>/dev/null || echo 0)
  [ "$q" -ge 2 ] && break
  # attempted = present in the manifest's "scenes" record (NOT merely in its predeclared "order" list; execution recovery 2026-11-24)
  "$PY" -c "import json,sys;sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1]))['scenes'] else 1)" \
      "$ROOT/scenes_new/manifest.json" "$s" 2>/dev/null && continue
  "$PY" -B "$HERE/scenes104kc.py" reference "$s" "$ROOT" > "$ROOT/logs/reference_$s.log" 2>&1 || say "reference $s error (recorded, next in order)"
  say "reference $s: $(tail -1 "$ROOT/logs/reference_$s.log")"
done
KBDATA="$HERE/../../results/ecro_wl84_104kb/primary/data"
run_fit() {  # name arm warm starts data [extra...]
  local name=$1 arm=$2 warm=$3 starts=$4 data=$5; shift 5
  [ -f "$ROOT/fits/$name/states.npz" ] && return 0
  rm -rf "$ROOT/fits/$name"
  "$PY" -B "$HERE/fit104kc.py" "$ROOT/fits/$name" --arm "$arm" --warm "$warm" --starts "$starts" --data "$data" --data-dir "$KBDATA" "$@" \
      > "$ROOT/logs/fit_$name.log" 2>&1 || fail "fit $name"
  say "fit $name done"
}
R8="$ROOT/stageA/fits/R8/states.npz"; R4="$ROOT/stageA/fits/R4/states.npz"
run_fit CR CR "$R8" 8 original
run_fit O0C O0C "$R8" 8 original
run_fit E2W E2W "$R8" 8 original
run_fit CR_R4 CR "$R8" 8 original --warm-r4 "$R4"
run_fit CR_S CR "$ROOT/stageA/fits/S_O0r/states.npz" 4 original --synthetic
run_fit O0C_S O0C "$ROOT/stageA/fits/S_O0r/states.npz" 4 original --synthetic
run_fit E2W_S E2W "$ROOT/stageA/fits/S_O0r/states.npz" 4 original --synthetic
run_fit CR_SD CR "$ROOT/stageA/fits/S_Dr/states.npz" 4 coverage --synthetic
run_fit O0C_SD O0C "$ROOT/stageA/fits/S_Dr/states.npz" 4 coverage --synthetic
say "fits done"
[ -f "$ROOT/contracts/fitted.json" ] || { "$PY" -B "$HERE/run_kb.py" tests104kb.py "$ROOT/contracts/fitted.json" \
    $(for f in CR O0C E2W CR_R4 CR_S O0C_S E2W_S CR_SD O0C_SD; do echo --fitted "$ROOT/fits/$f"; done) \
    > "$ROOT/logs/tests_fitted.log" 2>&1 || fail fitted_contracts; say "fitted contracts passed"; }
[ -f "$ROOT/fit_decision/fit_decision.json" ] || { "$PY" -B "$HERE/eval104kc.py" fitdecision "$ROOT" > "$ROOT/logs/fit_decision.log" 2>&1 \
    || fail fit_decision; say "fit decision hashed: $(tail -1 "$ROOT/logs/fit_decision.log" | cut -c1-300)"; }
[ -f "$ROOT/contracts/leak.json" ] || { "$PY" -B "$HERE/tests104kc.py" leak "$ROOT/contracts/leak.json" "$ROOT" \
    > "$ROOT/logs/tests_leak.log" 2>&1 || fail leak_contracts; say "leak contracts passed"; }
[ -f "$ROOT/test_eval/test.json" ] || { "$PY" -B "$HERE/eval104kc.py" test "$ROOT" > "$ROOT/logs/test.log" 2>&1 || fail test; \
    say "test evaluation: $(tail -1 "$ROOT/logs/test.log")"; }
NEWQ=$("$PY" -c "import json,sys;m=json.load(open(sys.argv[1]));print(' '.join(s for s in m['order'] if m['scenes'].get(s,{}).get('precision_passed')))" \
       "$ROOT/scenes_new/manifest.json" 2>/dev/null)
say "new qualified scenes: ${NEWQ:-none}"
for s in $NEWQ test83_hard_1_L1 test83_iid_159_L2 test83_hard_53_L2 test83_block_186_L2 test83_iid_1_L2; do
  [ -f "$ROOT/scenes/$s/analysis104kc.json" ] && continue
  if [ ! -f "$ROOT/scenes/$s/fixed_00_12.json" ]; then
    rm -f "$ROOT"/scenes/"$s"/fixed_r*.npz
    "$PY" -B "$HERE/scenes104kc.py" fixed "$s" "$ROOT" > "$ROOT/logs/scene_$s.log" 2>&1 || { say "scene $s fixed FAILED"; continue; }
  fi
  "$PY" -B "$HERE/scenes104kc.py" analyze "$s" "$ROOT" >> "$ROOT/logs/scene_$s.log" 2>&1 || { say "scene $s analyze FAILED"; continue; }
  say "scene $s: $(tail -1 "$ROOT/logs/scene_$s.log" | cut -c1-300)"
done
"$PY" -B "$HERE/decision104kc.py" "$ROOT" > "$ROOT/logs/decision.log" 2>&1 || fail decision
say "decision: $(tail -1 "$ROOT/logs/decision.log")"
say "DONE"
