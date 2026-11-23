#!/usr/bin/env bash
# WL104K-C Stage A chain (protocol_stageA.json). Usage: bash chainA.sh <run_root>   (resumable: completed steps are skipped)
# Every python call uses ABSOLUTE script paths so that gpu_duty.ps1 (match 'ecro_wl84_104kc') throttles it to the 50 % GPU cap.
set -u
mkdir -p "$1" && ROOT="$(cd "$1" && (pwd -W 2>/dev/null || pwd))"
HERE="$(cd "$(dirname "$0")" && (pwd -W 2>/dev/null || pwd))"
PY="$HERE/../../external/8dna26/.venv-cu128/Scripts/python.exe"
export PYTHONUTF8=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
STATUS="$ROOT/status.txt"; mkdir -p "$ROOT/logs" "$ROOT/stageA/fits"
say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }
fail() { say "FAILED $*"; exit 1; }
say "STAGE A START commit=$(git -C "$HERE" rev-parse HEAD)"
[ -f "$ROOT/stageA/shape.json" ] || { "$PY" -B "$HERE/stageA104kc.py" shape "$ROOT/stageA" > "$ROOT/logs/stageA_shape.log" 2>&1 || fail shape; say "shape done"; }
KBDATA="$HERE/../../results/ecro_wl84_104kb/primary/data"
for spec in "R4 --starts 4 --data original" "R8 --starts 8 --data original" \
            "S_O0r --starts 4 --data original --synthetic" "S_Dr --starts 4 --data coverage --synthetic"; do
  set -- $spec; name=$1; shift
  [ -f "$ROOT/stageA/fits/$name/states.npz" ] && continue
  rm -rf "$ROOT/stageA/fits/$name"
  "$PY" -B "$HERE/run_kb.py" fit104kb.py "$ROOT/stageA/fits/$name" --method adam --objective original --budget 12000 \
      --data-dir "$KBDATA" "$@" > "$ROOT/logs/stageA_fit_$name.log" 2>&1 || fail "fit $name"
  say "fit $name done"
done
"$PY" -B "$HERE/stageA104kc.py" compare "$ROOT" > "$ROOT/logs/stageA_compare.log" 2>&1 || fail compare
say "stage A: $(tail -1 "$ROOT/logs/stageA_compare.log")"
say "STAGE A DONE"
