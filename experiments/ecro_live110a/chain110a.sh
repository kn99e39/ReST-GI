#!/usr/bin/env bash
# WL110A remaining measurements on LabServer69, as one detached local chain (resumable; skips finished steps):
#   setsid -f bash experiments/ecro_live110a/chain110a.sh results/ecro_live110a/chain > results/ecro_live110a/chain.out 2>&1
# Progress: <out>/status.txt  (ends with DONE). Steps:
#   stage runtime -> app from the staged tree outside the repo (info) -> selfcheck CUDA (package path) -> selfcheck LLVM
#   (relation full, eager) -> selfcheck LLVM (relation selective, eager) -> LLVM window loop 512^2 / 256^2 (full relation,
#   uncapped, 150 frames) -> LLVM selective 256^2 (kernel-compile attribution) -> CUDA window loop through the package (512^2).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../ecro_live110/env110.sh"
OUT="$(mkdir -p "$1" && cd "$1" && pwd)"
PY=$LIVE110_PY
mkdir -p "$OUT/logs"
STATUS="$OUT/status.txt"
say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }
say "START commit=$(git -C "$HERE" rev-parse HEAD) dirty=$(git -C "$HERE" status --porcelain --untracked-files=no | wc -l)"

idle_gate() {
  local t0=$(date +%s) ok=0
  while [ $ok -lt 10 ]; do
    busy=$(timeout 5 nvidia-smi pmon -s u -c 1 2>/dev/null | awk '!/^#/ && $4 ~ /^[0-9]+$/ && $4 > 2 {n++} END {print n+0}')
    if [ "$busy" = "0" ]; then ok=$((ok+1)); else ok=0; fi
    sleep 1
    [ $(( $(date +%s) - t0 )) -gt 1800 ] && { say "idle gate TIMEOUT before $1"; return; }
  done
  say "idle gate passed before $1 after $(( $(date +%s) - t0 )) s"
}

[ -d "$OUT/stage/runtime" ] || { python3 "$HERE/tools/stage_runtime.py" "$OUT/stage/runtime" > "$OUT/logs/stage.log" 2>&1 && say "staged: $(tail -1 "$OUT/logs/stage.log" | cut -c1-120)" || say "FAILED stage"; }
rm -rf "$OUT/pkg"; mkdir -p "$OUT/pkg"; cp -r "$HERE/rest_ecro_live" "$OUT/pkg/"; find "$OUT/pkg" -name __pycache__ -prune -exec rm -r {} +
export REST_ECRO_RUNTIME="$OUT/stage/runtime"
cd "$OUT/pkg"

step() {        # step <name> <args...>   (python -m rest_ecro_live <args>, from the copied package + staged runtime)
  local name=$1; shift
  [ -f "$OUT/logs/$name.done" ] && { say "skip $name"; return; }
  idle_gate "$name"
  nvidia-smi pmon -s um -d 1 > "$OUT/logs/$name.pmon" 2>&1 & local pm=$!
  "$PY" -u -m rest_ecro_live "$@" > "$OUT/logs/$name.log" 2>&1; local rc=$?
  kill $pm 2>/dev/null
  say "$name rc=$rc $(grep -E '^(SUMMARY|SELFCHECK)' "$OUT/logs/$name.log" | cut -c1-330)"
  [ $rc = 0 ] && touch "$OUT/logs/$name.done"
}

step info_llvm info --backend llvm
step selfcheck_cuda selfcheck --backend cuda --out "$OUT/selfcheck_cuda.json"
step selfcheck_llvm_full selfcheck --backend llvm --relation full --out "$OUT/selfcheck_llvm_full.json"
step selfcheck_llvm_selective selfcheck --backend llvm --relation selective --out "$OUT/selfcheck_llvm_selective.json"

Xvfb :113 -screen 0 1024x640x24 -nolisten tcp > "$OUT/logs/xvfb.log" 2>&1 &
XPID=$!
trap 'kill $XPID 2>/dev/null' EXIT
sleep 2
export DISPLAY=:113
step view_llvm_full_512 view --backend llvm --relation full --res 512 --fps-cap 0 --frames 150 --log "$OUT/view_llvm_full_512.json"
step view_llvm_full_256 view --backend llvm --relation full --res 256 --fps-cap 0 --frames 150 --log "$OUT/view_llvm_full_256.json"
step view_llvm_selective_256 view --backend llvm --relation selective --res 256 --fps-cap 0 --frames 60 --log "$OUT/view_llvm_selective_256.json"
step view_cuda_512 view --backend cuda --res 512 --fps-cap 0 --frames 630 --log "$OUT/view_cuda_512.json"
say "DONE"
