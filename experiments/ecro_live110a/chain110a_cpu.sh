#!/usr/bin/env bash
# WL110A CPU-gated LLVM measurements (detached):
#   setsid -f bash experiments/ecro_live110a/chain110a_cpu.sh results/ecro_live110a/cpu > results/ecro_live110a/cpu.out 2>&1
# The first thread sweep (chain110a_threads.sh) overlapped another session's CPU-heavy job (dev109b.py, load 20 on 16 threads).
# Every step here first waits until no foreign process uses > 30 % CPU and the 1-min load average is < 3 for 30 s (max 6 h).
# Steps (60 measured frames after 15 warm-up, full relation, uncapped): thread configs 8/16 (default), 8/8, 4/12 at 512^2 and
# 256^2 headless; then window (Tk on Xvfb) vs headless at both resolutions with the default threads (Tk-slowdown attribution).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../ecro_live110/env110.sh"
OUT="$(mkdir -p "$1" && cd "$1" && pwd)"
PY=$LIVE110_PY
STATUS="$OUT/status.txt"
say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }
say "START commit=$(git -C "$HERE" rev-parse HEAD) dirty=$(git -C "$HERE" status --porcelain --untracked-files=no | wc -l)"
cpu_gate() {
  local t0=$(date +%s) ok=0
  while [ $ok -lt 30 ]; do
    busy=$(ps -eo pid,pcpu --no-headers | awk -v me=$$ '$2 > 30 {n++} END {print n+0}')
    load=$(awk '{print ($1 < 3) ? 1 : 0}' /proc/loadavg)
    if [ "$busy" = "0" ] && [ "$load" = "1" ]; then ok=$((ok+1)); else ok=0; fi
    sleep 1
    [ $(( $(date +%s) - t0 )) -gt 21600 ] && { say "cpu gate TIMEOUT before $1 (load $(cut -d' ' -f1 /proc/loadavg))"; return; }
  done
  say "cpu gate passed before $1 after $(( $(date +%s) - t0 )) s (load $(cut -d' ' -f1 /proc/loadavg))"
}
Xvfb :114 -screen 0 1024x640x24 -nolisten tcp > "$OUT/xvfb.log" 2>&1 &
XPID=$!
trap 'kill $XPID 2>/dev/null' EXIT
cd "$HERE"
run() {        # run <name> <args...>
  local name=$1; shift
  [ -f "$OUT/$name.json" ] && { say "skip $name"; return; }
  cpu_gate "$name"
  (top -b -d 1 -n 120 > "$OUT/$name.top" 2>&1 &)
  "$PY" -u -m rest_ecro_live view --backend llvm --relation full --fps-cap 0 --frames 75 --warmup 15 --log "$OUT/$name.json" "$@" \
      > "$OUT/$name.log" 2>&1
  say "$name rc=$? $(grep '^SUMMARY' "$OUT/$name.log" | cut -c1-300)"
  pkill -u "$USER" -f "top -b -d 1 -n 120" 2>/dev/null
}
for res in 512 256; do
  for cfg in "0 0" "8 8" "4 12"; do
    set -- $cfg
    run "headless_r${res}_t$1_d$2" --res $res --present none --torch-threads $1 --drjit-threads $2
  done
done
export DISPLAY=:114
run window_r512 --res 512 --present tk
run window_r256 --res 256 --present tk
say "DONE"
