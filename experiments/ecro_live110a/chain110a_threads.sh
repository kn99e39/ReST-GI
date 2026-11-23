#!/usr/bin/env bash
# WL110A CPU thread-budget sweep for the LLVM backend (headless, 512^2 and 256^2, full relation, 60 frames each); detached:
#   setsid -f bash experiments/ecro_live110a/chain110a_threads.sh results/ecro_live110a/threads > results/ecro_live110a/threads.out 2>&1
# Configurations: torch threads / Dr.Jit threads = default (8/16), 8/8, 4/12, 12/4, 6/6, 16/16 on the 8C/16T host.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../ecro_live110/env110.sh"
OUT="$(mkdir -p "$1" && cd "$1" && pwd)"
PY=$LIVE110_PY
STATUS="$OUT/status.txt"
say() { echo "$(date '+%F %T') $*" >> "$STATUS"; }
say "START commit=$(git -C "$HERE" rev-parse HEAD) dirty=$(git -C "$HERE" status --porcelain --untracked-files=no | wc -l)"
cd "$HERE"
for res in 512 256; do
  for cfg in "0 0" "8 8" "4 12" "12 4" "6 6" "16 16"; do
    set -- $cfg
    name="r${res}_t$1_d$2"
    [ -f "$OUT/$name.json" ] && { say "skip $name"; continue; }
    "$PY" -u -m rest_ecro_live view --backend llvm --relation full --res $res --fps-cap 0 --frames 60 --warmup 15 --present none \
        --torch-threads $1 --drjit-threads $2 --log "$OUT/$name.json" > "$OUT/$name.log" 2>&1
    say "$name rc=$? $(grep '^SUMMARY' "$OUT/$name.log" | cut -c1-300)"
  done
done
say "DONE"
