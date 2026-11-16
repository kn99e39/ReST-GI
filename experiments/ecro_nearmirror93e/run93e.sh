#!/usr/bin/env bash
set -euo pipefail
R=$1
MODE=${2:-preview}
. "$R/env.sh"
C="$R/code/experiments/ecro_nearmirror93e"
mkdir -p "$R/results/logs"
ST="$R/${MODE}.status"
exec >> "$R/results/logs/${MODE}.log" 2>&1
trap 'rc=$?; echo "$(date -u +%FT%TZ) EXIT=$rc" >> "$ST"' EXIT
echo "$(date -u +%FT%TZ) START" >> "$ST"
if [ "$MODE" = preview ]; then
  for mat in S1 G075 G040 G010 MIRROR; do
    "$PY" "$C/render93e.py" render "$R" "$mat" bx38 512 128 931490
    echo "$(date -u +%FT%TZ) $mat done" >> "$ST"
  done
else
  "$PY" "$C/render93e.py" pre "$R"
  for mat in S1 G075 G040 G010 MIRROR; do
    for variant in bx38 none; do
      for seed in 931500 931501; do
        "$PY" "$C/render93e.py" render "$R" "$mat" "$variant" 1024 2048 "$seed"
        echo "$(date -u +%FT%TZ) $mat $variant $seed done" >> "$ST"
      done
    done
  done
  for variant in bx38 none; do
    for seed in 931500 931501; do
      "$PY" "$C/render93e.py" model "$R" "$variant" 1024 1024 "$seed" "$HOME/rest_x93d_ck2"
      echo "$(date -u +%FT%TZ) ECRO $variant $seed done" >> "$ST"
    done
  done
fi
echo "$(date -u +%FT%TZ) DONE" >> "$ST"
