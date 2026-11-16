#!/usr/bin/env bash
# Independent reference jobs, not agents. Same frozen settings/seeds/code.
# Finish these eight frames before the primary chain reaches them; the primary
# renderer skips completed JSON-marked renders. No performance claim is made.
set -euo pipefail
R=$1
. "$R/env.sh"
C="$R/code/experiments/ecro_nearmirror93e"
exec >> "$R/results/logs/high.log" 2>&1
trap 'rc=$?; echo "$(date -u +%FT%TZ) EXIT=$rc" >> "$R/high.status"' EXIT
echo "$(date -u +%FT%TZ) START" >> "$R/high.status"
for mat in G010 MIRROR; do
  for variant in bx38 none; do
    for seed in 931500 931501; do
      "$PY" "$C/render93e.py" render "$R" "$mat" "$variant" 1024 2048 "$seed"
      echo "$(date -u +%FT%TZ) $mat $variant $seed done" >> "$R/high.status"
    done
  done
done
echo "$(date -u +%FT%TZ) DONE" >> "$R/high.status"
