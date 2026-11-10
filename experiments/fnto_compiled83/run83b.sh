#!/bin/bash
# WL83 amendment 2 (execution only, user-approved 2026-11-10): finish the chain after the bench with image-parallel rendering.
# Waits for the running bench process to exit, then K concurrent render processes, merge, visuals, verdict (same commands otherwise).
set -u
ROOT=$1; K=${2:-5}
. "$ROOT/env.sh"
CODE=$ROOT/code/experiments/fnto_compiled83
R=$ROOT/results; ST=$ROOT/chain.status
W81=$HOME/rest_compiled81/results/data; W82F=$HOME/rest_compiled82/results/fit
export W81
log() { echo "$(date -u +%FT%TZ) $*" >> "$ST"; }
cd "$CODE" || exit 1
log "CHAIN_B_START amendment2 K=$K commit=$(cat $ROOT/code/COMMIT)"
while pgrep -f "[b]ench83.py" > /dev/null; do sleep 5; done
[ -f "$R/bench/bench83.json" ] || { log "CHAIN_FAILED (bench output missing)"; exit 1; }
log "END B_bench rc=0 (waited)"
pids=()
for i in $(seq 0 $((K-1))); do
  log "START F_render_$i"
  ( $PY render83.py render $R/data $W81 $R/fit $W82F $R/render --part=$i/$K > $R/logs/F_render_$i.log 2>&1; echo $? > $R/logs/F_render_$i.rc ) &
  pids+=($!)
done
wait
bad=0
for i in $(seq 0 $((K-1))); do rc=$(cat $R/logs/F_render_$i.rc); log "END F_render_$i rc=$rc"; [ "$rc" != 0 ] && bad=1; done
[ $bad -ne 0 ] && { log "CHAIN_FAILED"; exit 1; }
$PY render83.py merge $R/render $K > $R/logs/F_merge.log 2>&1 || { log "CHAIN_FAILED (merge)"; exit 1; }
log "END F_merge"
$PY viz83.py $ROOT $W81 > $R/logs/V_viz.log 2>&1; log "END V_viz rc=$?"
$PY gate83.py verdict $R $R/verdict83.json > $R/logs/X_verdict.log 2>&1; log "END X_verdict rc=$?"
log "OUTCOME $($PY -c "import json;print(json.load(open('$R/verdict83.json'))['OUTCOME'])" 2>/dev/null)"
log CHAIN_DONE
