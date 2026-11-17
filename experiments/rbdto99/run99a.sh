#!/bin/bash
# WL99A chain (ONE detached chain): Q3 v2 synthetic tests -> diagnosis + v2 qualification (qual99a.py).
#   setsid nohup bash run99a.sh <R> > <R>/chain99a.out 2>&1 &
set -u
R=$1
. "$R/env.sh"
CODE=$R/code99a/experiments/rbdto99; O=$R/results/wl99a; ST=$R/chain99a.status
mkdir -p "$O"
log() { echo "$(date -u +%FT%TZ) $*" >> "$ST"; }
( while true; do echo "$(date -u +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')" >> "$O/gpu.txt"; sleep 15; done ) & GP=$!; disown $GP
trap 'kill $GP 2>/dev/null' EXIT
cd "$CODE" || exit 1
log "CHAIN_START commit=$(cat $R/code99a/COMMIT) protocol=$(sha256sum protocol/rbdto99a.json | cut -c1-16) indep=$(sha256sum protocol/workloads99a_indep.json | cut -c1-16)"
$PY tests99a.py "$O/tests99a.json" > "$O/tests99a.log" 2>&1; rc=$?
log "TESTS99A rc=$rc all_pass=$(python3 -c "import json;print(json.load(open('$O/tests99a.json'))['all_pass'])" 2>/dev/null)"
[ $rc -ne 0 ] && { log CHAIN_FAILED_TESTS; exit 1; }
$PY qual99a.py "$R/results/stage0" "$O" > "$O/qual99a.log" 2>&1; rc=$?
log "QUAL99A rc=$rc verdict=$(python3 -c "import json;print(json.load(open('$O/qual99a.json'))['verdict'])" 2>/dev/null)"
[ $rc -ne 0 ] && { log CHAIN_FAILED; exit 1; }
log CHAIN_DONE
