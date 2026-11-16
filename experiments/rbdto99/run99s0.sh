#!/bin/bash
# WL99 Stage 0 chain (ONE detached chain): focused tests -> references + qualification.
#   setsid nohup bash run99s0.sh <R> > <R>/chain.out 2>&1 &
set -u
R=$1
. "$R/env.sh"
CODE=$R/code/experiments/rbdto99; O=$R/results; ST=$R/chain.status
mkdir -p "$O"
log() { echo "$(date -u +%FT%TZ) $*" >> "$ST"; }
( while true; do echo "$(date -u +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)" >> "$O/gpu.txt"; sleep 10; done ) & GP=$!; disown $GP
trap 'kill $GP 2>/dev/null' EXIT
cd "$CODE" || exit 1
log "CHAIN_START commit=$(cat $R/code/COMMIT 2>/dev/null) protocol=$(sha256sum protocol/rbdto99.json | cut -c1-16) workloads=$(sha256sum protocol/workloads99.json | cut -c1-16)"
$PY tests99.py "$O/tests99.json" > "$O/tests99.log" 2>&1; rc=$?
log "TESTS rc=$rc all_pass=$(python3 -c "import json;print(json.load(open('$O/tests99.json'))['all_pass'])" 2>/dev/null)"
[ $rc -ne 0 ] && { log CHAIN_FAILED_TESTS; exit 1; }
$PY qual99.py "$O/stage0" > "$O/qual99.log" 2>&1; rc=$?
log "QUAL rc=$rc verdict=$(python3 -c "import json;print(json.load(open('$O/stage0/qual99.json'))['stage0_verdict'])" 2>/dev/null)"
[ $rc -ne 0 ] && { log CHAIN_FAILED_QUAL; exit 1; }
log CHAIN_DONE
