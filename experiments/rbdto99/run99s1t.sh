#!/bin/bash
# WL99 Stage 1 timing (exclusive GPU): waits until no other process holds the GPU, then times E1 / oracle frames, then U0.
#   setsid nohup bash run99s1t.sh <R> > <R>/chain1t.out 2>&1 &
set -u
R=$1
. "$R/env.sh"
CODE=$R/code/experiments/rbdto99; O=$R/results; ST=$R/chain.status
log() { echo "$(date -u +%FT%TZ) $*" >> "$ST"; }
cd "$CODE" || exit 1
log "S1T_WAIT_EXCLUSIVE_GPU"
while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; do sleep 30; done
log "S1T_START gpu_idle=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"
( while true; do echo "$(date -u +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')" >> "$O/gpu_timing.txt"; sleep 5; done ) & GP=$!; disown $GP
trap 'kill $GP 2>/dev/null' EXIT
$PY s1a99.py timing "$O/stage1" "$O/stage0" > "$O/s1a_timing.log" 2>&1; rc=$?
log "S1A_TIMING rc=$rc"
[ $rc -ne 0 ] && { log S1T_FAILED; exit 1; }
$PY s1a99.py u0 "$O/stage0" "$O/stage1" > "$O/s1a_u0.log" 2>&1; rc=$?
log "S1A_U0 rc=$rc"
log S1T_DONE
