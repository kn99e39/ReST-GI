#!/bin/bash
# WL99 Stage 1 chain (non-timing part): U0 error ladders for every admitted workload, then the representation frontier (s1b) on the
# primary sharp workload and the broad workload.  Timing (exclusive GPU) is a separate step: run99s1t.sh.
#   setsid nohup bash run99s1.sh <R> > <R>/chain1.out 2>&1 &
set -u
R=$1
. "$R/env.sh"
CODE=$R/code/experiments/rbdto99; O=$R/results; ST=$R/chain.status
log() { echo "$(date -u +%FT%TZ) $*" >> "$ST"; }
cd "$CODE" || exit 1
log "S1_START commit=$(cat $R/code/COMMIT 2>/dev/null)"
$PY s1a99.py ladder "$O/stage0" "$O/stage1" > "$O/s1a_ladder.log" 2>&1; rc=$?
log "S1A_LADDER rc=$rc"
[ $rc -ne 0 ] && { log S1_FAILED; exit 1; }
for W in A1_g90s1 B1_g90s2; do
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('$O/stage0/qual99.json'))['workloads']['$W']['admitted'] else 1)"; then
    $PY s1b99.py "$O/stage0" "$O/stage1/s1b" "$W" > "$O/s1b_$W.log" 2>&1; rc=$?
    log "S1B $W rc=$rc"
    [ $rc -ne 0 ] && { log S1_FAILED; exit 1; }
  else
    log "S1B $W skipped (not admitted)"
  fi
done
log S1_NONTIMING_DONE
