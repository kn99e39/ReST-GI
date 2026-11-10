#!/bin/bash
# WL84 chain (one detached chain, all stages sequential and single-process; stage D runs alone by construction).
#   setsid nohup bash run84.sh <root> > <root>/chain.out 2>&1 &
set -u
ROOT=$1
. "$ROOT/env.sh"
CODE=$ROOT/code/experiments/fnto_compiled84
R=$ROOT/results; ST=$ROOT/chain.status
mkdir -p "$R/logs" "$R/ref" "$R/replicates" "$R/econ" "$R/audit"
log() { echo "$(date -u +%FT%TZ) $*" >> "$ST"; }
step() { local n=$1; shift; log "START $n"; nvidia-smi --query-compute-apps=pid --format=csv,noheader > "$R/logs/$n.gpu_before"; "$@" > "$R/logs/$n.log" 2>&1; local rc=$?; log "END $n rc=$rc"; [ $rc -ne 0 ] && { log CHAIN_FAILED; exit 1; }; }
cd "$CODE" || exit 1
log "CHAIN_START commit=$(cat $ROOT/code/COMMIT)"
step A_audit $PY render84.py audit $R/audit
g=$($PY gate84.py audit $R); log "GATE A $g"
[ "$g" = REPRODUCED ] || { log "STOP WL83_STATE_NOT_REPRODUCED"; log CHAIN_DONE; exit 0; }
for im in test83_iid_1_L2 test83_iid_0_L2 test83_hard_1_L1; do step B_ref_$im $PY render84.py refgen $im $R/ref; done
for im in test83_iid_1_L2 test83_iid_0_L2 test83_hard_1_L1; do step C_rep_$im $PY render84.py replicates $im $R/replicates; done
step D_econ $PY render84.py econ $R/econ
step C_eval $PY render84.py evaluate $R
step V_viz $PY viz84.py $R
step X_verdict $PY gate84.py verdict $R
log "OUTCOME $($PY -c "import json;print(json.load(open('$R/verdict84.json'))['OUTCOME'])")"
log CHAIN_DONE
