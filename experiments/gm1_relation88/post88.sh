#!/bin/bash
# WL88 post-freeze completion of the stopped branch (analysis-code fix only; no rule change)
R=$HOME/rest_rel88; . $R/env.sh; cd $R/code/experiments/gm1_relation88
ST=$R/chain.status; O=$R/results
log() { echo "$(date -u +%FT%TZ) $*" >> "$ST"; }
log "POSTFIX_START analysis fix of the STOPPED_WORKLOAD_UNQUALIFIED branch (analyze88.py, viz88.py)"
for st in stage1 stage2 verdict; do
  log "START A_$st"; $PY analyze88.py $st $R $HOME/rest_ecro86 > $O/logs/A_$st.log 2>&1; rc=$?; echo $rc > $O/logs/A_$st.rc; log "END A_$st rc=$rc"
done
log "START V_viz"; $PY viz88.py $R $HOME/rest_ecro86 $O/viz > $O/logs/V_viz.log 2>&1; rc=$?; echo $rc > $O/logs/V_viz.rc; log "END V_viz rc=$rc"
log POSTFIX_DONE
