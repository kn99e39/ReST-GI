#!/bin/bash
# WL86A chain (diagnostic; frozen WL86 data in <root86> are read only). setsid nohup bash run86a.sh <root86> <out> [--dev] > <out>/chain.out 2>&1 &
set -u
R86=$1; OUT=$2; DEV=${3:-}
. "$OUT/env.sh"
CODE=$OUT/code/experiments/ecro86a
ST=$OUT/chain.status
mkdir -p "$OUT/logs" "$OUT/stageB" "$OUT/results"
log() { echo "$(date -u +%FT%TZ) $*" >> "$ST"; }
pool() {
  local maxj=$1; shift; local names=()
  while [ $# -gt 0 ]; do
    local name=$1 cmd=$2; shift 2
    while [ "$(jobs -rp | wc -l)" -ge "$maxj" ]; do sleep 1; done
    log "START $name"
    ( bash -c "$cmd" > "$OUT/logs/$name.log" 2>&1; echo $? > "$OUT/logs/$name.rc" ) &
    names+=("$name")
  done
  wait
  local bad=0
  for n in "${names[@]}"; do local rc; rc=$(cat "$OUT/logs/$n.rc" 2>/dev/null || echo 99); log "END $n rc=$rc"; [ "$rc" != 0 ] && bad=1; done
  [ $bad -ne 0 ] && { log CHAIN_FAILED; exit 1; }
}
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $OUT/code/COMMIT 2>/dev/null)"
CFGS="E2_T1 E2_T0 E1_T1"
# Stage A / B on the frozen WL86 data + Stage C fields and beams (concurrent)
j=(A_baseline "$PY diag86a.py baseline $R86 $OUT/stageA" B_localize "$PY diag86a.py localize $R86 $OUT/stageB" C_beams "$PY attrib86a.py beams $R86 $OUT $DEV")
for c in $CFGS; do j+=(C_fields_$c "$PY attrib86a.py fields $R86 $OUT $c $DEV"); done
pool 6 "${j[@]}"
pool 1 D_model "$PY attrib86a.py model $R86 $OUT"
j=()
for c in $CFGS; do for t in TRUE ANG_PANEL ENC_PANEL ENC_ALL; do for r in 0 1; do j+=(R_${t}_${c}_r$r "$PY attrib86a.py render $R86 $OUT $c $t $r $DEV"); done; done; done
pool 6 "${j[@]}"
pool 1 E_analyze "$PY analyze86a.py $R86 $OUT"
pool 1 V_viz "$PY viz86a.py $R86 $OUT"
log CHAIN_DONE
