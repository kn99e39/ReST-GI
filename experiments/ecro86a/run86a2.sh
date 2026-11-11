#!/bin/bash
# WL86A amendment 1 chain. setsid nohup bash run86a2.sh <root86> <out> [--dev] > <out>/chain2.out 2>&1 &
set -u
R86=$1; OUT=$2; DEV=${3:-}
. "$OUT/env.sh"
CODE=$OUT/code/experiments/ecro86a
ST=$OUT/chain2.status
mkdir -p "$OUT/logs"
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
log "CHAIN_START amendment=1 dev=$DEV commit=$(cat $OUT/code/COMMIT 2>/dev/null)"
j=(F_fitdata "$PY d2_86a.py fitdata $R86 $OUT $DEV" T_terms "$PY d2_86a.py terms $R86 $OUT" I_indist "$PY d2_86a.py indist $R86 $OUT")
for c in E2_T1 E2_T0 E1_T1; do j+=(S_split_$c "$PY d2_86a.py split $R86 $OUT $c $DEV"); done
pool 6 "${j[@]}"
pool 2 FIT_ecro_s8601 "$PY d2_86a.py fit $R86 $OUT ecro_s8601 $DEV" FIT_ecro_s8602 "$PY d2_86a.py fit $R86 $OUT ecro_s8602 $DEV"
pool 1 E2_analyze "$PY d2_86a.py analyze $R86 $OUT"
log CHAIN_DONE
