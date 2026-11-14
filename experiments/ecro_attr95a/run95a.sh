#!/bin/bash
# WL95A chain (one detached chain): links (read-only) -> Stage 0 -> main (senders, draws, deliveries, tables) -> renders -> analysis
# [-> rule-based extension to 64 draws -> re-render pooled tables -> analysis].
#   setsid nohup bash run95a.sh <R> <W94> <W94D> <WL95CALC> <R92A> [--dev] > <R>/chain.out 2>&1 &
set -u
R=$1; W=$2; WD=$3; WC=$4; R92A=$5; DEV=${6:-}
. "$R/env.sh"
CODE=$R/code/experiments/ecro_attr95a; MT94=$R/code/experiments/ecro_mat94
O=$R/results; ST=$R/chain.status
mkdir -p "$O/logs" "$O/renders" "$O/lists" "$O/tables" "$O/pre" "$O/wl94d" "$O/wl95_calc"
log() { echo "$(date -u +%FT%TZ) $*" >> "$ST"; }
pool() {
  local maxj=$1; shift; local names=()
  while [ $# -gt 0 ]; do
    local name=$1 cmd=$2; shift 2
    if [ "$(cat "$O/logs/$name.rc" 2>/dev/null)" = 0 ]; then names+=("$name"); continue; fi
    while [ "$(jobs -rp | wc -l)" -ge "$maxj" ]; do sleep 1; done
    log "START $name"
    ( bash -c "$cmd" > "$O/logs/$name.log" 2>&1; echo $? > "$O/logs/$name.rc" ) &
    names+=("$name")
  done
  wait
  local bad=0
  for n in "${names[@]}"; do local rc; rc=$(cat "$O/logs/$n.rc" 2>/dev/null || echo 99); log "END $n rc=$rc"; [ "$rc" != 0 ] && bad=1; done
  [ $bad -ne 0 ] && { log CHAIN_FAILED; exit 1; }
}
run() { local n=$1; shift; pool 1 "$n" "$*"; }
( while true; do echo "$(date -u +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)" >> "$O/logs/gpu.txt"; sleep 20; done ) & GP=$!; disown $GP
trap 'kill $GP 2>/dev/null' EXIT
ln -sfn "$W/data" "$R/data"; ln -sfn "$W/results/fields" "$O/fields"
cp -n "$W/results/pre/masks94.npz" "$O/pre/"
for f in "$W"/results/renders/REF_ER_CK_none_*; do ln -sf "$f" "$O/renders/"; done
for f in "$W"/results/tables/FZ_own_s860?.npz; do ln -sf "$f" "$O/tables/"; done
ln -sf "$WD/results/audit94d/audit.json" "$O/wl94d/audit.json"
for s in 8601 8602; do ln -sf "$WD/results/sender/SN_own_s$s.json" "$O/wl94d/SN_own_s$s.json"; done
for f in S1_ops.npz wl95_calc.json wl95_calc2.json; do ln -sf "$WC/$f" "$O/wl95_calc/$f"; done
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null) protocol=$(sha256sum protocol/ecro_attr95a.json | cut -c1-16)"
A="$PY a95.py"; MT="$PY $MT94/mat94.py"
run S0 "$A stage0 $R $R92A $DEV"
S0=$(grep '^STAGE0' $O/logs/S0.log | tail -1 | cut -d' ' -f2); log "STAGE0 $S0"
[ -z "$DEV" ] && [ "$S0" != PASS ] && { log "STOP STAGE0"; log CHAIN_DONE; exit 0; }
renders() {
  local tag=$1; local J=()
  split -n l/6 -d "$O/lists/r95a.txt" "$O/lists/r95a_${tag}_"
  for f in "$O"/lists/r95a_${tag}_*; do J+=(R_${tag}_$(basename $f) "$MT render $R $f $DEV"); done
  pool 3 "${J[@]}"
}
run M1 "$A main $R $R92A $DEV"
renders p
run N1 "$A an $R $O/eval95a"
if grep -q '^AN EXTEND' $O/logs/N1.log; then
  log "EXTEND to 64 draws (rule: decision boundary within 2 SE)"
  run M2 "A95_EXTEND=1 $A main $R $R92A $DEV"
  rm -f "$O"/renders/CK_none_pool_*                    # pooled tables changed; deployed / floor / G tables are identical
  renders e
  run N2 "$A an $R $O/eval95a"
fi
log CHAIN_DONE
