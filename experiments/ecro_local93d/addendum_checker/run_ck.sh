#!/bin/bash
# WL93D checkerboard addendum rev. 2 chain (post hoc, qualitative; box checker reflected by the sharp panel): ONE detached chain.
#   setsid nohup bash run_ck.sh <R> <R92A> <R93A> <R93D> [--dev] > <R>/chain.out 2>&1 &
set -u
R=$1; R92A=$2; R93A=$3; R93D=$4; DEV=${5:-}
. "$R/env.sh"
CODE=$R/code/experiments/ecro_local93d/addendum_checker
O=$R/results; ST=$R/chain.status
mkdir -p "$O/logs" "$O/renders" "$O/lists" "$O/tables" "$O/pre" "$O/viz"
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
gpu() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> "$O/logs/gpu.txt"; }
( while true; do gpu; sleep 20; done ) & GP=$!; disown $GP          # not a job of this shell (pool counts jobs)
trap 'kill $GP 2>/dev/null' EXIT
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null) scene=$(sha256sum scene_ck2.json | cut -c1-16) R92A=$R92A R93A=$R93A R93D=$R93D"
RSPP=8192; ASPP=4096; RR="0 1 2 3"; [ -n "$DEV" ] && { RSPP=32; ASPP=32; RR="0 1"; }
CK="$PY checker93d.py"
run P_pre "$CK pre $R $R92A $R93A $R93D $DEV"
PS=$(grep '^PRE' $O/logs/P_pre.log | tail -1 | cut -d' ' -f2); log "PRE $PS"
[ "$PS" = PASS ] || { log "STOP $PS"; log CHAIN_DONE; exit 0; }
pool 3 T_none "$CK tables $R $R92A $R93A none" T_bx38 "$CK tables $R $R92A $R93A bx38" T_bx75 "$CK tables $R $R92A $R93A bx75"
pool 2 D_sg38 "$PY diag_sg93d.py $R $R92A $R93A bx38" D_sg75 "$PY diag_sg93d.py $R $R92A $R93A bx75"
j=()
for v in none bx38 bx75; do
  for r in $RR; do j+=(R_${v}_r$r "$CK ref $R $v $((86100+r)) $RSPP REF_ER_CK_${v}_CK2_T1_r$r"); done
  L=$O/lists/a_$v.txt; : > "$L"
  for r in 0 1; do for s in 8601 8602; do for a in C0 C2_src C2_rcv C2_both; do
    echo "results/tables/CK_${v}_CK2_T1.npz:B_own_s${s}_$a $((86600+r)) $ASPP CK_${v}_own_s${s}_${a}_CK2_T1_r$r" >> "$L"; done; done; done
  j+=(A_$v "$CK render $R $v $L")
done
pool 3 "${j[@]}"
run V_viz "$CK viz $R $O/viz $DEV"
log CHAIN_DONE
