#!/bin/bash
# WL93D checkerboard addendum chain (post hoc, qualitative): ONE detached chain, end to end.
#   setsid nohup bash run_ck.sh <R> <R86> <R92A> <R93A> <R93D> [--dev] > <R>/chain.out 2>&1 &
set -u
R=$1; R86=$2; R92A=$3; R93A=$4; R93D=$5; DEV=${6:-}
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
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null) R86=$R86 R92A=$R92A R93A=$R93A R93D=$R93D"
RSPP=8192; ASPP=4096; PSPP=64; RR="0 1 2 3"; [ -n "$DEV" ] && { RSPP=32; ASPP=32; PSPP=16; RR="0 1"; }
CK="$PY checker93d.py"
# ---- preflight (stop on failure) + 64-spp previews
run P_pre "$CK pre $R $R86 $R92A $R93A $R93D $DEV"
PS=$(grep '^PRE' $O/logs/P_pre.log | tail -1 | cut -d' ' -f2); log "PRE $PS"
pool 3 P_prev_none "$CK preview $R none $PSPP PREV_none" P_prev_ck38 "$CK preview $R ck38 $PSPP PREV_ck38" P_prev_ck75 "$CK preview $R ck75 $PSPP PREV_ck75"
# A4 (post hoc, after the full 25 cm results): ck38 materials with 1 m patch-aligned cells; its own preflight (same tests, variant ck38L)
run P_pre_ck38L "$CK pre $R $R86 $R92A $R93A $R93D ck38L $DEV"
PL=$(grep '^PRE' $O/logs/P_pre_ck38L.log | tail -1 | cut -d' ' -f2); log "PRE_ck38L $PL"
[ "$PL" = PASS ] || { log "STOP ck38L $PL"; log CHAIN_DONE; exit 0; }
[ "$PS" = PASS ] || { log "STOP $PS"; log CHAIN_DONE; exit 0; }
# ---- B tables (frozen own s8601 / s8602; arms C0, C2_src, C2_rcv, C2_both)
pool 3 T_ck38 "$CK tables $R $R92A $R93A ck38" T_ck75 "$CK tables $R $R92A $R93A ck75" T_ck38L "$CK tables $R $R92A $R93A ck38L"
# ---- references (common random numbers with the WL86 untextured E2 T1 references r0-r3) + arm renders (CRN with WL93D)
j=()
for v in ck38 ck75 ck38L; do
  for r in $RR; do j+=(R_${v}_r$r "$CK ref $R $v $((86100+r)) $RSPP REF_ER_CK_${v}_E2_T1_r$r"); done
  L=$O/lists/a_$v.txt; : > "$L"
  for r in 0 1; do for s in 8601 8602; do for a in C0 C2_src C2_rcv C2_both; do
    echo "results/tables/CK_${v}_E2_T1.npz:B_own_s${s}_$a $((86600+r)) $ASPP CK_${v}_own_s${s}_${a}_E2_T1_r$r" >> "$L"; done; done; done
  j+=(A_$v "$CK render $R $v $L")
done
pool 3 "${j[@]}"
run V_viz "$CK viz $R $R86 $R93D $O/viz $DEV"
log CHAIN_DONE
