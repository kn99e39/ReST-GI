#!/bin/bash
# WL94 chain (protocol ecro_mat94): ONE detached chain covering Stage A tests, Stage B frozen counterfactuals, the Stage B -> C gate,
# conditional Stage C paired smoke (seed 8601; seed 8602 only by the protocol rule) and the final analysis / figures.
#   setsid nohup bash run94.sh <R> <R86> <R92A> <R92B> <R93D> <RCK> [--dev] > <R>/chain.out 2>&1 &
set -u
R=$1; R86=$2; R92A=$3; R92B=$4; R93D=$5; RCK=$6; DEV=${7:-}
. "$R/env.sh"
CODE=$R/code/experiments/ecro_mat94
O=$R/results; ST=$R/chain.status
mkdir -p "$O/logs" "$O/renders" "$O/lists" "$O/tables" "$O/pre" "$O/an"
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
( while true; do echo "$(date -u +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)" >> "$O/logs/gpu.txt"; sleep 20; done ) & GP=$!; disown $GP
trap 'kill $GP 2>/dev/null' EXIT
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null) protocol=$(sha256sum protocol/ecro_mat94.json | cut -c1-16) R86=$R86 R92A=$R92A R92B=$R92B R93D=$R93D RCK=$RCK"
ASPP=4096; [ -n "$DEV" ] && ASPP=32
MT="$PY mat94.py"; C="$PY c94.py"; AN="$PY an94.py"

# ---------------- Stage A
run A_pre "$MT pre $R $R92A $R92B $R93D $RCK $DEV"
PS=$(grep '^PRE' $O/logs/A_pre.log | tail -1 | cut -d' ' -f2); log "PRE $PS"
[ "$PS" = PASS ] || { log "STOP $PS"; log CHAIN_DONE; exit 0; }
if [ -z "$DEV" ]; then
  for v in none bx38 bx75; do for r in 0 1 2 3; do
    f=REF_ER_CK_${v}_CK2_T1_r$r; cp -n "$RCK/results/renders/$f.npz" "$RCK/results/renders/$f.json" "$O/renders/"
    echo "$f $(sha256sum $O/renders/$f.npz | cut -c1-64)" >> "$O/pre/reused_refs.txt"; done; done
  log "REUSED_REFS $(wc -l < $O/pre/reused_refs.txt)"
fi

# ---------------- Stage B: references, frozen tables, frozen arm renders
j=()
for v in none U07 U03 bx75 bx38 bx38i; do j+=(B_fref_$v "$MT fref $R $v $DEV"); done
RV="U07 U03 bx38i"; [ -n "$DEV" ] && RV="none U07 U03 bx75 bx38 bx38i"
for v in $RV; do for r in 0 1 2 3; do j+=(B_iref_${v}_r$r "$MT iref $R $v $r $DEV"); done; done
for s in 8601 8602; do j+=(B_tab_s$s "$MT tables $R $R92A own:$s FZ_own_s$s $DEV"); done
pool 4 "${j[@]}"
j=()
for v in none U07 U03 bx75 bx38 bx38i; do
  L=$O/lists/fz_$v.txt; : > "$L"
  for s in 8601 8602; do for a in C0 C2_src C2_rcv C2_both C2_srcS1; do for r in 0 1; do
    echo "$v results/tables/FZ_own_s$s.npz:$v|B_$a $((86600+r)) $ASPP CK_${v}_own_s${s}_${a}_CK2_T1_r$r" >> "$L"; done; done; done
  j+=(B_render_$v "$MT render $R $L $DEV")
done
pool 3 "${j[@]}"
run B_an "$AN stageB $R"
GATE=$(grep '^GATE' $O/logs/B_an.log | tail -1 | cut -d' ' -f2); log "GATE $GATE"

# ---------------- Stage C (conditional)
stagec() {
  local s=$1
  pool 2 C_train_A_s$s "$C train $R $R86 $R92A $s A $DEV" C_train_B_s$s "$C train $R $R86 $R92A $s B $DEV"
  pool 3 C_tab_A_s$s "$MT tables $R $R92A cand:A_s$s CD_A_s$s $DEV" C_tab_B_s$s "$MT tables $R $R92A cand:B_s$s CD_B_s$s $DEV" \
         C_pop_own_s$s "$C popeval $R $R86 $R92A own:$s P_own_s$s $DEV" C_pop_A_s$s "$C popeval $R $R86 $R92A cand:A_s$s P_A_s$s $DEV" \
         C_pop_B_s$s "$C popeval $R $R86 $R92A cand:B_s$s P_B_s$s $DEV"
  local jj=()
  for c in A B; do for v in none U07 U03 bx75 bx38 bx38i; do
    L=$O/lists/cd_${c}_s${s}_$v.txt; : > "$L"
    for a in C0 C2_srcS1; do for r in 0 1; do
      echo "$v results/tables/CD_${c}_s$s.npz:$v|B_$a $((86600+r)) $ASPP CK_${v}_${c}_s${s}_${a}_CK2_T1_r$r" >> "$L"; done; done
    jj+=(C_render_${c}_s${s}_$v "$MT render $R $L $DEV")
  done; done
  pool 3 "${jj[@]}"
  run C_an_s$s "$AN stageC $R $s"
}
if [ "$GATE" = RUN_C ]; then
  run C_select "$C select $R $R86 $R92A $DEV"
  grep -q '^SELECT PASS' $O/logs/C_select.log || { log "STOP STOP_LEAK"; log CHAIN_DONE; exit 0; }
  NT=96; NV=24; CH=24; [ -n "$DEV" ] && { NT=8; NV=4; CH=4; }
  j=()
  for a in A B; do for st in $(seq 0 $CH $((NT-1))); do j+=(C_gen_train_${a}_$st "$C gen $R $R86 train $a $st $CH $DEV"); done; done
  for a in orig ck; do for st in $(seq 0 $((CH/2)) $((NV-1))); do j+=(C_gen_val_${a}_$st "$C gen $R $R86 val $a $st $((CH/2)) $DEV"); done; done
  pool 6 "${j[@]}"
  stagec 8601
  NX=$(grep '^STAGEC' $O/logs/C_an_s8601.log | tail -1 | awk '{print $6}'); log "RUN_NEXT_SEED $NX"
  [ "$NX" = True ] && stagec 8602
fi
run Z_final "$AN final $R $O/eval94"
log CHAIN_DONE
