#!/bin/bash
# WL94B chain: ONE detached chain - read-only links to the WL94 / WL94A roots, split re-render audit + T_REPRO tables, audit gates,
# candidate D (seeds 8601 / 8602), evaluation, final outcome / figures.
#   setsid nohup bash run94b.sh <R> <W94> <W94A> <R86> <R92A> [--dev] > <R>/chain.out 2>&1 &
set -u
R=$1; W=$2; WA=$3; R86=$4; R92A=$5; DEV=${6:-}
. "$R/env.sh"
CODE=$R/code/experiments/ecro_exp94b; MT94=$R/code/experiments/ecro_mat94
O=$R/results; ST=$R/chain.status
mkdir -p "$O/logs" "$O/renders" "$O/lists" "$O/tables" "$O/pre" "$O/an" "$O/models" "$O/pop" "$O/audit"
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
# read-only links to the WL94 / WL94A evidence roots
ln -sfn "$W/data" "$R/data"; ln -sfn "$W/results/fields" "$O/fields"
cp -n "$W/results/pre/masks94.npz" "$O/pre/"; cp -n "$W/results/select94.json" "$O/"
for f in "$WA"/results/renders/*; do ln -sf "$f" "$O/renders/"; done
for f in "$W"/results/tables/FZ_own_s860?.npz "$WA"/results/tables/CD_A_s860?.npz "$WA"/results/tables/CD_C_s860?.npz; do ln -sf "$f" "$O/tables/"; done
for f in "$WA"/results/models/*; do ln -sf "$f" "$O/models/"; done
for f in "$WA"/results/pop/*; do ln -sf "$f" "$O/pop/"; done
ln -sf "$W/results/an/stageB.json" "$O/an/wl94_stageB.json"
for s in 8601 8602; do ln -sf "$WA/results/an/r_s$s.json" "$O/an/wl94a_r_s$s.json"; done
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null) protocol=$(sha256sum protocol/ecro_exp94b.json | cut -c1-16) W94=$W W94A=$WA R86=$R86 R92A=$R92A"
ASPP=4096; [ -n "$DEV" ] && ASPP=32
B="$PY b94.py"; MT="$PY $MT94/mat94.py"; C="$PY $MT94/c94.py"
NA=96; CH=24; [ -n "$DEV" ] && { NA=8; CH=8; }

# Stage A: split re-render (target seeds + independent seeds) and T_REPRO tables, one wide pool
J=()
for w in tgt ind; do for ((s=0; s<NA; s+=CH)); do J+=(A_agen_${w}_$s "$B agen $R $R86 $w $s $CH $DEV"); done; done
for s in 8601 8602; do J+=(A_rp_own_s$s "$MT tables $R $R92A own:$s RP_own_s$s $DEV" A_rp_C_s$s "$MT tables $R $R92A cand:C_s$s RP_C_s$s $DEV"); done
pool 6 "${J[@]}"
pool 2 A_repro "$B repro $R $R92A $DEV" A_audit "$B audit $R $R86 $DEV"
PS=$(grep '^REPRO' $O/logs/A_repro.log | tail -1 | cut -d' ' -f2); AS=$(grep '^AUDIT' $O/logs/A_audit.log | tail -1 | cut -d' ' -f2)
log "REPRO $PS AUDIT $AS"
if [ -z "$DEV" ] && { [ "$PS" != PASS ] || [ "$AS" != PASS ]; }; then
  run Z_stop "$B stop $R $O/eval94b $([ "$PS" != PASS ] && echo STOP_REPRO || echo $AS)"; log CHAIN_DONE; exit 0
fi

# Stage C: candidate D, both seeds
pool 2 T_D_s8601 "$B train $R $R86 $R92A 8601 $DEV" T_D_s8602 "$B train $R $R86 $R92A 8602 $DEV"

# Stage D: tables, population evaluation, renders (C0 + exact panel sender C2_srcS1)
J=(); JJ=()
for s in 8601 8602; do
  J+=(E_tab_D_s$s "$MT tables $R $R92A cand:D_s$s CD_D_s$s $DEV" E_pop_D_s$s "$C popeval $R $R86 $R92A cand:D_s$s P_D_s$s $DEV")
  for v in none U07 U03 bx75 bx38 bx38i; do
    L=$O/lists/r_D_s${s}_$v.txt; : > "$L"
    for a in C0 C2_srcS1; do for r in 0 1; do
      echo "$v results/tables/CD_D_s$s.npz:$v|B_$a $((86600+r)) $ASPP CK_${v}_D_s${s}_${a}_CK2_T1_r$r" >> "$L"; done; done
    JJ+=(E_ren_D_s${s}_$v "$MT render $R $L $DEV")
  done
done
pool 4 "${J[@]}"
pool 3 "${JJ[@]}"
pool 2 N_an_s8601 "$B an $R 8601" N_an_s8602 "$B an $R 8602"
run Z_final "$B final $R $O/eval94b"
log CHAIN_DONE
