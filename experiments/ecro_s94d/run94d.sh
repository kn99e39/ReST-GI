#!/bin/bash
# WL94D chain: ONE detached chain. setsid nohup bash run94d.sh <R> <W94> <W94A> <W94B> <W94C> <R86> <R92A> [--dev|--feas] > <R>/chain.out 2>&1 &
set -u
R=$1; W=$2; WA=$3; WB=$4; WC=$5; R86=$6; R92A=$7; MODE=${8:-}
DEV=; [ "$MODE" = --dev ] && DEV=--dev
. "$R/env.sh"
CODE=$R/code/experiments/ecro_s94d; MT94=$R/code/experiments/ecro_mat94; B94=$R/code/experiments/ecro_exp94b/b94.py
O=$R/results; ST=$R/chain.status
mkdir -p "$O/logs" "$O/renders" "$O/lists" "$O/tables" "$O/pre" "$O/an" "$O/models" "$O/pop" "$O/audit94d" "$O/sender" "$O/manifest"
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
ln -sfn "$W/data" "$R/data"; ln -sfn "$W/results/fields" "$O/fields"; ln -sfn "$WB/results/audit" "$O/audit"
cp -n "$W/results/pre/masks94.npz" "$O/pre/"; cp -n "$W/results/select94.json" "$O/"
for f in "$WC"/results/renders/*; do ln -sf "$f" "$O/renders/"; done
for f in "$W"/results/tables/FZ_own_s860?.npz "$WA"/results/tables/CD_C_s860?.npz "$WC"/results/tables/CD_D_s860?.npz; do ln -sf "$f" "$O/tables/"; done
for f in "$WC"/results/models/*; do ln -sf "$f" "$O/models/"; done
for f in "$WC"/results/pop/*; do ln -sf "$f" "$O/pop/"; done
for f in "$WC"/results/manifest/W_s860?.npz; do ln -sf "$f" "$O/manifest/"; done
ln -sf "$W/results/an/stageB.json" "$O/an/wl94_stageB.json"
for s in 8601 8602; do ln -sf "$WA/results/an/r_s$s.json" "$O/an/wl94a_r_s$s.json"; done
cd "$CODE" || exit 1
log "CHAIN_START mode=$MODE commit=$(cat $R/code/COMMIT 2>/dev/null) protocol=$(sha256sum protocol/ecro_s94d.json | cut -c1-16) W94C=$WC"
ASPP=4096; [ -n "$DEV" ] && ASPP=32
S="$PY s94.py"; MT="$PY $MT94/mat94.py"; C="$PY $MT94/c94.py"

J=()
for s in 8601 8602; do J+=(A_rp_own_s$s "$MT tables $R $R92A own:$s RP_own_s$s" A_rp_C_s$s "$MT tables $R $R92A cand:C_s$s RP_C_s$s"); done
pool 4 "${J[@]}"
pool 2 A_repro "$PY $B94 repro $R $R92A" A_audit "$S audit $R $R86 $R92A $DEV"
PS=$(grep '^REPRO' $O/logs/A_repro.log | tail -1 | cut -d' ' -f2); AS=$(grep '^AUDIT94D' $O/logs/A_audit.log | tail -1 | cut -d' ' -f2)
log "REPRO $PS AUDIT $AS"
[ "$MODE" = --feas ] && { log FEASIBILITY_ONLY; log CHAIN_DONE; exit 0; }
if [ -z "$DEV" ] && { [ "$PS" != PASS ] || [ "$AS" != PASS ]; }; then
  run Z_stop "$S stop $R $([ "$PS" != PASS ] && echo STOP_REPRO || echo $AS)"; log CHAIN_DONE; exit 0
fi

# GPU memory (12 GB): at most 4 concurrent CUDA processes (dev run 1 failed with 10: OOM)
J=()
for s in 8601 8602; do for c in T R P; do J+=(T_${c}_s$s "$S train $R $R86 $R92A $s $c $DEV"); done; done
pool 4 "${J[@]}"

J=(); JJ=()
for s in 8601 8602; do J+=(S_own_s$s "$S sender $R $R86 $R92A own:$s SN_own_s$s $DEV" S_D_s$s "$S sender $R $R86 $R92A cand:D_s$s SN_D_s$s $DEV"); done
for s in 8601 8602; do for c in T R P; do
  J+=(E_tab_${c}_s$s "$MT tables $R $R92A cand:${c}_s$s CD_${c}_s$s $DEV" E_pop_${c}_s$s "$C popeval $R $R86 $R92A cand:${c}_s$s P_${c}_s$s $DEV"
      E_snd_${c}_s$s "$S sender $R $R86 $R92A cand:${c}_s$s SN_${c}_s$s $DEV")
  for v in none U07 U03 bx75 bx38 bx38i; do
    L=$O/lists/r_${c}_s${s}_$v.txt; : > "$L"
    for a in C0 C2_srcS1; do for r in 0 1; do
      echo "$v results/tables/CD_${c}_s$s.npz:$v|B_$a $((86600+r)) $ASPP CK_${v}_${c}_s${s}_${a}_CK2_T1_r$r" >> "$L"; done; done
    JJ+=(E_ren_${c}_s${s}_$v "$MT render $R $L $DEV")
  done
done; done
pool 4 "${J[@]}"
pool 3 "${JJ[@]}"
pool 2 N_an_s8601 "$S an $R 8601" N_an_s8602 "$S an $R 8602"
run Z_final "$S final $R $O/eval94d"
log CHAIN_DONE
