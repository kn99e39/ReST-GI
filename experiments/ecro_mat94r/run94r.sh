#!/bin/bash
# WL94A (WL94-R) chain: ONE detached chain - links to the WL94 root (read only), T_REPRO / T_MIX / T_LEAK, candidate C + Ar control,
# evaluation, conditional seed-8602 replication (C + compute-matched A), final outcome / figures.
#   setsid nohup bash run94r.sh <R> <W94> <R86> <R92A> [--dev] > <R>/chain.out 2>&1 &
set -u
R=$1; W=$2; R86=$3; R92A=$4; DEV=${5:-}
. "$R/env.sh"
CODE=$R/code/experiments/ecro_mat94r; MT94=$R/code/experiments/ecro_mat94
O=$R/results; ST=$R/chain.status
mkdir -p "$O/logs" "$O/renders" "$O/lists" "$O/tables" "$O/pre" "$O/an" "$O/models" "$O/pop"
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
# read-only links to the WL94 evidence root
ln -sfn "$W/data" "$R/data"; ln -sfn "$W/results/fields" "$O/fields"
cp -n "$W/results/pre/masks94.npz" "$O/pre/"; cp -n "$W/results/select94.json" "$O/"
for f in "$W"/results/renders/*; do ln -sf "$f" "$O/renders/"; done
for f in "$W"/results/tables/FZ_own_s860?.npz "$W"/results/tables/CD_?_s8601.npz; do ln -sf "$f" "$O/tables/"; done
for f in "$W"/results/models/*; do ln -sf "$f" "$O/models/"; done
for f in "$W"/results/pop/*; do ln -sf "$f" "$O/pop/"; done
ln -sf "$W/results/an/stageC_s8601.json" "$O/an/wl94_stageC_s8601.json"
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null) protocol=$(sha256sum protocol/ecro_mat94r.json | cut -c1-16) W94=$W R86=$R86 R92A=$R92A"
ASPP=4096; [ -n "$DEV" ] && ASPP=32
RP="$PY r94.py"; MT="$PY $MT94/mat94.py"; C="$PY $MT94/c94.py"

run A_repro "$RP repro $R $R86 $R92A $DEV"
PS=$(grep '^REPRO' $O/logs/A_repro.log | tail -1 | cut -d' ' -f2); log "REPRO $PS"
[ "$PS" = PASS ] || { log "STOP REPRO"; log CHAIN_DONE; exit 0; }

evalseed() {   # evalseed <seed> <cand...>: tables, popeval, renders for the listed candidates
  local s=$1; shift; local j=() jj=()
  for c in "$@"; do
    j+=(E_tab_${c}_s$s "$MT tables $R $R92A cand:${c}_s$s CD_${c}_s$s $DEV" E_pop_${c}_s$s "$C popeval $R $R86 $R92A cand:${c}_s$s P_${c}_s$s $DEV")
  done
  [ -e "$O/pop/P_own_s$s.npz" ] || j+=(E_pop_own_s$s "$C popeval $R $R86 $R92A own:$s P_own_s$s $DEV")
  pool 3 "${j[@]}"
  for c in "$@"; do
    AR="C0 C2_srcS1"; [ "$c" = Ar ] && AR="C0"
    for v in none U07 U03 bx75 bx38 bx38i; do
      L=$O/lists/r_${c}_s${s}_$v.txt; : > "$L"
      for a in $AR; do for r in 0 1; do
        echo "$v results/tables/CD_${c}_s$s.npz:$v|B_$a $((86600+r)) $ASPP CK_${v}_${c}_s${s}_${a}_CK2_T1_r$r" >> "$L"; done; done
      jj+=(E_ren_${c}_s${s}_$v "$MT render $R $L $DEV")
    done
  done
  pool 3 "${jj[@]}"
}
pool 2 T_C_s8601 "$RP train $R $R86 $R92A 8601 C $DEV" T_Ar_s8601 "$RP train $R $R86 $R92A 8601 Ar $DEV"
evalseed 8601 C Ar
run N_an_s8601 "$RP an $R 8601"
NX=$(grep '^RAN' $O/logs/N_an_s8601.log | tail -1 | awk '{print $4}'); log "RUN_NEXT_SEED $NX"
if [ "$NX" = True ]; then
  pool 2 T_A_s8602 "$RP train $R $R86 $R92A 8602 A $DEV" T_C_s8602 "$RP train $R $R86 $R92A 8602 C $DEV"
  evalseed 8602 A C
  run N_an_s8602 "$RP an $R 8602"
fi
run Z_final "$RP final $R $O/eval94a"
log CHAIN_DONE
