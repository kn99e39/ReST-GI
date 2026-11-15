#!/bin/bash
# WL96A chain (ONE detached chain, end to end): links (read-only) -> Stage 0 -> explicit tails -> renders -> Stage 1 analysis
# [-> predeclared extension: 6 field halves, tails, 12-rep renders, analysis] [-> Stage 2 only on S1_HEADROOM]
# -> Stage 3 lifecycle (ALONE on the GPU) [-> scaling only on 'COST EXTEND'] -> verdict + figures.
#   setsid nohup bash run96a.sh <R> <W94> <R95A> <R92A> [--dev] > <R>/chain.out 2>&1 &
set -u
R=$1; W=$2; R95A=$3; R92A=$4; DEV=${5:-}
. "$R/env.sh"
CODE=$R/code/experiments/ecro_util96a; MT94=$R/code/experiments/ecro_mat94
O=$R/results; ST=$R/chain.status
mkdir -p "$O/logs" "$O/renders" "$O/lists" "$O/tables" "$O/pre" "$O/fields" "$O/fit"
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
( while true; do echo "$(date -u +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)" >> "$O/logs/gpu.txt"; sleep 5; done ) & GP=$!; disown $GP
trap 'kill $GP 2>/dev/null' EXIT
ln -sfn "$W/data" "$R/data"
for f in "$W"/results/fields/CKF_none.*; do ln -sf "$f" "$O/fields/"; done
cp -n "$W/results/pre/masks94.npz" "$O/pre/"
for f in "$W"/results/renders/REF_ER_CK_none_*; do ln -sf "$f" "$O/renders/"; done
for f in "$W"/results/tables/FZ_own_s860?.npz; do ln -sf "$f" "$O/tables/"; done
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null) protocol=$(sha256sum protocol/ecro_util96a.json | cut -c1-16)"
A="$PY w96a.py"; MT="$PY $MT94/mat94.py"
renders() {
  local tag=$1 lst=$2; local J=()
  split -n l/6 -d "$lst" "$O/lists/${tag}_"
  for f in "$O"/lists/${tag}_*; do J+=(R_${tag}_$(basename $f) "$MT render $R $f $DEV"); done
  pool 4 "${J[@]}"                                  # probe: one 4,096-spp render uses ~23 % GPU
}
run S0 "$A stage0 $R $R95A $DEV"
S0=$(grep '^STAGE0' $O/logs/S0.log | tail -1 | cut -d' ' -f2); log "STAGE0 $S0"
[ -z "$DEV" ] && [ "$S0" != PASS ] && { log "STOP STAGE0"; log CHAIN_DONE; exit 0; }
run T1 "$A tails $R $DEV"
renders p "$O/lists/r96a.txt"
run N1 "$A an $R $DEV"
S1=$(grep '^S1 ' $O/logs/N1.log | tail -1 | cut -d' ' -f2); log "S1 $S1"
if grep -q '^AN EXTEND' $O/logs/N1.log; then
  log "EXTEND (rule: S1_UNRESOLVED)"
  pool 3 FX0 "$A fext $R 0 $DEV" FX1 "$A fext $R 1 $DEV" FX2 "$A fext $R 2 $DEV" FX3 "$A fext $R 3 $DEV" FX4 "$A fext $R 4 $DEV" FX5 "$A fext $R 5 $DEV"   # CPU-bound accumulation (probe ~3 % GPU)
  run T2 "$A tails $R --ext $DEV"
  renders e "$O/lists/r96a_ext.txt"
  run N2 "$A an $R --ext $DEV"
  S1=$(grep '^S1 ' $O/logs/N2.log | tail -1 | cut -d' ' -f2); log "S1_EXT $S1"
fi
if [ "$S1" = S1_HEADROOM ] || [ -n "$DEV" ]; then                # dev exercises the Stage 2 path regardless
  log "STAGE2 (rule: S1_HEADROOM)"
  pool 4 F2a "$A s2ref $R 0 40 $DEV" F2b "$A s2ref $R 40 80 $DEV" F2c "$A s2ref $R 80 120 $DEV" F2d "$A s2ref $R 120 160 $DEV"
  run F2fit "$A s2fit $R $DEV"
  renders s2 "$O/lists/r96a_s2.txt"
else
  log "STAGE2_NOT_TRIGGERED"
fi
run C1 "$A cost $R $R92A 1.0 $DEV"
if grep -q '^COST EXTEND' $O/logs/C1.log || [ -n "$DEV" ]; then   # dev exercises the scaling path regardless
  log "COST EXTEND (rule: U_ZERO >= 1.15)"
  run C15 "$A cost $R $R92A 1.5 $DEV"
  run C20 "$A cost $R $R92A 2.0 $DEV"
fi
run FIN "$A final $R $O/eval96a $DEV"
log "$(grep '^FINAL' $O/logs/FIN.log | tail -1)"
log CHAIN_DONE
