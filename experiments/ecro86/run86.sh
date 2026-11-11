#!/bin/bash
# WL86 chain: ONE detached chain, all stages and gates. setsid nohup bash run86.sh <root> [--dev] > <root>/chain.out 2>&1 &
# Dr.Jit launches are Python-tracing bound, so render / data stages run as MAXJ concurrent processes; the economics stage runs alone.
set -u
ROOT=$1; DEV=${2:-}
. "$ROOT/env.sh"
CODE=$ROOT/code/experiments/ecro86
R=$ROOT/results; ST=$ROOT/chain.status
mkdir -p "$R/logs" "$R/renders"
MAXJ=${MAXJ:-6}
log() { echo "$(date -u +%FT%TZ) $*" >> "$ST"; }
pool() {
  local maxj=$1; shift; local names=()
  while [ $# -gt 0 ]; do
    local name=$1 cmd=$2; shift 2
    if [ -n "${RESUME:-}" ] && [ "$(cat "$R/logs/$name.rc" 2>/dev/null)" = 0 ]; then names+=("$name"); continue; fi   # resume: completed job kept
    while [ "$(jobs -rp | wc -l)" -ge "$maxj" ]; do sleep 1; done
    log "START $name"
    ( bash -c "$cmd" > "$R/logs/$name.log" 2>&1; echo $? > "$R/logs/$name.rc" ) &
    names+=("$name")
  done
  wait
  local bad=0
  for n in "${names[@]}"; do local rc; rc=$(cat "$R/logs/$n.rc" 2>/dev/null || echo 99); log "END $n rc=$rc"; [ "$rc" != 0 ] && bad=1; done
  [ $bad -ne 0 ] && { log CHAIN_FAILED; exit 1; }
}
run() { local n=$1; shift; pool 1 "$n" "$*"; }
last() { grep -v -i warn "$R/logs/$1.log" | tail -1 | cut -d' ' -f1; }
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $ROOT/code/COMMIT 2>/dev/null)"
SPP=8192; NER=8; NP=8; NSRC=4; FSPP=4096; ESPP=4096; GTR=1024; GVAL=64; GHQ=32; SH=8; STEPS=""
if [ -n "$DEV" ]; then SPP=128; NER=2; NP=2; NSRC=2; FSPP=128; ESPP=128; GTR=24; GVAL=8; GHQ=8; SH=4; STEPS="--steps=600"; fi
CFGS="E1_T0 E1_T1 E2_T0 E2_T1"
SEEDS="8601 8602"
# ---- Stage 0: geometric masks; validation tests and references (concurrent)
if [ -z "${RESUME:-}" ]; then
run S0_masks "$PY masks86.py $ROOT E1 && $PY masks86.py $ROOT E2 && $PY masks86.py $ROOT E1 --ls=LMIX && $PY masks86.py $ROOT E1 --ls=LSPOT"
j=(S0_tests "$PY tests86.py $ROOT $DEV")
for r in $(seq 0 $((NER-1))); do for c in $CFGS; do j+=(REF_ER_${c}_r$r "$PY pt86.py render $ROOT $c ER $((86100+r)) $SPP REF_ER_${c}_r$r"); done; done
for r in $(seq 0 $((NP-1))); do for c in $CFGS; do j+=(REF_P_${c}_r$r "$PY pt86.py render $ROOT $c P $((86200+r)) $SPP REF_P_${c}_r$r"); done; done
for r in $(seq 0 $((NSRC-1))); do for ls in LMIX LSPOT; do j+=(REF_ER_E1_T1_${ls}_r$r "$PY pt86.py render $ROOT E1_T1 ER $((86300+r)) $SPP REF_ER_E1_T1_${ls}_r$r --ls=$ls"); done; done
pool $MAXJ "${j[@]}"
else
  # amendment 1 (post hoc, labelled): Stage 0 renders are reused; the frozen STOP record stays in chain.status / qualify_frozen.json
  log "RESUME amendment=1 commit=$(cat $ROOT/code/COMMIT 2>/dev/null)"
  cp -n "$R/qualify.json" "$R/qualify_frozen.json"
  rm -f "$R/logs/S0_amend1.rc" "$R/logs/S0_qualify.rc" "$R/logs/S9_viz.rc"        # these are recomputed on resume
  run S0_amend1 "$PY tests86.py $ROOT --only=V3"
fi
run S0_qualify "$PY analyze86.py qualify $ROOT"
g=$(last S0_qualify); log "GATE WORKLOAD $g"
[ "$g" = ADMITTED ] || [ -n "$DEV" ] || { log "STOP $g"; run S9_viz "$PY viz86.py $ROOT"; log CHAIN_DONE; exit 0; }
# ---- Stage 1: higher-order identifiability and oracle-projection ceiling
j=()
for c in $CFGS; do j+=(F_$c "$PY pt86.py field $ROOT $c 86450 $FSPP F_$c"); done
pool 4 "${j[@]}"
j=()
for r in 0 1; do for c in $CFGS; do j+=(ORA_${c}_r$r "$PY ecro86.py render $ROOT $c oracle:F_$c oracle $((86400+r)) $ESPP ORA_${c}_r$r"); done; done
pool $MAXJ "${j[@]}"
run S1_stage1 "$PY analyze86.py stage1 $ROOT"
g=$(last S1_stage1); log "GATE STAGE1 $g"
[ "$g" = PASS ] || [ -n "$DEV" ] || { log "STOP $g"; run S9_viz "$PY viz86.py $ROOT"; log CHAIN_DONE; exit 0; }
# ---- Stage 2: response supervision and training
j=()
per=$((GTR / SH))
for s in $(seq 0 $((SH-1))); do j+=(GEN_train_$s "$PY gen86.py $ROOT train $((s*per)) $per"); done
pool $SH "${j[@]}"
j=()
pv=$((GVAL / 4)); ph=$((GHQ / 4))
for s in 0 1 2 3; do j+=(GEN_val_$s "$PY gen86.py $ROOT val $((s*pv)) $pv" GEN_trainhq_$s "$PY gen86.py $ROOT trainhq $((s*ph)) $ph"); done
pool $SH "${j[@]}"
j=()
for sd in $SEEDS; do j+=(TRAIN_s$sd "$PY train86.py $ROOT $sd $STEPS"); done
pool 2 "${j[@]}"
j=()
for sd in $SEEDS; do j+=(CONTRACTS_s$sd "$PY ecro86.py contracts $ROOT ecro_s$sd"); done
pool 2 "${j[@]}"
for sd in $SEEDS; do log "GATE CONTRACTS_s$sd $(grep CONTRACTS $R/logs/CONTRACTS_s$sd.log | tail -1 | cut -d' ' -f2)"; done
# ---- Stage 3: learned operator renders (common random numbers across T0 / T1 and across variants)
j=()
for sd in $SEEDS; do m=ecro_s$sd; for r in 0 1; do
  for c in $CFGS; do j+=(EC_${m}_ecro_${c}_r$r "$PY ecro86.py render $ROOT $c $m ecro $((86600+r)) $ESPP EC_${m}_ecro_${c}_r$r"); done
  for e in E1 E2; do for v in stale nog nos; do j+=(EC_${m}_${v}_${e}_T1_r$r "$PY ecro86.py render $ROOT ${e}_T1 $m $v $((86600+r)) $ESPP EC_${m}_${v}_${e}_T1_r$r"); done; done
  for ls in LMIX LSPOT; do j+=(EC_${m}_ecro_E1_T1_${ls}_r$r "$PY ecro86.py render $ROOT E1_T1 $m ecro $((86600+r)) $ESPP EC_${m}_ecro_E1_T1_${ls}_r$r --ls=$ls"); done
done; done
pool $MAXJ "${j[@]}"
run S3_final "$PY analyze86.py final $ROOT"
g=$(last S3_final); log "GATE STAGE3 $g"
# ---- Stage 4: economics (alone). Always measured; the verdict is gated by stage 3 in the report.
m=$($PY -c "import json;print(json.load(open('$R/final.json'))['economics_seed'])")
run S4_econ "$PY econ86.py $ROOT $m $DEV"
run S9_viz "$PY viz86.py $ROOT"
log CHAIN_DONE
