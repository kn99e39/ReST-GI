#!/bin/bash
# WL88 chain: ONE detached chain covering every stage end to end.
#   setsid nohup bash run88.sh <R88> <R86> <R87> [--dev] > <R88>/chain.out 2>&1 &
# Render / table stages run as concurrent processes; the economics and scaling stages run alone on the GPU.
set -u
R=$1; R86=$2; R87=$3; DEV=${4:-}
. "$R/env.sh"
CODE=$R/code/experiments/gm1_relation88
O=$R/results; ST=$R/chain.status
mkdir -p "$O/logs" "$O/renders" "$O/lists" "$O/tables" "$O/fields"
MAXJ=${MAXJ:-5}
log() { echo "$(date -u +%FT%TZ) $*" >> "$ST"; }
pool() {
  local maxj=$1; shift; local names=()
  while [ $# -gt 0 ]; do
    local name=$1 cmd=$2; shift 2
    if [ -n "${RESUME:-}" ] && [ "$(cat "$O/logs/$name.rc" 2>/dev/null)" = 0 ]; then names+=("$name"); continue; fi
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
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null)"
RSPP=8192; NREF=8; NQ=4; NP=4; NS=4; FSPP=4096; OSPP=2048; ESPP=4096
if [ -n "$DEV" ]; then RSPP=64; NREF=2; NQ=2; NP=2; NS=2; FSPP=64; OSPP=64; ESPP=64; export WL88_DEV=1; fi
# ================================================================ Stage 0: workload choice (geometry), qualification partition
run S0_wscreen "$PY run88.py wscreen $R"
CANDS=$(grep '^W_GEOMETRY_ADMISSIBLE' "$O/logs/S0_wscreen.log" | cut -d' ' -f2-)
[ -n "$DEV" ] && [ -z "$CANDS" ] && CANDS=Wk
log "GEOMETRY_ADMISSIBLE $CANDS"
W1=""
for c in $CANDS; do
  j=()
  for r in $(seq 0 $((NQ-1))); do for t in T0 T1; do
    j+=(QER_${c}_${t}_r$r "$PY run88.py ref $R ${c}_$t ER $((88140+r)) $RSPP REF_ERQ_${c}_${t}_r$r")
    j+=(QP_${c}_${t}_r$r "$PY run88.py ref $R ${c}_$t P $((88150+r)) $RSPP REF_PQ_${c}_${t}_r$r")
  done; done
  pool 4 "${j[@]}"
  run S0_wqualify_$c "$PY analyze88.py wqualify $R $c"
  if grep -q "WQUALIFY $c ADMITTED" "$O/logs/S0_wqualify_$c.log" || [ -n "$DEV" ]; then W1=$c; break; fi
done
if [ -z "$W1" ]; then log "STOP STOPPED_WORKLOAD_UNQUALIFIED"; fi
[ -n "$W1" ] && run S0_wchoose "$PY run88.py wchoose $R $W1"
log "W1 ${W1:-none}"
# ================================================================ Stage 0: tests, references, truth fields
j=(S0_tests "$PY tests88.py $R $R87 $DEV")
if [ -n "$W1" ]; then
  for r in $(seq 0 $((NREF-1))); do for t in T0 T1; do
    j+=(REF_ER_W1_${t}_r$r "$PY run88.py ref $R W1_$t ER $((88100+r)) $RSPP REF_ER_W1_${t}_r$r")
    j+=(REF_ER_W2_${t}_r$r "$PY run88.py ref $R W2_$t ER $((88180+r)) $RSPP REF_ER_W2_${t}_r$r")
  done; done
  for r in $(seq 0 $((NS-1))); do for ls in LMIX88 LSPOT88; do
    j+=(REF_ER_W1_T1_${ls}_r$r "$PY run88.py ref $R W1_T1 ER $((88160+r)) $RSPP REF_ER_W1_T1_${ls}_r$r --ls=$ls")
  done; done
  for r in $(seq 0 $((NP-1))); do for c in W1_T1 W2_T1; do
    j+=(REF_P_${c}_r$r "$PY run88.py ref $R $c P $((88120+r)) $RSPP REF_P_${c}_r$r")
  done; done
  for c in W1_T0 W1_T1 W2_T0 W2_T1; do j+=(F88_$c "$PY run88.py field $R $c 88450 $FSPP F88_$c"); done
  for ls in LMIX88 LSPOT88; do j+=(F88_W1_T1_$ls "$PY run88.py field $R W1_T1 88450 $FSPP F88_W1_T1_$ls --ls=$ls"); done
  for c in W1_T0 W1_T1 W2_T1; do j+=(ORD_$c "$PY run88.py orders $R $c 88460 $OSPP"); done
fi
for c in E2_T0 E2_T1; do j+=(F88_$c "$PY run88.py field $R $c 88450 $FSPP F88_$c"); j+=(ORD_$c "$PY run88.py orders $R $c 88460 $OSPP"); done
pool 4 "${j[@]}"
log "GATE TESTS $(grep '^TESTS' $O/logs/S0_tests.log | tail -1 | cut -d' ' -f2)"
grep -q '^TESTS PASS' "$O/logs/S0_tests.log" || [ -n "$DEV" ] || { log "STOP STOPPED_INVALID_COMPARISON"; log CHAIN_DONE; exit 0; }
# ================================================================ Stage 1 / 2 tables
WC=""; [ -n "$W1" ] && WC="W1_T0 W1_T1 W2_T0 W2_T1"
j=()
for c in $WC E2_T0 E2_T1; do j+=(TAB_$c "$PY run88.py tables $R $c"); done
[ -n "$W1" ] && for ls in LMIX88 LSPOT88; do j+=(TAB_W1_T1_$ls "$PY run88.py tables $R W1_T1 --ls=$ls"); done
pool 2 "${j[@]}"
j=()
RC="E2_T1"; [ -n "$W1" ] && RC="W1_T0 W1_T1 E2_T1"
for c in $RC; do j+=(REAL_$c "$PY run88.py real $R $c"); done
pool 1 "${j[@]}"
# ================================================================ renders (common random numbers across arms)
ARMS="R0 R1 R1M R1m1 R1m16 RS0 RS1 R0D R0F POOL2 POOL4 C3 C3U C3UR"
HARMS="R0 R1 R1M C3"
RARMS="R0_POOL16 R1_POOL16 R0_INF R1_INF C3U_INF"
SEEDS2="88201 88202 88203 88204 88205 88206 88207 88208 88209 88210 88211 88212 88213 88214 88215 88216"
j=()
for c in $WC E2_T0 E2_T1 ${W1:+W1_T1_LMIX88 W1_T1_LSPOT88}; do
  case $c in E2_*) S0=86600;; *) S0=88600;; esac
  case $c in W1_T0|W1_T1|E2_T0|E2_T1) AL=$ARMS;; *) AL=$HARMS;; esac
  base=${c%_LMIX88}; base=${base%_LSPOT88}; lsopt=""; [ "$c" != "$base" ] && lsopt="--ls=${c#${base}_}"
  L1=$O/lists/s1_$c.txt; : > "$L1"
  for a in $AL; do for r in 0 1; do echo "results/tables/X_$c.npz:$a $((S0+r)) $ESPP X88_${a}_${c}_r$r" >> "$L1"; done; done
  for r in 0 1; do echo "results/fields/F88_$c.npz:B $((S0+r)) $ESPP X88_ORACLE_${c}_r$r" >> "$L1"; done
  j+=(RS1_$c "$PY run88.py render $R $base $L1 $lsopt")
done
for c in $RC; do
  case $c in E2_*) S0=86600;; *) S0=88600;; esac
  for h in 0 1; do
    L2=$O/lists/s2_${c}_$h.txt; : > "$L2"
    i=0
    for sd in $SEEDS2; do
      if [ $((i % 2)) = $h ]; then for r in 0 1; do for a in R0 R1; do
        echo "results/tables/R_$c.npz:${a}_s$sd $((S0+r)) $ESPP X88S_${a}_${c}_s${sd}_r$r" >> "$L2"
      done; done; fi
      i=$((i+1))
    done
    [ $h = 0 ] && for a in $RARMS; do for r in 0 1; do echo "results/tables/R_$c.npz:$a $((S0+r)) $ESPP X88R_${a}_${c}_r$r" >> "$L2"; done; done
    j+=(RS2_${c}_$h "$PY run88.py render $R $c $L2")
  done
done
pool 6 "${j[@]}"
# ================================================================ analysis 1, economics + scaling (alone), analysis 2 / 3, verdict, visuals
run A_stage1 "$PY analyze88.py stage1 $R $R86"
if [ -n "$W1" ]; then
  run S3_econ "$PY econ88.py econ $R $DEV"
  run S3_scale "$PY econ88.py scale $R $DEV"
  run A_stage2 "$PY analyze88.py stage2 $R $R86"
  run A_stage3 "$PY analyze88.py stage3 $R"
  run A_verdict "$PY analyze88.py verdict $R"
fi
run V_viz "$PY viz88.py $R $R86 $O/viz"
log CHAIN_DONE
