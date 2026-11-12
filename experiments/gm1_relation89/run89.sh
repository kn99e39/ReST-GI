#!/bin/bash
# WL89 chain: ONE detached chain covering every stage end to end (including the stop branches).
#   setsid nohup bash run89.sh <R89> <R87> [--dev] > <R89>/chain.out 2>&1 &
# Render / table stages run as concurrent processes; the economics and scaling stages run alone on the GPU.
set -u
R=$1; R87=$2; DEV=${3:-}
. "$R/env.sh"
CODE=$R/code/experiments/gm1_relation89
O=$R/results; ST=$R/chain.status
mkdir -p "$O/logs" "$O/renders" "$O/lists" "$O/tables" "$O/fields" "$O/viz"
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
finish() { run A_verdict "$PY analyze89.py verdict $R"; run V_viz "$PY viz89.py $R $O/viz"; log CHAIN_DONE; exit 0; }
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null)"
RSPP=8192; NREF=8; NS=4; NP=4; FSPP=4096; OSPP=2048; ESPP=4096
if [ -n "$DEV" ]; then RSPP=64; NREF=2; NS=2; NP=2; FSPP=64; OSPP=64; ESPP=64; export WL89_DEV=1; fi
# ================================================================ Stage 0: geometry screening, qualification partition (8 x 8192 per T)
run S0_wscreen "$PY run89.py wscreen $R"
CANDS=$(grep '^W_GEOMETRY_ADMISSIBLE' "$O/logs/S0_wscreen.log" | cut -d' ' -f2-)
[ -n "$DEV" ] && [ -z "$CANDS" ] && CANDS=Wk
log "GEOMETRY_ADMISSIBLE $CANDS"
W1=""; n=0
for c in $CANDS; do
  n=$((n+1)); [ $n -gt 3 ] && break
  j=()
  for r in $(seq 0 $((NREF-1))); do for t in T0 T1; do
    j+=(QER_${c}_${t}_r$r "$PY run89.py ref $R ${c}_$t ER $((89140+r)) $RSPP REF_ERQ_${c}_${t}_r$r")
  done; done
  pool 4 "${j[@]}"
  run S0_wqualify_$c "$PY analyze89.py wqualify $R $c"
  if grep -q "WQUALIFY $c ADMITTED" "$O/logs/S0_wqualify_$c.log" || [ -n "$DEV" ]; then W1=$c; break; fi
done
if [ -z "$W1" ]; then log "STOP WORKLOAD_UNQUALIFIED"; finish; fi
run S0_wchoose "$PY run89.py wchoose $R $W1"
log "W1 $W1"
# ================================================================ Stage 0: tests, evaluation references, truth fields
j=(S0_tests "$PY tests89.py $R $R87 $DEV")
for r in $(seq 0 $((NREF-1))); do for t in T0 T1; do
  j+=(REF_ER_W1_${t}_r$r "$PY run89.py ref $R W1_$t ER $((89100+r)) $RSPP REF_ER_W1_${t}_r$r")
  j+=(REF_ER_W2_${t}_r$r "$PY run89.py ref $R W2_$t ER $((89180+r)) $RSPP REF_ER_W2_${t}_r$r")
done; done
for r in $(seq 0 $((NS-1))); do for ls in LMIX89 LSPOT89; do
  j+=(REF_ER_W1_T1_${ls}_r$r "$PY run89.py ref $R W1_T1 ER $((89160+r)) $RSPP REF_ER_W1_T1_${ls}_r$r --ls=$ls")
done; done
for r in $(seq 0 $((NP-1))); do j+=(REF_P_W1_T1_r$r "$PY run89.py ref $R W1_T1 P $((89120+r)) $RSPP REF_P_W1_T1_r$r"); done
for c in W1_T0 W1_T1 W2_T0 W2_T1; do j+=(F89_$c "$PY run89.py field $R $c 89450 $FSPP F89_$c"); done
for ls in LMIX89 LSPOT89; do j+=(F89_W1_T1_$ls "$PY run89.py field $R W1_T1 89450 $FSPP F89_W1_T1_$ls --ls=$ls"); done
for c in W1_T0 W1_T1; do j+=(ORD_$c "$PY run89.py orders $R $c 89460 $OSPP"); done
pool 4 "${j[@]}"
TL=$(grep '^TESTS' $O/logs/S0_tests.log | tail -1 | cut -d' ' -f2)
log "GATE TESTS $TL"
[ "$TL" = PASS ] || [ -n "$DEV" ] || { log "STOP STOPPED_INVALID_COMPARISON"; finish; }
# ================================================================ tables and realizations
j=()
for c in W1_T0 W1_T1 W2_T0 W2_T1; do j+=(TAB_$c "$PY run89.py tables $R $c"); done
for ls in LMIX89 LSPOT89; do j+=(TAB_W1_T1_$ls "$PY run89.py tables $R W1_T1 --ls=$ls"); done
pool 2 "${j[@]}"
j=()
for c in W1_T0 W1_T1; do j+=(REAL_$c "$PY run89.py real $R $c"); done
pool 1 "${j[@]}"
# ================================================================ renders (common random numbers across arms)
ARMS="R0 C3UR R1 R1M C3UR16 POOL2 C3 C3U RS0 RS1"
HARMS="R0 C3UR R1 R1M C3"
RARMS="R0_POOL16 R1_POOL16 C3UR_POOL16 R0_INF R1_INF C3UR_INF"
SEEDS2="89201 89202 89203 89204 89205 89206 89207 89208 89209 89210 89211 89212 89213 89214 89215 89216"
j=()
for c in W1_T0 W1_T1 W2_T0 W2_T1 W1_T1_LMIX89 W1_T1_LSPOT89; do
  case $c in W1_T0|W1_T1) AL=$ARMS;; *) AL=$HARMS;; esac
  base=${c%_LMIX89}; base=${base%_LSPOT89}; lsopt=""; [ "$c" != "$base" ] && lsopt="--ls=${c#${base}_}"
  L1=$O/lists/s1_$c.txt; : > "$L1"
  for a in $AL; do for r in 0 1; do echo "results/tables/X_$c.npz:$a $((89600+r)) $ESPP X89_${a}_${c}_r$r" >> "$L1"; done; done
  for r in 0 1; do echo "results/fields/F89_$c.npz:B $((89600+r)) $ESPP X89_ORACLE_${c}_r$r" >> "$L1"; done
  j+=(RS1_$c "$PY run89.py render $R $base $L1 $lsopt")
done
for c in W1_T0 W1_T1; do
  for h in 0 1 2; do
    L2=$O/lists/s2_${c}_$h.txt; : > "$L2"
    i=0
    for sd in $SEEDS2; do
      if [ $((i % 3)) = $h ]; then for r in 0 1; do for a in R0 R1 C3UR; do
        echo "results/tables/R_$c.npz:${a}_s$sd $((89600+r)) $ESPP X89S_${a}_${c}_s${sd}_r$r" >> "$L2"
      done; done; fi
      i=$((i+1))
    done
    [ $h = 0 ] && for a in $RARMS; do for r in 0 1; do echo "results/tables/R_$c.npz:$a $((89600+r)) $ESPP X89R_${a}_${c}_r$r" >> "$L2"; done; done
    j+=(RS2_${c}_$h "$PY run89.py render $R $c $L2")
  done
done
pool 6 "${j[@]}"
# ================================================================ analysis 1 (validity gate), economics + scaling (alone), analysis 2 / 3
run A_stage1 "$PY analyze89.py stage1 $R"
grep -q '"tests_pass": true, "finite_nonneg": true, "target_identity": true, "raysend_repaired": true' "$O/logs/A_stage1.log" || [ -n "$DEV" ] || { log "STOP STOPPED_INVALID_COMPARISON (stage-1 validity)"; finish; }
run S3_econ "$PY econ89.py econ $R $DEV"
run S3_scale "$PY econ89.py scale $R $DEV"
run A_stage2 "$PY analyze89.py stage2 $R"
run A_stage3 "$PY analyze89.py stage3 $R"
finish
