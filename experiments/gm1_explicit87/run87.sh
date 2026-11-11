#!/bin/bash
# WL87 chain: ONE detached chain covering every stage. setsid nohup bash run87.sh <R87> <R86> <R86A> [--dev] > <R87>/chain.out 2>&1 &
# Render / table stages run as concurrent processes (Dr.Jit launches are Python-tracing bound); the economics stage runs alone.
set -u
R=$1; R86=$2; R86A=$3; DEV=${4:-}
. "$R/env.sh"
CODE=$R/code/experiments/gm1_explicit87
O=$R/results; ST=$R/chain.status
mkdir -p "$O/logs" "$O/renders" "$O/lists" "$O/tables" "$O/fields"
MAXJ=${MAXJ:-6}
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
RSPP=8192; NREF=8; FSPP=4096; OSPP=2048; ESPP=4096
if [ -n "$DEV" ]; then RSPP=64; NREF=2; FSPP=128; OSPP=128; ESPP=64; export WL87_DEV_FORCE_E3=1; fi
CFGS="E1_T0 E1_T1 E2_T0 E2_T1 E3_T0 E3_T1"
SEEDS2="86001 87201 87202 87203 87204 87205 87206 87207 87208 87209 87210 87211 87212 87213 87214 87215 87216"
# ---- Stage 0: E3 choice (geometry only), validation tests, E3 references / oracle fields, per-order truth fields
run S0_e3select "$PY run87.py e3select $R $DEV"
if [ "$($PY -c "import json;print(json.load(open('$O/e3_choice.json'))['chosen'] is not None)")" != True ]; then CFGS="E1_T0 E1_T1 E2_T0 E2_T1"; NE3=1; else NE3=0; fi
log "E3_CHOICE $($PY -c "import json;print(json.load(open('$O/e3_choice.json'))['chosen'])")"
j=(S0_tests "$PY tests87.py $R $R86 $DEV")
[ $NE3 = 1 ] && NREF=0
for r in $(seq 0 $((NREF-1))); do for t in T0 T1; do
  j+=(REF_ER_E3_${t}_r$r "$PY run87.py ref $R E3_$t ER $((87100+r)) $RSPP REF_ER_E3_${t}_r$r")
  j+=(REF_P_E3_${t}_r$r "$PY run87.py ref $R E3_$t P $((87120+r)) $RSPP REF_P_E3_${t}_r$r")
done; done
[ $NE3 = 0 ] && for t in T0 T1; do j+=(F87_E3_$t "$PY run87.py field $R E3_$t 87450 $FSPP F87_E3_$t"); done
for c in $CFGS; do j+=(ORD_$c "$PY run87.py orders $R $c 87460 $OSPP"); done
pool $MAXJ "${j[@]}"
log "GATE TESTS $(grep '^TESTS' $O/logs/S0_tests.log | tail -1 | cut -d' ' -f2)"
grep -q '^TESTS PASS' "$O/logs/S0_tests.log" || [ -n "$DEV" ] || { log "STOP STOPPED_INVALID_COMPARISON"; log CHAIN_DONE; exit 0; }
[ $NE3 = 0 ] && run S0_e3qualify "$PY analyze87.py e3qualify $R"
log "E3 $(tail -1 $O/logs/S0_e3qualify.log | cut -d' ' -f2)"
# ---- Stage 1: explicit tables + controls on the deployed state; Stage 2 realization tables
j=()
for c in $CFGS; do j+=(TAB_$c "$PY run87.py tables $R $R86 $c"); done
for ls in LMIX LSPOT; do j+=(TAB_E1_T1_$ls "$PY run87.py tables $R $R86 E1_T1 --ls=$ls"); done
pool 4 "${j[@]}"
j=()
for c in $CFGS; do j+=(REAL_$c "$PY run87.py real $R $R86 $c"); done
pool 3 "${j[@]}"
# ---- render lists (Stage 1 arms, E3 oracle, Stage 2 realizations)
j=()
for c in $CFGS; do
  L1=$O/lists/s1_$c.txt; : > "$L1"
  for a in EXP BINQ ENC RAYSEND C3; do for r in 0 1; do echo "results/tables/X_$c.npz:$a $((86600+r)) $ESPP X87_${a}_${c}_r$r" >> "$L1"; done; done
  case $c in E3_*) for r in 0 1; do echo "results/fields/F87_$c.npz:B $((86400+r)) $ESPP ORA87_${c}_r$r" >> "$L1"; done;; esac
  j+=(RS1_$c "$PY run87.py render $R $R86 $c $L1")
  for h in 0 1; do
    L2=$O/lists/s2_${c}_$h.txt; : > "$L2"
    i=0
    for sd in $SEEDS2; do
      if [ $((i % 2)) = $h ]; then for r in 0 1; do
        echo "results/tables/R_$c.npz:EXP_s$sd $((86600+r)) $ESPP X87R_EXP_${c}_s${sd}_r$r" >> "$L2"
        echo "results/tables/R_$c.npz:ECRO_ecro_s8601_s$sd $((86600+r)) $ESPP X87R_ECRO_${c}_s${sd}_r$r" >> "$L2"
      done; fi
      i=$((i+1))
    done
    [ $h = 0 ] && for r in 0 1; do echo "results/tables/R_$c.npz:EXP_POOL $((86600+r)) $ESPP X87R_POOL_${c}_r$r" >> "$L2"; done
    j+=(RS2_${c}_$h "$PY run87.py render $R $R86 $c $L2")
  done
done
for ls in LMIX LSPOT; do
  L1=$O/lists/s1_E1_T1_$ls.txt; : > "$L1"
  for a in EXP BINQ ENC RAYSEND C3; do for r in 0 1; do echo "results/tables/X_E1_T1_$ls.npz:$a $((86600+r)) $ESPP X87_${a}_E1_T1_${ls}_r$r" >> "$L1"; done; done
  j+=(RS1_E1_T1_$ls "$PY run87.py render $R $R86 E1_T1 $L1 --ls=$ls")
done
[ $NE3 = 0 ] && for sd in 8601 8602; do for r in 0 1; do
  for t in T0 T1; do j+=(EC87_s${sd}_ecro_E3_${t}_r$r "$PY run87.py ecro $R $R86 E3_$t ecro_s$sd ecro $((86600+r)) $ESPP EC87_ecro_s${sd}_ecro_E3_${t}_r$r"); done
  j+=(EC87_s${sd}_stale_E3_T1_r$r "$PY run87.py ecro $R $R86 E3_T1 ecro_s$sd stale $((86600+r)) $ESPP EC87_ecro_s${sd}_stale_E3_T1_r$r")
done; done
pool $MAXJ "${j[@]}"
run A_stage1 "$PY analyze87.py stage1 $R $R86 $R86A"
log "GATE STAGE1 $(grep '^STAGE1' $O/logs/A_stage1.log | tail -1 | cut -d' ' -f2)"
run A_stage2 "$PY analyze87.py stage2 $R $R86 $R86A"
# ---- Stage 3: economics alone
run S3_econ "$PY econ87.py $R $R86 $DEV"
run A_stage3 "$PY analyze87.py stage3 $R $R86"
run A_verdict "$PY analyze87.py verdict $R"
run V_viz "$PY viz87.py $R $R86 $O/viz"
log CHAIN_DONE
