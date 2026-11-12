#!/bin/bash
# WL90 chain: ONE detached chain, end to end.   setsid nohup bash run90.sh <R90> <R89> [--dev] > <R90>/chain.out 2>&1 &
set -u
R=$1; R89=$2; DEV=${3:-}
. "$R/env.sh"
CODE=$R/code/experiments/gm1_gap90
O=$R/results; ST=$R/chain.status
mkdir -p "$O/logs" "$O/renders" "$O/lists" "$O/tables" "$O/fields" "$O/viz" "$O/econ"
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
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null) R89=$R89"
ESPP=4096; OSPP=2048
if [ -n "$DEV" ]; then ESPP=64; OSPP=64; export WL89_DEV=1 WL90_DEV=1; fi
cp "$R89/results/w_choice.json" "$O/w_choice.json"
log "W1 $($PY -c "import json;print(json.load(open('$O/w_choice.json'))['chosen'])") (WL89 choice, sha256 $(sha256sum $O/w_choice.json | cut -c1-16))"
CFGS="W1_T1 W1_T0 W2_T1"; FCFGS="W1_T1 W1_T0"
# ---- tests, missing truth order field, pooled exact-read tables, feasibility tables
j=(S0_tests "$PY tests90.py $R $R89 $DEV")
[ -f "$R89/results/fields/ORD_W2_T1.npz" ] || j+=(ORD_W2_T1 "$PY run90.py orders $R W2_T1 90460 $OSPP")
[ -f "$R89/results/fields/F89_W2_T1.npz" ] || j+=(F90_W2_T1 "$PY run90.py field $R W2_T1 90450 4096 F90_W2_T1")
for c in $CFGS; do j+=(XTAB_$c "$PY run90.py xtab $R $c"); done
pool 3 "${j[@]}"
TL=$(grep '^TESTS' $O/logs/S0_tests.log | tail -1 | cut -d' ' -f2)
log "GATE TESTS $TL"
[ "$TL" = PASS ] || [ -n "$DEV" ] || { log "STOP STOPPED_INVALID_COMPARISON"; log CHAIN_DONE; exit 0; }
j=()
for c in $FCFGS; do j+=(FEAS_$c "$PY run90.py feas $R $c"); done
pool 2 "${j[@]}"
# ---- renders (WL89 render seeds: common random numbers)
XARMS="M0 M1 M2 M3 M4 axsx a1sx axs1 a1s1 a4sx a16sx axs4 axs16 a4s4 a16s16 P1only P2plus EA1 SE1"
SEEDS="86001 89201 89202 89203 89204 89205 89206 89207 89208 89209 89210 89211 89212 89213 89214 89215 89216"
[ -n "$DEV" ] && SEEDS="86001 89201 89202"
j=()
for c in $CFGS; do
  L1=$O/lists/x_$c.txt; : > "$L1"
  for a in $XARMS; do
    key=$a; case $a in a*s*) key=X_$a;; esac
    for r in 0 1; do echo "results/tables/X_$c.npz:$key $((89600+r)) $ESPP X90_${a}_${c}_r$r" >> "$L1"; done
  done
  ORA=$R89/results/fields/F89_$c.npz; [ -f "$ORA" ] || ORA=$O/fields/F90_$c.npz
  for r in 0 1; do echo "$ORA:B $((89600+r)) $ESPP X90_ORACLE_${c}_r$r" >> "$L1"; done
  j+=(RX_$c "$PY run90.py render $R $c $L1")
done
for c in $FCFGS; do
  for h in 0 1; do
    L2=$O/lists/f_${c}_$h.txt; : > "$L2"
    i=0
    for sd in $SEEDS; do
      if [ $((i % 2)) = $h ]; then for a in R0 R0X1 C3UR C3URX1; do for r in 0 1; do
        echo "results/tables/F_$c.npz:${a}_s$sd $((89600+r)) $ESPP X90F_${a}_${c}_s${sd}_r$r" >> "$L2"
      done; done; fi
      i=$((i+1))
    done
    j+=(RF_${c}_$h "$PY run90.py render $R $c $L2")
  done
done
pool 6 "${j[@]}"
# ---- costs (alone), analysis, visuals
run S3_econ "$PY econ90.py $R $DEV"
run A_final "$PY analyze90.py $R $R89"
run V_viz "$PY viz90.py $R $R89 $O/viz"
log CHAIN_DONE
