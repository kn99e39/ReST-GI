#!/bin/bash
# WL91 chain: ONE detached chain, end to end.   setsid nohup bash run91.sh <R91> <R89> <R90> [--dev] > <R91>/chain.out 2>&1 &
set -u
R=$1; R89=$2; R90=$3; DEV=${4:-}
. "$R/env.sh"
CODE=$R/code/experiments/gm1_c3urx1_91
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
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null) R89=$R89 R90=$R90"
ESPP=4096; OSPP=2048
if [ -n "$DEV" ]; then ESPP=64; OSPP=64; export WL89_DEV=1 WL90_DEV=1 WL91_DEV=1; fi
cp "$R89/results/w_choice.json" "$O/w_choice.json"
log "W1 $($PY -c "import json;print(json.load(open('$O/w_choice.json'))['chosen'])") (WL89 choice, sha256 $(sha256sum $O/w_choice.json | cut -c1-16))"
TAGS="W1_T0 W1_T1 W2_T0 W2_T1 W1_T1_LMIX89 W1_T1_LSPOT89"
split() { base=${1%_LMIX89}; base=${base%_LSPOT89}; lsopt=""; [ "$1" != "$base" ] && lsopt="--ls=${1#${base}_}"; }
# ---- tests, new truth order fields, tables
j=(S0_tests "$PY tests91.py $R $R89 $R90 $DEV")
for ls in LMIX89 LSPOT89; do j+=(ORD_W1_T1_$ls "$PY run91.py orders $R W1_T1 91460 $OSPP --ls=$ls"); done
for t in $TAGS; do split $t; j+=(TAB_$t "$PY run91.py tabs $R $base $lsopt"); done
pool 3 "${j[@]}"
TL=$(grep '^TESTS' $O/logs/S0_tests.log | tail -1 | cut -d' ' -f2)
log "GATE TESTS $TL"
[ "$TL" = PASS ] || [ -n "$DEV" ] || { log "STOP STOPPED_INVALID_COMPARISON"; log CHAIN_DONE; exit 0; }
# ---- renders (WL89 render seeds: common random numbers)
SEEDS="86001 89201 89202 89203 89204 89205 89206 89207 89208 89209 89210 89211 89212 89213 89214 89215 89216"
[ -n "$DEV" ] && SEEDS="86001 89201 89202"
j=()
for t in $TAGS; do
  split $t
  for h in 0 1; do
    L=$O/lists/r_${t}_$h.txt; : > "$L"
    i=0
    for sd in $SEEDS; do
      if [ $((i % 2)) = $h ]; then for a in R0 C3UR C3URX1; do for r in 0 1; do
        echo "results/tables/T_$t.npz:${a}_s$sd $((89600+r)) $ESPP X91_${a}_${t}_s${sd}_r$r" >> "$L"
      done; done; fi
      i=$((i+1))
    done
    if [ $h = 1 ]; then for r in 0 1; do
      echo "results/tables/T_$t.npz:R0X1_s86001 $((89600+r)) $ESPP X91_R0X1_${t}_s86001_r$r" >> "$L"
      echo "results/tables/T_$t.npz:C3UR_POOL $((89600+r)) $ESPP X91P_C3UR_POOL_${t}_r$r" >> "$L"
      echo "results/tables/T_$t.npz:C3URX1_POOL $((89600+r)) $ESPP X91P_C3URX1_POOL_${t}_r$r" >> "$L"
      echo "$R89/results/fields/F89_$t.npz:B $((89600+r)) $ESPP X91_ORACLE_${t}_r$r" >> "$L"
      if [ "$t" = W1_T1 ]; then for k in 4 16; do for a in C3UR C3URX1; do
        echo "results/tables/T_$t.npz:${a}_k$k $((89600+r)) $ESPP X91K_${a}_k${k}_${t}_r$r" >> "$L"
      done; done; fi
    done; fi
    j+=(RR_${t}_$h "$PY run91.py render $R $base $L $lsopt")
  done
done
pool 6 "${j[@]}"
# ---- end-to-end economics (alone): cold lifecycles in fresh processes, warm lifecycle, ladders, scaling
for a in C3URX1 C3UR R0 PT; do run S3_cold_$a "$PY e2e91.py cold $R $a"; done
run S3_life "$PY e2e91.py life $R $DEV"
run S3_ladder "$PY e2e91.py ladder $R $R89 $DEV"
run S3_scale "$PY e2e91.py scale $R $DEV"
# ---- analysis, visuals
run A_final "$PY analyze91.py $R $R89 $R90"
run V_viz "$PY viz91.py $R $R89 $O/viz"
log CHAIN_DONE
