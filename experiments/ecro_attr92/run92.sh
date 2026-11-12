#!/bin/bash
# WL92 chain: ONE detached chain, end to end (gate and conditional intervention included).
#   setsid nohup bash run92.sh <R> <R86> [--dev] > <R>/chain.out 2>&1 &
set -u
R=$1; R86=$2; DEV=${3:-}
. "$R/env.sh"
CODE=$R/code/experiments/ecro_attr92
O=$R/results; ST=$R/chain.status
mkdir -p "$O/logs" "$O/renders" "$O/lists" "$O/tables" "$O/fields" "$O/fits" "$O/masks" "$O/models" "$O/viz"
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
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null) R86=$R86"
SPP=4096; RSPP=8192; REPS="0 1 2 3 4 5 6 7"; SREPS="0 1 2 3"
if [ -n "$DEV" ]; then SPP=64; RSPP=64; REPS="0 1"; SREPS="0 1"; fi
HIST="E2_T1 E2_T0 E1_T1"
VARS="SUM-f8601 SUM-f8602 SUM-r1 SUM-r2 OWN-f8601 OWN-k16 OWN-r1"
# ---- S0 provenance, E4 choice (geometry only)
pool 2 S0_prov "$PY diag92.py prov $R $R86" S0_e4select "$PY interv92.py e4select $R $R86 $DEV"
# ---- S1 truth split, then local operator + fit data, then ladder tables
j=(); for t in $HIST; do j+=(S1_split_$t "$PY diag92.py split $R $R86 $t $DEV"); done
pool 3 "${j[@]}"
j=(S2_local "$PY diag92.py local $R $R86 $DEV"); for t in $HIST; do j+=(S2_fitdata_$t "$PY diag92.py fitdata $R $R86 $t $DEV"); done
pool 4 "${j[@]}"
j=(); for t in $HIST; do j+=(S3_tables_$t "$PY diag92.py tables $R $R86 $t $DEV"); done
pool 3 "${j[@]}"
run S3_tests "$PY tests92.py $R $R86 $DEV"
TL=$(grep '^TESTS' $O/logs/S3_tests.log | tail -1 | cut -d' ' -f2)
log "TESTS $TL"
# ---- S4 fixed-geometry fits
j=(); for t in $HIST; do for v in $VARS; do j+=(S4_fit_${t}_$v "$PY diag92.py fit $R $R86 $t $v $DEV"); done; done
pool 4 "${j[@]}"
# ---- S5 renders of every table (WL86 ECRO render seeds: common random numbers)
j=()
for t in $HIST; do
  L=$O/lists/r_$t.txt; : > "$L"
  for r in 0 1; do
    for k in ORA2 L_RS L_EXP L_ENCQ L_ENCB L_R16 P_EXP P_ENCB P_R16 X_EXP B_ecro_s8601 B_ecro_s8602 SL_ecro_s8601 SL_ecro_s8602 LG_ecro_s8601 LG_ecro_s8602; do
      echo "results/tables/T_$t.npz:$k $((86600+r)) $SPP X92_${k}_${t}_r$r" >> "$L"
    done
    for v in $VARS; do for k in B SL; do
      echo "results/fits/fit_${t}_$v.npz:$k $((86600+r)) $SPP X92F_${v}_${k}_${t}_r$r" >> "$L"
    done; done
  done
  j+=(S5_render_$t "$PY diag92.py render $R $R86 $t $L")
done
pool 3 "${j[@]}"
run S5_gate "$PY analyze92.py gate $R $R86"
GL=$(grep '^GATE' $O/logs/S5_gate.log | tail -1 | cut -d' ' -f2)
log "GATE $GL (tests $TL)"
if [ "$GL" = ADMITTED ] && [ "$TL" != PASS ] && [ -z "$DEV" ]; then GL=STOP_INTERVENTION_TESTS; log "GATE $GL"; fi
# ---- conditional intervention
if [ "$GL" = ADMITTED ] || [ -n "${FORCE_INTERVENTION:-}" ]; then
  log "INTERVENTION_START"
  if [ -n "$DEV" ]; then SH="0:24 24:24"; else SH="0:128 128:128 256:128 384:128 512:128 640:128 768:128 896:128"; fi
  j=(); for s in $SH; do j+=(I1_gen_train_${s%%:*} "$PY interv92.py gen $R $R86 train ${s%%:*} ${s##*:} $DEV"); done
  if [ -n "$DEV" ]; then j+=(I1_gen_val "$PY interv92.py gen $R $R86 val 0 8 $DEV" I1_gen_trainhq "$PY interv92.py gen $R $R86 trainhq 0 4 $DEV")
  else j+=(I1_gen_val "$PY interv92.py gen $R $R86 val 0 64" I1_gen_trainhq "$PY interv92.py gen $R $R86 trainhq 0 32"); fi
  pool 8 "${j[@]}"
  F2OK=$($PY -c "
import glob, json
rows = [r for f in glob.glob('$R/data/train_f2/check_*.json') for r in json.load(open(f))]
ok = bool(rows) and all(abs(r['flux_ratio'] - 1) <= 1e-3 and r['rel_l1'] <= 1e-3 for r in rows)
json.dump({'rows': rows, 'PASS': ok}, open('$O/t_f2_sum.json', 'w'), indent=1)
print('PASS' if ok else 'FAIL')")
  log "T_F2_SUM $F2OK"
  if [ "$F2OK" = PASS ] || [ -n "$DEV" ]; then
    E4=$($PY -c "import json; print(json.load(open('$O/e4_choice.json'))['chosen'])")
    if [ "$E4" != None ]; then
      j=()
      for t in T0 T1; do for r in $REPS; do
        j+=(I2_refER_${t}_$r "$PY interv92.py ref $R $R86 E4_$t ER $((92610+r)) $RSPP REF_ER_E4_${t}_r$r" I2_refP_${t}_$r "$PY interv92.py ref $R $R86 E4_$t P $((92630+r)) $RSPP REF_P_E4_${t}_r$r")
      done; done
      for ls in LMIX4 LSPOT4; do for r in $SREPS; do j+=(I2_refS_${ls}_$r "$PY interv92.py ref $R $R86 E4_T1_$ls ER $((92650+r)) $RSPP REF_ER_E4_T1_${ls}_r$r"); done; done
      pool 6 "${j[@]}"
      run I2_e4admit "$PY analyze92.py e4 $R $R86"
      pool 2 I2_split_E4_T0 "$PY diag92.py split $R $R86 E4_T0 $DEV" I2_split_E4_T1 "$PY diag92.py split $R $R86 E4_T1 $DEV"
    fi
    pool 4 I3_train_ctl_8601 "$PY interv92.py train $R $R86 8601 ctl $DEV" I3_train_ctl_8602 "$PY interv92.py train $R $R86 8602 ctl $DEV" \
           I3_train_own_8601 "$PY interv92.py train $R $R86 8601 own $DEV" I3_train_own_8602 "$PY interv92.py train $R $R86 8602 own $DEV"
    TAGS="E1_T0 E1_T1 E2_T0 E2_T1 E1_T1_LMIX E1_T1_LSPOT"
    [ "$E4" != None ] && TAGS="$TAGS E4_T0 E4_T1 E4_T1_LMIX4 E4_T1_LSPOT4"
    j=(); for t in $TAGS; do j+=(I4_mtables_$t "$PY interv92.py mtables $R $R86 $t"); done
    pool 4 "${j[@]}"
    j=()
    for t in $TAGS; do
      L=$O/lists/i_$t.txt; : > "$L"
      for m in ecro_s8601 ecro_s8602 ctl_s8601 ctl_s8602 own_s8601 own_s8602; do for r in 0 1; do
        echo "results/tables/I_$t.npz:B_$m $((86600+r)) $SPP I92_${m}_${t}_r$r" >> "$L"
      done; done
      j+=(I4_render_$t "$PY diag92.py render $R $R86 $t $L")
    done
    pool 5 "${j[@]}"
  else
    log "STOP_CONTROL_NOT_REPRODUCIBLE"
  fi
  log "INTERVENTION_END"
fi
# ---- final analysis, visuals
run A_final "$PY analyze92.py final $R $R86"
run V_viz "$PY viz92.py $R $R86 $O/viz"
log CHAIN_DONE
