#!/bin/bash
# WL93B chain: ONE detached chain, end to end.  setsid nohup bash run93b.sh <R> <R86> <R92> <R92A> <R92B> <R93A> [--dev] > <R>/chain.out 2>&1 &
set -u
R=$1; R86=$2; R92=$3; R92A=$4; R92B=$5; R93A=$6; DEV=${7:-}
. "$R/env.sh"
CODE=$R/code/experiments/ecro_stage93b
O=$R/results; ST=$R/chain.status
mkdir -p "$O/logs" "$O/renders" "$O/lists" "$O/tables" "$O/fields" "$O/masks" "$O/viz" "$O/models" "$R/data"
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
finish() { run A_final "$PY analyze93b.py $R $R86 $R92A $R92B $R93A"; log "LABEL $(grep '^LABEL' $O/logs/A_final.log | tail -1 | cut -c1-200)"; run V_viz "$PY viz93b.py $R $R86 $R92A $R92B $O/viz"; log CHAIN_DONE; exit 0; }
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null) R86=$R86 R92=$R92 R92A=$R92A R92B=$R92B R93A=$R93A"
SPP=4096; RSPP=8192; REPS="0 1 2 3 4 5 6 7"; [ -n "$DEV" ] && { SPP=64; RSPP=64; REPS="0 1"; }
EV="$PY eval93b.py"
# ---- S0 inputs: WL93A targets / references (read-only links), configurations, truth splits, E5 inputs; reuse verification
for d in qref train_q val_q; do [ -e "$R/data/$d" ] || ln -s "$R93A/data/$d" "$R/data/$d"; done
cp "$R93A/results/configs.json" "$O/configs.json"; cp "$R93A/results/taskA.json" "$O/taskA_wl93a.json"
cp "$R92/results/e4_choice.json" "$R92B/results/e5_candidates.json" "$O/"
cp "$R92B/results/masks/E5k995.npz" "$O/masks/"
cp "$R92B/results/fields/D_E2_T0.npz" "$R92B/results/fields/D_E2_T1.npz" "$O/fields/"
log "INPUTS $(cd $O && sha256sum configs.json taskA_wl93a.json e4_choice.json e5_candidates.json masks/E5k995.npz fields/*.npz | cut -c1-16 | tr '\n' ' ') links $(ls -l $R/data | grep -c '\->')"
run S0_reuse "$EV reuse $R $R86 $R92A $R93A"
RU=$(grep '^REUSE' $O/logs/S0_reuse.log | tail -1 | cut -d' ' -f2); log "REUSE $RU"
[ "$RU" = PASS ] || { log "STOP STOP_REUSE_NOT_VERIFIED"; finish; }
# ---- S1 pattern diagnostic: representability gate + masks (diagnostic only; a failure removes it from evidence, the chain continues)
run S1_pattern "$EV pattern $R $R86"
PG=$(grep '^PATTERN' $O/logs/S1_pattern.log | tail -1 | cut -d' ' -f2); log "PATTERN $PG $(grep '^PATTERN' $O/logs/S1_pattern.log | tail -1 | cut -c1-200)"
# ---- S2 focused tests on the untrained arms (stop on failure)
run S2_tests_pre "$EV tests $R $R86 $R92A pre"
TP=$(grep '^TESTS' $O/logs/S2_tests_pre.log | tail -1 | cut -d' ' -f3); log "TESTS_PRE $TP"
[ "$TP" = PASS ] || { log "STOP STOP_TEST_FAILURE"; finish; }
# ---- S3 training: U0 / UQ / C0 / CQ and the reproduction controls R_T0 / R_TQ, both seeds (WL93A contract; 4 concurrent)
j=(); for s in 8601 8602; do for a in U0 UQ C0 CQ R_T0 R_TQ; do j+=(S3_train_${a}_s$s "$PY train93b.py $R $R86 $R92A $a $s $DEV"); done; done
pool 4 "${j[@]}"
run S3_tests_post "$EV tests $R $R86 $R92A post"
log "TESTS_POST $(grep '^TESTS' $O/logs/S3_tests_post.log | tail -1 | cut -d' ' -f3)"
# ---- S4 stage / integrated accounting, operators
pool 2 S4_account "$EV account $R $R86 $R92A" S4_operators "$EV operators $R $R86 $R92A"
log "$(grep '^ACCOUNT' $O/logs/S4_account.log | tail -1 | cut -c1-400)"
# ---- S5 render tables, LPAT reference, renders (common random numbers)
TAGS="E2_T0 E2_T1 E1_T1_LSPOT E5k995_T0 E5k995_T1"; [ "$PG" = PASS ] && TAGS="$TAGS E2_T1_LPAT"
j=(); for t in $TAGS; do j+=(S5_tables_$t "$EV tables $R $R86 $R92A $t"); done
pool 3 "${j[@]}"
if [ "$PG" = PASS ]; then
  j=(); for r in $REPS; do j+=(S5_refLPAT_$r "$EV call $R $R86 interv92.ref E2_T1_LPAT ER $((93810+r)) $RSPP REF_ER_E2_T1_LPAT_r$r"); done
  pool 6 "${j[@]}"
fi
j=()
for t in $TAGS; do
  L=$O/lists/i_$t.txt; : > "$L"
  for r in 0 1; do for m in T0 TQ U0 UQ C0 CQ own92a; do for s in 8601 8602; do echo "results/tables/I_$t.npz:B_${m}_s$s $((86600+r)) $SPP I93B_${m}_s${s}_${t}_r$r" >> "$L"; done; done; done
  j+=(S5_render_$t "$EV call $R $R86 diag92.render $t $L")
done
pool 5 "${j[@]}"
finish
