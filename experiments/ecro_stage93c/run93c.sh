#!/bin/bash
# WL93C chain: ONE detached chain, end to end.  setsid nohup bash run93c.sh <R> <R86> <R92> <R92A> <R92B> <R93A> [--dev] > <R>/chain.out 2>&1 &
set -u
R=$1; R86=$2; R92=$3; R92A=$4; R92B=$5; R93A=$6; DEV=${7:-}
. "$R/env.sh"
CODE=$R/code/experiments/ecro_stage93c
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
render() {   # render <stage> <models...> : tables already written; one list per tag
  local stg=$1; shift; local j=()
  for t in E2_T0 E2_T1 E1_T1_LSPOT E5k995_T0 E5k995_T1; do
    L=$O/lists/${stg}_$t.txt; : > "$L"
    for r in 0 1; do for m in "$@"; do for s in 8601 8602; do echo "results/tables/I_$t.npz:B_${m}_s$s $((86600+r)) $SPP I93C_${m}_s${s}_${t}_r$r" >> "$L"; done; done; done
    j+=(${stg}_render_$t "$EV call $R $R86 diag92.render $t $L")
  done
  pool 5 "${j[@]}"
}
finish() { run A_final "$PY analyze93c.py $R $R86 $R92A $R92B"; log "VERDICT $(grep '^VERDICT' $O/logs/A_final.log | tail -1 | cut -c1-200)"; run V_viz "$PY viz93c.py $R $R86 $R92A $R92B $O/viz"; log CHAIN_DONE; exit 0; }
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null) R86=$R86 R92=$R92 R92A=$R92A R92B=$R92B R93A=$R93A"
SPP=4096; [ -n "$DEV" ] && SPP=64
EV="$PY eval93c.py"
# ---- S0 inputs (WL93A targets / references as read-only links; configs; truth splits; E5 inputs); reuse verification; tests
for d in qref train_q val_q; do [ -e "$R/data/$d" ] || ln -s "$R93A/data/$d" "$R/data/$d"; done
cp "$R93A/results/configs.json" "$O/configs.json"; cp "$R93A/results/taskA.json" "$O/taskA_wl93a.json"
cp "$R92/results/e4_choice.json" "$R92B/results/e5_candidates.json" "$O/"
cp "$R92B/results/masks/E5k995.npz" "$O/masks/"
cp "$R92B/results/fields/D_E2_T0.npz" "$R92B/results/fields/D_E2_T1.npz" "$O/fields/"
log "INPUTS $(cd $O && sha256sum configs.json taskA_wl93a.json e4_choice.json e5_candidates.json masks/E5k995.npz fields/*.npz | cut -c1-16 | tr '\n' ' ')"
run S0_reuse "$EV reuse $R $R86 $R92A $R93A"
RU=$(grep '^REUSE' $O/logs/S0_reuse.log | tail -1 | cut -d' ' -f2); log "REUSE $RU"
[ "$RU" = PASS ] || { log "STOP STOP_REUSE_NOT_VERIFIED"; finish; }
run S0_tests_pre "$EV tests $R $R86 $R92A $R92B pre"
TP=$(grep '^TESTS' $O/logs/S0_tests_pre.log | tail -1 | cut -d' ' -f3); log "TESTS_PRE $TP"
[ "$TP" = PASS ] || { log "STOP STOP_TEST_FAILURE"; finish; }
# ---- S1 Task A: oracle composition accounting + gate; oracle render tables and renders (always: Task A evidence)
run S1_oracle "$EV oracle $R $R86 $R92A"
OG=$(grep '^ORACLE' $O/logs/S1_oracle.log | tail -1 | cut -d' ' -f2); log "$(grep '^ORACLE' $O/logs/S1_oracle.log | tail -1 | cut -c1-300)"
j=(); for t in E2_T0 E2_T1 E1_T1_LSPOT E5k995_T0 E5k995_T1; do j+=(S1_tables_$t "$EV tables $R $R86 $R92A $t oracle"); done
pool 3 "${j[@]}"
render S1 ORA
[ -n "$DEV" ] && [ -n "${FORCE_B:-}" ] && OG=FEASIBLE                    # dev plumbing only
[ "$OG" = FEASIBLE ] || { log "STOP ORACLE_COMPOSITION_LIMITED (no training)"; finish; }
# ---- S2 Task B: TR / TQR (WL93A contract; 4 concurrent); post tests; accounting; render tables / renders of T0 / TQ / TR / TQR
pool 4 S2_train_TR_s8601 "$PY train93c.py $R $R86 $R92A TR 8601 $DEV" S2_train_TQR_s8601 "$PY train93c.py $R $R86 $R92A TQR 8601 $DEV" \
  S2_train_TR_s8602 "$PY train93c.py $R $R86 $R92A TR 8602 $DEV" S2_train_TQR_s8602 "$PY train93c.py $R $R86 $R92A TQR 8602 $DEV"
run S2_tests_post "$EV tests $R $R86 $R92A $R92B post"
log "TESTS_POST $(grep '^TESTS' $O/logs/S2_tests_post.log | tail -1 | cut -d' ' -f3)"
run S2_account "$EV account $R $R86 $R92A"
log "$(grep '^ACCOUNT' $O/logs/S2_account.log | tail -1 | cut -c1-400)"
j=(); for t in E2_T0 E2_T1 E1_T1_LSPOT E5k995_T0 E5k995_T1; do j+=(S2_tables_$t "$EV tables $R $R86 $R92A $t arms"); done
pool 1 "${j[@]}"
render S2 T0 TQ TR TQR
finish
