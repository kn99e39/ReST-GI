#!/bin/bash
# WL93A chain: ONE detached chain, end to end.  setsid nohup bash run93a.sh <R> <R86> <R92> <R92A> <R92B> <R93> [--dev] > <R>/chain.out 2>&1 &
set -u
R=$1; R86=$2; R92=$3; R92A=$4; R92B=$5; R93=$6; DEV=${7:-}
. "$R/env.sh"
CODE=$R/code/experiments/ecro_stage93a
O=$R/results; ST=$R/chain.status
mkdir -p "$O/logs" "$O/renders" "$O/lists" "$O/tables" "$O/fields" "$O/masks" "$O/viz" "$O/models" "$O/stage"
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
finish() { run A_final "$PY analyze93a.py $R $R86 $R92A $R92B"; log "VERDICT $(grep '^VERDICT' $O/logs/A_final.log | tail -1 | cut -c1-200)"; run V_viz "$PY viz93a.py $R $R86 $R92A $R92B $O/viz"; log CHAIN_DONE; exit 0; }
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null) R86=$R86 R92=$R92 R92A=$R92A R92B=$R92B R93=$R93"
SPP=4096; [ -n "$DEV" ] && SPP=64
QR="$PY qref93a.py"; EV="$PY eval93a.py"; PR="$PY ../ecro_stage93/prep93.py"
# ---- S0 inputs (copied, hashed), configurations
cp "$R92/results/e4_choice.json" "$R92B/results/e5_candidates.json" "$O/"
cp "$R92B/results/masks/E5k995.npz" "$O/masks/"
cp "$R92B/results/fields/D_E2_T1.npz" "$O/fields/"
log "INPUTS $(cd $O && sha256sum e4_choice.json e5_candidates.json masks/*.npz fields/*.npz | cut -c1-16 | tr '\n' ' ')"
run S0_cfgs "$QR cfgs $R $R86 $R92A $DEV"
NC=$($PY -c "import json; print(len(json.load(open('$O/configs.json'))['configs']))")
# ---- S1 Task A references (B0-B3 concurrently with the pre-training stage tables; B4; B5 alone: 11.5 GB)
pool 7 S1_ref_B0 "$QR refgen $R $R86 $R92A B0 0 $NC $DEV" S1_ref_B1 "$QR refgen $R $R86 $R92A B1 0 $NC $DEV" S1_ref_B2 "$QR refgen $R $R86 $R92A B2 0 $NC $DEV" \
  S1_ref_B3 "$QR refgen $R $R86 $R92A B3 0 $NC $DEV" S1_stage_enc "$EV stage $R $R86 $R92A enc" S1_stage_own_s8601 "$EV stage $R $R86 $R92A own_s8601" S1_stage_own_s8602 "$EV stage $R $R86 $R92A own_s8602"
S1=$(( (NC+2)/3 ))
pool 3 S1_ref_B4a "$QR refgen $R $R86 $R92A B4 0 $S1 $DEV" S1_ref_B4b "$QR refgen $R $R86 $R92A B4 $S1 $S1 $DEV" S1_ref_B4c "$QR refgen $R $R86 $R92A B4 $((2*S1)) $S1 $DEV"
run S1_ref_B5 "$QR refgen $R $R86 $R92A B5 0 $NC $DEV"
# ---- S2 Task A analysis: target reliability and the training budget (stop before training if unreliable)
run S2_taskA "$QR taskA $R $R86 $R92A $R93 $DEV"
log "$(grep '^TASKA' $O/logs/S2_taskA.log | tail -1)"
TR=$(grep '^TASKA' $O/logs/S2_taskA.log | tail -1 | cut -d' ' -f2); TB=$(grep '^TASKA' $O/logs/S2_taskA.log | tail -1 | cut -d' ' -f4)
if [ "$TR" != RELIABLE ] && [ -z "$DEV" ]; then log "STOP TARGET_UNRELIABLE"; finish; fi
[ "$TB" = None ] && TB=B0
# ---- S3 training / training-validation targets at the train budget (independent seeds)
TJ=2; [ "$TB" = B5 ] && TJ=1                                                 # B5 peaks at 9.4 GB (probe): never two at once
pool $TJ S3_tgen_train "$QR tgen $R $R86 $R92A $TB train $DEV" S3_tgen_val "$QR tgen $R $R86 $R92A $TB val $DEV"
# ---- S4 focused tests (stop on failure)
run S4_tests_pre "$EV tests $R $R86 $R92A pre"
TP=$(grep '^TESTS' $O/logs/S4_tests_pre.log | tail -1 | cut -d' ' -f3); log "TESTS_PRE $TP"
[ "$TP" = PASS ] || { log "STOP STOPPED_TEST_FAILURE"; finish; }
# ---- S5 training: T0 / TQ x 2 seeds (4 concurrent; identical subset, order, steps)
pool 4 S5_train_T0_s8601 "$PY train93a.py $R $R86 $R92A T0 8601 $DEV" S5_train_TQ_s8601 "$PY train93a.py $R $R86 $R92A TQ 8601 $DEV" \
  S5_train_T0_s8602 "$PY train93a.py $R $R86 $R92A T0 8602 $DEV" S5_train_TQ_s8602 "$PY train93a.py $R $R86 $R92A TQ 8602 $DEV"
run S5_tests_post "$EV tests $R $R86 $R92A post"
log "TESTS_POST $(grep '^TESTS' $O/logs/S5_tests_post.log | tail -1 | cut -d' ' -f3)"
# ---- S6 stage evaluation vs the B5 reference
pool 4 S6_stage_T0_s8601 "$EV stage $R $R86 $R92A T0_s8601" S6_stage_TQ_s8601 "$EV stage $R $R86 $R92A TQ_s8601" S6_stage_T0_s8602 "$EV stage $R $R86 $R92A T0_s8602" S6_stage_TQ_s8602 "$EV stage $R $R86 $R92A TQ_s8602"
run S6_evalB "$EV evalB $R $R86 $R92A"
log "$(grep '^EVALB' $O/logs/S6_evalB.log | tail -1 | cut -c1-300)"
# ---- S7 rendered smoke (diagnostic; WL86 ECRO render seeds, common random numbers)
TAGS="E2_T0 E2_T1 E1_T1_LSPOT E5k995_T0 E5k995_T1"
j=(); for t in $TAGS; do j+=(S7_tables_$t "$EV tables $R $R86 $R92A $t"); done
pool 3 "${j[@]}"
j=()
for t in $TAGS; do
  L=$O/lists/i_$t.txt; : > "$L"
  for r in 0 1; do for m in T0_s8601 T0_s8602 TQ_s8601 TQ_s8602 own92a_s8601 own92a_s8602; do echo "results/tables/I_$t.npz:B_$m $((86600+r)) $SPP I93A_${m}_${t}_r$r" >> "$L"; done; done
  j+=(S7_render_$t "$PR call $R $R86 diag92.render $t $L")
done
pool 5 "${j[@]}"
finish
