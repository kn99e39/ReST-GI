#!/bin/bash
# WL93 chain: ONE detached chain, end to end.  setsid nohup bash run93.sh <R> <R86> <R92> <R92A> <R92B> [--dev] > <R>/chain.out 2>&1 &
# Concurrency (QJ, TJ, RJ, EJ) set from the full-scale probes recorded in the worklog before the freeze.
set -u
R=$1; R86=$2; R92=$3; R92A=$4; R92B=$5; DEV=${6:-}
. "$R/env.sh"
CODE=$R/code/experiments/ecro_stage93
O=$R/results; ST=$R/chain.status
mkdir -p "$O/logs" "$O/renders" "$O/lists" "$O/tables" "$O/fields" "$O/masks" "$O/viz" "$O/models" "$O/cf92b"
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
finish() { run A_final "$PY analyze93.py $R $R86 $R92A $R92B"; log "LABEL $(grep '^LABEL' $O/logs/A_final.log | tail -1 | cut -c1-160)"; run V_viz "$PY viz93.py $R $R86 $R92A $R92B $O/viz"; log CHAIN_DONE; exit 0; }
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null) R86=$R86 R92=$R92 R92A=$R92A R92B=$R92B"
QJ=${QJ:-8}; TJ=${TJ:-4}; RJ=${RJ:-5}; EJ=${EJ:-6}
SPP=4096; RSPP=8192; REPS="0 1 2 3 4 5 6 7"; SREPS="0 1 2 3"; QTR="0 128 256 384 512 640 768 896"; QN=128; QV=64
if [ -n "$DEV" ]; then SPP=64; RSPP=64; REPS="0 1"; SREPS="0 1"; QTR="0 30"; QN=30; QV=16; fi
PR="$PY prep93.py"; EV="$PY eval93.py"
# ---- S0 inputs (copied, hashed) and the WL92B reconciliation record
cp "$R92/results/e4_choice.json" "$O/e4_choice.json"
cp "$R92B/results/e5_candidates.json" "$O/e5_candidates.json"
cp "$R92B/results/masks/E5k995.npz" "$R92B/results/masks/E5k995_LMIX.npz" "$R92B/results/masks/E5k995_LSPOT.npz" "$O/masks/"
for t in E2_T1 E2_T0 E1_T1_LSPOT E5k995_T0 E5k995_T1; do cp "$R92B/results/fields/D_$t.npz" "$R92B/results/fields/W_$t.npz" "$O/fields/"; done
cp "$R92B/results/tables/CF_E2_T1.json" "$R92B/results/tables/CF_E2_T0.json" "$O/cf92b/"
log "INPUTS $(cd $O && sha256sum e4_choice.json e5_candidates.json masks/*.npz fields/*.npz cf92b/*.json | cut -c1-16 | tr '\n' ' ')"
run S0_recon92b "$PR recon92b $R $R92B"
log "$(grep '^RECON92B' $O/logs/S0_recon92b.log | tail -1)"
# ---- S1 validation source-stage targets, identifiability gate (stop if not identifiable)
run S1_qgen_val "$PR qgen $R $R86 $R92A val 0 $QV $DEV"
run S1_ident "$EV ident $R $R86 $R92A $DEV"
IG=$(grep '^IDENT' $O/logs/S1_ident.log | tail -1 | cut -d' ' -f2)
log "IDENT $IG"
[ "$IG" = PASS ] || [ -n "$DEV" ] || { log "STOP STOP_TARGET_NOT_IDENTIFIABLE"; finish; }
# ---- S2 training source-stage targets || E6 geometry-only candidates
j=(); for a in $QTR; do j+=(S2_qgen_train_$a "$PR qgen $R $R86 $R92A train $a $QN $DEV"); done
j+=(S2_e6gen "$PR e6gen $R $R86 $DEV")
pool $QJ "${j[@]}"
# ---- S3 E6 qualification BEFORE training: references for the geometry-admissible candidates in order until one is admitted
KS=$($PY -c "import json; print(' '.join(str(c['k']) for c in json.load(open('$O/e6_candidates.json'))['geometry_admissible']))")
log "E6 GEOMETRY_ADMISSIBLE [$KS]"
CH=""
for k in $KS; do
  e=E6k$k; off=$((100*k))
  j=()
  for t in T0 T1; do for r in $REPS; do
    j+=(S3_${e}_refER_${t}_$r "$PR call $R $R86 interv92.ref ${e}_$t ER $((93710+r+off)) $RSPP REF_ER_${e}_${t}_r$r" S3_${e}_refP_${t}_$r "$PR call $R $R86 interv92.ref ${e}_$t P $((93730+r+off)) $RSPP REF_P_${e}_${t}_r$r")
  done; done
  pool $EJ "${j[@]}"
  pool 2 S3_${e}_split_T0 "$PR call $R $R86 diag92.split ${e}_T0 $DEV" S3_${e}_split_T1 "$PR call $R $R86 diag92.split ${e}_T1 $DEV"
  run S3_${e}_admit "$PR e6admit $R $R86 $k"
  A=$(grep '^E6ADMIT' $O/logs/S3_${e}_admit.log | tail -1 | cut -d' ' -f3)
  log "E6 candidate $k $A"
  if [ "$A" = ADMITTED ]; then CH=$k; break; fi
done
FORCED=False
[ -z "$CH" ] && [ -n "$DEV" ] && [ -n "${FORCE_E6:-}" ] && { CH=$(echo $KS | cut -d' ' -f1); FORCED=True; }          # dev plumbing only (first geometry-admissible candidate)
$PY -c "import json; json.dump({'chosen': ${CH:-None}, 'dev_forced': $FORCED}, open('$O/e6_choice.json', 'w'))"
if [ -z "$CH" ]; then log "STOP STOPPED_WORKLOAD_UNQUALIFIED"; finish; fi
e=E6k$CH
log "E6 CHOSEN $e (forced=$FORCED)"
j=()
for ls in LMIX LSPOT; do for r in $SREPS; do j+=(S3_refS_${ls}_$r "$PR call $R $R86 interv92.ref ${e}_T1_$ls ER $((93750+r)) $RSPP REF_ER_${e}_T1_${ls}_r$r"); done; done
j+=(S3_ofield_T0 "$EV ofield $R $R86 ${e}_T0 $DEV" S3_ofield_T1 "$EV ofield $R $R86 ${e}_T1 $DEV")
pool $EJ "${j[@]}"
# ---- S4 synthetic tests on the untrained arms (stop on failure)
run S4_tests_pre "$EV tests $R $R86 $R92A pre $DEV"
TP=$(grep '^TESTS' $O/logs/S4_tests_pre.log | tail -1 | cut -d' ' -f3)
log "TESTS_PRE $TP"
[ "$TP" = PASS ] || [ -n "$DEV" ] || { log "STOP STOPPED_TEST_FAILURE"; finish; }
# ---- S5 training: 6 arms x 2 seeds, identical budget / order per seed
j=(); for s in 8601 8602; do for a in T0 TQ U0 UQ C0 CQ; do j+=(S5_train_${a}_s$s "$PY train93.py $R $R86 $R92A $a $s $DEV"); done; done
pool $TJ "${j[@]}"
run S5_tests_post "$EV tests $R $R86 $R92A post $DEV"
log "TESTS_POST $(grep '^TESTS' $O/logs/S5_tests_post.log | tail -1 | cut -d' ' -f3)"
# ---- S6 tables, local matrices, timing
TAGS="E2_T0 E2_T1 E1_T1_LSPOT E5k995_T0 E5k995_T1 E5k995_T1_LMIX E5k995_T1_LSPOT ${e}_T0 ${e}_T1 ${e}_T1_LMIX ${e}_T1_LSPOT"
j=(); for t in $TAGS; do j+=(S6_tables_$t "$EV tables $R $R86 $R92A $t $DEV"); done
pool 3 "${j[@]}"
run S6_local "$EV local $R $R86 $R92A"
run S6_timing "$EV timing $R $R86 $R92A"
# ---- S7 renders (WL86 ECRO render seeds: common random numbers across models)
MODELS=""; for a in T0 TQ U0 UQ C0 CQ own92a sum92a; do for s in 8601 8602; do MODELS="$MODELS ${a}_s$s"; done; done
j=()
for t in $TAGS; do
  L=$O/lists/i_$t.txt; : > "$L"
  for r in 0 1; do
    for m in $MODELS; do echo "results/tables/I_$t.npz:B_$m $((86600+r)) $SPP I93_${m}_${t}_r$r" >> "$L"; done
    case $t in E6*_T0|E6*_T1) for k2 in ORACLE X_EXP; do echo "results/tables/I_$t.npz:$k2 $((86600+r)) $SPP I93_${k2}_${t}_r$r" >> "$L"; done;; esac
  done
  j+=(S7_render_$t "$PR call $R $R86 diag92.render $t $L")
done
pool $RJ "${j[@]}"
finish
