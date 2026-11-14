#!/bin/bash
# WL93D chain: ONE detached chain, end to end.  setsid nohup bash run93d.sh <R> <R86> <R92> <R92A> <R92B> <R93A> [--dev] > <R>/chain.out 2>&1 &
set -u
R=$1; R86=$2; R92=$3; R92A=$4; R92B=$5; R93A=$6; DEV=${7:-}
. "$R/env.sh"
CODE=$R/code/experiments/ecro_local93d
O=$R/results; ST=$R/chain.status
mkdir -p "$O/logs" "$O/renders" "$O/lists" "$O/tables" "$O/fields" "$O/masks" "$O/viz" "$O/ops" "$R/data"
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
finish() { run A_final "$PY analyze93d.py $R $R86 $R92A $R92B $R93A"; log "LABELS $(grep '^LABELS' $O/logs/A_final.log | tail -1 | cut -c1-220)"; run V_viz "$PY viz93d.py $R $R86 $R92A $R92B $O/viz"; log CHAIN_DONE; exit 0; }
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null) R86=$R86 R92=$R92 R92A=$R92A R92B=$R92B R93A=$R93A"
SPP=4096; RSPP=8192; REPS="0 1 2 3 4 5 6 7"; [ -n "$DEV" ] && { SPP=64; RSPP=64; REPS="0 1"; }
OPS="$PY ops93d.py"; PR="$PY ../ecro_stage93/prep93.py"
# ---- S0 inputs (WL93A references / configs as read-only links / copies; E2 / E5 truth splits and masks); checkpoint identity
for d in qref val_q; do [ -e "$R/data/$d" ] || ln -s "$R93A/data/$d" "$R/data/$d"; done
cp "$R93A/results/configs.json" "$O/configs.json"; cp "$R93A/results/taskA.json" "$O/taskA_wl93a.json"
cp "$R92/results/e4_choice.json" "$R92B/results/e5_candidates.json" "$O/"
cp "$R92B/results/masks/E5k995.npz" "$O/masks/"
for t in E2_T0 E2_T1 E5k995_T0 E5k995_T1; do cp "$R92B/results/fields/D_$t.npz" "$O/fields/"; done
log "INPUTS $(cd $O && sha256sum configs.json taskA_wl93a.json e4_choice.json e5_candidates.json masks/E5k995.npz fields/*.npz | cut -c1-16 | tr '\n' ' ')"
run S0_ckpt "$OPS ckpt $R $R92A $R93A $R92B"
CK=$(grep '^CKPT' $O/logs/S0_ckpt.log | tail -1 | cut -d' ' -f2); log "CKPT $CK"
[ "$CK" = PASS ] || { log "STOP STOP_CKPT_FAILED"; finish; }
# ---- S1 held-out E6 qualification under the frozen WL93 e6 rule (prep93 unchanged), BEFORE any model output on E6
run S1_e6gen "$PR e6gen $R $R86 $DEV"
KS=$($PY -c "import json; print(' '.join(str(c['k']) for c in json.load(open('$O/e6_candidates.json'))['geometry_admissible']))")
log "E6 GEOMETRY_ADMISSIBLE [$KS]"
CH=""
for k in $KS; do
  e=E6k$k; off=$((100*k)); j=()
  for t in T0 T1; do for r in $REPS; do
    j+=(S1_${e}_refER_${t}_$r "$PR call $R $R86 interv92.ref ${e}_$t ER $((93710+r+off)) $RSPP REF_ER_${e}_${t}_r$r" S1_${e}_refP_${t}_$r "$PR call $R $R86 interv92.ref ${e}_$t P $((93730+r+off)) $RSPP REF_P_${e}_${t}_r$r")
  done; done
  pool 6 "${j[@]}"
  pool 2 S1_${e}_split_T0 "$PR call $R $R86 diag92.split ${e}_T0 $DEV" S1_${e}_split_T1 "$PR call $R $R86 diag92.split ${e}_T1 $DEV"
  run S1_${e}_admit "$PR e6admit $R $R86 $k"
  A=$(grep '^E6ADMIT' $O/logs/S1_${e}_admit.log | tail -1 | cut -d' ' -f3); log "E6 candidate $k $A"
  if [ "$A" = ADMITTED ]; then CH=$k; break; fi
done
FORCED=False
[ -z "$CH" ] && [ -n "$DEV" ] && [ -n "${FORCE_E6:-}" ] && { CH=$(echo $KS | cut -d' ' -f1); FORCED=True; }          # dev plumbing only
$PY -c "import json; json.dump({'chosen': ${CH:-None}, 'dev_forced': $FORCED}, open('$O/e6_choice.json', 'w'))"
log "E6 CHOSEN ${CH:-NONE} (forced=$FORCED)"
# ---- S2 configurations, focused tests (stop on failure), incident fields, effective operators
run S2_cfgs "$OPS cfgs $R $R86 $DEV"
run S2_tests "$OPS tests $R $R86 $R92A $R93A $R92B"
TP=$(grep '^TESTS' $O/logs/S2_tests.log | tail -1 | cut -d' ' -f2); log "TESTS $TP"
[ "$TP" = PASS ] || { log "STOP STOP_TESTS_FAILED"; finish; }
run S2_fields "$OPS fields $R $R86 $R92A"
pool 4 S2_ops_own_8601 "$OPS ops $R $R86 $R92A $R93A own 8601" S2_ops_own_8602 "$OPS ops $R $R86 $R92A $R93A own 8602" S2_ops_T0_8601 "$OPS ops $R $R86 $R92A $R93A T0 8601" S2_ops_T0_8602 "$OPS ops $R $R86 $R92A $R93A T0 8602"
# ---- S3 Task A / C functional analysis || Task B table-level interventions
pool 2 S3_functional "$OPS functional $R $R86" S3_account "$OPS account $R $R86 $R92A $R93A"
log "$(grep '^FUNCTIONAL' $O/logs/S3_functional.log | head -2 | cut -c1-300 | tr '\n' ' ')"
# ---- S4 render tables and renders (common random numbers)
TAGS="E2_T0 E2_T1 E5k995_T0 E5k995_T1"; [ -n "$CH" ] && TAGS="$TAGS E6k${CH}_T0 E6k${CH}_T1"
j=(); for t in $TAGS; do j+=(S4_tables_$t "$OPS tables $R $R86 $R92A $R93A $t"); done
pool 3 "${j[@]}"
j=()
for t in $TAGS; do
  L=$O/lists/i_$t.txt; : > "$L"
  case $t in *_T1) ARMS="C0 C1_src C1_rcv C1_both C2_src C2_rcv C2_both";; *) ARMS="C0 C2_src C2_rcv C2_both";; esac
  for r in 0 1; do for f in own T0; do for s in 8601 8602; do for a in $ARMS; do echo "results/tables/I_$t.npz:B_${f}_s${s}_$a $((86600+r)) $SPP I93D_${f}_s${s}_${a}_${t}_r$r" >> "$L"; done; done; done; done
  j+=(S4_render_$t "$PR call $R $R86 diag92.render $t $L")
done
pool 5 "${j[@]}"
finish
