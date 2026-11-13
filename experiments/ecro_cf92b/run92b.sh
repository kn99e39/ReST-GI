#!/bin/bash
# WL92B chain: ONE detached chain, end to end.  setsid nohup bash run92b.sh <R> <R86> <R92> <R92A> [--dev] > <R>/chain.out 2>&1 &
set -u
R=$1; R86=$2; R92=$3; R92A=$4; DEV=${5:-}
. "$R/env.sh"
CODE=$R/code/experiments/ecro_cf92b
O=$R/results; ST=$R/chain.status
mkdir -p "$O/logs" "$O/renders" "$O/lists" "$O/tables" "$O/fields" "$O/masks" "$O/viz"
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
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null) R86=$R86 R92=$R92 R92A=$R92A"
SPP=4096; RSPP=8192; REPS="0 1 2 3 4 5 6 7"; SREPS="0 1 2 3"
if [ -n "$DEV" ]; then SPP=64; RSPP=64; REPS="0 1"; SREPS="0 1"; fi
CF="$PY cf92b.py"
# ---- S0 inputs (copied, hashed), reproduction (stop if not reproduced)
cp "$R92/results/e4_choice.json" "$O/e4_choice.json"
cp "$R92/results/masks/E4.npz" "$R92/results/masks/E4_LMIX4.npz" "$R92/results/masks/E4_LSPOT4.npz" "$O/masks/"
for t in E2_T1 E2_T0 E1_T1; do cp "$R92/results/fields/D_$t.npz" "$R92/results/fields/W_$t.npz" "$O/fields/"; done
for t in E4_T0 E4_T1; do cp "$R92A/results/fields/D_$t.npz" "$R92A/results/fields/W_$t.npz" "$O/fields/"; done
log "INPUTS $(cd $O && sha256sum e4_choice.json masks/*.npz fields/*.npz | cut -c1-16 | tr '\n' ' ')"
run S0_repro "$CF repro $R $R86 $R92A $DEV"
RP=$(grep '^REPRO' $O/logs/S0_repro.log | tail -1 | cut -d' ' -f2)
log "REPRO $RP"
[ "$RP" = PASS ] || [ -n "$DEV" ] || { log "STOP STOP_BASELINE_NOT_REPRODUCED"; run A_final "$PY analyze92b.py $R $R86 $R92A"; log CHAIN_DONE; exit 0; }
# ---- S1 new truth split (E1 LSPOT) and its oracle field || E5 geometry-only candidates
pool 3 S1_split_LSPOT "$CF call $R $R86 diag92.split E1_T1_LSPOT $DEV" S1_ofield_LSPOT "$CF ofield $R $R86 E1_T1_LSPOT $DEV" S1_e5gen "$CF e5gen $R $R86 $DEV"
# ---- S2 counterfactual tables, local matrices, tests
TAGS="E2_T1 E2_T0 E1_T1_LSPOT E4_T1 E4_T0"
j=(); for t in $TAGS; do j+=(S2_cf_$t "$CF cftables $R $R86 $R92A $t $DEV"); done; j+=(S2_local "$CF local $R $R86 $R92A")
pool 3 "${j[@]}"
run S2_tests "$PY tests92b.py $R $R86 $DEV"
log "TESTS $(grep '^TESTS' $O/logs/S2_tests.log | tail -1 | cut -d' ' -f2)"
# ---- S3 renders of every counterfactual table (WL86 ECRO render seeds: common random numbers)
MODELS="ecro_s8601 ecro_s8602 sum_s8601 sum_s8602 own_s8601 own_s8602"
CFS="ORIG X_L2 X_L3 X_M1 X_M2 X_M12 X_Q X_SHQ"
j=()
for t in $TAGS; do
  L=$O/lists/cf_$t.txt; : > "$L"
  for r in 0 1; do
    echo "results/tables/CF_$t.npz:ORA $((86600+r)) $SPP X92B_ORA_${t}_r$r" >> "$L"
    for m in $MODELS; do for c in $CFS; do echo "results/tables/CF_$t.npz:${c}_$m $((86600+r)) $SPP X92B_${c}_${m}_${t}_r$r" >> "$L"; done; done
  done
  j+=(S3_render_$t "$CF call $R $R86 diag92.render $t $L")
done
pool 5 "${j[@]}"
# ---- S4 E5 qualification: references for the geometry-admissible candidates in order until one is admitted
KS=$($PY -c "import json; print(' '.join(str(c['k']) for c in json.load(open('$O/e5_candidates.json'))['geometry_admissible']))")
log "E5 GEOMETRY_ADMISSIBLE [$KS]"
CH=""
for k in $KS; do
  e=E5k$k; off=$((100*k))
  j=()
  for t in T0 T1; do for r in $REPS; do
    j+=(S4_${e}_refER_${t}_$r "$CF call $R $R86 interv92.ref ${e}_$t ER $((92710+r+off)) $RSPP REF_ER_${e}_${t}_r$r" S4_${e}_refP_${t}_$r "$CF call $R $R86 interv92.ref ${e}_$t P $((92730+r+off)) $RSPP REF_P_${e}_${t}_r$r")
  done; done
  pool 6 "${j[@]}"
  pool 2 S4_${e}_split_T0 "$CF call $R $R86 diag92.split ${e}_T0 $DEV" S4_${e}_split_T1 "$CF call $R $R86 diag92.split ${e}_T1 $DEV"
  run S4_${e}_admit "$CF e5admit $R $R86 $k"
  A=$(grep '^E5ADMIT' $O/logs/S4_${e}_admit.log | tail -1 | cut -d' ' -f3)
  log "E5 candidate $k $A"
  if [ "$A" = ADMITTED ]; then CH=$k; break; fi
done
# ---- S5 evaluation of the admitted E5 (frozen before any model output on it)
[ -z "$CH" ] && [ -n "$DEV" ] && [ -n "${FORCE_E5:-}" ] && CH=$FORCE_E5          # dev plumbing only
if [ -n "$CH" ]; then
  e=E5k$CH
  log "E5 CHOSEN $e"
  j=()
  for ls in LMIX LSPOT; do for r in $SREPS; do j+=(S5_refS_${ls}_$r "$CF call $R $R86 interv92.ref ${e}_T1_$ls ER $((92750+r)) $RSPP REF_ER_${e}_T1_${ls}_r$r"); done; done
  j+=(S5_ofield_T0 "$CF ofield $R $R86 ${e}_T0 $DEV" S5_ofield_T1 "$CF ofield $R $R86 ${e}_T1 $DEV")
  pool 6 "${j[@]}"
  ETAGS="${e}_T0 ${e}_T1 ${e}_T1_LMIX ${e}_T1_LSPOT"
  j=(); for t in $ETAGS; do j+=(S5_tables_$t "$CF e5tables $R $R86 $R92A $t"); done
  pool 4 "${j[@]}"
  j=()
  for t in $ETAGS; do
    L=$O/lists/e5_$t.txt; : > "$L"
    for r in 0 1; do
      for m in $MODELS; do echo "results/tables/I_$t.npz:B_$m $((86600+r)) $SPP I92B_${m}_${t}_r$r" >> "$L"; done
      case $t in *_T1) for m in $MODELS; do echo "results/tables/I_$t.npz:STALE_$m $((86600+r)) $SPP I92B_STALE_${m}_${t}_r$r" >> "$L"; done;; esac
      case $t in *_T0|*_T1) for k2 in ORACLE X_EXP; do echo "results/tables/I_$t.npz:$k2 $((86600+r)) $SPP I92B_${k2}_${t}_r$r" >> "$L"; done;; esac
    done
    j+=(S5_render_$t "$CF call $R $R86 diag92.render $t $L")
  done
  pool 4 "${j[@]}"
else
  log "E5 NONE_ADMITTED"
fi
run A_final "$PY analyze92b.py $R $R86 $R92A"
log "OUTCOME $(grep '^OUTCOME' $O/logs/A_final.log | tail -1 | cut -c1-120)"
run V_viz "$PY viz92b.py $R $R86 $R92A $O/viz"
log CHAIN_DONE
