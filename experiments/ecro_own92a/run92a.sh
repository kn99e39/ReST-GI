#!/bin/bash
# WL92A chain: ONE detached chain, end to end.   setsid nohup bash run92a.sh <R> <R86> <R92> [--dev] > <R>/chain.out 2>&1 &
set -u
R=$1; R86=$2; R92=$3; DEV=${4:-}
. "$R/env.sh"
CODE=$R/code/experiments/ecro_own92a
A92=$R/code/experiments/ecro_attr92
O=$R/results; ST=$R/chain.status
mkdir -p "$O/logs" "$O/renders" "$O/lists" "$O/tables" "$O/fields" "$O/masks" "$O/models" "$O/viz"
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
log "CHAIN_START dev=$DEV commit=$(cat $R/code/COMMIT 2>/dev/null) R86=$R86 R92=$R92"
SPP=4096; RSPP=8192; REPS="0 1 2 3 4 5 6 7"; SREPS="0 1 2 3"
if [ -n "$DEV" ]; then SPP=64; RSPP=64; REPS="0 1"; SREPS="0 1"; fi
# ---- S0 inputs from WL92 (geometry-only E4 choice / masks; truth splits of the inspected conditions), exclusion, pre-tests
cp "$R92/results/e4_choice.json" "$O/e4_choice.json"
cp "$R92/results/masks/E4.npz" "$R92/results/masks/E4_LMIX4.npz" "$R92/results/masks/E4_LSPOT4.npz" "$O/masks/"
for t in E2_T1 E2_T0 E1_T1; do cp "$R92/results/fields/D_$t.npz" "$R92/results/fields/W_$t.npz" "$O/fields/"; done
log "INPUTS $(cd $O && sha256sum e4_choice.json masks/*.npz fields/*.npz | cut -c1-16 | tr '\n' ' ')"
run S0_excl "$PY own92a.py excl $R $R86"
run S0_tests "$PY tests92a.py pre $R $R86 $R92 $DEV"
TP=$(grep '^TESTS_PRE' $O/logs/S0_tests.log | tail -1 | cut -d' ' -f2)
log "TESTS_PRE $TP"
[ "$TP" = PASS ] || [ -n "$DEV" ] || { log "STOP D_STOPPED tests_pre"; run A_final "$PY analyze92a.py $R $R86"; log CHAIN_DONE; exit 0; }
# ---- S1 component targets (WL86 shards) || E4 references
j=()
if [ -n "$DEV" ]; then j+=(S1_gen_train_0 "$PY own92a.py gen $R $R86 train 0 48 $DEV" S1_gen_val "$PY own92a.py gen $R $R86 val 0 8 $DEV" S1_gen_trainhq "$PY own92a.py gen $R $R86 trainhq 0 4 $DEV")
else
  for s in 0 1 2 3 4 5 6 7; do j+=(S1_gen_train_$s "$PY own92a.py gen $R $R86 train $((s*128)) 128"); done
  for s in 0 1 2 3; do j+=(S1_gen_val_$s "$PY own92a.py gen $R $R86 val $((s*16)) 16" S1_gen_trainhq_$s "$PY own92a.py gen $R $R86 trainhq $((s*8)) 8"); done
fi
for t in T0 T1; do for r in $REPS; do
  j+=(S1_refER_${t}_$r "$PY $A92/interv92.py ref $R $R86 E4_$t ER $((92610+r)) $RSPP REF_ER_E4_${t}_r$r" S1_refP_${t}_$r "$PY $A92/interv92.py ref $R $R86 E4_$t P $((92630+r)) $RSPP REF_P_E4_${t}_r$r")
done; done
for ls in LMIX4 LSPOT4; do for r in $SREPS; do j+=(S1_refS_${ls}_$r "$PY $A92/interv92.py ref $R $R86 E4_T1_$ls ER $((92650+r)) $RSPP REF_ER_E4_T1_${ls}_r$r"); done; done
pool 8 "${j[@]}"
run S1_f2 "$PY tests92a.py f2 $R $DEV"
F2=$(grep '^T_F2_SUM' $O/logs/S1_f2.log | tail -1 | cut -d' ' -f2)
run S1_e4admit "$PY $A92/analyze92.py e4 $R $R86"
E4A=$(grep '^E4' $O/logs/S1_e4admit.log | tail -1 | cut -d' ' -f2)
log "T_F2_SUM $F2 E4 $E4A"
[ "$F2" = PASS ] || [ -n "$DEV" ] || { log "STOP D_STOPPED T_F2_SUM"; run A_final "$PY analyze92a.py $R $R86"; log CHAIN_DONE; exit 0; }
# ---- S2 training (4 runs) || E4 truth split and oracle fields
pool 6 S2_train_sum_8601 "$PY own92a.py train $R $R86 8601 sum $DEV" S2_train_sum_8602 "$PY own92a.py train $R $R86 8602 sum $DEV" \
       S2_train_own_8601 "$PY own92a.py train $R $R86 8601 own $DEV" S2_train_own_8602 "$PY own92a.py train $R $R86 8602 own $DEV" \
       S2_split_E4_T0 "$PY $A92/diag92.py split $R $R86 E4_T0 $DEV; $PY $A92/interv92.py ofield $R $R86 E4_T0 $DEV" \
       S2_split_E4_T1 "$PY $A92/diag92.py split $R $R86 E4_T1 $DEV; $PY $A92/interv92.py ofield $R $R86 E4_T1 $DEV"
# ---- S3 field evaluation, tables, local operator
TAGS="E1_T0 E1_T1 E2_T0 E2_T1 E1_T1_LMIX E1_T1_LSPOT E4_T0 E4_T1 E4_T1_LMIX4 E4_T1_LSPOT4"
j=(S3_fieldeval "$PY own92a.py fieldeval $R $R86 $DEV"); for t in $TAGS; do j+=(S3_mtables_$t "$PY own92a.py mtables $R $R86 $t"); done
pool 4 "${j[@]}"
log "FIELDEVAL $(grep '^FIELDEVAL' $O/logs/S3_fieldeval.log | tail -1 | cut -c1-200)"
run S3_local "$PY own92a.py local $R $R86"
# ---- S4 renders (WL86 ECRO render seeds)
MODELS="ecro_s8601 ecro_s8602 sum_s8601 sum_s8602 own_s8601 own_s8602"
j=()
for t in $TAGS; do
  L=$O/lists/i_$t.txt; : > "$L"
  for r in 0 1; do
    for m in $MODELS; do echo "results/tables/I_$t.npz:B_$m $((86600+r)) $SPP I92A_${m}_${t}_r$r" >> "$L"; done
    case $t in E2_T1|E4_T1) for m in $MODELS; do echo "results/tables/I_$t.npz:STALE_$m $((86600+r)) $SPP I92A_STALE_${m}_${t}_r$r" >> "$L"; done;; esac
    case $t in E4_T0|E4_T1) for k in ORACLE X_EXP; do echo "results/tables/I_$t.npz:$k $((86600+r)) $SPP I92A_${k}_${t}_r$r" >> "$L"; done;; esac
  done
  j+=(S4_render_$t "$PY $A92/diag92.py render $R $R86 $t $L")
done
pool 5 "${j[@]}"
# ---- S5 costs (alone), analysis, conditional ladder, visuals
run S5_timing "$PY own92a.py timing $R $R86"
run A_final "$PY analyze92a.py $R $R86"
OUT=$(grep '^VERDICT' $O/logs/A_final.log | tail -1 | cut -d' ' -f2)
log "VERDICT $OUT"
if [ "$OUT" = A_SUPPORTED ] || [ -n "$DEV" ]; then run S6_ladder "$PY own92a.py ladder $R $R86 $DEV"; fi
run V_viz "$PY viz92a.py $R $R86 $O/viz"
log CHAIN_DONE
