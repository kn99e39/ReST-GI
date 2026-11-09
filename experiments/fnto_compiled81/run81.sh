#!/bin/bash
# WL81 chain: ONE detached chain covering every stage end to end (A0 -> B -> A1 data -> C -> D -> E -> F), frozen rules in
# protocol/compiled81.json + amendment1_candidate.json. Usage: setsid nohup bash run81.sh <root> [--dev] &
# <root>/code = git archive of the frozen commit, <root>/env.sh = environment, results in <root>/results, status in <root>/chain.status
set -u
ROOT=$1
DEV=${2:-}
. "$ROOT/env.sh"
CODE=$ROOT/code/experiments/fnto_compiled81
R=$ROOT/results
ST=$ROOT/chain.status
C80=$HOME/rest_compiled80c/results/smoke
mkdir -p "$R/logs"
log() { echo "$(date -u +%FT%TZ) $*" >> "$ST"; }
run() {
  local name=$1; shift
  log "START $name"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >> "$R/logs/$name.gpu_before" 2>&1
  "$@" > "$R/logs/$name.log" 2>&1
  local rc=$?
  log "END $name rc=$rc"
  if [ $rc -ne 0 ]; then log "CHAIN_FAILED $name"; exit 1; fi
}
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $ROOT/code/COMMIT 2>/dev/null)"

# ---- Stage A0: reference qualification rerun with the frozen code
run A0_iface $PY refaudit81.py iface "$R/A0"
run A0_walk $PY refaudit81.py walk "$R/A0"
run A0_slab $PY refaudit81.py slab "$R/A0"
run A0_trunc $PY refaudit81.py trunc "$R/A0"
g=$($PY gate81.py a0 "$R/A0"); log "GATE A0 $g"
[ "$g" = QUALIFIED ] || { log "STOP REFERENCE_NOT_QUALIFIED (A0)"; log CHAIN_DONE; exit 0; }

# ---- Stage B: sampler regression on frozen WL80C parameters + horizon fix
run B $PY testsB81.py "$C80" "$R/B"
g=$($PY gate81.py b "$R/B"); log "GATE B $g"
[ "$g" = CONTINUE ] || { log "STOP SAMPLER_OR_HORIZON (B)"; log CHAIN_DONE; exit 0; }

# ---- Stage A1: populations and reference targets
run A1_materials $PY data81.py "$R/data" materials $DEV
run A1_pairs $PY data81.py "$R/data" pairs $DEV
for s in gate test_iid test_block train_eval; do run A1_eval_$s $PY data81.py "$R/data" eval $s $DEV; done
for s in gate test_iid; do run A1_hl_$s $PY data81.py "$R/data" highlight $s $DEV; done
g=$($PY gate81.py data "$R/data"); log "GATE DATA $g"
[ "$g" = QUALIFIED ] || { log "STOP REFERENCE_NOT_QUALIFIED (noise floor)"; log CHAIN_DONE; exit 0; }

# ---- Stage C: energy A/B by direct fits on the gate set
for a in control candidate unbounded; do run C_fit_$a $PY fit81.py direct "$R/data" "$R/C/fit" $a gate $DEV; done
run C_eval $PY eval81.py stageC "$R/data" "$R/C/fit" "$R/C/eval"
run C_contract $PY contract81.py stageC "$R/data" "$R/C/eval" "$R/C/contract"
q=$($PY gate81.py c "$R/C/eval" "$R/C/contract"); log "GATE C qualified=[$q]"
[ "$q" = NONE ] && { log "STOP NO_QUALIFIED_ARM (C): learned compiler NOT TRAINED"; log CHAIN_DONE; exit 0; }
P1=${q%% *}
if [ "$P1" = candidate ]; then ARMS="candidate control"; else ARMS="control candidate"; fi
log "PRIMARY $P1 ARMS $ARMS"

# ---- Stage D: learned compiler (both arms, 2 seeds), material-count scaling (primary), oracles, evaluation, contract
for a in $ARMS; do for s in 8101 8102; do run D_train_${a}_$s $PY fit81.py compiler "$R/data" "$R/D/fit" $a $s $DEV; done; done
if [ -z "$DEV" ]; then for n in 750 1500; do run D_train_${P1}_n$n $PY fit81.py compiler "$R/data" "$R/D/fit" $P1 8101 $n; done; fi
for a in $ARMS; do for s in test_iid test_block; do run D_oracle_${a}_$s $PY fit81.py oracle "$R/data" "$R/D/fit" $a $s "$R/D/fit/compiler_${a}_s8101.pt" $DEV; done; done
run D_eval $PY eval81.py stageD "$R/data" "$R/D/fit" "$R/D/eval" $ARMS
run D_contract $PY contract81.py stageD "$R/data" "$R/D/eval" "$R/D/contract" $ARMS
g=$($PY gate81.py d "$R/D/eval" "$R/D/contract" $P1); log "GATE D $g"

# ---- Stage E: GPU cost (valid closures only)
if [ "$g" != FAIL ]; then
  run E_bench $PY bench81.py "$R/data" "$R/D/eval" "$R/D/fit" "$R/E" $ARMS $DEV
else
  log "SKIP E: primary arm failed the Stage D physical contract"
fi
# ---- Compression comparator and Stage F rendering (learned rule + physical validity required)
if [ "$g" = LEARNED_PHYSICAL_PASS ]; then
  run E_compress $PY compress81.py "$R/data" "$R/D/eval" "$R/E" $P1
  run F_validate $PY render81.py validate "$R/F/validation" "$R/D/eval/states_${P1}_s8101_test_iid.npy"
  if $PY -c "import json,sys; sys.exit(0 if json.load(open('$R/F/validation/validation.json'))['PASS'] else 1)"; then
    run F_render $PY render81.py render "$R/data" "$R/D/eval" "$R/F" $ARMS $DEV
  else
    log "STOP F: renderer validation failed -> RENDERING_UTILITY UNVERIFIED"
  fi
else
  log "SKIP compression / F: learned rule or physical validity not passed"
fi
log CHAIN_DONE
