#!/bin/bash
# WL81 chain: ONE detached chain covering every stage end to end (A0 -> B -> A1 data -> C -> D -> E -> F), frozen rules in
# protocol/compiled81.json + amendments. Usage: setsid nohup bash run81.sh <root> [--dev] &
# Amendment 2: independent jobs inside a stage run as parallel processes (GPU-bound after the GPU-side walk reduction); timing
# stages (E bench, F render) always run alone. A stage whose result marker already exists from an earlier run of the same frozen
# code is reused (A0, B) and logged as REUSED.
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
# par NAME1 'cmd1' NAME2 'cmd2' ... : run jobs concurrently, stop the chain if any fails
par() {
  local names=() pids=()
  while [ $# -gt 0 ]; do
    local name=$1 cmd=$2; shift 2
    log "START $name"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$R/logs/$name.gpu_before" 2>&1
    bash -c "$cmd" > "$R/logs/$name.log" 2>&1 &
    pids+=($!); names+=($name)
  done
  local bad=0
  for i in "${!pids[@]}"; do
    wait ${pids[$i]}; local rc=$?
    log "END ${names[$i]} rc=$rc"
    [ $rc -ne 0 ] && bad=1
  done
  if [ $bad -ne 0 ]; then log "CHAIN_FAILED (${names[*]})"; exit 1; fi
}
run() { local n=$1; shift; par "$n" "$*"; }
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $ROOT/code/COMMIT 2>/dev/null)"

# ---- Stage A0: reference qualification rerun with the frozen code
if [ -f "$R/A0/trunc.json" ]; then log "REUSED A0"; else
  run A0_iface "$PY refaudit81.py iface $R/A0"
  run A0_walk "$PY refaudit81.py walk $R/A0"
  par A0_slab "$PY refaudit81.py slab $R/A0" A0_trunc "$PY refaudit81.py trunc $R/A0"
fi
g=$($PY gate81.py a0 "$R/A0"); log "GATE A0 $g"
[ "$g" = QUALIFIED ] || { log "STOP REFERENCE_NOT_QUALIFIED (A0)"; log CHAIN_DONE; exit 0; }

# ---- Stage B: sampler regression on frozen WL80C parameters + horizon fix
if [ -f "$R/B/stageB.json" ]; then log "REUSED B"; else run B "$PY testsB81.py $C80 $R/B"; fi
g=$($PY gate81.py b "$R/B"); log "GATE B $g"
[ "$g" = CONTINUE ] || { log "STOP SAMPLER_OR_HORIZON (B)"; log CHAIN_DONE; exit 0; }

# ---- Stage A1: populations and reference targets (4 parallel lanes)
D="$R/data"
run A1_materials "$PY data81.py $D materials $DEV"
par A1_pairs "$PY data81.py $D pairs $DEV" \
    A1_gate "$PY data81.py $D eval gate $DEV && $PY data81.py $D highlight gate $DEV" \
    A1_test_iid "$PY data81.py $D eval test_iid $DEV && $PY data81.py $D highlight test_iid $DEV" \
    A1_test_block "$PY data81.py $D eval test_block $DEV" \
    A1_train_eval "$PY data81.py $D eval train_eval $DEV"
g=$($PY gate81.py data "$D"); log "GATE DATA $g"
[ "$g" = QUALIFIED ] || { log "STOP REFERENCE_NOT_QUALIFIED (noise floor)"; log CHAIN_DONE; exit 0; }

# ---- Stage C: energy A/B by direct fits on the gate set
par C_fit_control "$PY fit81.py direct $D $R/C/fit control gate $DEV" \
    C_fit_candidate "$PY fit81.py direct $D $R/C/fit candidate gate $DEV" \
    C_fit_unbounded "$PY fit81.py direct $D $R/C/fit unbounded gate $DEV"
run C_eval "$PY eval81.py stageC $D $R/C/fit $R/C/eval"
par C_contract_control "$PY contract81.py stageC $D $R/C/eval $R/C/contract control" \
    C_contract_candidate "$PY contract81.py stageC $D $R/C/eval $R/C/contract candidate"
q=$($PY gate81.py c "$R/C/eval" "$R/C/contract"); log "GATE C qualified=[$q]"
[ "$q" = NONE ] && { log "STOP NO_QUALIFIED_ARM (C): learned compiler NOT TRAINED"; log CHAIN_DONE; exit 0; }
P1=${q%% *}
if [ "$P1" = candidate ]; then P2=control; else P2=candidate; fi
log "PRIMARY $P1 ARMS $P1 $P2"

# ---- Stage D: learned compiler. Candidate training needs ~5 GB, so at most one candidate job runs at a time.
F="$R/D/fit"
par D_train_candidate_8101 "$PY fit81.py compiler $D $F candidate 8101 $DEV" \
    D_train_control_8101 "$PY fit81.py compiler $D $F control 8101 $DEV" \
    D_train_control_8102 "$PY fit81.py compiler $D $F control 8102 $DEV"
if [ -z "$DEV" ] && [ "$P1" = control ]; then
  par D_train_candidate_8102 "$PY fit81.py compiler $D $F candidate 8102" \
      D_train_control_n750 "$PY fit81.py compiler $D $F control 8101 750" \
      D_train_control_n1500 "$PY fit81.py compiler $D $F control 8101 1500"
else
  run D_train_candidate_8102 "$PY fit81.py compiler $D $F candidate 8102 $DEV"
  if [ -z "$DEV" ]; then
    run D_train_candidate_n750 "$PY fit81.py compiler $D $F candidate 8101 750"
    run D_train_candidate_n1500 "$PY fit81.py compiler $D $F candidate 8101 1500"
  fi
fi
par D_oracle_candidate_iid "$PY fit81.py oracle $D $F candidate test_iid $F/compiler_candidate_s8101.pt $DEV" \
    D_oracle_candidate_block "$PY fit81.py oracle $D $F candidate test_block $F/compiler_candidate_s8101.pt $DEV" \
    D_oracle_control_iid "$PY fit81.py oracle $D $F control test_iid $F/compiler_control_s8101.pt $DEV" \
    D_oracle_control_block "$PY fit81.py oracle $D $F control test_block $F/compiler_control_s8101.pt $DEV"
run D_eval "$PY eval81.py stageD $D $F $R/D/eval $P1 $P2"
par D_contract_$P1 "$PY contract81.py stageD $D $R/D/eval $R/D/contract $P1" \
    D_contract_$P2 "$PY contract81.py stageD $D $R/D/eval $R/D/contract $P2" \
    E_compress "$PY compress81.py $D $R/D/eval $R/E $P1"
g=$($PY gate81.py d "$R/D/eval" "$R/D/contract" $P1); log "GATE D $g"

# ---- Stage E: GPU cost (valid closures only; runs alone)
if [ "$g" != FAIL ]; then
  run E_bench "$PY bench81.py $D $R/D/eval $F $R/E $P1 $P2 $DEV"
else
  log "SKIP E: primary arm failed the Stage D physical contract"
fi
# ---- Stage F rendering (learned rule + physical validity required; runs alone)
if [ "$g" = LEARNED_PHYSICAL_PASS ]; then
  run F_validate "$PY render81.py validate $R/F/validation $R/D/eval/states_${P1}_s8101_test_iid.npy"
  if $PY -c "import json,sys; sys.exit(0 if json.load(open('$R/F/validation/validation.json'))['PASS'] else 1)"; then
    run F_render "$PY render81.py render $D $R/D/eval $R/F $P1 $P2 $DEV"
  else
    log "STOP F: renderer validation failed -> RENDERING_UTILITY UNVERIFIED"
  fi
else
  log "SKIP F: learned rule or physical validity not passed (compression result is descriptive only)"
fi
log CHAIN_DONE
