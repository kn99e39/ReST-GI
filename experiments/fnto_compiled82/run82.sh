#!/bin/bash
# WL82 chain: ONE detached chain covering every stage end to end (protocol/compiled82.json).
#   setsid nohup bash run82.sh <root> [--dev] > <root>/chain.out 2>&1 &
# Stages: R reference reproduction -> D data (fit sets, real set, independent albedo sets) -> T focused tests -> FG gate fits ->
#         EG gate eval -> CONV convergence rule (+ escalation) -> FH holdout / real fits -> EH evals -> K physical contract -> V visuals -> X verdict
set -u
ROOT=$1
DEV=${2:-}
. "$ROOT/env.sh"
CODE=$ROOT/code/experiments/fnto_compiled82
R=$ROOT/results
ST=$ROOT/chain.status
W81=$HOME/rest_compiled81/results/data
W81C=$HOME/rest_compiled81/results/C
D=$R/data; FD=$R/fit; EV=$R/eval
mkdir -p "$R/logs" "$D" "$FD" "$EV"
MAXJ=${MAXJ:-2}
log() { echo "$(date -u +%FT%TZ) $*" >> "$ST"; }
# pool MAXJ NAME1 'cmd1' NAME2 'cmd2' ... : run with at most MAXJ concurrent jobs; stop the chain if any fails
pool() {
  local maxj=$1; shift
  local names=()
  while [ $# -gt 0 ]; do
    local name=$1 cmd=$2; shift 2
    while [ "$(jobs -rp | wc -l)" -ge "$maxj" ]; do sleep 2; done
    log "START $name"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$R/logs/$name.gpu_before" 2>&1
    ( bash -c "$cmd" > "$R/logs/$name.log" 2>&1; echo $? > "$R/logs/$name.rc" ) &
    names+=("$name")
  done
  wait
  local bad=0
  for n in "${names[@]}"; do
    local rc; rc=$(cat "$R/logs/$n.rc" 2>/dev/null || echo 99)
    log "END $n rc=$rc"
    [ "$rc" != 0 ] && bad=1
  done
  if [ $bad -ne 0 ]; then log "CHAIN_FAILED"; exit 1; fi
}
run() { local n=$1; shift; pool 1 "$n" "$*"; }
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $ROOT/code/COMMIT 2>/dev/null)"

# ---- R: reference reproduction (stop condition)
run R_repro "$PY data82.py $D $W81 repro $DEV"
g=$($PY gate82.py repro "$D"); log "GATE R $g"
[ "$g" = REPRODUCED ] || { log "STOP REFERENCE_NOT_REPRODUCED"; log CHAIN_DONE; exit 0; }

# ---- D: data (fit sets + real evaluation set in parallel, then independent albedo sets in shards)
pool 6 D_fit_gate "$PY data82.py $D $W81 fitset gate $DEV" D_fit_test_iid "$PY data82.py $D $W81 fitset test_iid $DEV" \
       D_fit_test_block "$PY data82.py $D $W81 fitset test_block $DEV" D_fit_real "$PY data82.py $D $W81 fitset real $DEV" \
       D_realeval "$PY data82.py $D $W81 realeval $DEV"
K=4
jobs_=()
for s in gate test_iid test_block real; do for i in $(seq 0 $((K-1))); do
  jobs_+=(D_alb_${s}_$i "$PY data82.py $D $W81 albset $s --shard=$i/$K $DEV"); done; done
pool 13 "${jobs_[@]}"
pool 4 D_albm_gate "$PY data82.py $D $W81 albmerge gate $K" D_albm_test_iid "$PY data82.py $D $W81 albmerge test_iid $K" \
       D_albm_test_block "$PY data82.py $D $W81 albmerge test_block $K" D_albm_real "$PY data82.py $D $W81 albmerge real $K"

# ---- T: focused tests (stop condition)
run T_tests "$PY contract82.py tests $D $W81 $R/tests $DEV"
g=$($PY gate82.py tests "$R/tests"); log "GATE T $g"
[ "$g" = PASS ] || { log "STOP TESTS_FAILED"; log CHAIN_DONE; exit 0; }

ARMS=$($PY -c "import json;print(' '.join('_'.join(a) for a in json.load(open('protocol/compiled82.json'))['arms']['fit_arms']))")
STEP1=12000; STEP0=6000; STEP2=24000
if [ -n "$DEV" ]; then STEP1=400; STEP0=200; STEP2=800; export W82_STEP1=400 W82_STEP0=200 W82_STEP2=800; fi
fitcmd() { local set=$1 tag=$2 steps=$3 init=$4; local a m o; IFS=_ read a m o <<< "$tag"; echo "$PY fit82.py $D $set $a $m $o $steps $init $FD $DEV"; }

# ---- FG: gate (development) fits: primary, convergence and perturbed-init runs of every arm
jobs_=()
for t in $ARMS; do
  jobs_+=(FG_${t}_p "$(fitcmd gate $t $STEP1 generic)" FG_${t}_c "$(fitcmd gate $t $STEP0 generic)" FG_${t}_r "$(fitcmd gate $t $STEP1 reinit)")
done
pool $MAXJ "${jobs_[@]}"
run EG_eval "$PY eval82.py $D $W81 $W81C $FD gate $EV $DEV"

# ---- CONV: convergence rule (predeclared); escalation of non-converged arms to 24000 steps
esc=$($PY gate82.py conv $EV/eval82_gate.json $R/steps.json); log "GATE CONV escalate=[$esc]"
if [ "$esc" != NONE ]; then
  jobs_=(); for t in $esc; do jobs_+=(FG_${t}_e "$(fitcmd gate $t $STEP2 generic)"); done
  pool $MAXJ "${jobs_[@]}"
  run EG_eval2 "$PY eval82.py $D $W81 $W81C $FD gate $EV $DEV"
  $PY gate82.py conv2 $EV/eval82_gate.json $R/steps.json >> "$ST"
fi
stp() { $PY -c "import json;print(json.load(open('$R/steps.json'))['arms']['$1']['steps'])"; }

# ---- FH: holdout and real-material fits (frozen arms and steps)
jobs_=()
for s in test_iid test_block real; do for t in $ARMS; do
  jobs_+=(FH_${s}_${t} "$(fitcmd $s $t $(stp $t) generic)"); done; done
pool $MAXJ "${jobs_[@]}"
pool 3 EH_test_iid "$PY eval82.py $D $W81 $W81C $FD test_iid $EV $DEV" EH_test_block "$PY eval82.py $D $W81 $W81C $FD test_block $EV $DEV" \
       EH_real "$PY eval82.py $D $W81 $W81C $FD real $EV $DEV"

# ---- K: physical contract on the primary fits of every set
STEPS_JSON=$($PY -c "import json;print(json.dumps({t:v['steps'] for t,v in json.load(open('$R/steps.json'))['arms'].items()}))")
run K_contract "$PY contract82.py contract $D $FD $R/contract '$STEPS_JSON' $DEV"

# ---- V: visuals, X: verdict
run V_viz "$PY viz82.py $ROOT $W81 $DEV"
run X_verdict "$PY gate82.py verdict $R $R/verdict82.json"
log "OUTCOME $(python3 -c "import json;print(json.load(open('$R/verdict82.json'))['OUTCOME'])" 2>/dev/null)"
log CHAIN_DONE
