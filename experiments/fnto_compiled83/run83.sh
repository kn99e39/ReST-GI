#!/bin/bash
# WL83 chain: ONE detached chain covering every stage end to end (protocol/compiled83.json).
#   setsid nohup bash run83.sh <root> [--dev] > <root>/chain.out 2>&1 &
set -u
ROOT=$1
DEV=${2:-}
. "$ROOT/env.sh"
CODE=$ROOT/code/experiments/fnto_compiled83
C82=$ROOT/code/experiments/fnto_compiled82
R=$ROOT/results
ST=$ROOT/chain.status
W81=$HOME/rest_compiled81/results/data
W82D=$HOME/rest_compiled82/results/data
W82F=$HOME/rest_compiled82/results/fit
export W81
D=$R/data; FD=$R/fit; EV=$R/eval; CK=$R/compiler
mkdir -p "$R/logs" "$D" "$FD" "$EV" "$CK"
log() { echo "$(date -u +%FT%TZ) $*" >> "$ST"; }
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
DFS=12000; [ -n "$DEV" ] && DFS=300 && export W83_DF_STEPS=300
STEPS_JSON='{"C0_cert_metric":12000,"C0_unb_metric":12000,"E_cert_metric":12000,"E_unb_metric":12000,"Einc_unb_metric":12000,"ER_cert_metric":12000,"ER_unb_metric":12000}'

# ---- Stage A (part 1): reference reproduction, WL82 focused tests and contract on the stored WL82 fits, new population
pool 3 A_repro "$PY $C82/data82.py $R/A/repro $W81 repro $DEV" \
       A_tests82 "$PY $C82/contract82.py tests $W82D $W81 $R/A/tests82 $DEV" \
       D_materials "$PY data83.py $D $W81 materials $DEV"
run A_contract82 "$PY $C82/contract82.py contract $W82D $W82F $R/A/contract82 '$STEPS_JSON' $DEV"
run A_independence "$PY tests83.py independence $D $W81 $R/A/independence"

# ---- data: response sets (train83 in 10 shards, val83, test83_*), evaluation sets (shards), highlight sets (shards)
jobs_=()
for i in $(seq 0 9); do jobs_+=(D_fit_train83_$i "$PY data83.py $D $W81 fitset train83 --shard=$i/10 $DEV"); done
for s in val83 test83_iid test83_block test83_hard; do jobs_+=(D_fit_$s "$PY data83.py $D $W81 fitset $s $DEV"); done
pool 13 "${jobs_[@]}"
jobs_=()
for sk in test83_iid:4 test83_block:3 test83_hard:2 val83:3; do s=${sk%%:*}; k=${sk##*:}
  for i in $(seq 0 $((k-1))); do jobs_+=(D_ev_${s}_$i "$PY data83.py $D $W81 evalset $s --shard=$i/$k $DEV"); done; done
pool 13 "${jobs_[@]}"
pool 4 D_evm_test83_iid "$PY data83.py $D $W81 evalmerge test83_iid 4 $DEV" D_evm_test83_block "$PY data83.py $D $W81 evalmerge test83_block 3 $DEV" \
       D_evm_test83_hard "$PY data83.py $D $W81 evalmerge test83_hard 2 $DEV" D_evm_val83 "$PY data83.py $D $W81 evalmerge val83 3 $DEV"
jobs_=()
for sk in test83_iid:8 test83_hard:4; do s=${sk%%:*}; k=${sk##*:}
  for i in $(seq 0 $((k-1))); do jobs_+=(D_hl_${s}_$i "$PY data83.py $D $W81 hlset $s --shard=$i/$k $DEV"); done; done
pool 12 "${jobs_[@]}"
pool 2 D_hlm_test83_iid "$PY data83.py $D $W81 hlmerge test83_iid 8" D_hlm_test83_hard "$PY data83.py $D $W81 hlmerge test83_hard 4"

# ---- Stage A (part 2): stored WL82 direct fits through the WL83 evaluator; gate
run A_eval_wl82 "W83_UNPATCHED=1 $PY eval83.py $D $W81 $FD $W82F wl82_test_iid $R/A/eval_unpatched $DEV"
g=$($PY gate83.py stageA "$R"); log "GATE A $g"
[ "$g" = QUALIFIED ] || { log "STOP STAGE_A_NOT_QUALIFIED"; log CHAIN_DONE; exit 0; }

# ---- runtime correctness (pre-evidence) and renderer validation
run T_runtime "$PY tests83.py runtime $R/runtime $DEV"
g=$($PY -c "import json;print('PASS' if json.load(open('$R/runtime/runtime_tests.json'))['PASS'] else 'FAIL')"); log "GATE RUNTIME $g"
[ "$g" = PASS ] || { log "STOP RUNTIME_TESTS_FAILED"; log CHAIN_DONE; exit 0; }
run R_validate "$PY render83.py validate $R/render"
g=$($PY -c "import json;print('PASS' if json.load(open('$R/render/validation.json'))['PASS'] else 'FAIL')"); log "GATE RENDER_VALIDATION $g"
[ "$g" = PASS ] || { log "STOP RENDERER_VALIDATION_FAILED"; log CHAIN_DONE; exit 0; }

# ---- compiler training (3 arms x 2 seeds) and direct fits (3 sets x 3 arms), 3 concurrent processes
jobs_=()
for a in E ER C0; do for sd in 8301 8302; do jobs_+=(TR_${a}_s$sd "$PY compiler83.py train $D $a $sd $CK $DEV"); done; done
for s in test83_iid test83_hard test83_block; do for a in E ER C0; do
  jobs_+=(DF_${s}_$a "$PY fit83.py $D $s $a cert metric $DFS generic $FD"); done; done
pool 3 "${jobs_[@]}"

# ---- learned closures for every evaluated set
jobs_=()
for a in C0 E ER; do for sd in 8301 8302; do
  jobs_+=(PR_${a}_s$sd "for s in test83_iid test83_block test83_hard val83 train_eval wl82_test_iid real83; do $PY compiler83.py predict $D $W81 $CK $a $sd \$s $FD || exit 1; done"); done; done
pool 3 "${jobs_[@]}"

# ---- evaluation
jobs_=()
for s in test83_iid test83_block test83_hard val83 train_eval wl82_test_iid real83; do jobs_+=(EV_$s "$PY eval83.py $D $W81 $FD $W82F $s $EV $DEV"); done
pool 3 "${jobs_[@]}"

# ---- Dr.Jit passivity audit of every DF / learned closure
run K_audit "$PY tests83.py audit $D $FD $R/runtime $DEV"

# ---- GPU economics (alone), rendering (alone), visuals, verdict
run B_bench "$PY bench83.py $D $W81 $FD $CK $EV $R/bench $DEV"
run F_render "$PY render83.py render $D $W81 $FD $W82F $R/render $DEV"
run V_viz "$PY viz83.py $ROOT $W81 $DEV"
run X_verdict "$PY gate83.py verdict $R $R/verdict83.json"
log "OUTCOME $($PY -c "import json;print(json.load(open('$R/verdict83.json'))['OUTCOME'])" 2>/dev/null)"
log CHAIN_DONE
