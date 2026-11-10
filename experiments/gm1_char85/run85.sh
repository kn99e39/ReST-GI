#!/bin/bash
# WL85 chain: ONE detached chain, all stages. setsid nohup bash run85.sh <root> [--dev] > <root>/chain.out 2>&1 &
# Reference renders are CPU-tracing-bound in Dr.Jit 0.4.6 (about 1-2 s of Python tracing per launch against ~0.1 s of GPU work), so they
# run as MAXJ concurrent processes (memory-limited); the timing stage runs alone.
set -u
ROOT=$1; DEV=${2:-}
. "$ROOT/env.sh"
CODE=$ROOT/code/experiments/gm1_char85
R=$ROOT/results; ST=$ROOT/chain.status
mkdir -p "$R/logs" "$R/renders"
MAXJ=${MAXJ:-5}
log() { echo "$(date -u +%FT%TZ) $*" >> "$ST"; }
pool() {
  local maxj=$1; shift; local names=()
  while [ $# -gt 0 ]; do
    local name=$1 cmd=$2; shift 2
    while [ "$(jobs -rp | wc -l)" -ge "$maxj" ]; do sleep 1; done
    log "START $name"
    ( bash -c "$cmd" > "$R/logs/$name.log" 2>&1; echo $? > "$R/logs/$name.rc" ) &
    names+=("$name")
  done
  wait
  local bad=0
  for n in "${names[@]}"; do local rc; rc=$(cat "$R/logs/$n.rc" 2>/dev/null || echo 99); log "END $n rc=$rc"; [ "$rc" != 0 ] && bad=1; done
  [ $bad -ne 0 ] && { log CHAIN_FAILED; exit 1; }
}
run() { local n=$1; shift; pool 1 "$n" "$*"; }
cd "$CODE" || exit 1
log "CHAIN_START dev=$DEV commit=$(cat $ROOT/code/COMMIT 2>/dev/null)"
SPP=8192; NR=8
[ -n "$DEV" ] && SPP=128
# ---- Stage 0: census + fixture materials; geometry buffer; validation
run S0_census "$PY census85.py $R/census $DEV"
run S1_gbuf "$PY pt85.py gbuf $ROOT"
run S2_tests "$PY tests85.py $ROOT $DEV"
g=$($PY -c "import json;print('PASS' if json.load(open('$R/tests/tests85.json'))['PASS'] else 'FAIL')"); log "GATE TESTS $g"
[ "$g" = PASS ] || { log "STOP RENDERER_VALIDATION_FAILED"; log CHAIN_DONE; exit 0; }
# ---- Stage 2 / 3: references with instrumentation (common random numbers across T0 / T1)
jobs_=()
for r in $(seq 0 $((NR-1))); do for cfg in T0 T1; do
  jobs_+=(REF_ER_${cfg}_r$r "$PY pt85.py render $ROOT $cfg ER $((85100+r)) $SPP ER_${cfg}_r$r")
  jobs_+=(REF_P_${cfg}_r$r "$PY pt85.py render $ROOT $cfg P $((85200+r)) $SPP P_${cfg}_r$r")
done; done
pool $MAXJ "${jobs_[@]}"
run S3_analyze "$PY analyze85.py $ROOT"
q=$(tail -1 "$R/logs/S3_analyze.log" | cut -d' ' -f1); log "GATE WORKLOAD $q"
# ---- Stage 4: economics (alone). Runs also for an UNSUITABLE workload (cost measurements are valid), conclusions are gated in the report.
run S4_econ "$PY econ85.py $ROOT $DEV"
run S5_viz "$PY viz85.py $ROOT"
log CHAIN_DONE
