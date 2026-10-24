#!/usr/bin/env bash
# WL42 server orchestrator (LabServer63): everything after the K_r1 training launch runs without a session, in the
# order fixed by protocol fnto_r1:
#   1. wait for K_r1 training (K_r1/train.json; a vanished trainer without it = failure)
#   2. Gate 4c: spectral radius of the composed map at T0 for K_r1 (r1_rho.py)
#   3. ONLY if rho_T0 >= 1: train the declared conditional variant K_r1P (--passivity 1.0), then its Gate 4c
#   4. wait for the amendment-2 references (a2_refs.status = done)
#   5. r1_eval run for K_r1, then for K_r1P if it was trained
# Status: ~/rest_fnto_r1/launcher/after_train.status ; log of steps: after_train_runs.log
# Launch: setsid -f nohup bash r1_after_train.sh <K_r1 trainer pid> > ~/rest_fnto_r1/launcher/after_train.stdout.log 2>&1 < /dev/null
set -u
export RNA_DIR=$HOME/NeuralShaderDemo/external/relightable-neural-assets; PY=$RNA_DIR/.venv/bin/python
H=$HOME/rest_fnto_r1; X=$H/repo/experiments/fnto_lego_r1; O=$H/repo/results/fnto_lego_r1; L=$H/launcher; TPID=$1
st() { echo "$1 $(date -Is)" > $L/after_train.status; }
log() { echo "$1 $(date -Is)" >> $L/after_train_runs.log; }
run() { log "$1 start"; "${@:2}" > $L/$1.log 2>&1; rc=$?; log "$1 rc=$rc"; [ $rc = 0 ] || { st "failed: $1"; exit $rc; }; }
cd $X
st "waiting: K_r1 training"
while kill -0 $TPID 2>/dev/null; do sleep 60; done
[ -f $O/K_r1/train.json ] || { st "failed: K_r1 training ended without train.json"; exit 1; }
log "K_r1 training done"
st "running: rho K_r1"; run rho_K_r1 $PY r1_rho.py $O/K_r1/kernels.pt $O/K_r1/rho.json
KS="K_r1"
if $PY -c "import json,sys; sys.exit(0 if json.load(open('$O/K_r1/rho.json'))['T0'] >= 1 else 1)"; then
  log "rho_T0(K_r1) >= 1 -> conditional variant K_r1P"
  st "running: train K_r1P"; run train_K_r1P $PY -u r1_train.py --out $O/K_r1P --passivity 1.0
  st "running: rho K_r1P"; run rho_K_r1P $PY r1_rho.py $O/K_r1P/kernels.pt $O/K_r1P/rho.json
  KS="K_r1 K_r1P"
fi
st "waiting: amendment-2 references"
while true; do s=$(cat $L/a2_refs.status 2>/dev/null); case "$s" in done*) break;; failed*) st "failed: references ($s)"; exit 1;; esac; sleep 60; done
log "references done"
for k in $KS; do st "running: eval $k"; run eval_$k $PY -u r1_eval.py run --kernels $O/$k/kernels.pt --out $O/eval_$k; done
st "done ($KS)"
