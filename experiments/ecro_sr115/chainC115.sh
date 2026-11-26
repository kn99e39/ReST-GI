#!/usr/bin/env bash
# WL115 Stage C-E chain: D115 generation (starts once the first Stage B construction process has finished; at most two GPU processes
# with Stage B), then - only if Stage B is QUALIFIED and the dev training probe passed (logs/train_go) - CAR1 and CAR2 training
# (sequential: host / GPU memory), Stage D / E readouts in two processes, rebuild check, analysis.
set -u
cd /c/Projects/ReST-GI/experiments/ecro_sr115
PY=../../external/8dna26/.venv-cu128/Scripts/python.exe
R=${WL115_ROOT:-../../results/ecro_sr115/primary}
export PYTHONUNBUFFERED=1
echo "chainC start $(date -Iseconds) commit $(git rev-parse HEAD)" >> $R/logs/chain_progress.log
fail() { echo "FAILED $1 $(date -Iseconds)" >> $R/logs/chain_progress.log; echo "FAILED $1" > $R/chainC.status; exit 1; }
run() { local name=$1; shift; if [ ! -f $R/logs/$name.ok ]; then echo "RUNNING $name $(date -Iseconds)" > $R/chainC.status; "$@" > $R/logs/$name.log 2>&1 || return 1; touch $R/logs/$name.ok; echo "$name done $(date -Iseconds)" >> $R/logs/chain_progress.log; fi; }
until [ -f $R/logs/conB_0.ok ]; do sleep 15; done
run d115_train $PY -B data115.py $R train 0 192 || fail d115_train
run d115_val $PY -B data115.py $R val 0 16 || fail d115_val
until [ -f $R/logs/analyzeB.ok ]; do grep -q FAILED $R/chain.status && fail stageB; sleep 15; done
Q=$($PY -c "import json;print(json.load(open('$R/analysis/stageB_decision.json'))['QUALIFIED'])")
echo "stageB QUALIFIED=$Q" >> $R/logs/chain_progress.log
if [ "$Q" != "True" ]; then echo "STOPPED_NOT_QUALIFIED" > $R/chainC.status; exit 0; fi
until [ -f $R/logs/train_go ]; do sleep 15; done
run train_CAR1 $PY -B train115.py $R CAR1 || fail train_CAR1
run train_CAR2 $PY -B train115.py $R CAR2 || fail train_CAR2
echo "RUNNING stageD streams $(date -Iseconds)" > $R/chainC.status
( run streamsD_0 $PY -B evalD115.py streams $R F4 N1 N2 N3 N4 SEQ_D_independent N5 N6 N7 N8 SEQ_E_fresh ) & P1=$!
( run streamsD_1 $PY -B evalD115.py streams $R F0 F1 F2 F3 SEQ_A_lift SEQ_B_slide SEQ_C_independent ) & P2=$!
wait $P1 || fail streamsD_0; wait $P2 || fail streamsD_1
run rebuild $PY -B evalD115.py rebuild $R || fail rebuild
run analyzeD $PY -B analyzeD115.py $R || fail analyzeD
echo "STAGE_DE_DONE $(date -Iseconds)" > $R/chainC.status
