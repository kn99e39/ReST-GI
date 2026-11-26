#!/usr/bin/env bash
# WL115 Stage B (frozen candidate CAR-16 oracle qualification): two GPU processes at a time.
set -u
cd /c/Projects/ReST-GI/experiments/ecro_sr115
PY=../../external/8dna26/.venv-cu128/Scripts/python.exe
R=${WL115_ROOT:-../../results/ecro_sr115/primary}
ST=$R/chain.status
export PYTHONUNBUFFERED=1
echo "chainB start $(date -Iseconds) commit $(git rev-parse HEAD)" >> $R/logs/chain_progress.log
fail() { echo "FAILED $1 $(date -Iseconds)" > $ST; exit 1; }
run() { local name=$1; shift; if [ ! -f $R/logs/$name.ok ]; then "$@" > $R/logs/$name.log 2>&1 || return 1; touch $R/logs/$name.ok; echo "$name done $(date -Iseconds)" >> $R/logs/chain_progress.log; fi; }
echo "RUNNING stageB con $(date -Iseconds)" > $ST
( run conB_0 $PY -B stageB115.py con $R F4 N1 N2 N3 N4 SEQ_D_independent N5 ) & P1=$!
( run conB_1 $PY -B stageB115.py con $R N6 N7 N8 SEQ_E_fresh ) & P2=$!
wait $P1 || fail conB_0; wait $P2 || fail conB_1
echo "RUNNING stageB streams $(date -Iseconds)" > $ST
( run streamsB_0 $PY -B stageB115.py streams $R F4 N1 N2 N3 N4 SEQ_D_independent N5 ) & P1=$!
( run streamsB_1 $PY -B stageB115.py streams $R N6 N7 N8 SEQ_E_fresh ) & P2=$!
wait $P1 || fail streamsB_0; wait $P2 || fail streamsB_1
run analyzeB $PY -B analyzeB115.py $R || fail analyzeB
echo "STAGE_B_DONE $(date -Iseconds)" > $ST
