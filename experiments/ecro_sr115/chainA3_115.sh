#!/usr/bin/env bash
# WL115 Stage A chain v3 (execution change: VRAM budget). At most two GPU processes at any time:
#   phase 1: capture (b 16) || [running con8 process finishes -> conF (b 8)]
#   phase 2: streams group 1 -> streams group 2   ||   FRESH references (after conF)
set -u
cd /c/Projects/ReST-GI/experiments/ecro_sr115
PY=../../external/8dna26/.venv-cu128/Scripts/python.exe
R=${WL115_ROOT:-../../results/ecro_sr115/primary}
ST=$R/chain.status
export PYTHONUNBUFFERED=1
echo "chainA3 start $(date -Iseconds) commit $(git rev-parse HEAD)" >> $R/logs/chain_progress.log
fail() { echo "FAILED $1 $(date -Iseconds)" > $ST; exit 1; }
nfiles() { ls $1 2>/dev/null | grep -v tmp | wc -l; }
run() { local name=$1; shift; if [ ! -f $R/logs/$name.ok ]; then "$@" > $R/logs/$name.log 2>&1 || return 1; touch $R/logs/$name.ok; echo "$name done $(date -Iseconds)" >> $R/logs/chain_progress.log; fi; }
echo "RUNNING con8 (finishing) || capture $(date -Iseconds)" > $ST
( run capture $PY -B stageA115.py capture $R ) & P2=$!
until [ $(nfiles "$R/stageA/con8/c8_*.npz") -ge 11 ]; do sleep 10; done; sleep 20
touch $R/logs/con8.ok; echo "con8 done $(date -Iseconds)" >> $R/logs/chain_progress.log
echo "RUNNING conF || capture $(date -Iseconds)" > $ST
( run conF $PY -B stageA115.py conF $R ) & P1=$!
wait $P1 || fail conF
( run refs $PY -B ref115.py $R ) & P3=$!
wait $P2 || fail capture
run tests2 $PY -B tests115.py $R/tests2 || fail tests2
echo "RUNNING streamsA || refs $(date -Iseconds)" > $ST
run streamsA_0 $PY -B stageA115.py streams $R F0 F1 SEQ_A_lift || fail streamsA_0
run streamsA_1 $PY -B stageA115.py streams $R F2 F3 SEQ_B_slide SEQ_C_independent || fail streamsA_1
touch $R/logs/streamsA.ok
wait $P3 || fail refs
echo "STAGE_A_DONE $(date -Iseconds)" > $ST
