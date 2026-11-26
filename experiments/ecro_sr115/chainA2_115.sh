#!/usr/bin/env bash
# WL115 Stage A chain v2 (execution change for GPU utilisation, no scientific change): independent stages run concurrently.
#   branch 1: wait for the running con8 process (11 files) -> conF (32 x 32 contact constructions)
#   branch 2: capture (camera-conditioned diagnostic; independent of the constructions) -> FRESH references (after con8, memory)
#   then: focused tests again (grouped readout) -> readout streams in two processes -> STAGE_A_DONE
set -u
cd /c/Projects/ReST-GI/experiments/ecro_sr115
PY=../../external/8dna26/.venv-cu128/Scripts/python.exe
R=${WL115_ROOT:-../../results/ecro_sr115/primary}
ST=$R/chain.status
export PYTHONUNBUFFERED=1
echo "chainA2 start $(date -Iseconds) commit $(git rev-parse HEAD)" >> $R/logs/chain_progress.log
fail() { echo "FAILED $1 $(date -Iseconds)" > $ST; exit 1; }
nfiles() { ls $1 2>/dev/null | grep -v tmp | wc -l; }
(
  until [ $(nfiles "$R/stageA/con8/c8_*.npz") -ge 11 ]; do sleep 10; done; sleep 20
  touch $R/logs/con8.ok; echo "con8 done $(date -Iseconds)" >> $R/logs/chain_progress.log
  if [ ! -f $R/logs/conF.ok ]; then $PY -B stageA115.py conF $R > $R/logs/conF.log 2>&1 || exit 1; touch $R/logs/conF.ok; echo "conF done $(date -Iseconds)" >> $R/logs/chain_progress.log; fi
) & B1=$!
(
  if [ ! -f $R/logs/capture.ok ]; then $PY -B stageA115.py capture $R > $R/logs/capture.log 2>&1 || exit 1; touch $R/logs/capture.ok; echo "capture done $(date -Iseconds)" >> $R/logs/chain_progress.log; fi
  until [ -f $R/logs/con8.ok ]; do sleep 10; done
  if [ ! -f $R/logs/refs.ok ]; then $PY -B ref115.py $R > $R/logs/refs.log 2>&1 || exit 1; touch $R/logs/refs.ok; echo "refs done $(date -Iseconds)" >> $R/logs/chain_progress.log; fi
) & B2=$!
echo "RUNNING con8/conF || capture/refs $(date -Iseconds)" > $ST
wait $B1 || fail branch1
if [ ! -f $R/logs/tests2.ok ]; then $PY -B tests115.py $R/tests2 > $R/logs/tests2.log 2>&1 || fail tests2; touch $R/logs/tests2.ok; fi
if [ ! -f $R/logs/streamsA.ok ]; then
  echo "RUNNING streamsA (2 processes) || refs $(date -Iseconds)" > $ST
  $PY -B stageA115.py streams $R F0 F1 SEQ_A_lift > $R/logs/streamsA_0.log 2>&1 & S1=$!
  $PY -B stageA115.py streams $R F2 F3 SEQ_B_slide SEQ_C_independent > $R/logs/streamsA_1.log 2>&1 & S2=$!
  wait $S1 || fail streamsA_0; wait $S2 || fail streamsA_1
  touch $R/logs/streamsA.ok; echo "streamsA done $(date -Iseconds)" >> $R/logs/chain_progress.log
fi
wait $B2 || fail branch2
echo "STAGE_A_DONE $(date -Iseconds)" > $ST
