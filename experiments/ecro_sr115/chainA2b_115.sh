#!/usr/bin/env bash
# WL115 Stage A2 (post hoc angular mechanism diagnostic): after the Stage A streams (two GPU processes at most with the refs).
set -u
cd /c/Projects/ReST-GI/experiments/ecro_sr115
PY=../../external/8dna26/.venv-cu128/Scripts/python.exe
R=${WL115_ROOT:-../../results/ecro_sr115/primary}
export PYTHONUNBUFFERED=1
until [ -f $R/logs/streamsA.ok ]; do grep -q FAILED $R/chain.status && exit 1; sleep 15; done
run() { local name=$1; shift; if [ ! -f $R/logs/$name.ok ]; then "$@" > $R/logs/$name.log 2>&1 || { echo "FAILED $name $(date -Iseconds)" >> $R/logs/chain_progress.log; exit 1; }; touch $R/logs/$name.ok; echo "$name done $(date -Iseconds)" >> $R/logs/chain_progress.log; fi; }
run conA2 $PY -B stageA2_115.py con $R
run streamsA2 $PY -B stageA2_115.py streams $R
echo "A2_DONE $(date -Iseconds)" >> $R/logs/chain_progress.log
