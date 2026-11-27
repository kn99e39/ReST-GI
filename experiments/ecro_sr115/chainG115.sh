#!/usr/bin/env bash
# WL115: after Stage D/E (chainC STAGE_DE_DONE), run Stage G cost (GPU otherwise idle) and the Stage D/E figures; no manual gates.
set -u
cd /c/Projects/ReST-GI/experiments/ecro_sr115
PY=../../external/8dna26/.venv-cu128/Scripts/python.exe
R=${WL115_ROOT:-../../results/ecro_sr115/primary}
export PYTHONUNBUFFERED=1
until grep -q "STAGE_DE_DONE" $R/chainC.status 2>/dev/null; do grep -q FAILED $R/chainC.status 2>/dev/null && exit 1; sleep 20; done
run() { local name=$1; shift; if [ ! -f $R/logs/$name.ok ]; then "$@" > $R/logs/$name.log 2>&1 || { echo "FAILED $name $(date -Iseconds)" >> $R/logs/chain_progress.log; return 1; }; touch $R/logs/$name.ok; echo "$name done $(date -Iseconds)" >> $R/logs/chain_progress.log; fi; }
run cost $PY -B cost115.py $R
run figD $PY -B fig115D.py $R ../../evaluation/115
echo "ALL_GPU_DONE $(date -Iseconds)" >> $R/logs/chain_progress.log
