#!/usr/bin/env bash
# WL115 Stage A detached, resumable chain (local RTX 5080). Any failure writes FAILED to chain.status and stops.
set -u
cd /c/Projects/ReST-GI/experiments/ecro_sr115
PY=../../external/8dna26/.venv-cu128/Scripts/python.exe
R=${WL115_ROOT:-../../results/ecro_sr115/primary}
mkdir -p $R/logs $R/analysis
ST=$R/chain.status
export PYTHONUNBUFFERED=1
echo "chainA start $(date -Iseconds) commit $(git rev-parse HEAD) dev=${WL115_DEV:-0}" >> $R/logs/chain_progress.log
step() {
  local name=$1; shift
  if [ -f $R/logs/$name.ok ]; then return 0; fi
  echo "RUNNING $name $(date -Iseconds)" > $ST
  "$@" > $R/logs/$name.log 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "FAILED $name rc=$rc $(date -Iseconds)" > $ST; exit 1; fi
  touch $R/logs/$name.ok
  echo "$name done $(date -Iseconds)" >> $R/logs/chain_progress.log
}
step tests $PY -B tests115.py $R
step con8 $PY -B stageA115.py con8 $R
step conF $PY -B stageA115.py conF $R
step capture $PY -B stageA115.py capture $R
step streamsA $PY -B stageA115.py streams $R
echo "STAGE_A_DONE $(date -Iseconds)" > $ST
echo "stageA streams done $(date -Iseconds)" >> $R/logs/chain_progress.log
