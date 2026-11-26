#!/usr/bin/env bash
# WL115: waits for the Stage A chain (chain.status STAGE_A_DONE), then renders the candidate-independent FRESH references.
set -u
cd /c/Projects/ReST-GI/experiments/ecro_sr115
PY=../../external/8dna26/.venv-cu128/Scripts/python.exe
R=${WL115_ROOT:-../../results/ecro_sr115/primary}
ST=$R/chain.status
export PYTHONUNBUFFERED=1
while true; do
  s=$(cat $ST 2>/dev/null)
  case "$s" in STAGE_A_DONE*) break;; FAILED*) echo "chainR: stage A failed, not starting" >> $R/logs/chain_progress.log; exit 1;; esac
  sleep 30
done
if [ ! -f $R/logs/refs.ok ]; then
  echo "RUNNING refs $(date -Iseconds)" > $ST
  $PY -B ref115.py $R > $R/logs/refs.log 2>&1 || { echo "FAILED refs $(date -Iseconds)" > $ST; exit 1; }
  touch $R/logs/refs.ok; echo "refs done $(date -Iseconds)" >> $R/logs/chain_progress.log
fi
echo "REFS_DONE $(date -Iseconds)" > $ST
