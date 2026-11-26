#!/usr/bin/env bash
# WL114A LR3 detached, resumable chain (local RTX 5080): data (Mitsuba 3.6.4 venv) -> matched training A (D6) / B (full), seed 10202
# (one predeclared rerun of BOTH arms with seed 10203 if either collapses) -> oracle + arm evaluation -> analysis -> figures.
set -u
cd /c/Projects/ReST-GI/experiments/ecro_lr3_114a
TPY=/c/Projects/TransGI_Reproduction/.venv/Scripts/python.exe
EPY=../../external/8dna26/.venv-cu128/Scripts/python.exe
XML='C:\Users\dna10\TransGI-Reproduction-data\scenes\living-room-3\scene_transgi.xml'
R=${WL114A_ROOT:-../../results/ecro_lr3_114a/primary}
mkdir -p $R/logs $R/eval
ST=$R/chain.status
export PYTHONUNBUFFERED=1
echo "chain start $(date -Iseconds) commit $(git rev-parse HEAD) dev=${WL114A_DEV:-0}" >> $R/logs/chain_progress.log
step() {
  local name=$1; shift
  if [ -f $R/logs/$name.ok ]; then return 0; fi
  echo "RUNNING $name $(date -Iseconds)" > $ST
  "$@" > $R/logs/$name.log 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "FAILED $name rc=$rc $(date -Iseconds)" > $ST; exit 1; fi
  touch $R/logs/$name.ok; echo "$name done $(date -Iseconds)" >> $R/logs/chain_progress.log
}
if [ "${WL114A_DEV:-0}" = "1" ]; then STEPS=200; else STEPS=24000; fi
step data timeout 10800 $TPY -B gen114a.py "$XML" $R/data
train_pair() {
  local seed=$1
  ( timeout 9000 $EPY -B train114a.py $R/data $R/model_A D6 $seed $STEPS > $R/logs/train_A_$seed.log 2>&1 ) &
  local pa=$!
  ( timeout 9000 $EPY -B train114a.py $R/data $R/model_B F $seed $STEPS > $R/logs/train_B_$seed.log 2>&1 ) &
  local pb=$!
  wait $pa; local ra=$?; wait $pb; local rb=$?
  [ $ra -eq 0 ] && [ $rb -eq 0 ]
}
collapsed() { $EPY -c "import json,sys;a=json.load(open('$R/model_A/training.json'))['validation']['whole_rel_l1'];b=json.load(open('$R/model_B/training.json'))['validation']['whole_rel_l1'];sys.exit(0 if (a>=0.99 or b>=0.99) else 1)"; }
if [ ! -f $R/logs/train.ok ]; then
  echo "RUNNING train $(date -Iseconds)" > $ST
  train_pair 10202 || { echo "FAILED train 10202 $(date -Iseconds)" > $ST; exit 1; }
  if collapsed; then
    echo "collapse at seed 10202 (predeclared rerun of both arms with 10203)" >> $R/logs/chain_progress.log
    mv $R/model_A $R/model_A_s10202; mv $R/model_B $R/model_B_s10202
    train_pair 10203 || { echo "FAILED train 10203 $(date -Iseconds)" > $ST; exit 1; }
  fi
  touch $R/logs/train.ok; echo "train done $(date -Iseconds)" >> $R/logs/chain_progress.log
fi
step eval $TPY -B eval114a.py $R/data $R/eval oracle arms
step analyze $TPY -B analyze114a.py $R
$TPY -B fig114a.py $R ${WL114A_EVAL:-../../evaluation/114A} > $R/logs/figures.log 2>&1 || echo "figures failed (non-fatal)" >> $R/logs/chain_progress.log
echo "DONE $(date -Iseconds)" > $ST
echo "chain done $(date -Iseconds)" >> $R/logs/chain_progress.log
