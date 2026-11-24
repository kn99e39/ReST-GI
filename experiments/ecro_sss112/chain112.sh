#!/usr/bin/env bash
# WL112 detached, resumable end-to-end chain (local RTX 5080; launched hidden from PowerShell).
# Every step writes its own log; any failure writes FAILED to chain.status and stops. Steps skip completed outputs.
set -u
cd /c/Projects/ReST-GI/experiments/ecro_sss112
PY=../../external/8dna26/.venv-cu128/Scripts/python.exe
R=../../results/ecro_sss112/primary
mkdir -p $R/logs
ST=$R/chain.status
unset WL112_DEV WL112_DEV_FORCE_OUTCOME
export PYTHONUNBUFFERED=1
echo "chain start $(date -Iseconds) commit $(git rev-parse HEAD)" >> $R/logs/chain_progress.log

step() {
  local name=$1; shift
  echo "RUNNING $name $(date -Iseconds)" > $ST
  "$@" > $R/logs/$name.log 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "FAILED $name rc=$rc $(date -Iseconds)" > $ST; exit 1; fi
  echo "$name done $(date -Iseconds)" >> $R/logs/chain_progress.log
}

par() {   # par NAME "cmd1" "cmd2" [...]: run in parallel, fail if any fails
  local name=$1; shift
  echo "RUNNING $name $(date -Iseconds)" > $ST
  local pids=(); local i=0
  for c in "$@"; do bash -c "$c" > $R/logs/${name}_$i.log 2>&1 & pids+=($!); i=$((i+1)); done
  local bad=0
  for p in "${pids[@]}"; do wait $p || bad=1; done
  if [ $bad -ne 0 ]; then echo "FAILED $name $(date -Iseconds)" > $ST; exit 1; fi
  echo "$name done $(date -Iseconds)" >> $R/logs/chain_progress.log
}

step tests $PY -B tests112.py $R
step stageA $PY -B stageA112.py $R
par stageB "$PY -B eval112.py stageB $R V0 H1 H2" "$PY -B eval112.py stageB $R H3 H4 H5"
step analyzeAB $PY -B analyzeAB112.py $R
OUT=$($PY -c "import json;print(json.load(open('$R/stageB/decision.json'))['outcome'])")
echo "stageB outcome $OUT" >> $R/logs/chain_progress.log
if [ "$OUT" = "PROJECTION" ] || [ "$OUT" = "SPATIAL_G" ] || [ "$OUT" = "SPATIAL_U" ]; then
  par data "$PY -B data112.py $R train 0 100" "$PY -B data112.py $R train 100 192 && $PY -B data112.py $R val 0 16"
  step precheck $PY -B precheck112.py $R
  PRE=$($PY -c "import json;print(json.load(open('$R/stageC/precheck.json'))['pass'])")
  if [ "$PRE" != "True" ]; then echo "STOPPED precheck failed $(date -Iseconds)" > $ST; exit 0; fi
  if [ "$OUT" = "PROJECTION" ]; then
    par train "$PY -B train112.py $R O_U" "$PY -B train112.py $R O_G"
  else
    par train "$PY -B train112.py $R O_U" "$PY -B train112.py $R O_G" "$PY -B train112.py $R S2"
  fi
  par stageD "$PY -B eval112.py stageD $R F0_V0 F0_H1 F0_H2 F0_H3 F0_H4" "$PY -B eval112.py stageD $R F0_H5 F1_W0 F1_W1 F1_W2"
  par stageE "$PY -B eval112.py stageE $R SEQ_A_lift" "$PY -B eval112.py stageE $R SEQ_B_slide SEQ_C_independent"
  step rebuild $PY -B eval112.py rebuild $R
  step analyzeDE $PY -B analyzeDE112.py $R
  step cost $PY -B cost112.py $R
else
  step analyzeDE $PY -B analyzeDE112.py $R
fi
if [ -f fig112.py ]; then $PY -B fig112.py $R ../../evaluation/112 > $R/logs/figures.log 2>&1 || echo "figures failed (non-fatal)" >> $R/logs/chain_progress.log; fi
echo "DONE $(date -Iseconds)" > $ST
echo "chain done $(date -Iseconds)" >> $R/logs/chain_progress.log
