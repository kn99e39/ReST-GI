#!/usr/bin/env bash
# WL113 detached, resumable end-to-end chain (local RTX 5080). Any failure writes FAILED to chain.status and stops.
set -u
cd /c/Projects/ReST-GI/experiments/ecro_sss113
PY=../../external/8dna26/.venv-cu128/Scripts/python.exe
R=${WL113_ROOT:-../../results/ecro_sss113/primary}
mkdir -p $R/logs $R/stageB $R/analysis
ST=$R/chain.status
export PYTHONUNBUFFERED=1
echo "chain start $(date -Iseconds) commit $(git rev-parse HEAD) dev=${WL113_DEV:-0}" >> $R/logs/chain_progress.log

step() {
  local name=$1; shift
  echo "RUNNING $name $(date -Iseconds)" > $ST
  "$@" > $R/logs/$name.log 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "FAILED $name rc=$rc $(date -Iseconds)" > $ST; exit 1; fi
  echo "$name done $(date -Iseconds)" >> $R/logs/chain_progress.log
}
par() {
  local name=$1; shift
  echo "RUNNING $name $(date -Iseconds)" > $ST
  local pids=(); local i=0
  for c in "$@"; do bash -c "$c" > $R/logs/${name}_$i.log 2>&1 & pids+=($!); i=$((i+1)); done
  local bad=0
  for p in "${pids[@]}"; do wait $p || bad=1; done
  if [ $bad -ne 0 ]; then echo "FAILED $name $(date -Iseconds)" > $ST; exit 1; fi
  echo "$name done $(date -Iseconds)" >> $R/logs/chain_progress.log
}

step tests $PY -B tests113.py $R
par stageA "$PY -B stageA113.py $R F0 F2" "$PY -B stageA113.py $R F1 F3"
par streamsA "$PY -B eval113.py stageA $R F0 F2" "$PY -B eval113.py stageA $R F1 F3"
step analyzeA $PY -B analyze113A.py $R
SEL=$($PY -c "import json;print(json.load(open('$R/stageB/decision.json'))['selected'])")
echo "stageB selected $SEL" >> $R/logs/chain_progress.log
if [ "$SEL" = "L4" ] || [ "$SEL" = "S4" ]; then
  if [ "${WL113_DEV:-0}" = "1" ]; then TA="0 3"; TB="3 6"; VR="0 2"; else TA="0 100"; TB="100 192"; VR="0 16"; fi
  par data "$PY -B data113.py $R train $TA" "$PY -B data113.py $R train $TB && $PY -B data113.py $R val $VR"
  step precheck $PY -B precheck113.py $R
  PRE=$($PY -c "import json;print(json.load(open('$R/stageC/precheck.json'))['pass'])")
  if [ "$PRE" != "True" ]; then
    echo "precheck failed (checks or dense budget): candidate path stopped" >> $R/logs/chain_progress.log
    step analyzeDE $PY -B analyze113DE.py $R
  else
    par train "$PY -B train113.py $R O_U" "$PY -B train113.py $R CAND"
    par stageD "$PY -B eval113.py stageD $R F0_V0 F0_H1 F0_H2 F0_H3 F0_H4 F0_H5 F1_W0 F1_W1 F1_W2 F2_W0 F2_W1 F2_W2 F3_W0 F3_W1 F3_W2" "$PY -B eval113.py stageD $R F4_W0 F4_W1 F4_W2 && $PY -B eval113.py oppoint $R"
    par stageE "$PY -B eval113.py stageE $R SEQ_A_lift" "$PY -B eval113.py stageE $R SEQ_B_slide SEQ_C_independent"
    step rebuild $PY -B eval113.py rebuild $R
    step analyzeDE $PY -B analyze113DE.py $R
    step cost $PY -B cost113.py $R
  fi
else
  step analyzeDE $PY -B analyze113DE.py $R
fi
if [ -f fig113.py ]; then $PY -B fig113.py $R ${WL113_EVAL:-../../evaluation/113} > $R/logs/figures.log 2>&1 || echo "figures failed (non-fatal)" >> $R/logs/chain_progress.log; fi
echo "DONE $(date -Iseconds)" > $ST
echo "chain done $(date -Iseconds)" >> $R/logs/chain_progress.log
