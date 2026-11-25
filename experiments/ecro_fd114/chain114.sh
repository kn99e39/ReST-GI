#!/usr/bin/env bash
# WL114 detached, resumable end-to-end chain (local RTX 5080). Any failure writes FAILED to chain.status and stops.
set -u
cd /c/Projects/ReST-GI/experiments/ecro_fd114
PY=../../external/8dna26/.venv-cu128/Scripts/python.exe
TPY=/c/Projects/TransGI_Reproduction/.venv/Scripts/python.exe
R=${WL114_ROOT:-../../results/ecro_fd114/primary}
mkdir -p $R/logs $R/analysis $R/stageA/lr3
ST=$R/chain.status
export PYTHONUNBUFFERED=1
echo "chain start $(date -Iseconds) commit $(git rev-parse HEAD) dev=${WL114_DEV:-0}" >> $R/logs/chain_progress.log

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
par() {
  local name=$1; shift
  if [ -f $R/logs/$name.ok ]; then return 0; fi
  echo "RUNNING $name $(date -Iseconds)" > $ST
  local pids=(); local i=0
  for c in "$@"; do bash -c "$c" > $R/logs/${name}_$i.log 2>&1 & pids+=($!); i=$((i+1)); done
  local bad=0
  for p in "${pids[@]}"; do wait $p || bad=1; done
  if [ $bad -ne 0 ]; then echo "FAILED $name $(date -Iseconds)" > $ST; exit 1; fi
  touch $R/logs/$name.ok
  echo "$name done $(date -Iseconds)" >> $R/logs/chain_progress.log
}
jget() { $PY -c "import json,sys;d=json.load(open('$1'));print(eval('d'+sys.argv[1]))" "$2"; }

step tests $PY -B tests114.py $R
# ---- Stage A (SSS views in two processes, then constructions alongside the LR3 reference ladder)
par stageA "$PY -B stageA114.py $R views 0 2 4 6 8 10" "$PY -B stageA114.py $R views 1 3 5 7 9 11"
par stageA_con "$PY -B stageA114.py $R con" "$TPY -B lr3ref114.py $R/stageA/lr3 masks auth ladder"
step analyzeA $PY -B analyze114A.py $R
QA=$(jget $R/analysis/stageA_qualification.json "['SSS']['verdict']")
echo "stageA SSS $QA K*=$(jget $R/analysis/stageA_qualification.json "['SSS']['K_star']") LR3 $(jget $R/analysis/stageA_qualification.json "['LR3']['verdict']")" >> $R/logs/chain_progress.log
if [ "$QA" != "QUALIFIED" ]; then
  echo "SSS full-depth target NOT QUALIFIED: SSS workload stopped (protocol stop condition)" >> $R/logs/chain_progress.log
  echo "DONE $(date -Iseconds)" > $ST; exit 0
fi
# ---- Stage B (oracle tables + streams; whole sequences stay in one process)
P1="F0 F1 F2 F3 F4 SEQ_B_slide SEQ_C_independent"
P2="N1 N2 N3 N4 SEQ_A_lift SEQ_D_independent"
par tables "$PY -B eval114.py tables $R $P1" "$PY -B eval114.py tables $R $P2"
par streamsB "$PY -B eval114.py streamsB $R $P1" "$PY -B eval114.py streamsB $R $P2"
step analyzeB $PY -B analyze114.py $R B
PRO=$(jget $R/stageB/decision.json "['proceed']")
echo "stageB proceed $PRO" >> $R/logs/chain_progress.log
if [ "$PRO" != "True" ]; then
  echo "oracle error prevents depth attribution: SSS learned workload stopped (protocol stop rule)" >> $R/logs/chain_progress.log
  echo "DONE $(date -Iseconds)" > $ST; exit 0
fi
# ---- Stage C (D114 alone on the GPU: b = 16 needs ~14.8 GB; budget 3.0 h; training budget 3.5 h)
if [ "${WL114_DEV:-0}" = "1" ]; then TR="0 6"; VA="0 2"; else TR="0 192"; VA="0 16"; fi
step data timeout 10800 bash -c "$PY -B data114.py $R train $TR && $PY -B data114.py $R val $VA"
step auth $PY -B data114.py $R auth
par train1 "timeout 7200 $PY -B train114.py $R A1r" "timeout 7200 $PY -B train114.py $R B1" "timeout 7200 $PY -B train114.py $R OB1"
par train2 "timeout 5400 $PY -B train114.py $R A2" "timeout 5400 $PY -B train114.py $R B2"
# ---- Stage D
par streamsD "$PY -B eval114.py streamsD $R $P1" "$PY -B eval114.py streamsD $R $P2"
step rebuild $PY -B eval114.py rebuild $R
step analyzeD $PY -B analyze114.py $R D
step cost $PY -B cost114.py $R
if [ -f fig114.py ]; then $PY -B fig114.py $R ${WL114_EVAL:-../../evaluation/114} > $R/logs/figures.log 2>&1 || echo "figures failed (non-fatal)" >> $R/logs/chain_progress.log; fi
echo "DONE $(date -Iseconds)" > $ST
echo "chain done $(date -Iseconds)" >> $R/logs/chain_progress.log
