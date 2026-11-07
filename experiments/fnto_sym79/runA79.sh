#!/bin/bash
# WL79 Stage 0 + Stage 1 chain on LabServer63 (one detached process; frozen code = the deployed commit): geometry contract -> reference
# controls (CPU) in parallel with the Stage 1 closure and oracle (GPU) -> Stage 1 gate (requires the Stage 0 contract). Other projects' jobs are
# never touched; GPU work waits for < 1 GB used on two checks 60 s apart.
set -u
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python;D=${R79_ROOT:-$HOME/rest_sym79};cd $D/repo/experiments/fnto_sym79
export R79_ROOT=$D P53_MI_VARIANT=cuda_ad_rgb WL57_WALKER=fast PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=$HOME/nvlibs/optix595
mkdir -p $D/logs $D/stage0 $D/stage1
st(){ echo "$1 $(date -u '+%F %T')" > $D/runA.status; }
gpu(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)" >> $D/logs/gpu_sampler.log; sleep 15; done ) & SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT
while :; do u=$(gpu); if [ "$u" -lt 1000 ]; then sleep 60; u=$(gpu); [ "$u" -lt 1000 ] && break; fi; st "waiting for a free GPU (${u} MiB used)"; sleep 120; done
if [ ! -f $D/stage0/geometry.json ]; then st "running stage0 geometry"; $PY stage0_79.py geometry $D/stage0/geometry.json > $D/logs/stage0_geometry.log 2>&1 || { st "FAILED stage0 geometry"; exit 1; }; fi
st "running stage0 refs || stage1 closure+oracle"
( [ -f $D/stage0/refs.json ] && grep -q '"summary"' $D/stage0/refs.json ) || $PY stage0_79.py refs $D/stage0/refs.json > $D/logs/stage0_refs.log 2>&1 & R=$!
( [ -f $D/stage1/closure.json ] || $PY oracle79.py closure > $D/logs/stage1_closure.log 2>&1 ) && ( [ -f $D/stage1/oracle.json ] || $PY oracle79.py oracle > $D/logs/stage1_oracle.log 2>&1 ); O=$?
wait $R; RR=$?
[ $RR -eq 0 ] || { st "FAILED stage0 refs"; exit 1; }
[ $O -eq 0 ] || { st "FAILED stage1 oracle"; exit 1; }
st "running stage1 decide"; $PY oracle79.py decide > $D/logs/stage1_decide.log 2>&1 || { st "FAILED stage1 decide"; exit 1; }
st "done stageA: $(tail -1 $D/logs/stage1_decide.log | cut -c1-120)"
