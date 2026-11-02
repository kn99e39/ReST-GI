#!/bin/bash
# WL71B server chain on LabServer63: tests -> stage A -> stage C -> label. Status in chain.status; stops on the first failure.
set -u
D=$HOME/rest_attrib71b; cd $D
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True R71_ROOT=$HOME/rest_cover71
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
st(){ echo "$1 $(date -u '+%F %T')" > $D/chain.status; }
mkdir -p logs
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)" >> logs/gpu_sampler.log; sleep 15; done ) & SAMP=$!
trap "kill $SAMP 2>/dev/null" EXIT
for s in tests stageA stageC label; do
  st "running $s"; $PY attrib.py $s > logs/$s.log 2>&1 || { st "FAILED $s"; exit 1; }
done
st "done"
