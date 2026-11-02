#!/bin/bash
# One server chain on LabServer63: tests -> train R and N concurrently -> eval (incl. Stage A re-scoring) -> label. Status in chain.status.
set -u
D=$HOME/rest_drift_smoke; cd $D
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595 P53_MI_VARIANT=cuda_ad_rgb WL57_WALKER=fast PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True R71_ROOT=$HOME/rest_cover71
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
st(){ echo "$1 $(date -u '+%F %T')" > $D/chain.status; }
mkdir -p logs
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)" >> logs/gpu_sampler.log; sleep 15; done ) & SAMP=$!
trap "kill $SAMP 2>/dev/null" EXIT
st "running tests"; $PY drift.py tests > logs/tests.log 2>&1 || { st "FAILED tests"; exit 1; }
st "running train R+N"; $PY drift.py train R > logs/train_R.log 2>&1 & PR=$!; $PY drift.py train N > logs/train_N.log 2>&1 & PN=$!
wait $PR; rR=$?; wait $PN; rN=$?
[ $rR -eq 0 ] && [ $rN -eq 0 ] || { st "FAILED train R=$rR N=$rN"; exit 1; }
st "running eval"; $PY drift.py eval > logs/eval.log 2>&1 || { st "FAILED eval"; exit 1; }
st "running label"; $PY drift.py label > logs/label.log 2>&1 || { st "FAILED label"; exit 1; }
st "done"
