#!/bin/bash
# WL77A Part 0: stage the frozen WL76 population and generate / bundle its references (detached; GPU sampler; status file)
D=$HOME/rest_arch77a;PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python;cd $D/repo/experiments/fnto_arch77a;mkdir -p $D/logs
export P53_MI_VARIANT=cuda_ad_rgb WL57_WALKER=fast PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=$HOME/nvlibs/optix595 R77A_ROOT=$D
st(){ echo "$1 $(date -u '+%F %T')" > $D/refs.status; }
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)" >> $D/logs/gpu_sampler.log; sleep 15; done ) & SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT
st "running stage";$PY probes77a.py stage > $D/logs/stage.log 2>&1 || { st "FAILED stage"; exit 1; }
if [ ! -f $D/refs/complete.json ]; then
  st "running generate";$PY probes77a.py generate > $D/logs/generate.log 2>&1 || { st "FAILED generate"; exit 1; }
  st "running bundle";$PY probes77a.py bundle > $D/logs/bundle.log 2>&1 || { st "FAILED bundle"; exit 1; }
fi
st "done"
