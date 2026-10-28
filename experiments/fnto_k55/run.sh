#!/usr/bin/env bash
# Run a WL55 python script on LabServer69 in the 8dna26 venv (GPU intersector). Usage: run.sh <script> <args...>
PM=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
[ -f /usr/lib64/libLLVM-11.so ] && export DRJIT_LIBLLVM_PATH=/usr/lib64/libLLVM-11.so
mkdir -p $HOME/rest_wl55/tmp; export TMPDIR=$HOME/rest_wl55/tmp P53_TMPDIR=$HOME/rest_wl55/tmp   # root filesystem is full: scratch on /home
export PYTHONUNBUFFERED=1 P53_MI_VARIANT=${P53_MI_VARIANT:-cuda_ad_rgb} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $HOME/rest_wl55/repo/experiments/fnto_k55 && exec $PM "$@"
