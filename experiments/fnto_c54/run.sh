#!/usr/bin/env bash
# Run a WL54 python script on LabServer69 in the 8dna26 venv (GPU intersector). Usage: run.sh <script> <args...>  (cwd = repo experiments/fnto_c54)
PM=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
[ -f /usr/lib64/libLLVM-11.so ] && export DRJIT_LIBLLVM_PATH=/usr/lib64/libLLVM-11.so
mkdir -p $HOME/rest_wl54/tmp; export TMPDIR=$HOME/rest_wl54/tmp   # LabServer69 root filesystem is full: Mitsuba PLY scratch goes to /home
export PYTHONUNBUFFERED=1 P53_MI_VARIANT=${P53_MI_VARIANT:-cuda_ad_rgb} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $HOME/rest_wl54/repo/experiments/fnto_c54 && exec $PM "$@"
