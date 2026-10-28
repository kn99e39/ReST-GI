#!/usr/bin/env bash
set -euo pipefail
R=$HOME/rest_wl56
export TMPDIR=$R/tmp P53_TMPDIR=$R/tmp PYTHONUNBUFFERED=1 P53_MI_VARIANT=cuda_ad_rgb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DRJIT_LIBLLVM_PATH=/usr/lib64/libLLVM-11.so
cd "$R/repo/experiments/fnto_sel56"
exec "$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python" "$@"
