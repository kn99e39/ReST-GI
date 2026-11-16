#!/usr/bin/env bash
set -euo pipefail
BASE=$HOME/rest_8dna_ck21b
UP=$HOME/NeuralShaderDemo/external/8dna26
export PATH=$UP/.venv/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.6
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export CK21B_BASE=$BASE CK21B_SOURCE=$HOME/rest_x93d_ck2/code
cd "$BASE"
"$UP/.venv/bin/python" -u scripts/render_epoch12_preview.py
