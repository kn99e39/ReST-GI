#!/usr/bin/env bash
set -euo pipefail
BASE=$HOME/rest_8dna_public21a
UP=$HOME/NeuralShaderDemo/external/8dna26
export PATH=$UP/.venv/bin:$PATH
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1
exec "$UP/.venv/bin/python" -u "$BASE/scripts/render.py" --upstream "$UP" \
  --out "$BASE/results/fabric_parallel_v1" --asset fabric --method neural \
  --spp "${1:-128}" --res 800 --chunk 4 --tile-size 128 \
  --workers "${2:-auto}" --task-timeout 120
