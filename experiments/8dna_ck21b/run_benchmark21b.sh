#!/usr/bin/env bash
# WL21B Amendment 1 benchmark launcher (takeover by session rest-gi-ef, 2026-11-16). Same environment as launch_stable21b.sh.
set -uo pipefail
BASE=$HOME/rest_8dna_ck21b
UP=$HOME/NeuralShaderDemo/external/8dna26
PY=$UP/.venv/bin/python
export PATH=$UP/.venv/bin:$PATH
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CK21B_BASE=$BASE
export CK21B_SOURCE=$HOME/rest_x93d_ck2/code
export PYTHONUNBUFFERED=1
cd "$BASE"
echo "benchmark_start $(date -Is)" >> execution_history.log
"$PY" -u scripts/benchmark_pipeline.py > logs/benchmark_pipeline.log 2>&1
rc=$?; echo "$rc $(date -Is)" > logs/benchmark_pipeline.status
echo "benchmark_end rc=$rc $(date -Is)" >> execution_history.log
