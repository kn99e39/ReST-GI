#!/usr/bin/env bash
set -uo pipefail
BASE=$HOME/rest_8dna_public21a
UP=$HOME/NeuralShaderDemo/external/8dna26
export PATH=$UP/.venv/bin:$PATH
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
for mode in 0 1; do
  output=$BASE/results/fabric_execution_probe/vcall$mode
  mkdir -p "$output"
  timeout --kill-after=5s 75s "$UP/.venv/bin/python" -u "$BASE/scripts/diagnose_fabric.py" \
    --upstream "$UP" --out "$output" --vcall "$mode" > "$output/run.log" 2>&1
  rc=$?
  echo "$rc" > "$output/exit_code.txt"
done
