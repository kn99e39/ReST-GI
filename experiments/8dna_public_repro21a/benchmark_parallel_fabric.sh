#!/usr/bin/env bash
set -uo pipefail
BASE=$HOME/rest_8dna_public21a
UP=$HOME/NeuralShaderDemo/external/8dna26
export PATH=$UP/.venv/bin:$PATH
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1
for workers in 1 2; do
  output=$BASE/results/fabric_execution_probe/parallel$workers
  mkdir -p "$output"
  timeout --kill-after=10s 180s "$UP/.venv/bin/python" -u "$BASE/scripts/render.py" \
    --upstream "$UP" --out "$output" --asset fabric --method neural --rep 0 \
    --spp 32 --res 256 --chunk 4 --tile-size 64 --workers "$workers" \
    --task-timeout 60 > "$output/run.log" 2>&1
  echo "$?" > "$output/exit_code.txt"
done
