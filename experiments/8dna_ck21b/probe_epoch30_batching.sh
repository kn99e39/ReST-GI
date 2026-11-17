#!/usr/bin/env bash
set -uo pipefail
BASE=$HOME/rest_8dna_ck21b
PUBLIC=$HOME/rest_8dna_public21a
UP=$HOME/NeuralShaderDemo/external/8dna26
export PATH=$UP/.venv/bin:$PATH
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1
for chunk in 16 32 64; do
  output=$BASE/results/epoch30_batching/chunk$chunk
  mkdir -p "$output"
  timeout --kill-after=5s 60s python -u "$PUBLIC/scripts/diagnose_fabric.py" \
    --upstream "$UP" --out "$output" --vcall 0 --asset ck2 \
    --checkpoint "$BASE/results/8dna_ck2_epoch30_4096spp/checkpoint.ckpt" \
    --scene-bridge "$BASE/scripts/bridge.py" --res 512 --spp "$chunk" \
    --crop 191 191 130 130 --repeats 2 > "$output/run.log" 2>&1
  echo "$?" > "$output/exit_code.txt"
done
