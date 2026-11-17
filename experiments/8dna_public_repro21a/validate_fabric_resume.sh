#!/usr/bin/env bash
set -uo pipefail
BASE=$HOME/rest_8dna_public21a
UP=$HOME/NeuralShaderDemo/external/8dna26
export PATH=$UP/.venv/bin:$PATH
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1
output=$BASE/results/fabric_execution_probe/resume
mkdir -p "$output"
for target in 16 32; do
  workers=1
  if [ "$target" = 32 ]; then workers=2; fi
  "$UP/.venv/bin/python" -u "$BASE/scripts/render.py" --upstream "$UP" --out "$output" \
    --asset fabric --method neural --spp "$target" --res 256 --tile-size 64 --workers "$workers" \
    --task-timeout 60 > "$output/run_s$target.log" 2>&1 || exit $?
done
"$UP/.venv/bin/python" "$BASE/scripts/validate_parallel_fabric.py" --upstream "$UP" \
  --base "$BASE/results/fabric_execution_probe" > "$BASE/results/fabric_execution_probe/validation.log" 2>&1
