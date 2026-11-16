#!/usr/bin/env bash
set -uo pipefail
BASE=$HOME/rest_8dna_public21a
UP=$HOME/NeuralShaderDemo/external/8dna26
export PATH=$UP/.venv/bin:$PATH
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
cd "$BASE"
exec 9>queue.lock
flock -n 9 || exit 75
assets=(seal teaset milk cat candle dragon bunny curlhair hair fabric)
echo "running $(date -Is)" > queue.status
failed=0
for target in 128 4096; do
  for asset in "${assets[@]}"; do
    for method in neural pt; do
      for rep in 0 1; do
        job=${asset}_${method}_r${rep}_s${target}
        echo "running $job $(date -Is)" > current.status
        timeout 8h python -u scripts/render.py --upstream "$UP" --out "$BASE/results" --asset "$asset" --method "$method" --rep "$rep" --spp "$target" > "logs/$job.log" 2>&1
        rc=$?
        echo "$rc $(date -Is)" > "logs/$job.status"
        if [ "$rc" != 0 ]; then failed=$((failed+1)); echo "failed $job rc=$rc" >> failures.log; fi
      done
    done
  done
done
for asset in seal milk dragon; do
  for rep in 0 1; do
    job=${asset}_pt_r${rep}_s32768
    echo "running $job $(date -Is)" > current.status
    timeout 8h python -u scripts/render.py --upstream "$UP" --out "$BASE/results" --asset "$asset" --method pt --rep "$rep" --spp 32768 > "logs/$job.log" 2>&1
    rc=$?; echo "$rc $(date -Is)" > "logs/$job.status"
    if [ "$rc" != 0 ]; then failed=$((failed+1)); echo "failed $job rc=$rc" >> failures.log; fi
  done
done
echo "complete failed=$failed $(date -Is)" > queue.status
