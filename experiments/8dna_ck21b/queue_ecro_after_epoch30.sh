#!/usr/bin/env bash
set -euo pipefail
BASE=$HOME/rest_8dna_ck21b
UP=$HOME/NeuralShaderDemo/external/8dna26
export PATH=$UP/.venv/bin:$PATH
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$BASE"
exec 8>ecro_followup.lock
flock -n 8 || exit 75
finish() { echo "$? $(date -Is)" > logs/ecro_followup.exit; }
trap finish EXIT
echo "waiting_for_8dna_epoch30_4096 $(date -Is)" > ecro_followup.status
until test -f results/8dna_ck2_epoch30_4096spp/matched4096_complete.json; do
  sleep 30
done
echo "rendering_ecro_wl100_near_ck2_4096 $(date -Is)" > ecro_followup.status
python -u scripts/ecro_followup.py
echo "complete $(date -Is)" > ecro_followup.status
