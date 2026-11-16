#!/usr/bin/env bash
# WL21B Amendment 1 migration launcher (takeover by session rest-gi-ef, 2026-11-16): same environment and public-batch hold as launch_stable21b.sh.
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
resume_previous() {
  if [ "$(ps -p 3579312 -o args=)" = "bash /home/nd/rest_8dna_public21a/scripts/launch.sh" ]; then kill -CONT 3579312; fi
}
cd "$BASE"
exec 9>chain.lock
flock -n 9 || exit 75
trap resume_previous EXIT
echo "waiting_for_idle_packed $(date -Is)" > chain.status
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]; do sleep 10; done
echo "training_packed $(date -Is)" > chain.status
echo "packed_training_start $(date -Is)" >> execution_history.log
"$PY" -u scripts/train_packed21b.py > logs/train_packed.log 2>&1
rc=$?; echo "$rc $(date -Is)" > logs/train_packed.status
echo "packed_training_end rc=$rc $(date -Is)" >> execution_history.log
if [ "$rc" != 0 ]; then echo "failed_training_packed $(date -Is)" > chain.status; exit "$rc"; fi
echo "complete $(date -Is)" > chain.status
