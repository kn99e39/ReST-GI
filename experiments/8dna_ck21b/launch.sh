#!/usr/bin/env bash
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
echo "waiting_for_idle $(date -Is)" > chain.status
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ] || pgrep -f '^[^ ]*python -u scripts/render.py .*--out /home/nd/rest_8dna_public21a/results' > /dev/null; do sleep 10; done
if [ -f results/preflight.json ] && [ results/preflight.json -nt scripts/bridge.py ] && grep -q '"status": "pass"' results/preflight.json; then
  echo "preflight_preserved_pass $(date -Is)" >> execution_history.log
else
  echo "preflight $(date -Is)" > chain.status
  "$PY" -u scripts/preflight.py > logs/preflight.log 2>&1
  rc=$?; echo "$rc $(date -Is)" > logs/preflight.status
  if [ "$rc" != 0 ]; then echo "failed_preflight $(date -Is)" > chain.status; exit "$rc"; fi
fi
echo "training $(date -Is)" > chain.status
"$PY" -u scripts/train_native.py > logs/train_native.log 2>&1
rc=$?; echo "$rc $(date -Is)" > logs/train_native.status
if [ "$rc" != 0 ]; then echo "failed_training $(date -Is)" > chain.status; exit "$rc"; fi
echo "complete $(date -Is)" > chain.status
