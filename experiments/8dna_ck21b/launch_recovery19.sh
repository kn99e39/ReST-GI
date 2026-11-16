#!/usr/bin/env bash
set -uo pipefail
BASE=$HOME/rest_8dna_ck21b
UP=$HOME/NeuralShaderDemo/external/8dna26
export PATH=$UP/.venv/bin:$PATH
export LD_LIBRARY_PATH=$HOME/nvlibs/optix595${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-17.so
export TORCH_CUDA_ARCH_LIST=8.6
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
export CK21B_BASE=$BASE CK21B_SOURCE=$HOME/rest_x93d_ck2/code
cd "$BASE"
exec 9>chain.lock
flock -n 9 || exit 75
held=0
resume_public() {
  if [ "$held" = 1 ] && [ "$(ps -p 3579312 -o args=)" = "bash /home/nd/rest_8dna_public21a/scripts/launch.sh" ]; then kill -CONT 3579312; fi
}
trap resume_public EXIT
if [ "$(ps -p 3579312 -o args=)" = "bash /home/nd/rest_8dna_public21a/scripts/launch.sh" ]; then
  kill -STOP 3579312; held=1
fi
echo "recovery19_waiting_for_own_public_render $(date -Is)" > chain.status
# Let already-dispatched public-asset evaluation finish. Never stop unrelated
# workloads such as rest_x99 and never overlap our own rendering with the fit.
while ps -eo args= | grep -F 'scripts/render.py --upstream /home/nd/NeuralShaderDemo/external/8dna26 --out /home/nd/rest_8dna_public21a/results' | grep -v 'grep' >/dev/null; do sleep 10; done
echo "training_recovery19 $(date -Is)" > chain.status
echo "recovery19_start $(date -Is)" >> execution_history.log
"$UP/.venv/bin/python" -u scripts/recover_training19.py > logs/train_recovery19.log 2>&1
rc=$?
echo "$rc $(date -Is)" > logs/train_recovery19.status
echo "recovery19_end rc=$rc $(date -Is)" >> execution_history.log
if [ "$rc" != 0 ]; then echo "failed_training_recovery19 $(date -Is)" > chain.status; exit "$rc"; fi
echo "complete $(date -Is)" > chain.status
