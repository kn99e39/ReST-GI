#!/usr/bin/env bash
set -eu -o pipefail
cd "$HOME/rest_sss101a"
out=results/ecro_denoise108
commit=$1
py="$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python"
src=experiments/ecro_denoise108
state_path="$out/${WL108_STATE:-training.state}"
export DYN101_MI_DIR="$PWD/wl101/external/mi_3.8.0" DYN101_MI_VERSION=3.8.0
export LD_LIBRARY_PATH="$HOME/nvlibs/optix595:${LD_LIBRARY_PATH:-}"
if [[ -n ${WL108_RUNTIME:-} ]]; then
  py="$WL108_RUNTIME/python/bin/python3.10"
  export DYN101_MI_DIR="$WL108_RUNTIME/python/lib/python3.10/site-packages"
  export LD_LIBRARY_PATH="$WL108_RUNTIME/drivers:$LD_LIBRARY_PATH"
fi
export OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR="$PWD/$out/mpl"
stage=preflight
state() { printf '%s %s\n' "$(date -Iseconds)" "$*" | tee "$state_path"; }
trap 'rc=$?; if (( rc )); then state "STOP stage=$stage exit=$rc"; fi' EXIT
"$py" "$src/tests108.py"
while [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) ]]; do
  state "WAITING_FOR_FREE_GPU"
  sleep 30
done
stage=qualification
state "RUNNING stage=$stage commit=$commit"
timeout 1800 "$py" "$src/qualify108.py" results/ecro_denoise105 results/ecro_denoise107 \
  evaluation/105/raw_manifest105.json wl101 "$out/descriptors" "$commit" > "$out/qualification.log" 2>&1
for row in '10550001 A0 B0 B1 A1' '10850002 B0 A1 A0 B1' '10850003 A1 B1 B0 A0'; do
  read -ra pieces <<<"$row"
  seed=${pieces[0]}
  for arm in "${pieces[@]:1}"; do
    stage="train_${arm}_${seed}"
    state "RUNNING stage=$stage commit=$commit"
    timeout 1800 "$py" "$src/train108.py" results/ecro_denoise105/data \
      "$out/descriptors" "$out/models" "$arm" "$seed" "$commit" > "$out/$stage.log" 2>&1
  done
done
state TRAINING_COMPLETE
