#!/usr/bin/env bash
# Run from the existing LabServer63 frozen-renderer tree.
set -euo pipefail
export DYN101_MI_DIR="$PWD/wl101/external/mi_3.8.0" DYN101_MI_VERSION=3.8.0
export LD_LIBRARY_PATH="$HOME/nvlibs/optix595:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$PWD/results/ecro_denoise107/mpl"
export OPENBLAS_NUM_THREADS=2
source_dir=experiments/ecro_denoise107
out=results/ecro_denoise107
base=results/ecro_denoise105
anchor=evaluation/105/raw_manifest105.json
python="$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python"
commit="${2:?full committed source hash required}"
mkdir -p "$out"
case "${1:?capture, train, assess, or oracle required}" in
  capture)
    "$python" "$source_dir/replay107.py" "$base" "$anchor" "$out/replay.json"
    "$python" "$source_dir/capture107.py" "$base" wl101 "$out/capture" "$anchor" "$commit" \
      2>&1 | tee "$out/renderer.log"
    if grep -Eq 'Disabling this ray tracing operation|COMPILE ERROR' "$out/renderer.log"; then
      exit 99
    fi
    "$python" "$source_dir/reference107.py" "$base" "$out/reference" "$anchor" \
      --capture "$out/capture" --replay "$out/replay.json"
    ;;
  train)
    "$python" "$source_dir/predict107.py" train "$base/data" "$out/train" "$out/replay.json" "$out/reference/reference107.json"
    ;;
  assess)
    "$python" "$source_dir/predict107.py" eval "$base/data" "$out/train/O2.pt" \
      "$out/O2" "$out/reference/reference107.json"
    "$python" "$source_dir/real107.py" "$base" "$out/capture" "$out/reference" \
      "$out/replay.json" "$out/real" "$commit" --diagnostic "$out/O2"
    ;;
  oracle)
    "$python" "$source_dir/real107.py" "$base" "$out/capture" "$out/reference" \
      "$out/replay.json" "$out/real_oracle" "$commit"
    ;;
  *) exit 2 ;;
esac
