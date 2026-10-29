#!/usr/bin/env bash
set -euo pipefail
export P59="${P59:-$HOME/rest_persist_v1}"
export WL57="$P59" WL57_TRAIN_TARGETS="$P59/data/training_targets"
export WL57_CONFIRM_PROTOCOL="$P59/repo/experiments/fnto_rule57/protocol/rule57_recovery_a2.json"
export P53_MI_VARIANT=cuda_ad_rgb DRJIT_LIBLLVM_PATH=/usr/lib64/libLLVM-11.so
export TMPDIR="$P59/tmp" P53_TMPDIR="$P59/tmp" PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WL57_WALKER=fast
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
cd "$P59/repo/experiments/fnto_persist59"
exec "$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python" "$@"
