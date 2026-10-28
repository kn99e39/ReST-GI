#!/usr/bin/env bash
set -euo pipefail
R="$HOME/rest_wl57_recovery"
export WL57="$R" WL57_CONFIRM_PROTOCOL="$R/repo/experiments/fnto_rule57/protocol/rule57_recovery_a2.json"
export WL57_TRAIN_TARGETS="$R/data/training_targets"
export WL57_RUNNER="$R/repo/experiments/fnto_rule57/run_recovery.sh"
export TMPDIR="$R/tmp" P53_TMPDIR="$R/tmp" PYTHONUNBUFFERED=1 P53_MI_VARIANT=cuda_ad_rgb
export DRJIT_LIBLLVM_PATH=/usr/lib64/libLLVM-11.so PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$R/repo/experiments/fnto_rule57"
exec "$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python" "$@"
