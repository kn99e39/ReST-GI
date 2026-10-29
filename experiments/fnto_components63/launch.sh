#!/usr/bin/env bash
set -euo pipefail
COMP_ROOT="${COMP_ROOT:?isolated root required}"
COMP_COMMIT="${COMP_COMMIT:?committed source required}"
ORIGINAL_ROOT="${ORIGINAL_ROOT:-$HOME/rest_persist_v1}"
COMP_PY="$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python"
export P53_MI_VARIANT=cuda_ad_rgb DRJIT_LIBLLVM_PATH=/usr/lib64/libLLVM-11.so
export TMPDIR="$COMP_ROOT/tmp" P53_TMPDIR="$COMP_ROOT/tmp" PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
mkdir -p "$COMP_ROOT/tmp"
cd "$COMP_ROOT/source"
trap 'code=$?; if [ "$code" -ne 0 ]; then printf "FAILED_EXIT_%s\n" "$code" > "$COMP_ROOT/status.txt"; fi' EXIT
printf 'RUNNING\n' > "$COMP_ROOT/status.txt"
"$COMP_PY" -m unittest -v test_components > "$COMP_ROOT/tests.log" 2>&1
"$COMP_PY" run63.py preflight --root "$COMP_ROOT" --original "$ORIGINAL_ROOT" --source-commit "$COMP_COMMIT" > "$COMP_ROOT/preflight.log" 2>&1
"$COMP_PY" run63.py generate --root "$COMP_ROOT" --original "$ORIGINAL_ROOT" --source-commit "$COMP_COMMIT" > "$COMP_ROOT/generate.log" 2>&1
CUDA_VISIBLE_DEVICES="" "$COMP_PY" analyze63.py --root "$COMP_ROOT" --original "$ORIGINAL_ROOT" --source-commit "$COMP_COMMIT" > "$COMP_ROOT/analyze.log" 2>&1
"$COMP_PY" -c 'import json,os; print(json.load(open(os.environ["COMP_ROOT"]+"/analysis.json"))["status"])' > "$COMP_ROOT/status.txt"
