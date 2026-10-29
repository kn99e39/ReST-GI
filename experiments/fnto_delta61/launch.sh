#!/usr/bin/env bash
set -euo pipefail
AUDIT_ROOT="${AUDIT_ROOT:?set isolated audit root}"
ORIGINAL_ROOT="${ORIGINAL_ROOT:-$HOME/rest_persist_v1}"
AUDIT_COMMIT="${AUDIT_COMMIT:?set committed source}"
AUDIT_PY="$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python"
export P53_MI_VARIANT=cuda_ad_rgb DRJIT_LIBLLVM_PATH=/usr/lib64/libLLVM-11.so
export TMPDIR="$AUDIT_ROOT/tmp" P53_TMPDIR="$AUDIT_ROOT/tmp" PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
mkdir -p "$AUDIT_ROOT/tmp"
cd "$AUDIT_ROOT/source"
trap 'code=$?; if [ "$code" -ne 0 ]; then printf "FAILED_EXIT_%s\n" "$code" > "$AUDIT_ROOT/status.txt"; fi' EXIT
printf 'RUNNING\n' > "$AUDIT_ROOT/status.txt"
"$AUDIT_PY" -m unittest -v test_metrics > "$AUDIT_ROOT/tests.log" 2>&1
"$AUDIT_PY" audit_run.py preflight --root "$AUDIT_ROOT" --original "$ORIGINAL_ROOT" --source-commit "$AUDIT_COMMIT" > "$AUDIT_ROOT/preflight.log" 2>&1
"$AUDIT_PY" audit_run.py generate --root "$AUDIT_ROOT" --original "$ORIGINAL_ROOT" --source-commit "$AUDIT_COMMIT" > "$AUDIT_ROOT/generate.log" 2>&1
CUDA_VISIBLE_DEVICES="" "$AUDIT_PY" analyze.py --root "$AUDIT_ROOT" --original "$ORIGINAL_ROOT" --source-commit "$AUDIT_COMMIT" > "$AUDIT_ROOT/analyze.log" 2>&1
"$AUDIT_PY" -c 'import json,os; p=os.environ["AUDIT_ROOT"]; print(json.load(open(p+"/analysis.json"))["completion_status"])' > "$AUDIT_ROOT/status.txt"
