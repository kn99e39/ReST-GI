#!/usr/bin/env bash
set -eu -o pipefail
cd "$HOME/rest_sss101a"
out=results/ecro_denoise108
commit=$1
frozen="$out/inputs"
py="$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python"
src=experiments/ecro_denoise108
base="$frozen/results/ecro_denoise105"
prior="$frozen/results/ecro_denoise107"
anchor=evaluation/107/heldout/manifest107_heldout.json
export DYN101_MI_DIR="$HOME/rest_dyn101/external/mi_3.8.0" DYN101_MI_VERSION=3.8.0
export LD_LIBRARY_PATH="$PWD/$out/runtime/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR="$PWD/$out/mpl"
stage=waiting_training
status() { printf '%s %s\n' "$(date -Iseconds)" "$*" | tee "$out/evaluation.state"; }
trap 'rc=$?; if (( rc )); then status "STOP stage=$stage exit=$rc"; fi' EXIT
status WAITING_FOR_QUALIFIED_FITS
while :; do
  if [[ ! -f "$out/training69.state" ]]; then sleep 30; continue; fi
  recovered=$(cat "$out/training69.state")
  case "$recovered" in
    *TRAINING_COMPLETE) break ;;
    *STOP*) echo "$recovered"; exit 10 ;;
  esac
  sleep 30
done
stage=original_evaluation
status "RUNNING stage=$stage commit=$commit"
timeout 2700 "$py" "$src/eval108.py" "$base" "$prior" "$out/models" "$out/descriptors" \
  "$out/original" "$anchor" "$commit" > "$out/original_eval.log" 2>&1
if "$py" -c 'import json,sys; x=json.load(open(sys.argv[1]))["decision"]; raise SystemExit(0 if x["original_input_pass"] or x["descriptor_incremental_pass"] else 3)' "$out/original/metrics.json"; then
  stage=confirmation_capture
  status "RUNNING stage=$stage commit=$commit"
  timeout 7200 "$py" "$src/confirm108.py" "$base" "$prior" "$frozen/wl101" "$out/original/metrics.json" \
    "$out/confirmation" > "$out/confirmation_capture.log" 2>&1
  stage=confirmation_evaluation
  status "RUNNING stage=$stage commit=$commit"
  timeout 2700 "$py" "$src/eval108.py" "$base" "$prior" "$out/models" "$out/confirmation/descriptors" \
    "$out/confirmation_eval" "$anchor" "$commit" --confirmation "$out/confirmation" > "$out/confirmation_eval.log" 2>&1
  status COMPLETE_WITH_CONFIRMATION
else
  rc=$?
  if (( rc != 3 )); then exit "$rc"; fi
  status COMPLETE_ORIGINAL_CONFIRMATION_NOT_TRIGGERED
fi
