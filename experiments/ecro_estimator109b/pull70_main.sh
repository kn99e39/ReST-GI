#!/usr/bin/env bash
# WL109B Amendment 4: mirror the frozen Python environment, code and analysis inputs from LabServer70 to Main WSL at the
# same /home/nd paths (pull over tailscale SSH; nothing on LabServer70 is modified).
set -eu -o pipefail
SRC=nd@100.115.109.101:/home/nd/./
log() { printf '%s %s\n' "$(date -Is)" "$*"; }
pull() { rsync -aR --partial -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no" "$@" /home/nd/; }
mkdir -p /home/nd
log START
pull "${SRC}.local/share/uv/python"; log "uv python done"
pull "${SRC}rest_dyn101/experiments" "${SRC}rest_dyn101/external/mi_3.8.0"; log "rest_dyn101 done"
pull --exclude 'rest_sss101a/results/' "${SRC}rest_sss101a"; log "code done"
R=rest_sss101a/results
pull "${SRC}$R/ecro_denoise108/inputs" "${SRC}$R/ecro_denoise107/capture" "${SRC}$R/ecro_denoise107/reference"; log "inputs done"
pull "${SRC}$R/ecro_estimator109/evidence/run/manifest.json" "${SRC}$R/ecro_estimator109/evidence/run/manifest_capture.json" \
     "${SRC}$R/ecro_estimator109/evidence/dev" "${SRC}$R/ecro_estimator109/evidence/metrics" "${SRC}$R/ecro_estimator109/evidence/synth"; log "wl109 small done"
pull --exclude 'tmp/' --exclude 'mpl/' --exclude 'smoke/' "${SRC}$R/ecro_estimator109b/evidence"; log "wl109b evidence done"
pull "${SRC}NeuralShaderDemo/external/8dna26/.venv"; log "venv done"
log COMPLETE
