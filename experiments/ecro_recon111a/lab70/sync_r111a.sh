#!/usr/bin/env bash
# Sync WL111A code (experiments/ecro_recon111a) and the unchanged WL111 code it imports (experiments/ecro_recon111) to
# LabServer70. No other experiment directory on the host is touched.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs --exclude=__pycache__ -cf - experiments/ecro_recon111a experiments/ecro_recon111 \
  | ssh LabServer70 'mkdir -p ~/rest_sss101a && tar -xmf - -C ~/rest_sss101a && chmod +x ~/rest_sss101a/experiments/ecro_recon111*/lab70/*.sh' 2>&1 | grep -v manpath || true
echo synced
