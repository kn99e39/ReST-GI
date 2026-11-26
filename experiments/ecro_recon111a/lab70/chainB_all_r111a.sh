#!/usr/bin/env bash
# WL111A Stage B launcher: wait (no GPU use) for Stage A, then a Stage B smoke in a separate OUT (200 training steps, 2 views,
# 4 frames, 20 oracle steps); the evidence-free Stage B run starts only if the smoke completes.
set -eu -o pipefail
R=$HOME/rest_sss101a
L=$R/experiments/ecro_recon111a/lab70
S=$R/results/ecro_recon111a/chainB_all.state
st() { printf '%s %s\n' "$(date -Iseconds)" "$*" >> "$S"; }
until grep -q STAGEA_COMPLETE "$R/results/ecro_recon111a/stageA.log" 2>/dev/null; do
  if ! docker ps --format '{{.Names}}' | grep -q '^recon111a_stageA$'; then st "Stage A ended without STAGEA_COMPLETE"; exit 5; fi
  sleep 60
done
st "Stage A complete; smoke"
RECON111A_SMOKE=1 RECON111A_ITERS=200 RECON111A_OUT=$R/results/ecro_recon111a_smoke bash "$L/chainB_r111a.sh" || { st "SMOKE FAILED"; exit 6; }
st "smoke complete; Stage B"
bash "$L/chainB_r111a.sh" && st "Stage B complete" || { st "Stage B FAILED"; exit 7; }
