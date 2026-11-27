#!/usr/bin/env bash
# WL111A: host monitor (no GPU use). When the RTX 3080 Ti has had no compute process for 120 s in a row, write OUT/use_ti
# (owner request 2026-11-27: move to the 3080 Ti when it frees up, so the RTX 3090 is released). Polls every 10 s.
O=${RECON111A_OUT:-$HOME/rest_sss101a/results/ecro_recon111a}
TI=GPU-746555dc-08d4-ae98-4ba9-cc74672b34e1
free_since=
while [[ ! -f $O/use_ti ]]; do
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$TI"; then free_since=
  else
    now=$(date +%s); free_since=${free_since:-$now}
    if (( now - free_since >= 120 )); then
      printf '%s RTX 3080 Ti free for 120 s\n' "$(date -Iseconds)" > "$O/use_ti"; break
    fi
  fi
  sleep 10
done
