#!/usr/bin/env bash
# Worklog 37 held-out references on LabServer63: T0, T1 x Cycles seeds 1, 2; 16 views, 1024^2, 2048 spp, RR_SEED 4702.
set -u
R=$HOME/nsd_b4
for v in T0 T1; do for s in 1 2; do
  bash $R/b4_gen_server.sh heldout_${v}_s$s 4702 2>&1 | grep -v "^Fra:\|bgl\|gvfs" | tail -2
  touch $R/launcher/heldout_${v}_s$s.done
done; done
echo "exit=0 end=$(date -Is)" > $R/launcher/heldout.status
