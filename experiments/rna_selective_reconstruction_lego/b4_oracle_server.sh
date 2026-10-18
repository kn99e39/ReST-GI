#!/usr/bin/env bash
# Worklog 37 GT-only pilot on LabServer63: oracle view sets (T0, T1, T1b x Cycles seeds 1, 2), 32 views 512^2, 2048 spp, RR_SEED 4703.
# The held-out render is paused (SIGSTOP) meanwhile so the pilot gate comes first, and resumed (SIGCONT) afterwards.
set -u
R=$HOME/nsd_b4
HP=$(pgrep -f "generate_dataset.py --mode rna --config $R/config/heldout" | head -1); [ -n "$HP" ] && kill -STOP $HP && echo "paused heldout pid $HP $(date -Is)"
for v in T0 T1 T1b; do for s in 1 2; do
  bash $R/b4_gen_server.sh oracle_${v}_s$s 4703 2>&1 | grep -v "^Fra:\|bgl\|gvfs" | tail -2
done; done
[ -n "$HP" ] && kill -CONT $HP && echo "resumed heldout pid $HP $(date -Is)"
echo "exit=0 end=$(date -Is)" > $R/launcher/oracle.status
