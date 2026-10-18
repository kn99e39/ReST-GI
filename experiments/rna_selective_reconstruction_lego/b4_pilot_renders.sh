#!/usr/bin/env bash
# Worklog 37 GT-only pilot: oracle view sets (T0, T1, T1b x Cycles seeds 1, 2), 32 views at 512^2, 2048 spp, RR_SEED 4703.
set -u
G=/mnt/c/Projects/NeuralShaderDemo/experiments/rna_selective_reconstruction_lego/b4_gen.sh
for v in T0 T1 T1b; do for s in 1 2; do bash $G oracle_${v}_s$s 4703 2>&1 | grep -v "^Fra:\|bgl\|gvfs" | tail -3; done; done
echo "PILOT RENDERS done $(date -Is)"
