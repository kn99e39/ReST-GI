#!/bin/bash
# WL74 wrapper on LabServer63 (one detached process): dev smoke chain (root ~/rest_ato74_dev: 2 rounds / 5 FISTA-Adam iterations,
# 40-step spatial fits, 0.5M-photon packets, Stage 3/4 forced to exercise the path; no metric used), then the evidence chain.
set -u
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python; D=$HOME/rest_ato74; cd $D/repo/experiments/fnto_ato74
st(){ echo "$1 $(date -u '+%F %T')" > $D/run74.status; }
st "running dev smoke"
R74_ROOT=$HOME/rest_ato74_dev R74_DEV_ITERS=5 R74_DEV_STEPS=40 R74_SMOKE_N=500000 R74_DEV_FORCE_GATE=1 $PY chain74.py > $HOME/rest_ato74_dev/chain.out 2>&1 || { st "FAILED dev smoke (see ~/rest_ato74_dev/chain.status)"; exit 1; }
grep -q '^done' $HOME/rest_ato74_dev/chain.status || { st "FAILED dev smoke (not done)"; exit 1; }
st "running evidence chain"
R74_ROOT=$D $PY chain74.py > $D/chain.out 2>&1 || { st "FAILED evidence chain (see chain.status)"; exit 1; }
st "done"
