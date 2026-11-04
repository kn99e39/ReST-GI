#!/bin/bash
# WL73 wrapper on LabServer63 (one detached process). User instruction 2026-11-04: take the server without waiting for other jobs.
# Dev smoke chain (root ~/rest_src73_dev: 0.5M-photon packets, 40-step fits/replays, 5-iteration oracle; pipeline check only, no metric
# used), then, only if it completes, the evidence chain (root ~/rest_src73).
set -u
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python; D=$HOME/rest_src73; cd $D/repo/experiments/fnto_src73
st(){ echo "$1 $(date -u '+%F %T')" > $D/run73.status; }
st "running dev smoke"
R73_ROOT=$HOME/rest_src73_dev R73_DEV_STEPS=40 R73_SMOKE_N=500000 R73_DEV_ORACLE_ITERS=5 $PY chain73.py > $HOME/rest_src73_dev/chain.out 2>&1 || { st "FAILED dev smoke (see ~/rest_src73_dev/chain.status)"; exit 1; }
grep -q '^done' $HOME/rest_src73_dev/chain.status || { st "FAILED dev smoke (not done)"; exit 1; }
st "running evidence chain"
R73_ROOT=$D $PY chain73.py > $D/chain.out 2>&1 || { st "FAILED evidence chain (see chain.status)"; exit 1; }
st "done"
