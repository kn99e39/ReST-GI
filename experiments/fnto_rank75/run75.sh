#!/bin/bash
# WL75 wrapper on LabServer63 (one detached process): dev smoke chain (root ~/rest_rank75_dev: 40-step fits, 10-step probe, 5 oracle
# iterations, 0.5M-photon packets on the dev slot subset, Stage 3 forced; no metric used), then the evidence chain (~/rest_rank75).
set -u
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python; D=$HOME/rest_rank75; cd $D/repo/experiments/fnto_rank75
st(){ echo "$1 $(date -u '+%F %T')" > $D/run75.status; }
# User decision 2026-11-06: start only when LabServer63's GPU is free (other projects' jobs are never touched): used memory < 1 GB
# on two checks 60 s apart.
gpu(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
waitgpu(){ while :; do u=$(gpu); if [ "$u" -lt 1000 ]; then sleep 60; u=$(gpu); [ "$u" -lt 1000 ] && break; fi; st "waiting for a free GPU ($1; ${u} MiB used by other jobs)"; sleep 120; done; }
mkdir -p $HOME/rest_rank75_dev/repo
[ -e $HOME/rest_rank75_dev/repo/experiments ] || ln -s $D/repo/experiments $HOME/rest_rank75_dev/repo/experiments
if ! grep -q '^done' $HOME/rest_rank75_dev/chain.status 2>/dev/null; then
  waitgpu "dev smoke"; st "running dev smoke"
  R75_ROOT=$HOME/rest_rank75_dev R75_DEV_STEPS=40 R75_DEV_ITERS=5 R75_PROBE_STEPS=10 R74_SMOKE_N=500000 R75_DEV_FORCE_GATE=1 $PY chain75.py > $HOME/rest_rank75_dev/chain.out 2>&1 || { st "FAILED dev smoke (see ~/rest_rank75_dev/chain.status)"; exit 1; }
  grep -q '^done' $HOME/rest_rank75_dev/chain.status || { st "FAILED dev smoke (not done)"; exit 1; }
fi
waitgpu "evidence chain"; st "running evidence chain"
R75_ROOT=$D $PY chain75.py > $D/chain.out 2>&1 || { st "FAILED evidence chain (see chain.status)"; exit 1; }
st "done"
