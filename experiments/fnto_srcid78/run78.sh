#!/bin/bash
# WL78 wrapper on LabServer63 (one detached process): dev smoke chain (root ~/rest_srcid78_dev: 40-step fits, 10-step probe, Stage 4 forced
# on the dev slots of the WL77A Part 0 references, after the protocol freeze; no metric used), then the evidence chain (~/rest_srcid78).
# Starts each chain only when the GPU is free (other projects' jobs are never touched): used memory < 1 GB on two checks 60 s apart.
set -u
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python; D=$HOME/rest_srcid78; cd $D/repo/experiments/fnto_srcid78
st(){ echo "$1 $(date -u '+%F %T')" > $D/run78.status; }
gpu(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
waitgpu(){ while :; do u=$(gpu); if [ "$u" -lt 1000 ]; then sleep 60; u=$(gpu); [ "$u" -lt 1000 ] && break; fi; st "waiting for a free GPU ($1; ${u} MiB used by other jobs)"; sleep 120; done; }
mkdir -p $HOME/rest_srcid78_dev
[ -e $HOME/rest_srcid78_dev/repo ] || ln -s $D/repo $HOME/rest_srcid78_dev/repo
if ! grep -q '^done' $HOME/rest_srcid78_dev/chain.status 2>/dev/null; then
  waitgpu "dev smoke"; st "running dev smoke"
  R78_ROOT=$HOME/rest_srcid78_dev R75_DEV_STEPS=40 R78_PROBE_STEPS=10 R78_DEV_FORCE_GATE=1 $PY chain78.py > $HOME/rest_srcid78_dev/chain.out 2>&1 || { st "FAILED dev smoke (see ~/rest_srcid78_dev/chain.status)"; exit 1; }
  grep -q '^done' $HOME/rest_srcid78_dev/chain.status || { st "FAILED dev smoke (not done)"; exit 1; }
fi
waitgpu "evidence chain"; st "running evidence chain"
R78_ROOT=$D $PY chain78.py > $D/chain.out 2>&1 || { st "FAILED evidence chain (see chain.status)"; exit 1; }
st "done"
