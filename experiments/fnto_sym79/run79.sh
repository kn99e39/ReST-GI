#!/bin/bash
# WL79 chain B wrapper on LabServer63 (one detached process): waits for chain A (Stage 0 + 1) -> dev smoke of chain B on ~/rest_sym79_dev
# (40-step fits, 10-step probe, 2 reference slots at 1M photons, forced gates; no metric used) -> evidence chain B (~/rest_sym79).
# GPU work starts only when no foreign job uses the GPU (< 1 GB on two checks 60 s apart); other projects' jobs are never touched.
set -u
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python;D=$HOME/rest_sym79;DD=$HOME/rest_sym79_dev;cd $D/repo/experiments/fnto_sym79
st(){ echo "$1 $(date -u '+%F %T')" > $D/run79.status; }
gpu(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
waitgpu(){ while :; do u=$(gpu); if [ "$u" -lt 1000 ]; then sleep 60; u=$(gpu); [ "$u" -lt 1000 ] && break; fi; st "waiting for a free GPU ($1; ${u} MiB used)"; sleep 120; done; }
while ! grep -qE '^(done|FAILED)' $D/runA.status 2>/dev/null; do st "waiting for chain A"; sleep 60; done
grep -q '^FAILED' $D/runA.status && { st "FAILED: chain A failed"; exit 1; }
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)" >> $D/logs/gpu_sampler_B.log; sleep 15; done ) & SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT
mkdir -p $DD/stage1 $DD/stage0
cp $D/stage1/gate.json $DD/stage1/gate.json 2>/dev/null; cp $D/stage0/*.json $DD/stage0/ 2>/dev/null; cp $D/stage1/*.json $DD/stage1/ 2>/dev/null
[ -e $DD/repo ] && [ ! -L $DD/repo ] && mv $DD/repo $DD/repo_predeploy_$(date +%s); ln -sfn $D/repo $DD/repo
if ! grep -q '^done' $DD/chain.status 2>/dev/null; then
  waitgpu "dev smoke"; st "running dev smoke"
  R79_ROOT=$DD R75_DEV_STEPS=40 R79_PROBE_STEPS=10 R79_DEV_FORCE=1 $PY chain79.py > $DD/chain.out 2>&1 || { st "FAILED dev smoke (see ~/rest_sym79_dev/chain.status)"; exit 1; }
  grep -q '^done' $DD/chain.status || { st "FAILED dev smoke (not done)"; exit 1; }
fi
waitgpu "evidence chain"; st "running evidence chain"
R79_ROOT=$D $PY chain79.py > $D/chain.out 2>&1 || { st "FAILED evidence chain (see chain.status)"; exit 1; }
st "done"
