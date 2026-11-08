#!/bin/bash
# WL80 chain B wrapper on LabServer63 (one detached process): waits for chain A (Stages 0-2) -> dev smoke of chain B on ~/rest_orbit80_dev
# (copied Stage 0-2 records and basis; 40-step fits, 10-step probe, 2 reference slots at 1M photons, forced gates; no metric used) ->
# evidence chain B (~/rest_orbit80). GPU work waits for < 1 GB used (two checks 60 s apart); other projects' jobs are never touched.
set -u
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python;D=$HOME/rest_orbit80;DD=$HOME/rest_orbit80_dev;cd $D/repo/experiments/fnto_orbit80
st(){ echo "$1 $(date -u '+%F %T')" > $D/run80.status; }
gpu(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
waitgpu(){ while :; do u=$(gpu); if [ "$u" -lt 1000 ]; then sleep 60; u=$(gpu); [ "$u" -lt 1000 ] && break; fi; st "waiting for a free GPU ($1; ${u} MiB used)"; sleep 120; done; }
st "waiting for chain A"
while ! grep -qE '^(done|FAILED)' $D/runA.status 2>/dev/null; do sleep 60; done
grep -q '^FAILED' $D/runA.status && { st "FAILED: chain A failed"; exit 1; }
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)" >> $D/logs/gpu_sampler_B.log; sleep 15; done ) & SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT
if grep -q '"passed": true' $D/stage2/gate.json; then
  mkdir -p $DD/stage2 $DD/basis $DD/stage0
  cp $D/stage2/gate.json $DD/stage2/; cp $D/stage0/stage0.json $DD/stage0/; cp $D/basis/G1.pt $D/basis/select.json $DD/basis/; cp $D/basis/B80_r*.pt $DD/basis/
  [ -e $DD/repo ] && [ ! -L $DD/repo ] && mv $DD/repo $DD/repo_predeploy_$(date +%s); ln -sfn $D/repo $DD/repo
  if ! grep -q '^done' $DD/chain.status 2>/dev/null; then
    waitgpu "dev smoke"; st "running dev smoke"
    R80_ROOT=$DD R75_DEV_STEPS=40 R80_PROBE_STEPS=10 R80_DEV_FORCE=1 $PY chain80.py > $DD/chain.out 2>&1 || { st "FAILED dev smoke (see ~/rest_orbit80_dev/chain.status)"; exit 1; }
    grep -q '^done' $DD/chain.status || { st "FAILED dev smoke (not done)"; exit 1; }
  fi
fi
waitgpu "evidence chain"; st "running evidence chain"
R80_ROOT=$D $PY chain80.py > $D/chain.out 2>&1 || { st "FAILED evidence chain (see chain.status)"; exit 1; }
st "done"
