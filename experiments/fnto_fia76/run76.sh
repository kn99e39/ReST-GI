#!/bin/bash
# WL76 wrapper on LabServer63 (one detached process): dev smoke chain (~/rest_fia76_dev: 30-step fits, 5-step probe, 0.5M-photon packets
# on the dev slots, Stage 4 forced; uses the evidence Stage 1 record; no metric used), then the evidence chain (~/rest_fia76).
set -u
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python; D=$HOME/rest_fia76; DV=$HOME/rest_fia76_dev; cd $D/repo/experiments/fnto_fia76
st(){ echo "$1 $(date -u '+%F %T')" > $D/run76.status; }
mkdir -p $DV/repo && { [ -e $DV/repo/experiments ] || ln -s $D/repo/experiments $DV/repo/experiments; }
if ! grep -q '^done' $DV/chain.status 2>/dev/null; then
  st "running dev smoke"
  R76_ROOT=$DV R76_STAGE1=$D/stage1/physics.json R76_DEV_STEPS=30 R76_PROBE_STEPS=5 R74_SMOKE_N=500000 R76_DEV_FORCE_GATE=1 $PY chain76.py > $DV/chain.out 2>&1 || { st "FAILED dev smoke (see $DV/chain.status)"; exit 1; }
  grep -q '^done' $DV/chain.status || { st "FAILED dev smoke (not done)"; exit 1; }
fi
st "running evidence chain"
R76_ROOT=$D $PY chain76.py > $D/chain.out 2>&1 || { st "FAILED evidence chain (see chain.status)"; exit 1; }
st "done"
