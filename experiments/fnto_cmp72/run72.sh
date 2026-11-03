#!/bin/bash
# WL72 wrapper on LabServer63 (one detached process): wait for the TransGI-Reproduction job to finish completely and the GPU to be idle
# for 10 consecutive minutes, then dev smoke chain, then (only if the smoke completes) the evidence chain.
set -u
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python; D=$HOME/rest_cmp72; cd $D/repo/experiments/fnto_cmp72
st(){ echo "$1 $(date -u '+%F %T')" > $D/run72.status; }
transgi(){ for p in /proc/[0-9]*; do c=$(readlink $p/cwd 2>/dev/null); case "$c" in *TransGI-Reproduction*) return 0;; esac; done; return 1; }
ok=0
while [ $ok -lt 10 ]; do
  if transgi || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then ok=0; st "waiting (TransGI job or GPU busy)"; else ok=$((ok+1)); st "idle check $ok/10"; fi
  sleep 60
done
st "running dev smoke"
R72_ROOT=$HOME/rest_cmp72_dev R72_DEV_STEPS=40 R72_SMOKE_N=500000 R72_DEV_ORACLE_ITERS=5 $PY chain72.py > $HOME/rest_cmp72_dev/chain.out 2>&1 || { st "FAILED dev smoke (see ~/rest_cmp72_dev/chain.status)"; exit 1; }
grep -q '^done' $HOME/rest_cmp72_dev/chain.status || { st "FAILED dev smoke (not done)"; exit 1; }
st "running evidence chain"
R72_ROOT=$D $PY chain72.py > $D/chain.out 2>&1 || { st "FAILED evidence chain (see chain.status)"; exit 1; }
st "done"
