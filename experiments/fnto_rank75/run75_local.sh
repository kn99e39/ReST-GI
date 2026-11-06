#!/usr/bin/env bash
# WL75 amendment 1 (user decision 2026-11-06: this batch only runs on the local RTX 5080 PC instead of LabServer63): Git Bash wrapper,
# started detached by PowerShell Start-Process. Dev smoke chain (root ~/rest_rank75_dev; settings as run75.sh), then the evidence chain
# (~/rest_rank75). No GPU wait (the local display holds ~2 GB permanently). Windows-native torch 2.8 cu128 venv.
set -u
PY=/c/Projects/ReST-GI/external/8dna26/.venv-cu128/Scripts/python.exe; D=/c/Users/dna10/rest_rank75; cd $D/repo/experiments/fnto_rank75
st(){ echo "$1 $(date -u '+%F %T')" > $D/run75.status; }
mkdir -p /c/Users/dna10/rest_rank75_dev/repo
[ -e /c/Users/dna10/rest_rank75_dev/repo/experiments ] || cp -r $D/repo/experiments /c/Users/dna10/rest_rank75_dev/repo/experiments
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
if ! grep -q '^done' /c/Users/dna10/rest_rank75_dev/chain.status 2>/dev/null; then
  st "running dev smoke"
  R75_ROOT='C:\Users\dna10\rest_rank75_dev' R75_DEV_STEPS=40 R75_DEV_ITERS=5 R75_PROBE_STEPS=10 R74_SMOKE_N=500000 R75_DEV_FORCE_GATE=1 $PY chain75.py > /c/Users/dna10/rest_rank75_dev/chain.out 2>&1 || { st "FAILED dev smoke (see rest_rank75_dev/chain.status)"; exit 1; }
  grep -q '^done' /c/Users/dna10/rest_rank75_dev/chain.status || { st "FAILED dev smoke (not done)"; exit 1; }
fi
[ "${R75_DEV_ONLY:-0}" = "1" ] && { st "dev smoke done (dev only)"; exit 0; }
st "running evidence chain"
R75_ROOT='C:\Users\dna10\rest_rank75' $PY chain75.py > $D/chain.out 2>&1 || { st "FAILED evidence chain (see chain.status)"; exit 1; }
st "done"
