#!/usr/bin/env bash
# Worklog 41: first-event attribution references of the held-out views (WSL CPU); launch via wsl_run_detached.sh.
set -u
U=/mnt/c/Projects/ReST-GI/external/relightable-neural-assets; X=/mnt/c/Projects/ReST-GI/experiments/fnto_lego
O=/mnt/c/Projects/ReST-GI/results/fnto_lego/firstevent
mkdir -p $O/work && cd $O/work
for s in heldout_T0 heldout_T1; do
  $U/.venv/bin/python $X/fnto_firstevent_render.py -- $O $s --spp 1024 > $O/$s.log 2>&1; echo "$s rc=$? $(date -Is)" >> $O/chain.log
done
