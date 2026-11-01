#!/usr/bin/env bash
# WL69 arm CY: quality-matched Full Recompute = Cycles volume path tracing with the WL55 reference renderer/settings (official Blender
# 3.5.1 CUDA, k55_cycles unchanged), spp ladder x 2 realizations, seeds 690000 + 1000*state + 10*level + r (protocol xbench69_v1).
set -Eeuo pipefail
R=$HOME/rest_xbench69; B=$HOME/rest_wl55/blender-3.5.1-linux-x64/blender
export TMPDIR=$R/tmp PYTHONPATH=$HOME/NeuralShaderDemo/external/relightable-neural-assets/.venv/lib/python3.10/site-packages
cd $R/repo/experiments/fnto_k55
si=0
for scene in floor floor_c4; do
  li=0
  for spp in 1 4 16 64 256 1024; do
    s0=$((690000 + 1000 * si + 10 * li))
    $B -b --factory-startup --python-use-system-env --python k55_cycles.py -- $R/cy/spp$spp render $scene all $s0,$((s0 + 1)) spp=$spp device=CUDA threads=4
    li=$((li + 1))
  done
  si=$((si + 1))
done
