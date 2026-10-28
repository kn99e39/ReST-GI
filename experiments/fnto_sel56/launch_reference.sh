#!/usr/bin/env bash
set -euo pipefail
R=$HOME/rest_wl56
STATUS=$R/launcher/reference.status
trap 'echo failed > "$STATUS"' ERR
echo running > "$STATUS"
export TMPDIR=$R/tmp PYTHONPATH=$HOME/NeuralShaderDemo/external/relightable-neural-assets/.venv/lib/python3.10/site-packages
cd "$R/repo/experiments/fnto_sel56"
"$HOME/rest_wl55/blender-3.5.1-linux-x64/blender" -b --factory-startup --python-use-system-env --python-exit-code 1 --python s56_cycles.py -- "$R/out/cycles"
"$R/repo/experiments/fnto_sel56/run.sh" s56.py reference_gate
echo done > "$STATUS"
