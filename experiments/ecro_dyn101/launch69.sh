#!/bin/bash
# WL101 Amendment 1: whole evidence chain on LabServer69 in ONE detached setsid process (resumable).
W=$HOME/rest_dyn101
export DYN101_PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
export DYN101_IDLE_S=${DYN101_IDLE_S:-10}
export DYN101_MI_DIR=$W/external/mi_3.8.0 DYN101_MI_VERSION=3.8.0
export LD_LIBRARY_PATH=$W/external/mi39_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
O=$W/results/ecro_dyn101/evidence
mkdir -p $O
cd $W/experiments/ecro_dyn101
setsid -f bash -c "$DYN101_PY -u chain101.py $O > $O/chain.out 2>&1"
echo launched
