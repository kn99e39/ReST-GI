#!/usr/bin/env bash
# WL55 Cycles references on the GPU (official Blender 3.5.1, CUDA; user decision 2026-10-28), replacing the CPU chain (stopped after 6
# gap8 renders, kept unused in out/cycles). Output: out/cycles_cuda (+ render_manifest.jsonl with build / device per render).
R=$HOME/rest_wl55; B=$R/blender-3.5.1-linux-x64/blender; S=$R/launcher/cycles_cuda.status; L=$R/launcher; O=$R/out/cycles_cuda; mkdir -p $O
export TMPDIR=$R/tmp PYTHONPATH=$HOME/NeuralShaderDemo/external/relightable-neural-assets/.venv/lib/python3.10/site-packages
cd $R/repo/experiments/fnto_k55; echo "running $(date -Is)" > $S
run() { echo "$(date -Is) start $*" >> $L/cycles_cuda_steps.log; $B -b --factory-startup --python-use-system-env --python k55_cycles.py -- $O "$@" device=CUDA threads=4 >> $L/cycles_cuda.log 2>&1 || echo "$(date -Is) FAILED $*" >> $L/cycles_cuda.fail; }
run render gap8 all 5501,5502
for sc in kf_gap8_U kf_gap8_X kf_gap8_S kf_gap4_U kf_gap4_X kf_gap4_S; do run render $sc all 5511,5512; done
for sc in gap4 floor bend90; do run render $sc all 5501,5502; done
echo "done $(date -Is)" > $S
