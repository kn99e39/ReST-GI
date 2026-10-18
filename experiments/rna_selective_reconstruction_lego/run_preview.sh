#!/usr/bin/env bash
P=/mnt/c/Projects/NeuralShaderDemo; U=$P/external/relightable-neural-assets; R=$P/results/rna_selective_reconstruction_lego
mkdir -p $R/preview; cd $R/preview
for T in T0 T1; do
  $U/.venv/bin/python $P/experiments/rna_selective_reconstruction_lego/b4_preview.py $R/scene/lego_b4_${T}_s0.blend $R/preview/front_${T}.png 0.6,-2.6,0.7 -0.4,-0.6,0.7 2>&1 | grep -i "error\|saved" | tail -2
done
