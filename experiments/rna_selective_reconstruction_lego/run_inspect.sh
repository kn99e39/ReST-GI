#!/usr/bin/env bash
P=/mnt/c/Projects/NeuralShaderDemo; U=$P/external/relightable-neural-assets; R=$P/results/rna_selective_reconstruction_lego
mkdir -p $R/scene
cd $P/results/rna_reproduction/work
$U/.venv/bin/python $P/experiments/rna_selective_reconstruction_lego/b4_inspect_blend.py data/scenes/Lego_rough_aovs_normalized.blend $R/scene/official_blend_inspect.json 2>&1 | tail -3
