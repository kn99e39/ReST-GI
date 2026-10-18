#!/usr/bin/env bash
P=/mnt/c/Projects/NeuralShaderDemo; U=$P/external/relightable-neural-assets; R=$P/results/rna_selective_reconstruction_lego
cd $P/results/rna_reproduction/work
$U/.venv/bin/python $P/experiments/rna_selective_reconstruction_lego/b4_scene.py data/scenes/Lego_rough_aovs_normalized.blend $R/scene 2>&1 | grep -v "^Read blend\|bgl\|^Info" | tail -5
