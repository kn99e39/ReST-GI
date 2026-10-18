#!/usr/bin/env bash
# Worklog 37 contract tests (RNA venv). Optional args: <oracle_T0_s1_val.h5> <oracle_T1_s1_val.h5> for the dataset-adapter test.
P=/mnt/c/Projects/NeuralShaderDemo; U=$P/external/relightable-neural-assets; R=$P/results/rna_selective_reconstruction_lego
export PYTHONPATH=$U
EXTRA=""; [ $# -ge 2 ] && EXTRA="--h5-T0 $1 --h5-T1 $2"
$U/.venv/bin/python $P/experiments/rna_selective_reconstruction_lego/tests/test_b4.py --out $R/tests $EXTRA
