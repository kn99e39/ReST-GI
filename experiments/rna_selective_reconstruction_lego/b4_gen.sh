#!/usr/bin/env bash
# Worklog 37: run the UNCHANGED upstream generator for one b4 config, in its own work dir (temp EXRs), seeded by RR_SEED.
#   b4_gen.sh <config name (results/.../config/<name>.yml)> <RR_SEED>
set -e
P=/mnt/c/Projects/NeuralShaderDemo; U=$P/external/relightable-neural-assets; R=$P/results/rna_selective_reconstruction_lego
name=$1; seed=$2
mkdir -p $R/work/$name; cd $R/work/$name; export PYTHONPATH=$U
echo "gen $name RR_SEED=$seed start $(date -Is)"
RR_SEED=$seed $U/.venv/bin/python $P/experiments/rna_reproduction/rr_run.py $U/scripts/generate_dataset.py --mode rna --config $R/config/$name.yml
echo "gen $name done $(date -Is)"
