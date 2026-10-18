#!/usr/bin/env bash
# Worklog 37 (LabServer63): the UNCHANGED upstream generator for one b4 config, own work dir, seeded by RR_SEED.
# Same upstream commit (66b5b09), same custom Cycles build (75b072e3ad79), torch 2.8.0, numpy 2.2.6 as the 5080 PC.
set -e
U=$HOME/NeuralShaderDemo/external/relightable-neural-assets; R=$HOME/nsd_b4
name=$1; seed=$2
mkdir -p $R/work/$name; cd $R/work/$name; export PYTHONPATH=$U
echo "gen $name RR_SEED=$seed start $(date -Is)"
RR_SEED=$seed $U/.venv/bin/python $R/rr_run.py $U/scripts/generate_dataset.py --mode rna --config $R/config/$name.yml
echo "gen $name done $(date -Is)"
