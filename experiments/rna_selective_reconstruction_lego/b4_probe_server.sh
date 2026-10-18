#!/usr/bin/env bash
# Worklog 37 feasibility probe (LabServer63 RTX 3080 Ti 12 GB): upstream RNA HQ training, WL35 native Lego data on host,
# 2 epochs; logs GPU memory every 5 s. Not evidence; only decides whether the RNA controls fit this GPU and how long they take.
set -u
R=$HOME/nsd_b4; U=$HOME/NeuralShaderDemo/external/relightable-neural-assets
until [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ]; do sleep 30; done
echo "gpu free $(date -Is)"
mkdir -p $R/probe/run; cd $R/probe/run; export PYTHONPATH=$U
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 5 > $R/probe/gpu_mem.csv &
NV=$!
echo "train start $(date -Is)"
$U/.venv/bin/python $R/rr_run.py $U/scripts/train.py --config $R/probe/probe_rna.yml --checkpoint_dir $R/probe/run/checkpoints > $R/probe/train.log 2>&1
echo "train rc=$? end $(date -Is)"
kill $NV
echo "exit=0 end=$(date -Is)" > $R/probe/probe.status
