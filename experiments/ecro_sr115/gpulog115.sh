#!/usr/bin/env bash
# GPU utilisation / memory log every 5 s (record of execution efficiency)
while true; do echo "$(date +%H:%M:%S),$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)" >> /c/Projects/ReST-GI/results/ecro_sr115/primary/logs/gpu_util.csv; sleep 5; done
