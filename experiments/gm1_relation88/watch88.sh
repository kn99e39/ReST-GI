#!/bin/bash
ST=$1/chain.status; OUT=$1/gpu_util_trace.txt; : > "$OUT"
while ! grep -qE "CHAIN_DONE|CHAIN_FAILED" "$ST" 2>/dev/null; do echo "$(date -u +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)" >> "$OUT"; sleep 2; done
