#!/bin/bash
# WL97 E5 timing run: bash run_e5.sh <tag> <s> [extra flags]   (one GPU process; writes ~/rest_x97/E5/out/<tag>.json)
set -u
B=$HOME/rest_x97/E5
. $HOME/rest_x95a/env.sh
tag=$1; s=$2; shift 2
mkdir -p $B/out
rm -f $B/out/$tag.done
{
  echo "PRE $(date -u +%FT%TZ) $(uptime)"
  echo "PRE apps: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"
  ( while true; do echo "$(date -u +%T) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; sleep 2; done ) > $B/out/$tag.gpu.txt & GP=$!
  $PY $B/code/e5_time.py $B $HOME/rest_x96a/code $HOME/rest_x92a $s $B/out/$tag.json "$@"
  echo "RC $?"
  kill $GP
  echo "POST $(date -u +%FT%TZ) $(uptime)"
} > $B/out/$tag.log 2>&1
touch $B/out/$tag.done
