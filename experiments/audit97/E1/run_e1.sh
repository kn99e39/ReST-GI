#!/bin/bash
# WL97 track E1 runner (LabServer63). Usage: bash run_e1.sh <stage>   (stage: conv | sender | train)
# Reads frozen code and artifacts read-only; writes only under ~/rest_x97/E1.
set -u
. ~/rest_x95a/env.sh
export PYTHONDONTWRITEBYTECODE=1
export E1_CODE=$HOME/rest_x95a/code/experiments      # frozen checkout c152120 (WL95A freeze), read-only
export E1_ROOT=$HOME/rest_x97/E1
export E1_ORIG=$HOME/rest_x95a/results               # WL95A outputs, read-only
export E1_R92A=$HOME/rest_x92a                       # OWN checkpoints, read-only
export E1_R94=$HOME/rest_x94                         # CK2 field references, read-only
export E1_R86=$HOME/rest_ecro86                      # WL86 training / validation data, read-only
cd "$E1_ROOT" || exit 1
ST=$1
rm -f "done_$ST"
$PY e1_audit.py "$ST" > "log_$ST.txt" 2>&1
echo $? > "done_$ST"
