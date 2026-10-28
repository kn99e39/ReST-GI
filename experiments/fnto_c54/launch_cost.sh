#!/usr/bin/env bash
# idle check + cost benchmark on LabServer69 (uncontended: nothing else is started while this runs)
R=$HOME/rest_wl54; S=$R/launcher/cost_chain.status; export TMPDIR=$R/tmp; mkdir -p $TMPDIR
cd $R/repo/experiments/fnto_c54
echo "running idle check $(date -Is)" > $S
( nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader; cat /proc/loadavg; ps -eo pcpu,comm --sort=-pcpu | head -5 ) > $R/out/idle_check_before_cost.txt 2>&1
echo "running cost $(date -Is)" > $S
$R/repo/experiments/fnto_c54/run.sh c54_cost.py $R/out/cost.json 5 > $R/launcher/final_cost.log 2>&1 && echo "done $(date -Is)" > $S || echo "failed $(date -Is)" > $S
