#!/usr/bin/env bash
# Worklog 41: Stage A probe renders on the PC (WSL CPU Cycles), one split after another; stops on the first failure.
# Launch through experiments/dynamic_transport_failure/scripts/wsl_run_detached.sh (AGENTS.md long-job rule).
set -u
U=/mnt/c/Projects/ReST-GI/external/relightable-neural-assets; X=/mnt/c/Projects/ReST-GI/experiments/fnto_lego
O=/mnt/c/Projects/ReST-GI/results/fnto_lego/stagea
run() { mkdir -p $O/work_$1 && cd $O/work_$1 && $U/.venv/bin/python $X/fnto_stagea_render.py -- $O $1 $2 $3 --res 512 --spp 512 ${4:-} > $O/$1.log 2>&1; rc=$?; echo "$1 rc=$rc $(date -Is)" >> $O/chain.log; [ $rc = 0 ] || exit $rc; }
echo "chain start $(date -Is)" >> $O/chain.log
run holdout 120 2002
run superpose 12 3003 --superpose
run train 1400 1001
echo "chain done $(date -Is)" >> $O/chain.log
