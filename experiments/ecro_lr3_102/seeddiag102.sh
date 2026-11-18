#!/bin/bash
# WL102 post-hoc seed diagnostic: D0 / D1 / D3 x seeds 10201-10203, 1,500 steps each, 5 concurrent (not evidence).
R=$HOME/rest_wl102/results/wl102; PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
cd $HOME/rest_wl102/experiments/ecro_lr3_102
run() { v=$1; s=$2; if [ $v = D3 ]; then export WL102_AREA_FLOOR=0.00390625; fi; $PY diag102.py $R/data $R/diag_seeds $v 1500 $s > $R/diag_seeds_${v}_$s.log 2>&1; echo "$(date -u +%T) done $v $s"; }
export -f run; export R PY
for v in D0 D1 D3; do for s in 10201 10202 10203; do echo $v $s; done; done | xargs -P 3 -L 1 bash -c "run \$0 \$1"
echo SEEDDIAG COMPLETE
