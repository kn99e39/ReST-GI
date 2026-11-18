#!/bin/bash
# WL102 Amendment 2 chain: after chain102 (TransGI steps) has finished, retrain ECRO with the generator's area INPUT floored
# (WL102_AREA_FLOOR = 0.0625 / 16) and evaluate it; GPU-exclusive, same step()/timing convention as chain102.sh.
set -u
W=$HOME/rest_wl102; R=$W/results/wl102; X=$HOME/TransGI-Reproduction-data/scenes/living-room-3/scene_transgi.xml
EPY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
while tmux has-session -t =chain102 2>/dev/null; do sleep 20; done   # = exact name (a prefix match found chain102b itself)
grep -q "CHAIN COMPLETE" $R/chain.log || { echo "chain102 did not complete; not starting"; exit 1; }
step() {
  local n=$1; shift
  [ -f $R/done/$n ] && { echo "$(date -u +%FT%TZ) SKIP $n"; return 0; }
  echo "$(date -u +%FT%TZ) START $n"
  local t0=$(date +%s.%N)
  "$@" > $R/$n.log 2>&1; local rc=$?
  local t1=$(date +%s.%N)
  echo "$(date -u +%FT%TZ) END $n rc=$rc wall_s=$(echo "$t1 - $t0" | bc)"
  [ $rc -eq 0 ] && echo "$t0 $t1" > $R/done/$n || { echo "STOP at $n"; exit 1; }
}
export WL102_AREA_FLOOR=0.00390625
cd $W/experiments/ecro_lr3_102
step ecro_train_a2  $EPY train102.py $R/data $R/model_a2
step ecro_eval_a2   $EPY eval102.py $X $R/data $R/model_a2 $R/eval_a2
echo "$(date -u +%FT%TZ) CHAIN_B COMPLETE"
