#!/bin/bash
# WL102 Amendment 3 chain: original configuration (no area floor), seed 10202 (recovered in the seed diagnostic), 24,000 steps,
# GPU-exclusive, then evaluation; same step()/timing convention as chain102.sh.
set -u
W=$HOME/rest_wl102; R=$W/results/wl102; X=$HOME/TransGI-Reproduction-data/scenes/living-room-3/scene_transgi.xml
EPY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
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
unset WL102_AREA_FLOOR
export WL102_SEED=10202
cd $W/experiments/ecro_lr3_102
step ecro_train_s10202  $EPY train102.py $R/data $R/model_s10202
step ecro_eval_s10202   $EPY eval102.py $X $R/data $R/model_s10202 $R/eval_s10202
echo "$(date -u +%FT%TZ) CHAIN_C COMPLETE"
