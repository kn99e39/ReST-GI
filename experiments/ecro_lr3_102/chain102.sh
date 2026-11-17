#!/bin/bash
# WL102 stage-3 chain on LabServer69: every step sequential, so each timed step has the GPU to itself (protocol: GPU-exclusive).
# ECRO: dataset -> training -> evaluation; TransGI (separate repository, recommended configuration): datagen (val, train) ->
# training (LeakyReLU) -> fast-frame evaluation -> frame-time benchmark. Each step logs UTC start/end; resumable (done markers).
# usage: bash chain102.sh   (inside tmux; log ~/rest_wl102/results/wl102/chain.log)
set -u
W=$HOME/rest_wl102; R=$W/results/wl102; X=$HOME/TransGI-Reproduction-data/scenes/living-room-3/scene_transgi.xml
EPY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
TG=$HOME/TransGI-Reproduction
mkdir -p $R/done
step() {   # step <name> <command...>
  local n=$1; shift
  [ -f $R/done/$n ] && { echo "$(date -u +%FT%TZ) SKIP $n"; return 0; }
  echo "$(date -u +%FT%TZ) START $n"
  local t0=$(date +%s.%N)
  "$@" > $R/$n.log 2>&1; local rc=$?
  local t1=$(date +%s.%N)
  echo "$(date -u +%FT%TZ) END $n rc=$rc wall_s=$(echo "$t1 - $t0" | bc)"
  [ $rc -eq 0 ] && echo "$t0 $t1" > $R/done/$n || { echo "STOP at $n"; exit 1; }
}
cd $W/experiments/ecro_lr3_102
step ecro_gen       $EPY gen102.py $X $R/data
step ecro_train     $EPY train102.py $R/data $R/model
step ecro_eval      $EPY eval102.py $X $R/data $R/model $R/eval
cd $TG
step tg_gen_val     bash -c "source scripts/remote_env.sh; python scripts/gen_data.py configs/living_room_3.yaml --split val --parallel 8"
step tg_gen_train   bash -c "source scripts/remote_env.sh; python scripts/gen_data.py configs/living_room_3.yaml --split train --parallel 8"
step tg_train       bash -c "source scripts/remote_env.sh; python scripts/train_model.py configs/living_room_3.yaml --seed 0 --tag wl102_leaky --model-override '{\"activation\": \"LeakyReLU\"}'"
step tg_eval        bash -c "TRANSGI_VENV=transgi_fast; source scripts/remote_env.sh; python scripts/b03/eval_fast.py configs/living_room_3.yaml --ckpt \$TRANSGI_DATA/runs/living_room_3/train/wl102_leaky/checkpoint.pt --arm D_vis --out \$TRANSGI_DATA/runs/living_room_3/eval/wl102_leaky/static_V0.json"
step tg_bench       bash -c "TRANSGI_VENV=transgi_fast; source scripts/remote_env.sh; python scripts/b03/bench_fast.py configs/living_room_3.yaml --ckpt \$TRANSGI_DATA/runs/living_room_3/train/wl102_leaky/checkpoint.pt --tag _wl102_leaky"
echo "$(date -u +%FT%TZ) CHAIN COMPLETE"
