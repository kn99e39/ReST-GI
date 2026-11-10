#!/bin/bash
cd ~/rest_compiled83 && . ./env.sh && cd code/experiments/fnto_compiled83
R=$HOME/rest_compiled83/results
echo "$(date -u +%FT%TZ) amendment2: parts 2 / 4 of 5 stalled (no output in 9 min, GPU 0 %); killed; images 2, 4, 7, 9 relaunched as parts j/10 (one image per process)" >> $HOME/rest_compiled83/chain.status
for j in 2 4 7 9; do
  setsid nohup $PY render83.py render $R/data $HOME/rest_compiled81/results/data $R/fit $HOME/rest_compiled82/results/fit $R/render --part=$j/10 > $R/logs/F_render10_$j.log 2>&1 < /dev/null &
done
