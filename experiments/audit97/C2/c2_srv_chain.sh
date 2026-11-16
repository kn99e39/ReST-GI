#!/bin/bash
# WL97 audit C2 server chain (own directory only): audit tables -> 4 renders (1 rep, 4,096 spp, ~10 s GPU each) -> learned-sender statistics
# (~10 s GPU) -> CPU image analysis.  One GPU process at a time.   setsid nohup bash c2_srv_chain.sh > chain.out 2>&1 &
set -u
B=$HOME/rest_x97/C2; R=$B/R; CODE=$HOME/rest_x96a/code/experiments
. $HOME/rest_x95a/env.sh
mkdir -p $R/results/renders $R/results/tables $R/results/lists $B/out
ln -sfn $HOME/rest_x94/data $R/data
rm -f $B/done
cd $B
date -u +%FT%TZ
$PY c2_srv_tables96a.py $R || { echo TABLES_FAILED; touch $B/done; exit 1; }
$PY $CODE/ecro_mat94/mat94.py render $R $R/results/lists/c2.txt || echo RENDER_FAILED
date -u +%FT%TZ
$PY c2_srv_learned95a.py $CODE $R $HOME/rest_x92a $B/out/learned95a.json > $B/out/learned95a.log 2>&1 || echo LEARNED_FAILED
date -u +%FT%TZ
CUDA_VISIBLE_DEVICES= $PY c2_img96a_server.py $R $B/out/img96a.json > $B/out/img96a.log 2>&1 || echo IMG_FAILED
date -u +%FT%TZ
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
touch $B/done
