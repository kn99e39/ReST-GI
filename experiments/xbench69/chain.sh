#!/usr/bin/env bash
# WL69 Stage 4 chain on LabServer69 (protocol xbench69_v1 + amendment 1). Detached: setsid -f bash chain.sh. Resumable (.done files).
set -Eeuo pipefail
R=$HOME/rest_xbench69; L=$R/launcher; mkdir -p $L $R/tmp
source $R/env_8dna.sh
export P53_MI_VARIANT=cuda_ad_rgb PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True XB69_PLY_DIR=$R/8dna_work/ply XB69_TEXTURES=$R/8dna_work/scenes/textures
K55=$R/repo/experiments/fnto_k55; XB=$R/repo/experiments/xbench69; W=$R/8dna_work
CUR=init
trap 'echo "failed $CUR $(date -Is)" > $L/chain.status; kill $SAMP 2>/dev/null || true' ERR
peers() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';'; }
( while true; do echo "$(date -Is) $(peers)" >> $L/gpu_sampler.log; sleep 15; done ) & SAMP=$!
wait_idle() { local n=0; while [ $n -lt 3 ]; do if [ -z "$(peers)" ]; then n=$((n + 1)); else n=0; fi; sleep 20; done; }
stage() {
  CUR=$1; shift
  if [ -f $L/$CUR.done ]; then echo "$(date -Is) skip $CUR" >> $L/steps.log; return; fi
  echo "running $CUR $(date -Is)" > $L/chain.status; wait_idle
  echo "$(date -Is) start $CUR peers=[$(peers)]" >> $L/steps.log
  "$@" > $L/$CUR.log 2>&1
  echo "$(date -Is) end $CUR peers=[$(peers)]" >> $L/steps.log; touch $L/$CUR.done
}
stage tests     bash -c "cd $K55 && $PM $XB/tests/test_xb69.py"
stage rg        bash -c "cd $K55 && $PM $XB/x_rg.py $R/rg"
stage cy        bash $XB/cy_ladder.sh
for s in 0 1; do
  stage d8_train_s$s bash -c "cd $W && $PM train.py --config configs/default.yaml scene=xbslab --max_epochs 30 --seed $s --device 0 --log_path $R/d8_logs --experiment_name xbslab_s$s"
  stage d8_eval_s$s  bash -c "cd $W && $PM $XB/d8_eval.py $R/d8 $R/d8_logs/xbslab_s$s/final.ckpt s$s"
done
stage analyze   bash -c "cd $K55 && $PM $XB/x_analyze.py $R $R/analysis.json"
echo "done $(date -Is)" > $L/chain.status; kill $SAMP 2>/dev/null || true
