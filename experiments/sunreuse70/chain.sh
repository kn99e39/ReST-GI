#!/usr/bin/env bash
# WL70 chain on LabServer69 (protocol sunreuse70_v1). Detached: setsid -f bash chain.sh. Resumable (.done files).
# Stages: tests -> identity (Stage 2/3) -> gate (stop unless the predeclared validity rule holds) -> e2e (Stage 5) -> analyze.
set -Eeuo pipefail
R=$HOME/rest_sunreuse70; L=$R/launcher; mkdir -p $L $R/tmp
source $R/env_8dna.sh
export P53_MI_VARIANT=cuda_ad_rgb PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K55=$R/repo/experiments/fnto_k55; SR=$R/repo/experiments/sunreuse70
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
stage tests    bash -c "cd $K55 && $PM $SR/tests/test_sr70.py"
stage identity bash -c "cd $K55 && $PM $SR/sr70_identity.py $R/identity"
if ! $PM $SR/sr70_gate.py $R/identity/identity.json > $L/gate.log 2>&1; then
  echo "stopped_at_gate $(date -Is)" > $L/chain.status; kill $SAMP 2>/dev/null || true; exit 0
fi
stage e2e      bash -c "cd $K55 && $PM $SR/sr70_e2e.py $R/e2e"
stage analyze  bash -c "cd $K55 && $PM $SR/sr70_analyze.py $R $R/analysis.json"
echo "done $(date -Is)" > $L/chain.status; kill $SAMP 2>/dev/null || true
