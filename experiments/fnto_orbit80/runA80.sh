#!/bin/bash
# WL80 chain A on LabServer63 (one detached process; frozen code = the deployed commit): Stage 0 reconciliation -> Stage 1 population ->
# basis restarts r0 || r1, then r2 -> selection -> Stage 2 oracle (G1 full || non-selected restarts descriptive) -> Stage 2 gate.
# Other projects' jobs are never touched; GPU work waits for < 1 GB used on two checks 60 s apart.
set -u
PY=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python;D=${R80_ROOT:-$HOME/rest_orbit80};cd $D/repo/experiments/fnto_orbit80
export R80_ROOT=$D P53_MI_VARIANT=cuda_ad_rgb WL57_WALKER=fast PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=$HOME/nvlibs/optix595 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $D/logs $D/stage0 $D/basis $D/stage2
st(){ echo "$1 $(date -u '+%F %T')" > $D/runA.status; }
gpu(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)" >> $D/logs/gpu_sampler.log; sleep 15; done ) & SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT
while :; do u=$(gpu); if [ "$u" -lt 1000 ]; then sleep 60; u=$(gpu); [ "$u" -lt 1000 ] && break; fi; st "waiting for a free GPU (${u} MiB used)"; sleep 120; done
run(){ name=$1; shift; st "running $name"; $PY "$@" > $D/logs/$name.log 2>&1 || { st "FAILED $name"; exit 1; }; }
[ -f $D/stage0/stage0.json ] || run stage0 stage0_80.py $D/stage0/stage0.json
grep -q '"valid": true' $D/stage0/stage0.json || { st "FAILED stage0 contract invalid"; exit 1; }
[ -f $D/basis/targets80.pt ] || run prepare basis80.py prepare
st "running basis r0 || r1"
( [ -f $D/basis/B80_r0.pt ] || $PY basis80.py fit 0 > $D/logs/basis_r0.log 2>&1 ) & P0=$!
( [ -f $D/basis/B80_r1.pt ] || $PY basis80.py fit 1 > $D/logs/basis_r1.log 2>&1 ) & P1=$!
wait $P0 || { st "FAILED basis r0"; exit 1; }; wait $P1 || { st "FAILED basis r1"; exit 1; }
[ -f $D/basis/B80_r2.pt ] || run basis_r2 basis80.py fit 2
[ -f $D/basis/G1.pt ] || run select basis80.py select
SEL=$($PY -c "import json;print(json.load(open('$D/basis/select.json'))['selected_restart'])")
st "running stage2 oracle G1 || descriptive restarts"
( [ -f $D/stage2/G1/stage1/oracle.json ] || $PY oracle80.py run G1 > $D/logs/oracle_G1.log 2>&1 ) & PG=$!
( for r in 0 1 2; do [ "$r" = "$SEL" ] && continue; [ -f $D/stage2/B80_r$r/stage1/oracle.json ] || $PY oracle80.py run B80_r$r > $D/logs/oracle_r$r.log 2>&1 || exit 1; done ) & PR=$!
wait $PG || { st "FAILED oracle G1"; exit 1; }; wait $PR || { st "FAILED oracle restarts"; exit 1; }
run stage2_decide oracle80.py decide
st "done stageA: $(tail -1 $D/logs/stage2_decide.log | cut -c1-140)"
