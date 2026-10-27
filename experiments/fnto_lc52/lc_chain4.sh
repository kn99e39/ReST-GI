#!/usr/bin/env bash
# WL52 chain 4: benchmark on a fully idle server (no GPU compute process AND 1-min load average < 3); chain 3's
# "bench_idle" had an idle GPU but ~14 / 16 CPU cores busy with an unrelated Cycles render (load ~17).
L=$HOME/rest_fnto_r1/launcher; P=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python; S=$L/lc_chain4.status; TL=$L/lc_times.log
X=$HOME/rest_fnto_r1/repo/experiments/fnto_lc52; O=$HOME/rest_fnto_r1/repo/results/fnto_lc52
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $X; echo "== chain4 $(date -Is)" >> $TL
echo "waiting fully idle server $(date -Is)" > $S
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || [ "$(awk '{print ($1 < 3.0)}' /proc/loadavg)" != "1" ]; do sleep 30; done
echo "load $(cat /proc/loadavg)" >> $TL; echo "running bench_clean $(date -Is)" > $S; t=$(date +%s)
$P lc_bench.py $O/train/ckpt_best.pt $O/bench_clean > $L/lc_bench_clean.log 2>&1 && { echo "TIME bench_clean $(( $(date +%s)-t ))" >> $TL; echo "load_after $(cat /proc/loadavg)" >> $TL; echo "done $(date -Is)" > $S; } || echo "failed bench_clean $(date -Is)" > $S
