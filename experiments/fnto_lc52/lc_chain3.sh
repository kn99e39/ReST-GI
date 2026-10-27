#!/usr/bin/env bash
# WL52 chain 3: amendment 3 post-hoc localized diagnostic, then the benchmark rerun on an IDLE GPU (the chain-2 benchmark ran
# while another job's GPU process was active and is kept as contended evidence).
L=$HOME/rest_fnto_r1/launcher; P=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python; S=$L/lc_chain3.status; TL=$L/lc_times.log
X=$HOME/rest_fnto_r1/repo/experiments/fnto_lc52; O=$HOME/rest_fnto_r1/repo/results/fnto_lc52
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $X; echo "== chain3 $(date -Is) local commit $1" >> $TL
run() { n=$1; shift; echo "running $n $(date -Is)" > $S; t=$(date +%s); "$@" > $L/lc_$n.log 2>&1 || { echo "failed $n $(date -Is)" > $S; exit 1; }; echo "TIME $n $(( $(date +%s)-t ))" >> $TL; }
run local $P lc_local.py $O/train/ckpt_best.pt $O/local
echo "waiting idle GPU $(date -Is)" > $S
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader >> $TL
run bench_idle $P lc_bench.py $O/train/ckpt_best.pt $O/bench_idle
echo "done $(date -Is)" > $S
