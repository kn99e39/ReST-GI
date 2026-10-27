#!/usr/bin/env bash
# WL52 primary chain after amendment 1 (data from chain 1) (protocol fnto_lc52): data -> training -> evaluation -> benchmark -> report.
L=$HOME/rest_fnto_r1/launcher; P=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python; S=$L/lc_chain2.status; TL=$L/lc_times.log
X=$HOME/rest_fnto_r1/repo/experiments/fnto_lc52; O=$HOME/rest_fnto_r1/repo/results/fnto_lc52
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $X; echo "== chain $(date -Is) local commit $1" >> $TL
run() { n=$1; shift; echo "running $n $(date -Is)" > $S; t=$(date +%s); "$@" > $L/lc_$n.log 2>&1 || { echo "failed $n $(date -Is)" > $S; exit 1; }; echo "TIME $n $(( $(date +%s)-t ))" >> $TL; }


run train $P lc_train.py $O/data $O/train 60000
run eval $P lc_eval.py $O/train/ckpt_best.pt $O/eval
run bench $P lc_bench.py $O/train/ckpt_best.pt $O/bench
run report $P lc_report.py $O
echo "done $(date -Is)" > $S
