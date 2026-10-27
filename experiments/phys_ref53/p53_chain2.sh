#!/usr/bin/env bash
# WL53 amendment 2 rerun of the oracle pipeline (float64 accumulators) on LabServer69; then the report once the CPU branch is done.
L=$HOME/rest_wl53/launcher; S=$L/p53_chain2.status; SC=$L/p53_cpu.status; TL=$L/p53_times.log
X=$HOME/rest_wl53/repo/experiments/phys_ref53; O=$HOME/rest_wl53/results
PM=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python
export PYTHONUNBUFFERED=1 P53_GPU_FRAC=0.8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True P53_MI_VARIANT=cuda_ad_rgb
cd $X; echo "== chain2 $(date -Is) commit $(cat $HOME/rest_wl53/repo_commit.txt)" >> $TL
[ -d $O/oracle_v1_float32acc ] || mv $O/oracle $O/oracle_v1_float32acc
mkdir -p $O/oracle; cp $O/oracle_v1_float32acc/g1a_iw.json $O/oracle/
for st in tr g2 compose ops tr_hi ops_hi; do
  echo "running $st $(date -Is)" > $S; t=$(date +%s)
  $PM p53_oracle.py $st $O/oracle > $L/p53_v2_$st.log 2>&1 || { echo "failed $st $(date -Is)" > $S; exit 1; }
  echo "TIME v2 $st $(( $(date +%s)-t ))" >> $TL
done
echo "waiting cpu $(date -Is)" > $S
while ! grep -qE "^(done|failed)" $SC; do sleep 60; done
grep -q "^done" $SC || { echo "failed cpu-branch $(date -Is)" > $S; exit 1; }
echo "running report $(date -Is)" > $S
$PM p53_report.py $O > $L/p53_report.log 2>&1 || { echo "failed report $(date -Is)" > $S; exit 1; }
echo "done $(date -Is)" > $S
