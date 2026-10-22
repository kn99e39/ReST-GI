#!/usr/bin/env bash
# Worklog 39, protocol spt_v1 amendment 1 (execution only): SPT_T1 ran out of GPU memory next to SPT_T0 (12 GB 3080 Ti),
# so it runs after SPT_T0 instead of concurrently. Same command, data and settings.
set -u
H=$HOME/rest_spt; P=$H/repo; U=$HOME/NeuralShaderDemo/external/relightable-neural-assets; PY=$U/.venv/bin/python
X=$P/experiments/spt_lego; O=$P/results/spt_lego; L=$H/launcher
echo "waiting for SPT_T0 $(date -Is)"
until [ -f $O/SPT_T0/train.json ] && ! pgrep -f "spt_train.py --data train_T0" > /dev/null; do sleep 60; done
echo "SPT_T0 finished; starting SPT_T1 $(date -Is)"
mv $L/train_SPT_T1.log $L/train_SPT_T1_oom_concurrent.log 2>/dev/null
[ -d $O/SPT_T1 ] && mv $O/SPT_T1 $O/SPT_T1_oom_concurrent
cd $X && $PY -u spt_train.py --data train_T1 --state T1 --out $O/SPT_T1 --epochs 150 --pixels 131072 --seed 0 > $L/train_SPT_T1.log 2>&1
rc=$?; echo "SPT_T1 rc=$rc $(date -Is)" >> $L/runs.log
[ $rc -eq 0 ] && [ -f $O/SPT_T1/state.pt ] && echo "done $(date -Is)" > $L/t1.status || echo "failed rc=$rc $(date -Is)" > $L/t1.status
