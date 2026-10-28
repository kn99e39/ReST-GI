#!/usr/bin/env bash
# WL55 GPU chain (sequential): official tests, WL53 / WL54 regression, Stage A, Stage B operators / sun / COR / compositions / known fields.
R=$HOME/rest_wl55; RUN=$R/repo/experiments/fnto_k55/run.sh; O=$R/out; L=$R/launcher; S=$L/gpuB.status; mkdir -p $O/tests $O/stageA
echo "running $(date -Is)" > $S
step() { echo "$(date -Is) start $*" >> $L/gpuB_steps.log; "$@" >> $L/gpuB.log 2>&1 || echo "$(date -Is) FAILED $*" >> $L/gpuB.fail; }
step $RUN tests/test_k55.py $O/tests/tests_k55.json
step $RUN ../phys_ref53/tests/test_p53.py $O/tests/wl53_regression.json
step $RUN ../fnto_c54/tests/test_c54.py $O/tests/wl54_regression.json $HOME/rest_wl54/out/cycles
step $RUN k55_stageA.py code $O/stageA/code_audit.json
step $RUN k55_stageA.py replay $O/stageA/replay.json
for op in slab_L1d_s1 slab_L1d_s2 slab_L2d_s1 slab_L1d_din64 slab_L4_s1 slab_L4_s2 bend90_L4_s1 bend90_L4_s2; do step $RUN k55_stageB.py ops $op; done
for sc in gap8 gap4 floor bend90 kf_gap8_S kf_gap4_S; do step $RUN k55_stageB.py sun $sc; done
step $RUN k55_stageB.py cor floor_lam 5571
for sc in gap8 gap4 floor bend90; do step $RUN k55_stageB.py cor $sc 5571,5572; done
for sc in kf_gap8_U kf_gap8_X kf_gap8_S kf_gap4_U kf_gap4_X kf_gap4_S; do step $RUN k55_stageB.py cor $sc 5573,5574; done
for sc in gap8 gap4 floor bend90; do step $RUN k55_stageB.py compose $sc all; done
for sc in kf_gap8_U kf_gap8_X kf_gap8_S kf_gap4_U kf_gap4_X kf_gap4_S; do step $RUN k55_stageB.py kf $sc all; done
step $RUN k55_stageB.py btest gap8; step $RUN k55_stageB.py btest gap4
echo "done $(date -Is)" > $S
