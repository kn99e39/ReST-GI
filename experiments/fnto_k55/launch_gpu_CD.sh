#!/usr/bin/env bash
# WL55 amendment A1: Stage D operator tests, dent operators, held-out gap6s and Stage E scenes (sun, COR), CAK and stale arms.
R=$HOME/rest_wl55; RUN=$R/repo/experiments/fnto_k55/run.sh; O=$R/out; L=$R/launcher; S=$L/gpuCD.status; mkdir -p $O/stageD
echo "running $(date -Is)" > $S
step() { echo "$(date -Is) start $*" >> $L/gpuCD_steps.log; "$@" >> $L/gpuCD.log 2>&1 || echo "$(date -Is) FAILED $*" >> $L/gpuCD.fail; }
step $RUN k55_stageD.py ops $O/stageD/ops_tests.json
for op in dent_L4_s1 dent_L4_s2; do step $RUN k55_stageB.py ops $op; done
for sc in gap6s floor_c4 dent; do step $RUN k55_stageB.py sun $sc; step $RUN k55_stageB.py cor $sc 5571,5572; done
for sc in gap8 gap4 floor bend90; do step $RUN k55_stageB.py compose $sc CAK_s1,CAK_s2; done
step $RUN k55_stageB.py compose gap6s all; step $RUN k55_stageB.py compose gap6s CAK_s1,CAK_s2
for sc in bend90 dent; do step $RUN k55_stageB.py compose $sc all; step $RUN k55_stageB.py compose $sc CAK_s1,CAK_s2,P10st_s1,P10st_s2,CAKst_s1,CAKst_s2,P025st_s1,P025st_s2; done
step $RUN k55_stageB.py compose floor_c4 all; step $RUN k55_stageB.py compose floor_c4 CAK_s1,CAK_s2
echo "done $(date -Is)" > $S
