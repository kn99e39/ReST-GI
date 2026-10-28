#!/usr/bin/env bash
# WL55 Cycles references (CPU, 12 threads): emitter units, gap8 full, known-field fixtures, gap4 / floor / bend90 full (protocol stage_B).
R=$HOME/rest_wl55; C="$R/repo/experiments/fnto_c54/bpy_native.sh $R/repo/experiments/fnto_k55/k55_cycles.py --"; S=$R/launcher/cycles.status; L=$R/launcher
cd $R/repo/experiments/fnto_k55; echo "running $(date -Is)" > $S
run() { echo "$(date -Is) start $*" >> $L/cycles_steps.log; $C $R/out/cycles "$@" >> $L/cycles.log 2>&1 || echo "$(date -Is) FAILED $*" >> $L/cycles.fail; }
run units
run render gap8 all 5501,5502
for sc in kf_gap8_U kf_gap8_X kf_gap8_S kf_gap4_U kf_gap4_X kf_gap4_S; do run render $sc all 5511,5512; done
for sc in gap4 floor bend90; do run render $sc all 5501,5502; done
echo "done $(date -Is)" > $S
