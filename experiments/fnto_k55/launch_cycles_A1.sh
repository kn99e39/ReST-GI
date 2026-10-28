#!/usr/bin/env bash
# WL55 amendment A1.2 pixel diagnostic + held-out gap6s + Stage E references (CUDA, official Blender 3.5.1).
R=$HOME/rest_wl55; B=$R/blender-3.5.1-linux-x64/blender; S=$R/launcher/cyclesA1.status; L=$R/launcher
export TMPDIR=$R/tmp PYTHONPATH=$HOME/NeuralShaderDemo/external/relightable-neural-assets/.venv/lib/python3.10/site-packages
cd $R/repo/experiments/fnto_k55; echo "running $(date -Is)" > $S
run() { o=$1; shift; mkdir -p $o; echo "$(date -Is) start $o $*" >> $L/cyclesA1_steps.log; $B -b --factory-startup --python-use-system-env --python k55_cycles.py -- $o "$@" device=CUDA threads=4 >> $L/cyclesA1.log 2>&1 || echo "$(date -Is) FAILED $*" >> $L/cyclesA1.fail; }
for px in 0.25 0.125; do
  for sc in gap8 gap4; do run $R/out/cycles_px$px render $sc obot,obotx 5541 px=$px; done
  for sc in kf_gap8_U kf_gap4_U; do run $R/out/cycles_px$px render $sc obot,obotx 5542 px=$px; done
done
run $R/out/cycles_cuda render gap6s all 5521,5522
for sc in floor_c4 dent; do run $R/out/cycles_cuda render $sc all 5531,5532; done
echo "done $(date -Is)" > $S
