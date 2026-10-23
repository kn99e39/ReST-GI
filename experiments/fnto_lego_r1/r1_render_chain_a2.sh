#!/usr/bin/env bash
# WL42 amendment 2 re-render on LabServer63 (CPU): identical configurations and seeds as r1_render_chain.sh, with the
# point-sampling film (GAUSSIAN 0.01; the first chain's BOX 0.01 was a 1 px box). The first chain's outputs are kept in
# stagea_box1px/ and refs_box1px/.
#   phase stagea : Stage A holdout 120 + train 1000          (then the PC prepares; the server GPU trains)
#   phase refs   : first-event T0/T1, full T0/T1 references   (may run on the server CPU while the server GPU trains)
# Launch: setsid -f nohup bash r1_render_chain_a2.sh <phase> > ~/rest_fnto_r1/launcher/a2_<phase>.stdout.log 2>&1 < /dev/null
set -u
export RNA_DIR=$HOME/NeuralShaderDemo/external/relightable-neural-assets; PY=$RNA_DIR/.venv/bin/python
H=$HOME/rest_fnto_r1; X=$H/repo/experiments/fnto_lego_r1; O=$H/repo/results/fnto_lego_r1; L=$H/launcher; P=$1
st() { echo "$1 $(date -Is)" > $L/a2_$P.status; }
run() { echo "$1 start $(date -Is)" >> $L/a2_runs.log; "${@:2}" > $L/a2_$1.log 2>&1; rc=$?; echo "$1 rc=$rc $(date -Is)" >> $L/a2_runs.log; [ $rc = 0 ] || { st "failed: $1"; exit $rc; }; }
st "running"
if [ "$P" = stagea ]; then
  mkdir -p $O/stagea/work; cd $O/stagea/work
  run stagea_holdout $PY $X/r1_stagea_render.py -- $O/stagea holdout 120 2102 --res 512 --spp 512
  run stagea_train   $PY $X/r1_stagea_render.py -- $O/stagea train 1000 1101 --res 512 --spp 512
else
  mkdir -p $O/refs/work; cd $O/refs/work
  run first_T0 $PY $X/r1_refs_render.py -- $O/refs $H/views_r1.json heldout_T0 first --spp 1024
  run first_T1 $PY $X/r1_refs_render.py -- $O/refs $H/views_r1.json heldout_T1 first --spp 1024
  run full_T0  $PY $X/r1_refs_render.py -- $O/refs $H/views_r1.json heldout_T0 full --spp 4096
  run full_T1  $PY $X/r1_refs_render.py -- $O/refs $H/views_r1.json heldout_T1 full --spp 4096
fi
st "done"
