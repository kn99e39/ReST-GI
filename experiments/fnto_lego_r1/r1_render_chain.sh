#!/usr/bin/env bash
# WL42 server render chain (LabServer63 CPU): Stage A r1 probes, then the r1 reference regime of the held-out views.
# Launch: setsid -f nohup bash r1_render_chain.sh > ~/rest_fnto_r1/launcher/render_chain.stdout.log 2>&1 < /dev/null
set -u
export RNA_DIR=$HOME/NeuralShaderDemo/external/relightable-neural-assets; PY=$RNA_DIR/.venv/bin/python
H=$HOME/rest_fnto_r1; X=$H/repo/experiments/fnto_lego_r1; O=$H/repo/results/fnto_lego_r1; L=$H/launcher
st() { echo "$1 $(date -Is)" > $L/render_chain.status; }
run() { echo "$1 start $(date -Is)" >> $L/render_runs.log; "${@:2}" > $L/$1.log 2>&1; rc=$?; echo "$1 rc=$rc $(date -Is)" >> $L/render_runs.log; [ $rc = 0 ] || { st "failed: $1"; exit $rc; }; }
mkdir -p $O/stagea/work $O/refs/work; st "running"
cd $O/stagea/work
run stagea_holdout $PY $X/r1_stagea_render.py -- $O/stagea holdout 120 2102 --res 512 --spp 512
run stagea_train   $PY $X/r1_stagea_render.py -- $O/stagea train 1000 1101 --res 512 --spp 512
echo "stagea done $(date -Is)" > $L/stagea.done
cd $O/refs/work
run first_T0 $PY $X/r1_refs_render.py -- $O/refs $H/views_r1.json heldout_T0 first --spp 1024
run first_T1 $PY $X/r1_refs_render.py -- $O/refs $H/views_r1.json heldout_T1 first --spp 1024
run full_T0  $PY $X/r1_refs_render.py -- $O/refs $H/views_r1.json heldout_T0 full --spp 4096
run full_T1  $PY $X/r1_refs_render.py -- $O/refs $H/views_r1.json heldout_T1 full --spp 4096
st "done"
