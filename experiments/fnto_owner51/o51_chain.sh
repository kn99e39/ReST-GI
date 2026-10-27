#!/usr/bin/env bash
# WL51 primary chain (protocol fnto_owner51): renderer control (CPU docker) runs in parallel with the GPU stages.
L=$HOME/rest_fnto_r1/launcher; P=$HOME/NeuralShaderDemo/external/relightable-neural-assets/.venv/bin/python; S=$L/o51_chain.status; TL=$L/o51_times.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
X=$HOME/rest_fnto_r1/repo/experiments/fnto_owner51; O=$HOME/rest_fnto_r1/repo/results/fnto_ownership51
cd $X; echo "== chain $(date -Is) local commit a066742 (files synced by tar; server repo HEAD $(git -C $X rev-parse --short HEAD))" >> $TL
( t=$(date +%s); docker run --rm -u 1013:1013 -v /home/nd:/home/nd -e HOME=/home/nd -e RNA_DIR=/home/nd/NeuralShaderDemo/external/relightable-neural-assets -w "$X" restgi/bpy-runtime:22.04 $P o51_slab_render.py -- $O/gate2_renderer 1024 4096 > $L/o51_slab.log 2>&1 && echo "TIME slab $(( $(date +%s)-t ))" >> $TL || echo "FAILED slab" >> $TL ) &
SLAB=$!
run() { echo "running $1 $(date -Is)" > $S; t=$(date +%s); $P o51_ops.py $1 $2 > $L/o51_$1.log 2>&1 || { echo "failed $1 $(date -Is)" > $S; exit 1; }; echo "TIME $1 $(( $(date +%s)-t ))" >> $TL; }
run g0 $O/gate0_provenance
run g3 $O/gate3_energy
run g1 $O/gate1_operator
run views $O/gate1_operator
echo "waiting slab $(date -Is)" > $S; wait $SLAB
$P o51_report.py $O > $L/o51_report.log 2>&1
echo "done $(date -Is)" > $S
