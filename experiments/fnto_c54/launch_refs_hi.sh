#!/usr/bin/env bash
# amendment 3: pixel-converged references (Cycles, CPU, docker) -- Lambert near-contact cameras at 0.25 mm and the WL53 gap8 volume oblique camera
R=$HOME/rest_wl54; PB=$HOME/NeuralShaderDemo/external/relightable-neural-assets/.venv/bin/python; S=$R/launcher/refs_hi.status; export TMPDIR=$R/tmp; mkdir -p $TMPDIR
# Cycles runs WITHOUT Docker (bpy_native.sh: same venv python + bpy, the image's Ubuntu loader/libs extracted to /home; bit-identical to the
# container on a Lambert and a volume test render, 2026-10-28). The previous docker line is kept for reference:
# D="docker run --rm -u 1013:1013 -v /home/nd:/home/nd -e HOME=/home/nd -w $R/repo/experiments/fnto_c54 restgi/bpy-runtime:22.04 $PB c54_cycles_res.py --"
cd $R/repo/experiments/fnto_c54
D="$R/repo/experiments/fnto_c54/bpy_native.sh c54_cycles_res.py --"
echo "running $(date -Is)" > $S
$D $R/out/cycles_hi_vol vol:gap8 obot 0.5,0.25 seeds=5499 spp=1024 > $R/launcher/refs_hi_vol.log 2>&1 || echo "vol failed" >> $R/launcher/refs_hi.fail
$D $R/out/cycles_hi gap4 gapdown,gapup,obot 0.25 seeds=5499,5498 > $R/launcher/refs_hi_1.log 2>&1 || echo "gap4 failed" >> $R/launcher/refs_hi.fail
$D $R/out/cycles_hi gap8 gapdown,gapup,obot 0.25 seeds=5499,5498 > $R/launcher/refs_hi_2.log 2>&1 || echo "gap8 failed" >> $R/launcher/refs_hi.fail
for f in floor_lam rot_lam lblock_lam; do $D $R/out/cycles_hi $f underup,underdown 0.25 seeds=5499 > $R/launcher/refs_hi_$f.log 2>&1 || echo "$f failed" >> $R/launcher/refs_hi.fail; done
echo "done $(date -Is)" > $S
