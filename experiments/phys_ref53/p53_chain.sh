#!/usr/bin/env bash
# WL53 primary chain on LabServer69 (protocol phys_ref53). GPU branch (TR / C oracle, capped torch memory, WL52 shares the
# GPU) and CPU branch (Cycles volume reference in docker, then Mitsuba diagnostic, then port classification) run in
# parallel after the tests; the report runs when both are done. Status: ~/rest_wl53/launcher/p53_{chain,cpu}.status
L=$HOME/rest_wl53/launcher; mkdir -p $L; S=$L/p53_chain.status; SC=$L/p53_cpu.status; TL=$L/p53_times.log
X=$HOME/rest_wl53/repo/experiments/phys_ref53; O=$HOME/rest_wl53/results; mkdir -p $O/render $O/oracle $O/cycles
PM=$HOME/NeuralShaderDemo/external/8dna26/.venv/bin/python; PB=$HOME/NeuralShaderDemo/external/relightable-neural-assets/.venv/bin/python
export PYTHONUNBUFFERED=1 P53_GPU_FRAC=0.35 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f /usr/lib64/libLLVM-11.so ] && export DRJIT_LIBLLVM_PATH=/usr/lib64/libLLVM-11.so   # LabServer69: Dr.Jit LLVM backend (renders verified exact)
cd $X; echo "== chain $(date -Is) commit $(cat $HOME/rest_wl53/repo_commit.txt) host $(hostname)" >> $TL
echo "running tests $(date -Is)" > $S
P53_MI_VARIANT=cuda_ad_rgb $PM tests/test_p53.py $O/tests.json > $L/p53_tests.log 2>&1 || { echo "failed tests $(date -Is)" > $S; exit 1; }
(
  run() { echo "running cpu $1 $(date -Is)" > $SC; t=$(date +%s); "${@:2}" > $L/p53_cpu_$1.log 2>&1 || { echo "failed cpu $1 $(date -Is)" > $SC; exit 1; }; echo "TIME cpu $1 $(( $(date +%s)-t ))" >> $TL; }
  run cycles docker run --rm -u 1013:1013 -v /home/nd:/home/nd -e HOME=/home/nd -w "$X" restgi/bpy-runtime:22.04 $PB p53_cycles.py -- $O/cycles
  run mitsuba env P53_MI_VARIANT=llvm_ad_rgb $PM p53_render.py all $O/render
  run cports env P53_MI_VARIANT=llvm_ad_rgb $PM p53_render.py cports $O/render
  echo "done cpu $(date -Is)" > $SC
) &
CPU=$!
run() { echo "running $1 $(date -Is)" > $S; t=$(date +%s); P53_MI_VARIANT=cuda_ad_rgb $PM p53_oracle.py $1 $O/oracle > $L/p53_$1.log 2>&1 || { echo "failed $1 $(date -Is)" > $S; exit 1; }; echo "TIME $1 $(( $(date +%s)-t ))" >> $TL; }
run tr
run g1a_iw
run g2
run compose
run ops
echo "waiting cpu $(date -Is)" > $S
wait $CPU
grep -q "^done" $SC || { echo "failed cpu-branch $(date -Is)" > $S; exit 1; }
echo "running report $(date -Is)" > $S
$PM p53_report.py $O > $L/p53_report.log 2>&1 || { echo "failed report $(date -Is)" > $S; exit 1; }
echo "done $(date -Is)" > $S
