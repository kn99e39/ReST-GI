#!/usr/bin/env bash
# WL44 server chain (LabServer63, protocol fnto_k44). Requires prep2_{holdout,train} and eval_vis_k44.npz in the server
# results tree (made on the PC by k44_pc_runner.sh). Steps (stop at the first failure):
#   1 smoke: k44_train --max-steps 100 (finite loss, peak memory)       4 r1_eval run K_k44P  (R1_EVAL_VIS=eval_vis_k44.npz)
#   2 K_k44P  = k44_train --passivity 1.0 --seed 0                       5 r1_eval run K_r1P_s1 (r1 eval_vis.npz)
#   3 K_r1P_s1 = r1_train --passivity 1.0 --seed 1 (seed control)        6 k44_decomp x3, k44_verdict
# Status: ~/rest_fnto_r1/launcher/k44_chain.status ; runs log k44_chain_runs.log
set -u
export RNA_DIR=$HOME/NeuralShaderDemo/external/relightable-neural-assets; PY=$RNA_DIR/.venv/bin/python
H=$HOME/rest_fnto_r1; X=$H/repo/experiments/fnto_lego_k44; X1=$H/repo/experiments/fnto_lego_r1; O=$H/repo/results/fnto_lego_r1; L=$H/launcher
st() { echo "$1 $(date -Is)" > $L/k44_chain.status; }
log() { echo "$1 $(date -Is)" >> $L/k44_chain_runs.log; }
run() { log "$1 start"; st "running: $1"; "${@:2}" > $L/k44_$1.log 2>&1; rc=$?; log "$1 rc=$rc"; [ $rc = 0 ] || { st "failed: $1"; exit $rc; }; }
cd $X
run smoke_train $PY -u k44_train.py --out $O/K_k44P_smoke --passivity 1.0 --seed 0 --max-steps 100 --eval-every 1000000
$PY - <<EOF || { st "failed: smoke gate"; exit 3; }
import json, math, sys
d = json.load(open("$O/K_k44P_smoke/train.json"))
ok = all(math.isfinite(x["loss"]) for x in d["log"]) and d["peak_mem_gib"] < 11.0
print("smoke loss", d["log"][-1], "peak", d["peak_mem_gib"]); sys.exit(0 if ok else 1)
EOF
log "smoke gate passed"
run train_K_k44P $PY -u k44_train.py --out $O/K_k44P --passivity 1.0 --seed 0
cd $X1
run train_K_r1P_s1 $PY -u r1_train.py --out $O/K_r1P_s1 --passivity 1.0 --seed 1
run eval_K_k44P env R1_EVAL_VIS=eval_vis_k44.npz $PY -u r1_eval.py run --kernels $O/K_k44P/kernels.pt --out $O/eval_K_k44P
run eval_K_r1P_s1 $PY -u r1_eval.py run --kernels $O/K_r1P_s1/kernels.pt --out $O/eval_K_r1P_s1
cd $X
for t in K_r1P K_r1P_s1 K_k44P; do run decomp_$t $PY -u k44_decomp.py $O/eval_$t $t; done
run verdict $PY -u k44_verdict.py
st "done"
