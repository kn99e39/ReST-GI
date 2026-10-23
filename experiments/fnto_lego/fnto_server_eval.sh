#!/usr/bin/env bash
# Worklog 41 on LabServer63 (protocol fnto_v1, amendment 1): evaluation of the trained kernels K_main.
# Port sun visibility per held-out view was precomputed on the PC (Mitsuba, eval_vis.npz); everything else is torch.
set -u
H=$HOME/rest_fnto; P=$H/repo; U=$HOME/NeuralShaderDemo/external/relightable-neural-assets; PY=$U/.venv/bin/python
X=$P/experiments/fnto_lego; O=$P/results/fnto_lego; L=$H/launcher; K=$O/K_main/kernels.pt
st() { echo "$1 $(date -Is)" > $L/eval.status; }
st "running"; cd $X
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ] || { st "failed: GPU not idle"; exit 1; }
$PY -u fnto_stagea_eval.py $K $O/eval_main/stagea > $L/eval_stagea.log 2>&1; echo "stagea rc=$? $(date -Is)" >> $L/eval_runs.log
$PY -u fnto_eval.py run --kernels $K --out $O/eval_main --first $O/firstevent --keep 6 15 > $L/eval_main.log 2>&1; echo "main rc=$? $(date -Is)" >> $L/eval_runs.log
$PY -u -c "
import json, numpy as np, torch, fnto_eval as EV, fnto_solve as SV
dev = torch.device('cuda'); KA, KB = EV.load_kernels('$K', dev); T = SV.Tables(KA, KB, dev); a = SV.uniform_albedo(T); A = np.load('$O/anchors.npz')
r = {'quantiles_50_90_99': np.quantile(a, [0.5, 0.9, 0.99]).tolist(), 'max': float(a.max()), 'frac_gt_1': float((a > 1).mean()),
     'frac_gt_1_by_material': {str(int(m)): float((a[A['mat'] == m] > 1).mean()) for m in np.unique(A['mat'])}}
open('$O/eval_main/passivity.json', 'w').write(json.dumps(r, indent=1)); print(r)
" > $L/eval_passivity.log 2>&1; echo "passivity rc=$? $(date -Is)" >> $L/eval_runs.log
[ -f $O/eval_main/eval.json ] && st "done" || st "failed"
