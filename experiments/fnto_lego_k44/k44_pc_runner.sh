#!/usr/bin/env bash
# WL44 PC runner (Mitsuba steps; protocol fnto_k44). Waits until the WL43 server chain has ended (user decision
# 2026-10-25: no GPU work for WL44 while WL43 runs; no PC/server parallel), then:
#   1 smoke: K44_MAX_IMAGES=3 k44_prep holdout -> gate: 1->0 bit flips <= 0.3 % of pixel-light pairs and some 0->1 flips
#   2 k44_prep holdout train (full) ; 3 k44_eval_vis
#   4 stream prep2_* and eval_vis_k44.npz to the server, verify hashes ; 5 start k44_chain.sh on the server
# Status: results/fnto_lego_k44/pc_runner.status
set -u
R=/c/Projects/ReST-GI; X=$R/experiments/fnto_lego_k44; O=$R/results/fnto_lego_r1; PY=$R/external/8dna26/.venv-cu128/Scripts/python.exe
S=$R/results/fnto_lego_k44/pc_runner.status; LG=$R/results/fnto_lego_k44/pc_runner.log
st() { echo "$1 $(date '+%F %T')" > $S; echo "$1 $(date '+%F %T')" >> $LG; }
st "waiting: WL43 chain"
while true; do
  s=$(ssh LabServer63 'cat ~/rest_8dna_baseline/launcher/dyn_chain.status' 2>/dev/null)
  case "$s" in done*|failed*|blocked*) break;; esac
  sleep 300
done
st "WL43 ended: [$s]"
cd $X
st "running: prep smoke"
K44_MAX_IMAGES=3 $PY -u k44_prep.py holdout >> $LG 2>&1 || { st "failed: prep smoke"; exit 1; }
$PY - <<EOF || { st "failed: prep smoke gate"; exit 2; }
import json, sys
m = json.load(open(r"$O/stagea/prep2smoke_holdout/meta.json"))
f10 = m["bits_1to0"] / max(m["pixel_light_pairs"], 1); print(m)
sys.exit(0 if (f10 <= 0.003 and m["bits_0to1"] > 0) else 1)
EOF
st "running: prep holdout train"
$PY -u k44_prep.py holdout train >> $LG 2>&1 || { st "failed: prep"; exit 3; }
st "running: eval_vis"
$PY -u k44_eval_vis.py >> $LG 2>&1 || { st "failed: eval_vis"; exit 4; }
st "running: transfer"
cd $O && tar cf - stagea/prep2_holdout stagea/prep2_train eval_vis_k44.npz | gzip -1 | ssh LabServer63 'cd ~/rest_fnto_r1/repo/results/fnto_lego_r1 && gunzip | tar xf -' || { st "failed: transfer"; exit 5; }
(cd $O/stagea && find prep2_holdout prep2_train -type f | sort | xargs sha256sum | sed 's/ \*/  /') > "$TEMP/k44_pc.txt"
ssh LabServer63 'cd ~/rest_fnto_r1/repo/results/fnto_lego_r1/stagea && find prep2_holdout prep2_train -type f | sort | xargs sha256sum' > "$TEMP/k44_srv.txt"
diff -q "$TEMP/k44_pc.txt" "$TEMP/k44_srv.txt" || { st "failed: hash mismatch"; exit 6; }
scp -q $X/*.py $X/k44_chain.sh LabServer63:rest_fnto_r1/repo/experiments/fnto_lego_k44/ && scp -q $X/protocol/fnto_k44.json LabServer63:rest_fnto_r1/repo/experiments/fnto_lego_k44/protocol/ && \
  scp -q $R/experiments/fnto_lego_r1/r1_eval.py LabServer63:rest_fnto_r1/repo/experiments/fnto_lego_r1/ || { st "failed: code sync"; exit 7; }
ssh LabServer63 'mkdir -p ~/rest_fnto_r1/repo/experiments/fnto_lego_k44/protocol; cd ~/rest_fnto_r1/repo/experiments/fnto_lego_k44 && setsid -f nohup bash k44_chain.sh > ~/rest_fnto_r1/launcher/k44_chain.stdout.log 2>&1 < /dev/null' || { st "failed: chain launch"; exit 8; }
st "done: server chain launched"
