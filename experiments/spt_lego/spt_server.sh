#!/usr/bin/env bash
# Worklog 39 on LabServer63 (protocol spt_v1): SPT_T0 (train_T0 + relations T0) and SPT_T1 (train_T1 + relations T1,
# the candidate's own full reconstruction), started concurrently on the one GPU. Evaluation runs on the PC afterwards
# (held-out references and the worklog-37 region record live there).
# Layout: $H/repo = git archive of the declaring commit + results/spt_lego inputs (sha256-checked, data.sha256).
# Launch: setsid -f nohup bash spt_server.sh > $H/launcher/chain.stdout.log 2>&1 < /dev/null
set -u
H=$HOME/rest_spt; P=$H/repo; U=$HOME/NeuralShaderDemo/external/relightable-neural-assets; PY=$U/.venv/bin/python
X=$P/experiments/spt_lego; O=$P/results/spt_lego; L=$H/launcher
log() { echo "$1 $(date -Is)"; }
st() { echo "$1 $(date -Is)" > $L/chain.status; }
die() { log "STOP: $1"; st "failed: $1"; exit 1; }
mkdir -p $L; st "running"; log "chain start (spt_v1)"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ] || die "GPU not idle"
( cd $H && sha256sum -c --quiet data.sha256 ) || die "data checksum mismatch"
cd $X
run() { $PY -u spt_train.py --data $1 --state $2 --out $O/$3 --epochs 150 --pixels 131072 --seed 0 ${4:-} > $L/train_$3.log 2>&1; echo "$3 rc=$? $(date -Is)" >> $L/runs.log; }
run train_T0 T0 SPT_T0 & sleep 60
run train_T1 T1 SPT_T1 &
log "2 runs started"; wait; log "2 runs finished"; cat $L/runs.log
for r in SPT_T0 SPT_T1; do [ -f $O/$r/state.pt ] && [ -f $O/$r/train.json ] || die "$r failed"; done
st "done"; log "ALL done"
