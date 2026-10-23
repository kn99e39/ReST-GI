#!/usr/bin/env bash
# Worklog 41 on LabServer63 (protocol fnto_v1): Stage A training of the FNTO kernels (fnto_train.py, declared settings).
# Evaluation runs on the PC afterwards (Mitsuba relations, held-out references and region records live there).
# Layout: $H/repo = git archive of the declaring commit + results/fnto_lego inputs (sha256-checked, data.sha256).
# Launch: setsid -f nohup bash fnto_server.sh > $H/launcher/chain.stdout.log 2>&1 < /dev/null
set -u
H=$HOME/rest_fnto; P=$H/repo; U=$HOME/NeuralShaderDemo/external/relightable-neural-assets; PY=$U/.venv/bin/python
X=$P/experiments/fnto_lego; O=$P/results/fnto_lego; L=$H/launcher
log() { echo "$1 $(date -Is)"; }
st() { echo "$1 $(date -Is)" > $L/chain.status; }
die() { log "STOP: $1"; st "failed: $1"; exit 1; }
mkdir -p $L; st "running"; log "chain start (fnto_v1)"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ] || die "GPU not idle"
( cd $H && sha256sum -c --quiet data.sha256 ) || die "data checksum mismatch"
cd $X
$PY -u fnto_train.py --out $O/K_main --steps 24000 --imgs 4 --px 16384 --seed 0 --eval-every 4000 > $L/train_K_main.log 2>&1
echo "K_main rc=$? $(date -Is)" >> $L/runs.log
[ -f $O/K_main/kernels.pt ] && [ -f $O/K_main/train.json ] || die "K_main failed"
st "done"; log "ALL done"
