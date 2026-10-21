#!/usr/bin/env bash
# Worklog 38 on LabServer63 (protocol/b4_seedrep_v1.json): seeds 1 and 2 of A_full and B_sel, started concurrently on the
# one GPU (quality only; wall-clock is not a cost measurement here), then one evaluation of seeds 0-2 on held-out T1.
# Layout: $H/repo = git archive of the commit that declared the protocol; PC-rendered data under $R/data (sha256-checked).
# Launch: setsid -f nohup bash b4_seedrep_server.sh > $H/launcher/chain.stdout.log 2>&1 < /dev/null
set -u
H=$HOME/rest_b4rep; P=$H/repo; U=$HOME/NeuralShaderDemo/external/relightable-neural-assets; R=$P/results/rna_selective_reconstruction_lego
X=$P/experiments/rna_selective_reconstruction_lego; PY=$U/.venv/bin/python; D=$R/data; L=$H/launcher; C=$H/config
export PYTHONPATH=$U
log() { echo "$1 $(date -Is)"; }
st() { echo "$1 $(date -Is)" > $L/chain.status; }
die() { log "STOP: $1"; st "failed: $1"; exit 1; }
so_ok() { [ -f $1/state.pt ] && [ -f $1/train.json ]; }
so_train() { local s=$1 o=$2; shift 2
  B4_STATE=T1 $PY $X/b4_train.py --config $C/b4_lego_T1.yml --partition $P/results/rna_surface_ownership_lego/partition_L16.npz --out $o --seed $s "$@" > $L/train_$(basename $o).log 2>&1
  echo "$(basename $o) rc=$? $(date -Is)" >> $L/runs.log; }
mkdir -p $L; st "running"
log "chain start (b4_seedrep_v1)"
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ] || die "GPU not idle"
( cd $H && sha256sum -c --quiet data.sha256 ) || die "data checksum mismatch"
DSTAR=$($PY -c "import json; print(' '.join(str(u) for u in json.load(open('$R/oracle/oracle.json'))['D_star']))")
[ "$DSTAR" = "0 2 3 6 7 8 13 16" ] || die "unexpected D* $DSTAR"
for o in A_full_s1 A_full_s2 B_sel_s1 B_sel_s2; do [ -e $R/$o ] && die "$o already exists"; done
so_train 1 $R/A_full_s1 & sleep 120
so_train 2 $R/A_full_s2 & sleep 120
so_train 1 $R/B_sel_s1 --init $R/SO_T0/state.pt --train-units $DSTAR & sleep 120
so_train 2 $R/B_sel_s2 --init $R/SO_T0/state.pt --train-units $DSTAR &
log "4 runs started"; wait; log "4 runs finished"; cat $L/runs.log
for o in A_full_s1 A_full_s2 B_sel_s1 B_sel_s2; do so_ok $R/$o || die "$o failed"; done
$PY $X/b4_eval.py --config $C/b4_lego_T1.yml --oracle $R/oracle/oracle.json --out $R/eval_seedrep --h 0.0045 \
  --set heldout_T1:T1:$D/heldout_T1_s1_val.h5:$D/heldout_T1_s2_val.h5:$D/heldout_T0_s1_val.h5:$D/heldout_T0_s2_val.h5 \
  --owned SO_T0=$R/SO_T0 A_full=$R/A_full B_sel=$R/B_sel A_full_s1=$R/A_full_s1 B_sel_s1=$R/B_sel_s1 A_full_s2=$R/A_full_s2 B_sel_s2=$R/B_sel_s2 \
  --influence B_sel:SO_T0 B_sel_s1:SO_T0 B_sel_s2:SO_T0 > $L/eval.log 2>&1 || die "evaluation failed"
log "eval done"; st "done"; log "ALL done"
