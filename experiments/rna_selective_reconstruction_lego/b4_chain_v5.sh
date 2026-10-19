#!/usr/bin/env bash
# Worklog 37: continuation after the pre-reboot pause (protocol/b4_v1_amendment2.json). Chain v4 was stopped on 2026-10-19
# 18:38 right after an RNA_T0 last.ckpt save (epoch 89) so the user could reboot; the project folder was then renamed
# NeuralShaderDemo -> ReST-GI. This chain resumes RNA_T0 from that checkpoint and then runs chain v4's remaining steps
# unchanged except for paths: configs come from config_restgi/ (config/ copies with only the folder prefix replaced).
#   GPU alone: RNA_T0 (resumed) -> AABB check -> A_full -> B_sel -> core eval -> RNA_T1 -> final eval
set -u
P=/mnt/c/Projects/ReST-GI; U=$P/external/relightable-neural-assets; R=$P/results/rna_selective_reconstruction_lego
X=$P/experiments/rna_selective_reconstruction_lego; PY=$U/.venv/bin/python; D=$R/data; L=$R/launcher; C=$R/config_restgi
export PYTHONPATH=$U
log() { echo "$1 $(date -Is)"; }
so_train() { local cfg=$1 s=$2 o=$3; shift 3
  B4_STATE=$s $PY $X/b4_train.py --config $C/$cfg.yml --partition $P/results/rna_surface_ownership_lego/partition_L16.npz --out $o "$@" > $L/train_$(basename $o).log 2>&1; }
rna_train() { local log=${2:-$L/train_RNA_$1.log}; mkdir -p $R/rna_ctrl/$1
  ( cd $R/rna_ctrl/$1 && $PY $P/experiments/rna_reproduction/rr_run.py $U/scripts/train.py --config $C/b4_lego_$1.yml --checkpoint_dir $R/rna_ctrl/$1/checkpoints > $log 2>&1 ); }
ck() { ls -d $R/rna_ctrl/$1/checkpoints/b4-lego-$1/version_*/checkpoints/last.ckpt | sort | tail -1; }
RESUME=$R/rna_ctrl/T0/checkpoints/b4-lego-T0/version_0/checkpoints/last_pre_reboot.ckpt
DSTAR=$($PY -c "import json; print(' '.join(str(u) for u in json.load(open('$R/oracle/oracle.json'))['D_star']))")
SETS="--set heldout_T1:T1:$D/heldout_T1_s1_val.h5:$D/heldout_T1_s2_val.h5:$D/heldout_T0_s1_val.h5:$D/heldout_T0_s2_val.h5 heldout_T0:T0:$D/heldout_T0_s1_val.h5:$D/heldout_T0_s2_val.h5"
cd $R
log "chain v5 start; D* = $DSTAR"
[ -f $RESUME ] || { log "ABORT: resume checkpoint missing"; exit 4; }
log "RNA_T0 resume start (epoch-89 checkpoint)"; RR_RESUME_CKPT=$RESUME rna_train T0 $L/train_RNA_T0_resumed.log; log "RNA_T0 rc=$?"
$PY - <<PYEOF || { log "ABORT: T0/T1 training AABBs differ"; exit 3; }
import h5py, numpy as np
a, b = h5py.File("$D/train_T0.h5", "r"), h5py.File("$D/train_T1.h5", "r")
assert np.array_equal(a.attrs["aabb_min"], b.attrs["aabb_min"]) and np.array_equal(a.attrs["aabb_max"], b.attrs["aabb_max"])
print("canonical AABB check: T0 == T1")
PYEOF
log "A_full start"; so_train b4_lego_T1 T1 $R/A_full; log "A_full rc=$?"
log "B_sel start"; so_train b4_lego_T1 T1 $R/B_sel --init $R/SO_T0/state.pt --train-units $DSTAR; log "B_sel rc=$?"
$PY $X/b4_eval.py --config $C/b4_lego_T1.yml --oracle $R/oracle/oracle.json --out $R/eval_core --h 0.0045 $SETS \
  --owned SO_T0=$R/SO_T0 A_full=$R/A_full B_sel=$R/B_sel --influence B_sel:SO_T0 > $L/eval_core.log 2>&1; log "core eval rc=$?"
log "RNA_T1 start"; rna_train T1; log "RNA_T1 rc=$?"
$PY $X/b4_eval.py --config $C/b4_lego_T1.yml --oracle $R/oracle/oracle.json --out $R/eval --h 0.0045 $SETS \
  --plain RNA_T1=$(ck T1)@T1 RNA_T0=$(ck T0)@T0 --owned SO_T0=$R/SO_T0 A_full=$R/A_full B_sel=$R/B_sel \
  --influence B_sel:SO_T0 --keep-views $(seq 0 15) > $L/eval.log 2>&1; log "eval rc=$?"
log "ALL done"
