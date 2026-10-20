#!/usr/bin/env bash
# Worklog 37: continuation after the user-requested pause of 2026-10-20 18:54 (protocol/b4_v1_amendment4.json).
# Chain v6 completed RNA_T0, A_full, B_sel and the core evaluation; RNA_T1 was stopped right after its epoch-33 save
# (last_pre_pause.ckpt) and v6 stopped before the final evaluation. This chain resumes RNA_T1 from that checkpoint and
# runs v6's final evaluation command unchanged, stopping on any failed step.
#   GPU alone: RNA_T1 (resumed) -> final eval
set -u
P=/mnt/c/Projects/ReST-GI; U=$P/external/relightable-neural-assets; R=$P/results/rna_selective_reconstruction_lego
X=$P/experiments/rna_selective_reconstruction_lego; PY=$U/.venv/bin/python; D=$R/data; L=$R/launcher; C=$R/config_restgi
export PYTHONPATH=$U
log() { echo "$1 $(date -Is)"; }
die() { log "STOP: $1"; exit 1; }
rna_train() { local log=${2:-$L/train_RNA_$1.log}; mkdir -p $R/rna_ctrl/$1
  ( cd $R/rna_ctrl/$1 && $PY $P/experiments/rna_reproduction/rr_run.py $U/scripts/train.py --config $C/b4_lego_$1.yml --checkpoint_dir $R/rna_ctrl/$1/checkpoints > $log 2>&1 ); }
ck() { ls -d $R/rna_ctrl/$1/checkpoints/b4-lego-$1/version_*/checkpoints/last.ckpt | sort | tail -1; }
rna_ok() { [ $2 -eq 0 ] && grep -q "max_epochs=250. reached" $1; }
RESUME=$R/rna_ctrl/T1/checkpoints/b4-lego-T1/version_0/checkpoints/last_pre_pause.ckpt
SETS="--set heldout_T1:T1:$D/heldout_T1_s1_val.h5:$D/heldout_T1_s2_val.h5:$D/heldout_T0_s1_val.h5:$D/heldout_T0_s2_val.h5 heldout_T0:T0:$D/heldout_T0_s1_val.h5:$D/heldout_T0_s2_val.h5"
cd $R
log "chain v7 start"
[ -f $RESUME ] || die "resume checkpoint missing"
[ -e $R/eval ] && die "eval already exists"
grep -q "max_epochs=250. reached" $L/train_RNA_T0_resumed2.log || die "RNA_T0 is not complete"
for d in SO_T0 A_full B_sel; do [ -f $R/$d/state.pt ] || die "$d/state.pt missing"; done
log "RNA_T1 resume start (epoch-33 checkpoint)"; RR_RESUME_CKPT=$RESUME rna_train T1 $L/train_RNA_T1_resumed.log; rc=$?
log "RNA_T1 rc=$rc"; rna_ok $L/train_RNA_T1_resumed.log $rc || die "RNA_T1 did not finish 250 epochs"
log "final eval start (RNA_T1=$(ck T1), RNA_T0=$(ck T0))"
$PY $X/b4_eval.py --config $C/b4_lego_T1.yml --oracle $R/oracle/oracle.json --out $R/eval --h 0.0045 $SETS \
  --plain RNA_T1=$(ck T1)@T1 RNA_T0=$(ck T0)@T0 --owned SO_T0=$R/SO_T0 A_full=$R/A_full B_sel=$R/B_sel \
  --influence B_sel:SO_T0 --keep-views $(seq 0 15) > $L/eval.log 2>&1; rc=$?; log "eval rc=$rc"; [ $rc -eq 0 ] || die "final eval failed"
log "ALL done"
