#!/usr/bin/env bash
# Worklog 37: continuation after the smoke amendment (b4_smoke_meeting_v1_amendment1). Chain v3 was stopped (RNA_T0 aborted at
# ~20 min and re-run here); its train_T1 render + owners-T1 subshell keeps running and is awaited via its log line.
#   GPU 24-31: A_full_smoke -> eval_smoke2 -> RNA_T0 ; then (render done) alone: A_full -> B_sel -> core eval -> RNA_T1 -> final eval
set -u
P=/mnt/c/Projects/NeuralShaderDemo; U=$P/external/relightable-neural-assets; R=$P/results/rna_selective_reconstruction_lego
X=$P/experiments/rna_selective_reconstruction_lego; PY=$U/.venv/bin/python; D=$R/data; L=$R/launcher; S=$R/smoke
export PYTHONPATH=$U
log() { echo "$1 $(date -Is)"; }
so_train() { local cfg=$1 s=$2 o=$3; shift 3
  B4_STATE=$s $PY $X/b4_train.py --config $R/config/$cfg.yml --partition $P/results/rna_surface_ownership_lego/partition_L16.npz --out $o "$@" > $L/train_$(basename $o).log 2>&1; }
rna_train() { mkdir -p $R/rna_ctrl/$1; ( cd $R/rna_ctrl/$1 && $PY $P/experiments/rna_reproduction/rr_run.py $U/scripts/train.py --config $R/config/b4_lego_$1.yml --checkpoint_dir $R/rna_ctrl/$1/checkpoints > $L/train_RNA_$1.log 2>&1 ); }
ck() { ls -d $R/rna_ctrl/$1/checkpoints/b4-lego-$1/version_*/checkpoints/last.ckpt | sort | tail -1; }
DSTAR=$($PY -c "import json; print(' '.join(str(u) for u in json.load(open('$R/oracle/oracle.json'))['D_star']))")
SETS="--set heldout_T1:T1:$D/heldout_T1_s1_val.h5:$D/heldout_T1_s2_val.h5:$D/heldout_T0_s1_val.h5:$D/heldout_T0_s2_val.h5 heldout_T0:T0:$D/heldout_T0_s1_val.h5:$D/heldout_T0_s2_val.h5"
cd $R
log "chain v4 start; D* = $DSTAR"
log "A_full_smoke start"; taskset -c 24-31 bash -c "$(declare -f so_train); P=$P U=$U R=$R X=$X PY=$PY L=$L PYTHONPATH=$U so_train b4_lego_T1_smoke T1 $S/A_full_smoke"; log "A_full_smoke rc=$?"
taskset -c 24-31 $PY $X/b4_eval.py --config $R/config/b4_lego_T1.yml --oracle $R/oracle/oracle.json --out $S/eval_smoke2 --h 0.0045 \
  --set heldout_T1:T1:$D/heldout_T1_s1_val.h5:$D/heldout_T1_s2_val.h5:$D/heldout_T0_s1_val.h5:$D/heldout_T0_s2_val.h5 \
  --owned SO_T0=$R/SO_T0 B_sel_smoke=$S/B_sel_smoke A_full_smoke=$S/A_full_smoke --influence B_sel_smoke:SO_T0 --keep-views $(seq 0 15) > $L/eval_smoke2.log 2>&1
log "eval smoke2 rc=$?"; touch $L/smoke2_done
log "RNA_T0 start"; taskset -c 24-31 bash -c "$(declare -f rna_train); P=$P U=$U R=$R PY=$PY L=$L PYTHONPATH=$U rna_train T0"; log "RNA_T0 rc=$?"
until grep -q "owners T1 rc=" $L/b4_chain_v3.stdout.log; do sleep 60; done; log "T1 data + owners ready ($(grep 'owners T1 rc=' $L/b4_chain_v3.stdout.log))"
$PY - <<PYEOF || { log "ABORT: T0/T1 training AABBs differ"; exit 3; }
import h5py, numpy as np
a, b = h5py.File("$D/train_T0.h5", "r"), h5py.File("$D/train_T1.h5", "r")
assert np.array_equal(a.attrs["aabb_min"], b.attrs["aabb_min"]) and np.array_equal(a.attrs["aabb_max"], b.attrs["aabb_max"])
print("canonical AABB check: T0 == T1")
PYEOF
log "A_full start"; so_train b4_lego_T1 T1 $R/A_full; log "A_full rc=$?"
log "B_sel start"; so_train b4_lego_T1 T1 $R/B_sel --init $R/SO_T0/state.pt --train-units $DSTAR; log "B_sel rc=$?"
$PY $X/b4_eval.py --config $R/config/b4_lego_T1.yml --oracle $R/oracle/oracle.json --out $R/eval_core --h 0.0045 $SETS \
  --owned SO_T0=$R/SO_T0 A_full=$R/A_full B_sel=$R/B_sel --influence B_sel:SO_T0 > $L/eval_core.log 2>&1; log "core eval rc=$?"
log "RNA_T1 start"; rna_train T1; log "RNA_T1 rc=$?"
$PY $X/b4_eval.py --config $R/config/b4_lego_T1.yml --oracle $R/oracle/oracle.json --out $R/eval --h 0.0045 $SETS \
  --plain RNA_T1=$(ck T1)@T1 RNA_T0=$(ck T0)@T0 --owned SO_T0=$R/SO_T0 A_full=$R/A_full B_sel=$R/B_sel \
  --influence B_sel:SO_T0 --keep-views $(seq 0 15) > $L/eval.log 2>&1; log "eval rc=$?"
log "ALL done"
