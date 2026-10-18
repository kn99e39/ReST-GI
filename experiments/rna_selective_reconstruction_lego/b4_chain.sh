#!/usr/bin/env bash
# Worklog 37 phase 2 on the 5080 PC, strictly sequential (user 2026-10-18: back to this PC; server GPU reserved for another
# project; no parallel jobs). The oracle D* (b4_v1.1) was computed on LabServer63 (all six oracle sets rendered there).
# Held-out references are rendered here (all four sets on one machine: a 0.0018 RMS PC-vs-server rendering difference
# was measured, oracle/machine_bias.json, so references of one pair never mix machines).
#   held-out refs -> train_T0 -> owners T0 -> train_T1 -> owners T1 -> AABB check
#   -> SO_T0 -> A_full -> B_selective -> core evaluation (owned arms) -> RNA_T0 -> RNA_T1 -> final evaluation
set -u
P=/mnt/c/Projects/NeuralShaderDemo; U=$P/external/relightable-neural-assets; R=$P/results/rna_selective_reconstruction_lego
X=$P/experiments/rna_selective_reconstruction_lego; PY=$U/.venv/bin/python; D=$R/data; L=$R/launcher
export PYTHONPATH=$U
log() { echo "$1 $(date -Is)"; }
gen() { bash $X/b4_gen.sh $1 $2 2>&1 | grep -v "^Fra:\|bgl\|gvfs\|unregister\|Namespace\|Traceback\|addon_utils\|mod.unreg" | tail -2; }
so_train() { local s=$1 o=$2; shift 2
  B4_STATE=$s $PY $X/b4_train.py --config $R/config/b4_lego_$s.yml --partition $P/results/rna_surface_ownership_lego/partition_L16.npz --out $o "$@" > $L/train_$(basename $o).log 2>&1; }
rna_train() { mkdir -p $R/rna_ctrl/$1; ( cd $R/rna_ctrl/$1 && $PY $P/experiments/rna_reproduction/rr_run.py $U/scripts/train.py --config $R/config/b4_lego_$1.yml --checkpoint_dir $R/rna_ctrl/$1/checkpoints > $L/train_RNA_$1.log 2>&1 ); }
ck() { ls -d $R/rna_ctrl/$1/checkpoints/b4-lego-$1/version_*/checkpoints/last.ckpt | sort | tail -1; }
SETS="--set heldout_T1:T1:$D/heldout_T1_s1_val.h5:$D/heldout_T1_s2_val.h5:$D/heldout_T0_s1_val.h5:$D/heldout_T0_s2_val.h5 heldout_T0:T0:$D/heldout_T0_s1_val.h5:$D/heldout_T0_s2_val.h5"
cd $R
log "chain start"
for v in T0 T1; do for s in 1 2; do [ -f $D/heldout_${v}_s${s}_val.h5 ] || gen heldout_${v}_s$s 4702; done; done; log "heldout done"
gen train_T0 4701; log "train_T0 rendered"
$PY $X/b4_precompute_owners.py T0 $D/train_T0.h5 $D/heldout_T0_s1_val.h5 > $L/owners_T0.log 2>&1; log "owners T0 rc=$?"
gen train_T1 4701; log "train_T1 rendered"
$PY $X/b4_precompute_owners.py T1 $D/train_T1.h5 $D/heldout_T1_s1_val.h5 > $L/owners_T1.log 2>&1; log "owners T1 rc=$?"
$PY - <<PYEOF || { log "ABORT: T0/T1 training AABBs differ"; exit 3; }
import h5py, numpy as np
a, b = h5py.File("$D/train_T0.h5", "r"), h5py.File("$D/train_T1.h5", "r")
assert np.array_equal(a.attrs["aabb_min"], b.attrs["aabb_min"]) and np.array_equal(a.attrs["aabb_max"], b.attrs["aabb_max"])
print("canonical AABB check: T0 == T1", a.attrs["aabb_min"], a.attrs["aabb_max"])
PYEOF
DSTAR=$($PY -c "import json; print(' '.join(str(u) for u in json.load(open('$R/oracle/oracle.json'))['D_star']))")
log "D* = $DSTAR"
log "SO_T0 start"; so_train T0 $R/SO_T0; log "SO_T0 rc=$?"
log "A_full start"; so_train T1 $R/A_full; log "A_full rc=$?"
log "B_sel start"; so_train T1 $R/B_sel --init $R/SO_T0/state.pt --train-units $DSTAR; log "B_sel rc=$?"
$PY $X/b4_eval.py --config $R/config/b4_lego_T1.yml --oracle $R/oracle/oracle.json --out $R/eval_core --h 0.0045 $SETS \
  --owned SO_T0=$R/SO_T0 A_full=$R/A_full B_sel=$R/B_sel --influence B_sel:SO_T0 > $L/eval_core.log 2>&1; log "core eval rc=$?"
log "RNA_T0 start"; rna_train T0; log "RNA_T0 rc=$?"
log "RNA_T1 start"; rna_train T1; log "RNA_T1 rc=$?"
$PY $X/b4_eval.py --config $R/config/b4_lego_T1.yml --oracle $R/oracle/oracle.json --out $R/eval --h 0.0045 $SETS \
  --plain RNA_T1=$(ck T1)@T1 RNA_T0=$(ck T0)@T0 --owned SO_T0=$R/SO_T0 A_full=$R/A_full B_sel=$R/B_sel \
  --influence B_sel:SO_T0 --keep-views $(seq 0 15) > $L/eval.log 2>&1; log "eval rc=$?"
log "ALL done"
