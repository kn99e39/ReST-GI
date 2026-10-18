#!/usr/bin/env bash
# Worklog 37 phase 2 (after a passing GT-only pilot and the oracle D*): references, training data, all arms, evaluation.
#   LabServer63 (user request 2026-10-18): held-out refs (T0, T1 x Cycles seeds 1, 2), fetched by b4_fetch_server.sh
#   CPU: train_T0 -> owners T0 -> train_T1 (cores 0-23) -> owners T1
#   GPU: SO_T0, RNA_T0 (cores 24-31, overlapping the train_T1 render) -> [T1 data ready] -> A_full -> B_selective -> RNA_T1 -> eval
# A_full and B_selective run alone on the machine (their wall-clock is the reported reconstruction cost).
set -u
P=/mnt/c/Projects/NeuralShaderDemo; U=$P/external/relightable-neural-assets; R=$P/results/rna_selective_reconstruction_lego
X=$P/experiments/rna_selective_reconstruction_lego; PY=$U/.venv/bin/python; D=$R/data; L=$R/launcher
export PYTHONPATH=$U
log() { echo "$1 $(date -Is)"; }
so_train() { # <state> <out> [rso_train args]
  local s=$1 o=$2; shift 2
  B4_STATE=$s $PY $X/b4_train.py --config $R/config/b4_lego_$s.yml --partition $P/results/rna_surface_ownership_lego/partition_L16.npz --out $o "$@" > $L/train_$(basename $o).log 2>&1
}
rna_train() { # <state>
  mkdir -p $R/rna_ctrl/$1; cd $R/rna_ctrl/$1
  $PY $P/experiments/rna_reproduction/rr_run.py $U/scripts/train.py --config $R/config/b4_lego_$1.yml --checkpoint_dir $R/rna_ctrl/$1/checkpoints > $L/train_RNA_$1.log 2>&1
  cd $R
}
cd $R
log "chain start"
waitf() { until [ -f $L/fetched_$1 ]; do sleep 60; done; log "have $1"; }
bash $X/b4_gen.sh train_T0 4701 2>&1 | grep -v "^Fra:\|bgl\|gvfs" | tail -2
waitf heldout_T0_s1
$PY $X/b4_precompute_owners.py T0 $D/train_T0.h5 $D/heldout_T0_s1_val.h5 > $L/owners_T0.log 2>&1; log "owners T0 rc=$?"
rm -f $L/T1_ready
( taskset -c 0-23 bash $X/b4_gen.sh train_T1 4701 2>&1 | grep -v "^Fra:\|bgl\|gvfs" | tail -2
  waitf heldout_T1_s1
  $PY $X/b4_precompute_owners.py T1 $D/train_T1.h5 $D/heldout_T1_s1_val.h5 > $L/owners_T1.log 2>&1; log "owners T1 rc=$?"
  touch $L/T1_ready ) &
CPU=$!
log "SO_T0 start"; taskset -c 24-31 bash -c "$(declare -f so_train); P=$P U=$U R=$R X=$X PY=$PY L=$L PYTHONPATH=$U so_train T0 $R/SO_T0"; log "SO_T0 rc=$?"
log "RNA_T0 start"; taskset -c 24-31 bash -c "$(declare -f rna_train); P=$P U=$U R=$R PY=$PY L=$L PYTHONPATH=$U rna_train T0"; log "RNA_T0 rc=$?"
wait $CPU
$PY - <<EOF || { log "ABORT: T0/T1 training AABBs differ"; exit 3; }
import h5py, numpy as np
a, b = h5py.File("$D/train_T0.h5", "r"), h5py.File("$D/train_T1.h5", "r")
assert np.array_equal(a.attrs["aabb_min"], b.attrs["aabb_min"]) and np.array_equal(a.attrs["aabb_max"], b.attrs["aabb_max"])
print("canonical AABB check: T0 == T1", a.attrs["aabb_min"], a.attrs["aabb_max"])
EOF
DSTAR=$($PY -c "import json; print(' '.join(str(u) for u in json.load(open('$R/oracle/oracle.json'))['D_star']))")
log "D* = $DSTAR"
log "A_full start"; so_train T1 $R/A_full; log "A_full rc=$?"
log "B_sel start"; so_train T1 $R/B_sel --init $R/SO_T0/state.pt --train-units $DSTAR; log "B_sel rc=$?"
log "RNA_T1 start"; rna_train T1; log "RNA_T1 rc=$?"
for n in heldout_T0_s2 heldout_T1_s2; do waitf $n; done
ck() { ls -d $R/rna_ctrl/$1/checkpoints/b4-lego-$1/version_*/checkpoints/last.ckpt | sort | tail -1; }
$PY $X/b4_eval.py --config $R/config/b4_lego_T1.yml --oracle $R/oracle/oracle.json --out $R/eval --h 0.0045 \
  --set heldout_T1:T1:$D/heldout_T1_s1_val.h5:$D/heldout_T1_s2_val.h5:$D/heldout_T0_s1_val.h5:$D/heldout_T0_s2_val.h5 \
        heldout_T0:T0:$D/heldout_T0_s1_val.h5:$D/heldout_T0_s2_val.h5 \
  --plain RNA_T1=$(ck T1)@T1 RNA_T0=$(ck T0)@T0 --owned SO_T0=$R/SO_T0 A_full=$R/A_full B_sel=$R/B_sel \
  --influence B_sel:SO_T0 --keep-views $(seq 0 15) > $L/eval.log 2>&1; log "eval rc=$?"
log "ALL done"
