#!/usr/bin/env bash
# Worklog 37 on LabServer63 (user request 2026-10-18: run on the server until the 5080 PC is released).
# Automates the predeclared protocol b4_v1: pilot gate -> references -> training data -> arms -> evaluation.
#   1 wait for the oracle renders; b4_oracle.py; stop if any pilot criterion P1-P5 fails (status pilot_failed)
#   2 held-out references (resumed by the oracle script) -> train_T0 render -> owners T0
#   3 train_T1 render + owners T1  ||  GPU: SO_T0, RNA_T0
#   4 canonical AABB check; GPU alone: A_full, B_selective (their wall-clock is the reported cost); RNA_T1; evaluation
set -u
H=$HOME/nsd_b4; P=$H/repo; U=$HOME/NeuralShaderDemo/external/relightable-neural-assets; R=$P/results/rna_selective_reconstruction_lego
X=$P/experiments/rna_selective_reconstruction_lego; PY=$U/.venv/bin/python; D=$H/data; L=$H/launcher
export PYTHONPATH=$U
log() { echo "$1 $(date -Is)"; }
st() { echo "$1 $(date -Is)" > $L/chain.status; }
so_train() { local s=$1 o=$2; shift 2
  B4_STATE=$s $PY $X/b4_train.py --config $H/config/b4_lego_$s.yml --partition $P/results/rna_surface_ownership_lego/partition_L16.npz --out $o "$@" > $L/train_$(basename $o).log 2>&1; }
rna_train() { mkdir -p $R/rna_ctrl/$1; ( cd $R/rna_ctrl/$1 && $PY $H/rr_run.py $U/scripts/train.py --config $H/config/b4_lego_$1.yml --checkpoint_dir $R/rna_ctrl/$1/checkpoints > $L/train_RNA_$1.log 2>&1 ); }
gen() { bash $H/b4_gen_server.sh $1 $2 2>&1 | grep -v "^Fra:\|bgl\|gvfs" | tail -2; }
log "chain start"
until [ -f $L/oracle.status ]; do sleep 60; done
$PY $X/b4_oracle.py --data $D --out $R/oracle > $L/oracle_analysis.log 2>&1; log "oracle analysis rc=$?"
PASS=$($PY -c "import json; print(json.load(open('$R/oracle/oracle.json'))['pilot_b4_v1_1']['all_pass'])")
log "pilot (b4_v1.1, post-hoc P4 amendment) all_pass=$PASS"
if [ "$PASS" != "True" ]; then st "pilot_failed"; exit 0; fi
until [ -f $L/heldout.status ]; do sleep 60; done; log "heldout done"
gen train_T0 4701
$PY $X/b4_precompute_owners.py T0 $D/train_T0.h5 $D/heldout_T0_s1_val.h5 > $L/owners_T0.log 2>&1; log "owners T0 rc=$?"
( gen train_T1 4701; $PY $X/b4_precompute_owners.py T1 $D/train_T1.h5 $D/heldout_T1_s1_val.h5 > $L/owners_T1.log 2>&1; log "owners T1 rc=$?" ) &
CPU=$!
log "SO_T0 start"; so_train T0 $R/SO_T0; log "SO_T0 rc=$?"
log "RNA_T0 start"; rna_train T0; log "RNA_T0 rc=$?"
wait $CPU
$PY - <<PYEOF || { log "ABORT: T0/T1 training AABBs differ"; st "aabb_mismatch"; exit 3; }
import h5py, numpy as np
a, b = h5py.File("$D/train_T0.h5", "r"), h5py.File("$D/train_T1.h5", "r")
assert np.array_equal(a.attrs["aabb_min"], b.attrs["aabb_min"]) and np.array_equal(a.attrs["aabb_max"], b.attrs["aabb_max"])
print("canonical AABB check: T0 == T1", a.attrs["aabb_min"], a.attrs["aabb_max"])
PYEOF
DSTAR=$($PY -c "import json; print(' '.join(str(u) for u in json.load(open('$R/oracle/oracle.json'))['D_star']))")
log "D* = $DSTAR"
log "A_full start"; so_train T1 $R/A_full; log "A_full rc=$?"
log "B_sel start"; so_train T1 $R/B_sel --init $R/SO_T0/state.pt --train-units $DSTAR; log "B_sel rc=$?"
log "RNA_T1 start"; rna_train T1; log "RNA_T1 rc=$?"
ck() { ls -d $R/rna_ctrl/$1/checkpoints/b4-lego-$1/version_*/checkpoints/last.ckpt | sort | tail -1; }
$PY $X/b4_eval.py --config $H/config/b4_lego_T1.yml --oracle $R/oracle/oracle.json --out $R/eval --h 0.0045 \
  --set heldout_T1:T1:$D/heldout_T1_s1_val.h5:$D/heldout_T1_s2_val.h5:$D/heldout_T0_s1_val.h5:$D/heldout_T0_s2_val.h5 \
        heldout_T0:T0:$D/heldout_T0_s1_val.h5:$D/heldout_T0_s2_val.h5 \
  --plain RNA_T1=$(ck T1)@T1 RNA_T0=$(ck T0)@T0 --owned SO_T0=$R/SO_T0 A_full=$R/A_full B_sel=$R/B_sel \
  --influence B_sel:SO_T0 --keep-views $(seq 0 15) > $L/eval.log 2>&1; log "eval rc=$?"
st "done"; log "ALL done"
