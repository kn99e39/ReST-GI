#!/usr/bin/env bash
# Worklog 37 phase 2 on the 5080 PC, with the lab-meeting smoke test (protocol b4_smoke_meeting_v1) inserted at the earliest
# point. Overlap only inside this PC (cores 0-23 render / 24-31 training). Replaces b4_chain.sh, which was stopped while it
# was rendering heldout_T0_s1; that render is NOT interrupted (this chain waits for it and checks the file is complete).
#   CPU: [heldout_T0_s1] -> train_T0 -> owners T0
#   then CPU 0-23: heldout_T1_s1, heldout_T1_s2, heldout_T0_s2 -> train_T1_smoke (128) -> owners -> [smoke_ready] -> train_T1 -> owners
#        GPU 24-31: SO_T0 -> (smoke_ready) B_sel_smoke -> eval_smoke -> RNA_T0
#   then alone: AABB check -> A_full -> B_selective -> core evaluation -> RNA_T1 -> final evaluation
set -u
P=/mnt/c/Projects/NeuralShaderDemo; U=$P/external/relightable-neural-assets; R=$P/results/rna_selective_reconstruction_lego
X=$P/experiments/rna_selective_reconstruction_lego; PY=$U/.venv/bin/python; D=$R/data; L=$R/launcher; S=$R/smoke
export PYTHONPATH=$U
log() { echo "$1 $(date -Is)"; }
gen() { bash $X/b4_gen.sh $1 $2 2>&1 | grep -v "^Fra:\|bgl\|gvfs\|unregister\|Namespace\|Traceback\|addon_utils\|mod.unreg" | tail -2; }
so_train() { local cfg=$1 s=$2 o=$3; shift 3
  B4_STATE=$s $PY $X/b4_train.py --config $R/config/$cfg.yml --partition $P/results/rna_surface_ownership_lego/partition_L16.npz --out $o "$@" > $L/train_$(basename $o).log 2>&1; }
rna_train() { mkdir -p $R/rna_ctrl/$1; ( cd $R/rna_ctrl/$1 && $PY $P/experiments/rna_reproduction/rr_run.py $U/scripts/train.py --config $R/config/b4_lego_$1.yml --checkpoint_dir $R/rna_ctrl/$1/checkpoints > $L/train_RNA_$1.log 2>&1 ); }
ck() { ls -d $R/rna_ctrl/$1/checkpoints/b4-lego-$1/version_*/checkpoints/last.ckpt | sort | tail -1; }
complete() { $PY - "$1" <<'PYEOF'
import sys, h5py, numpy as np
with h5py.File(sys.argv[1], "r") as f:
    n = f["color"].shape[0]
    ok = all(np.abs(np.asarray(f["color"][i][::997], np.float32)).max() > 0 for i in range(n)) and np.asarray(f["diffuse_direct"][n - 1], np.float32).max() > 0
sys.exit(0 if ok else 1)
PYEOF
}
aabb_eq() { $PY - "$1" "$2" <<'PYEOF'
import sys, h5py, numpy as np
a, b = h5py.File(sys.argv[1], "r"), h5py.File(sys.argv[2], "r")
assert np.array_equal(a.attrs["aabb_min"], b.attrs["aabb_min"]) and np.array_equal(a.attrs["aabb_max"], b.attrs["aabb_max"])
print("canonical AABB equal:", sys.argv[1].split("/")[-1], sys.argv[2].split("/")[-1])
PYEOF
}
DSTAR=$($PY -c "import json; print(' '.join(str(u) for u in json.load(open('$R/oracle/oracle.json'))['D_star']))")
SETS="--set heldout_T1:T1:$D/heldout_T1_s1_val.h5:$D/heldout_T1_s2_val.h5:$D/heldout_T0_s1_val.h5:$D/heldout_T0_s2_val.h5 heldout_T0:T0:$D/heldout_T0_s1_val.h5:$D/heldout_T0_s2_val.h5"
cd $R; mkdir -p $S
log "chain v3 start; D* = $DSTAR"
while pgrep -f "generate_dataset.py --mode rna --config $R/config/heldout_T0_s1.yml" > /dev/null; do sleep 30; done
sleep 5
complete $D/heldout_T0_s1_val.h5 || { log "heldout_T0_s1 incomplete -> re-render"; rm -f $D/heldout_T0_s1_val.*; gen heldout_T0_s1 4702; }
log "heldout_T0_s1 ready"
gen train_T0 4701; log "train_T0 rendered"
$PY $X/b4_precompute_owners.py T0 $D/train_T0.h5 $D/heldout_T0_s1_val.h5 > $L/owners_T0.log 2>&1; log "owners T0 rc=$?"
rm -f $L/smoke_ready $L/smoke_done
( for n in heldout_T1_s1 heldout_T1_s2 heldout_T0_s2; do taskset -c 0-23 bash -c "$(declare -f gen); X=$X gen $n 4702"; log "$n rendered"; done
  taskset -c 0-23 bash -c "$(declare -f gen); X=$X gen train_T1_smoke 4705"; log "train_T1_smoke rendered"
  $PY $X/b4_precompute_owners.py T1 $D/train_T1_smoke.h5 $D/heldout_T1_s1_val.h5 > $L/owners_T1_smoke.log 2>&1; log "owners T1 smoke rc=$?"
  touch $L/smoke_ready
  taskset -c 0-23 bash -c "$(declare -f gen); X=$X gen train_T1 4701"; log "train_T1 rendered"
  $PY $X/b4_precompute_owners.py T1 $D/train_T1.h5 > $L/owners_T1.log 2>&1; log "owners T1 rc=$?" ) &
CPU=$!
log "SO_T0 start"; taskset -c 24-31 bash -c "$(declare -f so_train); P=$P U=$U R=$R X=$X PY=$PY L=$L PYTHONPATH=$U so_train b4_lego_T0 T0 $R/SO_T0"; log "SO_T0 rc=$?"
until [ -f $L/smoke_ready ]; do sleep 30; done
if aabb_eq $D/train_T0.h5 $D/train_T1_smoke.h5; then
  log "B_sel_smoke start"; taskset -c 24-31 bash -c "$(declare -f so_train); P=$P U=$U R=$R X=$X PY=$PY L=$L PYTHONPATH=$U so_train b4_lego_T1_smoke T1 $S/B_sel_smoke --init $R/SO_T0/state.pt --train-units $DSTAR"; log "B_sel_smoke rc=$?"
  taskset -c 24-31 $PY $X/b4_eval.py --config $R/config/b4_lego_T1.yml --oracle $R/oracle/oracle.json --out $S/eval_smoke --h 0.0045 \
    --set heldout_T1:T1:$D/heldout_T1_s1_val.h5:$D/heldout_T1_s2_val.h5:$D/heldout_T0_s1_val.h5:$D/heldout_T0_s2_val.h5 \
    --owned SO_T0=$R/SO_T0 B_sel_smoke=$S/B_sel_smoke --influence B_sel_smoke:SO_T0 --keep-views $(seq 0 15) > $L/eval_smoke.log 2>&1; log "eval smoke rc=$?"
else log "SMOKE BLOCKED: smoke T1 AABB differs from T0"; fi
touch $L/smoke_done
log "RNA_T0 start"; taskset -c 24-31 bash -c "$(declare -f rna_train); P=$P U=$U R=$R PY=$PY L=$L PYTHONPATH=$U rna_train T0"; log "RNA_T0 rc=$?"
wait $CPU
aabb_eq $D/train_T0.h5 $D/train_T1.h5 || { log "ABORT: T0/T1 training AABBs differ"; exit 3; }
log "A_full start"; so_train b4_lego_T1 T1 $R/A_full; log "A_full rc=$?"
log "B_sel start"; so_train b4_lego_T1 T1 $R/B_sel --init $R/SO_T0/state.pt --train-units $DSTAR; log "B_sel rc=$?"
$PY $X/b4_eval.py --config $R/config/b4_lego_T1.yml --oracle $R/oracle/oracle.json --out $R/eval_core --h 0.0045 $SETS \
  --owned SO_T0=$R/SO_T0 A_full=$R/A_full B_sel=$R/B_sel --influence B_sel:SO_T0 > $L/eval_core.log 2>&1; log "core eval rc=$?"
log "RNA_T1 start"; rna_train T1; log "RNA_T1 rc=$?"
$PY $X/b4_eval.py --config $R/config/b4_lego_T1.yml --oracle $R/oracle/oracle.json --out $R/eval --h 0.0045 $SETS \
  --plain RNA_T1=$(ck T1)@T1 RNA_T0=$(ck T0)@T0 --owned SO_T0=$R/SO_T0 A_full=$R/A_full B_sel=$R/B_sel \
  --influence B_sel:SO_T0 --keep-views $(seq 0 15) > $L/eval.log 2>&1; log "eval rc=$?"
log "ALL done"
