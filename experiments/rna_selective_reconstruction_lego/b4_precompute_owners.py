"""Worklog 37: fill the per-H5 owner caches of one edit state (b4_model.EditOwner keys) once, so training and evaluation
use identical owners. Also records the dataset AABB (canonical-frame check).
Usage: python b4_precompute_owners.py <state T0|T1> <h5> ..."""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b4_model as E  # noqa: E402

own = E.make_owner(sys.argv[1])
for h5 in sys.argv[2:]:
    t0 = time.time()
    with h5py.File(h5, "r") as f:
        aabb = (np.array(f.attrs["aabb_min"], np.float64), np.array(f.attrs["aabb_max"], np.float64))
    ds = E.EditOwnedSurfaceDataset(datafile=h5, use_for_visibility="diffuse_direct", use_positions=True, device="cpu", ops_device="cuda",
                                   owner=own, canonical_aabb=aabb)
    print(json.dumps({"h5": h5, "state": sys.argv[1], "key": own.key, "slices": ds.total_slices, "owner_counts": ds.owner_counts.tolist(),
                      "max_dist": ds.owner_max_dist, "shifted_px": ds.shifted_px, "aabb": [a.tolist() for a in aabb], "seconds": time.time() - t0}), flush=True)
    del ds
