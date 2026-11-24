"""WL109B Amendment 1: cross-host reproducibility check after the owner-directed move LabServer69 -> LabServer70 (GPU; run
under guard109.py on the target host).

python xcheck109b.py CAPTURE_DIR OUT_JSON

1. The unchanged WL109 admission on the new host: sources, checkpoint, scene, learned-table hash (must equal the WL109 run
   manifest), archived 4096-spp V0 replay (renderer108.frozen_runtime).
2. For every paired unit completed on the old host, the first replicate (same seed) is re-rendered here and compared with the
   stored frame0_R0 (all 9 components) and frame0_C1H (C1 HIGH). PASS iff every relative L1 <= the WL109 structural
   tolerance (1e-6); bitwise equality is recorded.
Outcome PASS -> the completed units are kept and only the missing units are captured here; FAIL -> every paired unit is
recaptured on the new host (old-host units moved aside and kept, descriptive only).
"""
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

from common109b import P9, BIAS_SPP, EVAL_VIEWS, read, sha, write, seed_paired, HERE, W109
import runtime109 as RT


def main(cap, out):
    import torch
    from gpu109 import Engine
    ev9 = Path.home() / P9["environment"]["remote_root"] / "results/ecro_estimator109/evidence"
    table0 = read(ev9 / "run/manifest.json")["table_sha_after"]
    M = int(read(ev9 / "dev/dev.json")["selected_M"]); assert M == 1
    t0 = time.time()
    Rr, rt, Bd, audit, r105 = RT.admit()
    rec = {"host": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "gpu": torch.cuda.get_device_name(0), "versions": RT.versions(Rr), "admission": audit,
           "table_equal_wl109": audit["table_bytes_sha256"] == table0, "units": {}, "code_sha256": sha(Path(__file__))}
    assert rec["table_equal_wl109"], "table differs from WL109"
    eng = Engine(Rr, rt, Bd)
    cams = RT.cameras()
    tol = P9["synthetic"]["structural_rel_l1"]
    rl = lambda a, b: float(np.abs(a - b).sum() / max(float(np.abs(b).sum()), 1e-30))
    for d in sorted((cap / "paired").glob("*_spp*")):
        if d.name.startswith(".") or not (d / "unit.json").exists():
            continue
        u = read(d / "unit.json")
        for f, h in u["files"].items():
            if f.startswith("frame0"):
                assert sha(d / f) == h, (d, f)
        view, spp = u["view"], u["spp"]
        vi, si = EVAL_VIEWS.index(view), list(BIAS_SPP).index(spp)
        seed = seed_paired(vi, si, 0)
        assert seed == u["seed_first"], (d, seed, u["seed_first"])
        r0 = eng.r0(cams[view], spp, seed).cpu().numpy()
        c1 = eng.c1_frame(cams[view], spp, seed, M, orig_hl=True)[6:9].cpu().numpy()
        a0 = np.load(d / "frame0_R0.npy"); a1 = np.load(d / "frame0_C1H.npy")
        rec["units"][d.name] = {"r0_rel_l1": rl(r0, a0), "c1h_rel_l1": rl(c1, a1), "r0_bitwise": bool(np.array_equal(r0, a0)),
                                "c1h_bitwise": bool(np.array_equal(c1, a1))}
        print(d.name, rec["units"][d.name], flush=True)
    rec["tolerance_rel_l1"] = tol
    rec["max_rel_l1"] = max(max(v["r0_rel_l1"], v["c1h_rel_l1"]) for v in rec["units"].values())
    rec["all_bitwise"] = all(v["r0_bitwise"] and v["c1h_bitwise"] for v in rec["units"].values())
    rec["outcome"] = "PASS" if rec["max_rel_l1"] <= tol else "FAIL"
    rec["table_sha_after"] = RT.table_bytes_sha(Bd)
    assert rec["table_sha_after"] == table0
    rec["elapsed_s"] = time.time() - t0
    write(out, rec)
    print("XCHECK", rec["outcome"], rec["max_rel_l1"], "bitwise", rec["all_bitwise"], flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
