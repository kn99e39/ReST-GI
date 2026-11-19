"""Audit the immutable local WL105 and WL106 reviewer and input mirrors."""
import argparse
from pathlib import Path

import numpy as np

from common107 import SPPS, VIEWS, read, sha, total, write


def audit(baseline, eval105, eval106):
    anchors = read(eval105 / "raw_manifest105.json")
    raw = anchors["raw_files"]
    man = baseline / "data/run_manifest.json"
    assert sha(man) == raw["data/run_manifest.json"]
    payload = read(man)
    cams = [r["camera"] for r in payload["views"].values()]
    assert all(cam not in cams[:i] for i, cam in enumerate(cams))
    count, seeds, missing = 0, [], []
    for view in VIEWS:
        item = f"eval/{view}"
        for filename, digest in payload["views"][item]["files"].items():
            sub = f"data/{item}/{filename}"
            path = baseline / sub
            assert sha(path) == digest == raw[sub]
            if filename.startswith("noisy"):
                spp, rep = map(int, filename.removeprefix("noisy_spp").removesuffix(".npz").split("_r"))
                seed = 11200000 + 1000 * VIEWS.index(view) + 100 * SPPS.index(spp) + rep
                with np.load(path) as z:
                    assert int(z["seed"]) == seed
                    x = z["comp"]
                    y = total(x)
                    assert x.shape == (9, 512, 512) and np.isfinite(y).all() and y.min() >= 0
                seeds.append(seed)
            count += 1
    assert len(seeds) == len(set(seeds)) == 144
    immutable = read(eval106 / "manifest106.json")
    assert all(sha(eval106 / f) == h for f, h in immutable["files"].items())
    sg = read(eval106 / "synthetic106.json")["gate"]
    assert not sg["SG2_guide_consistent_exact"] and not sg["pass"]
    for required in ("models/R1.pt", "models/R2.pt", "models/R3.pt", "eval/eval_manifest.json"):
        if not (baseline / required).exists():
            missing.append(required)
    return {"all_heldout_inputs_hash_verified": count, "heldout_noisy_seed_count": len(seeds),
            "view_count": len(cams), "duplicate_camera_count": 0,
            "historical_WL106_reviewer_files_hash_verified": len(immutable["files"]),
            "historical_WL106_gates": sg, "local_replay_dependencies_missing": missing,
            "WL105_data_manifest_sha256": sha(man), "WL105_anchor_sha256": sha(eval105 / "raw_manifest105.json"),
            "WL106_manifest_sha256": sha(eval106 / "manifest106.json")}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("baseline", "eval105", "eval106", "output"):
        p.add_argument(arg, type=Path)
    a = p.parse_args()
    result = audit(a.baseline, a.eval105, a.eval106)
    write(a.output, result)
    print(result, flush=True)
