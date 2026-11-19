"""Verify archived WL105 content/addressing and capture independent frozen renderer draws."""
import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np

from common107 import BASE, PROTOCOL, VIEWS, panel, read, sha, stencil, take_stencil, write


def admit(baseline, anchor):
    raw = read(anchor)["raw_files"]
    data = baseline / "data"
    e = read(baseline / "eval/eval_manifest.json")
    d = read(data / "run_manifest.json")
    assert tuple(e["views"]) == VIEWS and e["reps"] == 8
    assert d["gates"]["G4_seed_disjoint"] and not d["smoke"]
    assert e["data_manifest_sha256"] == sha(data / "run_manifest.json")
    expected = {
        "data/run_manifest.json": e["data_manifest_sha256"],
        **{f"models/{arm}.pt": h for arm, h in e["weights_sha256"].items()},
        **{f"data/{view}/{name}": h for view, rec in d["views"].items()
           for name, h in rec["files"].items()},
        **{f"eval/{name}": h for name, h in e["files"].items()},
    }
    for filename, digest in expected.items():
        assert raw[filename] == digest == sha(baseline / filename), filename
    assert raw["eval/eval_manifest.json"] == sha(baseline / "eval/eval_manifest.json")
    cameras = [r["camera"] for r in d["views"].values()]
    assert all(cam not in cameras[:i] for i, cam in enumerate(cameras)), "camera split overlap"
    print("Archived WL105 content verified", len(expected) + 1, flush=True)
    return d, e, len(expected) + 1


def new_seed(family, view, block):
    return {"fit": 11700000, "assessment": 11710000, "covariance": 11720000}[family] + 1000 * view + block


def capture(args):
    started = time.monotonic()
    args.output.mkdir(exist_ok=True, parents=True)
    rec_path = args.output / "capture_manifest.json"
    config = {"base_commit": BASE, "capture_commit": args.commit, "protocol_sha256": sha(PROTOCOL),
              "anchor_sha256": sha(args.anchor), "gpu_budget_seconds": 15300}
    if rec_path.exists():
        rec = read(rec_path)
        assert rec["configuration"] == config
        assert all(sha(args.output / f) == r["sha256"] for f, r in rec["files"].items())
    else:
        rec = {"configuration": config, "files": {}, "seconds_recorded": 0.0}
    spent = rec["seconds_recorded"]
    dm, em, count = admit(args.baseline, args.anchor)
    rec.update({"verified_archive_files": count, "R1_R2_R3_sha256": em["weights_sha256"],
                "data_manifest_sha256": em["data_manifest_sha256"]})
    import render105 as r
    import torch
    old_by_family = r.seed_sets(r.P["n_train_views"], r.P["n_val_views"], len(VIEWS))
    old = {s for vals in old_by_family.values() for s in vals} | {
        s for val in r.P["seeds"].values() for s in (val if isinstance(val, list) else [val])}
    new = [new_seed(k, i, h) for k, n in (("fit", 8), ("assessment", 46), ("covariance", 64))
           for i in range(len(VIEWS)) for h in range(n)]
    assert len(set(new)) == len(new) and not (old & set(new))
    assert max(r.P["target_spp_eval"] // 32, 8192 // 32) < 1000003
    reader, learned, _, _, prior, history, _, scene_manifest = r.RG.F101.inputs(args.wl101)
    R, rt = reader.R, reader.rt
    bd = R.PT.fl(learned)
    rec.update({"scene": scene_manifest, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "python": platform.python_version(), "numpy": np.__version__, "mitsuba": R.mi.__version__,
                "drjit": R.dr.__version__, "seed_and_stream_disjoint": True})
    errors = []
    for row in history["matched"]:
        comp = r.frame_components(R, rt, bd, r.CAM.V0, 4096, row["seed"])
        with np.load(prior / row["path"]) as f:
            ref = np.concatenate([f[k].reshape(512, 512, 3).transpose(2, 0, 1)
                                  for k in ("e1", "e2", "hl")]).astype(np.float64)
        for c in range(3):
            sl = slice(3 * c, 3 * (c + 1))
            errors.append(float(np.abs(comp[sl] - ref[sl]).sum() / np.abs(ref[sl]).sum()))
    rec["renderer_replay_relative_l1"] = errors
    assert max(errors) <= r.P["gates"]["replay_relative_l1"]
    aux = {}
    for view in VIEWS:
        with np.load(args.baseline / "data/eval" / view / "aux.npz") as a:
            aux[view] = {k: a[k] for k in a.files}
    for family, n, spp in (("covariance", 64, 32), ("fit", 8, 8192), ("assessment", 46, 8192)):
        for v, view in enumerate(VIEWS):
            existing = sum(f"{view}/{family}_{h:02d}.npy" in rec["files"] for h in range(n))
            if existing:
                print("resume", view, family, existing, flush=True)
        for h in range(n):
            for v, view in enumerate(VIEWS):
                if spent + time.monotonic() - started >= config["gpu_budget_seconds"]:
                    raise TimeoutError("Frozen capture GPU wall budget exhausted")
                name = f"{view}/{family}_{h:02d}.npy"
                dst = args.output / name
                if name in rec["files"]:
                    continue
                seed = new_seed(family, v, h)
                cam = dm["views"][f"eval/{view}"]["camera"]
                ts = time.monotonic()
                comp = r.frame_components(R, rt, bd, cam, spp, seed)
                assert comp.shape == (9, 512, 512) and np.isfinite(comp).all() and comp.min() >= 0
                if family == "covariance":
                    idx = panel(aux[view])
                    comp = take_stencil(comp, stencil(idx, 512, 512))
                dst.parent.mkdir(exist_ok=True)
                tmp = dst.with_suffix(".part")
                with tmp.open("wb") as file:
                    np.save(file, comp, allow_pickle=False)
                os.replace(tmp, dst)
                rec["files"][name] = {"sha256": sha(dst), "seed": seed, "spp": spp}
                rec["seconds_recorded"] = spent + time.monotonic() - started
                write(rec_path, rec)
                if family != "covariance" or h % 16 == 15:
                    print(family, view, h, round(time.monotonic() - ts, 2), flush=True)
    rec["complete"] = True
    write(rec_path, rec)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("baseline", "wl101", "output", "anchor"):
        p.add_argument(arg, type=Path)
    p.add_argument("commit")
    capture(p.parse_args())
