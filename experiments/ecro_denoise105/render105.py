"""WL105 data generation on the frozen WL101 ECRO renderer (LabServer63 WL101 stack). No reconstruction here.

python render105.py WL101_ROOT OUT [--smoke]

Gates: G0 WL101F input validation (scene, checkpoint, learned table, renderer sources, stack);
G1 frozen-renderer replay of the archived 4096-spp V0 ECRO predictions (seeds 101100 / 101101);
G2 LOW + HIGH = TOTAL; G3 same-seed determinism; G4 seed-set disjointness.
Outputs per view: aux.npz (8x8-subpixel shape coverage, mean normal, mean depth, centre-ray id/position),
target components (E1, E2, HL means) and noisy component means at 4 / 16 / 32 spp.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_sss101g"))
sys.path.insert(0, str(HERE))
argv = sys.argv[:]
sys.argv = sys.argv[:1]
import run_generalization as RG  # noqa: E402  (WL101F inputs + WL101A-F paths)
from run_oracle import A  # noqa: E402
import cameras105 as CAM  # noqa: E402

PROTOCOL = HERE / "protocol/denoise105.json"
P = json.loads(PROTOCOL.read_text())
SMOKE = "--smoke" in argv
if SMOKE:   # never consume evidence seeds in a code-path smoke run
    P["seeds"] = {k: (v if k == "replay" else v + 5000000) for k, v in P["seeds"].items()}
RES = 512
SPPS = P["spp"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def frame_components(R, rt, Bd, cam, spp, seed, b=32):
    """Per-pixel component MEANS [9, H, W] (e1 rgb, e2 rgb, hl rgb) via the original rt._frame_prog.
    Launch k uses seed * 1000003 + k with b samples (= render_hq); spp < 32 is one launch of spp samples."""
    dr, mi = R.dr, R.mi
    import torch
    cam = dict(cam, res=RES)
    b = min(b, spp)
    acc = None
    for k in range(spp // b):
        out = rt._frame_prog(rt.scene, rt.tabs(), rt.light_args(), Bd, dr.opaque(mi.UInt64, seed * 1000003 + k), (cam, RES, b))
        dr.eval(out)
        t = torch.stack([o.torch().double() for o in out])
        acc = t if acc is None else acc + t
    return (acc / spp).cpu().numpy().reshape(9, RES, RES).astype(np.float32)


def aux_buffers(R, rt, cam, sub=8):
    """Geometry-only guides averaged over a fixed sub x sub grid: per-shape coverage (9), mean normal (3), mean depth."""
    dr, mi, PT = R.dr, R.mi, R.PT
    F, U, I32 = mi.Float, mi.UInt32, mi.Int32
    cam = dict(cam, res=RES)
    o, fwd, right, up, s = PT.cam_basis(cam)
    cover = np.zeros((9, RES, RES), np.float32)
    nrm = np.zeros((3, RES, RES), np.float32)
    dep = np.zeros((RES, RES), np.float32)
    valid = np.zeros((RES, RES), np.float32)
    for k in range(sub * sub):
        jx, jy = (k % sub + 0.5) / sub, (k // sub + 0.5) / sub
        pix = dr.arange(U, RES * RES)
        ii, jj = pix // RES, pix % RES
        x = (F(jj) + jx) * (2.0 / RES) - 1.0
        y = 1.0 - (F(ii) + jy) * (2.0 / RES)
        d = dr.normalize(mi.Vector3f(*[float(fwd[q]) + s * x * float(right[q]) + s * y * float(up[q]) for q in range(3)]))
        si = rt.scene.ray_intersect(mi.Ray3f(mi.Point3f(*[F(float(o[q])) for q in range(3)]), d))
        sid = dr.select(si.is_valid(), I32(rt.shape_id(si)), I32(-1))
        n = dr.select(dr.dot(si.n, d) > 0, -si.n, si.n)
        dr.eval(sid, n, si.t)
        sid = np.asarray(sid).reshape(RES, RES)
        ok = sid >= 0
        for c in range(9):
            cover[c] += (sid == c)
        nn = np.stack([np.asarray(n[q]) for q in range(3)]).reshape(3, RES, RES)
        nrm += np.where(ok, nn, 0)
        dep += np.where(ok, np.asarray(si.t).reshape(RES, RES), 0)
        valid += ok
    w = np.maximum(valid, 1)
    return {"cover": cover / (sub * sub), "normal": nrm / w, "depth": dep / w, "valid_frac": valid / (sub * sub)}


def centre_gbuffer(R, rt, cam):
    g = rt.gbuffer(dict(cam, res=RES), RES)
    return {"sid": g["sid"].astype(np.int16), "ok": g["ok"], "p": g["p"].astype(np.float32)}


def seed_sets(n_train, n_val, n_eval):
    S = P["seeds"]
    sets = {"train_noisy": [], "val_noisy": [], "eval_noisy": [], "train_target": [], "val_target": [], "eval_target": []}
    for split, nv, base, reps in (("train", n_train, S["train_noisy"], P["train_reps"]), ("val", n_val, S["val_noisy"], P["train_reps"]),
                                  ("eval", n_eval, S["eval_noisy"], P["eval_reps"])):
        for v in range(nv):
            for i, spp in enumerate(SPPS):
                for r in range(reps):
                    sets[f"{split}_noisy"].append(base + 1000 * v + 100 * i + r)
    sets["train_target"] = [S["train_target"] + v for v in range(n_train)]
    sets["val_target"] = [S["val_target"] + v for v in range(n_val)]
    sets["eval_target"] = [S["eval_target"] + 10 * v + h for v in range(n_eval) for h in range(2)]
    flat = [s for v in sets.values() for s in v] + P["seeds"]["replay"]
    assert len(flat) == len(set(flat)), "seed sets overlap"
    # Launch-level streams (seed * 1000003 + k) must not collide either.
    streams = set()
    for key, vals in sets.items():
        for s in vals:
            n_launch = (P["target_spp_train"] if "target" in key and not key.startswith("eval") else P["target_spp_eval"]) // 32 if "target" in key else 1
            for k in range(n_launch):
                assert s * 1000003 + k not in streams, ("stream collision", key, s)
                streams.add(s * 1000003 + k)
    return sets


def main(root, out):
    reader, learned, base, wl101e, prior, history, reference, manifest = RG.F101.inputs(root)
    R, rt = reader.R, reader.rt
    Bd = R.PT.fl(learned)
    out.mkdir(parents=True, exist_ok=False)
    run = {**manifest, "protocol_sha256": sha(PROTOCOL), "smoke": SMOKE, "views": {}, "gates": {},
           "experiment_code_sha256": {f.name: sha(f) for f in sorted(HERE.glob("*.py"))}}
    # G1 frozen-renderer replay (V0, archived seeds), skipped in smoke mode.
    if not SMOKE:
        g1 = []
        for j, row in enumerate(history["matched"]):
            comp = frame_components(R, rt, Bd, CAM.V0, 4096, row["seed"])
            with np.load(prior / row["path"]) as f:
                for k, key in enumerate(("e1", "e2", "hl")):
                    ref = f[key].astype(np.float64).reshape(RES, RES, 3).transpose(2, 0, 1)
                    mine = comp[3 * k:3 * k + 3].astype(np.float64)
                    g1.append({"seed": row["seed"], "component": key,
                               "relative_l1": float(np.abs(mine - ref).sum() / np.abs(ref).sum())})
        run["gates"]["G1_replay"] = {"rows": g1, "pass": max(r["relative_l1"] for r in g1) <= P["gates"]["replay_relative_l1"]}
        assert run["gates"]["G1_replay"]["pass"], g1
    # G3 determinism and G2 composition identity on one small frame.
    a = frame_components(R, rt, Bd, CAM.V0, 4, P["seeds"]["determinism"])
    b = frame_components(R, rt, Bd, CAM.V0, 4, P["seeds"]["determinism"])
    run["gates"]["G3_determinism"] = bool(np.array_equal(a, b))
    low, high, total = a[0:3] + a[3:6], a[6:9], a[0:3] + a[3:6] + a[6:9]
    run["gates"]["G2_composition_max_abs"] = float(np.abs((low + high) - total).max())
    assert run["gates"]["G3_determinism"] and run["gates"]["G2_composition_max_abs"] == 0.0
    # Cameras: evaluation views and accepted random training / validation views (geometry-only acceptance).
    ev = CAM.eval_cameras()
    n_train, n_val = (4, 2) if SMOKE else (P["n_train_views"], P["n_val_views"])
    cand = CAM.random_cameras(4 * (n_train + n_val), P["seeds"]["cameras"], A["geometry"], list(ev.values()))
    accepted = []
    for cam in cand:
        aux = aux_buffers(R, rt, cam, sub=2)
        if aux["valid_frac"].mean() >= P["min_valid_fraction"]:
            accepted.append(cam)
        if len(accepted) == n_train + n_val:
            break
    assert len(accepted) == n_train + n_val
    splits = {"train": accepted[:n_train], "val": accepted[n_train:], "eval": [ev[k] for k in ev]}
    names = {"train": [f"T{v:02d}" for v in range(n_train)], "val": [f"Q{v:02d}" for v in range(n_val)], "eval": list(ev)}
    sets = seed_sets(n_train, n_val, len(ev))
    run["seed_sets_sizes"] = {k: len(v) for k, v in sets.items()}
    run["gates"]["G4_seed_disjoint"] = True
    t_spp = {"train": P["target_spp_train"], "val": P["target_spp_train"], "eval": P["target_spp_eval"]}
    if SMOKE:
        t_spp = {k: 64 for k in t_spp}
        splits["eval"] = splits["eval"][:1]
        names["eval"] = names["eval"][:1]
    S = P["seeds"]
    for split in ("train", "val", "eval"):
        reps = P["eval_reps"] if split == "eval" else P["train_reps"]
        if SMOKE:
            reps = 1
        for v, (name, cam) in enumerate(zip(names[split], splits[split])):
            d = out / split / name
            d.mkdir(parents=True)
            stamp = time.perf_counter()
            aux = aux_buffers(R, rt, cam, sub=8)
            np.savez_compressed(d / "aux.npz", **aux, **centre_gbuffer(R, rt, cam))
            rec = {"camera": cam, "files": {}}
            if split == "eval":
                tseeds = [S["eval_target"] + 10 * v + h for h in range(2)]
                for h, ts in enumerate(tseeds):
                    np.savez_compressed(d / f"target_half{h}.npz", comp=frame_components(R, rt, Bd, cam, t_spp[split], ts))
                rec["target_seeds"] = tseeds
            else:
                ts = (S["train_target"] if split == "train" else S["val_target"]) + v
                np.savez_compressed(d / "target.npz", comp=frame_components(R, rt, Bd, cam, t_spp[split], ts))
                rec["target_seeds"] = [ts]
            base_seed = {"train": S["train_noisy"], "val": S["val_noisy"], "eval": S["eval_noisy"]}[split]
            for i, spp in enumerate(SPPS):
                for r in range(reps):
                    seed = base_seed + 1000 * v + 100 * i + r
                    np.savez_compressed(d / f"noisy_spp{spp}_r{r}.npz", comp=frame_components(R, rt, Bd, cam, spp, seed), seed=seed)
            rec["target_spp"] = t_spp[split]
            rec["seconds"] = time.perf_counter() - stamp
            rec["files"] = {p.name: sha(p) for p in sorted(d.iterdir())}
            run["views"][f"{split}/{name}"] = rec
            print(split, name, round(rec["seconds"], 1), flush=True)
            (out / "progress.json").write_text(json.dumps(run, indent=1))
    # ECRO frame cost on the ORIGINAL frozen frame path (V0, 512^2), synchronized, 3 warm-ups + 20 timed frames.
    timing = {}
    for spp in SPPS:
        for i in range(3):
            rt.frame(Bd, dict(CAM.V0, res=RES), RES, spp, P["seeds"]["determinism"] + 7000 + i)
        ts = []
        for i in range(20):
            R.sync()
            t0 = time.perf_counter()
            rt.frame(Bd, dict(CAM.V0, res=RES), RES, spp, P["seeds"]["determinism"] + 8000 + 100 * spp + i)
            R.sync()
            ts.append(1e3 * (time.perf_counter() - t0))
        timing[spp] = {"frame_ms_median": float(np.median(ts)), "frame_ms_p95": float(np.percentile(ts, 95))}
    run["ecro_frame_cost"] = timing
    (out / "run_manifest.json").write_text(json.dumps(run, indent=1))
    print("RENDER COMPLETE", flush=True)


if __name__ == "__main__":
    main(Path(argv[1]).resolve(), Path(argv[2]).resolve())
