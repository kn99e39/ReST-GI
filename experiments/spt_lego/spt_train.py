"""SPT-1 training (torch; RNA venv on LabServer63 or WSL).

    python spt_train.py --data train_T0 --state T0 --out <dir> [--epochs 150] [--pixels 65536] [--seed 0] [--resume]

One geometry state's training set (B4 RNA-format data, per-pixel random sun direction). The relation table of the
same state supplies the dynamic incident state; persistent fields and heads are the only trainable parameters.
Loss: L2 on log(1 + radiance) (RNA's 'l2' + use_log). Adam 1e-3, x0.5 every 50 epochs (RNA's schedule).
Each step samples --pixels pixels of one training image (images shuffled per epoch). The multi-bounce cache is
refreshed every --cache-every epochs from the current albedo (no gradient). Final weights are used; a checkpoint is
written every 10 epochs and --resume continues from it (label the run 'resumed' if used).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

import spt_common as C
import spt_model as SM


def build_cache(name, state):
    """Masked RNA fields of a dataset in host-memory-friendly .npy files (results/spt_lego/data_cache/<name>/)."""
    d = C.OUT / "data_cache" / name
    if (d / "done").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    f = h5py.File(C.B4 / f"data/{name}.h5", "r")
    obj = np.load(C.OUT / f"pixels/{name}.obj.npy"); start = np.load(C.OUT / f"pixels/{name}.start.npy")
    offs = np.stack(C.offsets(state))
    out = {k: np.lib.format.open_memmap(d / f"{k}.npy", "w+", np.float16, (start[-1], 3)) for k in ("pos", "nrm", "cam", "light", "col")}
    vis = np.lib.format.open_memmap(d / "vis.npy", "w+", np.uint8, (start[-1],))
    for i in range(f["alpha"].shape[0]):
        a = np.nan_to_num(f["alpha"][i, :, 0].astype(np.float32)); m = a > 0; s, e = start[i], start[i + 1]
        assert m.sum() == e - s
        p = f["position"][i][m].astype(np.float64); n = f["normal"][i][m].astype(np.float32); c = f["camera_dir"][i][m].astype(np.float32)
        n = np.where(((n * c).sum(1) < 0)[:, None], -n, n)
        out["pos"][s:e] = p - offs[obj[s:e]]                       # canonical position
        out["nrm"][s:e] = n; out["cam"][s:e] = c
        out["light"][s:e] = np.nan_to_num(f["light_dir"][i][m].astype(np.float32))
        out["col"][s:e] = np.nan_to_num(f["color"][i][m].astype(np.float32))
        vis[s:e] = (np.nan_to_num(f["diffuse_direct"][i][m, 0].astype(np.float32)) > 0)
    for v in out.values():
        v.flush()
    vis.flush(); (d / "done").write_text("ok")
    return d


def load(name, state):
    d = build_cache(name, state)
    D = {k: np.load(d / f"{k}.npy") for k in ("pos", "nrm", "cam", "light", "col", "vis")}
    for k in ("idx", "w", "obj", "start"):
        D[k] = np.load(C.OUT / f"pixels/{name}.{k}.npy")
    return D


def aabbs():
    j = C.OUT / "aabb_A.json"                        # the train_T0 dataset AABB (canonical frame of A), stored once
    if not j.exists():
        f = h5py.File(C.B4 / "data/train_T0.h5", "r")
        j.write_text(json.dumps({"min": f.attrs["aabb_min"].tolist(), "max": f.attrs["aabb_max"].tolist()}))
    q = json.loads(j.read_text()); A = np.stack([q["min"], q["max"]]).astype(np.float64)
    bV = np.load(C.B4 / "scene/B_box_T0.npz")["V"]; lo, hi = bV.min(0), bV.max(0); pad = 0.02 * (hi - lo)
    return A, np.stack([lo - pad, hi + pad])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--state", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=150); ap.add_argument("--pixels", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--cache-every", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test only")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    t_load = time.time(); D = load(a.data, a.state); t_load = time.time() - t_load
    R = SM.Relations(a.state, dev)
    A_aabb, B_aabb = aabbs()
    model = SM.SPT(A_aabb, B_aabb).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, 50, 0.5)
    rec = {"args": vars(a), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "load_s": t_load,
           "parameters": int(sum(p.numel() for p in model.parameters())), "epochs": [], "resumed_from": None,
           "relations_sha16": C.sha16(C.OUT / f"rel_{a.state}.npz"), "anchors_sha16": C.sha16(C.OUT / "anchors.npz")}
    ep0 = 0; ck = out / "checkpoint.pt"
    if a.resume and ck.exists():
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"]); ep0 = s["epoch"]
        rec = s["rec"]; rec["resumed_from"] = (rec.get("resumed_from") or []) + [ep0]
        rng = np.random.default_rng(a.seed + 1000 * ep0)
    start = D["start"]; n_img = len(start) - 1
    T = lambda x, dt=torch.float32: torch.as_tensor(x).to(dev, dt, non_blocking=True)  # noqa: E731
    ecache = torch.zeros(R.Na, C.N_BINS, 3, device=dev, dtype=torch.float16)
    steps = 0; t_train = time.time()
    for ep in range(ep0, a.epochs):
        t_ep = time.time()
        if ep % a.cache_every == 0:
            tc = time.time(); ecache = SM.multibounce_cache(model, R); torch.cuda.synchronize(); tc = time.time() - tc
        else:
            tc = 0.0
        losses = []
        for i in rng.permutation(n_img):
            s, e = start[i], start[i + 1]
            sel = s + rng.choice(e - s, size=min(a.pixels, e - s), replace=False)
            sel.sort()
            canon, nrm, cam, light = T(D["pos"][sel]), T(D["nrm"][sel]), T(D["cam"][sel]), T(D["light"][sel])
            col, vis, obj = T(D["col"][sel]), T(D["vis"][sel]), T(D["obj"][sel], torch.int64)
            idx, w = T(D["idx"][sel], torch.int64), T(D["w"][sel])
            alb = model.anchor_albedo(R)
            g = SM.pixel_state(model, R, alb, idx, w, light, ecache)
            pred, _ = model.decode(model.feat(canon, obj), nrm, cam, light, vis, g)
            loss = ((torch.log1p(pred.clamp(min=0)) - torch.log1p(col)) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(float(loss)); steps += 1
            if a.max_steps and steps >= a.max_steps:
                break
        sched.step()
        torch.cuda.synchronize()
        rec["epochs"].append({"epoch": ep, "loss": float(np.mean(losses)), "seconds": time.time() - t_ep, "cache_s": tc,
                              "esun": float(model.log_esun.exp())})
        print(json.dumps(rec["epochs"][-1]), flush=True)
        if a.max_steps and steps >= a.max_steps:
            break
        if (ep + 1) % 10 == 0 or ep + 1 == a.epochs:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep + 1, "rec": rec}, ck)
    rec["train_s"] = time.time() - t_train; rec["steps"] = steps
    rec["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    torch.save(model.state_dict(), out / "state.pt"); rec["final_digest"] = SM.digest(model)
    (out / "train.json").write_text(json.dumps(rec, indent=1))
    print("done", json.dumps({k: rec[k] for k in ("parameters", "train_s", "steps", "peak_mem_gib")}))


if __name__ == "__main__":
    main()
