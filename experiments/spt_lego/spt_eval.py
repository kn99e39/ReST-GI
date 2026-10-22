"""SPT-1 evaluation on the B4 held-out sets (same references, inputs, display transform and regions as worklog 37).

    python spt_eval.py --t0 <SPT_T0 state.pt> [--t1 <SPT_T1 state.pt>] --out <dir>

Held-out T1 (16 views; reference = mean of Cycles seeds 1, 2; inputs from seed 1) arms:
  C_upd   : SPT_T0 parameters + relations of T1 (the edit is applied by state recomputation only)
  D_stale : SPT_T0 parameters + relations of T0 (factorization control: identical parameters, stale state)
  D_zero  : SPT_T0 parameters, incident state set to 0 (how much the persistent direct path carries alone)
  C_full  : SPT_T1 (trained on T1 data with T1 relations): the candidate's own full reconstruction
Held-out T0: S_T0 = SPT_T0 + relations of T0 (static representability).
Regions come from the worklog-37 evaluation record (results/rna_selective_reconstruction_lego/eval/views.npz).
Per view the light is the view's single sun direction: the multi-bounce cache is solved for that exact direction.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch

import spt_common as C
import spt_model as SM
from spt_train import aabbs

DSTAR = [0, 2, 3, 6, 7, 8, 13, 16]
REG = ["all", "Dstar_owned", "Vstar_owned", "A_changed", "A_weak", "A_unchanged", "A_uncorresponded", "A_vis_changed", "B"]


def ldr3(x):
    x = np.clip(x, 0, None)
    return (x / (1 + x)) ** (1 / 2.2)


@torch.no_grad()
def single_light_cache(model, R, alb, wl, iters=4):
    """E_ind[a, 3] for one light direction wl [3] (exact cosine, visibility of its bin)."""
    esun = model.log_esun.exp(); b = int(SM.light_bin_t(wl[None])[0])
    edir = esun * (R.nrm @ wl).clamp(min=0) * R.vis[:, b].float()
    E = torch.zeros(R.Na, 3, device=R.pos.device)
    valid = (R.hit >= 0).float(); S = R.hit.clamp(min=0)
    for _ in range(iters):
        rad = alb / math.pi * (edir[:, None] + E)
        E = (rad[S] * valid[..., None]).sum(1) * (math.pi / R.K)
    return E


@torch.no_grad()
def render_view(model, R, alb, wl, canon, obj, nrm, cam, vis, idx, w, zero=False):
    t = time.perf_counter()
    if zero:
        g = torch.zeros(canon.shape[0], 9, 3, device=canon.device)
        t_state = 0.0
    else:
        E = single_light_cache(model, R, alb, wl)
        ga = SM.anchor_state(model, R, alb, torch.arange(R.Na, device=canon.device), wl[None].expand(R.Na, 3), E)   # [Na, 9, 3]
        use = (idx >= 0).float()
        g = (ga[idx.clamp(min=0)] * (w * use)[..., None, None]).sum(1)
        torch.cuda.synchronize(); t_state = time.perf_counter() - t
    t2 = time.perf_counter()
    out = []
    for s in range(0, canon.shape[0], 1 << 18):
        sl = slice(s, s + (1 << 18))
        pr, parts = model.decode(model.feat(canon[sl], obj[sl]), nrm[sl], cam[sl], wl[None].expand(cam[sl].shape[0], 3), vis[sl], g[sl])
        out.append(torch.cat([pr, parts["direct"], parts["indirect"]], 1))
    torch.cuda.synchronize()
    o = torch.cat(out).cpu().numpy()
    return o[:, :3], o[:, 3:6], o[:, 6:9], t_state, time.perf_counter() - t2


def load_model(path, dev):
    A, B = aabbs()
    m = SM.SPT(A, B).to(dev); m.load_state_dict(torch.load(path, map_location=dev)); m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", required=True); ap.add_argument("--t1", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=int, nargs="*", default=[6, 15])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    m0 = load_model(a.t0, dev); m1 = load_model(a.t1, dev) if a.t1 else None
    dig0 = SM.digest(m0)
    R = SM.Relations("T0", dev)
    Rv = {"T0": (R.hit.clone(), R.vis.clone())}
    R.set_state("T1"); Rv["T1"] = (R.hit.clone(), R.vis.clone())
    use = lambda st: (setattr(R, "hit", Rv[st][0]), setattr(R, "vis", Rv[st][1]))  # noqa: E731
    z = np.load(C.B4 / "eval/views.npz")
    record = {"sets": {}, "digest_T0_model_unchanged": None, "timing": {}}
    keep = {}
    for set_name, state, arms in (("heldout_T1", "T1", ["C_upd", "D_stale", "D_zero"] + (["C_full"] if m1 else [])),
                                  ("heldout_T0", "T0", ["S_T0"])):
        f1 = h5py.File(C.B4 / f"data/{set_name}_s1_val.h5", "r"); f2 = h5py.File(C.B4 / f"data/{set_name}_s2_val.h5", "r")
        px = {k: np.load(C.OUT / f"pixels/{set_name}_s1_val.{k}.npy") for k in ("idx", "w", "obj", "start")}
        offs = np.stack(C.offsets(state)); res = int(f1.attrs["resolution"][0]); nv = f1["color"].shape[0]
        acc = {n: {"sq": [], "logl1": 0.0, "n3": 0, "hf": [], "reg": {r: [0.0, 0] for r in REG}, "ts": [], "td": []} for n in arms}
        for v in range(nv):
            g = lambda f, k: np.nan_to_num(f[k][v].astype(np.float32))  # noqa: E731
            mk = g(f1, "alpha")[:, 0] > 0; s, e = px["start"][v], px["start"][v + 1]
            obj = px["obj"][s:e].astype(np.int64)
            P = g(f1, "position")[mk]; N = g(f1, "normal")[mk]; Cd = g(f1, "camera_dir")[mk]
            N = np.where(((N * Cd).sum(1) < 0)[:, None], -N, N)
            L = g(f1, "light_dir")[mk]; vis = g(f1, "diffuse_direct")[mk, 0] > 0
            tgt = 0.5 * (g(f1, "color") + g(f2, "color"))[mk]
            wl = torch.tensor(L.mean(0) / np.linalg.norm(L.mean(0)), dtype=torch.float32, device=dev)
            T = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt, device=dev)  # noqa: E731
            canon = T(P - offs[obj]); objt = T(obj, torch.int64); nrm, cam, vist = T(N), T(Cd), T(vis.astype(np.float32))
            idx, w = T(px["idx"][s:e], torch.int64), T(px["w"][s:e].astype(np.float32))
            masks = {"all": np.ones(len(P), bool)}
            if set_name == "heldout_T1":
                own = z[f"heldout_T1_v{v}_own"].ravel()[mk]
                for r in ("A_changed", "A_weak", "A_unchanged", "A_uncorresponded", "A_vis_changed"):
                    masks[r] = z[f"heldout_T1_v{v}_reg_{r}"].ravel()[mk]
                masks["B"] = own == 16; masks["Dstar_owned"] = np.isin(own, DSTAR) & (own >= 0); masks["Vstar_owned"] = (own >= 0) & ~np.isin(own, DSTAR)
            lum = ldr3(tgt).mean(1); full = np.zeros(res * res); full[mk] = lum; fm = mk.reshape(res, res); L2 = full.reshape(res, res); grad = np.zeros_like(L2)
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                ok = fm & np.roll(fm, (dy, dx), (0, 1)); grad = np.maximum(grad, np.where(ok, np.abs(L2 - np.roll(L2, (dy, dx), (0, 1))), 0))
            gm = grad.reshape(-1)[mk]; hf = gm >= np.quantile(gm, 0.9)
            for n in arms:
                model = m1 if n == "C_full" else m0
                use("T0" if n in ("D_stale", "S_T0") else "T1")
                alb = model.anchor_albedo(R).detach()
                pr, dpart, ipart, ts, td = render_view(model, R, alb, wl, canon, objt, nrm, cam, vist, idx, w, zero=(n == "D_zero"))
                dl = ldr3(pr) - ldr3(tgt); sq = (dl ** 2).mean(1); A_ = acc[n]
                A_["sq"].append(float(sq.mean())); A_["logl1"] += float(np.abs(np.log1p(np.clip(pr, 0, None)) - np.log1p(tgt)).sum()); A_["n3"] += pr.size
                A_["hf"].append(float(np.abs(dl).mean(1)[hf].mean())); A_["ts"].append(ts); A_["td"].append(td)
                for r, mm in masks.items():
                    A_["reg"][r][0] += float(sq[mm].sum()); A_["reg"][r][1] += int(mm.sum())
                if v in a.keep:
                    img = np.zeros((res * res, 9), np.float16); img[mk] = np.concatenate([pr, dpart, ipart], 1)
                    keep[f"{set_name}_v{v}_{n}"] = img.reshape(res, res, 9)
        rows = {}
        for n, A_ in acc.items():
            sq = np.array(A_["sq"])
            rows[n] = {"view_mean_psnr": float(np.mean(10 * np.log10(1 / sq))), "per_view_psnr": (10 * np.log10(1 / sq)).tolist(),
                       "logl1": A_["logl1"] / A_["n3"], "hf_abs": float(np.mean(A_["hf"])),
                       "regions": {r: {"mse": (v_[0] / v_[1]) if v_[1] else None, "px": v_[1]} for r, v_ in A_["reg"].items()},
                       "state_s_per_view": float(np.mean(A_["ts"])), "decode_s_per_view": float(np.mean(A_["td"]))}
            rows[n]["pooled_psnr"] = float(10 * np.log10(1 / (A_["reg"]["all"][0] / A_["reg"]["all"][1])))
        record["sets"][set_name] = rows
    record["digest_T0_model_unchanged"] = SM.digest(m0) == dig0
    record["params"] = {"SPT_T0": int(sum(p.numel() for p in m0.parameters()))}
    np.savez_compressed(out / "views.npz", **keep)
    (out / "eval.json").write_text(json.dumps(record, indent=1))
    for s_, rows in record["sets"].items():
        print(s_, {n: round(r["view_mean_psnr"], 3) for n, r in rows.items()})


if __name__ == "__main__":
    main()
