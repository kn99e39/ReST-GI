"""FNTO evaluation on the worklog-37 held-out sets (same references, inputs, display transform, regions) (.venv-cu128).

    python fnto_eval.py prep                                  # per-pixel geometry caches (once; geometry only)
    python fnto_eval.py run --kernels <kernels.pt> --out <dir> [--views 0 1 ...] [--arms ...]

Held-out T1 (16 views; reference = mean of Cycles seeds 1, 2; inputs from seed 1):
  E_upd   : frozen K + the T1 coupling state (incrementally updated relation table rel_T1_inc, T1 port sun visibility)
  D_stale : frozen K + the T0 coupling state (relation table rel_T0, T0 port sun visibility) on the T1 view inputs
  E_full  : frozen K + a full T1 relation rebuild (check of the incremental update; not a separate method)
Held-out T0: S_T0 = frozen K + the T0 coupling state (static representability).
Per view the light is the view's single sun direction; the multi-bounce state is solved for that exact direction.
Pixel inputs (as for RNA / SPT-1): position, shading normal, camera direction, the pixel's own sun visibility
(diffuse_direct > 0). Box pixels are those the worklog-37 owner labels 16. Material per pixel: the mesh triangle under
the pixel (traced from the camera side), as in worklog 39's material diagnostic.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree

import fnto_common as C
import fnto_model as M
import fnto_relations as R
import fnto_solve as SV

SETS = {"heldout_T1": "T1", "heldout_T0": "T0"}
REG = ["all", "A_changed", "A_weak", "A_unchanged", "A_uncorresponded", "A_vis_changed", "B"]
PX = C.OUT / "eval_px"


def prep():
    A = R.load_anchors()
    trees = {o: (cKDTree(A["pos"][A["obj"] == o]), np.flatnonzero(A["obj"] == o)) for o in (0, 1)}
    z = np.load(C.B4 / "eval/views.npz")
    V_, F_ = R.mesh_a()
    PX.mkdir(parents=True, exist_ok=True)
    for set_name, state in SETS.items():
        f1 = h5py.File(C.B4 / f"data/{set_name}_s1_val.h5", "r"); f2 = h5py.File(C.B4 / f"data/{set_name}_s2_val.h5", "r")
        offs = np.stack(C.offsets(state)); geo = R.Geo(*C.offsets(state))
        out = {}; nv = f1["color"].shape[0]
        for v in range(nv):
            g = lambda f, k: np.nan_to_num(f[k][v].astype(np.float32))  # noqa: E731
            mk = g(f1, "alpha")[:, 0] > 0
            P = g(f1, "position")[mk].astype(np.float64); N = g(f1, "normal")[mk]; Cd = g(f1, "camera_dir")[mk]
            N = np.where(((N * Cd).sum(1) < 0)[:, None], -N, N)
            L = g(f1, "light_dir")[mk]; vis = g(f1, "diffuse_direct")[mk, 0] > 0
            tgt = 0.5 * (g(f1, "color") + g(f2, "color"))[mk]
            own = z[f"{set_name}_v{v}_own"].ravel()[mk]
            dom = (own == C.B_UNIT).astype(np.int64)
            canon = P - offs[dom]
            port4 = np.full((len(P), 4), -1, np.int64); w4 = np.zeros((len(P), 4)); port = np.full(len(P), -1, np.int64)
            for o in (0, 1):
                m = dom == o
                if not m.any():
                    continue
                tree, gi = trees[o]
                d, ii = tree.query(canon[m], k=8, workers=-1)
                cand = gi[ii]
                ok = (A["nrm"][cand] * N[m][:, None, :]).sum(-1) > 0.5
                rank = np.where(ok, np.arange(8)[None], 99)
                sel = np.argsort(rank, 1, kind="stable")[:, :4]
                okk = np.take_along_axis(ok, sel, 1); dd = np.take_along_axis(d, sel, 1); cc = np.take_along_axis(cand, sel, 1)
                none = ~okk.any(1); okk[none, 0] = True
                ww = np.where(okk, 1.0 / (dd + 2e-3), 0.0); ww /= ww.sum(1, keepdims=True)
                port4[m] = np.where(okk, cc, -1); w4[m] = ww; port[m] = cc[:, 0]
            # material: the triangle under the pixel, traced from 1 cm on the camera side
            o_ = P + 0.01 * Cd; valid, hp, prim, sid, t = geo.intersect(o_, -Cd)
            mat = np.full(len(P), 8, np.int64)
            ma = valid & (sid == geo.names.index("A")) & (dom == 0)
            mat[ma] = A["tri_mat"][prim[ma]]
            mat[(dom == 0) & ~ma] = A["mat"][port[(dom == 0) & ~ma]]
            out[v] = dict(mk=np.flatnonzero(mk).astype(np.int32), pos=canon.astype(np.float32), nrm=N.astype(np.float32), cam=Cd.astype(np.float32),
                          light=L.mean(0) / np.linalg.norm(L.mean(0)), vis=vis.astype(np.uint8), tgt=tgt.astype(np.float32), dom=dom.astype(np.int8),
                          port=port.astype(np.int32), port4=port4.astype(np.int32), w4=w4.astype(np.float32), mat=mat.astype(np.int8),
                          mat_traced=ma.astype(np.uint8))
            print(set_name, v, len(P), "box px", int(dom.sum()), "traced", float(ma[dom == 0].mean()), flush=True)
        np.savez_compressed(PX / f"{set_name}.npz", **{f"v{v}_{k}": a for v, d in out.items() for k, a in d.items()}, nv=nv,
                            res=int(f1.attrs["resolution"][0]))


def load_kernels(path, dev):
    KA, KB = M.make_kernels(dev)
    s = torch.load(path, map_location=dev)
    KA.load_state_dict(s["A"]); KB.load_state_dict(s["B"]); KA.eval(); KB.eval()
    return KA, KB


def hf_mask(tgt, mk, res):
    lum = C.ldr3(tgt).mean(1); full = np.zeros(res * res); full[mk] = lum; fm = np.zeros(res * res, bool); fm[mk] = True
    fm = fm.reshape(res, res); L2 = full.reshape(res, res); grad = np.zeros_like(L2)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        ok = fm & np.roll(fm, (dy, dx), (0, 1)); grad = np.maximum(grad, np.where(ok, np.abs(L2 - np.roll(L2, (dy, dx), (0, 1))), 0))
    gm = grad.reshape(-1)[mk]
    return gm >= np.quantile(gm, 0.9)


def run(a):
    dev = torch.device("cuda")
    KA, KB = load_kernels(a.kernels, dev)
    dig0 = M.digest(KA, KB)
    t = time.time(); T = SV.Tables(KA, KB, dev); torch.cuda.synchronize(); t_tables = time.time() - t
    A = R.load_anchors()
    rel = {k: np.load(C.OUT / f"rel_{k}.npz")["hit"] for k in ("T0", "T1", "T1_inc")}
    geos = {st: R.Geo(*C.offsets(st)) for st in ("T0", "T1")}
    z = np.load(C.B4 / "eval/views.npz")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    record = {"kernels": str(a.kernels), "tables_s": t_tables, "sets": {}, "per_view": {}}
    keep = {}
    plan = {"heldout_T1": {"E_upd": ("T1_inc", "T1"), "D_stale": ("T0", "T0"), "E_full": ("T1", "T1")},
            "heldout_T0": {"S_T0": ("T0", "T0")}}
    for set_name, arms in plan.items():
        arms = {k: v for k, v in arms.items() if not a.arms or k in a.arms}
        if not arms:
            continue
        D = np.load(PX / f"{set_name}.npz"); res = int(D["res"]); nv = int(D["nv"])
        acc = {n: {"sq": [], "logl1": 0.0, "n3": 0, "hf": [], "reg": {r: [0.0, 0] for r in REG}, "mat": np.zeros((9, 2)),
                   "lit": [0.0, 0], "shadow": [0.0, 0], "solve_s": [], "iters": [], "render_s": [], "vis_s": []} for n in arms}
        for v in (a.views if a.views else range(nv)):
            g = lambda k: D[f"v{v}_{k}"]  # noqa: E731
            mk = g("mk"); tgt = g("tgt")
            px = {k: torch.as_tensor(g(k).astype(np.float32 if k in ("pos", "nrm", "cam", "vis", "w4") else np.int64), device=dev)
                  for k in ("pos", "nrm", "cam", "vis", "dom", "port", "port4", "w4")}
            light = g("light"); mat = g("mat").astype(np.int64); vis = g("vis").astype(bool)
            masks = {"all": np.ones(len(tgt), bool)}
            if set_name == "heldout_T1":
                for r in ("A_changed", "A_weak", "A_unchanged", "A_uncorresponded", "A_vis_changed"):
                    masks[r] = z[f"heldout_T1_v{v}_reg_{r}"].ravel()[mk]
                masks["B"] = g("dom") == 1
            hf = hf_mask(tgt, mk, res); ref = C.ldr3(tgt)
            for n, (rk, vk) in arms.items():
                tv = time.time()
                Vp = np.zeros(len(A["obj"]), bool); idx = np.arange(len(A["obj"]))
                cosv = A["nrm"] @ light; lit = idx[cosv > 0]
                Vp[lit] = R.sun_visible(A, geos[vk], lit, light); t_vis = time.time() - tv
                S = SV.State(T, rel[rk], Vp, light).solve()
                tr = time.time(); pr, parts = SV.render_pixels(T, S, px); torch.cuda.synchronize(); t_r = time.time() - tr
                pr = pr.cpu().numpy(); parts = parts.cpu().numpy()
                dl = C.ldr3(pr) - ref; sq = (dl ** 2).mean(1); Q = acc[n]
                Q["sq"].append(float(sq.mean())); Q["logl1"] += float(np.abs(np.log1p(np.clip(pr, 0, None)) - np.log1p(tgt)).sum()); Q["n3"] += pr.size
                Q["hf"].append(float(np.abs(dl).mean(1)[hf].mean())); Q["solve_s"].append(S.solve_s); Q["iters"].append(S.iters)
                Q["render_s"].append(t_r); Q["vis_s"].append(t_vis)
                for r, mm in masks.items():
                    Q["reg"][r][0] += float(sq[mm].sum()); Q["reg"][r][1] += int(mm.sum())
                np.add.at(Q["mat"][:, 0], mat, sq); np.add.at(Q["mat"][:, 1], mat, 1)
                Q["lit"][0] += float(sq[vis].sum()); Q["lit"][1] += int(vis.sum()); Q["shadow"][0] += float(sq[~vis].sum()); Q["shadow"][1] += int((~vis).sum())
                record["per_view"].setdefault(set_name, {}).setdefault(n, []).append({"v": v, "psnr": float(10 * np.log10(1 / sq.mean())), "iters": S.iters,
                                                                                       "solve_s": S.solve_s, "resid": S.hist[-1], "diverged": S.diverged})
                if v in a.keep:
                    img = np.zeros((res * res, 15), np.float16); img[mk] = np.concatenate([pr, parts.reshape(len(pr), -1)], 1)
                    keep[f"{set_name}_v{v}_{n}"] = img.reshape(res, res, 15)
                print(set_name, v, n, f"psnr {10 * np.log10(1 / sq.mean()):.3f} it {S.iters} solve {S.solve_s:.2f}s render {t_r:.2f}s", flush=True)
                del S
        rows = {}
        for n, Q in acc.items():
            sq = np.array(Q["sq"])
            rows[n] = {"view_mean_psnr": float(np.mean(10 * np.log10(1 / sq))), "per_view_psnr": (10 * np.log10(1 / sq)).tolist(),
                       "pooled_psnr": float(10 * np.log10(Q["reg"]["all"][1] / Q["reg"]["all"][0])),
                       "logl1": Q["logl1"] / Q["n3"], "hf_abs": float(np.mean(Q["hf"])),
                       "regions": {r: {"mse": (x[0] / x[1]) if x[1] else None, "px": x[1]} for r, x in Q["reg"].items()},
                       "materials": {C.MATERIALS[i] if i < 8 else "B": {"mse": float(Q["mat"][i, 0] / Q["mat"][i, 1]) if Q["mat"][i, 1] else None,
                                                                         "px": int(Q["mat"][i, 1])} for i in range(9)},
                       "lit_mse": Q["lit"][0] / max(Q["lit"][1], 1), "shadow_mse": Q["shadow"][0] / max(Q["shadow"][1], 1),
                       "lit_px": Q["lit"][1], "shadow_px": Q["shadow"][1],
                       "solve_s_mean": float(np.mean(Q["solve_s"])), "iters_mean": float(np.mean(Q["iters"])), "iters_max": int(np.max(Q["iters"])),
                       "render_s_mean": float(np.mean(Q["render_s"])), "port_sun_vis_s_mean": float(np.mean(Q["vis_s"]))}
        record["sets"][set_name] = rows
    record["kernel_parameters_bit_identical"] = M.digest(KA, KB) == dig0
    record["kernel_parameters"] = {"A": sum(p.numel() for p in KA.parameters()), "B": sum(p.numel() for p in KB.parameters())}
    np.savez_compressed(out / "views.npz", **keep)
    (out / "eval.json").write_text(json.dumps(record, indent=1))
    for s_, rows in record["sets"].items():
        print(s_, {n: round(r["view_mean_psnr"], 3) for n, r in rows.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd")
    ap.add_argument("--kernels"); ap.add_argument("--out"); ap.add_argument("--views", type=int, nargs="*", default=[])
    ap.add_argument("--arms", nargs="*", default=[]); ap.add_argument("--keep", type=int, nargs="*", default=[6, 15])
    a = ap.parse_args()
    prep() if a.cmd == "prep" else run(a)
