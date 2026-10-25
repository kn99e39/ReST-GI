"""WL42 r1 evaluation on the worklog-37 held-out views (same cameras, suns, pixels, regions as worklogs 37-41).

    python r1_eval.py prep                               # PC (Mitsuba): pixel identity + port sun fractions per view
    python r1_eval.py run --kernels <kernels.pt> --out <dir>                 # server (torch)

Targets (labelled, never mixed): hist = historical held-out references (mean of Cycles seeds 1/2, 4096 spp,
clamp_direct 20 / clamp_indirect 5, point-sampled); r1 = reference regime r1 (no clamp, point-sampled, 4096 spp, server);
first_r1 = intrinsic-all first-event r1 references (attribution of K alone).
Arms (same frozen K in all of them; parameter digests compared):
  K_only     coupling switched off: K applied to the exact delta sun (port visible fractions, pixel visibility)
  S_T0       T0 coupling state (rel_T0, T0 port sun fractions) on held-out T0
  D_stale    T0 coupling state on held-out T1 inputs (the pixel's own T1 sun visibility, as Batch 6 / SPT-1)
  E_upd      T1 coupling state from the INCREMENTAL relation update (rel_T1_inc) + T1 port sun fractions
  E_relfull  E_upd with a full T1 relation rebuild: a consistency check of the incremental update, NOT a refit and NOT
             a reference (audit #15)
Signed change (contract l.153, audit #14): on pixels that show the Lego in both views (T0 and T1 share cameras and suns),
d_pred = E_upd(T1) - S_T0(T0), d_stale = D_stale(T1) - S_T0(T0), d_ref = ref(T1) - ref(T0); reported: correlation, least-
squares slope of d_pred on d_ref, sign agreement where |d_ref| exceeds the T0/T1 reference noise, and explained fraction
1 - |d_pred - d_ref|^2 / |d_ref|^2, on the changed region A_changed and on the stationary remainder.
Solver: per view status (converged / unconverged / diverged), iterations, residual and ratio histories; spectral radius
of the homogeneous composed map per relation table.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import r1_common as C

SETS = {"heldout_T0": "T0", "heldout_T1": "T1"}
PX = C.OUT1 / "eval_px"
REG = ["all", "A_changed", "A_weak", "A_unchanged", "A_uncorresponded", "A_vis_changed", "B"]


def prep():
    import h5py
    import fnto_relations as R
    import r1_map as Mp
    import r1_relations as RR
    A = RR.load(); PX.mkdir(parents=True, exist_ok=True)
    z = np.load(C.B4 / "eval/views.npz"); vis_out = {}
    geos = {st: R.Geo(*C.offsets(st)) for st in ("T0", "T1")}
    for set_name, state in SETS.items():
        f1 = h5py.File(C.B4 / f"data/{set_name}_s1_val.h5", "r"); f2 = h5py.File(C.B4 / f"data/{set_name}_s2_val.h5", "r")
        offs = np.stack(C.offsets(state)); geo = geos[state]; out = {}; nv = f1["color"].shape[0]
        for v in range(nv):
            g = lambda f, k: np.nan_to_num(f[k][v].astype(np.float32))  # noqa: E731
            mk = g(f1, "alpha")[:, 0] > 0
            P = g(f1, "position")[mk].astype(np.float64); N = g(f1, "normal")[mk]; Cd = g(f1, "camera_dir")[mk].astype(np.float64)
            N = np.where(((N * Cd).sum(1) < 0)[:, None], -N, N)
            L = g(f1, "light_dir")[mk]; light = L.mean(0) / np.linalg.norm(L.mean(0)); vis = g(f1, "diffuse_direct")[mk, 0] > 0
            tgt = 0.5 * (g(f1, "color") + g(f2, "color"))[mk]
            own = z[f"{set_name}_v{v}_own"].ravel()[mk]; dom = (own == C.B_UNIT).astype(np.int64)
            port = np.full(len(P), -1, np.int64)
            pa, ok = Mp.pixels_to_ports(A, geo, P[dom == 0], Cd[dom == 0]); port[np.flatnonzero(dom == 0)[ok]] = pa[ok]
            if (dom == 1).any():
                mb = dom == 1
                port[mb] = RR.map_hits(A, geo, np.ones(int(mb.sum()), bool), P[mb], np.zeros(int(mb.sum()), np.int64),
                                       np.full(int(mb.sum()), geo.names.index("B")), -Cd[mb])
            canon = P - offs[dom]
            valid = port >= 0
            mat = np.where(dom == 1, 8, A["mat"][np.clip(port, 0, None)])
            out[v] = dict(mk=np.flatnonzero(mk).astype(np.int32), pos=canon.astype(np.float32), nrm=N.astype(np.float32), cam=Cd.astype(np.float32),
                          light=light, vis=vis.astype(np.uint8), tgt=tgt.astype(np.float32), dom=dom.astype(np.int8), port=port.astype(np.int32),
                          valid=valid.astype(np.uint8), mat=mat.astype(np.int8), own=own.astype(np.int16))
            for st in (("T0", "T1") if set_name == "heldout_T1" else ("T0",)):
                t = time.time(); frac, _ = RR.sun_vis(A, geos[st], light)
                vis_out[f"{set_name}_v{v}_{st}"] = frac.astype(np.float16); vis_out[f"{set_name}_v{v}_{st}_s"] = np.float32(time.time() - t)
            print(set_name, v, "px", len(P), "identity traced", float(valid.mean()), flush=True)
        np.savez_compressed(PX / f"{set_name}.npz", **{f"v{v}_{k}": a for v, d in out.items() for k, a in d.items()}, nv=nv, res=int(f1.attrs["resolution"][0]))
    np.savez_compressed(C.OUT1 / "eval_vis.npz", **vis_out)


def load_kernels(path, dev):
    import r1_train as TR
    KA, KB = TR.make_kernels(dev); s = torch.load(path, map_location=dev)
    KA.load_state_dict(s["A"]); KB.load_state_dict(s["B"]); KA.eval(); KB.eval()
    return KA, KB


def hf_mask(tgt, mk, res):
    lum = C.ldr3(tgt).mean(1); full = np.zeros(res * res); full[mk] = lum; fm = np.zeros(res * res, bool); fm[mk] = True
    fm = fm.reshape(res, res); L2 = full.reshape(res, res); grad = np.zeros_like(L2)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        ok = fm & np.roll(fm, (dy, dx), (0, 1)); grad = np.maximum(grad, np.where(ok, np.abs(L2 - np.roll(L2, (dy, dx), (0, 1))), 0))
    gm = grad.reshape(-1)[mk]; return gm >= np.quantile(gm, 0.9)


def metrics(pr, tgt, masks, mat, vis, hf):
    dl = C.ldr3(pr) - C.ldr3(tgt); sq = (dl ** 2).mean(1)
    r = {"mse": float(sq.mean()), "logl1_sum": float(np.abs(np.log1p(np.clip(pr, 0, None)) - np.log1p(tgt)).sum()), "n3": int(pr.size),
         "hf": float(np.abs(dl).mean(1)[hf].mean()), "reg": {k: [float(sq[m].sum()), int(m.sum())] for k, m in masks.items()},
         "mat": [[float(sq[mat == i].sum()), int((mat == i).sum())] for i in range(9)],
         "lit": [float(sq[vis].sum()), int(vis.sum())], "shadow": [float(sq[~vis].sum()), int((~vis).sum())]}
    return r


def run(a):
    import r1_solve as SV
    import fnto_model as M
    from scipy.spatial import cKDTree
    dev = torch.device("cuda"); KA, KB = load_kernels(a.kernels, dev); dig0 = M.digest(KA, KB)
    T = SV.Tables(KA, KB, dev)
    rel = {k: np.load(C.OUT1 / f"rel_{k}.npz")["hit"] for k in ("T0", "T1", "T1_inc")}
    pv = np.load(C.OUT1 / os.environ.get("R1_EVAL_VIS", "eval_vis.npz")); z = np.load(C.B4 / "eval/views.npz")   # WL44 candidate: eval_vis_k44.npz
    refs = {}
    for t_name, f_name in (("r1", "full"), ("first_r1", "first")):
        refs[t_name] = {s: (dict(np.load(C.OUT1 / f"refs/{s}_{f_name}.npz")) if (C.OUT1 / f"refs/{s}_{f_name}.npz").exists() else None) for s in SETS}
    rec = {"kernels": str(a.kernels), "eval_vis": os.environ.get("R1_EVAL_VIS", "eval_vis.npz"), "spectral_radius": {}, "sets": {}, "per_view": {}, "signed_change": {}}
    for k in ("T0", "T1_inc"):
        rec["spectral_radius"][k] = SV.spectral_radius(T, rel[k])
        print("spectral radius", k, rec["spectral_radius"][k]["rho_estimate"], flush=True)
    plan = {"heldout_T0": {"K_only": (None, "T0"), "S_T0": ("T0", "T0")},
            "heldout_T1": {"K_only": (None, "T1"), "D_stale": ("T0", "T0"), "E_upd": ("T1_inc", "T1"), "E_relfull": ("T1", "T1")}}
    preds = {}; views = {}
    for set_name, arms in plan.items():
        D = np.load(PX / f"{set_name}.npz"); res = int(D["res"]); nv = int(D["nv"]); acc = {}; acc_all = {}
        for v in range(nv):
            g = lambda k: D[f"v{v}_{k}"]  # noqa: E731
            valid = g("valid").astype(bool); mk = g("mk")
            rec.setdefault("pixels", {}).setdefault(set_name, []).append({"v": v, "alpha": int(len(mk)), "valid": int(valid.sum())})
            yx = np.stack([mk // res, mk % res], 1)                                           # nearest traced pixel for the all-pixel row
            nn_idx = cKDTree(yx[valid]).query(yx[~valid])[1] if (~valid).any() else np.zeros(0, np.int64)
            px = {k: torch.as_tensor(g(k)[valid].astype(np.float32 if k in ("pos", "nrm", "cam", "vis") else np.int64), device=dev)
                  for k in ("pos", "nrm", "cam", "vis", "dom", "port")}
            tg = {"hist": g("tgt")[valid]}
            for t_name in ("r1", "first_r1"):
                if refs[t_name][set_name] is not None:
                    tg[t_name] = refs[t_name][set_name][f"v{v}_rgb"][valid]
            masks = {"all": np.ones(int(valid.sum()), bool)}
            if set_name == "heldout_T1":
                for r_ in ("A_changed", "A_weak", "A_unchanged", "A_uncorresponded", "A_vis_changed"):
                    masks[r_] = z[f"heldout_T1_v{v}_reg_{r_}"].ravel()[mk][valid]
                masks["B"] = g("dom")[valid] == 1
            mat = g("mat")[valid].astype(np.int64); vis = g("vis")[valid].astype(bool); hf = hf_mask(g("tgt"), mk, res)[valid]
            for n, (rk, vk) in arms.items():
                Vf = pv[f"{set_name}_v{v}_{vk}"].astype(np.float32)
                S = SV.State(T, rel[rk] if rk else np.full((T.N, C.K_GATHER), -1), Vf, g("light"))
                if rk:
                    S.solve()
                else:
                    S.iters, S.status, S.hist, S.ratio, S.solve_s = 0, "coupling off", [], [], 0.0
                    Eind, h = S.aggregates(); S.Eind, S.h = Eind, h; S.outgoing(Eind, h)
                t = time.time(); pr, parts = SV.render_pixels(T, S, px); torch.cuda.synchronize(); tr = time.time() - t
                pr = pr.cpu().numpy()
                preds[(set_name, n, v)] = pr
                views[f"{set_name}_v{v}_{n}"] = np.concatenate([pr, parts.reshape(len(pr), -1).cpu().numpy()], 1).astype(np.float16)
                views[f"{set_name}_v{v}_vidx"] = mk[valid].astype(np.int32)                 # for r1_exports (reviewer panels)
                if n != "K_only":                                                           # declared extra row: all alpha>0 pixels, hist target
                    pr_all = np.empty((len(mk), 3), np.float32); pr_all[valid] = pr; pr_all[~valid] = pr[nn_idx]
                    e_all = ((C.ldr3(pr_all) - C.ldr3(g("tgt"))) ** 2).mean(1); acc_all.setdefault(n, []).append((float(e_all.sum()), int(len(e_all)), float(e_all.mean())))
                for t_name, tgt in tg.items():
                    if n == "K_only" and t_name != "first_r1":
                        continue
                    if n != "K_only" and t_name == "first_r1":
                        continue
                    q = acc.setdefault((n, t_name), []); q.append(metrics(pr, tgt, masks, mat, vis, hf))
                rec["per_view"].setdefault(set_name, {}).setdefault(n, []).append({"v": v, "status": S.status, "iters": S.iters, "solve_s": S.solve_s, "render_s": tr,
                                                                                   "resid_hist": S.hist, "ratio_hist": S.ratio,
                                                                                   "spec_max": getattr(S, "spec_max", 0.0), "spec_clamped": getattr(S, "spec_clamped", 0)})
                print(set_name, v, n, S.status, S.iters, flush=True)
                del S
        rows = {}
        for (n, t_name), lst_all in acc.items():
            sq_all = np.array([x["mse"] for x in lst_all]); ok = np.isfinite(sq_all); lst = [x for x, o_ in zip(lst_all, ok) if o_]
            pv_psnr = [float(10 * np.log10(1 / q)) if np.isfinite(q) and q > 0 else None for q in sq_all]
            if not lst:                                                                     # every view non-finite (diverged): nothing to pool
                rows[f"{n}|{t_name}"] = {"view_mean_psnr": float("nan"), "per_view_psnr": pv_psnr, "views_pooled": 0, "views_excluded_nonfinite": int((~ok).sum())}
                continue
            sq = sq_all[ok]
            reg = {k: [sum(x["reg"][k][0] for x in lst), sum(x["reg"][k][1] for x in lst)] for k in lst[0]["reg"]}
            matt = np.array([x["mat"] for x in lst]).sum(0)
            rows[f"{n}|{t_name}"] = {"view_mean_psnr": float(np.mean(10 * np.log10(1 / sq))), "per_view_psnr": pv_psnr,
                                     "views_pooled": int(ok.sum()), "views_excluded_nonfinite": int((~ok).sum()),
                                     "pooled_psnr": float(10 * np.log10(reg["all"][1] / reg["all"][0])),
                                     "logl1": sum(x["logl1_sum"] for x in lst) / sum(x["n3"] for x in lst), "hf_abs": float(np.mean([x["hf"] for x in lst])),
                                     "regions": {k: {"mse": (v_[0] / v_[1]) if v_[1] else None, "px": v_[1]} for k, v_ in reg.items()},
                                     "materials": {(C.MATERIALS[i] if i < 8 else "B"): {"mse": float(matt[i, 0] / matt[i, 1]) if matt[i, 1] else None, "px": int(matt[i, 1])} for i in range(9)},
                                     "lit_mse": sum(x["lit"][0] for x in lst) / max(sum(x["lit"][1] for x in lst), 1),
                                     "shadow_mse": sum(x["shadow"][0] for x in lst) / max(sum(x["shadow"][1] for x in lst), 1)}
        for n, lst in acc_all.items():
            m = np.array([x[2] for x in lst]); ok = np.isfinite(m)
            rows[f"{n}|hist_allpx_nnfill"] = {"view_mean_psnr": float(np.mean(10 * np.log10(1 / m[ok]))) if ok.any() else float("nan"),
                                              "pooled_psnr": float(10 * np.log10(sum(x[1] for x, o_ in zip(lst, ok) if o_) / max(sum(x[0] for x, o_ in zip(lst, ok) if o_), 1e-30))),
                                              "views_pooled": int(ok.sum()), "note": "all alpha>0 pixels; untraced pixels take the prediction of the nearest traced pixel (declared extra row, not a protocol input)"}
        rec["sets"][set_name] = rows
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        (out / "eval_partial.json").write_text(json.dumps(rec, indent=1, default=float))
    np.savez_compressed(out / "views.npz", **views)
    # signed change on Lego pixels present in both views
    D0 = np.load(PX / "heldout_T0.npz"); D1 = np.load(PX / "heldout_T1.npz")
    for t_name in ("hist", "r1"):
      try:
        acc = {"A_changed": [], "stationary": []}; skipped = []
        for v in range(int(D1["nv"])):
            mk0, mk1 = D0[f"v{v}_mk"], D1[f"v{v}_mk"]; va0, va1 = D0[f"v{v}_valid"].astype(bool), D1[f"v{v}_valid"].astype(bool)
            lego0 = (D0[f"v{v}_dom"] == 0) & va0; lego1 = (D1[f"v{v}_dom"] == 0) & va1
            common, i0, i1 = np.intersect1d(mk0[lego0], mk1[lego1], return_indices=True)
            if t_name == "hist":
                r0 = D0[f"v{v}_tgt"][lego0][i0]; r1_ = D1[f"v{v}_tgt"][lego1][i1]
            else:
                if refs["r1"]["heldout_T0"] is None or refs["r1"]["heldout_T1"] is None:
                    continue
                r0 = refs["r1"]["heldout_T0"][f"v{v}_rgb"][lego0][i0]; r1_ = refs["r1"]["heldout_T1"][f"v{v}_rgb"][lego1][i1]
            p0 = preds[("heldout_T0", "S_T0", v)][np.cumsum(va0)[lego0] - 1][i0]          # predictions exist only for valid pixels
            pu = preds[("heldout_T1", "E_upd", v)][np.cumsum(va1)[lego1] - 1][i1]; ps = preds[("heldout_T1", "D_stale", v)][np.cumsum(va1)[lego1] - 1][i1]
            if not (np.isfinite(p0).all() and np.isfinite(pu).all() and np.isfinite(ps).all()):
                skipped.append(v); continue                                                 # a non-finite (diverged) view would poison the pools
            chg = z[f"heldout_T1_v{v}_reg_A_changed"].ravel()[common]
            lum = lambda x: x @ np.array([0.2126, 0.7152, 0.0722])  # noqa: E731
            dr, du, ds = lum(r1_ - r0), lum(pu - p0), lum(ps - p0)
            for key, m in (("A_changed", chg), ("stationary", ~chg)):
                acc[key].append((dr[m], du[m], ds[m]))
        sc = {}
        drs = np.concatenate([x[0] for x in acc["stationary"]]) if acc["stationary"] else np.zeros(0)
        sigma = float(1.4826 * np.median(np.abs(drs - np.median(drs)))) if len(drs) else 0.0; thr = 3 * sigma   # T0/T1 reference noise: MAD of the stationary pool
        for key, lst in acc.items():
            if not lst:
                continue
            dr = np.concatenate([x[0] for x in lst]); du = np.concatenate([x[1] for x in lst]); ds = np.concatenate([x[2] for x in lst])
            big = np.abs(dr) > thr
            f = lambda d: {"corr": float(np.corrcoef(d, dr)[0, 1]) if np.std(d) > 0 else None,  # noqa: E731
                           "slope": float((d * dr).sum() / max((dr * dr).sum(), 1e-30)),
                           "sign_agree_where_ref_change_gt_noise": float((np.sign(d[big]) == np.sign(dr[big])).mean()) if big.any() else None,
                           "explained_fraction": float(1 - ((d - dr) ** 2).sum() / max((dr ** 2).sum(), 1e-30))}
            sc[key] = {"px": int(len(dr)), "noise_sigma_stationary_mad": sigma, "ref_change_threshold": float(thr), "px_ref_change_gt_threshold": int(big.sum()),
                       "E_upd": f(du), "D_stale": f(ds)}
        sc["skipped_views_nonfinite"] = skipped
        rec["signed_change"][t_name] = sc
      except Exception:                                                                     # keep the arms' results even if this block fails
        rec["signed_change"][t_name] = {"error": traceback.format_exc()}
    rec["kernel_parameters_bit_identical"] = M.digest(KA, KB) == dig0
    rec["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    (out / "eval.json").write_text(json.dumps(rec, indent=1, default=float))
    for s_, rows in rec["sets"].items():
        print(s_, {n: round(r["view_mean_psnr"], 3) for n, r in rows.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--kernels"); ap.add_argument("--out")
    a = ap.parse_args()
    prep() if a.cmd == "prep" else run(a)
