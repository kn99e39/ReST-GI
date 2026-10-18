"""Architecture Batch 4 evaluation (worklog 37; RNA venv, GPU). Protocol b4_v1.

    python b4_eval.py --config <yml> --oracle <oracle.json> --out <dir> --h 0.0045 \
        --set <name>:<state>:<ref_s1.h5>:<ref_s2.h5>[:<pair_T0_s1.h5>:<pair_T0_s2.h5>] ... \
        --plain RNA_T1=<ckpt>@T1 RNA_T0=<ckpt>@T0 --owned SO_T0=<dir> A_full=<dir> B_sel=<dir> \
        [--influence B_sel:SO_T0] [--keep-views 0 5]

Reference = mean of the two realisations (4096 spp); inputs (AOVs, visibility) from realisation 1. Owned arms take
canonical positions (b4_model) and are evaluated with hard selection and with the worklog-34/36 PU composition (the
qualified composition, primary). '@state' restricts a plain arm to sets of that state (a plain RNA model has no
canonical frame). Owned arms are evaluated on every set; SO_T0 on a T1 set is arm C_frozen.

Metrics: the worklog-36 set (view-mean/pooled PSNR on the code tonemap, logL1, hf |e|, per unit, per material, seam J,
band/off-band) plus region display MSE and |e| on: D*-owned, V*-owned, V* pixels inside a D* unit's PU band, physically
changed / weak / unchanged / uncorresponded A pixels and visibility-changed A pixels (from the paired T0 references), and
B pixels. Influence: output changes of arm X vs arm Y inside / outside the predicted support of the units X retrained.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership_lego"))
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership"))
import b3_compose as C  # noqa: E402
import b4_model as E  # noqa: E402
import b4_oracle as O  # noqa: E402
import rso_model as M  # noqa: E402

COS20 = float(np.cos(np.deg2rad(20.0)))
TAU = O.TAU
K = 17
REGIONS = ["all", "Dstar_owned", "Vstar_owned", "Vstar_in_Dstar_band", "A_changed", "A_weak", "A_unchanged", "A_uncorresponded", "A_vis_changed", "B"]


def ldr3(x):
    x = np.clip(x, 0, None)
    return (x / (1 + x)) ** (1 / 2.2)


def load_plain(path, conf, dev):
    import torch
    from rna import models

    sd = torch.load(path, map_location=dev, weights_only=False)
    if "state_dict" in sd:
        sd = {k[6:]: v for k, v in sd["state_dict"].items() if k.startswith("model.")}
    m = models.NeuralSurfaceTriPlane(**dict(conf.model_params)).to(dev); m.load_state_dict(sd); m.eval(); m.set_iteration_index(10 ** 9)
    return m


def box(x, r=2):
    """Sum over the (2r+1)^2 neighbourhood (zero padding) of a (res, res[, c]) array."""
    p = np.pad(x, [(r, r), (r, r)] + [(0, 0)] * (x.ndim - 2)); c = np.cumsum(np.cumsum(p, 0), 1)
    c = np.pad(c, [(1, 0), (1, 0)] + [(0, 0)] * (x.ndim - 2)); n = 2 * r + 1
    return c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--oracle", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--h", type=float, required=True); ap.add_argument("--set", nargs="+", required=True)
    ap.add_argument("--plain", nargs="*", default=[]); ap.add_argument("--owned", nargs="*", default=[])
    ap.add_argument("--influence", nargs="*", default=[]); ap.add_argument("--keep-views", type=int, nargs="*", default=[])
    args = ap.parse_args()
    import h5py
    import rna
    import rna.config
    import torch

    sys.modules.setdefault("neumat", rna); sys.modules.setdefault("neumat.config", rna.config)
    from rna import config

    dev = "cuda"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    conf = config.get_neumat_config(config_filename=str(Path(args.config).resolve()))
    orc = json.loads(Path(args.oracle).read_text()); Dstar = set(orc["D_star"])
    mesh_mat = np.concatenate([np.load(E.ROOT / "results/rna_surface_ownership_lego/mesh/lego_mesh.npz")["material"], np.full(12, 8)])
    part = E.ROOT / "results/rna_surface_ownership_lego/partition_L16.npz"
    pz = np.load(part); bm = np.load(E.ROOT / "results/rna_selective_reconstruction_lego/scene/B_box_T0.npz")
    centres = np.concatenate([pz["centres"], bm["V"].mean(0, keepdims=True)])
    labels = np.concatenate([pz["label"], [E.B_UNIT]]); comps = np.concatenate([pz["comp"], [E.B_COMP]])
    arms, timing, plain_state = {}, {}, {}
    for spec in args.plain:
        n, p = spec.split("=", 1); p, st = p.split("@"); t0 = time.time()
        arms[n] = ("plain", load_plain(p, conf, dev)); plain_state[n] = st; timing[n] = {"load_s": time.time() - t0, "inf": []}
    for spec in args.owned:
        n, d = spec.split("=", 1); d = Path(d); t0 = time.time()
        specs = [{"params": s["params"], "lo": np.array(s["lo"]), "hi": np.array(s["hi"])} for s in json.loads((d / "specs.json").read_text())]
        sd = torch.load(d / "state.pt", map_location=dev)
        m = M.OwnedTriplaneSet(specs, (sd["dmin"].cpu().numpy(), sd["dmax"].cpu().numpy())).to(dev); m.load_state_dict(sd); m.eval(); m.set_iteration_index(10 ** 9)
        arms[n] = ("owned", C.LegoPUComposer(m, centres, labels, comps, args.h)); timing[n] = {"load_s": time.time() - t0, "inf": []}; timing[n + "-PU"] = {"inf": []}
    T = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev).unsqueeze(0)
    record = {"protocol": "b4_v1", "h": args.h, "D_star": sorted(Dstar), "sets": {},
              "params": {a: int(sum(p.numel() for p in (o.model.parameters() if k == "owned" else o.parameters()))) for a, (k, o) in arms.items()}}
    owners = {}
    keep = {}
    for sspec in args.set:
        parts = sspec.split(":"); sname, state, r1, r2 = parts[:4]; pair = parts[4:6] if len(parts) >= 6 else None
        if state not in owners:
            owners[state] = E.make_owner(state)
        own_s = owners[state]
        f1, f2 = h5py.File(r1, "r"), h5py.File(r2, "r")
        g1, g2 = (h5py.File(pair[0], "r"), h5py.File(pair[1], "r")) if pair else (None, None)
        if pair and "T0" not in owners:
            owners["T0"] = E.make_owner("T0")
        res = int(f1.attrs["resolution"][0]); nv = f1["color"].shape[0]
        aabb = (np.array(f1.attrs["aabb_min"], np.float64), np.array(f1.attrs["aabb_max"], np.float64))
        names = [a for a, (k, _) in arms.items() if k == "owned" or plain_state[a] == state]
        names += [a + "-PU" for a, (k, _) in arms.items() if k == "owned"]
        acc = {n: {"sq": [], "npx": [], "logl1": 0.0, "nall": 0, "hf": [], "unit": np.zeros((K, 3)), "mat": np.zeros((9, 2)),
                   "J_seam": [], "e_seam": [], "J_int": [], "e_int": [], "band": [], "off": [], "reg": {r: np.zeros(3) for r in REGIONS}} for n in names}
        dref_s, dref_i = [], []
        infl = {s: {"changed": 0, "predicted": 0, "outside_changed": 0, "owned_px": 0, "nbr_band_px": 0, "hard_outside_changed": 0} for s in args.influence}
        reg_px = {r: 0 for r in REGIONS}
        for v in range(nv):
            g = lambda f, k: np.nan_to_num(f[k][v].astype(np.float32))
            alpha = g(f1, "alpha")[:, 0]; m = alpha > 0
            P, N, Cd, Lt, vis = g(f1, "position"), g(f1, "normal"), g(f1, "camera_dir"), g(f1, "light_dir"), g(f1, "diffuse_direct")[:, 0] > 0
            tgt = 0.5 * (g(f1, "color") + g(f2, "color"))
            lab, _, (tri, comp, onsurf) = own_s(P[m].astype(np.float64), N[m].astype(np.float64), return_tri=True)
            canP = own_s.canonical(P[m].astype(np.float64), lab); canQ = own_s.canonical(onsurf, lab)
            own = -np.ones(res * res, np.int64); own[m] = lab
            cmp_ = -np.ones(res * res, np.int64); cmp_[m] = comp
            mat = -np.ones(res * res, np.int64); mat[m] = mesh_mat[tri]
            # physical regions from the paired T0 references (same camera and light)
            reg = {r: np.zeros(res * res, bool) for r in REGIONS}; reg["all"] = m.copy()
            if pair:
                a0 = (g(g1, "alpha")[:, 0] >= 0.999) & (g(g2, "alpha")[:, 0] >= 0.999); P0 = g(g1, "position"); N0 = g(g1, "normal"); vis0 = g(g1, "diffuse_direct")[:, 0] > 0
                l0 = -np.ones(res * res, np.int64); c0 = np.full((res * res, 3), np.nan)
                ll, _ = owners["T0"](P0[a0].astype(np.float64), N0[a0].astype(np.float64)); l0[a0] = ll; c0[a0] = owners["T0"].canonical(P0[a0].astype(np.float64), ll)
                c1 = np.full((res * res, 3), np.nan); c1[m] = canP
                a1 = m & (alpha >= 0.999) & (g(f2, "alpha")[:, 0] >= 0.999)
                isA = (own >= 0) & (own < E.B_UNIT)
                corr = a1 & a0 & isA & (l0 == own) & (np.linalg.norm(np.nan_to_num(c1 - c0, nan=1.0), axis=1) < 1e-3)
                d1 = ldr3(g(f1, "color")) - ldr3(g(g1, "color")); d2 = ldr3(g(f2, "color")) - ldr3(g(g2, "color"))
                db, e = (0.5 * (d1 + d2)).mean(1), (0.5 * (d1 - d2)).mean(1)
                cm = corr.reshape(res, res).astype(float); nb = np.maximum(box(cm), 1)
                Dbox = box((db * corr).reshape(res, res)) / nb; Sbox = np.sqrt(box((e ** 2 * corr).reshape(res, res))) / nb
                Dbox, Sbox = np.abs(Dbox).ravel(), Sbox.ravel()
                reg["A_changed"] = corr & (Dbox > np.maximum(2 * TAU, 4 * Sbox)); reg["A_unchanged"] = corr & (Dbox < TAU)
                reg["A_weak"] = corr & ~reg["A_changed"] & ~reg["A_unchanged"]; reg["A_uncorresponded"] = m & isA & ~corr
                reg["A_vis_changed"] = corr & (vis0 != vis)
            reg["B"] = own == E.B_UNIT
            reg["Dstar_owned"] = m & np.isin(own, list(Dstar)); reg["Vstar_owned"] = m & (own >= 0) & ~np.isin(own, list(Dstar))
            preds, wfull, nbfull = {}, None, None
            for a, (kind, obj) in arms.items():
                if a not in names:
                    continue
                full = np.zeros((res * res, 3), np.float32)
                if kind == "plain":
                    pos = (P[m] - aabb[0]) / (aabb[1] - aabb[0])
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    with torch.no_grad():
                        pr = torch.cat([obj(T(pos[s:s + (1 << 19)]), T(Cd[m][s:s + (1 << 19)]), T(Lt[m][s:s + (1 << 19)]), T(N[m][s:s + (1 << 19)])).squeeze(0)
                                        for s in range(0, len(pos), 1 << 19)])
                    rgb = torch.where(torch.tensor(vis[m], device=dev)[:, None], pr[:, :3], pr[:, 3:]).cpu().numpy()
                    torch.cuda.synchronize(); timing[a]["inf"].append(time.perf_counter() - t0)
                    full[m] = rgb; preds[a] = full
                else:
                    t0 = time.perf_counter()
                    pu, prt = obj(canP.astype(np.float32), Cd[m], Lt[m], N[m], vis[m], lab, comp, canQ, return_parts=True)
                    timing[a + "-PU"]["inf"].append(time.perf_counter() - t0)
                    hard = np.zeros((res * res, 3), np.float32); hard[m] = prt["hard"]; preds[a] = hard
                    full[m] = pu; preds[a + "-PU"] = full
                    if wfull is None:
                        wfull = np.zeros(res * res); wfull[m] = prt["w"]; nbfull = -np.ones(res * res, np.int64); nbfull[m] = prt["nbr"]
            if wfull is not None:
                reg["Vstar_in_Dstar_band"] = reg["Vstar_owned"] & np.isin(nbfull, list(Dstar)) & (wfull > 0)
            for r in REGIONS:
                reg_px[r] += int(reg[r].sum())
            Lr = ldr3(tgt).mean(-1)
            clean = alpha >= 0.999
            img = np.arange(res * res).reshape(res, res); pp, qq = [], []
            for dy, dx in ((0, 1), (1, 0)):
                a_ = img[: res - dy, : res - dx].ravel(); b_ = img[dy:, dx:].ravel(); ok = clean[a_] & clean[b_]; pp.append(a_[ok]); qq.append(b_[ok])
            p_, q_ = np.concatenate(pp), np.concatenate(qq)
            same_u = own[p_] == own[q_]; dist = np.linalg.norm(P[p_] - P[q_], axis=1); fpt = np.median(dist[same_u])
            okg = (cmp_[p_] == cmp_[q_]) & (dist <= 3 * fpt) & ((N[p_] * N[q_]).sum(1) >= COS20) & (vis[p_] == vis[q_])
            seam, inter = okg & ~same_u, okg & same_u
            dref = np.abs(Lr[p_] - Lr[q_]); dref_s.append(dref[seam]); dref_i.append(dref[inter])
            lum = Lr.reshape(res, res); mm = m.reshape(res, res); grad = np.zeros_like(lum)
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                okk = mm & np.roll(mm, (dy, dx), (0, 1)); grad = np.maximum(grad, np.where(okk, np.abs(lum - np.roll(lum, (dy, dx), (0, 1))), 0))
            gm = grad.reshape(-1)[m]; hf = np.zeros(res * res, bool); hf[np.flatnonzero(m)[gm >= np.quantile(gm, 0.9)]] = True
            band = m & (wfull > 0) if wfull is not None else np.zeros(res * res, bool)
            for n in names:
                pr = preds[n]; dl = ldr3(pr) - ldr3(tgt); e_ = np.abs(dl).mean(-1); sq_ = (dl ** 2).mean(-1)
                A_ = acc[n]
                A_["sq"].append(float(sq_[m].mean())); A_["npx"].append(int(m.sum()))
                A_["logl1"] += float(np.abs(np.log1p(np.clip(pr[m], 0, None)) - np.log1p(np.clip(tgt[m], 0, None))).sum()); A_["nall"] += int(m.sum()) * 3
                A_["hf"].append(e_[hf])
                np.add.at(A_["unit"], (own[m], 0), e_[m]); np.add.at(A_["unit"], (own[m], 1), 1); np.add.at(A_["unit"], (own[m], 2), sq_[m])
                np.add.at(A_["mat"], (mat[m], 0), e_[m]); np.add.at(A_["mat"], (mat[m], 1), 1)
                Lp = ldr3(pr).mean(-1); dp = np.abs(Lp[p_] - Lp[q_])
                A_["J_seam"].append(np.abs(dp - dref)[seam]); A_["J_int"].append(np.abs(dp - dref)[inter])
                A_["e_seam"].append(((e_[p_] + e_[q_]) / 2)[seam]); A_["e_int"].append(((e_[p_] + e_[q_]) / 2)[inter])
                A_["band"].append(e_[band]); A_["off"].append(e_[m & ~band])
                for r in REGIONS:
                    A_["reg"][r] += (float(sq_[reg[r]].sum()), float(e_[reg[r]].sum()), int(reg[r].sum()))
            for s in args.influence:
                X, Y = s.split(":"); xu = json.loads((Path([o for o in args.owned if o.startswith(X + "=")][0].split("=", 1)[1]) / "train.json").read_text())
                Dx = set(xu.get("args", {}).get("train_units") or [])
                a_, b_ = preds[X + "-PU"][m], preds[Y + "-PU"][m]
                pred_sup = np.isin(lab, list(Dx)) | (np.isin(nbfull[m], list(Dx)) & (wfull[m] > 0))
                ch = (a_ != b_).any(1)
                I = infl[s]; I["changed"] += int(ch.sum()); I["predicted"] += int(pred_sup.sum()); I["outside_changed"] += int((ch & ~pred_sup).sum())
                I["owned_px"] += int(np.isin(lab, list(Dx)).sum()); I["nbr_band_px"] += int((pred_sup & ~np.isin(lab, list(Dx))).sum())
                I["hard_outside_changed"] += int(((preds[X][m] != preds[Y][m]).any(1) & ~np.isin(lab, list(Dx))).sum())
            if v in args.keep_views:
                kv = f"{sname}_v{v}"
                keep[kv + "_ref"] = ldr3(tgt).reshape(res, res, 3).astype(np.float16)
                for n in names:
                    if n.endswith("-PU") or arms[n][0] == "plain":  # stored views: the primary (PU) owned arms and the RNA controls
                        keep[kv + "_" + n] = ldr3(preds[n]).reshape(res, res, 3).astype(np.float16)
                keep[kv + "_own"] = own.reshape(res, res).astype(np.int16)
                keep[kv + "_w"] = (wfull if wfull is not None else np.zeros(res * res)).reshape(res, res).astype(np.float16)
                for r in ("A_changed", "A_unchanged", "A_weak", "A_uncorresponded", "A_vis_changed", "Vstar_in_Dstar_band"):
                    keep[kv + "_reg_" + r] = reg[r].reshape(res, res)
                if pair:
                    keep[kv + "_gtdelta"] = (0.5 * (d1 + d2)).reshape(res, res, 3).astype(np.float16)
                    keep[kv + "_refT0"] = ldr3(0.5 * (g(g1, "color") + g(g2, "color"))).reshape(res, res, 3).astype(np.float16)
            print(sname, "view", v, flush=True)
        ds, di = np.concatenate(dref_s), np.concatenate(dref_i)
        edges = np.quantile(ds, np.linspace(0, 1, 11)); edges[0], edges[-1] = -np.inf, np.inf
        bs, bi = np.digitize(ds, edges[1:-1]), np.digitize(di, edges[1:-1])
        wi = (np.bincount(bs, minlength=10) / len(ds))[bi] / np.maximum((np.bincount(bi, minlength=10) / len(di))[bi], 1e-12)
        rows = {}
        for n in names:
            A_ = acc[n]; sq = np.array(A_["sq"]); npx = np.array(A_["npx"])
            rows[n] = {"view_mean_psnr": float(np.mean(-10 * np.log10(sq + 1e-8))), "pooled_psnr": float(-10 * np.log10((sq * npx).sum() / npx.sum())),
                       "per_view_psnr": (-10 * np.log10(sq + 1e-8)).tolist(), "logl1": A_["logl1"] / A_["nall"], "hf_abs": float(np.concatenate(A_["hf"]).mean()),
                       "seam_J": float(np.concatenate(A_["J_seam"]).mean()), "interior_J_matched": float(np.average(np.concatenate(A_["J_int"]), weights=wi)),
                       "seam_abs": float(np.concatenate(A_["e_seam"]).mean()), "interior_abs_matched": float(np.average(np.concatenate(A_["e_int"]), weights=wi)),
                       "band_abs": float(np.concatenate(A_["band"]).mean()) if sum(len(x) for x in A_["band"]) else None,
                       "offband_abs": float(np.concatenate(A_["off"]).mean()),
                       "per_unit_abs": (A_["unit"][:, 0] / np.maximum(A_["unit"][:, 1], 1)).tolist(), "per_unit_mse": (A_["unit"][:, 2] / np.maximum(A_["unit"][:, 1], 1)).tolist(),
                       "per_unit_px": A_["unit"][:, 1].astype(int).tolist(),
                       "per_material_abs": (A_["mat"][:, 0] / np.maximum(A_["mat"][:, 1], 1)).tolist(), "per_material_px": A_["mat"][:, 1].astype(int).tolist(),
                       "regions": {r: {"mse": A_["reg"][r][0] / max(A_["reg"][r][2], 1), "abs": A_["reg"][r][1] / max(A_["reg"][r][2], 1), "px": int(A_["reg"][r][2])} for r in REGIONS}}
        record["sets"][sname] = {"state": state, "refs": [r1, r2], "pair_T0": pair, "views": nv, "seam_pairs": int(len(ds)), "interior_pairs": int(len(di)),
                                 "region_px": reg_px, "rows": rows, "influence": {s: {**I, "changed_within_predicted": I["outside_changed"] == 0} for s, I in infl.items()}}
        print(sname, json.dumps({n: {k: round(v, 4) for k, v in r.items() if isinstance(v, float)} for n, r in rows.items()}), flush=True)
    record["timing"] = {n: {"load_s": t.get("load_s"), "inference_s_per_view": float(np.mean(t["inf"])) if t["inf"] else None} for n, t in timing.items()}
    (out / "eval.json").write_text(json.dumps(record, indent=1))
    if keep:
        np.savez_compressed(out / "views.npz", **keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
