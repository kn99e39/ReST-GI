"""Architecture Batch 3 evaluation on native Lego (worklog 36; RNA venv, GPU).

    python b3_eval.py --partition <L16.npz> --mesh <mesh.npz> --h 0.0045 --out <dir> \
        --sets heldout=<h5> dev=<h5> native40=<h5> --plain A=<ckpt|state> A_released=<ckpt> --owned B=<dir> REINIT=<dir>

Per set and arm (owned arms also under C = worklog-34 PU composition, b3_compose): view-mean PSNR (code tonemap,
alpha>0) and pooled; logL1; hf |e| (top-10% reference-gradient pixels per view); per unit; per material;
seam jump excess J and |e| on clean seam pairs vs |d ref|-matched clean interior pairs (full coverage, same
connected component, 3D gap <= 3 x the view footprint, normal angle <= 20 deg, same visibility); band (w>0)
vs off-band |e|; load and inference time. REINIT vs B under C: changed pixels vs predicted influence of u*.
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
sys.path.insert(0, str(HERE.parent / "rna_surface_ownership"))
import b3_compose as C  # noqa: E402
import b3_model as B  # noqa: E402
import rso_model as M  # noqa: E402

COS20 = float(np.cos(np.deg2rad(20.0)))


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True); ap.add_argument("--mesh", required=True); ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", required=True); ap.add_argument("--plain", nargs="*", default=[]); ap.add_argument("--owned", nargs="*", default=[])
    ap.add_argument("--u-star", type=int, default=None)
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
    owner = B.LegoOwner(args.partition, args.mesh)
    mesh_mat = np.load(args.mesh)["material"]
    arms, timing = {}, {}
    for spec in args.plain:
        n, p = spec.split("=", 1); t0 = time.time(); arms[n] = ("plain", load_plain(p, conf, dev)); timing[n] = {"load_s": time.time() - t0, "inf": []}
    for spec in args.owned:
        n, d = spec.split("=", 1); d = Path(d); t0 = time.time()
        specs = [{"params": s["params"], "lo": np.array(s["lo"]), "hi": np.array(s["hi"])} for s in json.loads((d / "specs.json").read_text())]
        sd = torch.load(d / "state.pt", map_location=dev)
        m = M.OwnedTriplaneSet(specs, (sd["dmin"].cpu().numpy(), sd["dmax"].cpu().numpy())).to(dev); m.load_state_dict(sd); m.eval(); m.set_iteration_index(10 ** 9)
        arms[n] = ("owned", C.LegoPUComposer.from_partition(m, args.partition, args.h)); timing[n] = {"load_s": time.time() - t0, "inf": []}; timing[n + "-PU"] = {"inf": []}
    names = [a for a in arms] + [a + "-PU" for a, (k, _) in arms.items() if k == "owned"]
    T = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev).unsqueeze(0)
    record = {"h": args.h, "sets": {}, "params": {a: int(sum(p.numel() for p in (o.model.parameters() if k == "owned" else o.parameters()))) for a, (k, o) in arms.items()}}
    reinit = None
    for sspec in args.sets:
        sname, h5p = sspec.split("=", 1)
        f = h5py.File(h5p, "r"); res = int(f.attrs["resolution"][0]); nv = f["color"].shape[0]
        aabb = (np.array(f.attrs["aabb_min"], np.float64), np.array(f.attrs["aabb_max"], np.float64))
        acc = {n: {"sq": [], "npx": [], "logl1": 0.0, "nall": 0, "hf": [], "unit": np.zeros((16, 2)), "mat": np.zeros((8, 2)),
                   "J_seam": [], "e_seam": [], "J_int": [], "e_int": [], "band": [], "off": []} for n in names}
        dref_s, dref_i = [], []
        for v in range(nv):
            g = lambda k: np.nan_to_num(f[k][v].astype(np.float32))
            alpha = g("alpha")[:, 0]; m = alpha > 0
            P, N, Cd, Lt, vis, tgt = g("position"), g("normal"), g("camera_dir"), g("light_dir"), g("diffuse_direct")[:, 0] > 0, g("color")
            lab, _, (tri, comp, onsurf) = owner(P[m].astype(np.float64), N[m].astype(np.float64), return_tri=True)
            own = -np.ones(res * res, np.int64); own[m] = lab
            cmp_ = -np.ones(res * res, np.int64); cmp_[m] = comp
            mat = -np.ones(res * res, np.int64); mat[m] = mesh_mat[tri]
            preds, wfull, nbfull = {}, None, None
            for a, (kind, obj) in arms.items():
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
                    pu, parts = obj(P[m], Cd[m], Lt[m], N[m], vis[m], lab, comp, onsurf, return_parts=True)
                    timing[a + "-PU"]["inf"].append(time.perf_counter() - t0)
                    hard = np.zeros((res * res, 3), np.float32); hard[m] = parts["hard"]; preds[a] = hard
                    full[m] = pu; preds[a + "-PU"] = full
                    if wfull is None:
                        wfull = np.zeros(res * res); wfull[m] = parts["w"]; nbfull = -np.ones(res * res, np.int64); nbfull[m] = parts["nbr"]
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
                pr = preds[n]; e = np.zeros(res * res); e[m] = np.abs(ldr3(pr[m]) - ldr3(tgt[m])).mean(-1)
                A_ = acc[n]
                A_["sq"].append(float(((ldr3(pr[m]) - ldr3(tgt[m])) ** 2).mean())); A_["npx"].append(int(m.sum()))
                A_["logl1"] += float(np.abs(np.log1p(np.clip(pr[m], 0, None)) - np.log1p(np.clip(tgt[m], 0, None))).sum()); A_["nall"] += int(m.sum()) * 3
                A_["hf"].append(e[hf])
                np.add.at(A_["unit"], (own[m], 0), e[m]); np.add.at(A_["unit"], (own[m], 1), 1)
                np.add.at(A_["mat"], (mat[m], 0), e[m]); np.add.at(A_["mat"], (mat[m], 1), 1)
                Lp = ldr3(pr).mean(-1); dp = np.abs(Lp[p_] - Lp[q_])
                A_["J_seam"].append(np.abs(dp - dref)[seam]); A_["J_int"].append(np.abs(dp - dref)[inter])
                A_["e_seam"].append(((e[p_] + e[q_]) / 2)[seam]); A_["e_int"].append(((e[p_] + e[q_]) / 2)[inter])
                A_["band"].append(e[band]); A_["off"].append(e[m & ~band])
            if "REINIT" in arms and "B" in arms and args.u_star is not None:
                reinit = reinit or {"changed": 0, "predicted": 0, "outside_changed": 0, "own_px": 0, "nbr_band_px": 0, "hard_outside_changed": 0, "sets": []}
                if sname not in reinit["sets"]:
                    reinit["sets"].append(sname)
                a_, b_ = preds["REINIT-PU"][m], preds["B-PU"][m]
                infl = (lab == args.u_star) | ((nbfull[m] == args.u_star) & (wfull[m] > 0))
                ch = (a_ != b_).any(1)
                reinit["changed"] += int(ch.sum()); reinit["predicted"] += int(infl.sum()); reinit["outside_changed"] += int((ch & ~infl).sum())
                reinit["own_px"] += int((lab == args.u_star).sum()); reinit["nbr_band_px"] += int((infl & (lab != args.u_star)).sum())
                reinit["hard_outside_changed"] += int(((preds["REINIT"][m] != preds["B"][m]).any(1) & (lab != args.u_star)).sum())
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
                       "per_unit_abs": (A_["unit"][:, 0] / np.maximum(A_["unit"][:, 1], 1)).tolist(), "per_unit_px": A_["unit"][:, 1].astype(int).tolist(),
                       "per_material_abs": (A_["mat"][:, 0] / np.maximum(A_["mat"][:, 1], 1)).tolist(), "per_material_px": A_["mat"][:, 1].astype(int).tolist()}
        record["sets"][sname] = {"h5": h5p, "views": nv, "seam_pairs": int(len(ds)), "interior_pairs": int(len(di)), "rows": rows}
        print(sname, json.dumps({n: {k: round(v, 4) for k, v in r.items() if isinstance(v, float)} for n, r in rows.items()}), flush=True)
    if reinit:
        reinit["changed_set_within_predicted"] = reinit["outside_changed"] == 0
        record["reinit_output_influence"] = reinit
    record["timing"] = {n: {"load_s": t.get("load_s"), "inference_s_per_view": float(np.mean(t["inf"])) if t["inf"] else None} for n, t in timing.items()}
    (out / "eval.json").write_text(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
