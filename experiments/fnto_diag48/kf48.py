"""WL48 Probe C secondary - known-field C transfer at kappa 0 / 2 / 8: decomposition of the worklog-46 R2 residual.

Same emit46 Blender oracle, views and 20,000 non-Yellow pixels as kf46_eval (seed 4602). Arms (protocol probe_C kf):
  K1  worklog 46 R2: receiver-port gathers (rel_T0), field at the hit PORT origin, port normal
  K2  same rays re-traced: field f_s and true triangle normal at the exact HIT POINT        (removes sender port aggregation)
  K3  128 cosine rays from the PIXEL's own position / normal, exact hit field               (removes receiver aggregation)
  K4  as K3 with 2048 rays                                                                  (limited quadrature control)
Misses and inside faces (-2) carry 0 in every arm.

    source mi_env.sh; <8dna26 venv python> kf48.py <emit46 dir> <out dir>
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_dir46")); sys.path.insert(0, str(HERE.parent / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parent / "fnto_lego"))
import r1_common as C  # noqa: E402
import r1_relations as RR  # noqa: E402
import fnto_relations as R  # noqa: E402
from kf46_eval import f_s, g  # noqa: E402


def tri_normals(geo, Va, Fa, Vb, Fb, ok, prim, sid):
    n = np.zeros((len(ok), 3))
    ma = ok & (sid == geo.names.index("A")); mb = ok & (sid == geo.names.index("B")) if "B" in geo.names else np.zeros_like(ok)
    for m, V, F in ((ma, Va, Fa), (mb, Vb, Fb)):
        if m.any():
            tri = prim[m]; cr = np.cross(V[F[tri, 1]] - V[F[tri, 0]], V[F[tri, 2]] - V[F[tri, 0]]); n[m] = cr / np.linalg.norm(cr, axis=1, keepdims=True)
    return n


def pixel_rays(pos, nrm, k, rng):
    rot = rng.random(len(pos)) * 2 * np.pi
    t, b, n = C.frames(np.repeat(nrm, k, 0), np.repeat(rot, k))
    L = C.cosine_dirs_local(k)[np.tile(np.arange(k), len(pos))]
    d = (L[:, 0:1] * t + L[:, 1:2] * b + L[:, 2:3] * n)
    o = np.repeat(pos + C.EPS1 * nrm, k, 0)
    return o, d


def main(edir, out, views=(6, 15), kappas=(0, 2, 8), npx=20000):
    edir, out = Path(edir), Path(out); out.mkdir(parents=True, exist_ok=True)
    A = RR.load(); pw = A["pos"].astype(np.float64); nq_all = A["nrm"].astype(np.float64); fs_port = f_s(pw)
    rel = np.load(C.OUT1 / "rel_T0.npz"); H, Tt = rel["hit"], rel["t"]
    px = np.load(C.OUT1 / "eval_px/heldout_T0.npz"); rng = np.random.default_rng(4602); rng2 = np.random.default_rng(4804); geo = R.Geo(*C.offsets("T0"))
    Va, Fa = R.mesh_a(); bx = np.load(C.BOX_T0); Vb, Fb = bx["V"].astype(np.float64), bx["F"]
    rows = {k: [] for k in ("view", "near", "inside", "E_in", "E_in_rep", "K1", "K2", "K3", "K4", "kappa", "K3_inside_frac")}
    for v in views:
        val = px[f"v{v}_valid"].astype(bool); dom = px[f"v{v}_dom"]; mat = np.where(dom == 1, 8, px[f"v{v}_mat"])
        cand = np.flatnonzero(val & (mat != 0)); sel = np.sort(rng.choice(cand, min(npx, len(cand)), replace=False))     # = kf46_eval selection
        prt = px[f"v{v}_port"][sel].astype(np.int64); up, inv = np.unique(prt, return_inverse=True)
        o, d = RR.ray_setup(A, up); h = H[up]; t = Tt[up].astype(np.float64); hc = np.clip(h, 0, None)
        ok, p, prim, sid, tt = geo.intersect(o.reshape(-1, 3), d.reshape(-1, 3))
        ntri = tri_normals(geo, Va, Fa, Vb, Fb, ok, prim, sid).reshape(*h.shape, 3); p = p.reshape(*h.shape, 3); okr = ok.reshape(h.shape)
        live = h >= 0; cq = np.abs((nq_all[hc] * -d).sum(-1)); ct = np.abs((ntri * -d).sum(-1))
        fs_port_v = np.where(live, fs_port[hc], 0.0); fs_hit = np.where(live & okr, f_s(p), 0.0)
        near = ((t < 3e-4) & live).mean(1)[inv]; inside = (h == -2).mean(1)[inv]
        ppos = px[f"v{v}_pos"][sel].astype(np.float64); pn = px[f"v{v}_nrm"][sel].astype(np.float64); pn /= np.linalg.norm(pn, axis=1, keepdims=True)
        K34 = {}
        for arm, k in (("K3", 128), ("K4", 2048)):
            vals = {kap: np.zeros(len(sel)) for kap in kappas}; insf = np.zeros(len(sel))
            for s0 in range(0, len(sel), max(1, 400000 // k)):
                s1 = min(s0 + max(1, 400000 // k), len(sel)); oo, dd = pixel_rays(ppos[s0:s1], pn[s0:s1], k, rng2)
                okk, pp, pr_, sd, _ = geo.intersect(oo, dd); port = RR.map_hits(A, geo, okk, pp, pr_, sd, dd)
                nt = tri_normals(geo, Va, Fa, Vb, Fb, okk, pr_, sd); lv = port >= 0; ctt = np.abs((nt * -dd).sum(-1))
                fsv = np.where(lv, f_s(pp), 0.0)
                insf[s0:s1] = (port == -2).reshape(-1, k).mean(1)
                for kap in kappas:
                    vals[kap][s0:s1] = (math.pi / k) * (fsv * g(ctt, kap)).reshape(-1, k).sum(1)
            K34[arm] = vals; K34[arm + "_ins"] = insf
        for kappa in kappas:
            fe = edir / f"emit_k{kappa}_v{v}.npz"; fo = edir / f"emitonly_k{kappa}_v{v}.npz"
            if not fe.exists():
                continue
            e = np.load(fe)["rgb"].astype(np.float64); eo = np.load(fo)["rgb"].astype(np.float64); E_in = np.pi * (e[sel] - eo[sel]).mean(1)
            fr = edir / f"emit_k{kappa}_v{v}_rep.npz"
            E_rep = np.pi * (np.load(fr)["rgb"].astype(np.float64)[sel] - eo[sel]).mean(1) if fr.exists() else np.full(len(sel), np.nan)
            w = math.pi / H.shape[1]
            k1 = (w * (fs_port_v * g(cq, kappa)).sum(1))[inv]; k2 = (w * (fs_hit * g(ct, kappa)).sum(1))[inv]
            for k_, arr in (("view", np.full(len(sel), v)), ("near", near), ("inside", inside), ("E_in", E_in), ("E_in_rep", E_rep), ("K1", k1), ("K2", k2),
                            ("K3", K34["K3"][kappa]), ("K4", K34["K4"][kappa]), ("kappa", np.full(len(sel), kappa)), ("K3_inside_frac", K34["K3_ins"])):
                rows[k_].append(arr)
        print("view", v, flush=True)
    d_ = {k: np.concatenate(x) for k, x in rows.items()}

    def st(a, b, m):
        a, b = a[m], b[m]
        return {"px": int(m.sum()), "pooled_ratio": float(a.sum() / b.sum()), "median_abs_rel": float(np.median(np.abs(a - b) / np.maximum(b, 1e-6))), "corr": float(np.corrcoef(a, b)[0, 1])}
    rec = {"per_kappa": {}}
    for kappa in kappas:
        mk = d_["kappa"] == kappa; clean = mk & (d_["near"] <= 0.05) & (d_["inside"] <= 0.10) & (d_["K3_inside_frac"] <= 0.10)
        r = {arm: {"all": st(d_[arm], d_["E_in"], mk), "clean": st(d_[arm], d_["E_in"], clean)} for arm in ("K1", "K2", "K3", "K4")}
        rep = mk & np.isfinite(d_["E_in_rep"])
        if rep.any():
            r["oracle_replicate"] = {"all": st(d_["E_in_rep"], d_["E_in"], rep), "clean": st(d_["E_in_rep"], d_["E_in"], rep & clean)}
        rec["per_kappa"][f"kappa{kappa}"] = r
    k8 = rec["per_kappa"].get("kappa8")
    if k8:
        crit = {}
        for a, b in (("K1", "K2"), ("K2", "K3"), ("K3", "K4")):
            for sub in ("all", "clean"):
                crit[f"{a}->{b}_{sub}"] = {"d_corr": k8[b][sub]["corr"] - k8[a][sub]["corr"], "d_median_abs_rel": k8[b][sub]["median_abs_rel"] - k8[a][sub]["median_abs_rel"],
                                           "contributor": bool(k8[b][sub]["corr"] - k8[a][sub]["corr"] >= 0.02 or k8[a][sub]["median_abs_rel"] - k8[b][sub]["median_abs_rel"] >= 0.02)}
        rec["criteria_kappa8"] = crit
    rec["rays"] = {"K3_per_px": 128, "K4_per_px": 2048, "pixels_per_view": npx}
    (out / "kf48.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1)[:5000], flush=True)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main(*sys.argv[1:3])
