"""WL46 KF (axes A / C, known directional field): frozen C relation table vs the emit46 Blender oracle.

For non-Yellow pixels of held-out T0 views, the frozen rel_T0 gathers of the pixel's port read the emitted radiance of the
hit port in three ways (identical rays, hits and spatial field f_s at the hit port's origin; production geometry):
  R1   value at the port normal (production sender semantics): f_s(q) g_kappa(1)
  R2   value toward the receiver: f_s(q) g_kappa(|n_q . (-d)|)                    (n_q = port normal)
  R2t  diagnostic: f_s(q) g_kappa(|n_tri . (-d)|) with the hit triangle's geometric normal (re-intersected with Mitsuba;
       the renderer's True Normal) - separates the port-normal approximation from the directional semantics
E_C = pi / 128 sum_k value; misses and inside faces carry 0. Oracle E_in = pi (P_emit - P_emitonly). Statistics as
worklog 45 B2 (pooled ratio, median |rel|, corr) per kappa, per material, near-hit / inside classes; replicate (seed + 100)
on kappa 8 view 6 gives the oracle noise.

    source mi_env.sh; <8dna26 venv python> kf46_eval.py <emit46 dir> <out dir> [views=6,15] [kappas=0,2,8]
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parent / "fnto_lego"))
import r1_common as C  # noqa: E402
import r1_relations as RR  # noqa: E402
import fnto_relations as R  # noqa: E402

MATS = C.MATERIALS[:8] + ["box"]


def f_s(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return np.maximum(0.05, 1 + 0.6 * np.sin(2 * np.pi * x / 0.4) * np.sin(2 * np.pi * y / 0.4) + 0.3 * np.sin(2 * np.pi * z / 0.25))


def g(c, kappa):
    return (kappa + 2) / 2.0 * np.abs(c) ** kappa


def main(edir, out, views=(6, 15), kappas=(0, 2, 8), npx=20000):
    edir, out = Path(edir), Path(out); out.mkdir(parents=True, exist_ok=True)
    A = RR.load(); N = len(A["obj"]); pw = A["pos"].astype(np.float64).copy(); nq_all = A["nrm"].astype(np.float64)
    fs_port = f_s(pw)
    rel = np.load(C.OUT1 / "rel_T0.npz"); H, Tt = rel["hit"], rel["t"]
    px = np.load(C.OUT1 / "eval_px/heldout_T0.npz"); rng = np.random.default_rng(4602); geo = R.Geo(*C.offsets("T0"))
    Va, Fa = R.mesh_a(); bx = np.load(C.BOX_T0); Vb, Fb = bx["V"].astype(np.float64), bx["F"]
    rows = {k: [] for k in ("view", "mat", "near", "inside", "E_in", "E_in_rep", "R1", "R2", "R2t", "kappa")}
    for v in views:
        val = px[f"v{v}_valid"].astype(bool); dom = px[f"v{v}_dom"]; mat = np.where(dom == 1, 8, px[f"v{v}_mat"])
        cand = np.flatnonzero(val & (mat != 0)); sel = np.sort(rng.choice(cand, min(npx, len(cand)), replace=False))
        prt = px[f"v{v}_port"][sel].astype(np.int64); up, inv = np.unique(prt, return_inverse=True)
        o, d = RR.ray_setup(A, up); h = H[up]; t = Tt[up].astype(np.float64); hc = np.clip(h, 0, None)
        ok, p, prim, sid, tt = geo.intersect(o.reshape(-1, 3), d.reshape(-1, 3))           # re-trace for the hit triangle normal
        ntri = np.zeros((len(ok), 3))
        ma = ok & (sid == geo.names.index("A")); mb = ok & (sid == geo.names.index("B")) if "B" in geo.names else np.zeros_like(ok)
        for m, V, F in ((ma, Va, Fa), (mb, Vb, Fb)):
            if m.any():
                tri = prim[m]; cr = np.cross(V[F[tri, 1]] - V[F[tri, 0]], V[F[tri, 2]] - V[F[tri, 0]]); ntri[m] = cr / np.linalg.norm(cr, axis=1, keepdims=True)
        ntri = ntri.reshape(*h.shape, 3)
        agree = float(((ok.reshape(h.shape)) == (h != -1)).mean())
        cq = np.abs((nq_all[hc] * -d).sum(-1)); ct = np.abs((ntri * -d).sum(-1))
        live = h >= 0; fsv = np.where(live, fs_port[hc], 0.0)
        near = ((t < 3e-4) & live).mean(1)[inv]; inside = (h == -2).mean(1)[inv]
        for kappa in kappas:
            fe = edir / f"emit_k{kappa}_v{v}.npz"; fo = edir / f"emitonly_k{kappa}_v{v}.npz"
            if not fe.exists():
                continue
            e = np.load(fe)["rgb"].astype(np.float64); eo = np.load(fo)["rgb"].astype(np.float64)
            E_in = np.pi * (e[sel] - eo[sel]).mean(1)
            fr = edir / f"emit_k{kappa}_v{v}_rep.npz"
            E_rep = np.pi * (np.load(fr)["rgb"].astype(np.float64)[sel] - eo[sel]).mean(1) if fr.exists() else np.full(len(sel), np.nan)
            w = math.pi / H.shape[1]
            r1 = w * (fsv * g(1.0, kappa)).sum(1); r2 = w * (fsv * g(cq, kappa)).sum(1); r2t = w * (fsv * g(ct, kappa)).sum(1)
            for k_, arr in (("view", np.full(len(sel), v)), ("mat", mat[sel]), ("near", near), ("inside", inside), ("E_in", E_in), ("E_in_rep", E_rep),
                            ("R1", r1[inv]), ("R2", r2[inv]), ("R2t", r2t[inv]), ("kappa", np.full(len(sel), kappa))):
                rows[k_].append(arr)
        print("view", v, "retrace agreement with rel_T0 hit/miss", agree, flush=True)
    d_ = {k: np.concatenate(x) for k, x in rows.items()}

    def st(a, b, m):
        a, b = a[m], b[m]
        return {"px": int(m.sum()), "pooled_ratio": float(a.sum() / b.sum()), "median_abs_rel": float(np.median(np.abs(a - b) / np.maximum(b, 1e-6))),
                "corr": float(np.corrcoef(a, b)[0, 1])}
    rec = {"views": list(views), "kappas": list(kappas), "per_kappa": {}}
    for kappa in kappas:
        mk = d_["kappa"] == kappa
        if not mk.any():
            continue
        r = {}
        for arm in ("R1", "R2", "R2t"):
            r[arm] = {"all": st(d_[arm], d_["E_in"], mk)}
            for i in range(1, 9):
                m = mk & (d_["mat"] == i)
                if m.sum() > 100:
                    r[arm][MATS[i]] = st(d_[arm], d_["E_in"], m)
            for lab, m in (("near_hit_frac>0.05", mk & (d_["near"] > 0.05)), ("inside_frac>0.10", mk & (d_["inside"] > 0.10))):
                if m.sum() > 100:
                    r[arm][lab] = st(d_[arm], d_["E_in"], m)
        rep = mk & np.isfinite(d_["E_in_rep"])
        if rep.any():
            r["oracle_replicate"] = {"median_abs_rel": float(np.median(np.abs(d_["E_in_rep"][rep] - d_["E_in"][rep]) / np.maximum(d_["E_in"][rep], 1e-6))),
                                     "corr": float(np.corrcoef(d_["E_in_rep"][rep], d_["E_in"][rep])[0, 1]), "px": int(rep.sum())}
        rec["per_kappa"][f"kappa{kappa}"] = r
    p0 = rec["per_kappa"].get("kappa0", {}).get("R2", {}).get("all")
    if p0:
        crit = {}
        for kappa in kappas:
            if kappa == 0 or f"kappa{kappa}" not in rec["per_kappa"]:
                continue
            a = rec["per_kappa"][f"kappa{kappa}"]["R2"]["all"]
            crit[f"kappa{kappa}"] = {"pooled_within_0.03_of_kappa0": abs(a["pooled_ratio"] - p0["pooled_ratio"]) <= 0.03, "corr_ge_kappa0_minus_0.02": a["corr"] >= p0["corr"] - 0.02,
                                     "median_rel_le_kappa0_plus_0.03": a["median_abs_rel"] <= p0["median_abs_rel"] + 0.03}
        rec["criteria_H_C_H_A_known_field"] = crit; rec["SUPPORTED"] = bool(crit) and all(all(x.values()) for x in crit.values())
    (out / "kf46_eval.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1)[:6000], flush=True)
    np.savez_compressed(out / "kf46_arrays.npz", **d_)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], tuple(int(x) for x in a[2].split(",")) if len(a) > 2 else (6, 15), tuple(int(x) for x in a[3].split(",")) if len(a) > 3 else (0, 2, 8))
