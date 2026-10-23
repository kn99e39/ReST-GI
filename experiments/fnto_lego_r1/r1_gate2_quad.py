"""WL42 Gate 2b - realised quadrature of the full-support band / mid kernels on the real Lego geometry (PC, Mitsuba).

Truth: dense sum over all FINE ports of the same domain within the radius, sum_s A_s E_s k(|x - s|).
r1 estimate: sum over the K stratified representatives of A_rep E_agg(rep) k(|x - rep|), E_agg = area-mean irradiance of
the representative's aggregate (1.5 cm band sources / 8 cm mid sources), exactly the inputs the kernels see.
Field: the sun of held-out T0 view 6 on the T0 geometry (port visible fractions). Test kernels k(r) = exp(-r/d)/(r + 5 mm),
band d = 1 cm (R 5 cm), mid d = 10 cm (R 30 cm). 20,000 random Lego receiver ports with a non-zero truth.
Predeclared pass (before measurement): median per-receiver |relative error| <= 0.15 for both.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import r1_common as C
import fnto_relations as R
import r1_relations as RR

A = RR.load(); nA = int(A["n_A"]); geo = R.Geo(*C.offsets("T0"))
D = np.load(C.OUT / "eval_px/heldout_T0.npz"); w = D["v6_light"]
frac, _ = RR.sun_vis(A, geo, w)
E = C.E_SUN * np.clip(A["nrm"] @ w, 0, None) * frac
rec = {}
for name, R_, d, rep, rA, f2x, xpos, xarea in (("band", C.R_LOCAL, 0.01, A["nbr"], A["nbrA"], A["f2b"], A["bpos"], A["barea"]),
                                                ("mid", C.R_MID, 0.10, A["mid"], A["midA"], A["f2c"], A["cpos"], A["carea"])):
    Eagg = np.bincount(f2x, E * A["area"], len(xarea)) / np.maximum(xarea, 1e-12)
    k = lambda r: np.exp(-r / d) / (r + 0.005)  # noqa: E731
    rng = np.random.default_rng(0); recv = rng.choice(nA, 20000, replace=False)
    tree = cKDTree(A["pos"][:nA]); truth = np.zeros(len(recv))
    for i, x in enumerate(recv):
        nb = tree.query_ball_point(A["pos"][x], R_)
        r = np.linalg.norm(A["pos"][nb] - A["pos"][x], axis=1); truth[i] = (A["area"][nb] * E[nb] * k(r)).sum()
    rp = rep[recv]; ok = rp >= 0
    r = np.linalg.norm(xpos[np.clip(rp, 0, None)] - A["pos"][recv][:, None], axis=-1)
    est = (np.where(ok, rA[recv] * Eagg[np.clip(rp, 0, None)] * k(r), 0)).sum(1)
    m = truth > 1e-3 * truth.max()
    rel = np.abs(est[m] - truth[m]) / truth[m]
    rec[name] = {"receivers": int(m.sum()), "median_rel_err": float(np.median(rel)), "q90_rel_err": float(np.quantile(rel, 0.9)),
                 "pooled_ratio_est_over_truth": float(est[m].sum() / truth[m].sum()), "pass": bool(np.median(rel) <= 0.15)}
    print(name, rec[name], flush=True)
out = C.OUT1 / "gate2"; out.mkdir(parents=True, exist_ok=True); (out / "quadrature.json").write_text(json.dumps(rec, indent=1))
