"""WL42 Gate 2b attribution - why the r1 representative quadrature misses (diagnostic only; changes nothing).

Pure numpy (runs on LabServer63 next to anchors.npz). For random Lego receivers and the Gate 2b test kernels
k(r) = exp(-r/d)/(r + 5 mm) it splits the per-receiver error of the stored r1 estimate into
  aggregation : all aggregate sources in the ball, each with its own area (no sampling)  vs  the fine-port truth
  pick bias   : expectation of the stored scheme (uniform pick in a stratum x stratum area)  vs  all aggregates
  sampling    : stored realised picks  vs  their expectation
and evaluates two unbiased alternatives with the same K slots (realised, seeded):
  area_pick   : same equal-count strata, representative drawn with probability proportional to its area
  near_exact  : the K/2 nearest aggregates exactly (own area), the remaining strata area-proportional
Fields (no shadows available without Mitsuba): uniform E = 1; E = max(n.w, 0) for the held-out T0 view-6 sun w;
the same x a 4 cm checker (shadow-like edges).

    python r1_gate2_attrib.py <anchors.npz> <out.json>
"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

W6 = np.array([0.69818145, 0.6463503, 0.30785364])


def strata(c, K):
    return np.linspace(0, c, K + 1).astype(np.int64)


def main(path, out):
    Z = np.load(path)
    nA = int(Z["n_A"]); pos = Z["pos"][:nA].astype(np.float64); nrm = Z["nrm"][:nA].astype(np.float64); area = Z["area"][:nA].astype(np.float64)
    ck = np.floor(pos / 0.04).astype(np.int64)
    fields = {"uniform": np.ones(nA), "cos": np.clip(nrm @ W6, 0, None),
              "cos_checker": np.clip(nrm @ W6, 0, None) * ((ck.sum(1) % 2) == 0)}
    rng = np.random.default_rng(7); recv = rng.choice(nA, 4000, replace=False)
    tree = cKDTree(pos)
    rec = {}
    for name, R, d, rep, rA, f2x, xpos, xarea, xobj in (
            ("band", 0.05, 0.01, Z["nbr"], Z["nbrA"], Z["f2b"], Z["bpos"], Z["barea"], Z["bobj"]),
            ("mid", 0.30, 0.10, Z["mid"], Z["midA"], Z["f2c"], Z["cpos"], Z["carea"], Z["cobj"])):
        K = rep.shape[1]; k = lambda r: np.exp(-r / d) / (r + 0.005)  # noqa: E731
        f2x = f2x[:nA]; xpos = xpos.astype(np.float64); xarea = xarea.astype(np.float64)
        gx = np.flatnonzero(xobj == 0); xtree = cKDTree(xpos[gx])
        balls = xtree.query_ball_point(pos[recv], R); fine = tree.query_ball_point(pos[recv], R)
        rec[name] = {}
        for fn, E in fields.items():
            Eagg = np.bincount(f2x, E * area, len(xarea)) / np.maximum(xarea, 1e-30)
            res = {key: np.zeros(len(recv)) for key in ("truth", "all_agg", "stored", "stored_expect", "area_pick", "near_exact")}
            r2 = np.random.default_rng(11)
            for i, x in enumerate(recv):
                nb = np.asarray(fine[i]); r = np.linalg.norm(pos[nb] - pos[x], axis=1); res["truth"][i] = (area[nb] * E[nb] * k(r)).sum()
                b = gx[np.asarray(balls[i], np.int64)]; rb = np.linalg.norm(xpos[b] - pos[x], axis=1); o = np.argsort(rb, kind="stable")
                b = b[o]; rb = rb[o]; a = xarea[b]; f = Eagg[b] * k(rb); c = len(b)
                res["all_agg"][i] = (a * f).sum()
                p = rep[x]; ok = p >= 0; rr = np.linalg.norm(xpos[np.clip(p, 0, None)] - pos[x], axis=1)
                res["stored"][i] = np.where(ok, rA[x] * Eagg[np.clip(p, 0, None)] * k(rr), 0).sum()
                if c <= K:
                    for key in ("stored_expect", "area_pick", "near_exact"):
                        res[key][i] = res["all_agg"][i]
                    continue
                e = strata(c, K); As = np.add.reduceat(a, e[:-1]); cs = np.diff(e)
                res["stored_expect"][i] = (As / cs * np.add.reduceat(f, e[:-1])).sum()
                tot = 0.0
                for s in range(K):
                    seg = slice(e[s], e[s + 1]); pa = a[seg] / As[s]; j = r2.choice(cs[s], p=pa); tot += As[s] * f[seg][j]
                res["area_pick"][i] = tot
                h = K // 2; tot = (a[:h] * f[:h]).sum(); e2 = h + strata(c - h, K - h)
                for s in range(K - h):
                    seg = slice(e2[s], e2[s + 1]); As2 = a[seg].sum(); j = r2.choice(e2[s + 1] - e2[s], p=a[seg] / As2); tot += As2 * f[seg][j]
                res["near_exact"][i] = tot
            m = res["truth"] > 1e-3 * res["truth"].max()

            def stat(est, ref):
                rel = np.abs(est[m] - ref[m]) / np.maximum(ref[m], 1e-30)
                return {"median_rel_err": float(np.median(rel)), "q90": float(np.quantile(rel, 0.9)), "pooled_ratio": float(est[m].sum() / ref[m].sum())}
            rec[name][fn] = {"receivers": int(m.sum()),
                             "vs_truth": {key: stat(res[key], res["truth"]) for key in res if key != "truth"},
                             "vs_all_agg": {key: stat(res[key], res["all_agg"]) for key in ("stored", "stored_expect", "area_pick", "near_exact")}}
            print(name, fn, json.dumps(rec[name][fn]), flush=True)
    json.dump(rec, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
