"""WL42 Gate 2b attribution, part 2 - sampling error of the representative quadrature vs the number of slots K
(diagnostic only; changes nothing). Same receivers, kernels and synthetic fields as r1_gate2_attrib.py.

Error is measured against the all-aggregates sum (the sampling part only; the aggregation part is in quadrature_attrib.json).
Schemes, realised with a fixed seed:
  near_exact     : the K/2 nearest aggregates exactly, the rest in K/2 equal-count distance strata, area-proportional pick
  near_exact_ori : as near_exact, but the far part is ordered by (8 distance shells, dominant normal axis, distance)
                   before the equal-count split, so a stratum holds one orientation class where possible

    python r1_gate2_ksweep.py <anchors.npz> <out.json>
"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

W6 = np.array([0.69818145, 0.6463503, 0.30785364])
KS = {"band": (48, 96, 128, 192), "mid": (64, 128, 192, 256)}


def axis_key(n):
    a = np.argmax(np.abs(n), 1)
    return a * 2 + (n[np.arange(len(n)), a] > 0)


def estimate(a, f, order_far, K, rng):
    c = len(a)
    if c <= K:
        return (a * f).sum()
    h = K // 2; tot = (a[:h] * f[:h]).sum()
    af, ff = a[h:][order_far], f[h:][order_far]
    e = np.linspace(0, c - h, K - h + 1).astype(np.int64)
    for s in range(K - h):
        aa = af[e[s]:e[s + 1]]; As = aa.sum(); j = rng.choice(len(aa), p=aa / As); tot += As * ff[e[s]:e[s + 1]][j]
    return tot


def main(path, out):
    Z = np.load(path)
    nA = int(Z["n_A"]); pos = Z["pos"][:nA].astype(np.float64); nrm = Z["nrm"][:nA].astype(np.float64); area = Z["area"][:nA].astype(np.float64)
    ck = np.floor(pos / 0.04).astype(np.int64)
    fields = {"cos": np.clip(nrm @ W6, 0, None), "cos_checker": np.clip(nrm @ W6, 0, None) * ((ck.sum(1) % 2) == 0)}
    recv = np.random.default_rng(7).choice(nA, 4000, replace=False)[:1500]
    rec = {}
    for name, R, d, f2x, xpos, xnrm, xarea, xobj in (
            ("band", 0.05, 0.01, Z["f2b"], Z["bpos"], Z["bnrm"], Z["barea"], Z["bobj"]),
            ("mid", 0.30, 0.10, Z["f2c"], Z["cpos"], Z["cnrm"], Z["carea"], Z["cobj"])):
        k = lambda r: np.exp(-r / d) / (r + 0.005)  # noqa: E731
        f2x = f2x[:nA]; xpos = xpos.astype(np.float64); xarea = xarea.astype(np.float64); ax = axis_key(xnrm.astype(np.float64))
        gx = np.flatnonzero(xobj == 0); balls = cKDTree(xpos[gx]).query_ball_point(pos[recv], R)
        rec[name] = {}
        for fn, E in fields.items():
            Eagg = np.bincount(f2x, E * area, len(xarea)) / np.maximum(xarea, 1e-30)
            ref = np.zeros(len(recv)); est = {(s, K): np.zeros(len(recv)) for s in ("near_exact", "near_exact_ori") for K in KS[name]}
            rng = np.random.default_rng(11)
            for i, x in enumerate(recv):
                b = gx[np.asarray(balls[i], np.int64)]; rb = np.linalg.norm(xpos[b] - pos[x], axis=1); o = np.argsort(rb, kind="stable")
                b = b[o]; rb = rb[o]; a = xarea[b]; f = Eagg[b] * k(rb); ref[i] = (a * f).sum()
                for K in KS[name]:
                    h = K // 2; nf = max(len(b) - h, 0)
                    est[("near_exact", K)][i] = estimate(a, f, np.arange(nf), K, rng)
                    if nf:
                        shell = np.minimum((rb[h:] / R * 8).astype(np.int64), 7)
                        ori = np.lexsort((rb[h:], ax[b[h:]], shell))
                    else:
                        ori = np.arange(0)
                    est[("near_exact_ori", K)][i] = estimate(a, f, ori, K, rng)
            m = ref > 1e-3 * ref.max()
            rec[name][fn] = {f"{s}_K{K}": {"median_rel_err": float(np.median(np.abs(v[m] - ref[m]) / ref[m])),
                                           "q90": float(np.quantile(np.abs(v[m] - ref[m]) / ref[m], 0.9))} for (s, K), v in est.items()}
            print(name, fn, json.dumps(rec[name][fn]), flush=True)
    json.dump(rec, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
