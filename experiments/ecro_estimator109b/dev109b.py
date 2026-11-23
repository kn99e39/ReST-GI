"""WL109B DEVELOPMENT study (CPU). Not evidence, not qualification.

Uses only WL109A development material: the WL109A synthetic generators (with fresh development seeds) and the WL109A
independent R0 null pool's raw small-ROI pixels. Its purpose is to choose ONE inference procedure and its group count
before the WL109B protocol is frozen; the sealed WL109B calibration later uses new seeds and a new GPU null pool.

Compared procedures, all on the same paired-difference data D [R, 3, P]:
  WL109 : original Q_g = mean(dbar_g^2 - s_g^2 / n) / scale per group, Student t (G - 1) on the G group values
  XJ    : cross-product U-statistic over distinct groups, U = sum_{g != h} <dbar_g, dbar_h> / (G (G - 1) X scale),
          delete-one-group jackknife SE, Student t (G - 1) bounds
python dev109b.py OUT [--reps N] [--workers W]
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_estimator109a"))
from common109a import EVAL_VIEWS, QLIM, all_masks, bias_pattern, t975, read, sha  # noqa: E402
import calib109a as CA  # noqa: E402

QT = (0.0, 0.5, 1.0, 2.0)
ACTIVE = ("WL109", "XJ", "XJL")
PATTERNS = ("uniform", "noise_proportional", "localized", "signed_smooth")


def wl109_bounds(dbar, s2, n, scale):
    G = dbar.shape[0]
    q = (dbar * dbar - s2 / n).mean(1) / scale
    pt = q.mean(); se = q.std(ddof=1) / math.sqrt(G); t = t975(G - 1)
    return pt, pt + t * se, pt - t * se


def xj_bounds(dbar, s2, n, scale):
    G = dbar.shape[0]                                            # dbar [G, X]
    S = dbar.sum(0); sq = (dbar * dbar).sum(0)
    U = (S * S - sq).mean() / (G * (G - 1)) / scale
    Sm = S[None] - dbar; sqm = sq[None] - dbar * dbar
    Um = (Sm * Sm - sqm).mean(1) / ((G - 1) * (G - 2)) / scale
    ps = G * U - (G - 1) * Um
    se = ps.std(ddof=1) / math.sqrt(G); t = t975(G - 1)
    return U, U + t * se, U - t * se


def lam_max_from_sub(D, sub=16):
    """largest eigenvalue of the per-pair noise covariance of D, estimated from 16-pair subgroup means (x 16), as the
    evidence stores it (upward-biased for wide ROIs: conservative)."""
    R = D.shape[0]; k = R // sub
    m = D[:k * sub].reshape(k, sub, -1).mean(1)
    m = m - m.mean(0, keepdims=True)
    s = np.linalg.svd(m, compute_uv=False)
    return float(s[0] ** 2 / (k - 1) * sub)


def xjl_bounds(dbar, s2, n, scale, lam=None, X=None, alpha_lo=0.025):
    """XJ with the worst-case linear-term variance b q added and the bound obtained by inversion in q."""
    from scipy.stats import t as tdist
    G = dbar.shape[0]; R = G * n
    U, _, _ = xj_bounds(dbar, s2, n, scale)
    S = dbar.sum(0); sq = (dbar * dbar).sum(0)
    Sm = S[None] - dbar; sqm = sq[None] - dbar * dbar
    Um = (Sm * Sm - sqm).mean(1) / ((G - 1) * (G - 2)) / scale
    ps = G * U - (G - 1) * Um
    se2 = ps.var(ddof=1) / G
    b = 4.0 * lam / (R * X * scale)
    a = t975(G - 1) ** 2
    disc = a * a * b * b / 4 + a * se2 + a * b * U
    up = max(0.0, U + a * b / 2 + math.sqrt(disc)) if disc > 0 else 0.0
    al = float(tdist.ppf(1 - alpha_lo, G - 1)) ** 2
    disc = al * al * b * b / 4 + al * se2 + al * b * U
    lo = U + al * b / 2 - math.sqrt(disc) if disc > 0 else -1.0
    return U, up, lo


def xjlb_bounds(dbar, s2, n, scale, lam=None, X=None):
    return xjl_bounds(dbar, s2, n, scale, lam, X, alpha_lo=0.05 / 84)


METHODS = {"WL109": wl109_bounds, "XJ": xj_bounds, "XJL": xjl_bounds, "XJLB": xjlb_bounds}


def run_methods(D, G, lim, scale, Pp, av, rs, acc, designs_G=None):
    """A fixed injected bias shifts every group mean by the same vector and leaves within-group variances unchanged, so
    injections are applied to the group means (exact)."""
    R = D.shape[0]; n = R // G
    X = D[:G * n].reshape(G, n, -1)
    dbar0 = X.mean(1); s2 = X.var(1, ddof=1)
    lam = lam_max_from_sub(D[:G * n]); XX = D.shape[1]
    for f in QT:
        pats = PATTERNS if f > 0 else ("none",)
        for pat in pats:
            dbar = dbar0 if f == 0 else dbar0 + math.sqrt(f * lim * scale) * bias_pattern(pat, Pp, av, rs).reshape(1, -1)
            q = f * lim
            for mn, fn in METHODS.items():
                if mn not in ACTIVE:
                    continue
                pt, up, lo = fn(dbar, s2, n, scale, lam, XX) if mn.startswith("XJL") else fn(dbar, s2, n, scale)
                a = acc.setdefault(f"{mn}/{pat}@{f}", {"noncov": 0, "fail": 0, "power": 0, "n": 0})
                a["noncov"] += up < q; a["fail"] += up > lim; a["power"] += lo > lim; a["n"] += 1


def noise(fam, rs, R, Pp, smooth):
    if fam in ("gauss_iid", "mc_like_4", "mc_like_32", "correlated"):
        return CA.synth_noise(fam, rs, R, Pp, smooth)
    if fam.startswith("heteroscedastic"):
        s = float(fam.split("_")[1])
        sd = np.exp(s * rs.standard_normal(Pp))
        return CA.synth_noise("heteroscedastic", rs, R, Pp, None, sd)
    if fam == "asym_mc4":                                        # X heavy-tailed, Y Gaussian with the same mean / variance
        k = 4; v = CA._mc_var(k); m = 0.3 * math.exp(1.5 ** 2 / 2)
        X = CA._mc_like(rs, (R, Pp), k)
        Y = m + math.sqrt(v) * rs.standard_normal((R, Pp))
        w = np.array([1.0, 0.8, 0.6])
        return (X - Y)[:, None, :] * w[None, :, None], (w[:, None] ** 2) * (2 * v) * np.ones((1, Pp))
    if fam == "het_mc4":                                         # firefly noise with per-pixel scale exp(N(0, 1))
        D, var = CA.synth_noise("mc_like_4", rs, R, Pp, None)
        sd = np.exp(rs.standard_normal(Pp))
        return D * sd[None, None, :], var * (sd ** 2)[None, :]
    raise KeyError(fam)


def synth_cell(args):
    fam, view, roi, R, G, reps, seed, idx = args
    rs = np.random.default_rng(seed)
    Pp = idx.size; xy = CA.xy_of(idx)
    smooth = CA._smooth_matrix(xy) if fam == "correlated" else None
    lim = QLIM[roi]
    acc = {}
    for _ in range(reps):
        D, var = noise(fam, rs, R, Pp, smooth)
        scale = float(var.mean() / 2)                                   # Var(D) / scale = 2 (unfavourable, as WL109A)
        av = {"noise_sd": np.sqrt(var.mean(0)), "xy": xy}
        run_methods(D.reshape(R, -1), G, lim, scale, Pp, av, rs, acc)
    out = {k: {kk: float(vv / v["n"]) for kk, vv in v.items() if kk != "n"} | {"n": v["n"]} for k, v in acc.items()}
    return f"{fam}/{view}:{roi}/R{R}G{G}", out


def real_cell(args):
    """WL109A null-pool raw small-ROI pixels: R0 vs R0 re-pairings (exact zero expectation) at design (R, G)."""
    rawdir, view, spp, roi, R, G, reps, seed, sel = args
    rs = np.random.default_rng(seed)
    raw = np.load(rawdir / "raw.npy", mmap_mode="r")
    ridx = np.load(rawdir / "raw_idx.npy")
    keep = np.isin(ridx, sel)
    A = np.asarray(raw[:, :, keep], np.float64)                    # [N0, 6, P]
    H = A[:, 3:6]; T = A[:, 0:3] + A[:, 3:6]
    scale = float(T.var(0, ddof=1).mean())
    Pp = H.shape[2]; xy = CA.xy_of(ridx[keep])
    lim = QLIM[roi]
    av = {"noise_sd": np.sqrt(2 * H.var(0, ddof=1).mean(0)), "xy": xy}
    N0 = H.shape[0]
    acc = {}
    per_perm = (N0 // 2) // R
    nperm = math.ceil(reps / per_perm)
    for _ in range(nperm):
        perm = rs.permutation(N0)
        Dall = H[perm[0::2]] - H[perm[1::2]]
        for b in range(per_perm):
            D = Dall[b * R:(b + 1) * R]
            run_methods(D.reshape(R, -1), G, lim, scale, Pp, av, rs, acc)
    out = {k: {kk: float(vv / v["n"]) for kk, vv in v.items() if kk != "n"} | {"n": v["n"]} for k, v in acc.items()}
    return f"real/{view}:{roi}/spp{spp}/R{R}G{G}", out


DESIGNS = [(512, 8), (512, 16), (512, 32), (1024, 16), (1024, 32)]


def main(out, reps, workers, wl109a_null):
    import runtime109 as RT
    from multiprocessing import Pool
    out.mkdir(parents=True, exist_ok=True)
    shapes = [("H4", "material_boundary"), ("H3", "material_boundary"), ("V0", "silhouette"), ("V0", "material_boundary"),
              ("H4", "silhouette")]
    idxs = {s: np.flatnonzero(all_masks(RT.aux(s[0]))[s[1]].reshape(-1)) for s in shapes}
    fams = ["gauss_iid", "mc_like_4", "asym_mc4", "heteroscedastic_1.0", "heteroscedastic_1.5", "het_mc4", "correlated"]
    designs = DESIGNS
    jobs = []; j = 0
    base = 77_000_001                                                # development seeds (not used by any evidence family)
    for fam in fams:
        for s in shapes:
            if idxs[s].size > 1500 and fam == "correlated":
                continue
            for R, G in designs:
                jobs.append((fam, s[0], s[1], R, G, reps, base + j, idxs[s])); j += 1
    rjobs = []
    for view in ("H3", "H4", "V0", "H1"):
        masks = all_masks(RT.aux(view))
        for spp in (4, 32):
            d = wl109a_null / f"{view}_spp{spp}"
            for roi in ("material_boundary", "silhouette"):
                sel = np.flatnonzero(masks[roi].reshape(-1))
                for R, G in [d for d in DESIGNS if d[0] == 512]:
                    rjobs.append((d, view, spp, roi, R, G, max(reps // 2, 100), base + 50_000 + j, sel)); j += 1
    t0 = time.time()
    with Pool(workers) as pool:
        res = dict(pool.imap_unordered(synth_cell, jobs))
        res.update(dict(pool.imap_unordered(real_cell, rjobs)))
    (out / "dev109b.json").write_text(json.dumps({"cells": res, "reps": reps, "code_sha256": sha(Path(__file__)),
                                                  "elapsed_s": time.time() - t0, "note": "DEVELOPMENT ONLY"}, indent=1))
    print("dev done", len(res), round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--reps", type=int, default=1000); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--wl109a-null", default=str(Path.home() / "rest_sss101a/results/ecro_estimator109a/evidence/cap/null"))
    ap.add_argument("--methods", default="WL109,XJ,XJL"); ap.add_argument("--designs", default="")
    a = ap.parse_args()
    ACTIVE = tuple(a.methods.split(","))
    if a.designs:
        DESIGNS = [tuple(int(v) for v in d.split("x")) for d in a.designs.split(",")]
    main(Path(a.out).resolve(), a.reps, a.workers, Path(a.wl109a_null))
