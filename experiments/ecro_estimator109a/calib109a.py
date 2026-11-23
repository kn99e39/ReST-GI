"""WL109A Stage B: audit and calibration of the original Q upper bound (CPU only; never initialises CUDA).

python calib109a.py hist   WL109_EVIDENCE_DIR OUT_DIR      historical null splits of the 64 WL109 R0 frames per (view, spp)
python calib109a.py synth  OUT_DIR [--workers K]           known-truth synthetic families
python calib109a.py pool   NULL_DIR OUT_DIR SYNTH_JSON     independent R0-versus-R0 null pool + gates + budget decision

Injected bias uses the exact identity Q_g(q) = Q_g(0) + 2 sqrt(q / scale) mean(b * Dbar_g) + q for D + sqrt(q * scale) b,
mean(b^2) = 1, so every Q_true and pattern is evaluated on the same noise realization.
"""
import argparse
from itertools import combinations
import math
import os
from pathlib import Path
import time

import numpy as np

from common109a import (P, P9, BIAS_SPP, DECISION_ROIS, EVAL_VIEWS, LUMA, QLIM, all_masks, b1_state, bias_pattern, read, sha,
                        t975, write, HERE, W109)

C = P["calibration"]
QT = C["injection"]["Q_true_over_limit"]
PATTERNS = C["injection"]["patterns"]
T7 = t975(7)


# ------------------------------------------------------------------------------------------------ core
def group_stats(Dg):
    """Dg [G, n, X] -> dbar [G, X], s2 [G, X]"""
    return Dg.mean(1), Dg.var(1, ddof=1)


def qg0(dbar, s2, n, scale):
    return (dbar * dbar - s2 / n).mean(1) / scale


def summaries(Qsub):
    """Qsub [..., S, 8] -> point, se, upper, lower [..., S]"""
    pt = Qsub.mean(-1)
    se = Qsub.std(-1, ddof=1) / math.sqrt(Qsub.shape[-1])
    return pt, se, pt + T7 * se, pt - T7 * se


def evaluate(Q0, mb, scale, limit, subsets):
    """Q0 [G] null group values; mb {pattern: [K, G]} mean(b * dbar_g) for K pattern instances; subsets [S, 8] group indices.
    -> dict of probabilities per (q, pattern)."""
    out = {}
    pt, se, up, lo = summaries(Q0[subsets])
    out["q0"] = {"noncov": float((up < 0).mean()), "fail": float((up > limit).mean()), "biased": float((lo > limit).mean()),
                 "point_mean": float(pt.mean()), "se_median": float(np.median(se)), "upper_median": float(np.median(up)),
                 "upper_q99": float(np.quantile(up, 0.99))}
    for pat, M in mb.items():
        for f in QT[1:]:
            q = f * limit
            Qg = Q0[None, :] + 2 * math.sqrt(q / scale) * M + q            # [K, G]
            pt, se, up, lo = summaries(Qg[:, subsets])
            out[f"{pat}@{f}"] = {"noncov": float((up < q).mean()), "pass": float((up <= limit).mean()),
                                 "power": float((lo > limit).mean()), "point_mean": float(pt.mean())}
    return out


def pattern_means(dbar, aux_vals, scale, rs, K):
    """mean over X of b * dbar_g for K instances of every pattern -> {pattern: [K, G]}"""
    G, X = dbar.shape
    Pp = X // 3
    out = {}
    for pat in PATTERNS:
        k = 1 if pat in ("uniform", "noise_proportional") else K
        bs = np.stack([bias_pattern(pat, Pp, aux_vals, rs).reshape(-1) for _ in range(k)])     # [k, X]
        out[pat] = bs @ dbar.T / X
    return out


def xy_of(mask_idx, W=512):
    return np.stack(np.divmod(mask_idx, W), 1).astype(np.float64)


# ------------------------------------------------------------------------------------------------ historical splits
def hist(ev, out):
    import runtime109 as RT
    out.mkdir(parents=True, exist_ok=True)
    run = ev / "run"
    man = read(run / "manifest.json")
    met = read(ev / "metrics/metrics109.json")
    cfg = C["historical_splits"]
    rs = np.random.default_rng(P["seeds"]["calibration_rng"])
    res = {"design": {"R": 32, "G": 8, "n": 4}, "settings": {}, "source_run_manifest_sha256": sha(run / "manifest.json")}
    for view in EVAL_VIEWS:
        masks = all_masks(RT.aux(view))
        for spp in BIAS_SPP:
            t0 = time.time()
            files = [f"paired/{view}/spp{spp}/r{r:02d}_R0.npy" for r in range(32)] + [f"grid/{view}/spp{spp:03d}/r{r:02d}.npy" for r in range(32)]
            for k in files:
                assert sha(run / "raw" / k) == man["files"][k], k
            fr = np.stack([np.load(run / "raw" / k) for k in files]).astype(np.float64)          # [64, 9, H, W]
            for roi in [r for r in DECISION_ROIS if r in masks]:
                m = masks[roi]
                idx = np.flatnonzero(m.reshape(-1))
                H = fr[:, 6:9].reshape(64, 3, -1)[:, :, idx]
                T = (fr[:, 0:3] + fr[:, 3:6] + fr[:, 6:9]).reshape(64, 3, -1)[:, :, idx]
                scale = float(T.var(0, ddof=1).mean())
                noise_sd = np.sqrt(2 * H.var(0, ddof=1).mean(0))
                av = {"noise_sd": noise_sd, "xy": xy_of(idx)}
                draws = cfg["draws_small_roi"] if idx.size < 20000 else cfg["draws_large_roi"]
                lim = QLIM[roi]
                acc = []
                for d in range(draws):
                    perm = rs.permutation(64)
                    D = (H[perm[0::2]] - H[perm[1::2]]).reshape(8, 4, -1)                           # 8 groups x n = 4
                    dbar, s2 = group_stats(D)
                    Q0 = qg0(dbar, s2, 4, scale)
                    mb = pattern_means(dbar, av, scale, rs, 1)
                    acc.append((Q0, mb))
                Q0s = np.stack([a[0] for a in acc])                                                  # [draws, 8]
                one = np.arange(8)[None, :]
                agg = {}
                for i, (Q0, mb) in enumerate(acc):
                    e = evaluate(Q0, mb, scale, lim, one)
                    for k, v in e.items():
                        for kk, vv in v.items():
                            agg.setdefault(k, {}).setdefault(kk, []).append(vv)
                ev_ = {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in agg.items()}
                pt, se, up, lo = summaries(Q0s)
                obs = met["views"][view]["paired"][str(spp)]["Q"][roi]
                ev_["observed_C1_WL109"] = {"point": obs["point"], "upper95": obs["upper95"],
                                            "null_point_percentile": float((pt < obs["point"]).mean()),
                                            "null_upper_percentile": float((up < obs["upper95"]).mean())}
                ev_["pixels"] = int(idx.size); ev_["scale"] = scale; ev_["draws"] = draws
                ev_["varD_over_scale"] = float((noise_sd ** 2).mean() / scale)
                res["settings"][f"{view}/{roi}/{spp}"] = ev_
            print(view, spp, "hist", round(time.time() - t0, 1), flush=True)
            del fr
    res["code_sha256"] = sha(Path(__file__))
    write(out / "hist.json", res)


# ------------------------------------------------------------------------------------------------ synthetic families
def _mc_like(rs, shape, k, p=0.3, s=1.5):
    cnt = rs.binomial(k, p, size=int(np.prod(shape)))
    tot = int(cnt.sum())
    v = rs.lognormal(0.0, s, tot + 1)
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    sums = np.add.reduceat(v, np.minimum(starts, tot))
    sums[cnt == 0] = 0.0
    return (sums / k).reshape(shape)


def _mc_var(k, p=0.3, s=1.5):
    return (p * math.exp(2 * s * s) - p * p * math.exp(s * s)) / k


def synth_noise(fam, rs, R, Pp, smooth=None, sd_pix=None):
    """-> D [R, 3, Pp] with known per-element variance map var [3, Pp]"""
    w = np.array([1.0, 0.8, 0.6])
    if fam in ("gauss_iid", "heteroscedastic"):
        z0 = rs.standard_normal((R, 1, Pp))
        z = math.sqrt(0.9) * z0 + math.sqrt(0.1) * rs.standard_normal((R, 3, Pp))
        sd = np.ones(Pp) if fam == "gauss_iid" else sd_pix
        return z * sd[None, None, :], np.repeat((sd ** 2)[None], 3, 0)
    k = 4 if fam in ("mc_like_4", "correlated") else 32
    base = _mc_like(rs, (R, Pp), k) - _mc_like(rs, (R, Pp), k)
    var = 2 * _mc_var(k) * np.ones(Pp)
    if fam == "correlated":
        base = base @ smooth.T                                                                 # rows: sum_j W_ij^2 = 1
    D = base[:, None, :] * w[None, :, None]
    return D, (w[:, None] ** 2) * var[None, :]


def _smooth_matrix(xy, sigma=1.5, radius=4.0):
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    W = np.where(d <= radius, np.exp(-0.5 * (d / sigma) ** 2), 0.0)
    return W / np.sqrt((W * W).sum(1, keepdims=True))


def _synth_cell(args):
    fam, view, roi, R, G, nrep, seed, idx = args
    rs = np.random.default_rng(seed)
    Pp = idx.size
    xy = xy_of(idx)
    smooth = _smooth_matrix(xy) if fam == "correlated" else None
    n = R // G
    lim = QLIM[roi]
    agg = {}
    t0 = time.time()
    for rep in range(nrep):
        sd_pix = np.exp(rs.standard_normal(Pp)) if fam == "heteroscedastic" else None
        D, var = synth_noise(fam, rs, R, Pp, smooth, sd_pix)
        scale = float(var.mean() / 2)
        X = D.reshape(G, n, -1)
        dbar, s2 = group_stats(X)
        Q0 = qg0(dbar, s2, n, scale)
        av = {"noise_sd": np.sqrt(var.mean(0)), "xy": xy}
        mb = pattern_means(dbar, av, scale, rs, 1)
        e = evaluate(Q0, mb, scale, lim, np.arange(G)[None, :]) if G == 8 else _evaluate_g(Q0, mb, scale, lim)
        for k, v in e.items():
            for kk, vv in v.items():
                agg.setdefault(k, {}).setdefault(kk, []).append(vv)
    res = {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in agg.items()}
    res.update(realizations=nrep, pixels=int(Pp), elapsed_s=time.time() - t0)
    return f"{fam}/{view}:{roi}/R{R}G{G}", res


def _evaluate_g(Q0, mb, scale, limit):
    """G != 8: Student t with G - 1 df (sensitivity only)."""
    G = Q0.size
    t = t975(G - 1)
    def summ(Qg):
        pt = Qg.mean(-1); se = Qg.std(-1, ddof=1) / math.sqrt(G)
        return pt, pt + t * se, pt - t * se
    out = {}
    pt, up, lo = summ(Q0[None])
    out["q0"] = {"noncov": float((up < 0).mean()), "fail": float((up > limit).mean()), "biased": float((lo > limit).mean())}
    for pat, M in mb.items():
        for f in QT[1:]:
            q = f * limit
            Qg = Q0[None, :] + 2 * math.sqrt(q / scale) * M + q
            pt, up, lo = summ(Qg)
            out[f"{pat}@{f}"] = {"noncov": float((up < q).mean()), "pass": float((up <= limit).mean()), "power": float((lo > limit).mean())}
    return out


def synth(out, workers):
    import runtime109 as RT
    from multiprocessing import Pool
    out.mkdir(parents=True, exist_ok=True)
    S = C["synthetic"]
    shapes = {}
    for view, roi in S["roi_shapes"]:
        m = all_masks(RT.aux(view))[roi]
        shapes[(view, roi)] = np.flatnonzero(m.reshape(-1))
    jobs = []
    base = P["seeds"]["calibration_rng"] + 1000
    j = 0
    for fam in S["families"]:
        for (view, roi), idx in shapes.items():
            for R, G in S["designs"]:
                if fam == "mc_like_32" and R >= 512 and idx.size > 400:
                    continue
                if R >= 1024 and fam not in ("gauss_iid", "mc_like_4"):
                    continue
                nrep = S["realizations"]["R<=128"] if R <= 128 else S["realizations"]["R>=512"]
                jobs.append((fam, view, roi, R, G, nrep, base + j, idx)); j += 1
    t0 = time.time()
    with Pool(workers) as pool:
        res = dict(pool.imap_unordered(_synth_cell, jobs))
    write(out / "synth.json", {"cells": res, "code_sha256": sha(Path(__file__)), "elapsed_s": time.time() - t0,
                               "skipped_rule": "mc_like_32 at R >= 512 only for shapes <= 400 px; R = 1024 only gauss_iid and mc_like_4"})
    print("synth done", len(res), round(time.time() - t0, 1), flush=True)


# ------------------------------------------------------------------------------------------------ null pool
def predicted_fail(se, limit, rs, draws=200000):
    z = rs.standard_normal(draws)
    shat = se * np.sqrt(rs.chisquare(7, draws) / 7)
    return float(((z * se + T7 * shat) > limit).mean())


def predicted_power(se_null, var_cross64, n, limit, rs, draws=200000):
    var = se_null ** 2 + var_cross64 * (64 / n) / 8
    se = math.sqrt(var)
    z = rs.standard_normal(draws)
    shat = se * np.sqrt(rs.chisquare(7, draws) / 7)
    return float(((2 * limit + z * se - T7 * shat) > limit).mean())


def pool(nulldir, out, synth_json, smoke=False):
    import runtime109 as RT
    from common109a import seed_null
    out.mkdir(parents=True, exist_ok=True)
    rs = np.random.default_rng(P["seeds"]["calibration_rng"] + 7)
    cfg = C["null_pool"]
    G = cfg["groups"]
    all8 = np.asarray(list(combinations(range(G), 8)))
    res = {"settings": {}, "small_n": {}, "direct_null": {}, "files": {}}
    pooled = {}
    views = [v for v in EVAL_VIEWS if (nulldir / "null" / f"{v}_spp4").exists()]
    assert smoke or views == list(EVAL_VIEWS), views
    for view in views:
        masks = all_masks(RT.aux(view))
        refc, vref, refrec = RT.reference(view)
        ref = refc[0:3] + refc[3:6] + refc[6:9]
        b1_lum = {}
        for spp in BIAS_SPP:
            t0 = time.time()
            u = nulldir / "null" / f"{view}_spp{spp}"
            rec = read(u / "unit.json")
            for f, h in rec["files"].items():
                assert sha(u / f) == h, (u, f)
            assert rec["groups"] == G and (smoke or rec["frames"] == cfg["frames"])
            S1 = np.load(u / "S1.npy").reshape(G, 3, -1); S2 = np.load(u / "S2.npy").reshape(G, 3, -1)
            sT = np.load(u / "sT.npy").reshape(3, -1); sT2 = np.load(u / "sT2.npy").reshape(3, -1)
            N0, n = rec["frames"], rec["group_pairs"]
            names = rec["mask_names"]
            lumH = np.load(u / "lumH.npy")
            varT = (sT2 - sT * sT / N0) / (N0 - 1)
            tS1, tS2 = S1.sum(0), S2.sum(0)
            varD = (tS2 - tS1 * tS1 / (N0 // 2)) / (N0 // 2 - 1)
            for roi in [r for r in DECISION_ROIS if r in masks]:
                idx = np.flatnonzero(masks[roi].reshape(-1))
                lim = QLIM[roi]
                scale = float(varT[:, idx].mean())
                s1 = S1[:, :, idx].reshape(G, -1); s2 = S2[:, :, idx].reshape(G, -1)
                dbar = s1 / n
                sg2 = (s2 - s1 * s1 / n) / (n - 1)
                Q0 = qg0(dbar, sg2, n, scale)
                av = {"noise_sd": np.sqrt(varD[:, idx].mean(0)), "xy": xy_of(idx)}
                mb = pattern_means(dbar, av, scale, rs, 8)
                e = evaluate(Q0, mb, scale, lim, all8)
                sd_g = float(Q0.std(ddof=1))
                se64 = sd_g / math.sqrt(8)
                e["sd_Qg_null"] = sd_g
                e["predicted_fail_n64"] = predicted_fail(se64, lim, rs)
                se128 = se64 * math.sqrt(64 * 63 / (128 * 127))
                e["predicted_fail_n128"] = predicted_fail(se128, lim, rs)
                # bias-induced group variance at q = 2 limit (worst pattern), for the n = 128 power prediction
                vc = 0.0
                for pat, M in mb.items():
                    Qg = Q0[None, :] + 2 * math.sqrt(2 * lim / scale) * M + 2 * lim
                    vc = max(vc, float(Qg.var(1, ddof=1).mean() - Q0.var(ddof=1)))
                e["var_cross_2L_n64"] = max(vc, 0.0)
                e["predicted_power_n128"] = predicted_power(se128, max(vc, 0.0), 128, lim, rs)
                e["pixels"] = int(idx.size); e["scale"] = scale; e["varD_over_scale"] = float(varD[:, idx].mean() / scale)
                # direct null realizations (groups 0-7, 8-15)
                for h, gsel in enumerate((np.arange(8), np.arange(8, 16))):
                    q = Q0[gsel]
                    pt = float(q.mean()); se = float(q.std(ddof=1) / math.sqrt(8))
                    res["direct_null"].setdefault(f"{view}/{roi}", {}).setdefault(f"half{h}", {})[str(spp)] = {
                        "point": pt, "se": se, "upper95": pt + T7 * se, "lower95": pt - T7 * se}
                res["settings"][f"{view}/{roi}/{spp}"] = e
                for k, v in e.items():
                    if isinstance(v, dict):
                        for kk, vv in v.items():
                            pooled.setdefault(k, {}).setdefault(kk, []).append(vv)
                # B1 null: per-pair luma differences of H over the ROI
                ki = names.index(roi)
                dl = lumH[0::2, ki] - lumH[1::2, ki]
                b1_lum.setdefault(roi, {})[spp] = dl
                half = dl.size // 2
            # small-n raw analysis (silhouette, material_boundary)
            raw = np.load(u / "raw.npy").astype(np.float64)                                   # [N0, 6, Praw]
            ridx = np.load(u / "raw_idx.npy")
            Hr = raw[:, 3:6]; Tr = raw[:, 0:3] + raw[:, 3:6]
            Dr = Hr[0::2] - Hr[1::2]                                                          # [1024, 3, Praw]
            for roi in P["raw_rois"]:
                sel = np.isin(ridx, np.flatnonzero(masks[roi].reshape(-1)))
                if sel.sum() == 0:
                    continue
                lim = QLIM[roi]
                scale = float(Tr[:, :, sel].var(0, ddof=1).mean())
                D = Dr[:, :, sel]
                av = {"noise_sd": np.sqrt(D.var(0, ddof=1).mean(0)), "xy": xy_of(ridx[sel])}
                for nn in (4, 8, 16, 32, 64):
                    Gp = D.shape[0] // nn
                    if Gp < 8:
                        continue
                    X = D[:Gp * nn].reshape(Gp, nn, -1)
                    dbar, sg2 = group_stats(X)
                    Q0 = qg0(dbar, sg2, nn, scale)
                    subs = all8 if Gp == 16 else np.stack([rs.choice(Gp, 8, replace=False) for _ in range(4000)])
                    mb = pattern_means(dbar, av, scale, rs, 8)
                    res["small_n"][f"{view}/{roi}/{spp}/n{nn}"] = evaluate(Q0, mb, scale, lim, subs)
            print(view, spp, "pool", round(time.time() - t0, 1), flush=True)
        for roi, dd in b1_lum.items():
            idx = masks[roi]
            refl = float((LUMA @ ref[:, idx]).mean())
            for h in range(2):
                sl = slice(h * half, (h + 1) * half)
                res["direct_null"][f"{view}/{roi}"][f"half{h}"]["B1"] = b1_state({s: v[sl] for s, v in dd.items()}, refl)
    # ---- gates and budget
    pooled_mean = {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in pooled.items()}
    res["pooled_n64"] = pooled_mean
    syn = read(synth_json)
    dn_states = {}
    for key, halves in res["direct_null"].items():
        roi = key.split("/")[1]
        for h, d in halves.items():
            ups = [d[str(s)]["upper95"] for s in BIAS_SPP]; los = [d[str(s)]["lower95"] for s in BIAS_SPP]
            qs = "BIASED" if any(l > QLIM[roi] for l in los) else ("EQUIVALENT" if all(x <= QLIM[roi] for x in ups) else "UNRESOLVED")
            dn_states[f"{key}/{h}"] = {"Q": qs, "B1": d["B1"]["state"]}

    def cov_ok(cell, gate=0.05):
        bad = []
        for k, v in cell.items():
            if not isinstance(v, dict):
                continue
            if k == "q0":
                if v["noncov"] > gate:
                    bad.append((k, v["noncov"]))
            elif "noncov" in v and v["noncov"] > gate:
                bad.append((k, v["noncov"]))
        return bad

    def design_gates(Rd):
        g = {}
        g["coverage_pool_bad"] = cov_ok(pooled_mean)
        syn_bad = {}
        for name, cell in syn["cells"].items():
            if name.endswith(f"/R{Rd}G8"):
                b = cov_ok(cell)
                if b:
                    syn_bad[name] = b
        g["coverage_synth_bad"] = syn_bad
        g["synth_cells_at_design"] = sum(1 for n in syn["cells"] if n.endswith(f"/R{Rd}G8"))
        return g

    gates = {"R512": design_gates(512)}
    power_bad = {k: v["power"] for k, v in pooled_mean.items() if "@2" in k and v["power"] < 0.80}
    fail_pooled = pooled_mean["q0"]["fail"]
    pred64 = {k: v["predicted_fail_n64"] for k, v in res["settings"].items() if v["predicted_fail_n64"] > 0.01}
    gates["R512"].update(power_bad=power_bad, pooled_fail_q0=fail_pooled, predicted_fail_bad=pred64)
    cov512 = not gates["R512"]["coverage_pool_bad"] and not gates["R512"]["coverage_synth_bad"]
    dn_ok = not any(v["Q"] == "BIASED" or v["B1"] == "BIASED" for v in dn_states.values())
    ok512 = cov512 and not power_bad and fail_pooled <= 0.01 and not pred64
    decision = {"direct_null_states": dn_states, "direct_null_ok": dn_ok, "coverage_ok_R512": cov512}
    if not dn_ok:
        decision.update(R=None, outcome="STOP_INCONCLUSIVE", reason="direct R0-versus-R0 null produced a BIASED state")
    elif ok512:
        decision.update(R=512, outcome="PROCEED")
    else:
        gates["R1024"] = design_gates(1024)
        cov1024 = not gates["R1024"]["coverage_synth_bad"] and not gates["R512"]["coverage_pool_bad"]
        pred128 = {k: v["predicted_fail_n128"] for k, v in res["settings"].items() if v["predicted_fail_n128"] > 0.01}
        powp = {k: v["predicted_power_n128"] for k, v in res["settings"].items() if v["predicted_power_n128"] < 0.80}
        gates["R1024"].update(predicted_fail_bad=pred128, predicted_power_bad=powp)
        if not cov512 and not cov1024:
            decision.update(R=None, outcome="STOP_INCONCLUSIVE", reason="Q calibration cannot be qualified (coverage)")
        elif cov1024 and not pred128 and not powp:
            decision.update(R=1024, outcome="PROCEED")
        else:
            decision.update(R=None, outcome="STOP_INCONCLUSIVE", reason="budget rule cannot reach the precision / power requirement")
    res["gates"] = gates
    res["decision"] = decision
    res["synth_json_sha256"] = sha(synth_json)
    res["code_sha256"] = sha(Path(__file__))
    res["protocol109a_sha256"] = sha(HERE / "protocol109a.json")
    write(out / "pool.json", res)
    write(out / "budget109a.json", {"decision": decision, "pool_json_sha256": sha(out / "pool.json"),
                                    "protocol109a_sha256": res["protocol109a_sha256"]})
    print("BUDGET", decision["outcome"], decision.get("R"), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "Stage B analyses run CPU-only (CUDA_VISIBLE_DEVICES='')"
    if a.cmd == "hist":
        hist(Path(a.args[0]).resolve(), Path(a.args[1]).resolve())
    elif a.cmd == "synth":
        synth(Path(a.args[0]).resolve(), a.workers)
    elif a.cmd == "pool":
        pool(Path(a.args[0]).resolve(), Path(a.args[1]).resolve(), Path(a.args[2]).resolve(), a.smoke)
    else:
        raise KeyError(a.cmd)
