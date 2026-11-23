"""WL109B Stage C: sealed statistical qualification of the frozen XJL procedure (CPU only).

python calib109b.py synth OUT [--stage 1|2] [--workers W]
    Synthetic known-truth families at the real ROI shapes and both candidate designs. Stage 1 = the base realizations;
    stage 2 = the uniform escalation (every cell, new seeds), run only when the decision step asks for it.
python calib109b.py pool CAPTURE_DIR OUT [--workers W]
    Independent GPU R0 null pool (exact zero expectation): random block re-groupings at both designs, Q = 0 and injected
    known biases, per-setting equivalence precision, power, and two disjoint direct R0-versus-R0 decisions.
python calib109b.py decide OUT
    Applies the frozen gates and the budget rule to synth (stage 1 [+ 2]) and pool; writes budget109b.json with outcome
    PROCEED (R), ESCALATE, or STOP_INCONCLUSIVE.

Nothing here reads a C1 frame.
"""
import argparse
import math
import os
from pathlib import Path
import time

import numpy as np

from common109b import (P, BIAS_SPP, DECISION_ROIS, EVAL_VIEWS, QLIM, LUMA, all_masks, b1_state, bias_pattern, clopper_pearson,
                        coverage_class, groups_from_subgroups, lam_max, read, sha, write, xjl_bounds, xy_of)

C = P["calibration"]
INF = P["inference"]
QT = C["injection"]["Q_true_over_limit"]
PATTERNS = C["injection"]["patterns"]
G = INF["groups"]
SUB = INF["subgroup_pairs"]
DESIGNS = list(C["designs_R"])
SMOKE = False


def set_smoke():
    """smoke: tiny designs that exercise every code path (G = 4 groups of one 16-pair subgroup, R = 64)."""
    global G, DESIGNS, SMOKE
    G, DESIGNS, SMOKE = 4, [64], True
TOL = C["gates"]["coverage_tolerance"]


def tally(acc, key, q, lim, r):
    a = acc.setdefault(key, {"noncov": 0, "fail": 0, "power": 0, "upper_le_limit": 0, "n": 0})
    a["noncov"] += int(r["upper"] < q); a["fail"] += int(r["upper"] > lim); a["power"] += int(r["lower_family"] > lim)
    a["upper_le_limit"] += int(r["upper"] <= lim); a["n"] += 1


def evaluate_groups(sub_means, scale, lim, X, R, av, rs, acc):
    """sub_means [k, X] means of 16-pair subgroups of D. Fixed injections shift every subgroup / group mean by the same vector
    (exact; lam is unaffected because it is computed from centred subgroup means)."""
    lam = lam_max(sub_means, SUB)
    gm0 = groups_from_subgroups(sub_means * SUB, SUB, G)
    tally(acc, "none@0", 0.0, lim, xjl_bounds(gm0, scale, lam, R))
    for f in QT:
        if f == 0:
            continue
        q = f * lim
        for pat in PATTERNS:
            b = bias_pattern(pat, X // 3, av, rs).reshape(1, -1)
            tally(acc, f"{pat}@{f}", q, lim, xjl_bounds(gm0 + math.sqrt(q * scale) * b, scale, lam, R))


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


def _smooth_matrix(xy, sigma, radius):
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    W = np.where(d <= radius, np.exp(-0.5 * (d / sigma) ** 2), 0.0)
    return W / np.sqrt((W * W).sum(1, keepdims=True))


W3 = np.array([1.0, 0.8, 0.6])


def synth_noise(fam, rs, R, Pp, smooth):
    """-> D [R, 3, Pp] (zero mean) and its per-element variance [3, Pp]; definitions in protocol109b calibration.synthetic."""
    if fam == "gauss_iid" or fam.startswith("heteroscedastic"):
        z = math.sqrt(0.9) * rs.standard_normal((R, 1, Pp)) + math.sqrt(0.1) * rs.standard_normal((R, 3, Pp))
        sd = np.ones(Pp) if fam == "gauss_iid" else np.exp(float(fam.split("_")[1]) * rs.standard_normal(Pp))
        return z * sd[None, None, :], np.repeat((sd ** 2)[None], 3, 0)
    if fam in ("mc_like_4", "mc_like_32", "correlated", "het_mc4"):
        k = 32 if fam == "mc_like_32" else 4
        base = _mc_like(rs, (R, Pp), k) - _mc_like(rs, (R, Pp), k)
        var = 2 * _mc_var(k) * np.ones(Pp)
        if fam == "correlated":
            base = base @ smooth.T
        if fam == "het_mc4":
            sd = np.exp(rs.standard_normal(Pp))
            base = base * sd[None, :]; var = var * sd ** 2
        return base[:, None, :] * W3[None, :, None], (W3[:, None] ** 2) * var[None, :]
    if fam == "asym_mc4":
        k = 4; v = _mc_var(k); m = 0.3 * math.exp(1.5 ** 2 / 2)
        X = _mc_like(rs, (R, Pp), k)
        Y = m + math.sqrt(v) * rs.standard_normal((R, Pp))
        return (X - Y)[:, None, :] * W3[None, :, None], (W3[:, None] ** 2) * (2 * v) * np.ones((1, Pp))
    raise KeyError(fam)


def _synth_cell(args):
    fam, view, roi, R, nrep, seed, idx = args
    S = C["synthetic"]
    rs = np.random.default_rng(seed)
    Pp = idx.size; xy = xy_of(idx)
    smooth = _smooth_matrix(xy, S["correlated_sigma_px"], S["correlated_radius_px"]) if fam == "correlated" else None
    lim = QLIM[roi]
    acc = {}
    t0 = time.time()
    for _ in range(nrep):
        D, var = synth_noise(fam, rs, R, Pp, smooth)
        scale = float(var.mean() / S["varD_over_scale"])
        sm = D.reshape(R // SUB, SUB, -1).mean(1)
        evaluate_groups(sm, scale, lim, 3 * Pp, R, {"noise_sd": np.sqrt(var.mean(0)), "xy": xy}, rs, acc)
    return f"{fam}/{view}:{roi}/R{R}", {"counts": acc, "pixels": int(Pp), "elapsed_s": time.time() - t0}


def synth(out, stage, workers):
    import runtime109 as RT
    from multiprocessing import Pool
    out.mkdir(parents=True, exist_ok=True)
    S = C["synthetic"]
    shapes = {(v, r): np.flatnonzero(all_masks(RT.aux(v))[r].reshape(-1)) for v, r in S["roi_shapes"]}
    nrep = (S["realizations_stage1"] if stage == 1 else S["realizations_stage2"]) if not SMOKE else 3
    base = P["seeds"]["calibration_rng"] + (1000 if stage == 1 else 500000)
    jobs = []; j = 0
    for fam in S["families"]:
        for (v, r), idx in shapes.items():
            for R in DESIGNS:
                jobs.append((fam, v, r, R, nrep, base + j, idx)); j += 1
    t0 = time.time()
    with Pool(workers) as pool:
        res = dict(pool.imap_unordered(_synth_cell, jobs))
    write(out / f"synth_stage{stage}.json", {"cells": res, "stage": stage, "realizations": nrep, "code_sha256": sha(Path(__file__)),
                                            "protocol109b_sha256": sha(Path(__file__).parent / "protocol109b.json"),
                                            "elapsed_s": time.time() - t0})
    print("synth stage", stage, "done", len(res), round(time.time() - t0, 1), flush=True)


# ------------------------------------------------------------------------------------------------ GPU null pool
def _pool_setting(args):
    import runtime109 as RT
    cap, view, spp, smoke = args
    if smoke:
        set_smoke()
    cfg = C["null_pool"]
    u = cap / "null" / f"{view}_spp{spp}"
    rec = read(u / "unit.json")
    for f, h in rec["files"].items():
        assert sha(u / f) == h, (u, f)
    masks = all_masks(RT.aux(view))
    refc, vref, refrec = RT.reference(view)
    ref = refc[0:3] + refc[3:6] + refc[6:9]
    Bk = np.load(u / "B1.npy", mmap_mode="r")
    nb, bp, N0 = rec["blocks"], rec["block_pairs"], rec["frames"]
    assert bp == SUB
    sT = np.load(u / "sT.npy").reshape(3, -1); sT2 = np.load(u / "sT2.npy").reshape(3, -1)
    varT = (sT2 - sT * sT / N0) / (N0 - 1)
    lumH = np.load(u / "lumH.npy"); names = rec["mask_names"]
    rs = np.random.default_rng(P["seeds"]["calibration_rng"] + 7000 + 100 * EVAL_VIEWS.index(view) + spp)
    out = {"settings": {}, "direct": {}, "b1_direct": {}}
    for roi in [r for r in DECISION_ROIS if r in masks]:
        idx = np.flatnonzero(masks[roi].reshape(-1))
        X = 3 * idx.size
        bm = np.asarray(Bk.reshape(nb, 3, -1)[:, :, idx], np.float64).reshape(nb, -1) / bp     # block means of D [nb, X]
        varD = bm.var(0, ddof=1) * bp
        scale = float(varT[:, idx].mean())
        lim = QLIM[roi]
        av = {"noise_sd": np.sqrt(varD.reshape(3, -1).mean(0)), "xy": xy_of(idx)}
        for R in DESIGNS:
            k = R // bp
            acc = {}
            for _ in range(cfg["draws"] if not SMOKE else 5):
                sel = rs.choice(nb, k, replace=False)
                evaluate_groups(bm[sel], scale, lim, X, R, av, rs, acc)
            out["settings"][f"{view}/{roi}/{spp}/R{R}"] = {"counts": acc, "pixels": int(idx.size), "scale": scale,
                                                         "varD_over_scale": float(varD.mean() / scale)}
            for h in range(2):                                                    # disjoint direct decisions
                sel = np.arange(h * k, (h + 1) * k)
                lam = lam_max(bm[sel], bp)
                r = xjl_bounds(groups_from_subgroups(bm[sel] * bp, bp, G), scale, lam, R)
                out["direct"][f"{view}/{roi}/{spp}/R{R}/half{h}"] = r
                ki = names.index(roi)
                dl = lumH[0::2, ki] - lumH[1::2, ki]
                out["b1_direct"].setdefault(f"{view}/{roi}/R{R}/half{h}", {})[spp] = dl[h * R:(h + 1) * R].tolist()
        out.setdefault("ref_luma", {})[roi] = float((LUMA @ ref[:, masks[roi]]).mean())
    print(view, spp, "pool done", flush=True)
    return out


def pool(cap, out, workers):
    from multiprocessing import Pool
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(cap, v, s, SMOKE) for v in EVAL_VIEWS for s in BIAS_SPP if (cap / "null" / f"{v}_spp{s}").exists()]
    assert SMOKE or len(jobs) == 18, len(jobs)
    t0 = time.time()
    with Pool(workers) as p:
        parts = p.map(_pool_setting, jobs)
    res = {"settings": {}, "direct": {}, "direct_states": {}, "code_sha256": sha(Path(__file__)), "elapsed_s": None}
    b1 = {}; refl = {}
    for (cap_, v, s, _), part in zip(jobs, parts):
        res["settings"].update(part["settings"]); res["direct"].update(part["direct"])
        for k, d in part["b1_direct"].items():
            b1.setdefault(k, {}).update(d)
        for roi, x in part["ref_luma"].items():
            refl[f"{v}/{roi}"] = x
    for R in DESIGNS:
        for v in EVAL_VIEWS:
            for roi in DECISION_ROIS:
                for h in range(2):
                    rr = {s: res["direct"].get(f"{v}/{roi}/{s}/R{R}/half{h}") for s in BIAS_SPP}
                    if None in rr.values():
                        continue
                    lim = QLIM[roi]
                    qs = ("BIASED" if any(x["lower_family"] > lim for x in rr.values()) else
                          ("EQUIVALENT" if all(x["upper"] <= lim for x in rr.values()) else "UNRESOLVED"))
                    bs = b1_state({s: np.asarray(b1[f"{v}/{roi}/R{R}/half{h}"][s]) for s in BIAS_SPP}, refl[f"{v}/{roi}"])["state"]
                    res["direct_states"][f"R{R}/{v}/{roi}/half{h}"] = {"Q": qs, "B1": bs}
    res["elapsed_s"] = time.time() - t0
    write(out / "pool109b.json", res)
    print("pool done", round(time.time() - t0, 1), flush=True)


# ------------------------------------------------------------------------------------------------ gates and budget
def merge_synth(paths):
    cells = {}
    for p in paths:
        for name, c in read(p)["cells"].items():
            m = cells.setdefault(name, {})
            for k, a in c["counts"].items():
                b = m.setdefault(k, {kk: 0 for kk in a})
                for kk, vv in a.items():
                    b[kk] += vv
    return cells


def design_gates(R, synth_cells, pool_res, escalated):
    g = {"R": R}
    # A. synthetic coverage, every family / shape / Q_true / pattern entry
    syn = {}
    worst = None
    for name, cell in synth_cells.items():
        if not name.endswith(f"/R{R}"):
            continue
        for k, a in cell.items():
            q = float(k.split("@")[1])
            cls, lo, hi = coverage_class(a["noncov"], a["n"], TOL)
            p = a["noncov"] / a["n"]
            if escalated and cls == "BORDERLINE":
                cls = "PASS" if p <= TOL else "FAIL"
            syn[f"{name}/{k}"] = {"class": cls, "rate": p, "cp95": [lo, hi], "n": a["n"], "Q_true_over_limit": q,
                                  "fail_rate_at_Q0": a["fail"] / a["n"] if q == 0 else None,
                                  "power_family": a["power"] / a["n"] if q == 2 else None}
            if worst is None or p > worst[1]:
                worst = (f"{name}/{k}", p)
    cls = [v["class"] for v in syn.values()]
    g["synthetic"] = {"entries": len(syn), "FAIL": cls.count("FAIL"), "BORDERLINE": cls.count("BORDERLINE"), "PASS": cls.count("PASS"),
                      "worst": worst, "failed": [k for k, v in syn.items() if v["class"] == "FAIL"],
                      "borderline": [k for k, v in syn.items() if v["class"] == "BORDERLINE"]}
    g["synthetic_entries"] = syn
    # B. real null pool (pooled over the 84 view x ROI x spp settings)
    sets = {k: v for k, v in pool_res["settings"].items() if k.endswith(f"/R{R}")}
    pooled = {}
    for k, v in sets.items():
        for kk, a in v["counts"].items():
            pooled.setdefault(kk, []).append(a)
    real_cov = {}
    for kk, lst in pooled.items():
        rate = float(np.mean([a["noncov"] / a["n"] for a in lst]))
        real_cov[kk] = {"pooled_noncov": rate, "pooled_power_family": float(np.mean([a["power"] / a["n"] for a in lst])),
                        "settings": len(lst)}
    g["real_coverage"] = real_cov
    g["real_coverage_ok"] = all(v["pooled_noncov"] <= TOL for v in real_cov.values())
    fail0 = {k: v["counts"]["none@0"]["fail"] / v["counts"]["none@0"]["n"] for k, v in sets.items()}
    g["precision"] = {"max_setting_fail": max(fail0.values()), "worst_setting": max(fail0, key=fail0.get),
                      "joint_pass_probability": float(np.prod([1 - x for x in fail0.values()])),
                      "settings_over_limit": {k: x for k, x in fail0.items() if x > C["gates"]["precision_setting_fail_max"]}}
    g["precision_ok"] = (g["precision"]["max_setting_fail"] <= C["gates"]["precision_setting_fail_max"]
                         and g["precision"]["joint_pass_probability"] >= C["gates"]["precision_joint_pass_min"])
    pw = {kk: v["pooled_power_family"] for kk, v in real_cov.items() if kk.endswith("@2.0") or kk.endswith("@2")}
    g["power"] = pw
    g["power_ok"] = all(x >= C["gates"]["power_min"] for x in pw.values())
    ds = {k: v for k, v in pool_res["direct_states"].items() if k.startswith(f"R{R}/")}
    g["direct_null"] = {"states": ds, "biased": [k for k, v in ds.items() if "BIASED" in v.values()],
                        "equivalent_Q": sum(v["Q"] == "EQUIVALENT" for v in ds.values()), "n": len(ds)}
    g["direct_null_ok"] = not g["direct_null"]["biased"]
    g["synthetic_ok"] = g["synthetic"]["FAIL"] == 0 and g["synthetic"]["BORDERLINE"] == 0
    g["all_ok"] = bool(g["synthetic_ok"] and g["real_coverage_ok"] and g["precision_ok"] and g["power_ok"] and g["direct_null_ok"])
    return g


def decide(out):
    s1 = out / "synth_stage1.json"; s2 = out / "synth_stage2.json"
    escalated = s2.exists()
    cells = merge_synth([s1] + ([s2] if escalated else []))
    pool_res = read(out / "pool109b.json")
    gates = {f"R{R}": design_gates(R, cells, pool_res, escalated) for R in DESIGNS}
    rec = {"protocol109b_sha256": sha(Path(__file__).parent / "protocol109b.json"), "escalated": escalated,
           "inputs": {p.name: sha(p) for p in [s1, s2, out / "pool109b.json"] if p.exists()}, "code_sha256": sha(Path(__file__))}
    summary = {k: {kk: vv for kk, vv in g.items() if kk not in ("synthetic_entries",)} for k, g in gates.items()}
    for k, g in summary.items():
        g["direct_null"] = {kk: vv for kk, vv in g["direct_null"].items() if kk != "states"}
    rec["gates"] = summary
    any_fail_cov = {R: gates[f"R{R}"]["synthetic"]["FAIL"] > 0 or not gates[f"R{R}"]["real_coverage_ok"] for R in DESIGNS}
    border = any(gates[f"R{R}"]["synthetic"]["BORDERLINE"] > 0 for R in DESIGNS)
    decision = None
    for R in DESIGNS:                                           # ascending budgets: the first fully qualified design
        if gates[f"R{R}"]["all_ok"]:
            decision = {"outcome": "PROCEED", "R": R}
            break
    if decision is None:
        if border and not escalated and not all(any_fail_cov.values()):
            decision = {"outcome": "ESCALATE", "R": None, "reason": "borderline synthetic coverage entries; uniform escalation of every cell"}
        else:
            decision = {"outcome": "STOP_INCONCLUSIVE", "R": None,
                        "reason": "no design passes every frozen gate" + (" after escalation" if escalated else "")}
    if SMOKE:
        decision = {"outcome": "SMOKE", "R": 64, "would_be": decision}
    rec["decision"] = decision
    write(out / "budget109b.json", rec)
    write(out / "synthetic_entries109b.json", {k: g["synthetic_entries"] for k, g in gates.items()})
    print("BUDGET", decision, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode"); ap.add_argument("a"); ap.add_argument("b", nargs="?")
    ap.add_argument("--stage", type=int, default=1); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        set_smoke()
    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "calibration runs CPU-only"
    if a.mode == "synth":
        synth(Path(a.a).resolve(), a.stage, a.workers)
    elif a.mode == "pool":
        pool(Path(a.a).resolve(), Path(a.b).resolve(), a.workers)
    elif a.mode == "decide":
        decide(Path(a.a).resolve())
    else:
        raise KeyError(a.mode)
