"""WL104J Stages 4-5: independent angular evaluation of A / B / C / C_noF (and external controls), contrasts, transfer,
reciprocity, synthetic controls, convergence diagnostics and lobe diagnostics.

Usage: python eval104j.py <run_root> <out_dir>
  run_root contains data/, fit_primary/, fit_extended/, synth_S0a/, synth_S0b/, synth_S1/, synth_S2/.
"""
import argparse
import time
from pathlib import Path

import numpy as np

import common104j as CM

GF, IO = CM.GF, CM.IO
LUMA = CM.LUMA
REGIONS = {"core": (0, 3), "shoulder": (3, 12), "tail": (12, 181)}
EFF = CM.P["evaluation"]
MAP_PROBES = [2, 6, 10, 18]           # polar 20, 50, 60, 86 at azimuth 22.5


def load_states(path, R, n):
    with np.load(path) as z:
        raw = {k: z[k] for k in z.files}
    out, assigned = {}, set()
    for arm in sorted(CM.ARMS, key=len, reverse=True):                 # C_noF before C
        keys = [k for k in raw if k.startswith(arm + "_") and k not in assigned]
        assigned.update(keys)
        if not keys:
            continue
        st = {k[len(arm) + 1:]: raw[k] for k in keys}
        out[arm] = [{k: v.reshape(R, n, *v.shape[1:])[r] for k, v in st.items()} for r in range(R)]
    truth = {k[6:]: v for k, v in raw.items() if k.startswith("truth_")}
    return out, truth


def as4(d, n):
    return {k: v.reshape(n, -1, *v.shape[1:]) for k, v in d.items()}


def metrics(pred, ref, rad=None):
    """pred [n, P, L, 3]; ref: dict(wi, wo, w, mean, a, b) -> per-probe metric arrays [n, P]."""
    w = ref["w"]
    out = {k: v for k, v in CM.wl82_terms(pred, ref["mean"], w).items()}
    if "a" in ref:
        for h in ("a", "b"):
            t = CM.wl82_terms(pred, ref[h], w)
            out[f"el1_{h}"], out[f"eshape_{h}"], out[f"ealb_{h}"] = t["el1"], t["eshape"], t["ealb"]
    if rad is not None:
        pl, rl = (pred @ LUMA) * w, (ref["mean"] @ LUMA) * w
        for k, (lo, hi) in REGIONS.items():
            m = (rad >= lo) & (rad < hi)
            den = np.maximum((rl * m).sum(-1), 1e-20)
            out[f"{k}_signed"] = ((pl - rl) * m).sum(-1) / den
            out[f"{k}_l1"] = (np.abs(pl - rl) * m).sum(-1) / den
            if "a" in ref:
                ra, rb = [((ref[h] @ LUMA) * w * m).sum(-1) for h in ("a", "b")]
                out[f"{k}_ref_half_rel"] = 0.5 * np.abs(ra - rb) / den
    return out


def peak_metric(arm, st, peak, n):
    pred = CM.predict(arm, st, peak["wi"], peak["wo"], CM.PROBE_GROUP) @ LUMA
    return pred.max(-1) / np.maximum(peak["ref"].max(-1), 1e-20) - 1


def evaluate_state(arm, st, sets, n):
    ev, fit = sets["eval"], sets["fit"]
    res = {"eval": metrics(CM.predict(arm, st, ev["wi"], ev["wo"], CM.PROBE_GROUP), ev, ev["rad"]),
           "fit": metrics(CM.predict(arm, st, fit["wi"], fit["wo"], CM.PROBE_GROUP), fit)}
    res["eval"]["peak_signed"] = peak_metric(arm, st, sets["peak"], n)
    t1 = sets["polar_transfer"]
    if arm == "A":
        res["polar_transfer"] = metrics(CM.predict(arm, st, t1["wi"], t1["wo"], t1["lo"]), t1)
    else:
        lo = metrics(CM.predict(arm, st, t1["wi"], t1["wo"], t1["lo"]), t1)
        hi = metrics(CM.predict(arm, st, t1["wi"], t1["wo"], t1["hi"]), t1)
        res["polar_transfer"] = {**{f"{k}_lower": v for k, v in lo.items()}, **{f"{k}_upper": v for k, v in hi.items()},
                                 **{k: 0.5 * (lo[k] + hi[k]) for k in lo}}
    t2 = sets["azimuth_transfer"]
    res["azimuth_transfer"] = metrics(CM.predict(arm, st, t2["wi"], t2["wo"], CM.PROBE_GROUP), t2)
    return res


def reciprocity(arm, st, n):
    """Model-only reciprocity inconsistency on 80 directions (group polars x 8 azimuths)."""
    pol = np.repeat(CM.ANGLES, 8)
    az = np.tile(22.5 + 45 * np.arange(8), CM.G)
    D = CM.sph(pol, az)
    N = len(D)
    wi = np.broadcast_to(D[None, :, None, :], (n, N, N, 3)).copy()
    wo = np.broadcast_to(D[None, None, :, :], (n, N, N, 3)).copy()
    grp = np.repeat(np.arange(CM.G), 8)
    F = CM.predict(arm, st, wi, wo, grp) @ LUMA                 # [n, N(wi), N(wo)]
    f = F / D[None, None, :, 2]                                 # f(wi=i, wo=j)
    fr = np.swapaxes(f, 1, 2)                                   # f(wi=j, wo=i) at [i, j]
    off = ~np.eye(N, dtype=bool)
    a, b = f[:, off], fr[:, off]
    tot = np.abs(a - b).sum(1) / np.maximum((0.5 * (a + b)).sum(1), 1e-300)
    medf = np.median(0.5 * (a + b), 1, keepdims=True)
    sel = 0.5 * (a + b) > 0.01 * medf
    relp = np.abs(a - b) / np.maximum(0.5 * (a + b), 1e-300)
    p95 = np.array([np.quantile(relp[m][sel[m]], 0.95) for m in range(n)])
    return {"sum_relative": tot.tolist(), "p95_pair_relative": p95.tolist()}


def pooled(E, mats):
    """E [R, n, P] -> pooled sums [R]."""
    return E[:, mats].sum((1, 2))


def contrast(EX, EY, mats, rng, nb=10000):
    sx, sy = pooled(EX, mats), pooled(EY, mats)
    per = sx / sy - 1
    k = len(mats)
    idx = rng.integers(0, k, (nb, k))
    bx = EX[:, mats].sum(-1)[:, idx].sum(-1)                   # [R, nb]
    by = EY[:, mats].sum(-1)[:, idx].sum(-1)
    med = np.median(bx / by - 1, 0)
    return {"per_start": per.tolist(), "median": float(np.median(per)), "lo": float(np.quantile(med, .025)),
            "hi": float(np.quantile(med, .975))}


def label(c, halves):
    """c: {cohort: contrast}, halves: {cohort: {'a': contrast, 'b': contrast}} (protocol effect_labels)."""
    coh = list(c)
    sub = all(c[k]["median"] <= -0.20 and c[k]["hi"] <= -0.10 and halves[k]["a"]["median"] <= -0.15 and
              halves[k]["b"]["median"] <= -0.15 and sum(x <= -0.10 for x in c[k]["per_start"]) >= 3 for k in coh)
    meas = all(c[k]["median"] <= -0.05 and c[k]["hi"] < 0 and halves[k]["a"]["median"] < 0 and halves[k]["b"]["median"] < 0
               for k in coh)
    lab = "SUBSTANTIAL" if sub else "MEASURABLE" if meas else "NEGLIGIBLE"
    reg = any(c[k]["median"] >= 0.05 for k in coh)
    return {"label": lab, "regression": bool(reg)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rows, raw, _, desc, ref, wl104i, seen = CM.inputs()
    n = len(rows)
    ref_sha_before = CM.sha(IO.ANGULAR)
    dj = IO.read(args.root / "data/data.json")
    for f, h in dj["outputs_sha256"].items():
        if CM.sha(args.root / "data" / f) != h:
            raise IO.Stop(f"DATA_HASH {f}")
    cohorts = {c: [i for i, r in enumerate(rows) if r["cohort"] == c] for c in ("development", "held_out")}
    # ---------------- evaluation and fit / transfer sets
    R4 = as4({k: ref[k] for k in ("wi", "wo", "use", "q", "a", "b")}, n)
    ev = {"wi": R4["wi"], "wo": R4["wo"], "w": R4["use"] / R4["q"], "a": R4["a"], "b": R4["b"], "mean": 0.5 * (R4["a"] + R4["b"])}
    ev["rad"] = CM.mirror_radius(ev["wi"], ev["wo"])
    pk = as4({k: ref[k] for k in ("peak_wi", "peak_wo", "peak_a", "peak_b")}, n)
    peak = {"wi": pk["peak_wi"], "wo": pk["peak_wo"], "ref": (0.5 * (pk["peak_a"] + pk["peak_b"])) @ LUMA}

    def load_set(name):
        with np.load(args.root / "data" / f"{name}.npz") as z:
            s = as4({k: z[k] for k in ("wi", "wo", "use", "q", "a", "b")}, n)
        return {"wi": s["wi"], "wo": s["wo"], "w": s["use"] / s["q"], "a": s["a"], "b": s["b"], "mean": 0.5 * (s["a"] + s["b"])}
    fit = load_set("fit_set")
    t1 = load_set("polar_transfer_set")
    t1["lo"] = np.repeat(np.arange(CM.G - 1), 2)
    t1["hi"] = t1["lo"] + 1
    t2 = load_set("azimuth_transfer_set")
    sets = {"eval": ev, "fit": fit, "peak": peak, "polar_transfer": t1, "azimuth_transfer": t2}
    # ---------------- primary states
    fj = IO.read(args.root / "fit_primary/fit.json")
    if CM.sha(args.root / "fit_primary/states.npz") != fj["states_sha256"]:
        raise IO.Stop("STATES_HASH primary")
    R = len(fj["start_keys"])
    prim, _ = load_states(args.root / "fit_primary/states.npz", R, n)
    per = {}
    for arm, lst in prim.items():
        per[arm] = [evaluate_state(arm, st, sets, n) for st in lst]
        print("evaluated", arm, flush=True)
    # external controls (shared, arm A coordinates)
    ext = {"R0_historical": {"y": GF.from_original(raw["R0"])}}
    for k in fj["start_keys"]:
        ext[f"WL104I_A1_{k}"] = {"y": GF.from_original(wl104i[f"A1_{k}"])}
        ext[f"WL104I_B1_{k}"] = {"y": wl104i[f"B1_{k}"]}
    ext_res = {k: evaluate_state("A", st, sets, n) for k, st in ext.items()}
    # ---------------- contrasts
    rng = np.random.default_rng(CM.P["seeds"]["bootstrap"])
    arr = lambda arm, part, key: np.stack([per[arm][r][part][key] for r in range(R)])      # [R, n, P]
    pairs = {"B-A": ("B", "A"), "C-B": ("C", "B"), "C-A": ("C", "A"), "C_noF-B": ("C_noF", "B"), "C-C_noF": ("C", "C_noF")}
    con = {}
    for name, (x, y) in pairs.items():
        c = {}
        for part, key in (("eval", "el1"), ("eval", "eshape"), ("eval", "ealb"), ("fit", "el1"), ("polar_transfer", "el1"),
                          ("azimuth_transfer", "el1"), ("eval", "el1_a"), ("eval", "el1_b")):
            c[f"{part}:{key}"] = {coh: contrast(arr(x, part, key), arr(y, part, key), ms, rng) for coh, ms in cohorts.items()}
        halves = {coh: {"a": c["eval:el1_a"][coh], "b": c["eval:el1_b"][coh]} for coh in cohorts}
        c["label"] = label(c["eval:el1"], halves)
        for coh in cohorts:                                      # overfitting rule
            tr, ho = -c["fit:el1"][coh]["median"], -c["eval:el1"][coh]["median"]
            c.setdefault("overfit", {})[coh] = {"training_reduction": tr, "heldout_reduction": ho,
                                                "overfit_dominated": bool(tr >= 0.05 and ho < 0.5 * tr)}
        c["transfer_label"] = {}
        for part in ("polar_transfer", "azimuth_transfer"):
            cc = c[f"{part}:el1"]
            pos = all(cc[k]["median"] <= -0.05 and cc[k]["hi"] < 0 for k in cohorts)
            neg = any(cc[k]["median"] >= 0.05 for k in cohorts)
            c["transfer_label"][part] = "TRANSFER_POSITIVE" if pos else "TRANSFER_NEGATIVE" if neg else "TRANSFER_NEUTRAL"
        con[name] = c
    # per-material / per-probe tables (median over starts)
    tables = {}
    for arm in prim:
        tables[arm] = {k: np.median(arr(arm, "eval", k), 0).tolist() for k in per[arm][0]["eval"]}
        tables[arm]["fit_el1"] = np.median(arr(arm, "fit", "el1"), 0).tolist()
        tables[arm]["el1_per_start"] = arr(arm, "eval", "el1").tolist()
        tables[arm]["polar_transfer_el1"] = np.median(arr(arm, "polar_transfer", "el1"), 0).tolist()
        tables[arm]["azimuth_transfer_el1"] = np.median(arr(arm, "azimuth_transfer", "el1"), 0).tolist()
        if arm != "A":
            for s in ("lower", "upper"):
                tables[arm][f"polar_transfer_el1_{s}"] = np.median(arr(arm, "polar_transfer", f"el1_{s}"), 0).tolist()
    ext_tables = {k: {m: v["eval"][m].tolist() for m in ("el1", "eshape", "ealb")} for k, v in ext_res.items()}
    # residual, resolution
    resid = {coh: float(np.median(pooled(arr("C", "eval", "el1"), ms) / pooled(arr("A", "eval", "el1"), ms))) for coh, ms in cohorts.items()}
    noise = {}
    for coh, ms in cohorts.items():
        nz = np.abs(arr("A", "eval", "el1_a") - arr("A", "eval", "el1_b"))[:, ms] / 2
        noise[coh] = {"half_split_el1_noise": float(nz.mean()), "A_el1": float(arr("A", "eval", "el1")[:, ms].mean()),
                      "ratio": float(nz.mean() / arr("A", "eval", "el1")[:, ms].mean())}
        noise[coh]["resolves"] = bool(noise[coh]["ratio"] <= 0.0125)
    # start stability
    stab = {name: {coh: int(sum(np.sign(x) == np.sign(con[name]["eval:el1"][coh]["median"]) for x in con[name]["eval:el1"][coh]["per_start"]))
                   for coh in cohorts} for name in ("B-A", "C-B", "C-A")}
    # ---------------- convergence
    conv = {}
    for arm in ("A", "B", "C", "C_noF"):
        if fj[arm]["last_2000_relative_change"] is None:                  # run shorter than 2,000 steps (dev only)
            conv[arm] = {"fraction_units_le_0.5pct": 0.0, "max": None}
            continue
        lc = np.array(fj[arm]["last_2000_relative_change"])
        conv[arm] = {"fraction_units_le_0.5pct": float((lc <= 0.005).mean()), "max": float(lc.max())}
    obj = {arm: np.array(fj[arm]["final_objective"]) for arm in ("A", "B", "C")}
    if "C_fallback_used" in fj:
        objF = np.array(fj["C_fallback_run"]["final_objective"])
        obj["C"] = np.where(np.array(fj["C_fallback_used"]), objF, obj["C"])
    nest = {}
    for coh, ms in cohorts.items():
        vb = obj["B"][:, ms] > obj["A"][:, ms] * 1.001
        vc = obj["C"][:, ms] > obj["B"][:, ms] * 1.001
        nest[coh] = {"B_above_A": int(vb.sum()), "C_above_B": int(vc.sum()), "units": int(vb.size),
                     "fraction": float((vb | vc).mean())}
    xj = IO.read(args.root / "fit_extended/fit.json")
    ext_obj = {arm: np.array(xj[arm]["final_objective"])[0] for arm in ("A", "B", "C")}
    g0 = fj["start_keys"].index("generic")
    ext_chk = {arm: float(np.median(np.abs(ext_obj[arm] - obj[arm][g0]) / obj[arm][g0])) for arm in ("A", "B", "C")}
    tc = {}
    for nm, (x, y) in (("B-A", ("B", "A")), ("C-B", ("C", "B"))):
        d12 = float(obj[x][g0].sum() - obj[y][g0].sum())
        dex = float(ext_obj[x].sum() - ext_obj[y].sum())
        tc[nm] = {"12k": d12, "24k": dex, "relative_change": abs(dex - d12) / max(abs(d12), 1e-12)}
    xs, _ = load_states(args.root / "fit_extended/states.npz", 1, n)
    ext_eval = {arm: evaluate_state(arm, xs[arm][0], sets, n)["eval"]["el1"].tolist() for arm in xs}
    conv_ok = (all(v["fraction_units_le_0.5pct"] >= 0.9 for v in conv.values()) and all(v["fraction"] <= 0.25 for v in nest.values())
               and all(v <= 0.01 for v in ext_chk.values()) and all(v["relative_change"] <= 0.25 for v in tc.values()))
    convergence = {"last_2000": conv, "nesting": nest, "extended_objective_median_relative_change": ext_chk,
                   "extended_training_contrast": tc, "extended_heldout_el1": ext_eval, "resolved": bool(conv_ok),
                   "C_fallback_triggered": fj.get("C_fallback_triggered"), "C_fallback_used": fj.get("C_fallback_used")}
    # ---------------- synthetic controls
    syn = {}
    for ctl in ("S0a", "S0b", "S1", "S2"):
        d = args.root / f"synth_{ctl}"
        sj = IO.read(d / "fit.json")
        if CM.sha(d / "states.npz") != sj["states_sha256"]:
            raise IO.Stop(f"STATES_HASH {ctl}")
        ss, truth = load_states(d / "states.npz", 1, n)
        tgt = CM.predict(sj["truth_arm"], truth, ev["wi"], ev["wo"], CM.PROBE_GROUP)
        tref = {"wi": ev["wi"], "wo": ev["wo"], "w": ev["w"], "mean": tgt}
        e = {arm: CM.wl82_terms(CM.predict(arm, ss[arm][0], ev["wi"], ev["wo"], CM.PROBE_GROUP), tgt, ev["w"])["el1"] for arm in ss}
        ftr = {arm: np.array(sj[arm]["final_objective"])[0].tolist() for arm in ss}
        s = {"heldout_el1_mean": {arm: float(v.mean()) for arm, v in e.items()},
             "heldout_el1_mean_by_cohort": {coh: {arm: float(v[ms].mean()) for arm, v in e.items()} for coh, ms in cohorts.items()},
             "heldout_el1_per_material": {arm: v.mean(1).tolist() for arm, v in e.items()}, "training_objective": ftr,
             "truth_objective_on_noisy_target": sj["truth_objective_on_noisy_fit_target"]}
        m = s["heldout_el1_mean"]
        if ctl == "S0a":
            s["pass"] = bool(abs(m["B"] - m["A"]) <= 1e-3 and abs(m["C"] - m["A"]) <= 1e-3)
        elif ctl == "S1":
            s["pass"] = bool(m["B"] <= 0.5 * m["A"])
        elif ctl == "S2":
            s["pass"] = bool(m["C"] <= 0.5 * m["B"])
        syn[ctl] = s
    slack = {}
    for coh, ms in cohorts.items():
        sb = syn["S0b"]["heldout_el1_mean_by_cohort"][coh]
        mean_med = lambda arm: float(np.median(arr(arm, "eval", "el1")[:, ms].mean((1, 2))))
        phys = {"B-A": mean_med("A") - mean_med("B"), "C-A": mean_med("A") - mean_med("C"), "C-B": mean_med("B") - mean_med("C")}
        null = {"B-A": sb["A"] - sb["B"], "C-A": sb["A"] - sb["C"], "C-B": sb["B"] - sb["C"]}
        slack[coh] = {k: {"physical_gain": phys[k], "null_generic_gain": null[k],
                          "separable": bool(phys[k] >= 2 * max(null[k], 0) and phys[k] > 0)} for k in phys}
    # ---------------- reciprocity
    recip = {arm: [reciprocity(arm, st, n) for st in prim[arm]] for arm in prim}
    # ---------------- best start (training objective) per arm and material; lanes, terms and parameter profiles
    best = {}
    for arm in prim:
        o = np.array(fj[arm]["final_objective"]) if arm != "C" else obj["C"]
        best[arm] = np.argmin(o, 0)                                        # [n]
    bst = {arm: {k: np.stack([prim[arm][best[arm][m]][k][m] for m in range(n)]) for k in prim[arm][0]} for arm in prim}
    lanes = {"ref_mean": ev["mean"], "ref_a": ev["a"], "ref_b": ev["b"], "w": ev["w"], "wi": ev["wi"], "wo": ev["wo"],
             "peak_wo": peak["wo"], "peak_wi": peak["wi"], "peak_ref": peak["ref"], "fit_wi": fit["wi"], "fit_wo": fit["wo"],
             "fit_w": fit["w"], "fit_ref": fit["mean"]}
    for arm in prim:
        lanes[f"pred_{arm}"] = CM.predict(arm, bst[arm], ev["wi"], ev["wo"], CM.PROBE_GROUP)
        lanes[f"peak_{arm}"] = CM.predict(arm, bst[arm], peak["wi"], peak["wo"], CM.PROBE_GROUP) @ LUMA
        lanes[f"fitpred_{arm}"] = CM.predict(arm, bst[arm], fit["wi"], fit["wo"], CM.PROBE_GROUP)
    lanes["pred_R0"] = CM.predict("A", ext["R0_historical"], ev["wi"], ev["wo"], CM.PROBE_GROUP)
    names = ("O0", "O1", "O2", "SL", "SI0", "SI1", "SI2", "SI3")
    lobe = {}
    for arm in ("A", "B", "C"):
        T = CM.terms(arm, bst[arm], ev["wi"], ev["wo"], CM.PROBE_GROUP)
        w = ev["w"]
        C = np.zeros((n, CM.NP, 8, 8))
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                num = (w[..., None] * T[a] * T[b]).sum((2, 3))
                den = np.sqrt((w[..., None] * T[a] ** 2).sum((2, 3)) * (w[..., None] * T[b] ** 2).sum((2, 3)))
                C[..., i, j] = num / np.maximum(den, 1e-300)
        en = np.stack([((T[a] @ LUMA) * w).sum(-1) for a in names], -1)
        share = en / np.maximum(en.sum(-1, keepdims=True), 1e-300)
        iu = np.triu_indices(8, 1)
        lobe[arm] = {"cos": C, "share": share, "pairs_ge_095": (C[..., iu[0], iu[1]] >= 0.95).sum(-1)}
        for i, a in enumerate(names):
            lanes[f"term_{arm}_{a}"] = (T[a][:, MAP_PROBES] @ LUMA).astype(np.float32)
    # parameter profiles (best start; amplitudes on the common section scale)
    prof = {}
    for arm in prim:
        st = bst[arm]
        loc = CM.expand_groups(CM.section(st["y"])) if arm == "A" else st
        p = CM.lane_params("B" if arm == "A" else arm, loc)
        prof[arm] = {k: v.detach().cpu().numpy().reshape(n, CM.G, *v.shape[1:]).tolist() for k, v in p.items() if k in ("lam", "au", "av", "f0")}
        prof[arm]["s_n"] = np.asarray(loc["s_n"]).tolist()
        prof[arm]["eta"] = np.asarray(loc["eta"]).tolist()
    np.savez_compressed(args.out / "lanes.npz", **{k: np.asarray(v, np.float32) for k, v in lanes.items()},
                        **{f"lobe_{arm}_{k}": v for arm, d in lobe.items() for k, v in d.items()})
    if CM.sha(IO.ANGULAR) != ref_sha_before:
        raise IO.Stop("EVALUATION_REFERENCE_CHANGED")
    res = {"materials": rows, "cohorts": cohorts, "probe_polar": CM.POLAR.tolist(), "start_keys": fj["start_keys"],
           "contrasts": con, "tables": tables, "external_controls": ext_tables, "residual_C_over_A": resid,
           "reference_resolution": noise, "start_stability": stab, "convergence": convergence, "synthetic": syn,
           "slack": slack, "reciprocity": recip, "best_start": {k: v.tolist() for k, v in best.items()},
           "lobe_pairs_ge_095": {arm: d["pairs_ge_095"].tolist() for arm, d in lobe.items()},
           "lobe_share": {arm: d["share"].tolist() for arm, d in lobe.items()}, "parameter_profiles": prof,
           "fit_cost": {arm: {"seconds": fj[arm]["seconds"], "peak_allocated_bytes": fj[arm]["peak_allocated_bytes"],
                              "trainable_per_material": fj["trainable_per_material"][arm]} for arm in CM.ARMS},
           "fit_cost_C_fallback_seconds": fj.get("C_fallback_run", {}).get("seconds"),
           "data": {"precision": dj["precision"], "support": dj["support"], "isolation": dj["isolation"],
                    "evaluation_reference_el1_half_noise": dj["evaluation_reference_el1_half_noise"]},
           "commit": CM.commit(), "protocol_sha256": CM.sha(CM.HERE / "protocol.json"), "inputs_sha256": seen,
           "lanes_sha256": CM.sha(args.out / "lanes.npz"), "seconds": time.monotonic() - t0}
    CM.write(args.out / "eval.json", res)
    print("eval done", flush=True)


if __name__ == "__main__":
    main()
