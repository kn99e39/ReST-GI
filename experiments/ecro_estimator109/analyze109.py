"""WL109 frozen analysis and verdict.

python analyze109.py RUN_DIR DEV_JSON SYNTH_JSON METRICS_OUT_DIR [--dynamic DYN_JSON]

Target: the frozen ECRO expectation, estimated by the WL107 qualified 48 x 8192-spp reference (independent of C1).
MSE(view, ROI, arm, spp) = mean_r mean_{ROI,RGB} (Y_r - ref)^2 - mean_{ROI,RGB} Var(ref)   (replicates r are frames)
Equal-time ratio rho_v(n) = MSE_C1,v(n) / MSE_R0,v(t_C1,v(n)), R0 interpolated log-log between bracketing measured grid levels.
Uncertainty: two-level bootstrap (views; replicates and timing samples within view), normal interval on log GM.
"""
import argparse
import math
from pathlib import Path
import time

import numpy as np

from common109 import (P, DECISION_ROIS, EVAL_VIEWS, GRID, LUMA, NOMINAL, apply_smoke, distance_bands, gm, read, rois, sha, write, HERE)
import runtime109 as RT

Z = 1.959963984540054


def load_paired(run, view, spp, R):
    r0 = np.stack([np.load(run / f"raw/paired/{view}/spp{spp}/r{r:02d}_R0.npy") for r in range(R)]).astype(np.float64)
    hc = np.stack([np.load(run / f"raw/paired/{view}/spp{spp}/r{r:02d}_C1H.npy") for r in range(R)]).astype(np.float64)
    L = r0[:, 0:3] + r0[:, 3:6]
    return L, r0[:, 6:9], hc


def load_grid(run, view, spp, R):
    g = np.stack([np.load(run / f"raw/grid/{view}/spp{spp:03d}/r{r:02d}.npy") for r in range(R)]).astype(np.float64)
    return g[:, 0:3] + g[:, 3:6] + g[:, 6:9]


def sse(Y, ref, mask):
    """per-replicate mean over ROI and RGB of the squared error."""
    d = Y[:, :, mask] - ref[None, :, mask]
    return (d ** 2).mean((1, 2))


def pooled_var(X, mask):
    return float(X.var(0, ddof=1)[:, mask].mean())


def pooled_cov(X, Y, mask):
    R = X.shape[0]
    return float((((X - X.mean(0)) * (Y - Y.mean(0))).sum(0) / (R - 1))[:, mask].mean())


def luma_means(X, mask):
    return np.tensordot(LUMA, X, ([0], [1]))[:, mask].mean(1)


def boot_idx(R, B, rs):
    return rs.integers(0, R, (B, R))


def _tv(X, idx):
    X = X if idx is None else X[idx]
    return X.var(0, unbiased=True).mean()


def _tc(X, Y, idx):
    if idx is not None:
        X, Y = X[idx], Y[idx]
    return (((X - X.mean(0)) * (Y - Y.mean(0))).sum(0) / (X.shape[0] - 1)).mean()


def var_components(Lg, HOg, HCg, mask, B, rs):
    """Pooled variance components over the ROI (torch, GPU); bootstrap SE over replicate frames.
    Lg, HOg, HCg: torch [R, 3, H, W] float64 on the GPU."""
    import torch
    m = torch.as_tensor(mask, device=Lg.device)
    L, HO, HC = Lg[:, :, m], HOg[:, :, m], HCg[:, :, m]

    def comp(idx):
        v = {"var_L": _tv(L, idx), "var_H": _tv(HO, idx), "two_cov_LH": 2 * _tc(L, HO, idx), "var_T_R0": _tv(L + HO, idx),
             "var_HC": _tv(HC, idx), "two_cov_L_HC": 2 * _tc(L, HC, idx), "var_T_C1": _tv(L + HC, idx), "cov_H_HC": _tc(HO, HC, idx)}
        v = {k: float(x) for k, x in v.items()}
        dv = lambda a, b: a / b if b != 0 else float("nan")
        v["var_T_ratio"] = dv(v["var_T_C1"], v["var_T_R0"])
        v["high_J_share"] = dv(v["cov_H_HC"], v["var_H"])
        v["ideal_conditional_ratio"] = dv(v["var_L"] + v["cov_H_HC"] + v["two_cov_L_HC"], v["var_T_R0"])
        return v
    point = comp(None)
    R = L.shape[0]
    draws = [comp(torch.as_tensor(rs.integers(0, R, R), device=Lg.device)) for _ in range(B)]
    se = {k: float(np.nanstd([d[k] for d in draws], ddof=1)) for k in point}
    point["identity_residual_R0"] = point["var_T_R0"] - (point["var_L"] + point["var_H"] + point["two_cov_LH"])
    point["identity_residual_C1"] = point["var_T_C1"] - (point["var_L"] + point["var_HC"] + point["two_cov_L_HC"])
    return {"point": point, "se": se}


def spatial_Q(Dg, mask, scale, groups):
    """Spatial bias power from G independent replicate groups (stats109.spatial_bias_power)."""
    import stats109 as ST
    return ST.spatial_bias_power(Dg, mask, scale, groups)


def interp_mse(t, times, mses):
    """log-log interpolation of MSE at time t between the bracketing measured levels (levels ordered by spp).
    Returns (value, flag) with flag 'ok' | 'below' | 'above' | 'nonmonotone'."""
    times, mses = np.asarray(times), np.asarray(mses)
    if np.any(np.diff(times) <= 0):
        order = np.argsort(times)
        times, mses = times[order], mses[order]
        flag = "nonmonotone"
    else:
        flag = "ok"
    if t < times[0]:
        return float("nan"), "below"
    if t > times[-1]:
        return float("nan"), "above"
    i = int(np.searchsorted(times, t, side="right") - 1)
    i = min(i, len(times) - 2)
    x0, x1 = math.log(times[i]), math.log(times[i + 1])
    y0, y1 = math.log(mses[i]), math.log(mses[i + 1])
    w = (math.log(t) - x0) / (x1 - x0) if x1 > x0 else 0.0
    return float(math.exp(y0 + w * (y1 - y0))), flag


def interp_vec(t, times, mses):
    """vectorized over bootstrap draws: t [B], times [B, L] (sorted per draw), mses [B, L] -> [B] (nan outside)."""
    order = np.argsort(times, axis=1)
    times = np.take_along_axis(times, order, 1)
    mses = np.take_along_axis(mses, order, 1)
    Bn, Ln = times.shape
    i = (times <= t[:, None]).sum(1) - 1
    bad = (i < 0) | (t > times[:, -1])
    i = np.clip(i, 0, Ln - 2)
    r = np.arange(Bn)
    x0, x1 = np.log(times[r, i]), np.log(times[r, i + 1])
    y0, y1 = np.log(mses[r, i]), np.log(mses[r, i + 1])
    w = np.where(x1 > x0, (np.log(t) - x0) / np.where(x1 > x0, x1 - x0, 1), 0.0)
    out = np.exp(y0 + w * (y1 - y0))
    out[bad] = np.nan
    return out


def main(run, dev_json, synth_json, mout, dyn_json):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = P["analysis"]
    man = read(run / "manifest.json")
    if man["smoke"]:
        apply_smoke()
    dev, syn = read(dev_json), read(synth_json)
    assert man["complete"] and man["protocol_sha256"] == sha(HERE / "protocol109.json")
    assert man["dev_json_sha256"] == sha(dev_json) and man["synth_json_sha256"] == sha(synth_json)
    mout.mkdir(parents=True, exist_ok=True)
    R = P["final"]["escalated_replicates"] if man.get("escalated") else P["final"]["replicates"]
    Rg = P["final"]["replicates"]
    B = A["bootstrap_draws"]
    rs = np.random.default_rng(A["bootstrap_seed"])
    res = {"protocol_sha256": man["protocol_sha256"], "run_manifest_sha256": sha(run / "manifest.json"), "selected_M": man["selected_M"],
           "paired_replicates": R, "grid_replicates": Rg, "views": {}, "analysis_code_sha256": sha(Path(__file__))}
    # raw-file integrity
    bad = [k for k, h in man["files"].items() if sha(run / "raw" / k) != h]
    assert not bad, bad[:5]
    roi_list = list(DECISION_ROIS) + ["contact"]
    store = {}          # bootstrap inputs per view
    for view in EVAL_VIEWS:
        t0 = time.time()
        aux = RT.aux(view)
        masks = rois(aux)
        bands = distance_bands(aux, A["band_width"])
        allm = {**{k: masks[k] for k in roi_list}, **{f"band{k}": v for k, v in bands.items()}}
        allm = {k: v for k, v in allm.items() if v.sum() > 0}
        refc, vref, refrec = RT.reference(view)
        ref = refc[0:3] + refc[3:6] + refc[6:9]
        V = {"reference": refrec, "pixels": {k: int(v.sum()) for k, v in allm.items()}, "paired": {}, "grid": {}}
        V["ref_mean_sq"] = {k: float((ref[:, m] ** 2).mean()) for k, m in allm.items()}
        V["ref_var_mean"] = {k: float(vref[:, m].mean()) for k, m in allm.items()}
        V["ref_luma"] = {k: float((LUMA @ ref[:, m]).mean()) for k, m in allm.items()}
        V["ref_rel_rms_se"] = {k: math.sqrt(V["ref_var_mean"][k] / V["ref_mean_sq"][k]) for k in allm}
        st = {"paired": {}, "grid": {}}
        for spp in NOMINAL:
            L, HO, HC = load_paired(run, view, spp, R)
            T0, T1 = L + HO, L + HC
            pr = {"sse_R0": {}, "sse_C1": {}, "mse_R0": {}, "mse_C1": {}, "relrmse_R0": {}, "relrmse_C1": {}, "components": {},
                  "paired_luma_diff": {}, "Q": {}, "signed_bias_vs_ref": {}}
            for k, m in allm.items():
                s0, s1 = sse(T0, ref, m), sse(T1, ref, m)
                pr["sse_R0"][k], pr["sse_C1"][k] = s0.tolist(), s1.tolist()
                pr["mse_R0"][k] = float(s0.mean() - V["ref_var_mean"][k])
                pr["mse_C1"][k] = float(s1.mean() - V["ref_var_mean"][k])
                pr["relrmse_R0"][k] = math.sqrt(max(pr["mse_R0"][k], 0) / V["ref_mean_sq"][k])
                pr["relrmse_C1"][k] = math.sqrt(max(pr["mse_C1"][k], 0) / V["ref_mean_sq"][k])
                d = luma_means(HC - HO, m)
                pr["paired_luma_diff"][k] = d.tolist()
                for arm, T in (("R0", T0), ("C1", T1)):
                    y = luma_means(T, m)
                    pr["signed_bias_vs_ref"].setdefault(arm, {})[k] = {
                        "relative": float((y.mean() - V["ref_luma"][k]) / V["ref_luma"][k]),
                        "se": float(math.sqrt(y.var(ddof=1) / R + 0.0) / abs(V["ref_luma"][k]))}
            import torch
            Lg, HOg, HCg = (torch.as_tensor(x, device="cuda") for x in (L, HO, HC))
            for k in list(DECISION_ROIS) + ["contact"]:
                if k in allm:
                    pr["components"][k] = var_components(Lg, HOg, HCg, allm[k], A["component_bootstrap_draws"], rs)
                    if k in DECISION_ROIS:
                        scale = pr["components"][k]["point"]["var_T_R0"]
                        pr["Q"][k] = spatial_Q(HCg - HOg, allm[k], scale, A["Q_groups"])
            del Lg, HOg, HCg
            torch.cuda.empty_cache()
            if view == "V0" and spp == 4:
                np.savez_compressed(mout / "V0_spp4_maps.npz", var_T_R0=T0.var(0, ddof=1).mean(0).astype(np.float32),
                                    var_T_C1=T1.var(0, ddof=1).mean(0).astype(np.float32),
                                    paired_mean_diff=(HC - HO).mean(0).astype(np.float32), ref=ref.astype(np.float32),
                                    r0_frame=T0[0].astype(np.float32), c1_frame=T1[0].astype(np.float32))
            V["paired"][str(spp)] = {k: v for k, v in pr.items() if not k.startswith("sse") and k != "paired_luma_diff"}
            st["paired"][spp] = pr
            del L, HO, HC, T0, T1
        for spp in GRID:
            T = load_grid(run, view, spp, Rg)
            g = {"sse": {}, "mse": {}, "luma": {}}
            for k, m in allm.items():
                s = sse(T, ref, m)
                g["sse"][k] = s.tolist()
                g["mse"][k] = float(s.mean() - V["ref_var_mean"][k])
                if k in DECISION_ROIS:
                    g["luma"][k] = luma_means(T, m).tolist()
            V["grid"][str(spp)] = {"mse": g["mse"]}
            st["grid"][spp] = g
            del T
        # report-only: WL107 reference against the independent hashed-seed R0 grid frames (spp-weighted mean of all levels)
        cons = {}
        for k in DECISION_ROIS:
            if k not in allm:
                continue
            num, var, wsum = 0.0, 0.0, sum(GRID)
            for spp in GRID:
                y = np.asarray(st["grid"][spp]["luma"][k])
                w = spp / wsum
                num += w * y.mean(); var += w * w * y.var(ddof=1) / len(y)
            rel = (num - V["ref_luma"][k]) / V["ref_luma"][k]
            cons[k] = {"relative_difference": float(rel), "se": float(math.sqrt(var) / abs(V["ref_luma"][k])),
                       "z": float(rel / (math.sqrt(var) / abs(V["ref_luma"][k])))}
        V["reference_vs_independent_r0"] = cons
        tim = man["timing"][view]
        V["timing_median_ms"] = {k: float(np.median(v)) for k, v in tim["frame_ms"].items()}
        V["e2e_median_ms"] = {k: float(np.median(v)) for k, v in tim["e2e_ms"].items()}
        V["first_call_ms"] = tim["first_call_ms"]
        V["watermark_MiB"] = {k: v / 2 ** 20 for k, v in tim["watermark_bytes"].items()}
        st["timing"] = tim["frame_ms"]
        store[view] = (st, V)
        res["views"][view] = V
        print(view, "loaded", round(time.time() - t0, 1), flush=True)

    # ------------------------------------------------------------------ equal-time and equal-spp comparisons
    def view_arrays(view, roi, spp):
        st, V = store[view]
        s1 = np.asarray(st["paired"][spp]["sse_C1"][roi])
        s0 = np.asarray(st["paired"][spp]["sse_R0"][roi])
        sg = np.asarray([st["grid"][g]["sse"][roi] for g in GRID])             # [L, Rg]
        tg = np.asarray([st["timing"][f"R0_{g}"] for g in GRID])                # [L, nt]
        tc = np.asarray(st["timing"][f"C1_{spp}"])
        return s1, s0, sg, tg, tc, V["ref_var_mean"][roi]

    def compare(roi, spp):
        per, flags = {}, {}
        draws_t = np.empty((B, len(EVAL_VIEWS)))
        draws_s = np.empty((B, len(EVAL_VIEWS)))
        for vi, view in enumerate(EVAL_VIEWS):
            s1, s0, sg, tg, tc, vr = view_arrays(view, roi, spp)
            mse1, mse0 = s1.mean() - vr, s0.mean() - vr
            msg = sg.mean(1) - vr
            tgm = np.median(tg, 1)
            tcm = float(np.median(tc))
            m_eq, flag = interp_mse(tcm, tgm, msg)
            flags[view] = flag
            # bootstrap within view
            ip = rs.integers(0, len(s1), (B, len(s1)))
            ig = rs.integers(0, sg.shape[1], (B, sg.shape[0], sg.shape[1]))
            it = rs.integers(0, tg.shape[1], (B, tg.shape[0], tg.shape[1]))
            ic = rs.integers(0, len(tc), (B, len(tc)))
            b1 = s1[ip].mean(1) - vr
            b0 = s0[ip].mean(1) - vr
            bg = np.take_along_axis(np.broadcast_to(sg, (B,) + sg.shape), ig, 2).mean(2) - vr
            btg = np.median(np.take_along_axis(np.broadcast_to(tg, (B,) + tg.shape), it, 2), 2)
            btc = np.median(tc[ic], 1)
            beq = interp_vec(btc, btg, bg)
            draws_t[:, vi] = b1 / beq
            draws_s[:, vi] = b1 / b0
            per[view] = {"mse_C1": float(mse1), "mse_R0_nominal": float(mse0), "mse_R0_equal_time": m_eq, "interp_flag": flag,
                         "t_C1_ms": tcm, "t_R0_nominal_ms": float(np.median(st_t(view, f"R0_{spp}"))),
                         "cost_ratio": tcm / float(np.median(st_t(view, f"R0_{spp}"))),
                         "equal_time_ratio": float(mse1 / m_eq) if m_eq == m_eq else float("nan"),
                         "equal_spp_ratio": float(mse1 / mse0),
                         "equal_time_ratio_ci95_within_view": [float(np.nanpercentile(b1 / beq, 2.5)), float(np.nanpercentile(b1 / beq, 97.5))]}
            # time to equal quality: R0 time reaching C1's MSE (log-log on the measured curve)
            per[view]["r0_time_to_equal_quality_ms"] = time_to_quality(mse1, tgm, msg)
        iv = rs.integers(0, len(EVAL_VIEWS), (B, len(EVAL_VIEWS)))
        out = {"per_view": per}
        for name, D, key in (("equal_time", draws_t, "equal_time_ratio"), ("equal_spp", draws_s, "equal_spp_ratio")):
            pt = gm([per[v][key] for v in EVAL_VIEWS]) if all(per[v][key] == per[v][key] for v in EVAL_VIEWS) else float("nan")
            lg = np.log(np.take_along_axis(D, iv, 1)).mean(1)
            se = float(np.nanstd(lg, ddof=1))
            out[name] = {"gm": pt, "log_se": se, "ci95": [pt * math.exp(-Z * se), pt * math.exp(Z * se)] if pt == pt else [float("nan")] * 2,
                         "nan_draw_fraction": float(np.isnan(lg).mean())}
        out["interp_flags"] = flags
        return out

    def st_t(view, key):
        return store[view][0]["timing"][key]

    def time_to_quality(target, times, mses):
        times, mses = np.asarray(times), np.asarray(mses)
        for i in range(len(times) - 1):
            if mses[i] >= target >= mses[i + 1]:
                y0, y1 = math.log(mses[i]), math.log(mses[i + 1])
                w = (math.log(target) - y0) / (y1 - y0) if y1 != y0 else 0.0
                return float(math.exp(math.log(times[i]) + w * (math.log(times[i + 1]) - math.log(times[i]))))
        return None

    comps = {}
    for spp in NOMINAL:
        for roi in list(DECISION_ROIS) + ["contact"]:
            if all(roi in store[v][1]["pixels"] for v in EVAL_VIEWS):
                comps[f"{roi}/{spp}"] = compare(roi, spp)
        for v in EVAL_VIEWS:
            pass
    band_names = sorted({k for v in EVAL_VIEWS for k in store[v][1]["pixels"] if k.startswith("band")})
    bands = {}
    for spp in (4,):
        for bn in band_names:
            if all(bn in store[v][1]["pixels"] for v in EVAL_VIEWS):
                bands[f"{bn}/{spp}"] = compare(bn, spp)
    res["comparisons"] = comps
    res["boundary_bands"] = bands

    # ------------------------------------------------------------------ bias gates
    tau, qmax = A["bias_tolerance_relative"], A["spatial_bias_power_max"]
    bias = {}
    for view in EVAL_VIEWS:
        st, V = store[view]
        for roi in DECISION_ROIS:
            if roi not in V["pixels"]:
                continue
            ests, ws = [], []
            for spp in A["bias_spp"]:
                d = np.asarray(st["paired"][spp]["paired_luma_diff"][roi])
                se = d.std(ddof=1) / math.sqrt(len(d))
                ests.append(d.mean()); ws.append(1 / se ** 2 if se > 0 else 1e30)
            ests, ws = np.asarray(ests), np.asarray(ws)
            m = float((ests * ws).sum() / ws.sum()); se = float(1 / math.sqrt(ws.sum()))
            rel, rse = m / V["ref_luma"][roi], se / abs(V["ref_luma"][roi])
            lo, hi = rel - Z * rse, rel + Z * rse
            state = "EQUIVALENT" if (lo >= -tau and hi <= tau) else ("BIASED" if (lo > tau or hi < -tau) else "UNRESOLVED")
            qs = {str(spp): V["paired"][str(spp)]["Q"][roi] for spp in A["bias_spp"]}
            qlim = qmax[roi] if isinstance(qmax, dict) else qmax
            q_ok = all(q["upper95"] <= qlim for q in qs.values())
            bias[f"{view}/{roi}"] = {"relative": rel, "se": rse, "ci95": [lo, hi], "state": state, "Q": qs, "Q_pass": bool(q_ok),
                                     "resolved_nonzero": bool(lo > 0 or hi < 0)}
    res["bias"] = bias
    b_states = [b["state"] for b in bias.values()]
    bias_pass = all(s == "EQUIVALENT" for s in b_states) and all(b["Q_pass"] for b in bias.values())
    bias_failed = any(s == "BIASED" for s in b_states) or any(not b["Q_pass"] for b in bias.values())
    bias_unresolved = (not bias_failed) and any(s == "UNRESOLVED" for s in b_states)

    # ------------------------------------------------------------------ reference precision gate
    refq = {}
    for view in EVAL_VIEWS:
        V = store[view][1]
        for roi in DECISION_ROIS:
            if roi not in V["pixels"]:
                continue
            for spp in NOMINAL:
                arm_min = min(V["paired"][str(spp)]["relrmse_C1"][roi], V["paired"][str(spp)]["relrmse_R0"][roi])
                refq[f"{view}/{roi}/{spp}"] = V["ref_rel_rms_se"][roi] / arm_min
    res["reference_noise_over_arm_rmse"] = refq
    ref_pass = max(refq.values()) <= A["reference_noise_fraction_max"]

    # ------------------------------------------------------------------ decision
    g = A["gates"]
    def get(roi, spp):
        return comps[f"{roi}/{spp}"]["equal_time"]
    extension_needed = any(f in ("below", "above") for c in comps.values() for f in c["interp_flags"].values())
    p2 = get("primary", 4)["gm"] <= g["primary_4spp_equal_time_ratio_max"]
    p3 = get("primary", 4)["ci95"][1] < g["primary_4spp_upper95_below"]
    p5 = all(get("primary", s)["gm"] <= g["higher_spp_ratio_point_max"] and get("primary", s)["ci95"][1] <= g["higher_spp_ratio_upper95_max"]
             for s in (16, 32))
    p6 = all(get(roi, s)["gm"] <= g["region_ratio_point_max"] and get(roi, s)["ci95"][1] <= g["region_ratio_upper95_max"]
             for roi in ("silhouette", "sss_interior") for s in (4, 16, 32))
    dec = {"P1_synthetic": bool(syn["pass"]), "P2_primary_4spp_ratio": bool(p2), "P3_primary_4spp_upper95": bool(p3),
           "P4_bias": bool(bias_pass), "P4_bias_failed": bool(bias_failed), "P4_bias_unresolved": bool(bias_unresolved),
           "P5_higher_spp_nonregression": bool(p5), "P6_region_nonregression": bool(p6), "reference_precision": bool(ref_pass),
           "equal_time_extension_needed": bool(extension_needed), "escalated": bool(man.get("escalated", False))}
    static_pass = all(dec[k] for k in ("P1_synthetic", "P2_primary_4spp_ratio", "P3_primary_4spp_upper95", "P4_bias",
                                       "P5_higher_spp_nonregression", "P6_region_nonregression", "reference_precision")) and not extension_needed
    dec["static_pass"] = bool(static_pass)
    if not ref_pass or extension_needed:
        verdict = "INCONCLUSIVE"
    elif bias_failed:
        verdict = "NOT_SUPPORTED"
    elif bias_unresolved:
        verdict = "PENDING_ESCALATION" if not man.get("escalated") else "INCONCLUSIVE"
    elif static_pass:
        if dyn_json is None:
            verdict = "STATIC_PASS_PENDING_DYNAMIC"
        else:
            dyn = read(dyn_json)
            dec["dynamic"] = {k: dyn.get(k) for k in ("correctness_pass", "benefit_pass", "executed")}
            if not dyn.get("executed"):
                verdict = "STATIC_ONLY_SUPPORTED"
            elif not dyn["correctness_pass"]:
                verdict = "NOT_SUPPORTED"
            elif dyn["benefit_pass"]:
                verdict = "SUPPORTED"
            else:
                verdict = "STATIC_ONLY_SUPPORTED"
    else:
        verdict = "NOT_SUPPORTED"
    res["decision"] = dec
    res["verdict"] = verdict
    write(mout / "metrics109.json", res)
    figures(res, store, mout, plt)
    print("VERDICT", verdict, {k: v for k, v in dec.items()}, flush=True)


def figures(res, store, mout, plt):
    views = EVAL_VIEWS
    # 1. quality-versus-time curves (primary ROI)
    fig, axs = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, view in zip(axs.ravel(), views):
        st, V = store[view]
        tg = [V["timing_median_ms"][f"R0_{g}"] for g in GRID]
        mg = [V["grid"][str(g)]["mse"]["primary"] for g in GRID]
        ax.loglog(tg, mg, "o-", color="0.35", ms=3, label="R0 grid (measured)")
        for spp in NOMINAL:
            tc = V["timing_median_ms"][f"C1_{spp}"]
            ax.loglog([tc], [V["paired"][str(spp)]["mse_C1"]["primary"]], "s", color="C3", ms=6,
                      label="C1" if spp == NOMINAL[0] else None)
            ax.loglog([V["timing_median_ms"][f"R0_{spp}"]], [V["paired"][str(spp)]["mse_R0"]["primary"]], "^", color="C0", ms=5,
                      label="R0 paired nominal" if spp == NOMINAL[0] else None)
        ax.set_title(view); ax.set_xlabel("synchronized GPU frame time [ms]"); ax.set_ylabel("primary MSE vs frozen-ECRO ref")
        ax.grid(True, which="both", alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    fig.savefig(mout / "quality_vs_time.png", dpi=130); plt.close(fig)
    # 2. variance components at 4 spp (pooled over views), signed shares of R0 TOTAL variance
    fig, axs = plt.subplots(1, len(DECISION_ROIS), figsize=(16, 3.8), constrained_layout=True)
    for ax, roi in zip(axs, DECISION_ROIS):
        vals = []
        for view in views:
            if roi not in res["views"][view]["paired"]["4"]["components"]:
                continue
            c = res["views"][view]["paired"]["4"]["components"][roi]["point"]
            vt = c["var_T_R0"]
            vals.append([c["var_L"] / vt, c["var_H"] / vt, c["two_cov_LH"] / vt, c["var_HC"] / vt, c["two_cov_L_HC"] / vt])
        v = np.mean(vals, 0)
        x = np.arange(2)
        ax.bar(x, [v[0], v[0]], color="C0", label="Var LOW")
        ax.bar(x, [v[1], v[3]], bottom=[v[0], v[0]], color="C1", label="Var HIGH")
        base = np.array([v[0] + v[1], v[0] + v[3]])
        cov = np.array([v[2], v[4]])
        ax.bar(x, cov, bottom=np.where(cov >= 0, base, base + cov), color="C2", label="2Cov(LOW,HIGH)")
        ax.set_xticks(x, ["R0", "C1"]); ax.set_title(f"{roi} ({len(vals)} views)"); ax.axhline(1, color="k", lw=0.6)
        ax.set_ylabel("share of R0 TOTAL variance (mean of views)")
    axs[0].legend(fontsize=8)
    fig.savefig(mout / "variance_components_4spp.png", dpi=130); plt.close(fig)
    # 3. equal-time and equal-spp ratios per view
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, roi in zip(axs, ("primary", "silhouette", "sss_interior")):
        for j, spp in enumerate(NOMINAL):
            c = res["comparisons"][f"{roi}/{spp}"]
            y = [c["per_view"][v]["equal_time_ratio"] for v in views]
            ax.plot(np.arange(len(views)) + 0.12 * j, y, "o", label=f"{spp} spp (GM {c['equal_time']['gm']:.3f})")
        ax.axhline(1, color="k", lw=0.6); ax.axhline(0.85, color="C3", lw=0.6, ls="--")
        ax.set_xticks(np.arange(len(views)), views); ax.set_title(f"{roi}: equal-time MSE ratio C1 / R0"); ax.legend(fontsize=7)
    fig.savefig(mout / "equal_time_ratios.png", dpi=130); plt.close(fig)
    # 4. V0 maps at 4 spp
    z = np.load(mout / "V0_spp4_maps.npz")
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.4), constrained_layout=True)
    vmax = float(np.percentile(z["var_T_R0"], 99.5))
    for ax, k, t in ((axs[0], "var_T_R0", "R0 TOTAL variance (4 spp)"), (axs[1], "var_T_C1", "C1 TOTAL variance (4 spp)")):
        im = ax.imshow(np.sqrt(z[k] / vmax).clip(0, 1), cmap="magma"); ax.set_title(t + ", sqrt scale"); ax.axis("off")
    rr = z["var_T_C1"] / np.maximum(z["var_T_R0"], 1e-12)
    im = axs[2].imshow(np.log2(np.clip(rr, 1 / 4, 4)), cmap="coolwarm", vmin=-2, vmax=2); axs[2].set_title("log2 variance ratio C1/R0")
    axs[2].axis("off"); fig.colorbar(im, ax=axs[2], fraction=0.046)
    d = np.tensordot(LUMA, z["paired_mean_diff"], 1)
    s = float(np.percentile(np.abs(d), 99.5)) or 1.0
    im = axs[3].imshow(d, cmap="coolwarm", vmin=-s, vmax=s); axs[3].set_title("paired mean C1-R0 HIGH (luma)"); axs[3].axis("off")
    fig.colorbar(im, ax=axs[3], fraction=0.046)
    fig.savefig(mout / "V0_maps_4spp.png", dpi=130); plt.close(fig)
    # 5. boundary-distance bands at 4 spp (equal-time ratio)
    names = sorted({k.split("/")[0] for k in res["boundary_bands"]}, key=lambda s: int(s[4:]))
    if names:
        fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
        xs = [int(n[4:]) for n in names]
        ax.plot(xs, [res["boundary_bands"][f"{n}/4"]["equal_time"]["gm"] for n in names], "o-", label="equal time")
        ax.plot(xs, [res["boundary_bands"][f"{n}/4"]["equal_spp"]["gm"] for n in names], "s-", label="equal spp")
        ax.axhline(1, color="k", lw=0.6); ax.set_xlabel("signed pixel distance to sphere silhouette (+ inside)")
        ax.set_ylabel("GM MSE ratio C1/R0 (4 spp)"); ax.legend()
        fig.savefig(mout / "boundary_bands_4spp.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("dev_json"); ap.add_argument("synth_json"); ap.add_argument("out")
    ap.add_argument("--dynamic", default=None)
    a = ap.parse_args()
    main(Path(a.run).resolve(), Path(a.dev_json).resolve(), Path(a.synth_json).resolve(), Path(a.out).resolve(),
         Path(a.dynamic).resolve() if a.dynamic else None)
