"""WL109A Stage C analysis and Stage E static decision (CPU only).

python analyze109a.py CAPTURE_DIR BUDGET_JSON OUT_DIR [--smoke]

Rules come from protocol109a.json (decision) and protocol109.json (definitions, limits, gates). Every Q uses the original
estimator form (common109a.q_from_group_sums == stats109.spatial_bias_power, tests109a), G = 8 contiguous replicate groups,
7-df Student t. The equal-time comparison repeats analyze109.compare on the new paired squared errors, the new R0 grid and the
new timing (same interpolation and two-level bootstrap code).
"""
import argparse
import math
import os
from pathlib import Path
import time

import numpy as np

from common109a import (P, P9, BIAS_SPP, DECISION_ROIS, EVAL_VIEWS, GRID, NOMINAL, LUMA, QLIM, all_masks, b1_state, q_from_group_sums,
                        q_state, q_summary, read, sha, t975, write, HERE)

Z = 1.959963984540054
CONCERN = ("H3/material_boundary", "H4/material_boundary", "H5/material_boundary", "V0/silhouette")


def tq(alpha, df=7):
    from scipy.stats import t
    return float(t.ppf(1 - alpha, df))


def load_unit(d):
    rec = read(d / "unit.json")
    for f, h in rec["files"].items():
        assert sha(d / f) == h, (d, f)
    return rec


def comp_maps(g, n):
    """glob sums (keys L H C LL HH CC LH LC HC) -> per-pixel component maps [3, H, W]"""
    v = lambda a, aa: (aa - a * a / n) / (n - 1)
    c = lambda a, b, ab: (ab - a * b / n) / (n - 1)
    vl, vh, vc = v(g["L"], g["LL"]), v(g["H"], g["HH"]), v(g["C"], g["CC"])
    clh, clc, chc = c(g["L"], g["H"], g["LH"]), c(g["L"], g["C"], g["LC"]), c(g["H"], g["C"], g["HC"])
    return {"var_L": vl, "var_H": vh, "two_cov_LH": 2 * clh, "var_T_R0": vl + vh + 2 * clh, "var_HC": vc, "two_cov_L_HC": 2 * clc,
            "var_T_C1": vl + vc + 2 * clc, "cov_H_HC": chc}


def main(cap, budget_json, out, smoke):
    import analyze109 as A9                                     # interp_mse / interp_vec reused unchanged
    import runtime109 as RT
    out.mkdir(parents=True, exist_ok=True)
    A = P9["analysis"]
    budget = read(budget_json) if budget_json.exists() else {"decision": {"R": None}}
    rs = np.random.default_rng(P["seeds"]["bootstrap"])
    B = A["bootstrap_draws"] if not smoke else 200
    views = [v for v in EVAL_VIEWS if (cap / "paired" / f"{v}_spp4").exists()]
    res = {"protocol109a_sha256": sha(HERE / "protocol109a.json"), "budget": budget["decision"], "views": {}, "smoke": smoke,
           "code_sha256": sha(Path(__file__))}
    store = {}
    maps_out = {}
    for view in views:
        t0 = time.time()
        masks = all_masks(RT.aux(view))
        refc, vref, refrec = RT.reference(view)
        ref = refc[0:3] + refc[3:6] + refc[6:9]
        V = {"reference": refrec, "pixels": {k: int(m.sum()) for k, m in masks.items()}, "spp": {}}
        V["ref_var_mean"] = {k: float(vref[:, m].mean()) for k, m in masks.items()}
        V["ref_mean_sq"] = {k: float((ref[:, m] ** 2).mean()) for k, m in masks.items()}
        V["ref_luma"] = {k: float((LUMA @ ref[:, m]).mean()) for k, m in masks.items()}
        V["ref_rel_rms_se"] = {k: math.sqrt(V["ref_var_mean"][k] / V["ref_mean_sq"][k]) for k in masks}
        st = {"paired": {}}
        for spp in BIAS_SPP:
            d = cap / "paired" / f"{view}_spp{spp}"
            rec = load_unit(d)
            R, sub, nsub = rec["R"], rec["subgroup"], rec["subgroups"]
            if not smoke:
                assert R == budget["decision"]["R"], (R, budget["decision"])
            names = rec["mask_names"]
            G = 8 if nsub >= 8 else nsub
            k = nsub // G
            n = R // G
            SG1 = np.load(d / "SG1.npy").reshape(nsub, 3, -1); SG2 = np.load(d / "SG2.npy").reshape(nsub, 3, -1)
            gl = np.load(d / "glob.npy").reshape(len(rec["glob_keys"]), 3, -1)
            g = dict(zip(rec["glob_keys"], gl))
            cm = comp_maps(g, R)
            pf = dict(np.load(d / "perframe.npz"))
            sc = dict(np.load(d / "subcomp.npz"))
            GS1 = SG1.reshape(G, k, 3, -1).sum(1); GS2 = SG2.reshape(G, k, 3, -1).sum(1)
            GSx = {GG: (SG1.reshape(GG, nsub // GG, 3, -1).sum(1), SG2.reshape(GG, nsub // GG, 3, -1).sum(1))
                   for GG in (16, 32) if nsub % GG == 0 and nsub >= GG}
            tS1, tS2 = SG1.sum(0), SG2.sum(0)
            Dmean = tS1 / R
            varD = (tS2 - tS1 * tS1 / R) / (R - 1)
            S = {"R": R, "G": G, "n": n, "identity": rec["identity"], "Q": {}, "Q_G16": {}, "Q_G32": {}, "components": {},
                 "mse_R0": {}, "mse_C1": {}, "relrmse_R0": {}, "relrmse_C1": {}, "mean_luma_D": {}, "abs_bias_power": {},
                 "varD_over_scale": {}}
            for key, m in masks.items():
                idx = np.flatnonzero(m.reshape(-1))
                scale = float(cm["var_T_R0"][:, idx].mean())
                qg = q_from_group_sums(GS1[:, :, idx], GS2[:, :, idx], n, scale)
                S["Q"][key] = q_summary(qg)
                S["Q"][key]["t"] = S["Q"][key]["point"] / S["Q"][key]["se"] if S["Q"][key]["se"] > 0 else float("nan")
                S["abs_bias_power"][key] = {"value": S["Q"][key]["point"] * scale, "se": S["Q"][key]["se"] * scale, "scale": scale}
                S["varD_over_scale"][key] = float(varD[:, idx].mean() / scale)
                for GG, slot in ((16, "Q_G16"), (32, "Q_G32")):
                    if GG in GSx:
                        S[slot][key] = q_summary(q_from_group_sums(GSx[GG][0][:, :, idx], GSx[GG][1][:, :, idx], R // GG, scale))
                ki = names.index(key)
                dl = pf["lumD"][:, ki]
                S["mean_luma_D"][key] = {"mean": float(dl.mean()), "se": float(dl.std(ddof=1) / math.sqrt(R)),
                                         "relative": float(dl.mean() / V["ref_luma"][key]) if V["ref_luma"][key] else float("nan")}
                s0, s1 = pf["sse0"][:, ki], pf["sse1"][:, ki]
                S["mse_R0"][key] = float(s0.mean() - V["ref_var_mean"][key]); S["mse_C1"][key] = float(s1.mean() - V["ref_var_mean"][key])
                S["relrmse_R0"][key] = math.sqrt(max(S["mse_R0"][key], 0) / V["ref_mean_sq"][key])
                S["relrmse_C1"][key] = math.sqrt(max(S["mse_C1"][key], 0) / V["ref_mean_sq"][key])
                if key in DECISION_ROIS or key == "contact":
                    pt = {c: float(cm[c][:, idx].mean()) for c in cm}
                    se = {c: float(sc[c][:, ki].std(ddof=1) / math.sqrt(nsub)) for c in sc}
                    dv = lambda a, b: a / b if b else float("nan")
                    pt["var_T_ratio"] = dv(pt["var_T_C1"], pt["var_T_R0"])
                    pt["high_J_share"] = dv(pt["cov_H_HC"], pt["var_H"])
                    pt["ideal_conditional_ratio"] = dv(pt["var_L"] + pt["cov_H_HC"] + pt["two_cov_L_HC"], pt["var_T_R0"])
                    pt["varD_ratio_vs_R0R0_null"] = dv(pt["var_H"] + pt["var_HC"] - 2 * pt["cov_H_HC"], 2 * pt["var_H"])
                    S["components"][key] = {"point": pt, "se_subgroups": se}
            st["paired"][spp] = {"sse0": pf["sse0"], "sse1": pf["sse1"], "lumD": pf["lumD"], "names": names}
            V["spp"][str(spp)] = S
            if view in ("V0", "H3", "H4", "H5"):
                maps_out[f"{view}_spp{spp}"] = {"Dmean": Dmean.reshape(3, 512, 512).astype(np.float32),
                                                "z": (Dmean / np.sqrt(np.maximum(varD, 1e-30) / R)).reshape(3, 512, 512).astype(np.float32),
                                                "meanT0": ((g["L"] + g["H"]) / R).reshape(3, 512, 512).astype(np.float32),
                                                "meanT1": ((g["L"] + g["C"]) / R).reshape(3, 512, 512).astype(np.float32)}
        # grid and timing
        gd = cap / "grid" / view
        grec = load_unit(gd)
        gz = np.load(gd / "grid.npz")
        gn = grec["mask_names"]
        st["grid"] = {"sse": gz["sse"], "names": gn}
        V["grid_mse"] = {key: [float(gz["sse"][gi, :, gn.index(key)].mean() - V["ref_var_mean"][key]) for gi in range(len(GRID))]
                         for key in list(DECISION_ROIS) + ["contact"] if key in gn}
        td = cap / "timing" / view
        load_unit(td)
        tim = read(td / "timing.json")
        st["timing"] = tim["frame_ms"]
        V["timing_median_ms"] = {k: float(np.median(v)) for k, v in tim["frame_ms"].items()}
        V["e2e_median_ms"] = {k: float(np.median(v)) for k, v in tim["e2e_ms"].items()}
        V["first_call_ms"] = tim["first_call_ms"]
        V["watermark_MiB"] = {k: v / 2 ** 20 for k, v in tim["watermark_bytes"].items()}
        V["timing_exclusive"] = all(x["clean"] for x in tim["exclusivity"])
        store[view] = (st, V)
        res["views"][view] = V
        print(view, "loaded", round(time.time() - t0, 1), flush=True)
    np.savez_compressed(out / "maps109a.npz", **{f"{k}__{kk}": vv for k, v in maps_out.items() for kk, vv in v.items()})

    # ------------------------------------------------------------------ equal-time comparison (analyze109.compare logic)
    def compare(roi, spp):
        per, flags = {}, {}
        draws_t = np.empty((B, len(views))); draws_s = np.empty((B, len(views)))
        for vi, view in enumerate(views):
            st, V = store[view]
            ki = st["paired"][spp]["names"].index(roi)
            s1 = st["paired"][spp]["sse1"][:, ki]; s0 = st["paired"][spp]["sse0"][:, ki]
            gk = st["grid"]["names"].index(roi)
            sg = st["grid"]["sse"][:, :, gk]                                     # [L, Rg]
            vr = V["ref_var_mean"][roi]
            tg = np.asarray([st["timing"][f"R0_{g}"] for g in GRID]); tc = np.asarray(st["timing"][f"C1_{spp}"])
            mse1, mse0 = s1.mean() - vr, s0.mean() - vr
            msg = sg.mean(1) - vr
            tgm = np.median(tg, 1); tcm = float(np.median(tc))
            m_eq, flag = A9.interp_mse(tcm, tgm, msg)
            flags[view] = flag
            ip = rs.integers(0, len(s1), (B, len(s1)))
            ig = rs.integers(0, sg.shape[1], (B, sg.shape[0], sg.shape[1]))
            it = rs.integers(0, tg.shape[1], (B, tg.shape[0], tg.shape[1]))
            ic = rs.integers(0, len(tc), (B, len(tc)))
            b1 = s1[ip].mean(1) - vr; b0 = s0[ip].mean(1) - vr
            bg = np.take_along_axis(np.broadcast_to(sg, (B,) + sg.shape), ig, 2).mean(2) - vr
            btg = np.median(np.take_along_axis(np.broadcast_to(tg, (B,) + tg.shape), it, 2), 2)
            btc = np.median(tc[ic], 1)
            beq = A9.interp_vec(btc, btg, bg)
            draws_t[:, vi] = b1 / beq; draws_s[:, vi] = b1 / b0
            t_r0 = float(np.median(st["timing"][f"R0_{spp}"]))
            per[view] = {"mse_C1": float(mse1), "mse_R0_nominal": float(mse0), "mse_R0_equal_time": m_eq, "interp_flag": flag,
                         "t_C1_ms": tcm, "t_R0_nominal_ms": t_r0, "cost_ratio": tcm / t_r0,
                         "equal_time_ratio": float(mse1 / m_eq) if m_eq == m_eq else float("nan"), "equal_spp_ratio": float(mse1 / mse0)}
        iv = rs.integers(0, len(views), (B, len(views)))
        o = {"per_view": per, "interp_flags": flags}
        for name, D, key in (("equal_time", draws_t, "equal_time_ratio"), ("equal_spp", draws_s, "equal_spp_ratio")):
            vals = [per[v][key] for v in views]
            pt = float(np.exp(np.mean(np.log(vals)))) if all(x == x for x in vals) else float("nan")
            lg = np.log(np.take_along_axis(D, iv, 1)).mean(1)
            se = float(np.nanstd(lg, ddof=1))
            o[name] = {"gm": pt, "log_se": se, "ci95": [pt * math.exp(-Z * se), pt * math.exp(Z * se)], "nan_draw_fraction": float(np.isnan(lg).mean())}
        return o

    comps = {f"{roi}/{spp}": compare(roi, spp) for spp in BIAS_SPP for roi in list(DECISION_ROIS) + ["contact"]
             if all(roi in store[v][0]["grid"]["names"] for v in views)}
    res["comparisons"] = comps

    # ------------------------------------------------------------------ bias states
    bias = {}
    for view in views:
        st, V = store[view]
        for roi in DECISION_ROIS:
            if roi not in V["pixels"]:
                continue
            qs = {str(s): V["spp"][str(s)]["Q"][roi] for s in BIAS_SPP}
            dd = {s: st["paired"][s]["lumD"][:, st["paired"][s]["names"].index(roi)] for s in BIAS_SPP}
            b1 = b1_state(dd, V["ref_luma"][roi])
            bias[f"{view}/{roi}"] = {"Q": qs, "Q_state": q_state(qs, QLIM[roi]), "B1": b1, "limit": QLIM[roi], "pixels": V["pixels"][roi]}
    res["bias"] = bias
    n_q = sum(len(b["Q"]) for b in bias.values())
    t_fam, t_con = tq(0.05 / n_q), tq(0.05 / (len(CONCERN) * len(BIAS_SPP)))
    z_b1 = float(__import__("scipy.stats", fromlist=["norm"]).norm.ppf(1 - 0.05 / (2 * len(bias))))
    flags = []
    for key, b in bias.items():
        if b["Q_state"] != "EQUIVALENT":
            flags.append({"key": key, "why": f"Q {b['Q_state']}"})
        if b["B1"]["state"] != "EQUIVALENT":
            flags.append({"key": key, "why": f"B1 {b['B1']['state']}"})
        for s, q in b["Q"].items():
            if q["t"] == q["t"] and q["t"] > t_fam:
                flags.append({"key": key, "spp": s, "why": "Q resolved > 0 (family alpha 0.05/n_Q)", "t": q["t"]})
            elif key in CONCERN and q["t"] == q["t"] and q["t"] > t_con:
                flags.append({"key": key, "spp": s, "why": "concern-region Q resolved > 0 (alpha 0.05/12)", "t": q["t"]})
        if abs(b["B1"]["relative"]) / b["B1"]["se"] > z_b1:
            flags.append({"key": key, "why": "B1 nonzero (alpha 0.05/n_B1)", "z": b["B1"]["relative"] / b["B1"]["se"]})
    res["stage_D_trigger"] = {"flags": flags, "triggered": bool(flags), "t_family": t_fam, "t_concern": t_con, "z_b1": z_b1}

    # ------------------------------------------------------------------ reconfirmation and reference precision
    g = A["gates"]
    get = lambda roi, s: comps[f"{roi}/{s}"]["equal_time"]
    refq = {}
    for view in views:
        V = store[view][1]
        for roi in DECISION_ROIS:
            if roi in V["pixels"]:
                for s in BIAS_SPP:
                    arm = min(V["spp"][str(s)]["relrmse_C1"][roi], V["spp"][str(s)]["relrmse_R0"][roi])
                    refq[f"{view}/{roi}/{s}"] = V["ref_rel_rms_se"][roi] / arm
    res["reference_noise_over_arm_rmse"] = refq
    rec_ = {"P2_primary_4spp_ratio": get("primary", 4)["gm"] <= g["primary_4spp_equal_time_ratio_max"],
            "P3_primary_4spp_upper95": get("primary", 4)["ci95"][1] < g["primary_4spp_upper95_below"],
            "P5_higher_spp": all(get("primary", s)["gm"] <= g["higher_spp_ratio_point_max"] and get("primary", s)["ci95"][1] <= g["higher_spp_ratio_upper95_max"] for s in (16, 32)),
            "P6_regions": all(get(r, s)["gm"] <= g["region_ratio_point_max"] and get(r, s)["ci95"][1] <= g["region_ratio_upper95_max"]
                              for r in ("silhouette", "sss_interior") for s in (4, 16, 32)),
            "reference_precision": max(refq.values()) <= A["reference_noise_fraction_max"],
            "grid_brackets": all(f == "ok" for c in comps.values() for f in c["interp_flags"].values()),
            "timing_exclusive": all(store[v][1]["timing_exclusive"] for v in views)}
    res["reconfirmation"] = {k: bool(v) for k, v in rec_.items()}
    qs_ = [b["Q_state"] for b in bias.values()]; bs_ = [b["B1"]["state"] for b in bias.values()]
    if "BIASED" in qs_ or "BIASED" in bs_:
        verdict = "BIAS_SUPPORTED"
    elif "UNRESOLVED" in qs_ or "UNRESOLVED" in bs_:
        verdict = "INCONCLUSIVE"
    elif not (rec_["reference_precision"] and rec_["grid_brackets"]):
        verdict = "INCONCLUSIVE"
    elif not all(rec_[k] for k in ("P2_primary_4spp_ratio", "P3_primary_4spp_upper95", "P5_higher_spp", "P6_regions")):
        verdict = "NOT_SUPPORTED"
    elif flags:
        verdict = "STAGE_D_REQUIRED"
    else:
        verdict = "STATIC_PASS"
    res["static_verdict"] = verdict
    res["counts"] = {"Q": {s: qs_.count(s) for s in set(qs_)}, "B1": {s: bs_.count(s) for s in set(bs_)}}
    write(out / "metrics109a.json", res)
    print("STATIC", verdict, res["counts"], res["reconfirmation"], "flags", len(flags), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cap"); ap.add_argument("budget"); ap.add_argument("out")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "analysis runs CPU-only"
    main(Path(a.cap).resolve(), Path(a.budget).resolve(), Path(a.out).resolve(), a.smoke)
