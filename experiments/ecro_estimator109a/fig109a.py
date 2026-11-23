"""WL109A reviewer figures (CPU, descriptive; no decision is computed here).

python fig109a.py EVIDENCE_DIR WL109_METRICS_JSON FIG_OUT
"""
import math
import os
from pathlib import Path
import sys

import numpy as np

from common109a import BIAS_SPP, DECISION_ROIS, EVAL_VIEWS, LUMA, QLIM, all_masks, read

COL = {4: "#2a6fdb", 16: "#d9822b", 32: "#3a9d5d"}


def tonemap(x, exposure):
    y = np.clip(x * exposure, 0, None)
    y = y / (1 + y)
    return np.clip(y, 0, 1) ** (1 / 2.2)


def bbox(m, pad=10):
    r, c = np.nonzero(m)
    return max(r.min() - pad, 0), min(r.max() + pad + 1, m.shape[0]), max(c.min() - pad, 0), min(c.max() + pad + 1, m.shape[1])


def main(ev, met9_path, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import runtime109 as RT
    out.mkdir(parents=True, exist_ok=True)
    # ---------------------------------------------------------------- 2. calibration
    hist = read(ev / "calib/hist.json"); pool = read(ev / "calib/pool.json")
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    small = [k for k in hist["settings"] if k.split("/")[1] in ("material_boundary", "silhouette")]
    x = np.arange(len(small))
    axs[0].bar(x, [hist["settings"][k]["q0"]["fail"] for k in small], color="#7f8c8d")
    for i, k in enumerate(small):
        o = hist["settings"][k]["observed_C1_WL109"]
        lim = QLIM[k.split("/")[1]]
        if o["upper95"] > lim:
            axs[0].plot(i, 1.02, "v", color="#c0392b")
    axs[0].set_xticks(x, small, rotation=80, fontsize=6); axs[0].set_ylim(0, 1.08)
    axs[0].set_title("WL109 design under an exact null (R0 vs R0 re-pairings):\nP(upper95 > limit)   ▼ = WL109 C1 failed here", fontsize=9)
    for roi, ls in (("material_boundary", "-"), ("silhouette", "--")):
        for view in EVAL_VIEWS:
            ns, fs, cs = [], [], []
            for nn in (4, 8, 16, 32, 64):
                k = f"{view}/{roi}/4/n{nn}"
                if k in pool["small_n"]:
                    ns.append(nn); fs.append(pool["small_n"][k]["q0"]["fail"])
                    cs.append(np.mean([v["noncov"] for kk, v in pool["small_n"][k].items() if kk.endswith("@1")]))
            if ns:
                axs[1].plot(ns, fs, ls, marker="o", ms=3, label=f"{view} {roi[:4]}")
                axs[2].plot(ns, cs, ls, marker="o", ms=3)
    axs[1].set_xscale("log", base=2); axs[1].set_xlabel("replicates per group n (G = 8; R = 8n)")
    axs[1].set_title("null pool, 4 spp: P(upper95 > limit | Q = 0)", fontsize=9); axs[1].legend(fontsize=6, ncol=2)
    axs[2].set_xscale("log", base=2); axs[2].axhline(0.05, color="#c0392b", ls="--", lw=1); axs[2].axhline(0.025, color="0.5", ls=":", lw=1)
    axs[2].set_xlabel("replicates per group n"); axs[2].set_title("null pool, 4 spp: non-coverage at Q_true = limit (mean of patterns)", fontsize=9)
    fig.savefig(out / "calibration.png", dpi=130); plt.close(fig)
    # ---------------------------------------------------------------- 2b. synthetic coverage (the frozen Stage B gate)
    syn = read(ev / "calib/synth.json")
    fams = ["gauss_iid", "mc_like_4", "mc_like_32", "heteroscedastic", "correlated"]
    designs = ["R32G8", "R128G8", "R512G8", "R1024G8", "R512G16"]
    fig, axs = plt.subplots(1, len(designs), figsize=(18, 4.2), constrained_layout=True, sharey=True)
    for ax, d in zip(axs, designs):
        for fi, f in enumerate(fams):
            cells = sorted(n for n in syn["cells"] if n.startswith(f + "/") and n.endswith("/" + d))
            for ci, n in enumerate(cells):
                c = syn["cells"][n]
                worst = max(v["noncov"] for k, v in c.items() if isinstance(v, dict) and "noncov" in v)
                q0 = c["q0"]["noncov"]; R = c["realizations"]
                x = fi + (ci - 2) * 0.12
                se = math.sqrt(0.05 * 0.95 / R)
                ax.errorbar(x, q0, yerr=1.96 * se, fmt="o", ms=3.5, color="#2a6fdb", lw=0.8, label="Q_true = 0" if (fi, ci) == (0, 0) else None)
                ax.plot(x, worst, "x", ms=4, color="#c0392b", label="worst injected cell" if (fi, ci) == (0, 0) else None)
        ax.axhline(0.05, color="#c0392b", ls="--", lw=1); ax.axhline(0.025, color="0.5", ls=":", lw=1)
        ax.set_xticks(range(len(fams)), fams, rotation=40, fontsize=7); ax.set_title(d, fontsize=9); ax.grid(axis="y", alpha=0.3)
    axs[0].set_ylabel("one-sided non-coverage P(upper95 < Q_true)"); axs[0].legend(fontsize=7)
    fig.suptitle("Synthetic known-truth families (5 ROI shapes each; bars = ±1.96 binomial SE at 0.05). Gate: every cell at the design <= 0.05", fontsize=9)
    fig.savefig(out / "synthetic_coverage.png", dpi=130); plt.close(fig)
    if not (ev / "metrics/metrics109a.json").exists():
        print("Stage C metrics absent: Stage B figures only")
        return
    met = read(ev / "metrics/metrics109a.json")
    met9 = read(met9_path)
    # ---------------------------------------------------------------- 1. Q bounds, WL109 vs WL109A (Q / limit)
    keys = list(met["bias"])
    fig, axs = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True, sharex=True)
    for ax, (name, src) in zip(axs, (("WL109 (R = 32, n = 4 per group)", "w9"), (f"WL109A (R = {met['budget'].get('R')}, independent seeds)", "wa"))):
        for j, key in enumerate(keys):
            lim = QLIM[key.split("/")[1]]
            for k, s in enumerate(BIAS_SPP):
                q = met9["bias"][key]["Q"][str(s)] if src == "w9" else met["bias"][key]["Q"][str(s)]
                lo = q.get("lower95", q["point"] - (q["upper95"] - q["point"]))
                x = j + (k - 1) * 0.22
                ax.plot([x, x], [lo / lim, q["upper95"] / lim], color=COL[s], lw=1.6)
                ax.plot(x, q["point"] / lim, "o", color=COL[s], ms=3.5, label=f"{s} spp" if j == 0 else None)
        ax.axhline(1, color="#c0392b", lw=1, ls="--"); ax.axhline(0, color="0.5", lw=0.6)
        ax.set_ylabel("Q / limit (point, 95% t bounds)"); ax.set_title(name); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-1.2, 3.0)
    axs[0].legend(ncol=3, fontsize=8, loc="upper left")
    axs[1].set_xticks(range(len(keys)), keys, rotation=70, fontsize=7)
    fig.savefig(out / "q_bounds_wl109_vs_wl109a.png", dpi=130); plt.close(fig)
    # ---------------------------------------------------------------- 3. maps and crops (4 spp)
    z = np.load(ev / "metrics/maps109a.npz")
    for view in ("V0", "H3", "H4", "H5"):
        if f"{view}_spp4__Dmean" not in z:
            continue
        aux = RT.aux(view)
        masks = all_masks(aux)
        refc, _, _ = RT.reference(view)
        ref = refc[0:3] + refc[3:6] + refc[6:9]
        T0, T1 = z[f"{view}_spp4__meanT0"], z[f"{view}_spp4__meanT1"]
        D, Zm = np.tensordot(LUMA, z[f"{view}_spp4__Dmean"], 1), np.tensordot(LUMA, z[f"{view}_spp4__z"], 1)
        e0, e1 = np.tensordot(LUMA, T0 - ref, 1), np.tensordot(LUMA, T1 - ref, 1)
        expo = 1.0 / max(float(np.percentile(np.tensordot(LUMA, ref, 1), 99)), 1e-6)
        roi = "silhouette" if view == "V0" else "material_boundary"
        r0, r1, c0, c1 = bbox(masks[roi], 12)
        s = max(float(np.percentile(np.abs(e0), 99.5)), 1e-9)
        sd = max(float(np.percentile(np.abs(D), 99.5)), 1e-9)
        panels = [("R0 mean (4 spp, all replicates)", tonemap(T0.transpose(1, 2, 0), expo), None),
                  ("C1 mean (same exposure)", tonemap(T1.transpose(1, 2, 0), expo), None),
                  ("paired mean C1 - R0 (luma)", D, ("coolwarm", -sd, sd)),
                  ("paired z (luma of per-channel z)", Zm, ("coolwarm", -4, 4)),
                  ("R0 mean - reference (luma)", e0, ("coolwarm", -s, s)),
                  ("C1 mean - reference (luma)", e1, ("coolwarm", -s, s))]
        fig, axs = plt.subplots(2, 6, figsize=(22, 8), constrained_layout=True)
        for j, (t, img, cm) in enumerate(panels):
            for i, crop in enumerate((False, True)):
                ax = axs[i, j]
                im = img[r0:r1, c0:c1] if crop else img
                if cm is None:
                    ax.imshow(im)
                else:
                    h = ax.imshow(im, cmap=cm[0], vmin=cm[1], vmax=cm[2], interpolation="nearest")
                    if crop:
                        fig.colorbar(h, ax=ax, fraction=0.046)
                if crop:
                    ax.contour(masks[roi][r0:r1, c0:c1], levels=[0.5], colors="k", linewidths=0.4)
                ax.set_title(t + (f"\ncrop: {roi} bbox" if crop else ""), fontsize=8); ax.axis("off")
        fig.suptitle(f"{view}: R = {met['budget'].get('R')} paired replicates at 4 spp; consistent scales within each column", fontsize=10)
        fig.savefig(out / f"maps_{view}_4spp.png", dpi=110); plt.close(fig)
    # ---------------------------------------------------------------- 4. boundary-distance profiles
    fig, axs = plt.subplots(2, 3, figsize=(16, 7.5), constrained_layout=True)
    for ax, view in zip(axs.ravel(), EVAL_VIEWS):
        V = met["views"][view]
        for s in BIAS_SPP:
            S = V["spp"][str(s)]
            names = sorted([k for k in S["mean_luma_D"] if k.startswith("band")], key=lambda k: int(k[4:]))
            xs = [int(k[4:]) for k in names]
            ys = [S["mean_luma_D"][k]["relative"] for k in names]
            es = [2 * S["mean_luma_D"][k]["se"] / abs(V["ref_luma"][k]) for k in names]
            ax.errorbar(np.asarray(xs) + (s - 16) * 0.012, ys, yerr=es, fmt="o-", ms=3, lw=1, color=COL[s], label=f"{s} spp")
        ax.axhline(0, color="k", lw=0.6); ax.set_title(f"{view}: paired mean C1 - R0 / ref luma, ±2 SE", fontsize=9)
        ax.set_xlabel("signed distance to sphere silhouette [px] (+ inside)")
    axs[0, 0].legend(fontsize=8)
    fig.savefig(out / "profile_sphere_silhouette.png", dpi=130); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, view in zip(axs, ("H3", "H4", "H5")):
        V = met["views"][view]
        for s in BIAS_SPP:
            S = V["spp"][str(s)]
            names = sorted([k for k in S["mean_luma_D"] if k.startswith("mbband")], key=lambda k: int(k[6:]))
            xs = [int(k[6:]) for k in names]
            ax.errorbar(np.asarray(xs) + (s - 16) * 0.012, [S["Q"][k]["point"] for k in names],
                        yerr=[2.365 * S["Q"][k]["se"] for k in names], fmt="o-", ms=3, color=COL[s], label=f"{s} spp")
        ax.axhline(0.01, color="#c0392b", ls="--", lw=1); ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{view}: Q by distance to material boundary (0 = ROI)", fontsize=9); ax.set_xlabel("distance [px]")
    axs[0].legend(fontsize=8)
    fig.savefig(out / "profile_material_boundary.png", dpi=130); plt.close(fig)
    # ---------------------------------------------------------------- 5. bias versus spp (absolute bias power)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for key, mk in zip(("H3/material_boundary", "H4/material_boundary", "H5/material_boundary", "V0/silhouette"), "osD^"):
        view, roi = key.split("/")
        V = met["views"][view]
        ab = [V["spp"][str(s)]["abs_bias_power"][roi] for s in BIAS_SPP]
        axs[0].errorbar(BIAS_SPP, [a["value"] for a in ab], yerr=[2.365 * a["se"] for a in ab], marker=mk, capsize=3, label=key)
        axs[1].errorbar(BIAS_SPP, [V["spp"][str(s)]["Q"][roi]["point"] for s in BIAS_SPP],
                        yerr=[2.365 * V["spp"][str(s)]["Q"][roi]["se"] for s in BIAS_SPP], marker=mk, capsize=3, label=key)
    axs[0].axhline(0, color="k", lw=0.6); axs[0].set_xscale("log", base=2)
    axs[0].set_title("absolute bias power Q x Var(TOTAL_R0) [radiance^2]\n(a fixed per-pixel bias is constant across spp)", fontsize=9)
    axs[1].axhline(0, color="k", lw=0.6); axs[1].axhline(0.01, color="#c0392b", ls="--", lw=1); axs[1].set_xscale("log", base=2)
    axs[1].set_title("Q (a fixed bias would grow in proportion to spp)", fontsize=9); axs[0].legend(fontsize=7)
    fig.savefig(out / "bias_vs_spp.png", dpi=130); plt.close(fig)
    # ---------------------------------------------------------------- 6. variance and cost
    fig, axs = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for j, s in enumerate(BIAS_SPP):
        c = met["comparisons"][f"primary/{s}"]
        axs[0].plot(np.arange(6) + 0.12 * j, [c["per_view"][v]["equal_time_ratio"] for v in EVAL_VIEWS], "o", color=COL[s],
                    label=f"WL109A {s} spp (GM {c['equal_time']['gm']:.3f})")
        c9 = met9["comparisons"][f"primary/{s}"]
        axs[0].plot(np.arange(6) + 0.12 * j, [c9["per_view"][v]["equal_time_ratio"] for v in EVAL_VIEWS], "x", color=COL[s], alpha=0.6)
    axs[0].axhline(0.85, color="#c0392b", ls="--", lw=1); axs[0].axhline(1, color="k", lw=0.6)
    axs[0].set_xticks(range(6), EVAL_VIEWS); axs[0].set_title("primary equal-time MSE ratio C1 / R0 (o WL109A, x WL109)", fontsize=9)
    axs[0].legend(fontsize=7)
    names = ("var_L", "var_H", "two_cov_LH", "var_HC", "two_cov_L_HC")
    for i, view in enumerate(EVAL_VIEWS):
        cp = met["views"][view]["spp"]["4"]["components"]["primary"]["point"]
        vt = cp["var_T_R0"]
        axs[1].bar(i - 0.2, cp["var_L"] / vt, 0.35, color="#2a6fdb"); axs[1].bar(i - 0.2, cp["var_H"] / vt, 0.35, bottom=cp["var_L"] / vt, color="#d9822b")
        axs[1].bar(i - 0.2, cp["two_cov_LH"] / vt, 0.35, bottom=(cp["var_L"] + cp["var_H"]) / vt, color="#3a9d5d")
        axs[1].bar(i + 0.2, cp["var_L"] / vt, 0.35, color="#2a6fdb"); axs[1].bar(i + 0.2, cp["var_HC"] / vt, 0.35, bottom=cp["var_L"] / vt, color="#d9822b")
        base = (cp["var_L"] + cp["var_HC"]) / vt; c2 = cp["two_cov_L_HC"] / vt
        axs[1].bar(i + 0.2, c2, 0.35, bottom=base if c2 >= 0 else base + c2, color="#3a9d5d")
    axs[1].set_xticks(range(6), [f"{v}\nR0 | C1" for v in EVAL_VIEWS], fontsize=8)
    axs[1].set_title("primary 4 spp: Var LOW (blue), Var HIGH (orange), 2Cov(LOW,HIGH) (green), share of Var(TOTAL_R0)", fontsize=9)
    fig.savefig(out / "variance_cost.png", dpi=130); plt.close(fig)
    print("figures written", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve())
