"""WL109B reviewer export (CPU only): figures, reviewer copies of the decision JSONs and a hash manifest -> evaluation/109B.

usage: export109b.py EVIDENCE_DIR OUT_DIR
Reads only finished WL109B outputs; computes nothing that enters a decision.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

VIEWS = ["V0", "H1", "H2", "H3", "H4", "H5"]
ROIS = ["primary", "sss_interior", "silhouette", "material_boundary", "contact", "control"]
SPP = ["4", "16", "32"]
COL = {"4": "#2a6fdb", "16": "#e08a1e", "32": "#3a9d5d"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fig_calibration(ev, out):
    s = json.loads((ev / "calib/synthetic_entries109b.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key in zip(axs, ["R512", "R1024"]):
        ent = s[key]
        fam = sorted({k.split("/")[0] for k in ent})
        for i, f in enumerate(fam):
            vals = [(v["Q_true_over_limit"], v["cp95"][1], v["rate"]) for k, v in ent.items() if k.split("/")[0] == f]
            x = np.array([i + 0.08 * (q - 1) for q, _, _ in vals])
            ax.scatter(x, [r for *_, r in vals], s=8, color="#2a6fdb", alpha=0.6, label="observed rate" if i == 0 else None)
            ax.scatter(x, [u for _, u, _ in vals], s=8, color="#999", marker="_", label="CP95 upper" if i == 0 else None)
        ax.axhline(0.05, color="#c0392b", lw=1, label="gate 0.05")
        ax.axhline(0.025, color="#c0392b", lw=0.8, ls=":", label="nominal 0.025")
        ax.set_xticks(range(len(fam)), fam, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(-0.002, 0.06)
        ax.set_ylabel("one-sided non-coverage of the XJL upper bound")
        n = len(ent); npass = sum(v["class"] == "PASS" for v in ent.values())
        ax.set_title(f"{key}: synthetic entries {npass}/{n} PASS (2,000 realizations each)", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=130)
    plt.close(fig)


def fig_bias_bounds(ev, out):
    b = json.loads((ev / "metrics/bias109b.json").read_text())["bias"]
    keys = [f"{v}/{r}" for r in ROIS for v in VIEWS if f"{v}/{r}" in b]
    fig, ax = plt.subplots(figsize=(14, 4.8))
    for i, k in enumerate(keys):
        lim = b[k]["limit"]
        for j, s in enumerate(SPP):
            x = i + (j - 1) * 0.22
            ax.scatter(x, max(b[k]["Qxjl"][s]["upper"] / lim, 1e-4), color=COL[s], s=18, label=f"XJL upper, {s} spp" if i == 0 else None)
            o = b[k]["Q_original_wl109_descriptive"][s]["upper95"] / lim
            ax.scatter(x, max(o, 1e-4), facecolors="none", edgecolors=COL[s], s=18, marker="s",
                       label=f"WL109 Q upper (descriptive), {s} spp" if i == 0 else None)
    ax.axhline(1.0, color="#c0392b", lw=1.2, label="limit (EQUIVALENT if every XJL upper <= 1)")
    for i, k in enumerate(keys):
        if k in ("H3/material_boundary", "H4/material_boundary"):
            ax.axvspan(i - 0.45, i + 0.45, color="#f4d03f", alpha=0.25, lw=0)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 3)
    ax.set_xticks(range(len(keys)), keys, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("upper bound / limit")
    ax.set_title("Stage D, R = 512: spatial-bias upper bounds (28 ROI x view, 3 spp). Shaded: WL109's two failed ROIs. "
                 "WL109-design markers below 1e-4 are clipped (negative bounds).", fontsize=9)
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "bias_bounds.png", dpi=130)
    plt.close(fig)


def fig_equal_time(ev, out):
    m = json.loads((ev / "metrics/metrics109b.json").read_text())["comparisons"]
    d = json.loads((ev / "dyn/dyn109b.json").read_text())["D5"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    ax = axs[0]
    for j, r in enumerate(["primary", "sss_interior", "silhouette", "material_boundary", "contact"]):
        pv = m[f"{r}/4"]["per_view"]
        ax.plot(VIEWS, [pv[v]["equal_time_ratio"] for v in VIEWS], marker="o", label=r)
    ax.axhline(0.85, color="#c0392b", lw=1, label="P2 threshold 0.85 (primary GM)")
    ax.axhline(1.0, color="#555", lw=0.8, ls=":")
    ax.set_ylabel("equal-time MSE ratio C1 / R0(t_C1), 4 spp"); ax.set_ylim(0.5, 1.05)
    ax.set_title("Stage F: per-view equal-time ratio (lower is better)", fontsize=10)
    ax.legend(fontsize=7)
    ax = axs[1]
    labels, gm, lo, hi = [], [], [], []
    for r in ["primary", "sss_interior", "silhouette", "material_boundary", "contact"]:
        for s in SPP:
            e = m[f"{r}/{s}"]["equal_time"]
            labels.append(f"{r} {s}"); gm.append(e["gm"]); lo.append(e["ci95"][0]); hi.append(e["ci95"][1])
    labels.append("DYNAMIC P7 primary 4"); gm.append(d["equal_time_ratio"]); lo.append(d["ci95"][0]); hi.append(d["ci95"][1])
    y = np.arange(len(labels))[::-1]
    ax.errorbar(gm, y, xerr=[np.subtract(gm, lo), np.subtract(hi, gm)], fmt="o", ms=3, color="#2a6fdb", ecolor="#2a6fdb")
    ax.scatter([gm[-1]], [y[-1]], color="#c0392b", zorder=3)
    ax.axvline(0.85, color="#c0392b", lw=1); ax.axvline(1.0, color="#555", lw=0.8, ls=":")
    ax.set_yticks(y, labels, fontsize=7); ax.set_xlim(0.6, 1.05)
    ax.set_xlabel("geometric-mean equal-time ratio with 95% CI")
    ax.set_title("Static GM over 6 views; dynamic (red): one view, one edit", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "equal_time.png", dpi=130)
    plt.close(fig)


def fig_maps(ev, out):
    z = np.load(ev / "metrics/maps109b.npz")
    fig, axs = plt.subplots(3, 6, figsize=(16, 8.4))
    for j, v in enumerate(VIEWS):
        img = z[f"{v}_spp4__meanT0"].transpose(1, 2, 0)
        axs[0, j].imshow(np.clip(img / max(np.percentile(img, 99.5), 1e-6), 0, 1) ** (1 / 2.2))
        axs[0, j].set_title(f"{v}: mean R0 (4 spp)", fontsize=8)
        for i, s in enumerate(["4", "32"]):
            zz = z[f"{v}_spp{s}__z"].mean(0) * np.sqrt(3)
            im = axs[i + 1, j].imshow(zz, cmap="RdBu_r", vmin=-4, vmax=4)
            axs[i + 1, j].set_title(f"{v}: z, {s} spp", fontsize=8)
    for a in axs.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=axs[1:, :].ravel().tolist(), shrink=0.6, label="z (N(0,1) under no bias, per pixel)")
    fig.suptitle("Stage D, R = 512: per-pixel z of mean D = H_C1 - H_R0 (RGB mean x sqrt3); no spatial structure expected under no bias. White = zero-variance background", fontsize=10)
    fig.savefig(out / "bias_maps.png", dpi=110)
    plt.close(fig)


def main(ev, out):
    ev, out = Path(ev).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fig_calibration(ev, out)
    fig_bias_bounds(ev, out)
    fig_equal_time(ev, out)
    fig_maps(ev, out)
    copies = {"budget109b.json": "calib/budget109b.json", "pool109b.json": "calib/pool109b.json",
              "synthetic_entries109b.json": "calib/synthetic_entries109b.json", "bias109b.json": "metrics/bias109b.json",
              "metrics109b.json": "metrics/metrics109b.json", "static109b.json": "metrics/static109b.json",
              "dyn109b.json": "dyn/dyn109b.json", "xcheck109b.json": "xcheck/xcheck109b.json",
              "chain.state": "chain.state", "guard109b_lab69.jsonl": "guard109b.jsonl",
              "guard109b_lab70.jsonl": "guard109b_lab70.jsonl", "gpu_util_lab70.csv": "gpu_util_lab70.csv",
              "gpu_apps_lab70.csv": "gpu_apps_lab70.csv"}
    for v in VIEWS:
        copies[f"timing/{v}.json"] = f"cap/timing/{v}/timing.json"
    files = {}
    for dst, src in copies.items():
        (out / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ev / src, out / dst)
        files[dst] = {"sha256": sha(out / dst), "bytes": (out / dst).stat().st_size, "source": f"evidence/{src}"}
    for f in ["calibration.png", "bias_bounds.png", "equal_time.png", "bias_maps.png"]:
        files[f] = {"sha256": sha(out / f), "bytes": (out / f).stat().st_size, "source": "export109b.py (figure)"}
    dyn = json.loads((ev / "dyn/dyn109b.json").read_text())
    man = {"worklog": "docs/worklogs/109B_WORKLOG_ECRO_C1_ROBUST_SPATIAL_BIAS_INFERENCE.md",
           "protocol": "experiments/ecro_estimator109b/protocol109b.json",
           "protocol_sha256": sha(Path(__file__).resolve().parent / "protocol109b.json"),
           "server_run_dir": "LabServer70:~/rest_sss101a/results/ecro_estimator109b/evidence (Stages A-C and 16/18 paired units on LabServer69)",
           "final_verdict": dyn["verdict"],
           "large_evidence_not_copied": ["cap/paired/*/*.npy (9.7 GB)", "dyn/paired_accumulators.npz (409 MB)",
                                         "metrics/maps109b.npz (135 MB; rendered in bias_maps.png)"],
           "files": files}
    (out / "manifest109b.json").write_text(json.dumps(man, indent=1))
    print("exported", len(files), "files ->", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
