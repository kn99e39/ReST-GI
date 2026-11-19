"""WL106 frozen analysis: R0-R4 against the qualified frozen-ECRO target, coverage-bin bias, mixing / weight-noise
decomposition (R3, R4), target trust, paired two-level bootstrap, decisions (protocol denoise106.json), figures.

python analyze106.py DATA105 EVAL106 REVIEW [SYNTH_JSON]
"""
import html
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_denoise105"))
from analyze105 import rois, grad_ratio, LUMA, tone  # noqa: E402

P = json.loads((HERE / "protocol/denoise106.json").read_text())
ARMS = ("R0", "R1", "R2", "R3", "R4")
SPPS = (4, 16, 32)
ROIS = ("primary", "sss_interior", "silhouette", "material_boundary", "control", "contact_excluded")
BINS = [0.0, 1 / 64, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 63 / 64, 1.0 + 1e-9]
COL = {"R0": "#2a78d6", "R1": "#eb6834", "R2": "#1baf7a", "R3": "#6250d6", "R4": "#c2185b"}
MK = {"R0": "o", "R1": "s", "R2": "^", "R3": "D", "R4": "P"}
DRAWS, BSEED = 10000, 10699001


def lum(x):
    return np.tensordot(LUMA, x, 1)


def boot_idx(V, Rn):
    rs = np.random.default_rng(BSEED)
    for _ in range(DRAWS):
        vs = rs.integers(0, V, V)
        yield vs[:, None], rs.integers(0, Rn, (V, Rn))


def boot_stat(f, *arrays):
    """Two-level bootstrap of f(*resampled (V, R) arrays); paired (same indices for every array)."""
    V, Rn = arrays[0].shape
    vals = np.array([f(*[a[vi, ri] for a in arrays]) for vi, ri in boot_idx(V, Rn)])
    return {"point": float(f(*arrays)), "ci95": [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]}


def per_view_ci(f, *arrays):
    out = []
    rs = np.random.default_rng(BSEED)
    V, Rn = arrays[0].shape
    for v in range(V):
        vals = []
        for _ in range(2000):
            ri = rs.integers(0, Rn, Rn)
            vals.append(f(*[a[v:v + 1, ri] for a in arrays]))
        out.append({"point": float(f(*[a[v:v + 1] for a in arrays])), "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]})
    return out


def main(data, ev, review, synth=None):
    review.mkdir(parents=True, exist_ok=True)
    eman = json.loads((ev / "eval_manifest.json").read_text())
    views, reps = eman["views"], eman["reps"]
    V = len(views)
    E = {a: {s: {r: np.full((V, reps), np.nan) for r in ROIS} for s in SPPS} for a in ARMS}
    BIAS = {a: {s: {r: np.full((V, reps), np.nan) for r in ROIS} for s in SPPS} for a in ARMS}
    # ratio-of-sums bias pieces (numerator sum of luma residual, denominator sum of luma target) for pooled statistics
    NUM = {a: {s: {r: np.zeros((V, reps)) for r in ROIS} for s in SPPS} for a in ARMS + ("mix_R3", "mix_R4")}
    DEN = {r: np.zeros(V) for r in ROIS}
    GR = {a: {s: np.full((V, reps), np.nan) for s in SPPS} for a in ARMS}
    BINACC = {(a, s): np.zeros((len(BINS) - 1, 2)) for a in ARMS + ("mix_R3", "mix_R4") for s in SPPS}
    MOM = {(t, s): [] for t in ("R3_low", "R3_high", "R4_low", "R4_high") for s in SPPS}
    TN = {r: np.zeros(V) for r in ROIS}
    TN105 = {r: np.zeros(V) for r in ROIS}
    worst = []
    keep = {}
    for vi, name in enumerate(views):
        with np.load(data / "eval" / name / "aux.npz") as a:
            aux = {k: a[k] for k in a.files}
        m = rois(aux)
        sph = aux["cover"][6]
        with np.load(ev / f"target_{name}.npz") as t:
            Tm = t["comp"].astype(np.float64)
            var_mean = t["var_mean"].astype(np.float64)
        Y = Tm[0:3] + Tm[3:6] + Tm[6:9]
        h = [np.load(data / "eval" / name / f"target_half{i}.npz")["comp"].astype(np.float64) for i in range(2)]
        half = 0.5 * ((h[0][0:3] + h[0][3:6] + h[0][6:9]) - (h[1][0:3] + h[1][3:6] + h[1][6:9]))
        for r in ROIS:
            mm = m[r]
            if mm.any():
                TN[r][vi] = math.sqrt(var_mean[:, mm].sum() / (Y[:, mm] ** 2).sum())
                TN105[r][vi] = math.sqrt((half[:, mm] ** 2).mean() / (Y[:, mm] ** 2).mean())
                DEN[r][vi] = lum(Y)[mm].sum()
        bidx = np.digitize(sph, BINS) - 1
        sil = m["silhouette"]
        for s in SPPS:
            for rep in range(reps):
                with np.load(ev / f"{name}_spp{s}_r{rep}.npz") as f:
                    for a in ARMS + ("mix_R3", "mix_R4"):
                        y = f[a].astype(np.float64)
                        dif = y - Y
                        dl = lum(dif)
                        for r in ROIS:
                            mm = m[r]
                            if not mm.any():
                                continue
                            NUM[a][s][r][vi, rep] = dl[mm].sum()
                            if a in ARMS:
                                E[a][s][r][vi, rep] = math.sqrt((dif[:, mm] ** 2).mean() / (Y[:, mm] ** 2).mean())
                                BIAS[a][s][r][vi, rep] = float(dl[mm].mean() / lum(Y)[mm].mean())
                        for b in range(1, len(BINS) - 2):
                            mm = sil & (bidx == b)
                            BINACC[(a, s)][b] += [dl[mm].sum(), lum(Y)[mm].sum()]
                        if a in ARMS:
                            GR[a][s][vi, rep] = grad_ratio(y, Y, sil) if sil.any() else np.nan
                            if rep == 0:
                                keep[(name, s, a)] = y.astype(np.float32)
                    for t in ("R3_low", "R3_high", "R4_low", "R4_high"):
                        MOM[(t, s)].append(float(f[f"moment_{t}"].astype(np.float64)[sil].mean()) if sil.any() else np.nan)
        keep[(name, "target")] = Y.astype(np.float32)
        keep[(name, "rois")] = m
        keep[(name, "sph")] = sph
        print("analysed", name, flush=True)

    def pooled_bias(num, den):
        return num.sum() / (den[:, None] * np.ones_like(num)).sum()

    summary = {}
    for a in ARMS:
        summary[a] = {}
        for s in SPPS:
            row = {}
            for r in ROIS:
                if np.isnan(E[a][s][r]).all():
                    continue
                row[r] = {"rel_rmse": float(np.nanmean(E[a][s][r])), "signed_bias": float(np.nanmean(BIAS[a][s][r])),
                          "pooled_signed_bias": float(NUM[a][s][r].sum() / (DEN[r][:, None] * np.ones((V, reps))).sum())}
            row["silhouette"]["gradient_ratio"] = float(np.nanmean(GR[a][s]))
            summary[a][s] = row
    tn = {r: float(np.mean(TN[r][TN[r] > 0])) if (TN[r] > 0).any() else None for r in ROIS}
    tn105 = {r: float(np.mean(TN105[r][TN105[r] > 0])) if (TN105[r] > 0).any() else None for r in ROIS}
    bins = {f"{a}_spp{s}": (BINACC[(a, s)][1:-1, 0] / np.maximum(BINACC[(a, s)][1:-1, 1], 1e-12)).tolist() for a in ARMS + ("mix_R3", "mix_R4") for s in SPPS}
    moments = {f"{t}_spp{s}": float(np.nanmean(MOM[(t, s)])) for t in ("R3_low", "R3_high", "R4_low", "R4_high") for s in SPPS}

    dec = {"B1": {}, "S1": {}, "N1": {}, "T1": {}, "decomposition": {}, "per_view": {}}
    for s in SPPS:
        den = DEN["silhouette"][:, None] * np.ones((V, reps))
        n3, n4 = NUM["R3"][s]["silhouette"], NUM["R4"][s]["silhouette"]
        rb = boot_stat(lambda a, b, d: 1 - abs(b.sum() / d.sum()) / abs(a.sum() / d.sum()), n3, n4, den)
        dec["B1"][s] = {**rb, "bias_R3": float(n3.sum() / den.sum()), "bias_R4": float(n4.sum() / den.sum()),
                        "bias_R3_ci": boot_stat(lambda a, d: a.sum() / d.sum(), n3, den)["ci95"],
                        "bias_R4_ci": boot_stat(lambda a, d: a.sum() / d.sum(), n4, den)["ci95"],
                        "material": bool(rb["point"] >= 0.5 and rb["ci95"][0] > 0)}
        sb = {r: summary["R4"][s][r]["signed_bias"] for r in ("primary", "sss_interior", "silhouette", "material_boundary", "control")}
        s1 = {"bias": sb, "silhouette_gradient_ratio": summary["R4"][s]["silhouette"]["gradient_ratio"]}
        s1["pass"] = bool(all(abs(v) <= 0.01 for v in sb.values()) and 0.8 <= s1["silhouette_gradient_ratio"] <= 1.2)
        dec["S1"][s] = s1
        nr = boot_stat(lambda a, b: b.mean() / a.mean(), E["R3"][s]["primary"], E["R4"][s]["primary"])
        bnd = {r: summary["R4"][s][r]["rel_rmse"] / summary["R3"][s][r]["rel_rmse"] for r in ("silhouette", "material_boundary")}
        dec["N1"][s] = {"primary_ratio": nr, "boundary_ratio": bnd,
                        "pass": bool(nr["point"] <= 1.02 and nr["ci95"][1] <= 1.05 and all(v <= 1.05 for v in bnd.values()))}
        t1 = {}
        for r in ("primary", "silhouette", "material_boundary"):
            emin = min(summary["R3"][s][r]["rel_rmse"], summary["R4"][s][r]["rel_rmse"])
            t1[r] = {"target_noise": tn[r], "min_error": emin, "ratio": tn[r] / emin, "pass": bool(tn[r] <= 0.1 * emin)}
        dec["T1"][s] = {**t1, "pass": all(v["pass"] for v in t1.values())}
        for arm in ("R3", "R4"):
            mix = NUM[f"mix_{arm}"][s]["silhouette"].sum() / den.sum()
            tot = NUM[arm][s]["silhouette"].sum() / den.sum()
            mixmb = NUM[f"mix_{arm}"][s]["material_boundary"].sum() / (DEN["material_boundary"][:, None] * np.ones((V, reps))).sum()
            totmb = NUM[arm][s]["material_boundary"].sum() / (DEN["material_boundary"][:, None] * np.ones((V, reps))).sum()
            dec["decomposition"].setdefault(arm, {})[s] = {"silhouette": {"total": float(tot), "mixing": float(mix), "weight_noise": float(tot - mix)},
                                                          "material_boundary": {"total": float(totmb), "mixing": float(mixmb), "weight_noise": float(totmb - mixmb)}}
        dec["per_view"][s] = {arm: dict(zip(views, per_view_ci(lambda a, d: a.sum() / d.sum(), NUM[arm][s]["silhouette"], den))) for arm in ("R3", "R4")}
    # Cost (C7, reported separately).
    cost = eman["cost"]
    c7 = {"latency_ratio": cost["R4"]["latency_ms_median"] / cost["R3"]["latency_ms_median"],
          "memory_ratio": cost["R4"]["peak_extra_mib"] / max(cost["R3"]["peak_extra_mib"], 1e-9)}
    c7["within_budget"] = bool(c7["latency_ratio"] <= 1.5 and c7["memory_ratio"] <= 1.5)
    dec["C7"] = c7
    # Synthetic gate.
    sg = json.loads(Path(synth).read_text())["gate"] if synth else None
    dec["synthetic_gate"] = sg
    # Verdict (protocol decision.verdict).
    T1 = all(dec["T1"][s]["pass"] for s in SPPS)
    B1 = [dec["B1"][s]["material"] for s in SPPS]
    S1 = [dec["S1"][s]["pass"] for s in SPPS]
    N1 = [dec["N1"][s]["pass"] for s in SPPS]
    straddle = sum(dec["B1"][s]["ci95"][0] <= 0 <= dec["B1"][s]["ci95"][1] for s in SPPS)
    s1_fail = [s for s in SPPS if not dec["S1"][s]["pass"]]
    def r4_mix_exceeds(s):
        d = dec["decomposition"]["R4"][s]
        return any(abs(d[r]["mixing"]) > 0.01 for r in ("silhouette", "material_boundary"))
    def r4_noise_dominates(s):
        d = dec["decomposition"]["R4"][s]
        return any(abs(d[r]["weight_noise"]) > abs(d[r]["mixing"]) for r in ("silhouette", "material_boundary") if abs(d[r]["total"]) > 0.01)
    if sg is not None and not sg["pass"]:
        verdict = "STOPPED_AT_SYNTHETIC_GATE"
    elif not T1:
        verdict = "INCONCLUSIVE"
    elif all(B1) and all(S1) and all(N1):
        verdict = "SUPPORTED"
    elif sum(B1) < 2 or not all(N1):
        verdict = "NOT_SUPPORTED"
    elif s1_fail and any(r4_mix_exceeds(s) for s in s1_fail):
        verdict = "INPUT_REPRESENTATION_INSUFFICIENT"
    elif s1_fail and any(r4_noise_dominates(s) for s in s1_fail):
        verdict = "NOT_SUPPORTED"
    elif straddle >= 2:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "NOT_SUPPORTED"
    res = {"views": views, "reps": reps, "target_noise_relative_rms": tn, "target_noise_WL105_two_halves": tn105,
           "n_halves": {n: eman["targets"][n]["n_halves"] for n in views}, "summary": summary, "coverage_bins_silhouette": bins,
           "bins": BINS, "silhouette_moment_L1": moments, "decision": dec, "verdict": verdict, "cost": cost,
           "eval_manifest": {k: eman[k] for k in ("weights_sha256", "data_manifest_sha256", "deterministic_inference_R4", "gpu",
                                                  "reproduction_max_rel_vs_WL105", "input_hash_mismatch", "R4_fused_vs_reference_max_rel")}}
    # Worst boundary cases: per (view, spp) silhouette bias of R4 and R3, sorted by |R4|.
    for vi, name in enumerate(views):
        for s in SPPS:
            den = DEN["silhouette"][vi] * reps
            worst.append({"view": name, "spp": s, "R3": float(NUM["R3"][s]["silhouette"][vi].sum() / den),
                          "R4": float(NUM["R4"][s]["silhouette"][vi].sum() / den)})
    res["worst_boundary_cases"] = sorted(worst, key=lambda w: -abs(w["R4"]))[:6]
    (review / "metrics.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    figures(review, keep, views, summary, bins)
    page(review, res)
    print(verdict, flush=True)


def figures(review, keep, views, summary, bins):
    fig, ax = plt.subplots(1, 4, figsize=(17, 4), layout="constrained")
    for a_, roi in zip(ax, ("primary", "sss_interior", "silhouette", "material_boundary")):
        for arm in ARMS:
            a_.plot(SPPS, [summary[arm][s][roi]["rel_rmse"] for s in SPPS], MK[arm] + "-", color=COL[arm], label=arm)
        a_.set_xscale("log", base=2); a_.set_yscale("log"); a_.set_xticks(SPPS, [str(s) for s in SPPS])
        a_.set_title(roi.replace("_", " ")); a_.set_xlabel("spp"); a_.grid(alpha=.25)
    ax[0].set_ylabel("relative HDR RMSE vs qualified target"); ax[0].legend()
    fig.suptitle("WL106 · error vs spp on held-out views (6 views x 8 seeds)")
    fig.savefig(review / "error_vs_spp.png", dpi=140); plt.close(fig)
    b = np.array(BINS)
    centres = 0.5 * (b[1:-2] + b[2:-1])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained", sharey=True)
    for a_, s in zip(ax, SPPS):
        for arm in ("R1", "R2", "R3", "R4"):
            a_.plot(centres, bins[f"{arm}_spp{s}"], MK[arm] + "-", color=COL[arm], label=arm)
        for arm, ls in (("mix_R3", ":"), ("mix_R4", ":")):
            a_.plot(centres, bins[f"{arm}_spp{s}"], ls, color=COL[arm[4:]], label=f"{arm[4:]} mixing term")
        a_.axhline(0, color="#58616b", lw=.8); a_.axhspan(-0.01, 0.01, color="#9aa0a6", alpha=.15)
        a_.set_title(f"{s} spp"); a_.set_xlabel("sphere coverage fraction of the pixel"); a_.grid(alpha=.25)
    ax[0].set_ylabel("signed residual / target (luminance, pooled)"); ax[0].legend(fontsize=8)
    fig.suptitle("WL106 · silhouette bias by coverage fraction (shaded: ±1% safety band)")
    fig.savefig(review / "coverage_profile.png", dpi=140); plt.close(fig)
    for name in views:
        Y = keep[(name, "target")]
        m = keep[(name, "rois")]
        sph = keep[(name, "sph")]
        exposure = 1.0 / max(np.percentile(lum(Y), 99), 1e-3)
        ys, xs = np.nonzero(m["silhouette"])
        crops = {"full": np.s_[:, :]}
        if len(ys):
            for tag, k in (("silhouette left", np.argmin(xs)), ("silhouette top", np.argmin(ys))):
                cy, cx = ys[k], xs[k]
                crops[tag] = np.s_[max(cy - 32, 0):cy + 32, max(cx - 32, 0):cx + 32]
        for s in (4, 32):
            cols = [("R3", keep[(name, s, "R3")]), ("R4", keep[(name, s, "R4")]), ("target", Y)]
            fig, axs = plt.subplots(3 * len(crops), 3, figsize=(10, 3.2 * 3 * len(crops) / 1.6), layout="constrained", squeeze=False)
            sl0 = crops.get("silhouette left", np.s_[:, :])
            cap = float(np.percentile(np.abs(lum(keep[(name, s, "R3")] - Y))[sl0], 99.5))
            for ci, (cname, sl) in enumerate(crops.items()):
                for j, (label, img) in enumerate(cols):
                    axs[3 * ci, j].imshow(tone(img[(slice(None),) + sl], exposure)); axs[3 * ci, j].set_title(f"{label} · {cname}", fontsize=8)
                    if label != "target":
                        e = lum(img - Y)[sl]
                        axs[3 * ci + 1, j].imshow(e, cmap="RdBu_r", vmin=-cap, vmax=cap); axs[3 * ci + 1, j].set_title(f"signed error (±{cap:.3g})", fontsize=8)
                        axs[3 * ci + 2, j].imshow(np.abs(e), cmap="magma", vmin=0, vmax=cap); axs[3 * ci + 2, j].set_title("|error|", fontsize=8)
                    else:
                        axs[3 * ci + 1, j].imshow(sph[sl], cmap="gray", vmin=0, vmax=1); axs[3 * ci + 1, j].set_title("sphere coverage", fontsize=8)
                        axs[3 * ci + 2, j].imshow(((sph > 0) & (sph < 1))[sl], cmap="gray"); axs[3 * ci + 2, j].set_title("silhouette ROI", fontsize=8)
                for a_ in axs[3 * ci:3 * ci + 3].ravel():
                    a_.set_axis_off()
            fig.suptitle(f"WL106 · {name} · {s} spp replicate 0 · error scale from R3's left-silhouette crop (linear HDR luminance)")
            fig.savefig(review / f"view_{name}_spp{s}.png", dpi=110); plt.close(fig)


def page(review, res):
    d = res["decision"]
    rows = "".join(f"<tr><td>{s}</td><td>{d['B1'][s]['bias_R3']:+.4f}</td><td>{d['B1'][s]['bias_R4']:+.4f}</td>"
                   f"<td>{d['B1'][s]['point']:+.3f} {[round(x, 3) for x in d['B1'][s]['ci95']]}</td><td>{d['S1'][s]['pass']}</td>"
                   f"<td>{d['N1'][s]['primary_ratio']['point']:.3f} {[round(x, 3) for x in d['N1'][s]['primary_ratio']['ci95']]}</td>"
                   f"<td>{d['T1'][s]['pass']}</td></tr>" for s in SPPS)
    imgs = "".join(f"<p><img src='{p.name}' alt='{p.stem}'></p>" for p in sorted(review.glob("view_*.png")))
    (review / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WL106 boundary-safe reconstruction</title><style>:root{{--bg:#fcfcfb;--ink:#1f1f1f;--line:#d9d8d2}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--ink:#f0efec;--line:#3b3b38}}}}
body{{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:16px}}
img{{max-width:100%;height:auto;background:#fff}} td,th{{border-bottom:1px solid var(--line);padding:3px 8px}}</style></head><body>
<h1>WL106 · coverage-moment-constrained reconstruction (R4) vs R3</h1><p><b>Verdict:</b> {html.escape(res['verdict'])}</p>
<table><tr><th>spp</th><th>silhouette bias R3</th><th>R4</th><th>Rb = 1 - |R4|/|R3|</th><th>R4 safety</th><th>primary RMSE R4/R3</th><th>target trust</th></tr>{rows}</table>
<p><img src="coverage_profile.png" alt="coverage profile"></p><p><img src="error_vs_spp.png" alt="error vs spp"></p>{imgs}<p><a href="metrics.json">metrics.json</a></p></body></html>""")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(), sys.argv[4] if len(sys.argv) > 4 else None)
