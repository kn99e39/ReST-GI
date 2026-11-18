"""WL105 frozen analysis: ROI metrics against the frozen-ECRO target, paired two-level bootstrap, decisions, figures.

python analyze105.py DATA EVAL PHYSICAL_REF_DIR REVIEW
"""
import html
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol/denoise105.json").read_text())
LUMA = np.array([0.2126, 0.7152, 0.0722])
ARMS = ("R0", "R1", "R2", "R3")
SPPS = (4, 16, 32)
ROIS = ("primary", "sss_interior", "silhouette", "material_boundary", "control", "contact_excluded")


def erode(m, k):
    p = np.pad(m, k, constant_values=False)
    H, W = m.shape
    return np.logical_and.reduce([p[y:y + H, x:x + W] for y in range(2 * k + 1) for x in range(2 * k + 1)])


def rois(aux):
    cover, sid, ok, p, vf = aux["cover"], aux["sid"], aux["ok"], aux["p"], aux["valid_frac"]
    r = np.hypot(p[..., 0] - 0.9, p[..., 1] - 0.4)
    contact = ok & (sid == 0) & (r < 0.25)
    sph = cover[6]
    m = {"contact_excluded": contact, "primary": (vf > 0) & ~contact,
         "sss_interior": erode(sph >= 1.0, 2), "silhouette": (sph > 0) & (sph < 1),
         "material_boundary": (cover.max(0) < 1) & (sph == 0) & (vf >= 1) & ~contact,
         "control": erode(cover[2] >= 1.0, 4)}
    return m


def grad_ratio(y, t, m):
    yl, tl = np.tensordot(LUMA, y, 1), np.tensordot(LUMA, t, 1)
    num = den = 0.0
    for ax in (0, 1):
        dy, dt = np.diff(yl, axis=ax), np.diff(tl, axis=ax)
        mm = (m[:-1, :] | m[1:, :]) if ax == 0 else (m[:, :-1] | m[:, 1:])
        num += float((dy[mm] ** 2).sum()); den += float((dt[mm] ** 2).sum())
    return math.sqrt(num / den) if den > 0 else float("nan")


def boot(values_a, values_b=None, stat="ratio", draws=10000, seed=10599001):
    """values: (V, R) arrays. Two-level bootstrap of D = 1 - mean(b)/mean(a) (paired) or of mean(a)."""
    rs = np.random.default_rng(seed)
    V, Rn = values_a.shape
    out = np.empty(draws)
    for i in range(draws):
        vs = rs.integers(0, V, V)
        rr = rs.integers(0, Rn, (V, Rn))
        a = values_a[vs[:, None], rr]
        if values_b is None:
            out[i] = a.mean()
        else:
            b = values_b[vs[:, None], rr]
            out[i] = 1 - b.mean() / a.mean()
    point = (values_a.mean() if values_b is None else 1 - values_b.mean() / values_a.mean())
    return {"point": float(point), "ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]}


def main(data, ev, phys, review):
    review.mkdir(parents=True, exist_ok=True)
    dman = json.loads((data / "run_manifest.json").read_text())
    eman = json.loads((ev / "eval_manifest.json").read_text())
    views, reps = eman["views"], eman["reps"]
    E = {a: {s: {r: np.zeros((len(views), reps)) for r in ROIS} for s in SPPS} for a in ARMS}
    L1 = {a: {s: np.zeros((len(views), reps)) for s in SPPS} for a in ARMS}
    BIAS = {a: {s: {r: np.zeros((len(views), reps)) for r in ROIS} for s in SPPS} for a in ARMS}
    GR = {a: {s: {r: np.zeros((len(views), reps)) for r in ("silhouette", "material_boundary")} for s in SPPS} for a in ARMS}
    VAR = {a: {s: {r: np.zeros(len(views)) for r in ROIS} for s in SPPS} for a in ARMS}
    target_noise = {r: [] for r in ROIS}
    keep = {}
    phys_err = {}
    for vi, name in enumerate(views):
        d = data / "eval" / name
        with np.load(d / "aux.npz") as a:
            aux = {k: a[k] for k in a.files}
        m = rois(aux)
        h = [np.load(d / f"target_half{i}.npz")["comp"].astype(np.float64) for i in range(2)]
        T = 0.5 * (h[0] + h[1])
        Y = T[0:3] + T[3:6] + T[6:9]
        half = 0.5 * ((h[0][0:3] + h[0][3:6] + h[0][6:9]) - (h[1][0:3] + h[1][3:6] + h[1][6:9]))
        for r in ROIS:
            mm = m[r]
            if mm.any():
                target_noise[r].append(float(np.sqrt((half[:, mm] ** 2).mean() / (Y[:, mm] ** 2).mean())))
        for s in SPPS:
            outs = {a: [] for a in ARMS}
            for rep in range(reps):
                with np.load(ev / f"{name}_spp{s}_r{rep}.npz") as f:
                    for a in ARMS:
                        y = f[a].astype(np.float64)
                        outs[a].append(np.tensordot(LUMA, y, 1))
                        dif = y - Y
                        for r in ROIS:
                            mm = m[r]
                            if not mm.any():
                                E[a][s][r][vi, rep] = np.nan
                                continue
                            E[a][s][r][vi, rep] = math.sqrt((dif[:, mm] ** 2).mean() / (Y[:, mm] ** 2).mean())
                            BIAS[a][s][r][vi, rep] = float((np.tensordot(LUMA, dif, 1)[mm]).mean() / np.tensordot(LUMA, Y, 1)[mm].mean())
                        mp = m["primary"]
                        L1[a][s][vi, rep] = float(np.abs(dif[:, mp]).sum() / np.abs(Y[:, mp]).sum())
                        for r in ("silhouette", "material_boundary"):
                            GR[a][s][r][vi, rep] = grad_ratio(y, Y, m[r]) if m[r].any() else np.nan
                        if rep == 0:
                            keep[(name, s, a)] = y.astype(np.float32)
            for a in ARMS:
                st = np.stack(outs[a])
                v = st.var(0, ddof=1)
                for r in ROIS:
                    VAR[a][s][r][vi] = float(v[m[r]].mean()) if m[r].any() else np.nan
        keep[(name, "target")] = Y.astype(np.float32)
        keep[(name, "rois")] = m
        if name == "V0" and phys is not None:
            refs = [np.load(phys / f"REF_0_{j}.npz") for j in range(2)]
            PH = np.mean([(f["e1"] + f["e2"] + f["h"]).astype(np.float64) for f in refs], 0).reshape(512, 512, 3).transpose(2, 0, 1)
            for s in SPPS:
                for a in ARMS:
                    errs = []
                    for rep in range(reps):
                        with np.load(ev / f"{name}_spp{s}_r{rep}.npz") as f:
                            y = f[a].astype(np.float64)
                        mm = m["primary"]
                        errs.append(math.sqrt(((y - PH)[:, mm] ** 2).mean() / (PH[:, mm] ** 2).mean()))
                    phys_err[f"{a}_spp{s}"] = float(np.mean(errs))
            mm = m["primary"]
            phys_err["frozen_target_vs_physical"] = float(math.sqrt(((Y - PH)[:, mm] ** 2).mean() / (PH[:, mm] ** 2).mean()))
        print("analysed", name, flush=True)
    res = {"views": views, "reps": reps, "target_noise_relative_rms": {r: float(np.mean(v)) if v else None for r, v in target_noise.items()}}
    summary = {}
    for a in ARMS:
        summary[a] = {}
        for s in SPPS:
            row = {r: {"rel_rmse": float(np.nanmean(E[a][s][r])), "signed_bias": float(np.nanmean(BIAS[a][s][r])),
                       "output_variance_luma": float(np.nanmean(VAR[a][s][r]))} for r in ROIS}
            row["primary_rel_l1"] = float(L1[a][s].mean())
            for r in ("silhouette", "material_boundary"):
                row[r]["gradient_ratio"] = float(np.nanmean(GR[a][s][r]))
            summary[a][s] = row
    res["summary"] = summary
    # Decisions.
    cost = eman["cost"]
    frame = dman["ecro_frame_cost"]
    fs = np.array(SPPS, float)
    fm = np.array([frame[str(s)]["frame_ms_median"] if str(s) in frame else frame[s]["frame_ms_median"] for s in SPPS])
    slope, icpt = np.polyfit(fs, fm, 1)
    dec = {"R1_qualification": {}, "quality": {}, "cost_matched": {}, "safety": {}, "R3_attribution": {}, "effective_spp": {}}
    q_ok = True
    for s in SPPS:
        e0, e1 = summary["R0"][s]["primary"]["rel_rmse"], summary["R1"][s]["primary"]["rel_rmse"]
        sil = summary["R1"][s]["silhouette"]
        okq = (e1 < e0 and 0.8 <= sil["gradient_ratio"] <= 1.2 and
               all(abs(summary["R1"][s][r]["signed_bias"]) <= 0.01 for r in ("primary", "silhouette", "material_boundary")))
        dec["R1_qualification"][s] = {"pass": bool(okq), "R0": e0, "R1": e1, "silhouette_gradient_ratio": sil["gradient_ratio"],
                                      "bias": {r: summary["R1"][s][r]["signed_bias"] for r in ("primary", "silhouette", "material_boundary")}}
        q_ok &= okq
        D = boot(E["R1"][s]["primary"], E["R2"][s]["primary"])
        dec["quality"][s] = {**D, "material": bool(D["point"] >= 0.05 and D["ci95"][0] > 0)}
        D3 = boot(E["R1"][s]["primary"], E["R3"][s]["primary"])
        D23 = boot(E["R3"][s]["primary"], E["R2"][s]["primary"])
        dec["R3_attribution"][s] = {"R3_vs_R1": D3, "R2_vs_R3": D23}
        # cost-matched: R1 at s* with equal end-to-end time
        t2 = icpt + slope * s + cost["R2"]["latency_ms_median"]
        s_star = (t2 - cost["R1"]["latency_ms_median"] - icpt) / slope
        logs = np.log(fs)
        E1v = np.stack([E["R1"][k]["primary"] for k in SPPS])            # (3, V, R)
        lg = np.log(E1v)
        def interp(ls):
            x = math.log(s_star)
            if x <= logs[-1]:
                i = 0 if x <= logs[1] else 1
            else:
                i = 1
            w = (x - logs[i]) / (logs[i + 1] - logs[i])
            return np.exp(ls[i] * (1 - w) + ls[i + 1] * w)
        E1star = interp(lg)
        Dc = boot(E1star, E["R2"][s]["primary"])
        dec["cost_matched"][s] = {**Dc, "s_star": float(s_star), "end_to_end_ms_R2": float(t2),
                                  "material": bool(Dc["point"] >= 0.05 and Dc["ci95"][0] > 0)}
        safe = {"bias": {r: summary["R2"][s][r]["signed_bias"] for r in ("primary", "sss_interior", "silhouette", "material_boundary", "control")},
                "boundary_rmse_ratio": {r: summary["R2"][s][r]["rel_rmse"] / summary["R1"][s][r]["rel_rmse"] for r in ("silhouette", "material_boundary")},
                "silhouette_gradient_ratio": summary["R2"][s]["silhouette"]["gradient_ratio"]}
        safe["pass"] = bool(all(abs(v) <= 0.01 for v in safe["bias"].values()) and all(v <= 1.05 for v in safe["boundary_rmse_ratio"].values())
                            and 0.8 <= safe["silhouette_gradient_ratio"] <= 1.2)
        dec["safety"][s] = safe
        # effective spp of R2 relative to R1 and R0 (log-log)
        e2 = summary["R2"][s]["primary"]["rel_rmse"]
        r1c = [summary["R1"][k]["primary"]["rel_rmse"] for k in SPPS]
        a1 = np.polyfit(np.log(fs), np.log(r1c), 1)
        dec["effective_spp"][s] = {"R1_equivalent_spp": float(math.exp((math.log(e2) - a1[1]) / a1[0])),
                                   "R0_equivalent_spp": float(s * (summary["R0"][s]["primary"]["rel_rmse"] / e2) ** 2)}
    tn = res["target_noise_relative_rms"]["primary"]
    trust = all(tn <= 0.1 * summary["R1"][s]["primary"]["rel_rmse"] for s in SPPS)
    dec["target_trust"] = {"target_noise": tn, "pass": bool(trust)}
    qm = [dec["quality"][s]["material"] for s in SPPS]
    cm = [dec["cost_matched"][s]["material"] for s in SPPS]
    sf = [dec["safety"][s]["pass"] for s in SPPS]
    both = [q and c and f for q, c, f in zip(qm, cm, sf)]
    if not trust:
        verdict = "STOP_TARGETS_NOT_TRUSTWORTHY"
    elif not q_ok:
        verdict = "STOP_NO_QUALIFIED_RGB_BASELINE"
    elif all(both):
        verdict = "SUPPORTED"
    elif any(both):
        verdict = "PARTIALLY_SUPPORTED"
    elif any(q and f for q, f in zip(qm, sf)) and not any(cm):
        verdict = "QUALITY_ONLY_NOT_SUPPORTED_AT_MATCHED_COST"
    elif sum(abs(dec["quality"][s]["point"]) >= 0.05 and dec["quality"][s]["ci95"][0] <= 0 <= dec["quality"][s]["ci95"][1] for s in SPPS) >= 2:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "NOT_SUPPORTED"
    res["decision"] = dec
    res["verdict"] = verdict
    res["cost"] = {"reconstruction": cost, "ecro_frame": frame, "frame_fit_ms": [float(icpt), float(slope)]}
    res["physical_reference_V0"] = phys_err
    res["eval_manifest"] = {k: eman[k] for k in ("weights_sha256", "data_manifest_sha256", "deterministic_inference", "gpu")}
    (review / "metrics.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    figures(review, keep, views, summary, E)
    page(review, res)
    print(verdict, flush=True)


def tone(x, exposure):
    y = np.clip(x * exposure, 0, None)
    y = y / (1 + y)
    return np.clip(y ** (1 / 2.2), 0, 1).transpose(1, 2, 0)


def figures(review, keep, views, summary, E):
    colors = {"R0": "#2a78d6", "R1": "#eb6834", "R2": "#1baf7a", "R3": "#6250d6"}
    marks = {"R0": "o", "R1": "s", "R2": "^", "R3": "D"}
    fig, ax = plt.subplots(1, 4, figsize=(17, 4), layout="constrained")
    for a_, roi in zip(ax, ("primary", "sss_interior", "silhouette", "material_boundary")):
        for arm in ("R0", "R1", "R2", "R3"):
            a_.plot(SPPS, [summary[arm][s][roi]["rel_rmse"] for s in SPPS], marks[arm] + "-", color=colors[arm], label=arm)
        a_.set_xscale("log", base=2); a_.set_yscale("log"); a_.set_xticks(SPPS, [str(s) for s in SPPS])
        a_.set_title(roi.replace("_", " ")); a_.set_xlabel("spp"); a_.grid(alpha=.25)
    ax[0].set_ylabel("relative HDR RMSE vs frozen-ECRO target"); ax[0].legend()
    fig.suptitle("WL105 · error vs spp on held-out views (mean over 6 views x 8 seeds)")
    fig.savefig(review / "error_vs_spp.png", dpi=140); plt.close(fig)
    for name in views:
        Y = keep[(name, "target")]
        m = keep[(name, "rois")]
        exposure = 1.0 / max(np.percentile(np.tensordot(LUMA, Y, 1), 99), 1e-3)
        ys, xs = np.nonzero(m["silhouette"])
        cy, cx = (int(np.median(ys)), int(np.median(xs))) if len(ys) else (256, 256)
        crops = {"full": np.s_[:, :], "silhouette": np.s_[max(cy - 64, 0):cy + 64, max(cx - 64, 0):cx + 64]}
        mb = np.nonzero(m["material_boundary"])
        if len(mb[0]):
            k = len(mb[0]) // 2
            crops["material boundary"] = np.s_[max(mb[0][k] - 64, 0):mb[0][k] + 64, max(mb[1][k] - 64, 0):mb[1][k] + 64]
        for s in (4, 32):
            cols = [("raw R0", keep[(name, s, "R0")]), ("R1 RGB", keep[(name, s, "R1")]), ("R2 coupled", keep[(name, s, "R2")]),
                    ("target", Y)]
            fig, axs = plt.subplots(3 * len(crops), 4, figsize=(13, 9.6 * len(crops) / 2), layout="constrained", squeeze=False)
            cap = float(np.percentile(np.abs(np.tensordot(LUMA, keep[(name, s, "R0")] - Y, 1)), 99))
            for ci, (cname, sl) in enumerate(crops.items()):
                for j, (label, img) in enumerate(cols):
                    axs[3 * ci, j].imshow(tone(img[(slice(None),) + sl], exposure)); axs[3 * ci, j].set_title(f"{label} · {cname}", fontsize=8)
                    if label != "target":
                        e = np.tensordot(LUMA, img - Y, 1)[sl]
                        axs[3 * ci + 1, j].imshow(e, cmap="RdBu_r", vmin=-cap, vmax=cap); axs[3 * ci + 1, j].set_title("signed error", fontsize=8)
                        axs[3 * ci + 2, j].imshow(np.abs(e), cmap="magma", vmin=0, vmax=cap); axs[3 * ci + 2, j].set_title("|error|", fontsize=8)
                for a_ in axs[3 * ci:3 * ci + 3].ravel():
                    a_.set_axis_off()
            fig.suptitle(f"WL105 · {name} · {s} spp · common exposure; error scale ±{cap:.3g} (linear HDR luminance)")
            fig.savefig(review / f"view_{name}_spp{s}.png", dpi=110); plt.close(fig)


def page(review, res):
    d = res["decision"]
    rows = "".join(f"<tr><td>{s}</td><td>{d['quality'][s]['point']:+.3f} {[round(x, 3) for x in d['quality'][s]['ci95']]}</td>"
                   f"<td>{d['cost_matched'][s]['point']:+.3f} {[round(x, 3) for x in d['cost_matched'][s]['ci95']]} (s*={d['cost_matched'][s]['s_star']:.1f})</td>"
                   f"<td>{d['safety'][s]['pass']}</td><td>{d['R1_qualification'][s]['pass']}</td></tr>" for s in SPPS)
    imgs = "".join(f"<p><img src='{p.name}'></p>" for p in sorted(review.glob("view_*.png")))
    (review / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WL105 coupled reconstruction</title><style>:root{{--bg:#fcfcfb;--ink:#1f1f1f;--line:#d9d8d2}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--ink:#f0efec;--line:#3b3b38}}}}
body{{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:16px}}
img{{max-width:100%;height:auto;background:#fff}} td,th{{border-bottom:1px solid var(--line);padding:3px 8px}}</style></head><body>
<h1>WL105 · coupled LOW/HIGH vs combined-RGB reconstruction</h1><p><b>Verdict:</b> {html.escape(res['verdict'])}</p>
<table><tr><th>spp</th><th>D (R2 vs R1, equal spp)</th><th>D_cost (equal end-to-end time)</th><th>R2 safety</th><th>R1 qualified</th></tr>{rows}</table>
<p><img src="error_vs_spp.png"></p>{imgs}<p><a href="metrics.json">metrics.json</a></p></body></html>""")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), None if sys.argv[3] == "-" else Path(sys.argv[3]).resolve(),
         Path(sys.argv[4]).resolve())
