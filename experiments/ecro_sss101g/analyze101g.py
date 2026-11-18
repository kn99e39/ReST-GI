"""Offline WL101G metrics, frozen decision and reviewer figures.

python analyze101g.py RAW_RUN WL101_REFERENCE_DIR REVIEW_DIR
"""
import argparse
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_sss101e"))
sys.path.insert(0, str(HERE))
from analyze_recovery import interval, pixel_score  # noqa: E402  WL101E definitions
from analyze_contact import CAP, LUMA, tone  # noqa: E402  WL101C display
from common import sha, write_json  # noqa: E402
from regions101g import EDGES, P, PROTOCOL, VIEWS, contact_valid, view_regions  # noqa: E402
import tables101g as T  # noqa: E402

DECISION_ARMS = ("G", "S_G", "S_Gad")
RMSE_REGIONS = ("sphere_boundary", "sphere", "contact_shadow", "floor", "full")
REGIONS = ("full", "sphere", "sphere_interior", "sphere_boundary", "floor", "contact_shadow",
           "near", "flank", "near_all", "flank_all")
SHAPES = ("floor", "ceiling", "back wall", "left wall", "right wall", "panel", "sphere", "box A", "box B")
STYLE = {"U_E": ("#2a78d6", "o", "Uniform U (WL101E)"), "G": ("#eb6834", "s", "Etendue G (camera-independent)"),
         "S_G": ("#1baf7a", "^", "G + 2x2 spatial"), "S_Gad": ("#eda100", "D", "G + adaptive 10% spatial"),
         "P_F": ("#e87ba4", "v", "WL101F P (V0 near/flank)"), "Pv_own": ("#008300", "P", "Own-view P_v (upper bound)"),
         "learned": ("#6250d6", "X", "Learned HIGH")}


def families(names):
    out = {}
    for name in names:
        fam, i = name.rsplit("_", 1)
        out.setdefault(fam, []).append(int(i))
    return out


def scalars(grid, ref, mask):
    """grid (I,2,H,W,3) or (2,H,W,3); returns (I,2) signed luminance residual."""
    ref_mu = (ref[:, mask, :] @ LUMA).mean(-1)
    if grid.ndim == 4:
        v = (grid[:, mask, :].astype(np.float64) @ LUMA).mean(-1) - ref_mu
        return np.broadcast_to(v, (4, 2))
    return (grid[:, :, mask, :].astype(np.float64) @ LUMA).mean(-1) - ref_mu[None]


def arm_metrics(grid, weight, ref, total, reg, bands):
    res = {"regions": {}, "bands": []}
    mean_img = grid.mean(tuple(range(grid.ndim - 3)), dtype=np.float64)
    refm = ref.mean(0)
    for name in REGIONS:
        m = reg[name]
        if m.sum() == 0:
            res["regions"][name] = None
            continue
        row = {"pixels": int(m.sum()), "signed": interval(scalars(grid, ref, m)),
               "raw": pixel_score(mean_img, refm, m), "converged": T.cross_rmse(grid, ref, m)}
        if grid.ndim == 5:
            row["single_table"] = T.cross_rmse(grid, ref, m, same_replica=True)
        if weight is not None:
            frac = np.array([[weight[i, j][m].sum() / total[j][m].sum() for j in range(2)] for i in range(len(weight))])
            row["native_fraction"] = {"min": float(frac.min()), "mean": float(frac.mean())}
        res["regions"][name] = row
    if reg["near"].sum() and reg["flank"].sum():
        res["contrast"] = interval(scalars(grid, ref, reg["near"]) - scalars(grid, ref, reg["flank"]))
    for (a, b), m in zip(zip(EDGES, EDGES[1:]), bands):
        res["bands"].append({"range_m": [a, b], "pixels": int(m.sum()),
                             "signed": interval(scalars(grid, ref, m)) if m.sum() else None})
    return res


def load_view(raw, run, vid, refs):
    rows = sorted((r for r in run["stages"]["3"]["streams"] if r["view"] == vid), key=lambda r: r["stream"])
    assert len(rows) == 2
    files = []
    for r in rows:
        assert sha(raw / r["file"]) == run["files"][r["file"]]["sha256"]
        files.append(np.load(raw / r["file"]))
    if vid == "V0":
        ref = np.stack([np.load(refs / f"REF_0_{j}.npz")["h"] for j in range(2)]).astype(np.float64)
    else:
        ref = np.stack([f["query"] for f in files]).astype(np.float64)
    return files, ref


def view_analysis(raw, run, vid, refs):
    files, ref = load_view(raw, run, vid, refs)
    gfile = refs / "MAP_0.npz" if vid == "V0" else raw / f"gbuffer_{vid}.npz"
    reg, bands = view_regions(gfile, vid == "V0")
    total = np.stack([f["total_weight"] for f in files]).astype(np.float64)
    shape_w = np.stack([f["shape_weight"] for f in files]).astype(np.float64).mean(0)
    support = {}
    for name in REGIONS:
        m = reg[name]
        if m.sum():
            t = shape_w[:, m].sum(1)
            support[name] = {SHAPES[k]: float(t[k] / t.sum()) for k in range(9) if t.sum() > 0}
    result = {"contact_valid": contact_valid(reg), "region_pixels": {k: int(reg[k].sum()) for k in REGIONS},
              "queried_shape_fraction_of_high_weight": support, "arms": {}}
    means = {"physical": ref.mean(0)}
    learned = np.stack([f["learned"] for f in files])
    result["arms"]["learned"] = arm_metrics(learned, None, ref, total, reg, bands)
    means["learned"] = learned.mean(0, dtype=np.float64)
    for fam, reps in families(run["arms"]).items():
        reps = sorted(reps)
        grid = np.stack([np.stack([f[f"{fam}_{i}"] for f in files]) for i in reps])
        weight = np.stack([np.stack([f[f"{fam}_{i}:weight"] for f in files]) for i in reps]).astype(np.float64)
        result["arms"][fam] = arm_metrics(grid, weight, ref, total, reg, bands)
        if fam in ("U_E", "G", "S_G", "S_Gad", "P_F", "S_F", f"Pv_{vid}", f"Sv_{vid}"):
            means["Pv_own" if fam == f"Pv_{vid}" else "Sv_own" if fam == f"Sv_{vid}" else fam] = grid.mean((0, 1), dtype=np.float64)
        del grid, weight
    for f in files:
        f.close()
    return result, means, reg


def within(ci, band):
    return ci[0] >= -band and ci[1] <= band


def decide(views):
    D = P["decision"]
    band, prec = D["band_values"], D["precision_values"]
    rows = {}
    for arm in (*DECISION_ARMS, "U_E", "learned", "Un", "V", "S_U", "A_G", "SA_G", "P_F", "S_F", "SA_F", "Q_F", "A_F"):
        per = {}
        for vid, v in views.items():
            fam = v["arms"].get(arm)
            if fam is None:
                continue
            base = v["arms"]["U_E"]
            near, con = fam["regions"]["near"], fam.get("contrast")
            entry = {}
            if v["contact_valid"]:
                entry["contact_pass"] = within(near["signed"]["ci95"], band["near"]) and within(con["ci95"], band["contrast"])
                entry["precision_pass"] = near["signed"]["half_width95"] <= prec["near"] and con["half_width95"] <= prec["contrast"]
                entry["near"] = near["signed"]["mean"]
                entry["contrast"] = con["mean"]
            nd = {}
            for r in RMSE_REGIONS:
                a, b = fam["regions"][r]["converged"]["relative"], base["regions"][r]["converged"]["relative"]
                nd[r] = {"arm": a, "U_E": b, "pass": bool(a <= 1.10 * b + 0.0025)}
            entry["non_degradation"] = nd
            entry["non_degradation_pass"] = all(x["pass"] for x in nd.values())
            cov = [fam["regions"][r]["native_fraction"]["min"] for r in REGIONS
                   if fam["regions"].get(r) and "native_fraction" in fam["regions"][r]]
            entry["coverage_min"] = min(cov) if cov else None
            entry["coverage_pass"] = (min(cov) >= 0.995) if cov else True
            per[vid] = entry
        valid = [e for e in per.values() if "contact_pass" in e]
        summary = {
            "contact_pass_all": all(e["contact_pass"] for e in valid),
            "contact_pass_count": sum(e["contact_pass"] for e in valid), "contact_valid_views": len(valid),
            "precision_pass_all": all(e["precision_pass"] for e in valid),
            "non_degradation_pass_all": all(e["non_degradation_pass"] for e in per.values()),
            "coverage_pass_all": all(e["coverage_pass"] for e in per.values()),
            "max_abs_near": max(abs(e["near"]) for e in valid) if valid else None,
            "max_abs_contrast": max(abs(e["contrast"]) for e in valid) if valid else None,
            "near_spread": (max(e["near"] for e in valid) - min(e["near"] for e in valid)) if valid else None,
            "contrast_spread": (max(e["contrast"] for e in valid) - min(e["contrast"] for e in valid)) if valid else None}
        summary["sufficient"] = all(summary[k] for k in ("contact_pass_all", "precision_pass_all",
                                                          "non_degradation_pass_all", "coverage_pass_all"))
        rows[arm] = {"views": per, "summary": summary}
    gates_ok = {a: rows[a]["summary"]["precision_pass_all"] and rows[a]["summary"]["coverage_pass_all"] for a in DECISION_ARMS}
    gates_ok["U_E_precision"] = rows["U_E"]["summary"]["precision_pass_all"]
    if rows["G"]["summary"]["sufficient"]:
        verdict = "A_EXISTING_LAYOUT_SUFFICES_WITH_REUSABLE_PROJECTION"
    elif rows["S_G"]["summary"]["sufficient"] or rows["S_Gad"]["summary"]["sufficient"]:
        which = [a for a in ("S_G", "S_Gad") if rows[a]["summary"]["sufficient"]]
        verdict = "B_SPATIAL_REFINEMENT_NEEDED:" + "+".join(which)
    elif not all(gates_ok.values()):
        verdict = "D_EVIDENCE_INSUFFICIENT"
    else:
        verdict = "C_NEITHER_CANDIDATE_SATISFIES_CROSS_VIEW_BEHAVIOR"
    return verdict, rows, gates_ok


def improvements(views):
    """Material reduction of |excess| vs U_E and attribution contrasts (descriptive)."""
    out = {}
    for vid, v in views.items():
        if not v["contact_valid"]:
            continue
        def grid(arm, key):
            a = v["arms"][arm]
            g = a["regions"]["near"]["signed"]["cells"] if key == "near" else a["contrast"]["cells"]
            return np.asarray(g)
        row = {}
        for key in ("near", "contrast"):
            u = grid("U_E", key)
            ui = interval(u)
            resolved = ui["ci95"][0] > 0 or ui["ci95"][1] < 0
            entries = {}
            for arm in ("Un", "V", "G", "S_G", "S_Gad", "A_G", "SA_G", "S_U", "P_F", "S_F", f"Pv_{vid}", f"Sv_{vid}"):
                if arm not in v["arms"]:
                    continue
                red = interval(np.abs(u) - np.abs(grid(arm, key)))
                entries[arm] = {"reduction": red["mean"], "ci95": red["ci95"],
                                "material": bool(resolved and red["ci95"][0] > 0.10 * abs(ui["mean"]))}
            steps = {}
            for a, b in (("U_E", "Un"), ("Un", "V"), ("V", "G"), ("G", "S_G"), ("G", "A_G"), ("G", "SA_G"),
                         ("Un", "S_U"), ("G", "S_Gad"), ("G", f"Pv_{vid}"), ("S_G", f"Sv_{vid}"), ("G", "P_F")):
                if a in v["arms"] and b in v["arms"]:
                    steps[f"{a}->{b}"] = interval(grid(a, key) - grid(b, key))
            row[key] = {"U_E_excess": ui, "U_E_resolved": resolved, "reductions": entries, "signed_steps": steps}
        out[vid] = row
    return out


def transfer_matrix(views):
    mat = {}
    for src in VIEWS:
        mat[src] = {}
        for vid, v in views.items():
            a = v["arms"].get(f"Pv_{src}")
            s = v["arms"].get(f"Sv_{src}")
            if a is None:
                continue
            mat[src][vid] = {
                "P_near": a["regions"]["near"]["signed"]["mean"] if v["contact_valid"] else None,
                "S_near": s["regions"]["near"]["signed"]["mean"] if v["contact_valid"] else None,
                "P_full_converged": a["regions"]["full"]["converged"]["relative"],
                "S_full_converged": s["regions"]["full"]["converged"]["relative"],
                "P_boundary_converged": a["regions"]["sphere_boundary"]["converged"]["relative"],
                "P_native_full": a["regions"]["full"]["native_fraction"]["min"]}
    return mat


# ---------------------------------------------------------------- figures
def crop_box(reg, pad=24):
    m = reg["near_all"] | reg["flank_all"] | reg["sphere_boundary"]
    ys, xs = np.nonzero(reg["near_all"] | reg["flank_all"]) if (reg["near_all"] | reg["flank_all"]).any() else np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    return np.s_[max(0, y0 - pad):min(512, y1 + pad), max(0, x0 - pad):min(512, x1 + pad)]


def view_figure(out, vid, means, reg, which, filename, crop):
    labels = [k for k in which if k in means]
    sl = crop_box(reg) if crop else np.s_[:, :]
    n = len(labels)
    fig, axs = plt.subplots(2, n, figsize=(2.3 * n, 4.9), layout="constrained", squeeze=False)
    ref = means["physical"]
    for c, k in enumerate(labels):
        top, bottom = axs[:, c]
        top.imshow(tone(means[k][sl]), interpolation="none")
        top.set_title("Physical HIGH" if k == "physical" else STYLE.get(k, (None, None, k))[2], fontsize=8)
        scale = bottom.imshow(((means[k] - ref)[sl]) @ LUMA, vmin=-CAP, vmax=CAP, cmap="RdBu_r", interpolation="none")
        bottom.set_title("signed residual", fontsize=8)
        for a in (top, bottom):
            a.set_axis_off()
    fig.colorbar(scale, ax=axs.ravel().tolist(), shrink=.6, label="Signed linear HDR luminance")
    fig.suptitle(f"WL101G · view {vid} · {'contact crop' if crop else 'full frame'} · shared exposure and residual scale", fontsize=10)
    fig.savefig(out / filename, dpi=130)
    plt.close(fig)


def radial_figure(out, views):
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.2), layout="constrained", sharey=True)
    for ax, vid in zip(axs.flat, VIEWS):
        v = views[vid]
        for arm in ("U_E", "G", "S_G", "S_Gad", "P_F", "Pv_own", "learned"):
            fam = f"Pv_{vid}" if arm == "Pv_own" else arm
            bands = v["arms"][fam]["bands"]
            x = [np.mean(b["range_m"]) for b in bands if b["signed"]]
            y = [b["signed"]["mean"] for b in bands if b["signed"]]
            if not x:
                continue
            color, marker, label = STYLE[arm]
            ax.plot(x, y, marker=marker, ms=4, lw=1.6, color=color, label=label)
        ax.axhline(0, color="#58616b", lw=.8)
        ax.axvspan(0, .18, color="#bc7040", alpha=.07)
        ax.axvspan(.26, .40, color="#2b4260", alpha=.06)
        ax.set_title(f"{vid}{'' if v['contact_valid'] else ' (contact endpoints n/a)'}", fontsize=10)
        ax.grid(alpha=.2)
        ax.set_xlabel("Floor radius from support point (m)")
    axs[0, 0].set_ylabel("Signed HIGH luminance residual")
    axs[1, 0].set_ylabel("Signed HIGH luminance residual")
    axs[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("WL101G · radial floor residual by view (V0: frozen front half; held-out: all visible floor)")
    fig.savefig(out / "radial_profiles.png", dpi=140)
    plt.close(fig)


def endpoint_figure(out, views):
    band = P["decision"]["band_values"]
    arms = ("U_E", "G", "S_G", "S_Gad", "P_F", "Pv_own", "learned")
    valid = [v for v in VIEWS if views[v]["contact_valid"]]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for ax, key, b in ((axs[0], "near", band["near"]), (axs[1], "contrast", band["contrast"])):
        for k, arm in enumerate(arms):
            xs, ys, lo, hi = [], [], [], []
            for vi, vid in enumerate(valid):
                fam = f"Pv_{vid}" if arm == "Pv_own" else arm
                a = views[vid]["arms"][fam]
                r = a["regions"]["near"]["signed"] if key == "near" else a["contrast"]
                xs.append(vi + (k - 3) * 0.11)
                ys.append(r["mean"]); lo.append(r["mean"] - r["ci95"][0]); hi.append(r["ci95"][1] - r["mean"])
            color, marker, label = STYLE[arm]
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=marker, ms=6, color=color, label=label, capsize=2, lw=1.2)
        ax.axhspan(-b, b, color="#7a7a7a", alpha=.12, label="frozen band")
        ax.axhline(0, color="#58616b", lw=.8)
        ax.set_xticks(range(len(valid)), valid)
        ax.set_title("Near-floor signed excess" if key == "near" else "Near-minus-flank excess", fontsize=10)
        ax.grid(alpha=.2, axis="y")
    axs[0].legend(fontsize=7, ncol=2)
    fig.suptitle("WL101G · contact endpoints across contact-valid views (95% intervals)")
    fig.savefig(out / "contact_endpoints.png", dpi=140)
    plt.close(fig)


def rmse_figure(out, views):
    arms = ("U_E", "G", "S_G", "S_Gad", "P_F", "Pv_own", "learned")
    fig, axs = plt.subplots(1, len(RMSE_REGIONS), figsize=(16, 4.2), layout="constrained", sharex=True)
    for ax, region in zip(axs, RMSE_REGIONS):
        for k, arm in enumerate(arms):
            ys = []
            for vid in VIEWS:
                fam = f"Pv_{vid}" if arm == "Pv_own" else arm
                ys.append(100 * views[vid]["arms"][fam]["regions"][region]["converged"]["relative"])
            color, marker, label = STYLE[arm]
            ax.plot(range(len(VIEWS)), ys, marker=marker, ms=5, lw=1.2, color=color, label=label)
        ax.set_xticks(range(len(VIEWS)), VIEWS)
        ax.set_title(region.replace("_", " "), fontsize=10)
        ax.grid(alpha=.2)
    axs[0].set_ylabel("Noise-corrected relative RGB HDR RMSE (%)")
    axs[0].legend(fontsize=7)
    fig.suptitle("WL101G · region fidelity by view (converged-table bias)")
    fig.savefig(out / "region_rmse.png", dpi=140)
    plt.close(fig)


def transfer_figure(out, mat):
    srcs = [s for s in VIEWS if mat.get(s)]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    for ax, key, title in ((axs[0], "P_full_converged", "Full-frame RMSE (%) of P_v"),
                           (axs[1], "P_near", "Near-floor signed excess of P_v")):
        data = np.full((len(srcs), len(VIEWS)), np.nan)
        for a, s in enumerate(srcs):
            for b, vid in enumerate(VIEWS):
                val = mat[s].get(vid, {}).get(key)
                if val is not None:
                    data[a, b] = 100 * val if key.endswith("converged") else val
        lim = np.nanmax(np.abs(data))
        im = ax.imshow(data, cmap="Blues" if key.endswith("converged") else "RdBu_r",
                       vmin=0 if key.endswith("converged") else -lim, vmax=lim)
        for a in range(len(srcs)):
            for b in range(len(VIEWS)):
                if np.isfinite(data[a, b]):
                    ax.text(b, a, f"{data[a, b]:.2f}" if key.endswith("converged") else f"{data[a, b]:+.3f}",
                            ha="center", va="center", fontsize=7, color="#1f1f1f")
        ax.set_xticks(range(len(VIEWS)), VIEWS)
        ax.set_yticks(range(len(srcs)), srcs)
        ax.set_xlabel("Evaluated view")
        ax.set_ylabel("Construction view of P_v")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle("WL101G · camera-conditioned projection transfer (diagnostic, never a candidate)")
    fig.savefig(out / "transfer_matrix.png", dpi=140)
    plt.close(fig)


def page(out, result):
    d = result["decision"]
    def cell(x, fmt="{:+.5f}"):
        return "" if x is None else (fmt.format(x) if isinstance(x, float) else html.escape(str(x)))
    rows = []
    for arm, r in d["arms"].items():
        s = r["summary"]
        rows.append(f"<tr><td>{arm}</td><td>{s['contact_pass_count']}/{s['contact_valid_views']}</td>"
                    f"<td>{cell(s['max_abs_near'])}</td><td>{cell(s['max_abs_contrast'])}</td>"
                    f"<td>{s['non_degradation_pass_all']}</td><td>{s['coverage_pass_all']}</td>"
                    f"<td>{s['precision_pass_all']}</td><td><b>{s['sufficient']}</b></td></tr>")
    views = []
    for vid, v in result["views"].items():
        cells = []
        for arm in ("U_E", "G", "S_G", "S_Gad", "P_F", f"Pv_{vid}", "learned"):
            a = v["arms"][arm]
            near = a["regions"]["near"]["signed"]["mean"] if v["contact_valid"] else None
            con = a["contrast"]["mean"] if v["contact_valid"] else None
            b = a["regions"]["sphere_boundary"]["converged"]["relative"]
            f = a["regions"]["full"]["converged"]["relative"]
            cells.append(f"<tr><td>{arm}</td><td>{cell(near)}</td><td>{cell(con)}</td>"
                         f"<td>{100*b:.2f}%</td><td>{100*f:.2f}%</td></tr>")
        views.append(f"<h3>{vid} {'(contact-valid)' if v['contact_valid'] else '(contact endpoints n/a)'}</h3>"
                     f"<table><tr><th>arm</th><th>near</th><th>near−flank</th><th>silhouette RMSE</th><th>full RMSE</th></tr>"
                     + "".join(cells) + "</table>"
                     f"<p><img src='view_{vid}.png'><br><img src='contact_{vid}.png'></p>")
    text = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WL101G cross-view projection</title><style>
:root{{--bg:#fcfcfb;--ink:#1f1f1f;--muted:#5e5e5a;--line:#d9d8d2}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#3b3b38}}}}
body{{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1200px;padding:16px}}
img{{max-width:100%;height:auto;background:#fff}} table{{border-collapse:collapse;margin:8px 0;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}}
td,th{{border-bottom:1px solid var(--line);padding:3px 8px;text-align:right}} td:first-child,th:first-child{{text-align:left}}
p.note{{color:var(--muted)}}</style></head><body>
<h1>WL101G · camera-independent HIGH projection across views</h1>
<p><b>Frozen verdict:</b> {html.escape(d['verdict'])}</p>
<p class="note">Physical-Oracle tables only. Contact bands ±{P['decision']['band_values']['near']:.5f} (near) and ±{P['decision']['band_values']['contrast']:.5f} (near−flank). RMSE values are noise-corrected converged-table estimates.</p>
<table><tr><th>arm</th><th>contact pass</th><th>max |near|</th><th>max |near−flank|</th><th>regions ok</th><th>coverage ok</th><th>precision ok</th><th>sufficient</th></tr>{''.join(rows)}</table>
<p><img src="contact_endpoints.png"></p><p><img src="region_rmse.png"></p><p><img src="radial_profiles.png"></p><p><img src="transfer_matrix.png"></p>
{''.join(views)}
<p><a href="metrics.json">metrics.json</a> · <a href="manifest101g.json">manifest</a></p></body></html>"""
    (out / "index.html").write_text(text)


def main(raw, refs, out):
    run = json.loads((raw / "run_manifest.json").read_text())
    assert run.get("completed") and run["protocol_sha256"] == sha(PROTOCOL)
    for name, rec in run["files"].items():
        assert (raw / name).stat().st_size == rec["bytes"]
    out.mkdir(parents=True, exist_ok=True)
    views, means_all, regs = {}, {}, {}
    for vid in VIEWS:
        views[vid], means_all[vid], regs[vid] = view_analysis(raw, run, vid, refs)
        print("analysed", vid, flush=True)
    verdict, arms, gates = decide(views)
    result = {"protocol_sha256": run["protocol_sha256"], "run_manifest_sha256": sha(raw / "run_manifest.json"),
              "decision": {"verdict": verdict, "arms": arms, "decision_gates": gates},
              "improvements": improvements(views), "pv_transfer": transfer_matrix(views),
              "construction": run["stages"]["1"], "lookup_checks": run["stages"]["1b"],
              "benchmark": run["stages"]["4"], "table_bytes": run["table_bytes"], "views": views}
    write_json(out / "metrics.json", result)
    for vid in VIEWS:
        which = ["physical", "U_E", "G", "S_G", "S_Gad", "P_F", "Pv_own", "learned"]
        view_figure(out, vid, means_all[vid], regs[vid], which, f"view_{vid}.png", False)
        view_figure(out, vid, means_all[vid], regs[vid], which, f"contact_{vid}.png", True)
    radial_figure(out, views)
    endpoint_figure(out, views)
    rmse_figure(out, views)
    transfer_figure(out, result["pv_transfer"])
    np.savez_compressed(raw / "review_maps.npz", **{f"{vid}:{k}": v for vid, m in means_all.items() for k, v in m.items()})
    page(out, result)
    manifest = {"source_run_manifest_sha256": result["run_manifest_sha256"],
                "review_files": {p.name: sha(p) for p in sorted(out.iterdir()) if p.name != "manifest101g.json"},
                "raw_files": {k: v["sha256"] for k, v in run["files"].items()},
                "review_maps_sha256": sha(raw / "review_maps.npz")}
    write_json(out / "manifest101g.json", manifest)
    print(verdict, flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "references", "output"):
        a.add_argument(name, type=Path)
    x = a.parse_args()
    main(x.raw, x.references, x.output)
