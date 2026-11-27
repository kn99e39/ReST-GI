"""WL111A reviewer artifacts (not part of any decision; reads the final evidence only) and the evaluation/111A package.

python figs_r111a.py FINAL_DIR OUT_DIR [COMMIT]
  frontier_<regime>.png         primary RMSE ratio vs R3 at 4 spp against median E2E (both realizations, S111 control, R3's
                                own curve), 95 % intervals
  equal_spp_<roi>.png           relative RMSE vs spp per arm (decision set)
  roi_ratio_4spp.png            GM RMSE ratio vs R3 at 4 spp per ROI (decision / historical / edited), both realizations
  bias_variance_4spp.png        relative bias RMS and stochastic std per ROI and component (decision set)
  runtime_breakdown.png         synchronized spans at 4 spp (update, ECRO frame, guide acquisition, reconstruction)
  images_<key>_spp<k>.png       identically exposed LDR (reference, R0, R3, S111 r0, S111A r0, S111A r1), signed luminance
                                error of replicate 0, expectation error (mean over replicates - reference), std
  crops_<key>_spp<k>.png        silhouette / material-boundary crops (windows chosen from geometry masks only)
  tables.md                     per-view and per-ROI tables, realization comparison, edited-state qualification, timing
Display: LDR = clip(x, 0, 1) ^ (1 / 2.2) at exposure 1 (WL110 / WL111 convention); error maps are luminance differences
divided by the mean primary-ROI reference luminance, fixed symmetric scale per figure.
"""
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common_r111a as A
import common_r111 as C
import analyze_r111 as AZ
import analyzeF_r111a as AF

COL = {"R0": "#9e9e9e", "R3": "#d62728", "S111_r0": "#1f77b4", "S111_r1": "#9ecae1", "S111A_r0": "#2ca02c", "S111A_r1": "#98df8a"}
LAB = {"R0": "R0 native", "R3": "R3 (frozen)", "S111_r0": "S111 r0", "S111_r1": "S111 r1", "S111A_r0": "S111A r0", "S111A_r1": "S111A r1"}
IMG_ARMS = ("R0", "R3", "S111_r0", "S111A_r0", "S111A_r1")


def ldr(x):
    return np.clip(np.transpose(np.asarray(x, np.float32), (1, 2, 0)), 0, 1) ** (1 / 2.2)


def lum(x):
    return np.tensordot(C.LUMA, np.asarray(x, np.float64), 1)


def frontier(an, data, out):
    V = AF.P["views"]
    for regime in ("static", "historical", "edited"):
        rr = an["routes"].get(regime, {})
        if "S111A_r0" not in rr:
            continue
        views = rr["views"]
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        for arm, mk in (("S111A_r0", "o-"), ("S111A_r1", "s--"), ("S111_r0", "^-"), ("S111_r1", "v--")):
            pts = rr[arm]["points"]
            ax.plot([p["e2e_ms"] for p in pts], [p["rmse_ratio"] for p in pts], mk, color=COL[arm], ms=4, label=LAB[arm])
            for p in pts:
                ax.plot([p["e2e_ms"]] * 2, p["ci95"], color=COL[arm], lw=0.7)
                if arm == "S111A_r0":
                    ax.annotate(str(p["spp"]), (p["e2e_ms"], p["rmse_ratio"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        t = AF.timing(an["_root"])
        treg = "edit" if regime == "edited" else "static"
        xs, ys = [], []
        for spp in AF.GRID:
            e = AZ.e2e(t, treg, "R3", spp)[0]
            r = AZ.gm_ratio(data, views, ("R3", spp, "single"), ("R3", 4, "single"), "primary")
            if e is not None and r == r:
                xs.append(e)
                ys.append(r)
        ax.plot(xs, ys, "d:", color=COL["R3"], ms=4, label="R3 at its own spp")
        r3t = rr["S111A_r0"]["R3_e2e_ms"]
        ax.axhline(1.0, color=COL["R3"], lw=0.8)
        ax.axhline(1.05, color=COL["R3"], lw=0.6, ls=":")
        ax.axhline(0.90, color="k", lw=0.5, ls=":")
        ax.axvline(r3t, color=COL["R3"], lw=0.8)
        ax.axvline(0.85 * r3t, color=COL["R3"], lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"median end-to-end frame time, {treg} timing (ms)")
        ax.set_ylabel("primary RMSE ratio vs R3 at 4 spp (GM over views)")
        ax.set_title(f"cold frames, {regime} ({len(views)} views / states); route A: y<=1.05 & x<=0.85 R3; route B: x<=R3 & y<=0.90",
                     fontsize=8)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"frontier_{regime}.png", dpi=130)
        plt.close(fig)


def equal_spp(data, views, out):
    for roi in ("primary", "silhouette", "material_boundary", "sss_interior", "contact"):
        curves = {}
        for arm in AF.ARMS:
            ys = [[x for v in views if (x := AF.rm(data, v, arm, s, roi)) is not None] for s in AF.GRID]
            if all(ys):
                curves[arm] = [float(np.mean(y)) for y in ys]
        if not curves:
            continue
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        for arm, ys in curves.items():
            ax.plot(AF.GRID, ys, "o-", color=COL[arm], ms=3, label=LAB[arm])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("spp")
        ax.set_ylabel(f"debiased relative HDR RMSE ({roi}, mean over views)")
        ax.set_title(f"decision set K1-K6 + N1-N6, {roi}", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"equal_spp_{roi}.png", dpi=130)
        plt.close(fig)


def roi_ratio(an, data, out):
    sets = {"decision": an["routes"]["static"]["views"], "historical": an["routes"]["historical"]["views"],
            "edited (all states)": list(an["edited_qualification"])}
    rois = ("primary", "sss_interior", "silhouette", "material_boundary", "control", "contact")
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (name, views) in zip(axs, sets.items()):
        w = 0.2
        for k, arm in enumerate(("S111_r0", "S111_r1", "S111A_r0", "S111A_r1")):
            ys = [AZ.gm_ratio(data, views, (arm, 4, "single"), ("R3", 4, "single"), roi) for roi in rois]
            ax.bar(np.arange(len(rois)) + (k - 1.5) * w, ys, w, color=COL[arm], label=LAB[arm])
        ax.axhline(1.0, color=COL["R3"], lw=0.8)
        ax.axhline(1.05, color=COL["R3"], lw=0.6, ls=":")
        ax.set_xticks(range(len(rois)))
        ax.set_xticklabels(rois, rotation=30, fontsize=7)
        ax.set_title(f"{name}: RMSE / R3, 4 spp", fontsize=9)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "roi_ratio_4spp.png", dpi=130)
    plt.close(fig)


def bias_variance(an, out):
    d = an["decomposition_4spp"]["decision"]
    rows = [(c, r) for c in ("total", "low", "high") for r in ("primary", "silhouette", "material_boundary", "sss_interior")]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.0), sharey=True)
    for ax, q in zip(axs, ("rel_bias_rms", "rel_std")):
        w = 0.16
        for k, arm in enumerate(("R3", "S111_r0", "S111_r1", "S111A_r0", "S111A_r1")):
            ys = [d[arm].get(f"{c}_{r}", {}).get(q, np.nan) for c, r in rows]
            ax.bar(np.arange(len(rows)) + (k - 2) * w, ys, w, color=COL[arm], label=LAB[arm])
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([f"{c}:{r}" for c, r in rows], rotation=40, fontsize=6)
        ax.set_title({"rel_bias_rms": "systematic: relative RMS of (mean of 8 - reference), debiased",
                      "rel_std": "stochastic: relative std over 8 replicates"}[q], fontsize=8)
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "bias_variance_4spp.png", dpi=130)
    plt.close(fig)


def runtime(an, out):
    c = an["timing"].get("components")
    if not c:
        return
    med = c["median_ms"]
    arms = ("R0", "R3", "S111", "S111A")
    parts = ("update", "frame", "guides", "recon")
    cols = {"update": "#bdbdbd", "frame": "#636363", "guides": "#fdae6b", "recon": "#3182bd"}
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for i, a in enumerate(arms):
        left = 0.0
        for p in parts:
            v = med[a][p]
            ax.barh(i, v, left=left, color=cols[p], label=p if i == 0 else None)
            left += v
        ax.text(left, i, f" {left:.2f} ms (subsystem {med[a]['guides'] + med[a]['recon']:.2f})", va="center", fontsize=7)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arms)
    ax.set_xlabel("synchronized span per edited frame at 512 x 512, 4 spp (ms, median of 30)")
    ax.legend(fontsize=7, ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "runtime_breakdown.png", dpi=130)
    plt.close(fig)


def images(final, out):
    for f in sorted((final / "maps").glob("*.npz")):
        z = np.load(f)
        ref = z["ref"].astype(np.float64)
        prim = z["mask_primary"] if "mask_primary" in z.files else np.ones(ref.shape[1:], bool)
        s = float(lum(ref)[prim].mean())
        cols = ["ref"] + list(IMG_ARMS)
        fig, axs = plt.subplots(4, len(cols), figsize=(2.3 * len(cols), 9.4))
        lim = 0.5
        for j, a in enumerate(cols):
            img = ref if a == "ref" else z[f"{a}_rep0"].astype(np.float64)
            axs[0, j].imshow(ldr(img))
            axs[0, j].set_title("T_ECRO reference" if a == "ref" else LAB[a], fontsize=8)
            if a == "ref":
                if "phys" in z.files:
                    axs[1, j].imshow(ldr(z["phys"].astype(np.float64)))
                    axs[1, j].set_title("T_PHYSICAL (descriptive)", fontsize=7)
                axs[2, j].imshow(prim, cmap="gray")
                axs[2, j].set_title("primary ROI", fontsize=7)
                continue
            axs[1, j].imshow((lum(img) - lum(ref)) / s, cmap="RdBu_r", vmin=-lim, vmax=lim)
            axs[1, j].set_title("signed error, replicate 0", fontsize=7)
            m = z[f"{a}_mean"].astype(np.float64)
            axs[2, j].imshow((lum(m) - lum(ref)) / s, cmap="RdBu_r", vmin=-lim / 4, vmax=lim / 4)
            axs[2, j].set_title(f"expectation error (mean of 8), +-{lim / 4}", fontsize=7)
            axs[3, j].imshow(np.sqrt(np.maximum(lum(z[f"{a}_var"].astype(np.float64)), 0)) / s, cmap="magma", vmin=0, vmax=lim / 4)
            axs[3, j].set_title(f"std over replicates, 0..{lim / 4}", fontsize=7)
        for ax in axs.ravel():
            ax.axis("off")
        fig.suptitle(f"{f.stem}: LDR exposure 1; errors = luminance difference / mean primary reference luminance", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"images_{f.stem}.png", dpi=110)
        plt.close(fig)
        if "mask_silhouette" in z.files:
            crops(z, ref, s, f.stem, out)


def crops(z, ref, s, stem, out, size=96):
    wins = []
    for roi in ("silhouette", "material_boundary"):
        m = z[f"mask_{roi}"].astype(np.float64)
        if m.sum() == 0:
            continue
        cs = np.cumsum(np.cumsum(np.pad(m, ((1, 0), (1, 0))), 0), 1)
        H = m.shape[0] - size
        dens = cs[size:, size:] - cs[:-size, size:] - cs[size:, :-size] + cs[:-size, :-size]
        y, x = np.unravel_index(np.argmax(dens[:H + 1, :H + 1]), dens[:H + 1, :H + 1].shape)
        wins.append((roi, y, x))
    if not wins:
        return
    cols = ["ref"] + list(IMG_ARMS)
    fig, axs = plt.subplots(2 * len(wins), len(cols), figsize=(2.0 * len(cols), 4.2 * len(wins)))
    axs = np.atleast_2d(axs)
    for i, (roi, y, x) in enumerate(wins):
        for j, a in enumerate(cols):
            img = ref if a == "ref" else z[f"{a}_rep0"].astype(np.float64)
            sl = (slice(None), slice(y, y + size), slice(x, x + size))
            axs[2 * i, j].imshow(ldr(img[sl]), interpolation="nearest")
            axs[2 * i, j].set_title(("reference" if a == "ref" else LAB[a]) + f" [{roi}]", fontsize=6)
            if a != "ref":
                m = z[f"{a}_mean"].astype(np.float64)
                axs[2 * i + 1, j].imshow(((lum(m) - lum(ref)) / s)[y:y + size, x:x + size], cmap="RdBu_r", vmin=-0.25, vmax=0.25,
                                         interpolation="nearest")
                axs[2 * i + 1, j].set_title("expectation error", fontsize=6)
            else:
                axs[2 * i + 1, j].imshow(z[f"mask_{roi}"][y:y + size, x:x + size], cmap="gray", interpolation="nearest")
                axs[2 * i + 1, j].set_title(f"{roi} mask", fontsize=6)
    for ax in axs.ravel():
        ax.axis("off")
    fig.suptitle(f"{stem}: {size} px crops at the densest {', '.join(w[0] for w in wins)} window (geometry masks only)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"crops_{stem}.png", dpi=120)
    plt.close(fig)


def fmt(x, p=4):
    return "n/a" if x is None or x != x else f"{x:.{p}f}"


def tables(an, data, out):
    L = ["# WL111A final tables (generated by figs_r111a.py from analysis.json and the quality evidence)", "",
         f"Verdict: **{an['verdict']}**  inputs: `{an['verdict_inputs']}`", ""]
    L += ["## Routes (anchor R3 at 4 spp)", "", "| regime | arm | A pass | A point | B pass | B point / D |", "|---|---|---|---|---|---|"]
    for regime, rr in an["routes"].items():
        for arm in ("S111A_r0", "S111A_r1", "S111_r0", "S111_r1"):
            if arm in rr:
                a, b = rr[arm]["A"], rr[arm]["B"]
                L.append(f"| {regime} | {arm} | {a.get('pass')} | {a.get('point', a.get('reason', ''))} | {b.get('pass')} | "
                         f"{b.get('point', b.get('reason', ''))} / {fmt(b.get('D'), 3)} |")
    L += ["", "## Mechanism: S111A / S111 (same index), decision set, GM primary RMSE ratio [95 %]", "", "| spp | r0 | r1 |", "|---|---|---|"]
    for spp in AF.GRID:
        row = [f"{fmt(an['mechanism'][f'r{ri}'][str(spp)]['primary'][0], 3)} [{fmt(an['mechanism'][f'r{ri}'][str(spp)]['primary'][1][0], 3)}, "
               f"{fmt(an['mechanism'][f'r{ri}'][str(spp)]['primary'][1][1], 3)}]" for ri in AF.REAL]
        L.append(f"| {spp} | {row[0]} | {row[1]} |")
    L += ["", "## Per-ROI GM ratio vs R3 at 4 spp", ""]
    for name, views in (("decision", an["routes"]["static"]["views"]), ("historical", an["routes"]["historical"]["views"]),
                        ("edited (all)", list(an["edited_qualification"]))):
        L += [f"### {name}", "", "| arm | " + " | ".join(AF.ROIS) + " |", "|---" * (len(AF.ROIS) + 1) + "|"]
        for arm in AF.ARMS:
            L.append(f"| {arm} | " + " | ".join(fmt(AZ.gm_ratio(data, views, (arm, 4, "single"), ("R3", 4, "single"), r), 3) for r in AF.ROIS) + " |")
        L.append("")
    L += ["## Per-view primary relative RMSE at 4 spp", "", "| view | " + " | ".join(AF.ARMS) + " | ref SE |", "|---" * (len(AF.ARMS) + 2) + "|"]
    for v, d in an["per_view"].items():
        q = an["static_qualification"].get(v) or an["historical_reference_precision"].get(v) or an["edited_qualification"].get(v, {})
        L.append(f"| {v} | " + " | ".join(fmt(d["4"][a]["primary"]) for a in AF.ARMS) + f" | {fmt(q.get('ref_se'), 5)} |")
    L += ["", "## Edited-state qualification", "", "| state | change | noise | ref SE | R3 4 spp | qualifies |", "|---|---|---|---|---|---|"]
    for k, q in an["edited_qualification"].items():
        L.append(f"| {k} | {fmt(q['total_change'])} | {fmt(q['noise_total'], 5)} | {fmt(q['ref_se'], 5)} | {fmt(q['R3_4spp_rmse'])} | {q['qualifies']} |")
    L += ["", "## Contract C (static, routed point else 4 spp)", ""]
    for ri, c in (an.get("contract_C") or {}).items():
        L.append(f"- {ri}: point {c['point']}, pass {c['pass']}, bias {{{', '.join(f'{k}: {fmt(v)}' for k, v in c['bias'].items())}}}, "
                 f"silhouette grad {fmt(c['silhouette_grad_ratio'], 3)}, boundary vs R3 "
                 f"{{{', '.join(f'{k}: {fmt(v[0], 3)} [{fmt(v[1][0], 3)}, {fmt(v[1][1], 3)}]' for k, v in c['boundary_vs_R3'].items())}}}")
    L += ["", "## Timing (median ms)", ""]
    for name in ("static", "edit"):
        tt = an["timing"].get(name, {})
        if tt:
            L += [f"### {name}", "", "| spp | " + " | ".join(("R0", "R3", "S111_r0", "S111A_r0")) + " |", "|---" * 5 + "|"]
            for spp in AF.GRID:
                L.append(f"| {spp} | " + " | ".join(fmt(tt.get(f"{a}_{spp}"), 2) for a in ("R0", "R3", "S111_r0", "S111A_r0")) + " |")
            L.append("")
    fu = an["timing"].get("futility")
    if fu:
        L.append(f"Futility: camera-form ratio S111A / R3 {fmt(fu['S111A_over_R3_camera_form'], 3)}, static overhead "
                 f"{fmt(fu['S111A_static_overhead_ms'], 2)} ms, stop {fu['stop']}")
    L += ["", "## T_PHYSICAL (descriptive), 4 spp primary relative RMSE vs T_PHYSICAL; transport-depth difference", ""]
    for k, p in an["physical"].items():
        L.append(f"- {k}: |T_PHYSICAL - T_ECRO| primary {fmt(p['phys_minus_ecro_rel']['primary'])} (signed "
                 f"{fmt(p['phys_minus_ecro_signed']['primary'])}); " + ", ".join(f"{a} {fmt(p['vs_T_PHYSICAL_4spp'][a]['primary'])}" for a in AF.ARMS))
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def package(final, figs, dest, commit):
    dest.mkdir(parents=True, exist_ok=True)
    man = {"batch": "WL111A", "code_commit": commit, "protocol_sha256": C.sha(A.HERE / "protocol" / "recon111a.json"),
           "files": {}, "sources": {}}
    for f in sorted(figs.iterdir()):
        if f.suffix in (".png", ".md"):
            shutil.copy2(f, dest / f.name)
            man["files"][f.name] = {"sha256": C.sha(dest / f.name), "generator": "figs_r111a.py"}
    for name, src in (("analysis111A.json", final / "analysis.json"), ("sealed_new111A.json", final / "sealed_new.json"),
                      ("states111A.json", final / "states.json"), ("futility111A.json", final / "timing" / "futility.json")):
        if src.exists():
            shutil.copy2(src, dest / name)
            man["files"][name] = {"sha256": C.sha(dest / name), "source": str(src.relative_to(final))}
    for sub in ("quality", "timing", "refs", "maps"):
        d = final / sub
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.suffix in (".json", ".npz"):
                    man["sources"][f"{sub}/{f.name}"] = C.sha(f)
    C.write(dest / "manifest111A.json", man)
    print("PACKAGED", len(man["files"]), "files;", len(man["sources"]), "evidence files hashed", flush=True)


def main(final, out, commit=None):
    final, out = Path(final), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    an = C.read(final / "analysis.json")
    an["_root"] = final
    recs = AF.load(final)
    data = AF.tables(recs)
    frontier(an, data, out)
    equal_spp(data, an["routes"]["static"]["views"], out)
    roi_ratio(an, data, out)
    bias_variance(an, out)
    runtime(an, out)
    images(final, out)
    tables(an, data, out)
    if commit:
        package(final, out, out.parent / "package", commit)
    print("FIGS_COMPLETE", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
