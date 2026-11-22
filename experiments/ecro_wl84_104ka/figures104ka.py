"""WL104K-A reviewer package (evaluation/104K-A): readable figures, index.html, manifest.

Usage: python figures104ka.py <run_root> <out_dir>
"""
import argparse
import html
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import FuncNorm  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common104ka as KA  # noqa: E402

CM = KA.CM
COL = {"reference": "#111111", "A": "#1f77b4", "T": "#e08214", "P": "#2ca02c"}
SHORT = {"A": "A shared ER", "T": "T shared gate", "P": "P per-component gates"}
SCN = {"test83_iid_1_L2": "iid_1 scene", "test83_hard_1_L1": "hard_1 scene", "test83_iid_159_L2": "iid_159 scene (held out)"}
REGN = {"objects": "all object pixels", "sphere": "sphere", "cube": "cube", "highlight_core": "highlight core",
        "highlight_ring": "highlight shoulder (ring)", "grazing": "grazing view region", "sphere_lower": "lower sphere"}
MEAS = "MEASURED"


def tag(fig, t=MEAS):
    fig.text(0.995, 0.995, t, ha="right", va="top", fontsize=8, color="white", bbox=dict(boxstyle="round,pad=0.25", fc="#2b6cb0", ec="none"))


def foot(fig, t, y=0.005, w=210):
    fig.text(0.01, y, "\n".join(textwrap.fill(p, w) for p in t.split("\n")), fontsize=7.5, va="bottom", color="#333")


def ssq(M):
    return FuncNorm((lambda x: np.sign(x) * (np.abs(x) / M) ** 0.5, lambda y: np.sign(y) * y ** 2 * M), vmin=-M, vmax=M)


class Pack:
    def __init__(self, root, out):
        self.root, self.out = root, out
        self.D = KA.read(root / "decision.json")
        self.XF = KA.read(root / "ext/fit.json")
        self.XE = KA.read(root / "ext_angular.json")
        self.SA = KA.read(root / "stageA.json")
        self.KF = KA.read(KA.K_ROOT / "fit_primary/fit.json")
        self.LA = {s: KA.read(root / "lanes" / s / "attribution.json") for s in KA.SCENES}
        self.figs = []

    def save(self, fig, name, q, cap, src):
        fig.savefig(self.out / name, dpi=115)
        plt.close(fig)
        self.figs.append({"file": name, "question": q, "caption": cap, "source": src})


def fig_optimization(pk):
    fig, axs = plt.subplots(2, 3, figsize=(17, 8.5))
    for j, arm in enumerate(CM.ARMS):
        H1 = np.array(pk.KF[arm]["history"]); s1 = np.array(pk.KF[arm]["history_steps"])
        H2 = np.array(pk.XF[arm]["history"]); s2 = np.array(pk.XF[arm]["history_steps"]) + s1[-1]
        ax = axs[0, j]
        for r in range(H1.shape[1]):
            for m in range(H1.shape[2]):
                ax.plot(s1, H1[:, r, m], color=plt.cm.tab10(m), lw=0.6)
                ax.plot(s2, H2[:, r, m], color=plt.cm.tab10(m), lw=0.6, ls="-")
        ax.axvline(s1[-1], color="k", ls="--", lw=0.8)
        ax.set_yscale("log"); ax.set_xlabel("optimizer step (dashed: 12k end; then constant lr 3e-4, fresh Adam moments)")
        ax.set_title(f"{SHORT[arm]}: rule (a) after extension {100 * pk.XF[arm]['rule_a_fraction']:.0f} % of units\n"
                     f"({'PASS' if pk.XF[arm]['rule_a_pass'] else 'FAIL'}; WL104K at 12k: "
                     f"{100 * np.mean(np.array(pk.KF[arm]['last_2000_relative_change']) <= 0.005):.0f} %)", fontsize=9)
        if j == 0:
            ax.set_ylabel("training objective (WL82 el1 + ealb)")
        ax = axs[1, j]
        st = pk.XF[arm]["stationarity"]
        vals = [st["start"]["projected_grad_norm"], st["12k"]["projected_grad_norm"], st["extended"]["projected_grad_norm"]]
        ax.bar(range(3), vals, color=COL[arm])
        ax.set_xticks(range(3), ["common start", "12k state", "extended state"])
        ax.set_yscale("log"); ax.set_ylabel("KKT-projected gradient norm (full batch)")
        txt = ""
        if arm != "A":
            txt = f"gate knots at 0 / 1: 12k {100 * st['12k']['gate_knots_at_0']:.0f} / {100 * st['12k']['gate_knots_at_1']:.0f} %, " \
                  f"extended {100 * st['extended']['gate_knots_at_0']:.0f} / {100 * st['extended']['gate_knots_at_1']:.0f} %"
        ax.set_title(f"{SHORT[arm]} stationarity\n{txt}", fontsize=9)
    fig.suptitle("Optimization closure: 12k histories plus the matched 12k-step extension, convergence rule (a) and stationarity",
                 x=0.01, ha="left", weight="bold")
    tag(fig)
    foot(fig, "One line per material and start (4 starts × 9 materials). Rule (a) = share of units whose training objective changes by ≤ 0.5 % over the last "
              "2,000 steps (≥ 90 % required, unchanged from WL104K). The 12k Adam state was not archived, so the extension restarts the moments.")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.12, hspace=0.45, wspace=0.25)
    pk.save(fig, "fig01_optimization_closure_curves_and_stationarity.png", "Does a matched extension close the optimization of A / T / P?",
            "Training curves, rule (a) and projected-gradient stationarity.", "ext/fit.json, WL104K fit_primary/fit.json")


def fig_per_start(pk):
    D = pk.D
    fig, axs = plt.subplots(1, 4, figsize=(19, 5.2))
    ax = axs[0]
    for j, arm in enumerate(CM.ARMS):
        for tg, mk in (("12k", "o"), ("extended", "s")):
            v = pk.XE[tg]["per_start_el1_mean"][arm]
            ax.plot(np.arange(len(v)) + (j - 1) * 0.2, v, mk, color=COL[arm], mfc="none" if tg == "12k" else COL[arm],
                    label=f"{SHORT[arm]}, {tg}")
    ax.set_xticks(range(4), ["start 1\n(generic)", "start 2", "start 3", "start 4"])
    ax.set_ylabel("held-out angular el1 (mean over materials, probes)"); ax.legend(fontsize=7)
    ax.set_title("angular quality per start (open = 12k, filled = extended)", fontsize=9.5)
    for k, s in enumerate(KA.SCENES):
        ax = axs[k + 1]
        for e, (ev, evn) in enumerate((("e1", "first bounce"), ("e2", "second bounce"), ("total", "total"))):
            for tg, off, c in (("12k", -0.12, "#888"), ("extended", 0.12, "#2ca02c")):
                ps = D["scene_utility"][tg][s][ev]["per_start"]
                for r, q in enumerate(ps):
                    x = e + off + (r - 1.5) * 0.04
                    ax.errorbar(x, 100 * q["point"], yerr=[[100 * (q["point"] - q["lo"])], [100 * (q["hi"] - q["point"])]], fmt="o", ms=3,
                                color=c, capsize=2, label=(f"{tg} states" if (e, r) == (0, 0) else None))
            ax.text(e, ax.get_ylim()[1] if False else 0, "", fontsize=7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(3), ["first bounce", "second bounce", "total"])
        labs = ", ".join(f"{ev}: {D['scene_utility']['extended'][s][ev]['label']}" for ev in ("e1", "total"))
        ax.set_title(f"{SCN[s]}: P − A objects stable RMS per start\n(extended) {labs}", fontsize=8.5)
        ax.set_ylabel("relative change (%), jackknife 95 % CI")
        ax.legend(fontsize=7)
    fig.suptitle("Per-start angular and scene quality: is the advantage stable across fitting starts?", x=0.01, ha="left", weight="bold")
    tag(fig)
    foot(fig, "Scene contrasts use the fixed common-proposal replay (same sampled directions for every arm); 4 dots per group = 4 fitting starts. "
              "Total = first + second bounce + the archived R0 higher-order term (identical for all arms).")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.17, wspace=0.3)
    pk.save(fig, "fig02_per_start_angular_and_scene_quality.png", "Is the angular and scene advantage stable across starts?",
            "Angular el1 per start; scene P − A per start with jackknife intervals.", "ext_angular.json, decision.json")


def fig_scene_maps(pk):
    for s in KA.SCENES:
        with np.load(pk.root / "lanes" / s / "maps.npz") as z:
            m = {k: z[k] for k in z.files}
        obj = m["mask_objects"]
        ref = m["ref"]
        M = float(np.abs(np.stack([m["12k_A_mean"] - ref, m["12k_P_mean"] - ref, m["rendered_A_mean"] - m["rendered_reference_er"],
                                    m["rendered_P_mean"] - m["rendered_reference_er"]]))[:, obj].max())
        fig, axs = plt.subplots(2, 4, figsize=(17, 8.6), layout="constrained")
        panels = [("ref", "physical response at the scene's\nfirst-bounce directions (fresh walks)", "gray"),
                  ("12k_A_mean", "A − physical, same directions", "div"), ("12k_P_mean", "P − physical, same directions", "div"),
                  ("ext_P_mean", "P (extended) − physical", "div")]
        for j, (k, t, cm) in enumerate(panels):
            ax = axs[0, j]
            if cm == "gray":
                ax.imshow(np.where(obj, ref, np.nan) / np.nanpercentile(np.where(obj, ref, np.nan), 99.5), cmap="gray", vmin=0, vmax=1)
            else:
                im = ax.imshow(np.where(obj, m[k] - ref, np.nan), cmap="RdBu_r", norm=ssq(M))
            ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        rp = [("rendered_reference_er", "rendered layered reference\n(first bounce, ER part)", "gray"),
              ("rendered_A_mean", "rendered A − rendered reference", "div"), ("rendered_P_mean", "rendered P − rendered reference", "div"),
              ("ref_noise", "physical-response walk noise\n(|half a − half b| / 2)", "div")]
        for j, (k, t, cm) in enumerate(rp):
            ax = axs[1, j]
            if cm == "gray":
                v = np.where(obj, m[k], np.nan)
                ax.imshow(v / np.nanpercentile(v, 99.5), cmap="gray", vmin=0, vmax=1)
            elif k == "ref_noise":
                ax.imshow(np.where(obj, m[k], np.nan), cmap="RdBu_r", norm=ssq(M))
            else:
                ax.imshow(np.where(obj, m[k] - m["rendered_reference_er"], np.nan), cmap="RdBu_r", norm=ssq(M))
            ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axs, shrink=0.5, label=f"signed first-bounce luminance difference (shared scale ±{M:.3g}, square-root display)")
        fig.suptitle(f"{SCN[s]}: physical angular response vs model response at the exact directions used by the image (top), "
                     f"and the rendered images (bottom); red = model brighter", fontsize=11, x=0.01, ha="left")
        tag(fig)
        pk.save(fig, f"fig03_scene_conditioned_response_vs_rendered_error_{SCN[s].split()[0]}.png",
                f"Does the image error of {SCN[s]} come from the angular response at the directions the image uses?",
                "Top: lane-level response residual at the scene's own first-bounce directions (start-mean); bottom: rendered residual.",
                "lanes/<scene>/maps.npz")


def fig_regions(pk):
    regs = ("objects", "highlight_core", "highlight_ring", "grazing", "sphere_lower", "cube")
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.4), sharey=False)
    for ax, s in zip(axs, KA.SCENES):
        A = pk.LA[s]["attribution"]
        x = 0
        ticks = []
        for reg in regs:
            if reg not in A:
                continue
            for j, arm in enumerate(CM.ARMS):
                for tg, mk in (("12k", "o"), ("ext", "s")):
                    for r in range(4):
                        e = A[reg][f"{tg}|{arm}|{r}"]
                        xx = x + (j - 1) * 0.25 + (0.06 if tg == "ext" else -0.06)
                        ax.errorbar(xx, 100 * e["signed_excess"], yerr=196 * e["se"], fmt=mk, ms=3, color=COL[arm],
                                    mfc="none" if tg == "12k" else COL[arm], capsize=1.5,
                                    label=f"{SHORT[arm]} ({tg})" if (x, r) == (0, 0) else None)
            ticks.append(REGN[reg])
            x += 1
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(ticks)), ticks, rotation=25, fontsize=8)
        ax.set_ylabel("signed excess of model over physical (%)\nΣ w (F_model − F_phys) / Σ w F_phys")
        lab = pk.LA[s]["labels"].get("highlight_core") or {}
        ax.set_title(f"{SCN[s]}\nhighlight-core label: {lab.get('label', 'n/a')}", fontsize=9.5)
    axs[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Region attribution at the scene's own first-bounce directions: where is each model too bright or too dark?", x=0.01, ha="left", weight="bold")
    tag(fig)
    foot(fig, "w = light weight (rgb / d²) of each lane; F_phys from fresh layered random walks (8,192 per lane, two halves); bars = 95 % walk-noise "
              "intervals; 4 markers per arm = 4 fitting starts (open 12k, filled extended).")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.83, bottom=0.25, wspace=0.3)
    pk.save(fig, "fig04_region_attribution_core_shoulder_grazing.png", "Is the hard_1 highlight overshoot an angular-response error, noise or start-dependent?",
            "Signed excess per region, arm, start.", "lanes/<scene>/attribution.json")


def fig_mirror_bins(pk):
    fig, axs = plt.subplots(2, 3, figsize=(19, 8.6))
    for k, s in enumerate(KA.SCENES):
        Bn = pk.LA[s]["binned"]
        edges = np.array(pk.LA[s]["mirror_bins_deg"])
        lab = [f"{edges[i]:g}–{edges[i + 1]:g}°" for i in range(len(edges) - 1)]
        for row, reg in enumerate(("highlight_core", "objects")):
            ax = axs[row, k]
            if reg not in Bn:
                ax.axis("off"); continue
            b = Bn[reg]
            x = np.arange(len(lab))
            ax.bar(x, 100 * np.array(b["ref_share"]), 0.8, color="#ddd", label="share of physical contribution")
            for j, arm in enumerate(CM.ARMS):
                v = 100 * np.mean(np.array(b[f"12k|{arm}"]), 0)
                ax.plot(x, v, "o-", color=COL[arm], ms=3, label=f"{SHORT[arm]} signed excess (12k, start-mean)")
                ve = 100 * np.mean(np.array(b[f"ext|{arm}"]), 0)
                ax.plot(x, ve, "x--", color=COL[arm], ms=4, lw=0.8)
            ax.axhline(0, color="k", lw=0.7)
            ax.set_xticks(x, lab, rotation=45, fontsize=7)
            ax.set_title(f"{SCN[s]} — {REGN[reg]}\nview polar 5/50/95 %: {', '.join(f'{q:.0f}' for q in b['view_polar_q05_50_95'])}°; "
                         f"within ±2.5° of a probe polar: {100 * b['share_within_2.5deg_of_a_probe_polar']:.0f} %", fontsize=8.5)
            ax.set_ylabel("% of the region's physical contribution")
        axs[1, k].set_xlabel("angle between the light direction and the mirror of the view direction")
    axs[0, 0].legend(fontsize=6.5)
    fig.suptitle("Which angular directions carry the image error? Signed model excess binned by distance from the mirror direction",
                 x=0.01, ha="left", weight="bold")
    tag(fig)
    foot(fig, "Grey bars: how much of the region's physical light arrives in each mirror-offset bin. Lines: model − physical contribution in that bin, "
              "as a share of the region total (solid 12k, dashed × extended). Positive = model too bright there.")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.12, hspace=0.6, wspace=0.25)
    pk.save(fig, "fig05_error_by_mirror_offset_at_scene_directions.png", "Is the overshoot concentrated at specific near-mirror directions?",
            "Binned signed excess vs mirror offset at the scene's own directions.", "lanes/<scene>/attribution.json")


def fig_events(pk):
    SA = pk.SA["scenes"]
    D = pk.D
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, s in zip(axs, KA.SCENES):
        x = np.arange(3)
        for j, arm in enumerate(CM.ARMS):
            v = [100 * np.mean(SA[s]["per_start_stable_rms"][arm][ev]) for ev in ("e1", "e2", "total")]
            ax.bar(x + (j - 1) * 0.27, v, 0.26, color=COL[arm], label=SHORT[arm])
        a0 = [100 * SA[s]["A0_stable_rms"][ev] for ev in ("e1", "e2", "total")]
        ax.plot(x, a0, "k_", ms=25, mew=2, label="historical R0")
        ax.set_xticks(x, ["first bounce", "second bounce", "total"])
        ax.set_ylabel("objects stable RMS (12k states, start-mean)")
        ax.set_title(SCN[s], fontsize=10)
    axs[0].legend(fontsize=7.5)
    fig.suptitle("E1 / E2 / TOTAL decomposition (12k states; total = E1 + fixed E2 + the common archived R0 higher-order term)", x=0.01, ha="left", weight="bold")
    tag(fig)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.12, wspace=0.3)
    pk.save(fig, "fig06_first_second_total_decomposition.png", "Which event carries each scene's change?", "Stable RMS per event.", "stageA.json")


def fig_noise(pk):
    SA = pk.SA
    fig, ax = plt.subplots(figsize=(13, 4.8))
    rows = []
    for c in ("development", "held_out"):
        u = SA["uncertainty"][c]
        rows.append((f"angular reference half-split noise\n({c}, % of A el1)", 100 * u["reference_half_split_el1_ratio_of_A"]))
        rows.append((f"angular P − A across-start range\n({c}, pp)", u["P-A_across_start_range_pp"]))
        rows.append((f"angular P − A material-bootstrap CI width\n({c}, pp)", u["P-A_material_bootstrap_ci_width_pp"]))
    for s in KA.SCENES:
        sc = SA["scenes"][s]
        rows.append((f"{SCN[s]}: render-replicate SE of A total\n(% of A stable RMS)",
                     100 * np.mean(sc["per_start_jackknife_se"]["A"]["total"]) / np.mean(sc["per_start_stable_rms"]["A"]["total"])))
        pr = np.array(sc["per_start_stable_rms"]["P"]["total"]); ar = np.array(sc["per_start_stable_rms"]["A"]["total"])
        rows.append((f"{SCN[s]}: across-start range of P − A total\n(pp)", 100 * float(np.ptp(pr / ar - 1))))
        hc = pk.LA[s]["attribution"].get("highlight_core", pk.LA[s]["attribution"]["objects"])["_reference"]
        rows.append((f"{SCN[s]}: lane walk-noise SE of the physical\nhighlight-core sum (%)", 100 * hc["se_rel"]))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color="#6c8ebf")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7)
    ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlabel("magnitude (units in each label)")
    fig.suptitle("Noise and variability accounting: reference noise, render replicates and fitting starts", x=0.01, ha="left", weight="bold")
    tag(fig)
    fig.subplots_adjust(left=0.3, right=0.99, top=0.9, bottom=0.12)
    pk.save(fig, "fig07_noise_and_variability_accounting.png", "Can noise or start variability explain the observed differences?",
            "Separated uncertainty sources.", "stageA.json, lanes/*/attribution.json")


def fig_scatter(pk):
    s = "test83_hard_1_L1"
    with np.load(pk.root / "lanes" / s / "lanes.npz") as z:
        Lz = {k: z[k] for k in z.files}
    with np.load(pk.root / "lanes" / s / "maps.npz") as z:
        core = z["mask_highlight_core"].reshape(-1)
    sel = core[Lz["pix"]]
    lf = Lz["lf"].astype(np.float64)
    cref = (0.5 * (Lz["ref_a"] + Lz["ref_b"]) * lf) @ CM.LUMA
    mr = CM.mirror_radius(Lz["wi"].astype(np.float64), Lz["wo"].astype(np.float64))
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2))
    edges = np.r_[0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8]
    ctr = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(mr[sel], edges) - 1
    ok = (idx >= 0) & (idx < len(ctr))
    for ax, arm in zip(axs, CM.ARMS):
        cm_ = np.mean([Lz[f"12k_{arm}_{r}"] for r in range(4)], 0)
        ax.scatter(cref[sel], cm_[sel], s=1, c=mr[sel], cmap="viridis", vmin=0, vmax=8, alpha=0.4)
        lim = max(cref[sel].max(), cm_[sel].max())
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        ax.set_xlabel("physical contribution per lane (layered walks)"); ax.set_ylabel(f"{SHORT[arm]} contribution, same lane (start-mean)")
        ins = ax.inset_axes([0.58, 0.08, 0.38, 0.35])
        num = np.bincount(idx[ok], (cm_[sel] - cref[sel])[ok], len(ctr)); den = np.bincount(idx[ok], cref[sel][ok], len(ctr))
        ins.plot(ctr, 100 * num / np.maximum(den, 1e-30), "o-", color=COL[arm], ms=3); ins.axhline(0, color="k", lw=0.6)
        ins.set_xlabel("mirror offset (deg)", fontsize=7); ins.set_ylabel("excess (%)", fontsize=7); ins.tick_params(labelsize=6)
        ax.set_title(f"{SHORT[arm]} vs physical, hard_1 highlight-core lanes", fontsize=9.5)
    fig.suptitle("Matched-direction comparison in the hard_1 highlight core: each dot is one camera sample (colour = mirror offset, 0–8°)",
                 x=0.01, ha="left", weight="bold")
    tag(fig)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.13, wspace=0.28)
    pk.save(fig, "fig08_hard1_highlight_matched_direction_scatter.png", "At the exact directions of the hard_1 highlight, is P too bright near the mirror peak?",
            "Per-lane model vs physical contribution; inset: signed excess by mirror offset.", "lanes/test83_hard_1_L1/lanes.npz")


def index(pk):
    D = pk.D
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td><b>{html.escape(json.dumps(v) if not isinstance(v, str) else v)}</b></td></tr>"
                   for k, v in D.items() if k in ("optimization_qualification", "rule_a_fraction", "angular_stability",
                                                 "wl104k_gain_criteria_on_extended_states_descriptive", "extended_state_contracts_passed",
                                                 "lane_correspondence", "attribution_labels", "integration_work_justified", "adoption_readiness"))
    su = "".join(f"<tr><td>{html.escape(SCN[s])}</td>" + "".join(f"<td>{D['scene_utility'][tg][s][ev]['label']}</td>" for tg in ("12k", "extended")
                                                               for ev in ("e1", "e2", "total")) + "</tr>" for s in KA.SCENES)
    figs = "".join(f"<section><h3>{html.escape(f['file'])} <span class='tag'>MEASURED</span></h3><p><b>Question:</b> {html.escape(f['question'])}</p>"
                   f"<p>{html.escape(f['caption'])} <i>Source: {html.escape(f['source'])}</i></p><img src='{html.escape(f['file'])}' loading='lazy'></section>"
                   for f in pk.figs)
    page = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>WL104K-A closure</title><style>:root{{--bg:#fff;--fg:#1a1a1a;--mut:#555;--bd:#ddd}}
@media (prefers-color-scheme: dark){{:root:not([data-theme=light]){{--bg:#16181d;--fg:#e8e8e8;--mut:#aaa;--bd:#333}}}}
body{{background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif;max-width:1400px;margin:0 auto;padding:16px}}
img{{max-width:100%;background:#fff;border:1px solid var(--bd)}} .tag{{font-size:12px;padding:2px 6px;border-radius:4px;color:#fff;background:#2b6cb0}}
td,th{{border:1px solid var(--bd);padding:3px 6px}} table{{border-collapse:collapse;font-size:13px}} p{{color:var(--mut)}}</style></head><body>
<h1>WL104K-A — optimization closure and scene-conditioned fidelity (ER-v1-SDM, candidate)</h1>
<p>Diagnostic follow-up of WL104K. WL104K's frozen verdict and appended post hoc section are unchanged. A / T / P = shared ER-v1-GF / one shared
symmetric gate / per-component symmetric gates. Physical reference: WL104D layered angular reference and fresh layered walks at the scenes' own
first-bounce directions; scene images from the fixed common-proposal replay.</p>
<h2>Frozen-rule outcomes</h2><table>{rows}</table>
<h2>Scene utility across starts (P − A objects stable RMS; STABLE_* = all 4 starts agree with 95 % intervals)</h2>
<table><tr><th>scene</th><th>12k E1</th><th>12k E2</th><th>12k total</th><th>ext E1</th><th>ext E2</th><th>ext total</th></tr>{su}</table>
<h2>Figures</h2>{figs}</body></html>"""
    (pk.out / "index.html").write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pk = Pack(args.root, args.out)
    fig_optimization(pk); fig_per_start(pk); fig_scene_maps(pk); fig_regions(pk); fig_mirror_bins(pk); fig_events(pk); fig_noise(pk); fig_scatter(pk)
    index(pk)
    man = {"figures": pk.figs, "decision_sha256": CM.sha(args.root / "decision.json"), "script_sha256": CM.sha(Path(__file__)),
           "files_sha256": {f.name: CM.sha(f) for f in sorted(args.out.glob("*")) if f.suffix in (".png", ".html")}}
    KA.write(args.out / "figures_manifest.json", man)
    print("figures", len(pk.figs), flush=True)


if __name__ == "__main__":
    main()
