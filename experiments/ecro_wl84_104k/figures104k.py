"""WL104K reviewer package (evaluation/104K): readable figures, comparison table, index.html, manifest.

Usage: python figures104k.py <run_root> <out_dir>
Reads only <run_root>/{eval/eval.json, eval/lanes.npz, decision.json, fit_*/fit.json, scenes/*/analysis.json, maps.npz}.
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
import common104k as CM  # noqa: E402

LUMA = CM.LUMA
COL = {"reference": "#111111", "A": "#1f77b4", "T": "#e08214", "P": "#2ca02c", "R0": "#7f7f7f"}
NAME = {"A": "A  shared ER-v1-GF (no gates)", "T": "T  one shared symmetric gate", "P": "P  per-component symmetric gates (ER-v1-SDM)"}
SHORT = {"A": "A shared ER", "T": "T shared gate", "P": "P per-component gates"}
COMPN = {"O0": "Outer 1", "O1": "Outer 2", "O2": "Outer 3", "L": "Lambert", "I0": "Inner 1", "I1": "Inner 2", "I2": "Inner 3", "I3": "Inner 4"}
MEAS, CONC = "MEASURED", "CONCEPTUAL (illustration, no data)"
MAP_PROBES, MAP_POLAR = [2, 6, 10, 18], [20, 50, 60, 86]
SCENES = ("test83_iid_1_L2", "test83_hard_1_L1", "test83_iid_159_L2")


def short(t):
    return t.replace("test83_", "")


def ssqrt(M):
    return FuncNorm((lambda x: np.sign(x) * np.sqrt(np.abs(x) / M), lambda y: np.sign(y) * y ** 2 * M), vmin=-M, vmax=M)


def tag(fig, t):
    fig.text(0.995, 0.995, t, ha="right", va="top", fontsize=8, color="white",
             bbox=dict(boxstyle="round,pad=0.25", fc="#2b6cb0" if t.startswith("MEAS") else "#b7791f", ec="none"))


def foot(fig, t, y=0.005, w=200):
    fig.text(0.01, y, "\n".join(textwrap.fill(p, w) for p in t.split("\n")), fontsize=7.5, va="bottom", color="#333")


def mirror_frame(wi):
    m = np.array([-wi[0], -wi[1], wi[2]])
    e1 = np.cross([0, 0, 1.0], m)
    e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-9 else np.array([1.0, 0, 0])
    return m, e1, np.cross(m, e1)


def bin_map(wi, wo, v, R=45.0, nb=41):
    m, e1, e2 = mirror_frame(wi)
    r = np.degrees(np.arccos(np.clip(wo @ m, -1, 1)))
    az = np.arctan2(wo @ e2, wo @ e1)
    x, y = r * np.cos(az), r * np.sin(az)
    edges = np.linspace(-R, R, nb + 1)
    s, _, _ = np.histogram2d(y, x, [edges, edges], weights=v)
    c, _, _ = np.histogram2d(y, x, [edges, edges])
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def ring_profile(wi, wo, w, F, edges):
    r = CM.mirror_radius(wi, wo)
    lum = F @ LUMA
    idx = np.digitize(r, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    num = np.bincount(idx[ok], (w * lum)[ok], len(edges) - 1)
    den = np.bincount(idx[ok], w[ok], len(edges) - 1)
    return np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)


class Pack:
    def __init__(self, root, out):
        self.root, self.out = root, out
        self.E = CM.read(root / "eval/eval.json")
        self.D = CM.read(root / "decision.json")
        if CM.sha(root / "eval/lanes.npz") != self.E["lanes_sha256"]:
            raise CM.Stop("LANES_HASH")
        with np.load(root / "eval/lanes.npz") as z:
            self.L = {k: z[k].astype(np.float64) for k in z.files}
        self.rows = self.E["materials"]
        self.n = len(self.rows)
        self.figs, self.checks = [], []

    def check(self, name, a, b, tol=1e-9):
        d = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
        self.checks.append({"check": name, "max_abs_diff": d, "pass": bool(d <= tol)})

    def save(self, fig, name, question, kind, caption, source):
        fig.savefig(self.out / name, dpi=120)
        plt.close(fig)
        self.figs.append({"file": name, "question": question, "kind": kind, "caption": caption, "source": source})


def fig_concept(pk):
    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.25, 1], left=0.04, right=0.99, top=0.82, bottom=0.2, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_title("Original certified ER (A)", loc="left", fontsize=10, weight="bold")
    txt = ("f(wi,wo) = Outer1 + Outer2 + Outer3\n        + s(μi) s(μo) · [ Lambert + Inner1..4 ]\n\n"
           "• 7 GGX lobes (3 outer, 4 refracted inner) + Lambert\n• one shared attenuation s(μ) on Lambert + inner only\n"
           "• every term is symmetric in (wi, wo) → reciprocal\n• each term ≥ 0, total albedo ≤ Σλ ≤ 1 → passive")
    ax.text(0, 0.9, txt, va="top", family="monospace", fontsize=9.5)
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.set_title("Candidate ER-v1-SDM (P) and shared-gate control (T)", loc="left", fontsize=10, weight="bold")
    txt = ("f(wi,wo) = Σ_j  q_j(μi) · q_j(μo) · term_j(wi,wo)\n\n"
           "• the same 8 terms, GGX shapes unchanged\n• q_j: 5-knot piecewise-linear curve in μ = cos θ,\n  0 ≤ q_j ≤ 1, per RGB channel\n"
           "• same q_j for the incoming and the outgoing side\n  → product symmetric → still reciprocal\n"
           "• q ≤ 1 can only remove energy → f_SDM ≤ f_base → passive\n• q_j = 1 everywhere reproduces A exactly\n"
           "• T: one q shared by all 8 terms (15 numbers)\n• P: one q per term (120 numbers)")
    ax.text(0, 0.9, txt, va="top", family="monospace", fontsize=9.5)
    ax = fig.add_subplot(gs[0, 2])
    mu = np.linspace(0, 1, 200)
    for lab, kn, c in (("identity (= A)", [1, 1, 1, 1, 1], "k"), ("example: dim near grazing", [0.4, 0.6, 0.85, 1, 1], "#2ca02c"),
                       ("example: dim near normal", [1, 1, 0.85, 0.6, 0.45], "#d62728")):
        ax.plot(mu, np.interp(mu, CM.KNOTS, kn), color=c, label=lab)
        ax.plot(CM.KNOTS, kn, "o", color=c, ms=4)
    ax.set_xlabel("μ = cos θ  (0 = grazing, 1 = normal)")
    ax.set_ylabel("gate value q(μ)")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("What a gate curve looks like (illustration)", fontsize=9.5)
    fig.suptitle("Original ER versus reciprocal per-component directional modulation", x=0.01, ha="left", weight="bold")
    tag(fig, CONC)
    foot(fig, "Physical meaning: a gate attenuates one scattering term more for some incidence/exit polar angles than for others. "
              "Because the same curve multiplies both sides, the response stays reciprocal; because it never exceeds 1, energy can only be removed. "
              "Example curves on the right are illustrations, not fitted results (fitted curves: figure 5).")
    pk.save(fig, "fig01_original_er_vs_reciprocal_per_component_gates.png", "What does the candidate change, and why is it reciprocal and passive?",
            CONC, "Schematic of A, T and P; example gate curves are illustrative.", "protocol.json (no data)")


def fig_profiles(pk):
    L = pk.L
    edges = np.r_[0, 1, 2, 3, 4.5, 6, 8, 10, 12, 15, 19, 24, 30, 37, 45, 55, 70, 90]
    ctr = np.sqrt(np.maximum(edges[:-1], 0.5) * edges[1:])
    for m, r in enumerate(pk.rows):
        fig, axs = plt.subplots(2, 4, figsize=(17, 8.2), gridspec_kw=dict(height_ratios=[2.2, 1]), sharex="col")
        for j, p in enumerate(MAP_PROBES):
            ax, ar = axs[0, j], axs[1, j]
            wi, wo, w = L["wi"][m, p], L["wo"][m, p], L["w"][m, p]
            ref = ring_profile(wi, wo, w, L["ref_mean"][m, p], edges)
            for a in (ax, ar):
                a.axvspan(0.5, 3, color="#f2e6ff", zorder=0); a.axvspan(3, 12, color="#e6f2ff", zorder=0)
            ax.plot(ctr, ref, "o-", color=COL["reference"], lw=2.2, ms=4, label="layered reference (independent)")
            for arm in CM.ARMS:
                v = ring_profile(wi, wo, w, L[f"pred_{arm}"][m, p], edges)
                ax.plot(ctr, v, "-", color=COL[arm], lw=1.5, label=SHORT[arm])
                ar.plot(ctr, (v - ref) / np.maximum(ref, 1e-12) * 100, "-", color=COL[arm], lw=1.5)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"incident polar {MAP_POLAR[j]}°, azimuth 22.5°", fontsize=9.5)
            ar.axhline(0, color="k", lw=0.8); ar.set_ylim(-60, 60)
            ar.set_xlabel("angle from the mirror direction (deg, log)")
            if j == 0:
                ax.set_ylabel("ring-mean response f·cos θo\n(luminance, sr⁻¹)")
                ar.set_ylabel("model − reference\n(% of reference)")
                ax.text(1.2, ax.get_ylim()[0] * 1.5, "core", fontsize=8, color="#7a3fbf")
                ax.text(5, ax.get_ylim()[0] * 1.5, "shoulder", fontsize=8, color="#2b6cb0")
                ax.text(25, ax.get_ylim()[0] * 1.5, "tail", fontsize=8, color="#555")
            if j == 3:
                ax.legend(fontsize=8, loc="lower left")
        coh = "development" if r["cohort"] == "development" else "held out"
        fig.suptitle(f"Material {short(r['tag'])} ({coh}): layered reference vs A / T / P at four incident angles", x=0.01, ha="left", weight="bold")
        tag(fig, MEAS)
        foot(fig, "Ring-mean of the quadrature-weighted response at each angle from the mirror direction, on the independent WL104D evaluation lanes "
                  "(never fitted). Shaded: mirror core (< 3°, purple) and shoulder (3–12°, blue); beyond 12° = tail. Best training-objective start per arm.")
        fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.12, hspace=0.08, wspace=0.25)
        pk.save(fig, f"fig02_angular_profiles_{short(r['tag'])}_reference_vs_A_T_P.png",
                f"Does P (or T) follow the reference better than A for {short(r['tag'])}, in core, shoulder or tail?", MEAS,
                "Reference in black; lower row relative difference (clipped ±60 %).", "eval/lanes.npz")


def fig_maps(pk):
    L = pk.L
    maps, M = {}, 0.0
    for m in range(pk.n):
        for j, p in enumerate(MAP_PROBES):
            wi, wo, w = L["wi"][m, p, 0], L["wo"][m, p], L["w"][m, p]
            A = (w[:, None] * L["ref_mean"][m, p]).sum(0) @ LUMA
            res = {a: (L["ref_mean"][m, p] - L[f"pred_{a}"][m, p]) @ LUMA / A for a in CM.ARMS}
            mm = {f"res_{a}": bin_map(wi, wo, v) for a, v in res.items()}
            mm["red_PA"] = bin_map(wi, wo, np.abs(res["A"]) - np.abs(res["P"]))
            mm["red_PT"] = bin_map(wi, wo, np.abs(res["T"]) - np.abs(res["P"]))
            maps[(m, j)] = mm
            M = max(M, max(np.nanmax(np.abs(v)) for v in mm.values()))
    rowsn = [("res_A", "A shared:\nreference − model"), ("res_T", "T shared gate:\nreference − model"),
             ("res_P", "P per-component:\nreference − model"), ("red_PA", "error reduction\nP vs A: |res A| − |res P|"),
             ("red_PT", "error reduction\nP vs T: |res T| − |res P|")]
    for m, r in enumerate(pk.rows):
        fig, axs = plt.subplots(5, 4, figsize=(13, 16), layout="constrained")
        for j in range(4):
            for i, (k, lab) in enumerate(rowsn):
                ax = axs[i, j]
                ax.imshow(maps[(m, j)][k], origin="lower", extent=[-45, 45, -45, 45], cmap="RdBu_r" if k.startswith("res") else "PiYG", norm=ssqrt(M))
                ax.set_xticks([-30, 0, 30]); ax.set_yticks([-30, 0, 30]); ax.tick_params(labelsize=7)
                if i == 0:
                    ax.set_title(f"incident polar {MAP_POLAR[j]}°", fontsize=10)
                if j == 0:
                    ax.set_ylabel(lab, fontsize=8.5)
        c0 = fig.colorbar(axs[0, 0].images[0], ax=axs[:3, :], shrink=0.6)
        c0.set_label("signed residual per sr / reference directional albedo (luminance)")
        c1 = fig.colorbar(axs[3, 0].images[0], ax=axs[3:, :], shrink=0.6)
        c1.set_label("reduction of |residual| (green = P closer to the reference)")
        coh = "development" if r["cohort"] == "development" else "held out"
        fig.suptitle(f"Signed residuals and actual error reduction — material {short(r['tag'])} ({coh})\naxes: outgoing direction relative to the "
                     f"mirror direction (deg); one absolute colour scale (±{M:.3g}) for every WL104K map, square-root display", fontsize=10.5, x=0.01, ha="left")
        tag(fig, MEAS)
        pk.save(fig, f"fig03_residual_and_error_reduction_maps_{short(r['tag'])}.png",
                f"Where in the outgoing hemisphere does P change the error of {short(r['tag'])}?", MEAS,
                "Rows 1–3 signed residual (red = model too dark); rows 4–5 change of absolute error (green = improvement).", "eval/lanes.npz")
    return M


def fig_tables(pk):
    T = pk.E["tables"]
    ang = CM.ANGLES
    fig, axs = plt.subplots(3, pk.n, figsize=(21, 9), sharex=True)
    for m, r in enumerate(pk.rows):
        for arm in CM.ARMS:
            v = np.array(T[arm]["el1"])[m].reshape(-1, 2).mean(1)
            axs[0, m].plot(ang, v, "o-", ms=3, color=COL[arm], label=SHORT[arm])
        a = np.array(T["A"]["el1"])[m].reshape(-1, 2).mean(1)
        for arm in ("T", "P"):
            axs[1, m].plot(ang, 100 * (np.array(T[arm]["el1"])[m].reshape(-1, 2).mean(1) / a - 1), "o-", ms=3, color=COL[arm])
        for k, ls in (("core_l1_of_total", ":"), ("shoulder_l1_of_total", "--"), ("tail_l1_of_total", "-")):
            for arm in ("A", "P"):
                axs[2, m].plot(ang, np.array(T[arm][k])[m].reshape(-1, 2).mean(1), ls, color=COL[arm], lw=1.2,
                               label=f"{SHORT[arm]}, {k.split('_')[0]}" if m == 0 else None)
        axs[1, m].axhline(0, color="k", lw=0.7); axs[1, m].axhline(-10, color="#2ca02c", lw=0.7, ls="--")
        axs[0, m].set_title(f"{short(r['tag'])}\n({'dev' if r['cohort'] == 'development' else 'held out'})", fontsize=9)
        axs[2, m].set_xlabel("incident polar (deg)", fontsize=8)
        for a_ in axs[:, m]:
            a_.tick_params(labelsize=7)
    axs[0, 0].set_ylabel("held-out angular el1", fontsize=8.5)
    axs[1, 0].set_ylabel("el1 change vs A (%)\n(dashed: −10 % practical tier)", fontsize=8.5)
    axs[2, 0].set_ylabel("region L1 / total reference energy\n(dotted core, dashed shoulder, solid tail)", fontsize=8.5)
    axs[0, 0].legend(fontsize=7); axs[2, 0].legend(fontsize=6)
    fig.suptitle("Per-material and per-incidence angular error: A / T / P (median over 4 starts, mean of both azimuths)", x=0.01, ha="left", weight="bold")
    tag(fig, MEAS)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.88, bottom=0.07, hspace=0.18, wspace=0.4)
    pk.save(fig, "fig04_error_by_material_and_incidence_A_T_P.png", "Which materials and incident angles gain or lose, and in which region?", MEAS,
            "Every material and incidence; region rows show A and P only for legibility (T in the table).", "eval/eval.json tables")


def fig_gates(pk):
    G = pk.E["gates"]
    share = np.array(pk.E["component_share"]["P"])                       # [n, P, 8]
    best = pk.E["best_start"]
    mu = np.linspace(0, 1, 101)
    fig, axs = plt.subplots(3, pk.n, figsize=(22, 9.5), sharex=True, sharey=True)
    cols = plt.cm.tab10(np.arange(8))
    for m, r in enumerate(pk.rows):
        gP = np.array(G["P"][best["P"][m]])[m]                               # [8, 3, 5]
        gT = np.array(G["T"][best["T"][m]])[m, 0]                            # [3, 5]
        sh = share[m].mean(0)
        for c, (ax, cn) in enumerate(zip(axs[:, m], ("red", "green", "blue"))):
            for j, comp in enumerate(CM.COMP):
                lw = 0.6 + 3.0 * sh[j]
                ax.plot(mu, np.interp(mu, CM.KNOTS, gP[j, c]), color=cols[j], lw=lw, alpha=0.9 if sh[j] > 0.02 else 0.35,
                        label=f"{COMPN[comp]} ({100 * sh[j]:.0f} %)" if c == 0 else None)
            ax.plot(mu, np.interp(mu, CM.KNOTS, gT[c]), "k--", lw=1.5, label="T shared gate" if c == 0 else None)
            if m == 0:
                ax.set_ylabel(f"{cn} channel gate q(μ)", fontsize=8.5)
            ax.set_ylim(-0.03, 1.05); ax.tick_params(labelsize=7)
        axs[0, m].set_title(f"{short(r['tag'])}\n({'dev' if r['cohort'] == 'development' else 'held out'})", fontsize=9)
        axs[0, m].legend(fontsize=5.5, loc="lower right")
        axs[2, m].set_xlabel("μ = cos θ", fontsize=8)
    fig.suptitle("Learned directional gates per component (P, best start; dashed = T's single shared gate). Line width and legend % = the "
                 "component's share of P's reflected energy", x=0.01, ha="left", weight="bold", fontsize=10.5)
    tag(fig, MEAS)
    foot(fig, "Distinct curves show that the fit used the freedom differently per component; they do not establish that the corresponding "
              "scattering paths are physically independent. Gates on components with a small energy share (faint lines) are weakly determined.")
    fig.subplots_adjust(left=0.05, right=0.995, top=0.88, bottom=0.1, hspace=0.12, wspace=0.12)
    pk.save(fig, "fig05_learned_gate_curves_per_component.png", "Do different lobes learn distinct directional modulation?", MEAS,
            "Per RGB channel; best training-objective start per material.", "eval/eval.json gates, component_share")


def fig_cost(pk):
    E = pk.E
    cost = E["cost"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, key, lab in ((axs[0], "added_bytes_float32", "added storage per material (bytes)"),
                         (axs[1], "torch_float32_seconds_per_4M_lanes", "torch float32 eval time, 4.2M lanes (s)"),
                         (axs[2], "drjit_seconds_per_4M_lanes", "Dr.Jit eval time, 4.2M lanes (s)")):
        for c, mk in (("development", "o"), ("held_out", "s")):
            ms = E["cohorts"][c]
            for arm in CM.ARMS:
                el = np.median(np.array(E["tables"][arm]["el1_per_start"])[:, ms].sum((1, 2)))
                ax.plot(cost[arm][key], el / len(ms) / CM.NP, mk, color=COL[arm], ms=9,
                        label=f"{SHORT[arm]}, {'development' if c == 'development' else 'held out'}")
        ax.set_xlabel(lab)
        ax.set_ylabel("pooled held-out el1 (mean per probe)")
    axs[0].legend(fontsize=7)
    fig.suptitle("Angular fidelity versus added parameters, storage and evaluation time", x=0.01, ha="left", weight="bold")
    tag(fig, MEAS)
    foot(fig, f"Trainable parameters per material: A 58, T 73, P 178; runtime state 69 / 84 / 189 floats. "
              f"Timings: one material, 2^22 random lanes, median of 5 calls on the local RTX 5080. The T / P Dr.Jit path is an "
              f"unoptimized experiment-local evaluator (component split + gates), A is the production compiled runtime.", y=0.01)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.25, wspace=0.3)
    pk.save(fig, "fig06_fidelity_vs_parameters_storage_runtime.png", "What does the fidelity change cost?", MEAS,
            "Each point = arm × cohort.", "eval/eval.json cost, tables")


def fig_summary(pk):
    E, D = pk.E, pk.D
    con = E["contrasts"]
    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.99, top=0.9, bottom=0.12, hspace=0.45, wspace=0.3)
    names = [("T-A", "T − A\none shared gate"), ("P-T", "P − T\nper-component\nbeyond shared"), ("P-A", "P − A\ntotal candidate effect")]

    def ci_plot(ax, part, title, ylabel):
        for i, (k, lab) in enumerate(names):
            for j, c in enumerate(("development", "held_out")):
                v = con[k][part][c]
                x = i + (j - 0.5) * 0.3
                ax.errorbar(x, 100 * v["median"], yerr=[[100 * (v["median"] - v["lo"])], [100 * (v["hi"] - v["median"])]], fmt="o",
                            color="#444" if j == 0 else "#c0392b", capsize=4, label=("development" if j == 0 else "held out") if i == 0 else None)
                for s in v["per_start"]:
                    ax.plot(x + 0.06, 100 * s, ".", color="#999", ms=4)
                pk.check(f"{part} {k} {c} median", v["median"], np.median(v["per_start"]), 1e-12)
        ax.axhline(0, color="k", lw=0.8); ax.axhline(-10, color="#2ca02c", ls="--", lw=1)
        ax.set_xticks(range(3), [l for _, l in names], fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9); ax.legend(fontsize=8)
    ci_plot(fig.add_subplot(gs[0, 0]), "eval:el1", "(a) held-out angular el1 change\nmedian of 4 starts, 95 % cluster bootstrap; dashed −10 % tier",
            "relative change (%)")
    ci_plot(fig.add_subplot(gs[0, 1]), "polar_transfer:el1", "(b) nearby held-out incident polars (transfer)", "relative change (%)")
    ci_plot(fig.add_subplot(gs[1, 1]), "azimuth_transfer:el1", "(d) new azimuths (transfer)", "relative change (%)")
    ax = fig.add_subplot(gs[0, 2])
    ctl = [("S0", "null target\n(no gates)"), ("ST", "known shared\ngate"), ("SP", "known per-\ncomponent gates")]
    for i, (k, lab) in enumerate(ctl):
        mm = E["synthetic"][k]["heldout_el1_mean"]
        for j, arm in enumerate(CM.ARMS):
            ax.bar(i + (j - 1) * 0.25, mm[arm], 0.23, color=COL[arm], label=SHORT[arm] if i == 0 else None)
        ps = E["synthetic"][k]["pass"]
        ax.text(i, max(mm.values()) * 1.15, "PASS" if ps else "FAIL", ha="center", fontsize=8, color="#2ca02c" if ps else "#c0392b")
    ax.set_yscale("log"); ax.set_xticks(range(3), [l for _, l in ctl], fontsize=8)
    ax.set_ylabel("held-out el1 vs the known truth"); ax.legend(fontsize=7.5)
    ax.set_title("(c) synthetic controls: can the joint fit use the gates?", fontsize=9)
    ax = fig.add_subplot(gs[1, 0])
    for i, (k, lab) in enumerate(names):
        for j, c in enumerate(("development", "held_out")):
            tr = -100 * con[k]["fit:el1"][c]["median"]; ho = -100 * con[k]["eval:el1"][c]["median"]
            ax.bar(i + (j - 0.5) * 0.36 - 0.08, tr, 0.16, color="#bbb", label="training lanes" if (i, j) == (0, 0) else None)
            ax.bar(i + (j - 0.5) * 0.36 + 0.08, ho, 0.16, color="#444" if j == 0 else "#c0392b",
                   label=f"held-out lanes ({'dev' if j == 0 else 'held out'})" if i == 0 else None)
    ax.set_xticks(range(3), [l for _, l in names], fontsize=8); ax.set_ylabel("el1 reduction (%)"); ax.legend(fontsize=7.5)
    ax.set_title("(e) training vs independent held-out lanes", fontsize=9)
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    lines = [f"{k.replace('_', ' ')}: {v}" for k, v in D["decision_table"].items()] + ["", f"optimization resolved: {D['optimization_resolved']}"]
    ax.text(0, 1, "\n".join(textwrap.fill(x, 60, subsequent_indent="  ") if x else x for x in lines), va="top", fontsize=9, family="monospace")
    ax.set_title("(f) frozen decision table", fontsize=9, loc="left")
    fig.suptitle("Summary: does reciprocal per-component directional modulation recover angular fidelity?", x=0.01, ha="left", weight="bold")
    tag(fig, MEAS)
    foot(fig, "el1 = WL82 energy-weighted relative L1 against the mean of two independent evaluation halves, pooled over (material, probe) per cohort "
              "(development 3 materials, held out 6). P and T are reciprocal and passive (verified); B / C of WL104J are NOT comparators.", y=0.01)
    pk.save(fig, "fig07_outcome_summary.png", "What is the overall outcome under the frozen rules?", MEAS,
            "Contrasts, transfer, synthetic controls, train vs held-out, decision table.", "eval/eval.json, decision.json")


def fig_training(pk):
    fj = CM.read(pk.root / "fit_primary/fit.json")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, arm in zip(axs, CM.ARMS):
        H = np.array(fj[arm]["history"]); st = np.array(fj[arm]["history_steps"])
        for r in range(H.shape[1]):
            for m in range(H.shape[2]):
                ax.plot(st, H[:, r, m], color=plt.cm.tab10(m), lw=0.7, label=short(pk.rows[m]["tag"]) if r == 0 else None)
        ax.set_yscale("log"); ax.set_xlabel("optimizer step")
        ax.set_title(f"{SHORT[arm]} ({fj['trainable_per_material'][arm]} trainable / material), {fj[arm]['seconds']:.0f} s", fontsize=9)
    axs[0].set_ylabel("training objective (WL82 el1 + ealb)"); axs[-1].legend(fontsize=7, ncol=2)
    fig.suptitle("Training histories: all arms start from the same physical response and optimize jointly (4 starts × 9 materials)",
                 x=0.01, ha="left", weight="bold")
    tag(fig, MEAS)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.14, wspace=0.08)
    pk.save(fig, "fig08_training_histories.png", "Did every arm converge, and do starts agree?", MEAS, "One line per material and start.",
            "fit_primary/fit.json")


def fig_scenes(pk):
    an = {s: pk.root / "scenes" / s / "analysis.json" for s in SCENES}
    if not all(p.exists() for p in an.values()):
        return None
    A = {s: CM.read(p) for s, p in an.items()}
    maps = {s: dict(np.load(pk.root / "scenes" / s / "maps.npz")) for s in SCENES}
    M = max(float(np.abs(maps[s][k]).max()) for s in SCENES for k in maps[s] if "err" in k or "reduction" in k)
    fig, axs = plt.subplots(len(SCENES), 5, figsize=(17, 10.5), layout="constrained")
    for i, s in enumerate(SCENES):
        mp = maps[s]
        panels = [("reference_e1", "layered reference, first bounce"), ("A_0_e1_err", "A − reference, first bounce"),
                  ("P_0_e1_err", "P − reference, first bounce"), ("abs_err_reduction_P_vs_A_e1", "|A err| − |P err|, first bounce"),
                  ("abs_err_reduction_P_vs_A_e2", "|A err| − |P err|, second bounce")]
        for j, (k, lab) in enumerate(panels):
            ax = axs[i, j]
            v = mp[k] @ LUMA
            if k.startswith("reference"):
                ax.imshow(v / np.percentile(v, 99.5), cmap="gray", vmin=0, vmax=1)
            else:
                im = ax.imshow(v, cmap="RdBu_r" if "err" in k and "reduction" not in k else "PiYG", norm=ssqrt(M))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{short(s)}: {lab}", fontsize=8.5)
        tot = A[s]["primary"]["P-A"]["total"]
        axs[i, 0].set_ylabel(f"P − A objects RMS (total)\n{100 * tot['point']:+.2f} % [{100 * tot['lo']:+.2f}, {100 * tot['hi']:+.2f}]", fontsize=8)
    fig.colorbar(im, ax=axs[:, 1:], shrink=0.5, label=f"luminance difference (shared absolute scale ±{M:.3g}, square-root display)")
    fig.suptitle("Scene check (common R0 proposal, fixed replay): does the angular gain survive scene transport?", fontsize=11, x=0.01, ha="left")
    tag(fig, MEAS)
    pk.save(fig, "fig09_scene_residuals_and_error_reduction.png", "Do angular gains carry into rendered first- and second-bounce images?",
            MEAS, "Start 1 shown for A / P residuals; reduction maps use the start-mean.", "scenes/*/maps.npz, analysis.json")
    return {s: A[s]["primary"] for s in SCENES}


def table_html(pk):
    T = pk.E["tables"]
    ang = CM.POLAR
    head = "<tr><th>material</th><th>cohort</th><th>incidence</th><th>A el1</th><th>T el1</th><th>P el1</th><th>T−A</th><th>P−T</th><th>P−A</th></tr>"
    body = []
    with open(pk.out / "table_el1_by_material_and_incidence.csv", "w", encoding="utf-8") as f:
        f.write("material,cohort,polar_deg,azimuth_deg,A_el1,T_el1,P_el1,A_eshape,T_eshape,P_eshape,A_ealb,T_ealb,P_ealb\n")
        for m, r in enumerate(pk.rows):
            for p in range(CM.NP):
                a, t, q = (T[x]["el1"][m][p] for x in CM.ARMS)
                body.append(f"<tr><td>{short(r['tag'])}</td><td>{r['cohort']}</td><td>{ang[p]:g}° / {22.5 if p % 2 == 0 else 112.5}°</td>"
                            f"<td>{a:.4f}</td><td>{t:.4f}</td><td>{q:.4f}</td><td>{100 * (t / a - 1):+.1f}%</td><td>{100 * (q / t - 1):+.1f}%</td>"
                            f"<td>{100 * (q / a - 1):+.1f}%</td></tr>")
                f.write(",".join([short(r["tag"]), r["cohort"], f"{ang[p]:g}", "22.5" if p % 2 == 0 else "112.5"] +
                                 [f"{T[x][k][m][p]:.6g}" for k in ("el1", "eshape", "ealb") for x in CM.ARMS]) + "\n")
    return "<table class='t'>" + head + "".join(body) + "</table>"


def index(pk, M, scene):
    D = pk.D
    figs = "".join(f"<section><h3>{html.escape(f['file'])} <span class='tag {'m' if f['kind'] == MEAS else 'c'}'>{html.escape(f['kind'])}</span></h3>"
                   f"<p><b>Question:</b> {html.escape(f['question'])}</p><p>{html.escape(f['caption'])} <i>Source: {html.escape(f['source'])}</i></p>"
                   f"<img src='{html.escape(f['file'])}' loading='lazy'></section>" for f in pk.figs)
    dec = "".join(f"<tr><td>{html.escape(k)}</td><td><b>{html.escape(str(v))}</b></td></tr>" for k, v in D["decision_table"].items())
    page = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>WL104K symmetric gates</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--mut:#555;--bd:#ddd}}
@media (prefers-color-scheme: dark){{:root:not([data-theme=light]){{--bg:#16181d;--fg:#e8e8e8;--mut:#aaa;--bd:#333}}}}
body{{background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif;margin:0 auto;max-width:1400px;padding:16px}}
img{{max-width:100%;background:#fff;border:1px solid var(--bd)}} .tag{{font-size:12px;padding:2px 6px;border-radius:4px;color:#fff}}
.tag.m{{background:#2b6cb0}} .tag.c{{background:#b7791f}} table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid var(--bd);padding:3px 6px}}
.scroll{{overflow-x:auto}} p{{color:var(--mut)}}</style></head><body>
<h1>WL104K — Reciprocal per-component directional modulation (ER-v1-SDM candidate)</h1>
<p>Arms: <b>A</b> shared ER-v1-GF; <b>T</b> one shared symmetric gate q(μi)q(μo) on all 8 terms; <b>P</b> one gate per term (candidate).
All arms are reciprocal and passive, start from the same physical response and optimize jointly with the WL82 objective on the WL104J fit set.
Evaluation: WL104D Amendment 1 independent layered reference (never fitted). WL104J oracles B / C are historical context only. Candidate, not adopted.</p>
<h2>Frozen decision table</h2><table>{dec}</table>
<h2>Every material and incidence (median over 4 starts)</h2><div class='scroll'>{table_html(pk)}</div>
<p>CSV: <a href='table_el1_by_material_and_incidence.csv'>table_el1_by_material_and_incidence.csv</a>. Angular map scale ±{M:.4g} (shared, square root).</p>
<h2>Figures</h2>{figs}</body></html>"""
    (pk.out / "index.html").write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pk = Pack(args.root, args.out)
    fig_concept(pk); fig_profiles(pk); M = fig_maps(pk); fig_tables(pk); fig_gates(pk); fig_cost(pk); fig_summary(pk); fig_training(pk)
    scene = fig_scenes(pk)
    for arm in CM.ARMS:
        pk.check(f"table {arm} el1 = median over starts", pk.E["tables"][arm]["el1"], np.median(np.array(pk.E["tables"][arm]["el1_per_start"]), 0), 1e-12)
    index(pk, M, scene)
    man = {"figures": pk.figs, "checks": pk.checks, "all_checks_pass": all(c["pass"] for c in pk.checks), "angular_map_scale": M,
           "eval_sha256": CM.sha(args.root / "eval/eval.json"), "decision_sha256": CM.sha(args.root / "decision.json"),
           "lanes_sha256": pk.E["lanes_sha256"], "script_sha256": CM.sha(Path(__file__)),
           "files_sha256": {f.name: CM.sha(f) for f in sorted(args.out.glob("*")) if f.suffix in (".png", ".csv", ".html")}}
    CM.write(args.out / "figures_manifest.json", man)
    print("figures", len(pk.figs), "checks pass", man["all_checks_pass"], flush=True)


if __name__ == "__main__":
    main()
