"""WL104J reviewer package: readable figures, comparison tables, index.html and a manifest in evaluation/104J.

Usage: python figures104j.py <run_root> <out_dir>
Reads only <run_root>/eval/{eval.json, decision.json, lanes.npz} and <run_root>/fit_*/fit.json; every plotted number is
re-checked against eval.json (figure data consistency checks recorded in the manifest).
"""
import argparse
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import FuncNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common104j as CM  # noqa: E402

LUMA = CM.LUMA
ARMC = {"reference": "#111111", "A": "#1f77b4", "B": "#e08214", "C": "#2ca02c", "C_noF": "#9467bd", "R0": "#7f7f7f"}
ARMN = {"A": "A  shared ER (one parameter set for all incidences)",
        "B": "B  amplitude-local oracle (amplitudes per incidence)",
        "C": "C  full-local-lobe oracle (amplitudes + lobe shapes per incidence)",
        "C_noF": "C without Fresnel (amplitudes + roughness per incidence)"}
ARMS_SHORT = {"A": "A shared", "B": "B amplitude-local", "C": "C full-local", "C_noF": "C without Fresnel"}
TERMN = {"O0": "Outer 1", "O1": "Outer 2", "O2": "Outer 3", "SL": "Lambert", "SI0": "Inner 1", "SI1": "Inner 2",
         "SI2": "Inner 3", "SI3": "Inner 4"}
TERMS = list(TERMN)
MEAS = "MEASURED"
CONC = "CONCEPTUAL (no data)"
MAP_PROBES = [2, 6, 10, 18]
MAP_POLAR = [20, 50, 60, 86]


def short(tag):
    return tag.replace("test83_", "")


def ssqrt_norm(M):
    return FuncNorm((lambda x: np.sign(x) * np.sqrt(np.abs(x) / M), lambda y: np.sign(y) * y ** 2 * M), vmin=-M, vmax=M)


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


def tag(fig, text):
    fig.text(0.995, 0.995, text, ha="right", va="top", fontsize=8, color="white",
             bbox=dict(boxstyle="round,pad=0.25", fc="#2b6cb0" if text.startswith("MEAS") else "#b7791f", ec="none"))


def foot(fig, text, y=0.005):
    import textwrap
    fig.text(0.01, y, "\n".join(textwrap.fill(p, 190) for p in text.split("\n")), fontsize=7.5, va="bottom", color="#333")


class Pack:
    def __init__(self, root, out):
        self.root, self.out = root, out
        self.E = CM.read(root / "eval/eval.json")
        self.D = CM.read(root / "eval/decision.json")
        if CM.sha(root / "eval/lanes.npz") != self.E["lanes_sha256"]:
            raise CM.Stop("LANES_HASH")
        with np.load(root / "eval/lanes.npz") as z:
            self.L = {k: z[k] for k in z.files}
        self.rows = self.E["materials"]
        self.n = len(self.rows)
        self.coh = self.E["cohorts"]
        self.checks = []
        self.figs = []

    def check(self, name, a, b, tol=1e-6):
        d = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
        self.checks.append({"check": name, "max_abs_diff": d, "pass": bool(d <= tol)})

    def save(self, fig, name, question, kind, caption):
        f = self.out / name
        fig.savefig(f, dpi=120)
        plt.close(fig)
        self.figs.append({"file": name, "question": question, "kind": kind, "caption": caption})


# ------------------------------------------------------------------ figure 1 (conceptual)
def fig_concept(pk):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
    groups = ["5°", "20°", "35°", "50°", "57.5°", "60°", "62.5°", "70°", "80°", "86°"]
    blocks = [("amplitudes (Lambert, 3 outer, 4 inner)", 0), ("lobe shapes: roughness, anisotropy", 1), ("lobe Fresnel F0", 2),
              ("attenuation s(μ) and η", 3)]
    local = {"A": set(), "B": {0}, "C": {0, 1, 2}}
    for ax, arm in zip(axs, ("A", "B", "C")):
        for bi, (bn, k) in enumerate(blocks):
            y = 3 - bi
            if k in local[arm]:
                for g in range(10):
                    ax.add_patch(Rectangle((g, y + 0.1), 0.9, 0.8, fc=ARMC[arm], ec="white", alpha=0.85))
            else:
                ax.add_patch(Rectangle((0, y + 0.1), 9.9, 0.8, fc="#bbbbbb", ec="white"))
                ax.text(4.95, y + 0.5, "one shared value", ha="center", va="center", fontsize=8.5)
            if arm == "A":
                ax.text(-0.2, y + 0.5, bn, ha="right", va="center", fontsize=8.5)
        ax.set_xlim(-0.1, 10)
        ax.set_ylim(0, 4.3)
        ax.set_xticks(np.arange(10) + 0.45, groups, fontsize=7.5)
        ax.set_yticks([])
        ax.set_xlabel("incident polar angle group")
        ax.set_title(ARMN[arm].replace(" (", "\n("), fontsize=9.5, loc="left")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Why parameter sharing may constrain the certified ER: one parameter set must serve every incident angle",
                 x=0.01, ha="left", weight="bold")
    tag(fig, CONC)
    foot(fig, "Coloured cells = parameters that may take a different value in each incident-angle group; grey bar = one value shared by all "
              "incident angles. The response family (7 GGX lobes, Lambert, attenuation, refraction mapping) is identical in all three arms. "
              "B and C are diagnostic oracles: each group's state is a valid certified ER state, but the incident-conditioned response is not "
              "a reciprocal BSDF and is never rendered.")
    fig.subplots_adjust(left=0.17, right=0.99, top=0.8, bottom=0.2, wspace=0.08)
    pk.save(fig, "fig01_parameter_sharing_concept.png", "What does relaxing cross-incidence sharing mean for the three arms?",
            CONC, "Schematic only. A shares every parameter across incident angles; B lets the amplitudes vary per incident-angle group; "
                  "C additionally lets roughness, anisotropy and Fresnel F0 of all seven lobes vary per group. Attenuation and η stay "
                  "shared (frozen to A) in B and C.")


# ------------------------------------------------------------------ figure 2 (profiles)
def radial_profile(wi, wo, w, F, edges):
    r = CM.mirror_radius(wi, wo)
    lum = F @ LUMA
    idx = np.digitize(r, edges) - 1
    num = np.bincount(idx[(idx >= 0) & (idx < len(edges) - 1)], (w * lum)[(idx >= 0) & (idx < len(edges) - 1)], len(edges) - 1)
    den = np.bincount(idx[(idx >= 0) & (idx < len(edges) - 1)], w[(idx >= 0) & (idx < len(edges) - 1)], len(edges) - 1)
    return np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)


def fig_profiles(pk):
    L = pk.L
    edges = np.r_[0, 1, 2, 3, 4.5, 6, 8, 10, 12, 15, 19, 24, 30, 37, 45, 55, 70, 90]
    ctr = np.sqrt(np.maximum(edges[:-1], 0.5) * edges[1:])
    names = []
    for m, r in enumerate(pk.rows):
        fig, axs = plt.subplots(2, 4, figsize=(17, 8.2), gridspec_kw=dict(height_ratios=[2.2, 1]), sharex="col")
        for j, p in enumerate(MAP_PROBES):
            ax, ar = axs[0, j], axs[1, j]
            wi, wo, w = L["wi"][m, p], L["wo"][m, p], L["w"][m, p]
            ref = radial_profile(wi, wo, w, L["ref_mean"][m, p], edges)
            ax.plot(ctr, ref, "o-", color=ARMC["reference"], lw=2.2, ms=4, label="layered reference (independent)")
            for arm in ("A", "B", "C"):
                v = radial_profile(wi, wo, w, L[f"pred_{arm}"][m, p], edges)
                ax.plot(ctr, v, "-", color=ARMC[arm], lw=1.5, label=ARMS_SHORT[arm])
                ar.plot(ctr, (v - ref) / np.maximum(ref, 1e-12) * 100, "-", color=ARMC[arm], lw=1.5)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"incident polar {MAP_POLAR[j]}°, azimuth 22.5°", fontsize=9.5)
            ar.axhline(0, color="k", lw=0.8)
            ar.set_ylim(-60, 60)
            ar.set_xlabel("angle from the mirror direction (deg, log scale)")
            if j == 0:
                ax.set_ylabel("mean response f·cos θo over the ring\n(luminance, sr⁻¹)")
                ar.set_ylabel("model − reference\n(% of reference)")
            if j == 3:
                ax.legend(fontsize=8, loc="lower left")
        coh = "development" if r["cohort"] == "development" else "held out"
        fig.suptitle(f"Material {short(r['tag'])} ({coh}): the same material at four incident angles — reference vs A / B / C",
                     x=0.01, ha="left", weight="bold")
        tag(fig, MEAS)
        foot(fig, "Each point is the quadrature-weighted mean response over a ring of outgoing directions at the given angle from the mirror "
                  "direction, computed on the independent evaluation lanes (never used for fitting). States: best training-objective start per arm. "
                  "Lower row: relative difference to the reference, clipped to ±60 %.")
        fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.12, hspace=0.08, wspace=0.25)
        name = f"fig02_response_profiles_{short(r['tag'])}_reference_vs_shared_and_local_oracles.png"
        pk.save(fig, name, f"How do A, B and C reproduce the angular response of {short(r['tag'])} at 20/50/60/86°?", MEAS,
                "Ring-averaged response vs angle from mirror; reference in black.")
        names.append(name)
    return names


# ------------------------------------------------------------------ figure 3 (residual and reduction maps)
def fig_maps(pk):
    L = pk.L
    maps, M = {}, 0.0
    for m in range(pk.n):
        for j, p in enumerate(MAP_PROBES):
            wi = L["wi"][m, p, 0]; wo = L["wo"][m, p]; w = L["w"][m, p]
            A = (w[:, None] * L["ref_mean"][m, p]).sum(0) @ LUMA
            res = {arm: (L["ref_mean"][m, p] - L[f"pred_{arm}"][m, p]) @ LUMA / A for arm in ("A", "B", "C")}
            mm = {f"res_{a}": bin_map(wi, wo, v) for a, v in res.items()}
            mm["red_BA"] = bin_map(wi, wo, np.abs(res["A"]) - np.abs(res["B"]))
            mm["red_CB"] = bin_map(wi, wo, np.abs(res["B"]) - np.abs(res["C"]))
            maps[(m, j)] = mm
            M = max(M, max(np.nanmax(np.abs(v)) for v in mm.values()))
    names = []
    rowsn = [("res_A", "A shared:\nreference − model"), ("res_B", "B amplitude-local:\nreference − model"),
             ("res_C", "C full-local:\nreference − model"), ("red_BA", "error reduction B vs A:\n|res A| − |res B|"),
             ("red_CB", "error reduction C vs B:\n|res B| − |res C|")]
    for m, r in enumerate(pk.rows):
        fig, axs = plt.subplots(5, 4, figsize=(13, 16), layout="constrained")
        for j in range(4):
            for i, (k, lab) in enumerate(rowsn):
                ax = axs[i, j]
                cmap = "RdBu_r" if k.startswith("res") else "PiYG"
                im = ax.imshow(maps[(m, j)][k], origin="lower", extent=[-45, 45, -45, 45], cmap=cmap, norm=ssqrt_norm(M))
                ax.set_xticks([-30, 0, 30]); ax.set_yticks([-30, 0, 30])
                ax.tick_params(labelsize=7)
                if i == 0:
                    ax.set_title(f"incident polar {MAP_POLAR[j]}°", fontsize=10)
                if j == 0:
                    ax.set_ylabel(lab, fontsize=8.5)
            cb = fig.colorbar(im, ax=axs[3:, :], shrink=0.6, location="right") if j == 3 else None
        cb0 = fig.colorbar(axs[0, 0].images[0], ax=axs[:3, :], shrink=0.6, location="right")
        cb0.set_label("signed residual per sr, as a fraction of the reference directional albedo (luminance)")
        if cb is not None:
            cb.set_label("reduction of |residual| (green = local arm closer to the reference)")
        coh = "development" if r["cohort"] == "development" else "held out"
        fig.suptitle(f"Where each relaxation changes the error — material {short(r['tag'])} ({coh})\n"
                     f"axes: outgoing direction relative to the mirror direction (deg); one absolute colour scale (±{M:.3g}) for every "
                     f"map of WL104J, square-root display", fontsize=10.5, x=0.01, ha="left")
        tag(fig, MEAS)
        name = f"fig03_signed_residual_and_error_reduction_maps_{short(r['tag'])}.png"
        pk.save(fig, name, f"Where in the outgoing hemisphere do B and C change the error of {short(r['tag'])}?", MEAS,
                "Rows 1-3: signed residual (red = model too dark, blue = too bright). Rows 4-5: actual change of the absolute error "
                "(green = improvement). Shared absolute scale across all WL104J maps.")
        names.append(name)
    return names, M


# ------------------------------------------------------------------ figure 4 (regions)
def fig_regions(pk):
    T = pk.E["tables"]
    pol = np.array(pk.E["probe_polar"])
    ang = CM.ANGLES
    fig, axs = plt.subplots(4, pk.n, figsize=(20, 11), sharex=True)
    keys = [("core_l1", "mirror core (< 3°)\nL1 error / reference energy"), ("shoulder_l1", "shoulder (3–12°)\nL1 error / reference energy"),
            ("tail_l1", "tail (≥ 12°)\nL1 error / reference energy"), ("peak_signed", "peak height\n(model / reference − 1)")]
    for i, (k, lab) in enumerate(keys):
        for m, r in enumerate(pk.rows):
            ax = axs[i, m]
            for arm in ("A", "B", "C"):
                v = np.array(T[arm][k])[m].reshape(len(ang), 2).mean(1)
                ax.plot(ang, v, "o-", ms=3, color=ARMC[arm], label=ARMS_SHORT[arm])
            if k == "peak_signed":
                ax.axhline(0, color="k", lw=0.7)
            if i == 0:
                ax.set_title(f"{short(r['tag'])}\n({'dev' if r['cohort'] == 'development' else 'held out'})", fontsize=9)
            if m == 0:
                ax.set_ylabel(lab, fontsize=8.5)
            if i == 3:
                ax.set_xlabel("incident polar (deg)", fontsize=8)
            ax.tick_params(labelsize=7)
    axs[0, 0].legend(fontsize=7.5)
    fig.suptitle("Mirror core, shoulder and tail error by material and incident angle (independent evaluation lanes; median over 4 starts; "
                 "mean of the two azimuths)", x=0.01, ha="left", weight="bold")
    tag(fig, MEAS)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.88, bottom=0.07, hspace=0.25, wspace=0.35)
    pk.save(fig, "fig04_core_shoulder_tail_and_peak_error_by_material_and_incidence.png",
            "Do the relaxations reduce the error in the mirror core, the shoulder or the tail, and at which incidences?", MEAS,
            "Region L1 = sum over the region of |model − reference| (luminance, quadrature-weighted) divided by the reference energy in "
            "that region. Peak from the independent peak grid.")
    for arm in ("A", "B", "C"):
        for k in ("core_l1", "shoulder_l1", "tail_l1", "peak_signed"):
            pk.check(f"fig04 {arm} {k} source = eval tables", np.array(T[arm][k]).shape, (pk.n, CM.NP), 0)


# ------------------------------------------------------------------ figure 5 (lobes)
def fig_lobes(pk):
    Pp = pk.E["parameter_profiles"]
    ang = CM.ANGLES
    lob = ["Outer 1", "Outer 2", "Outer 3", "Inner 1", "Inner 2", "Inner 3", "Inner 4"]
    cols = plt.cm.tab10(np.arange(7))
    fig, axs = plt.subplots(3, pk.n, figsize=(21, 9.5), sharex=True)
    for m, r in enumerate(pk.rows):
        aA = np.sqrt(np.array(Pp["A"]["au"])[m, 0] * np.array(Pp["A"]["av"])[m, 0])
        aC = np.sqrt(np.array(Pp["C"]["au"])[m] * np.array(Pp["C"]["av"])[m])        # [G, 7]
        fA = np.array(Pp["A"]["f0"])[m, 0]
        fC = np.array(Pp["C"]["f0"])[m]
        lamA = np.array(Pp["A"]["lam"])[m, 0] @ LUMA if False else (np.array(Pp["A"]["lam"])[m, 0].T @ LUMA)   # [8]
        lamB = np.einsum("gck,c->gk", np.array(Pp["B"]["lam"])[m], LUMA)
        lamC = np.einsum("gck,c->gk", np.array(Pp["C"]["lam"])[m], LUMA)
        for k in range(7):
            axs[0, m].plot(ang, aC[:, k], "o-", ms=2.5, color=cols[k], label=lob[k])
            axs[0, m].axhline(aA[k], color=cols[k], ls=":", lw=1)
            axs[1, m].plot(ang, fC[:, k], "o-", ms=2.5, color=cols[k])
            axs[1, m].axhline(fA[k], color=cols[k], ls=":", lw=1)
            axs[2, m].plot(ang, lamC[:, k + 1], "o-", ms=2.5, color=cols[k])
            axs[2, m].plot(ang, lamB[:, k + 1], "x--", ms=3, lw=0.8, color=cols[k])
            axs[2, m].axhline(lamA[k + 1], color=cols[k], ls=":", lw=1)
        axs[2, m].plot(ang, lamC[:, 0], "o-", ms=2.5, color="k")
        axs[2, m].axhline(lamA[0], color="k", ls=":", lw=1)
        axs[0, m].set_yscale("log")
        axs[0, m].set_title(f"{short(r['tag'])}\n({'dev' if r['cohort'] == 'development' else 'held out'})", fontsize=9)
        axs[2, m].set_xlabel("incident polar (deg)", fontsize=8)
        for a in axs[:, m]:
            a.tick_params(labelsize=7)
    axs[0, 0].set_ylabel("GGX roughness α (geometric mean u, v)\nC per incidence (solid), A shared (dotted)", fontsize=8)
    axs[1, 0].set_ylabel("Fresnel F0\nC (solid), A (dotted)", fontsize=8)
    axs[2, 0].set_ylabel("amplitude λ (luminance)\nC (solid), B (dashed ×), A (dotted);\nblack = Lambert", fontsize=8)
    axs[0, 0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("Lobe behaviour: locally fitted lobe parameters vs incident angle, with the shared A value for comparison "
                 "(best training-objective start)", x=0.01, ha="left", weight="bold")
    tag(fig, MEAS)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.88, bottom=0.07, hspace=0.12, wspace=0.4)
    pk.save(fig, "fig05a_local_lobe_parameters_vs_incident_angle.png",
            "How do the locally fitted lobe parameters change with incident angle compared with the shared fit?", MEAS,
            "Amplitudes on the common section scale (attenuation knots at their maximal admissible scale), so A, B and C amplitudes are "
            "directly comparable. A lobe with negligible amplitude can take any roughness without affecting the response.")
    # dependency
    pairs = {arm: np.array(pk.E["lobe_pairs_ge_095"][arm]) for arm in ("A", "B", "C")}       # [n, P]
    share = {arm: np.array(pk.E["lobe_share"][arm]) for arm in ("A", "B", "C")}             # [n, P, 8]
    fig, axs = plt.subplots(2, pk.n, figsize=(21, 7.5), sharex=True)
    for m, r in enumerate(pk.rows):
        for arm in ("A", "B", "C"):
            axs[0, m].plot(ang, pairs[arm][m].reshape(-1, 2).mean(1), "o-", ms=3, color=ARMC[arm], label=ARMS_SHORT[arm])
        sh = share["C"][m].reshape(CM.G, 2, 8).mean(1)
        shA = share["A"][m].reshape(CM.G, 2, 8).mean(1)
        bottom = np.zeros(CM.G); bottomA = np.zeros(CM.G)
        for k, t in enumerate(TERMS):
            axs[1, m].bar(ang - 1.2, sh[:, k], 2.2, bottom=bottom, color=plt.cm.tab10(k), label=TERMN[t])
            axs[1, m].bar(ang + 1.2, shA[:, k], 2.2, bottom=bottomA, color=plt.cm.tab10(k), alpha=0.45)
            bottom += sh[:, k]; bottomA += shA[:, k]
        axs[0, m].set_title(f"{short(r['tag'])}\n({'dev' if r['cohort'] == 'development' else 'held out'})", fontsize=9)
        axs[1, m].set_xlabel("incident polar (deg)", fontsize=8)
        for a in axs[:, m]:
            a.tick_params(labelsize=7)
    axs[0, 0].set_ylabel("number of term pairs with\nangular-response cosine ≥ 0.95\n(of 28 pairs)", fontsize=8)
    axs[1, 0].set_ylabel("share of reflected energy per term\nleft bar C, right (faded) bar A", fontsize=8)
    axs[0, 0].legend(fontsize=7)
    axs[1, -1].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Lobe dependency: do previously overlapping terms stay overlapping when lobe parameters may vary with incidence?",
                 x=0.01, ha="left", weight="bold")
    tag(fig, MEAS)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.86, bottom=0.08, hspace=0.15, wspace=0.4)
    pk.save(fig, "fig05b_lobe_overlap_and_energy_share_vs_incident_angle.png",
            "Do near-duplicate lobes become functionally distinct when parameters are local?", MEAS,
            "Cosine = per-probe lane-weighted RGB cosine between the angular responses of two terms (WL104H definition). Cosine ignores "
            "amplitude, so a pair can overlap even when one term carries almost no energy (see the energy shares).")


# ------------------------------------------------------------------ figure 6 (summary)
def fig_summary(pk):
    E, D = pk.E, pk.D
    con = E["contrasts"]
    fig = plt.figure(figsize=(17, 11.5))
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.99, top=0.9, bottom=0.12, hspace=0.45, wspace=0.3)
    # (a) contrasts
    ax = fig.add_subplot(gs[0, 0])
    names = [("B-A", "B − A\namplitude sharing\nrelaxed"), ("C-B", "C − B\nlobe shape / Fresnel\nsharing relaxed"),
             ("C-A", "C − A\ntotal relaxation")]
    for i, (k, lab) in enumerate(names):
        for j, c in enumerate(("development", "held_out")):
            v = con[k]["eval:el1"][c]
            x = i + (j - 0.5) * 0.3
            ax.errorbar(x, 100 * v["median"], yerr=[[100 * (v["median"] - v["lo"])], [100 * (v["hi"] - v["median"])]], fmt="o",
                        color="#444" if j == 0 else "#c0392b", capsize=4, label=("development" if j == 0 else "held out") if i == 0 else None)
            for s in v["per_start"]:
                ax.plot(x + 0.06, 100 * s, ".", color="#999", ms=4)
            pk.check(f"fig06 {k} {c} median", v["median"], np.median(v["per_start"]), 1e-12)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(-20, color="#2ca02c", ls="--", lw=1)
    ax.text(2.45, -20, "substantial (−20 %)", fontsize=7.5, color="#2ca02c", ha="right", va="bottom")
    ax.axhline(-5, color="#e08214", ls=":", lw=1)
    ax.text(2.45, -5, "measurable (−5 %)", fontsize=7.5, color="#e08214", ha="right", va="bottom")
    ax.set_xticks(range(3), [l for _, l in names], fontsize=8)
    ax.set_ylabel("relative change of pooled held-out el1 (%)")
    ax.set_title("(a) held-out angular error change\nmedian over 4 starts, 95 % cluster bootstrap; grey = single starts", fontsize=9)
    ax.legend(fontsize=8)
    # (b) training vs held-out
    ax = fig.add_subplot(gs[0, 1])
    for i, (k, lab) in enumerate(names):
        for j, c in enumerate(("development", "held_out")):
            o = con[k]["overfit"][c]
            ax.bar(i + (j - 0.5) * 0.36 - 0.08, 100 * o["training_reduction"], 0.16, color="#bbb",
                   label="training lanes" if (i, j) == (0, 0) else None)
            ax.bar(i + (j - 0.5) * 0.36 + 0.08, 100 * o["heldout_reduction"], 0.16, color="#444" if j == 0 else "#c0392b",
                   label=f"held-out lanes ({'development' if j == 0 else 'held out'})" if i == 0 else None)
    ax.set_xticks(range(3), [l for _, l in names], fontsize=8)
    ax.set_ylabel("el1 reduction (%)")
    ax.set_title("(b) does the improvement persist on held-out outgoing directions?\n(training lanes vs independent evaluation lanes)",
                 fontsize=9)
    ax.legend(fontsize=7.5)
    # (c) synthetic controls
    ax = fig.add_subplot(gs[0, 2])
    ctl = [("S0a", "null target,\nnear-truth start"), ("S0b", "null target,\ngeneric start"), ("S1", "known amplitude\nvariation"),
           ("S2", "known shape\nvariation")]
    for i, (k, lab) in enumerate(ctl):
        m = E["synthetic"][k]["heldout_el1_mean"]
        for j, arm in enumerate(("A", "B", "C")):
            ax.bar(i + (j - 1) * 0.25, m[arm], 0.23, color=ARMC[arm], label=ARMS_SHORT[arm] if i == 0 else None)
        ps = E["synthetic"][k].get("pass")
        ax.text(i, max(m.values()) * 1.05, "" if ps is None else ("PASS" if ps else "FAIL"), ha="center", fontsize=8,
                color="#2ca02c" if ps else "#c0392b")
    ax.set_xticks(range(4), [l for _, l in ctl], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("held-out el1 vs the known truth")
    ax.set_title("(c) synthetic controls: does the diagnostic detect\nthe mechanism it should, and nothing else?", fontsize=9)
    ax.legend(fontsize=7.5)
    # (d) transfer
    ax = fig.add_subplot(gs[1, 0])
    for i, (k, lab) in enumerate(names):
        for j, part in enumerate(("polar_transfer", "azimuth_transfer")):
            for jj, c in enumerate(("development", "held_out")):
                v = con[k][f"{part}:el1"][c]
                x = i + (2 * j + jj - 1.5) * 0.17
                ax.errorbar(x, 100 * v["median"], yerr=[[100 * (v["median"] - v["lo"])], [100 * (v["hi"] - v["median"])]], fmt="s" if j == 0 else "^",
                            color="#444" if jj == 0 else "#c0392b", capsize=3,
                            label=(f"{'nearby polar' if j == 0 else 'new azimuth'}, {'development' if jj == 0 else 'held out'}") if i == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(3), [l for _, l in names], fontsize=8)
    ax.set_ylabel("relative el1 change vs the less local arm (%)")
    ax.set_title("(d) transfer to independently held-out nearby incident directions\n(never used for fitting or selection)", fontsize=9)
    ax.legend(fontsize=7)
    # (e) reciprocity
    ax = fig.add_subplot(gs[1, 1])
    for i, arm in enumerate(("A", "B", "C", "C_noF")):
        v = np.array([x["sum_relative"] for x in E["reciprocity"][arm]])        # [R, n]
        ax.plot(np.full(v.size, i) + np.random.default_rng(0).uniform(-0.15, 0.15, v.size), np.maximum(v.ravel(), 1e-17), ".",
                color=ARMC[arm], ms=5)
    ax.set_yscale("log")
    ax.set_xticks(range(4), [ARMS_SHORT[a] for a in ("A", "B", "C", "C_noF")], fontsize=8)
    ax.set_ylabel("Σ|f(wi,wo) − f(wo,wi)| / Σ mean")
    ax.set_title("(e) reciprocity inconsistency of the model (no reference needed)\none dot per material and start", fontsize=9)
    # (f) verdict text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    L = D["effect_labels"]
    lines = [f"Verdict (frozen rules): {D['verdict']}" + (f"  [{', '.join(D['reasons'])}]" if D["reasons"] else ""),
             "", f"B − A amplitude sharing:  {L['B-A']['label']}", f"C − B lobe-parameter sharing:  {L['C-B']['label']}",
             f"C − A total:  {L['C-A']['label']}",
             f"Attribution: {D['attribution']['label'] if D['attribution'] else 'not applicable'}",
             f"Residual after C / A: dev {D['residual_C_over_A']['development']:.2f}, held out {D['residual_C_over_A']['held_out']:.2f}",
             f"Synthetic controls: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in D["synthetic_controls"].items()),
             f"Optimization resolved: {D['optimization_resolved']}",
             f"Overfit-dominated: " + ", ".join(f"{k} {v}" for k, v in D["overfit_dominated"].items()),
             f"Separable from optimizer slack: " + ", ".join(f"{k} {v}" for k, v in D["slack_separable"].items()),
             f"Reference resolves comparisons: {D['reference_resolves']}"]
    import textwrap
    ax.text(0, 1, "\n".join(textwrap.fill(x, 58, subsequent_indent="  ") if x else x for x in lines), va="top", fontsize=9,
            family="monospace")
    ax.set_title("(f) hypothesis outcome", fontsize=9, loc="left")
    fig.suptitle("Summary: which part of the angular residual is explained by cross-incidence parameter sharing?", x=0.01, ha="left",
                 weight="bold")
    tag(fig, MEAS)
    foot(fig, "el1 = WL82 energy-weighted relative L1 of f·cos θo against the mean of the two independent evaluation halves, pooled over "
              "(material, probe) within a cohort. Development = 3 materials, held out = 6. B and C are diagnostic oracles (not reciprocal "
              "BSDFs): their gains are empirical approximation headroom on these incidences, not a deployable representation and not a "
              "capacity lower bound.", y=0.01)
    pk.save(fig, "fig06_hypothesis_outcome_summary.png",
            "Does relaxing cross-incidence sharing expose reproducible held-out headroom, and is it amplitude or lobe-parameter freedom?",
            MEAS, "Panels: held-out contrasts, training vs held-out, synthetic controls, transfer, reciprocity, frozen-rule labels.")


# ------------------------------------------------------------------ supporting figures
def fig_training(pk):
    fj = CM.read(pk.root / "fit_primary/fit.json")
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
    for ax, arm in zip(axs, ("A", "B", "C", "C_noF")):
        H = np.array(fj[arm]["history"])                       # [T, R, n]
        st = np.array(fj[arm]["history_steps"])
        for r in range(H.shape[1]):
            for m in range(H.shape[2]):
                ax.plot(st, H[:, r, m], color=plt.cm.tab10(m), lw=0.7, alpha=0.8,
                        label=short(pk.rows[m]["tag"]) if r == 0 else None)
        ax.set_yscale("log")
        ax.set_title(f"{ARMS_SHORT[arm]} ({fj['trainable_per_material'][arm]} trainable / material)\n"
                     f"{fj[arm]['seconds']:.0f} s for 4 starts × 9 materials", fontsize=9)
        ax.set_xlabel("optimizer step")
    axs[0].set_ylabel("training objective (WL82 el1 + ealb)")
    axs[-1].legend(fontsize=7, ncol=2)
    fig.suptitle("Training histories (fit lanes only; B starts from A, C and C-without-Fresnel from B)", x=0.01, ha="left", weight="bold")
    tag(fig, MEAS)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.8, bottom=0.14, wspace=0.08)
    pk.save(fig, "fig07_training_histories_all_arms.png", "Did every arm converge, and how stable are the starts?", MEAS,
            "One line per material and start. Wall times were measured while another project's job shared the GPU (contended).")


def fig_data(pk):
    dj = pk.E["data"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, k in zip(axs, ("fit_set", "polar_transfer", "azimuth_transfer")):
        pr = dj["precision"][k]
        x = np.arange(pk.n)
        ax.bar(x - 0.2, [r["albedo"] for r in pr], 0.2, label="albedo noise (limit 0.008)")
        ax.bar(x, [r["band_shape"] or 0 for r in pr], 0.2, label="55–65° shape noise (limit 0.015)")
        ax.bar(x + 0.2, [r["grazing_shape"] or 0 for r in pr], 0.2, label="≥ 80° shape noise (limit 0.025)")
        ax.set_xticks(x, [short(r["tag"]) for r in pk.rows], rotation=60, fontsize=7.5)
        ax.set_title({"fit_set": "fit set (targets of every arm)", "polar_transfer": "nearby-polar transfer set",
                      "azimuth_transfer": "new-azimuth transfer set"}[k], fontsize=9.5)
    axs[0].set_ylabel("half-split noise (WL104D precision rule)")
    axs[0].legend(fontsize=7)
    fig.suptitle("Fresh layered reference data: precision of the fit and transfer sets (independent of the WL104D evaluation reference)",
                 x=0.01, ha="left", weight="bold")
    tag(fig, MEAS)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.25)
    pk.save(fig, "fig08_fit_and_transfer_data_precision.png", "Are the new fitting and transfer targets precise enough?", MEAS,
            "1024 lanes per probe, 8192 layered walks per lane (depth 1024), fresh direction and walk streams.")


def table_html(pk):
    T = pk.E["tables"]
    ang = CM.POLAR
    head = "<tr><th>material</th><th>cohort</th><th>incidence</th>" + "".join(
        f"<th>{a} el1</th>" for a in ("A", "B", "C")) + "<th>B−A</th><th>C−B</th><th>C−A</th><th>A eshape</th><th>C eshape</th><th>A ealb</th><th>C ealb</th></tr>"
    body = []
    for m, r in enumerate(pk.rows):
        for p in range(CM.NP):
            a, b, c = (T[x]["el1"][m][p] for x in ("A", "B", "C"))
            body.append(f"<tr><td>{short(r['tag'])}</td><td>{r['cohort']}</td><td>{ang[p]:g}° / {CM.AZ[p % 2]:g}°</td>"
                        f"<td>{a:.4f}</td><td>{b:.4f}</td><td>{c:.4f}</td><td>{100 * (b / a - 1):+.1f}%</td><td>{100 * (c / b - 1):+.1f}%</td>"
                        f"<td>{100 * (c / a - 1):+.1f}%</td><td>{T['A']['eshape'][m][p]:.4f}</td><td>{T['C']['eshape'][m][p]:.4f}</td>"
                        f"<td>{T['A']['ealb'][m][p]:.4f}</td><td>{T['C']['ealb'][m][p]:.4f}</td></tr>")
    rows = []
    with open(pk.out / "table_el1_by_material_and_incidence.csv", "w", encoding="utf-8") as f:
        f.write("material,cohort,polar_deg,azimuth_deg,A_el1,B_el1,C_el1,C_noF_el1,A_eshape,B_eshape,C_eshape,A_ealb,B_ealb,C_ealb\n")
        for m, r in enumerate(pk.rows):
            for p in range(CM.NP):
                f.write(",".join([short(r["tag"]), r["cohort"], f"{ang[p]:g}", f"{CM.AZ[p % 2]:g}"] +
                                 [f"{T[x]['el1'][m][p]:.6g}" for x in ("A", "B", "C", "C_noF")] +
                                 [f"{T[x]['eshape'][m][p]:.6g}" for x in ("A", "B", "C")] +
                                 [f"{T[x]['ealb'][m][p]:.6g}" for x in ("A", "B", "C")]) + "\n")
    return "<table class='t'>" + head + "".join(body) + "</table>"


def index(pk, maps_M):
    E, D = pk.E, pk.D
    figs = "".join(f"<section><h3>{html.escape(f['file'])} <span class='tag {'m' if f['kind'] == MEAS else 'c'}'>{html.escape(f['kind'])}</span></h3>"
                   f"<p><b>Question:</b> {html.escape(f['question'])}</p><p>{html.escape(f['caption'])}</p>"
                   f"<img src='{html.escape(f['file'])}' loading='lazy'></section>" for f in pk.figs)
    dec = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(json.dumps(v))}</td></tr>" for k, v in D.items()
                  if k not in ("commit", "eval_sha256"))
    page = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>WL104J sharing ablation</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--mut:#555;--bd:#ddd}}
@media (prefers-color-scheme: dark){{:root:not([data-theme=light]){{--bg:#16181d;--fg:#e8e8e8;--mut:#aaa;--bd:#333}}}}
body{{background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif;margin:0 auto;max-width:1400px;padding:16px}}
img{{max-width:100%;background:#fff;border:1px solid var(--bd)}} .tag{{font-size:12px;padding:2px 6px;border-radius:4px;color:#fff}}
.tag.m{{background:#2b6cb0}} .tag.c{{background:#b7791f}} table{{border-collapse:collapse;font-size:13px}} td,th{{border:1px solid var(--bd);padding:3px 6px}}
.t{{font-size:12px}} .scroll{{overflow-x:auto}} p{{color:var(--mut)}}</style></head><body>
<h1>WL104J — Cross-incidence parameter sharing ablation (diagnostic oracles)</h1>
<p>Arms: <b>A</b> shared ER-v1-GF (58 parameters per material, one state for every incidence); <b>B</b> amplitude-only local oracle
(8 amplitudes per channel per incident-angle group, lobe shapes, attenuation and η frozen to A); <b>C</b> full local lobe oracle
(B + roughness, anisotropy and Fresnel F0 of all 7 lobes per group); <b>C without Fresnel</b> (secondary). B and C are DIAGNOSTIC ONLY:
not reciprocal BSDFs, never rendered. Physical reference: WL104D Amendment 1 independent layered angular reference (evaluation only);
fitting targets: fresh layered walks at the same incidences with independent outgoing directions. Historical R0 and WL104I states are
external controls only.</p>
<h2>Frozen decision</h2><table>{dec}</table>
<h2>Comparison table (every material and incidence; median over 4 starts)</h2><div class='scroll'>{table_html(pk)}</div>
<p>CSV: <a href='table_el1_by_material_and_incidence.csv'>table_el1_by_material_and_incidence.csv</a>. Error-map scale: ±{maps_M:.4g}
(shared by every WL104J map, square-root display).</p>
<h2>Figures</h2>{figs}</body></html>"""
    (pk.out / "index.html").write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pk = Pack(args.root, args.out)
    fig_concept(pk)
    fig_profiles(pk)
    _, M = fig_maps(pk)
    fig_regions(pk)
    fig_lobes(pk)
    fig_summary(pk)
    fig_training(pk)
    fig_data(pk)
    # consistency: plotted el1 tables vs per-start arrays
    for arm in ("A", "B", "C"):
        pk.check(f"table {arm} el1 = median over starts", pk.E["tables"][arm]["el1"],
                 np.median(np.array(pk.E["tables"][arm]["el1_per_start"]), 0), 1e-12)
    index(pk, M)
    man = {"figures": pk.figs, "checks": pk.checks, "all_checks_pass": all(c["pass"] for c in pk.checks), "map_scale": M,
           "map_display": "signed square root of |x| / M", "eval_sha256": CM.sha(args.root / "eval/eval.json"),
           "decision_sha256": CM.sha(args.root / "eval/decision.json"), "lanes_sha256": pk.E["lanes_sha256"],
           "script_sha256": CM.sha(Path(__file__)), "files_sha256": {f.name: CM.sha(f) for f in sorted(args.out.glob("*"))
                                                                     if f.suffix in (".png", ".csv", ".html")}}
    CM.write(args.out / "figures_manifest.json", man)
    print("figures", len(pk.figs), "checks pass", man["all_checks_pass"], flush=True)


if __name__ == "__main__":
    main()
