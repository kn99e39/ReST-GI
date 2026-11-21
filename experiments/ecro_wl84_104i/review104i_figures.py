"""WL104I reviewer figures, researcher-readable version (presentation only; 2026-11-21).

Redraws the WL104I reviewer figures from the archived, hash-checked WL104I results (and, for the
lobe-level supporting panels only, the archived WL104H results). Nothing is recomputed from a model, nothing is
re-fitted or re-rendered; every plotted number is read from an archived file or is a direct arithmetic
re-expression of archived arrays, and is written to figures_readable_values.json together with the
consistency checks against the frozen decision / stage files.

The original package figures (package104i.py) stay in evaluation/104I unchanged for provenance.

Usage (from experiments/ecro_wl84_104i, any Python with numpy + matplotlib + torch for the WL104H helper import):
  python review104i_figures.py ../../evaluation/104I
"""
import argparse
import hashlib
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "ecro_wl84_104h"))
from package104h import bin_map, ssqrt_norm, sroot4_norm  # noqa: E402  (same helpers as the original figures)

RAW = REPO / "results/ecro_wl84_104i/primary"
H_EVAL = REPO / "evaluation/104H"
G_RAW = REPO / "results/ecro_wl84_104g"
MATERIALS = REPO / "evaluation/104D/amendment1/materials.json"
LUMA = np.array([0.2126, 0.7152, 0.0722])
ANGLES = np.array([5, 20, 35, 50, 57.5, 60, 62.5, 70, 80, 86], float)
POLAR = np.repeat(ANGLES, 2)          # WL104H/I probe layout: 10 polar angles x 2 azimuths

# ---------------------------------------------------------------- readable vocabulary (IDs go to metadata only)
C_HIST, C_ORIG, C_GF = "#8c8c8c", "#2f6db5", "#e07b1f"
ARM = {"A0": "Historical original fit (archived, WL83 recipe)",
       "A1": "Original ER, matched refit (64 parameters)",
       "B1": "ER-v1-GF, matched refit (58 parameters)"}
ARM_SHORT = {"A0": "historical fit", "A1": "original ER refit", "B1": "ER-v1-GF refit"}
SEEDS = ("104911", "104912", "104913")
START = {"generic": "reproduction start", "104911": "start 1", "104912": "start 2", "104913": "start 3"}
BANDS = ("near_normal", "mid", "band_55_65", "theta_70", "grazing", "all")
BAND = {"near_normal": "5–20°", "mid": "35–50°", "band_55_65": "55–65°", "theta_70": "70°",
        "grazing": "80–86°", "all": "all angles"}
COHORT = {"development": "development materials (3)", "held_out": "held-out materials (6)"}
SCENES = ("test83_iid_1_L2", "test83_hard_1_L1", "test83_iid_159_L2")
SCENE = {"test83_iid_1_L2": "material iid_1, light 2 (development)",
         "test83_hard_1_L1": "material hard_1, light 1 (development)",
         "test83_iid_159_L2": "material iid_159, light 2 (held out)"}
SCENE_SHORT = {"test83_iid_1_L2": "iid_1 scene", "test83_hard_1_L1": "hard_1 scene", "test83_iid_159_L2": "iid_159 scene (held out)"}
EVENT = {"e1": "first-bounce light", "e2": "second-bounce light", "total": "total light", "h": "third+ bounce light"}
TERM = {"O0": "Outer 1", "O1": "Outer 2", "O2": "Outer 3", "SL": "Lambert", "SI0": "Inner 1", "SI1": "Inner 2",
        "SI2": "Inner 3", "SI3": "Inner 4"}
GROUP = {"outer_amplitude": "outer-lobe\namplitudes", "split": "Lambert/inner\nsplit", "roughness": "lobe\nroughness",
         "anisotropy": "lobe\nanisotropy", "fresnel": "lobe\nFresnel F0", "attenuation": "attenuation\nknots", "eta": "coat\nindex η"}
MEASURED = "MEASURED"
CONCEPT = "CONCEPTUAL ILLUSTRATION — not measured data"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "figure.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False})


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def tag(fig, text, measured=True):
    fig.text(0.995, 0.995, text, ha="right", va="top", fontsize=8, weight="bold",
             color="#1b7a3a" if measured else "#b03030",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1b7a3a" if measured else "#b03030"))


def footnote(fig, text):
    """Wrapped footnote at the bottom; returns the figure-height fraction it occupies."""
    width = int(fig.get_figwidth() * 17.5)
    lines = []
    for para in text.split("\n\n"):
        lines += textwrap.wrap(" ".join(para.split()), width) or [""]
    fig.text(0.005, 0.004, "\n".join(lines), ha="left", va="bottom", fontsize=7, color="#555555", linespacing=1.3)
    return (len(lines) * 7 * 1.3 / 72 + 0.08) / fig.get_figheight()


def reserve(fig, frac, top=0.93):
    """Constrained-layout figures: keep axes clear of the footnote and the title line."""
    fig.get_layout_engine().set(rect=(0, frac + 0.01, 1, top - frac - 0.01))


# ================================================================ authentication
def authenticate(ev):
    """Every archived input must match the frozen WL104I / WL104H manifests before anything is drawn."""
    man = read(ev / "manifest.json")
    hman = read(H_EVAL / "manifest.json")
    checks = {}
    for name in ("decision.json", "angular.json", "fit.json", "ident.json", "posthoc_rank.json", "native_audit.json",
                 "equivalence.json", *[f"scene_{s}.json" for s in SCENES]):
        checks[f"evaluation/104I/{name}"] = sha(ev / name) == man["review_sha256"][name]
    for rel in ("angular/lanes.npz", *[f"scenes/{s}/maps.npz" for s in SCENES]):
        checks[f"{man['raw_root']}/{rel}"] = sha(RAW / rel) == man["raw_sha256"][rel]
    for name in ("summary.json", "posthoc_term_collinearity.json"):
        checks[f"evaluation/104H/{name}"] = sha(H_EVAL / name) == hman["review_sha256"][name]
    # region masks: the same files WL104I's scene stage hashed (scene analysis inputs_sha256)
    for s in SCENES:
        a = read(ev / f"scene_{s}.json")
        key = f"results/ecro_wl84_104g/scenes/{s}/region_masks.npz"
        checks[key] = sha(REPO / key) == a["inputs_sha256"][key]
    bad = [k for k, v in checks.items() if not v]
    if bad:
        raise SystemExit(f"authentication failed: {bad}")
    return checks


# ================================================================ figure 1: model structure (conceptual)
def fig_model_structure(out):
    fig = plt.figure(figsize=(13, 7.2))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.86]); ax.set_xlim(0, 13); ax.set_ylim(0, 7.2); ax.axis("off")
    fig.suptitle("What the four component families of the certified ER closure represent", x=0.01, ha="left", weight="bold")
    fig.text(0.01, 0.905, "F(ωi, ωo) = Outer₁ + Outer₂ + Outer₃  +  S(μi)·S(μo) · ( Lambert + Inner₁ + Inner₂ + Inner₃ + Inner₄ )"
             "      (per colour channel; F = BSDF × cos θo)", fontsize=10, family="DejaVu Sans")
    tag(fig, CONCEPT, measured=False)
    # layered slab sketch
    ax.add_patch(Rectangle((0.4, 1.0), 5.6, 1.6, fc="#dfe9f5", ec="none"))
    ax.add_patch(Rectangle((0.4, 0.55), 5.6, 0.45, fc="#b8b8b8", ec="none"))
    ax.plot([0.4, 6.0], [2.6, 2.6], color="#2f6db5", lw=2)
    ax.text(6.1, 2.6, "coat top surface", va="center", fontsize=8, color="#2f6db5")
    ax.text(6.1, 1.8, "clear coat\n(index η)", va="center", fontsize=8)
    ax.text(6.1, 0.78, "base\n(conductor / diffuse)", va="center", fontsize=8)
    arrow = lambda a, b, c, ls="-": ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=c, lw=1.6, ls=ls))
    arrow((0.9, 4.6), (2.4, 2.62), "k"); ax.text(0.6, 4.75, "incident light ωi", fontsize=8)
    # outer reflection
    arrow((2.4, 2.62), (3.6, 4.4), C_ORIG); arrow((2.4, 2.62), (3.3, 4.55), C_ORIG); arrow((2.4, 2.62), (3.9, 4.2), C_ORIG)
    ax.text(3.7, 4.55, "Outer GGX lobes\n(reflection at the coat)", color=C_ORIG, fontsize=8, weight="bold")
    # refraction into the coat, inner reflection, exit
    arrow((2.4, 2.62), (3.0, 1.02), "#7a4fb0", "--"); arrow((3.0, 1.02), (3.7, 2.58), "#7a4fb0", "--")
    arrow((3.7, 2.6), (5.0, 4.0), "#7a4fb0")
    ax.text(5.05, 4.0, "Inner GGX lobes\n(glossy reflection below the coat,\nseen through refraction)", color="#7a4fb0", fontsize=8, weight="bold")
    # lambert
    for dx in (-0.5, -0.2, 0.15, 0.45):
        arrow((1.7, 1.02), (1.7 + dx, 2.5), "#3a9a4a", ":")
    arrow((1.6, 2.62), (1.0, 3.5), "#3a9a4a")
    ax.text(0.45, 3.55, "Lambert term\n(diffuse light\nbelow the coat)", color="#3a9a4a", fontsize=8, weight="bold")
    ax.text(0.5, 0.2, "Attenuation S(μi)·S(μo): one learned transmission curve per channel (5 knots over μ = cos θ),\n"
            "applied once on the way in and once on the way out; it multiplies only the Lambert + Inner block.",
            fontsize=8, color="#a0522d")
    # right-hand definitions table
    rows = [
        ("Outer GGX (3 lobes)", C_ORIG,
         "Anisotropic GGX microfacet lobes with Schlick Fresnel F0, evaluated at the\nactual directions. Intended role: glossy reflection "
         "from the top surface."),
        ("Inner GGX (4 lobes)", "#7a4fb0",
         "GGX lobes evaluated at the refracted directions t(ωi), t(ωo) and scaled by\n1/η². Intended role: light that enters the coat, "
         "reflects below it, and exits."),
        ("Lambert (1 term)", "#3a9a4a",
         "Constant BSDF (cos θo / π). Intended role: diffuse sub-coat scattering."),
        ("Attenuation S(μ)", "#a0522d",
         "Per-channel transmission factor through the coat (piecewise-linear, 5 knots).\nIt is a multiplier, not an additive term."),
        ("Amplitudes λ", "#444444",
         "Per channel: how much energy each of the 8 terms carries (Σλ ≤ 1, passive).\nOriginal ER: softmax over 9 slots incl. an 'absorption' slot."),
    ]
    y = 6.55
    for name, col, desc in rows:
        ax.add_patch(FancyBboxPatch((7.35, y - 0.86), 5.5, 0.98, boxstyle="round,pad=0.04", fc="white", ec=col, lw=1.2))
        ax.text(7.5, y - 0.02, name, fontsize=9, weight="bold", color=col, va="top")
        ax.text(7.5, y - 0.3, desc, fontsize=7.6, va="top")
        y -= 1.12
    ax.text(7.35, 0.62, "These are fitting components of a learned closure. The intended roles above come from the\n"
            "definitions; WL104H measured that several terms overlap strongly (see figure 5), so a term is\n"
            "not guaranteed to correspond one-to-one to a physical light path.", fontsize=7.6, style="italic", va="top")
    f = out / "fig01_er_model_components_conceptual.png"
    fig.savefig(f, dpi=130); plt.close(fig)
    return f, {"source": "definitions in docs/plans/ER_V1_GF_GAUGE_FIXED_PARAMETERIZATION.md and WL104H term table; no data"}


# ================================================================ figure 2: measured energy share per component (WL104H)
def fig_energy_share(out):
    S = read(H_EVAL / "summary.json")["response"]
    groups = {"Outer GGX (3 lobes)": ("O0", "O1", "O2"), "Lambert × attenuation": ("SL",),
              "Inner GGX (4 lobes) × attenuation": ("SI0", "SI1", "SI2", "SI3")}
    gcol = [C_ORIG, "#3a9a4a", "#7a4fb0"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.8), layout="constrained", sharey=True)
    vals = {}
    for a, c in zip(axs, ("development", "held_out")):
        x = np.arange(len(BANDS)); bottom = np.zeros(len(BANDS))
        for (gname, terms), col in zip(groups.items(), gcol):
            v = np.array([sum(S[f"{c}|{b}|all"]["term_luma_share"][t] for t in terms) for b in BANDS])
            vals[f"{c}|{gname}"] = v.tolist()
            a.bar(x, v, 0.65, bottom=bottom, color=col, label=gname)
            for i in range(len(BANDS)):
                if v[i] > 0.06:
                    a.text(x[i], bottom[i] + v[i] / 2, f"{100 * v[i]:.0f}%", ha="center", va="center", color="white", fontsize=8)
            bottom += v
        a.set_xticks(x, [BAND[b] for b in BANDS]); a.set_xlabel("incidence angle θi of the probe")
        a.set_title(COHORT[c]); a.set_ylim(0, 1.0)
    axs[0].set_ylabel("share of the directional reflectance (luminance)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=False)
    fig.suptitle("How much reflected energy each component carries (historical original fit, 9 materials)", x=0.01, ha="left", weight="bold")
    tag(fig, MEASURED + " — WL104H, R0 state")
    fr = footnote(fig, "Source: WL104H response phase (evaluation/104H/summary.json, term_luma_share; region 'all'). Attenuation is not a separate "
                  "bar: it multiplies the Lambert and Inner terms, so their shares already include it. The same shares hold for ER-v1-GF "
                  "at an equivalent state (WL104I: every composed term equal to ≤ 6.3e-16).")
    reserve(fig, fr, top=0.88)
    f = out / "fig02_measured_energy_share_by_component.png"
    fig.savefig(f, dpi=130); plt.close(fig)
    return f, {"values": vals}


# ================================================================ figure 3: redundancies removed
def fig_redundancies(ev, out):
    idt = read(ev / "ident.json")["results"]
    ph = read(ev / "posthoc_rank.json")["results"]
    dec = read(ev / "decision.json")
    fig = plt.figure(figsize=(14, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], left=0.02, right=0.99, top=0.86, bottom=0.27, wspace=0.08)
    ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_title("A. Parameter count: what changes from 64 to 58  (conceptual accounting)", loc="left", fontsize=10)
    blocks = [("lobe shapes (roughness, anisotropy, F0)", 21, 21, "#9db7d9"),
              ("amplitudes per channel", 27, 21, "#e9b77a"),
              ("attenuation knots", 15, 15, "#d9a98a"),
              ("coat index η", 1, 1, "#c9c9c9")]
    y = 8.6
    for name, n0, n1, col in blocks:
        ax.add_patch(Rectangle((0.2, y), n0 / 64 * 4.2, 0.65, fc=col, ec="k", lw=0.5))
        ax.add_patch(Rectangle((5.3, y), n1 / 64 * 4.2, 0.65, fc=col, ec="k", lw=0.5))
        ax.text(0.2, y + 0.82, name, fontsize=8)
        ax.text(0.25 + n0 / 64 * 4.2, y + 0.3, f" {n0}", fontsize=8, va="center")
        ax.text(5.35 + n1 / 64 * 4.2, y + 0.3, f" {n1}", fontsize=8, va="center")
        y -= 1.45
    ax.text(0.2, 9.75, "Original ER: 64", weight="bold"); ax.text(5.3, 9.75, "ER-v1-GF: 58", weight="bold")
    ax.text(0.2, 2.9, "Removed exact redundancies (3 colour channels each):", weight="bold", fontsize=9)
    ax.text(0.2, 2.45, "1. Softmax shift — adding the same constant to all 9 amplitude logits of a channel\n"
                       "    changes nothing. ER-v1-GF drops the 'absorption' slot and uses log-ratio coordinates.", fontsize=8, va="top")
    ax.text(0.2, 1.35, "2. Attenuation-scale trade — knots s → κ·s together with Lambert/inner amplitudes\n"
                       "    λ → λ/κ² gives the same F (the absorption slot takes the slack). ER-v1-GF pins the\n"
                       "    Lambert/inner block to the remaining budget, so the knots carry the scale.", fontsize=8, va="top")
    ax.text(0.2, -0.05, "Response family: identical (144 states; float64 ≤ 1.1e-15 of peak, runtime ≤ 5.5e-7).", fontsize=8, style="italic")
    # B. measured singular values
    b = fig.add_subplot(gs[1])
    keys = [k for k in idt]
    xo, yo, xc, yc = [], [], [], []
    for i, k in enumerate(keys):
        so = np.sort(np.asarray(idt[k]["original"]["singular_relative"]))[:10]
        sc = np.sort(np.asarray(idt[k]["candidate"]["singular_relative"]))[:10]
        xo += [i - 0.18] * len(so); yo += list(so); xc += [i + 0.18] * len(sc); yc += list(sc)
    b.axhspan(1e-22, 1e-15, color="#f3dede", zorder=0)
    b.axhline(1e-15, color="#b03030", ls="--", lw=1)
    b.text(len(keys) - 0.5, 1.4e-15, "frozen 'exact null' cutoff 1e-15", color="#b03030", ha="right", va="bottom", fontsize=8)
    b.text(len(keys) - 0.5, 3e-22, "below cutoff = counted as exact redundancy", color="#b03030", ha="right", va="bottom", fontsize=7.5)
    b.scatter(xo, yo, s=10, color=C_ORIG, label="original ER (64): 10 smallest", zorder=3)
    b.scatter(xc, yc, s=10, color=C_GF, marker="s", label="ER-v1-GF (58): 10 smallest", zorder=3)
    b.set_yscale("log"); b.set_ylim(1e-22, 1e-3)
    lab = []
    for k in keys:
        st, m = k.split("/")
        lab.append(f"{m.replace('test83_', '')} ({ {'R0': 'hist.', 'A': 'A', 'C': 'C'}[st] })")
    b.set_xticks(range(len(keys)), lab, rotation=90, fontsize=6.5)
    b.set_ylabel("normalised singular value σ / σmax of the response Jacobian")
    b.set_title("B. Measured: the six exact redundancies appear in the original only  (27 fitted states, 9 materials × 3 fits)", loc="left", fontsize=10)
    b.legend(loc="upper left", fontsize=8)
    n_orig = [idt[k]["original"]["exact_nulls"] for k in keys]
    n_cand = [idt[k]["candidate"]["exact_nulls"] for k in keys]
    chain = max(max(v["chain_rule"].values()) for v in ph.values())
    fig.suptitle("Which parameter redundancies ER-v1-GF removes", x=0.01, ha="left", weight="bold")
    tag(fig, "A: CONCEPTUAL accounting   ·   B: " + MEASURED, measured=True)
    footnote(fig, f"Frozen rule result: exact-gauge removal = {dec['decision']['exact_gauge_removal']}. ER-v1-GF had {max(n_cand)} exact nulls in all 27 states; "
                  f"the original had 6 in 26 states and {min(n_orig)} in one (historical iid_159: one gauge direction at 1.04e-15, just above the cutoff). "
                  f"Post hoc: the original's gauge directions change the response by ≤ {dec['gauge_vector_response_max']:.1e}; tangent spaces of the two charts "
                  f"are identical by the chain rule (≤ {chain:.1e}).\nSpectra are coordinate dependent: only the count of exact nulls is compared, not conditioning. "
                  "x-axis: material (hist. = archived historical fit R0; A and C = the two WL104D comparison fits). Source: evaluation/104I/ident.json, posthoc_rank.json.")
    f = out / "fig03_parameter_redundancies_removed.png"
    fig.savefig(f, dpi=130); plt.close(fig)
    return f, {"exact_nulls_original": dict(zip(keys, n_orig)), "exact_nulls_candidate": dict(zip(keys, n_cand)), "chain_rule_max": chain}


# ================================================================ figure 4: unresolved dependencies
def fig_dependencies(ev, out):
    idt = read(ev / "ident.json")["results"]
    T = read(H_EVAL / "posthoc_term_collinearity.json")["results"]
    gnames = list(GROUP)
    r0 = [k for k in idt if k.startswith("R0/")]
    cnt = np.zeros((7, 7)); med = np.full((7, 7), np.nan)
    for i in range(7):
        for j in range(7):
            if i == j:
                continue
            key = f"{gnames[i]}|{gnames[j]}" if f"{gnames[i]}|{gnames[j]}" in idt[r0[0]]["candidate"]["max_principal_cosine"] else f"{gnames[j]}|{gnames[i]}"
            v = np.array([idt[k]["candidate"]["max_principal_cosine"][key] for k in r0])
            cnt[i, j] = (v >= 0.99).sum(); med[i, j] = np.median(v)
    fig = plt.figure(figsize=(15, 6.8))
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.99, top=0.83, bottom=0.27, wspace=0.25)
    a = fig.add_subplot(gs[0, 0])
    im = a.imshow(np.where(np.eye(7, dtype=bool), np.nan, cnt), cmap="Reds", vmin=0, vmax=9)
    for i in range(7):
        for j in range(7):
            if i != j:
                a.text(j, i, f"{int(cnt[i, j])}/9\n{med[i, j]:.3f}", ha="center", va="center", fontsize=7,
                       color="white" if cnt[i, j] >= 6 else "k")
    a.set_xticks(range(7), [GROUP[g].replace("\n", " ") for g in gnames], fontsize=7.5, rotation=40, ha="right"); a.set_yticks(range(7), [GROUP[g] for g in gnames], fontsize=7.5)
    a.set_title("A. Parameter groups that remain nearly interchangeable in ER-v1-GF\n"
                "cell: materials (of 9) with largest principal cosine ≥ 0.99, and the median cosine", loc="left", fontsize=9.5)
    plt.colorbar(im, ax=a, shrink=0.8, label="materials with cosine ≥ 0.99")
    # C: weak directions
    c = fig.add_subplot(gs[0, 1])
    keys = list(idt)
    wo = np.array([idt[k]["original"]["weak_1e-3"] for k in keys]); wc = np.array([idt[k]["candidate"]["weak_1e-3"] for k in keys])
    x = np.arange(len(keys))
    c.bar(x - 0.2, wo, 0.4, color=C_ORIG, label="original ER (exact redundancies not counted)")
    c.bar(x + 0.2, wc, 0.4, color=C_GF, label="ER-v1-GF")
    c.set_xticks(x, [f"{k.split('/')[1].replace('test83_', '')} ({ {'R0': 'hist.', 'A': 'A', 'C': 'C'}[k.split('/')[0]] })" for k in keys],
                 rotation=90, fontsize=6.5)
    c.set_ylabel("weak directions per fitted state\n(1e-15 ≤ σ/σmax < 1e-3)")
    c.set_title("B. Approximate (weak) directions: unchanged by ER-v1-GF", loc="left", fontsize=9.5)
    c.set_ylim(0, 52); c.legend(fontsize=7.5, loc="upper left", ncol=2)
    fig.suptitle("Parameter dependencies that ER-v1-GF does NOT resolve", x=0.01, ha="left", weight="bold")
    tag(fig, MEASURED + " — WL104I Stage C")
    footnote(fig, "These are approximate dependencies (near-degenerate, not exact). Removing the six exact gauges changes neither the tangent space nor "
                  "these overlaps (WL104I post hoc chain-rule identity). No experiment here tests whether a further reduction would help. "
                  "A: principal cosine between the column spaces of two parameter groups of the ER-v1-GF response Jacobian at the 9 historical fits "
                  "(1.000 = one group can imitate the other to first order). B: counts at all 27 states (hist. = historical fit R0; A, C = WL104D comparison fits). "
                  "Source: evaluation/104I/ident.json (candidate max_principal_cosine, weak_1e-3).")
    f = out / "fig04_unresolved_parameter_group_dependencies.png"
    fig.savefig(f, dpi=130); plt.close(fig)
    return f, {"group_count_ge_0.99": {f"{gnames[i]}|{gnames[j]}": int(cnt[i, j]) for i in range(7) for j in range(i + 1, 7)},
               "group_median_cos": {f"{gnames[i]}|{gnames[j]}": float(med[i, j]) for i in range(7) for j in range(i + 1, 7)},
               "weak_original": dict(zip(keys, wo.tolist())), "weak_candidate": dict(zip(keys, wc.tolist()))}


# ================================================================ figure 5: near-duplicate lobes (WL104H evidence, all materials)
WL104H_TABLE = {"test83_iid_1": {("O1", "O2"), ("SI0", "SI1")}, "test83_block_186": {("SI0", "SI1"), ("SI0", "SI2"), ("SI1", "SI2")},
                "test83_hard_76": {("SI0", "SI1")}, "test83_iid_191": {("O0", "O1")}}   # pairs printed in the WL104H worklog table


def fig_near_duplicates(out):
    T = read(H_EVAL / "posthoc_term_collinearity.json")["results"]
    rows = read(MATERIALS)
    fig, axs = plt.subplots(3, 3, figsize=(13, 13.5))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.12, hspace=0.45, wspace=0.05)
    pairs, extra = {}, {}
    for ax, r in zip(axs.flat, rows):
        m = r["tag"]; names = T[m]["names"]
        M = np.array(T[m]["median_over_probes"])
        im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1)
        for i in range(8):
            for j in range(8):
                if i == j:
                    continue
                if M[i, j] >= 0.95:
                    listed = (names[min(i, j)], names[max(i, j)]) in WL104H_TABLE.get(m, set())
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#d0021b", lw=2, ls="-" if listed else "--"))
                    if i < j:
                        pairs.setdefault(m, []).append([TERM[names[i]], TERM[names[j]], round(float(M[i, j]), 4)])
                        if not listed:
                            extra.setdefault(m, []).append([TERM[names[i]], TERM[names[j]], round(float(M[i, j]), 4)])
                if M[i, j] >= 0.9:
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white")
        ax.set_xticks(range(8), [TERM[t] for t in names], rotation=90, fontsize=7.5)
        ax.set_yticks(range(8), [TERM[t] for t in names], fontsize=7.5)
        ax.set_title(f"material {m.replace('test83_', '')} ({'development' if r['cohort'] == 'development' else 'held out'})", fontsize=9.5)
    cb = fig.colorbar(im, ax=axs, shrink=0.5, location="right", pad=0.01)
    cb.set_label("median cosine between the angular responses of two terms (1 = same shape)")
    fig.suptitle("Near-duplicate lobes: pairs of terms with almost the same angular response (not resolved by ER-v1-GF)", x=0.01, ha="left", weight="bold")
    fig.text(0.01, 0.935, "Red frame: cosine ≥ 0.95 (solid = pair listed in the WL104H worklog table; dashed = present in the archived WL104H data "
                          "but not listed in that table). Grey level: similarity of every pair of terms.", fontsize=9)
    tag(fig, MEASURED + " — WL104H (post hoc), historical fit")
    footnote(fig, "Similarity = per-probe cosine of the lane-weighted RGB responses of two terms, median over the 20 probe directions, at the historical original fit "
                  "(WL104H post hoc diagnostic). Cosine ignores amplitude, so a duplicated pair can include a term that carries little energy (e.g. Inner 4 carries "
                  "≈ 0.3 % of the development-cohort reflectance, figure 2). ER-v1-GF composes exactly the same terms (WL104I: ≤ 6.3e-16), so every overlap here "
                  "is present in both parameterizations. Which lobe of a duplicated pair 'owns' a shape is not identifiable from these data. "
                  "Source: evaluation/104H/posthoc_term_collinearity.json.")
    f = out / "fig05_unresolved_near_duplicate_lobes.png"
    fig.savefig(f, dpi=120); plt.close(fig)
    return f, {"pairs_ge_0.95": pairs, "pairs_not_in_wl104h_worklog_table": extra}


# ================================================================ figure 5: angular fidelity
def fig_angular(ev, out):
    S = read(ev / "angular.json")["summary"]
    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], left=0.06, right=0.99, top=0.87, bottom=0.13, hspace=0.45, wspace=0.15)
    vals = {}
    for n, c in enumerate(("development", "held_out")):
        a = fig.add_subplot(gs[0, n])
        x = np.arange(len(BANDS))
        for j, (arm, key, col) in enumerate((("A0", "A0_mean", C_HIST), ("A1", "A1_mean", C_ORIG), ("B1", "B1_mean", C_GF))):
            v = [S[c][b]["el1"][key] for b in BANDS]
            vals[f"{c}|{arm}"] = v
            a.bar(x + (j - 1) * 0.27, v, 0.27, color=col, label=ARM[arm])
        a.set_xticks(x, [BAND[b] for b in BANDS]); a.set_xlabel("incidence angle θi")
        a.set_ylabel("angular error el1\n(relative L1 vs layered reference)")
        a.set_title(f"A{n + 1}. Absolute angular error — {COHORT[c]}", loc="left", fontsize=9.5)
        if n == 0:
            a.legend(fontsize=7.5, loc="upper left")
    b = fig.add_subplot(gs[1, :])
    labels, pts, lo, hi = [], [], [], []
    for c in ("development", "held_out"):
        for bnd in BANDS:
            e = S[c][bnd]["el1"]
            labels.append(f"{BAND[bnd]}\n{'dev.' if c == 'development' else 'held out'}")
            pts.append(e["point"]); lo.append(e["lo"]); hi.append(e["hi"])
    pts, lo, hi = 100 * np.array(pts), 100 * np.array(lo), 100 * np.array(hi)
    x = np.arange(len(pts)) + np.repeat([0, 0.6], len(BANDS))
    b.axhspan(5, 9, color="#f6dcdc", zorder=0); b.axhline(5, color="#b03030", ls="--", lw=1)
    b.text(x[-1] + 0.4, 5.2, "+5 % non-inferiority margin\n(worse than this = regression)", color="#b03030", fontsize=8, ha="right", va="bottom")
    b.axhline(0, color="k", lw=0.6)
    b.errorbar(x, pts, yerr=[pts - lo, hi - pts], fmt="o", color=C_GF, ecolor="#555", capsize=3)
    for xi, p in zip(x, pts):
        b.text(xi + 0.12, p, f"{p:+.1f}%", fontsize=7, va="center")
    b.set_xticks(x, labels, fontsize=7.5); b.set_ylim(-9, 9)
    b.set_ylabel("change of angular error, ER-v1-GF vs original refit\n(B − A)/A in %, 95 % CI")
    b.set_title("B. Paired comparison of the two matched refits (same physical starts, same data, same optimiser settings)", loc="left", fontsize=9.5)
    vals["relative_change_percent"] = dict(zip([lab.replace("\n", " ") for lab in labels], pts.tolist()))
    fig.suptitle("Angular fidelity: original ER vs ER-v1-GF after matched retraining", x=0.01, ha="left", weight="bold")
    tag(fig, MEASURED + " — WL104I Stage E")
    footnote(fig, "Physical reference: WL104D Amendment 1 independent layered-BSDF reference (two independent random-walk halves; half-split el1 noise 3.6e-4). "
                  "el1 = albedo-normalised L1 error over 20 probe directions × 1,024 lanes per material.\n"
                  "A and B: mean over materials × 3 training starts. CI: paired bootstrap over (material, start). The predeclared question was preservation "
                  "(every upper bound ≤ +5 %): PASS. Negative values are NOT claimed as improvement — with 3 starts × 9 materials\n"
                  "the worklog cannot separate an optimiser-path effect of the new coordinates from chance. Source: evaluation/104I/angular.json.")
    f = out / "fig06_angular_fidelity_original_vs_gf.png"
    fig.savefig(f, dpi=130); plt.close(fig)
    return f, vals


# ================================================================ figure 6: angular residual maps
def fig_angular_maps(out):
    rows = read(MATERIALS)
    with np.load(RAW / "angular/lanes.npz") as z:
        L = {k: z[k] for k in z.files}
    s = SEEDS[0]
    probes = [2, 10, 18]
    sets, M = {}, 0.0
    for m, r in enumerate(rows):
        if r["cohort"] != "development":
            continue
        sl = slice(m * 20, (m + 1) * 20)
        w = L["w"][sl]
        ref = 0.5 * (L["ref_a"][sl] + L["ref_b"][sl])
        A = (w[..., None] * ref).sum((1, 2))
        for p in probes:
            g = lambda v: bin_map(L["ref_wi"][sl][p, 0].astype(np.float64), L["ref_wo"][sl][p].astype(np.float64), (v[p] @ LUMA) / A[p])
            mp = {"ref - A1": g(ref - L[f"A1_{s}"][sl]), "ref - B1": g(ref - L[f"B1_{s}"][sl]), "B1 - A1": g(L[f"B1_{s}"][sl] - L[f"A1_{s}"][sl])}
            sets[(r["tag"], p)] = mp
            M = max(M, max(np.nanmax(np.abs(v)) for v in mp.values()))
    tags = [r["tag"] for r in rows if r["cohort"] == "development"]
    col_title = {"ref - A1": "reference − original ER refit", "ref - B1": "reference − ER-v1-GF refit", "B1 - A1": "ER-v1-GF − original ER"}
    fig, ax = plt.subplots(len(probes), 3 * len(tags), figsize=(18, 7.2), layout="constrained")
    for ti, t in enumerate(tags):
        for pi, p in enumerate(probes):
            for j, (k, v) in enumerate(sets[(t, p)].items()):
                a = ax[pi, 3 * ti + j]
                im = a.imshow(v, origin="lower", extent=[-45, 45, -45, 45], cmap="RdBu_r", norm=ssqrt_norm(M))
                a.set_xticks([-40, 0, 40]); a.set_yticks([-40, 0, 40]); a.tick_params(labelsize=6)
                if pi == 0:
                    a.set_title(f"{t.replace('test83_', 'material ')}\n{col_title[k]}", fontsize=8)
                if j == 0:
                    a.set_ylabel(f"θi = {POLAR[p]:g}°\noffset from mirror (deg)", fontsize=7.5)
                if pi == len(probes) - 1:
                    a.set_xlabel("offset from mirror (deg)", fontsize=7)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, location="right")
    cb.set_label("reflected luminance error / probe albedo\n(signed square-root scale)", fontsize=8)
    fig.suptitle("Where the angular error sits: mirror-centred error maps, development materials (first training start)", x=0.01, ha="left", weight="bold")
    tag(fig, MEASURED + " — WL104I Stage E lanes")
    fr = footnote(fig, f"Red = model too dark, blue = model too bright (reference − model). Shared colour scale ±{M:.4f} for every panel (same value as the original figure). "
                  "Reference: WL104D independent layered-BSDF reference (mean of both halves). Third column of each block: difference between the two refits. "
                  "Source: results/ecro_wl84_104i/primary/angular/lanes.npz.")
    reserve(fig, fr, top=0.95)
    f = out / "fig07_angular_error_maps_original_vs_gf.png"
    fig.savefig(f, dpi=120); plt.close(fig)
    return f, {"shared_scale": float(M)}


# ================================================================ figure 7: scene fidelity
def fig_scene_forest(ev, out):
    sc = {s: read(ev / f"scene_{s}.json") for s in SCENES}
    fig = plt.figure(figsize=(14.5, 8.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], left=0.2, right=0.99, top=0.84, bottom=0.24, wspace=0.3)
    a = fig.add_subplot(gs[0])
    rows, vals = [], {}
    for s in SCENES:
        for ev_ in ("e1", "e2", "total"):
            rows.append((s, ev_))
    y = np.arange(len(rows))[::-1].astype(float)
    a.axvspan(5, 25, color="#f6dcdc", zorder=0); a.axvline(5, color="#b03030", ls="--", lw=1); a.axvline(0, color="k", lw=0.6)
    for yi, (s, ev_) in zip(y, rows):
        v = sc[s]["primary_fixed"][ev_]
        p, lo, hi = 100 * v["point"], 100 * v["lo"], 100 * v["hi"]
        a.errorbar([p], [yi], xerr=[[p - lo], [hi - p]], fmt="D", color=C_GF, ecolor="k", capsize=3, zorder=3)
        for k, ps in enumerate(v["per_seed_rel"]):
            a.scatter(100 * ps, yi - 0.22, s=14, marker=["o", "s", "^"][k], color="#6b6b6b", zorder=2,
                      label=["start 1", "start 2", "start 3"][k] if (s, ev_) == rows[0] else None)
        vals[f"{s}|{ev_}"] = {"mean": p, "lo": lo, "hi": hi, "per_start": [100 * q for q in v["per_seed_rel"]]}
    a.set_yticks(y, [f"{SCENE_SHORT[s]} · {EVENT[e]}" for s, e in rows], fontsize=8)
    a.set_xlabel("change of object RMS error, ER-v1-GF vs original refit, (B − A)/A in %")
    a.set_xlim(-22, 8)
    a.set_title("A. Paired scene comparison (fixed historical sampling)\n◆ change of the 3-start mean RMS, 95 % CI;  grey = each start alone", loc="left", fontsize=9.5)
    a.legend(fontsize=7.5, loc="lower left")
    a.text(5.3, y[0] + 0.3, "+5 % margin", color="#b03030", fontsize=8)
    # B: absolute RMS level
    b = fig.add_subplot(gs[1])
    for yi, (s, ev_) in zip(y, rows):
        v = sc[s]["primary_fixed"][ev_]
        a0 = sc[s]["fixed"]["A0"]["objects"]["rms"][ev_]["point"]
        b.scatter([a0], [yi], color=C_HIST, marker="x", s=22, label=ARM_SHORT["A0"] if (s, ev_) == rows[0] else None)
        b.scatter([v["A1_rms"]], [yi + 0.15], color=C_ORIG, s=22, label=ARM_SHORT["A1"] if (s, ev_) == rows[0] else None)
        b.scatter([v["B1_rms"]], [yi - 0.15], color=C_GF, marker="s", s=22, label=ARM_SHORT["B1"] if (s, ev_) == rows[0] else None)
    b.set_yticks(y, []); b.set_xscale("log")
    b.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    b.set_xticks([0.003, 0.01, 0.03, 0.06], ["0.003", "0.01", "0.03", "0.06"])
    b.set_xlabel("object RMS error vs layered reference\n(stable RGB RMS, radiance units; mean over 3 starts)")
    b.set_title("B. Absolute error level", loc="left", fontsize=9.5)
    b.legend(fontsize=7.5, loc="lower left")
    # C: native (unqualified)
    c = fig.add_subplot(gs[2])
    c.axvline(0, color="k", lw=0.6); c.axvline(5, color="#b03030", ls="--", lw=1)
    for yi, (s, ev_) in zip(y, rows):
        v = sc[s]["secondary_native"].get(ev_)
        if v is None:
            continue
        p, lo, hi = 100 * v["point"], 100 * v["lo"], 100 * v["hi"]
        c.errorbar([p], [yi], xerr=[[p - lo], [hi - p]], fmt="o", color="#b5b5b5", ecolor="#b5b5b5", capsize=3)
    c.set_yticks(y, []); c.set_xlim(-22, 8)
    c.set_xlabel("(B − A)/A in %")
    c.set_title("C. Each model's own sampling\n(secondary; UNQUALIFIED — native audit\nfailed for both arms)", loc="left", fontsize=9.5, color="#777")
    fig.suptitle("Scene fidelity: original ER vs ER-v1-GF in three rendered scenes", x=0.01, ha="left", weight="bold")
    tag(fig, MEASURED + " — WL104I Stage E scenes")
    footnote(fig, "Physical reference: the same scenes rendered with the layered-BSDF random walk (depth cap 1024), same tracer and camera (WL104G). "
                  "Fixed historical sampling: every arm is evaluated at the identical light paths drawn by the historical fit's sampler, so only the\n"
                  "material response differs. First-bounce = light reflected once at the visible object; second-bounce = at the next hit; total = first + second + "
                  "archived third+ bounce light of the historical fit. 12 replicates; CI: delete-one-replicate jackknife,\n"
                  "conditional on the 3 realised training starts (so it does not include start-to-start variation; see the grey points). "
                  "Preservation (all upper bounds ≤ +5 %): PASS. Per-start changes differ widely (second-bounce light, iid_1: "
                  + " / ".join(f"{q:+.1f}" for q in vals["test83_iid_1_L2|e2"]["per_start"]) + " %; iid_159: "
                  + " / ".join(f"{q:+.1f}" for q in vals["test83_iid_159_L2|e2"]["per_start"]) + " % for starts 1 / 2 / 3).\n"
                  "Sources: evaluation/104I/scene_*.json (primary_fixed, secondary_native, fixed.A0).")
    f = out / "fig08_scene_fidelity_original_vs_gf.png"
    fig.savefig(f, dpi=130); plt.close(fig)
    return f, vals


# ================================================================ figure 8: scene error maps
def fig_scene_maps(out):
    maps, boxes, M = {}, {}, 0.0
    for s in SCENES:
        with np.load(RAW / "scenes" / s / "maps.npz") as z:
            maps[s] = {k: z[k] @ LUMA for k in z.files}
        with np.load(G_RAW / "scenes" / s / "region_masks.npz") as z:
            obj = z["objects"]
        ys, xs = np.nonzero(obj)
        boxes[s] = (max(ys.min() - 6, 0), min(ys.max() + 7, 256), max(xs.min() - 6, 0), min(xs.max() + 7, 256))
        M = max(M, max(float(np.abs(v).max()) for v in maps[s].values()))
    s0 = SEEDS[0]
    cols = (("A0_{}_err", "historical original fit − reference"), (f"A1_{s0}" + "_{}_err", "original ER refit − reference"),
            (f"B1_{s0}" + "_{}_err", "ER-v1-GF refit − reference"), ("B1_minus_A1_{}", "ER-v1-GF − original ER\n(mean of 3 starts)"))
    fig, ax = plt.subplots(len(SCENES) * 2, 4, figsize=(14, 12.5), layout="constrained")
    for i, s in enumerate(SCENES):
        y0, y1, x0, x1 = boxes[s]
        for e, ev_ in enumerate(("e1", "e2")):
            r = 2 * i + e
            for j, (c, title) in enumerate(cols):
                im = ax[r, j].imshow(maps[s][c.format(ev_)][y0:y1, x0:x1], cmap="RdBu_r", norm=sroot4_norm(M))
                ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
                if r == 0:
                    ax[r, j].set_title(title, fontsize=10)
            ax[r, 0].set_ylabel(f"{SCENE_SHORT[s]}\n{EVENT[ev_]}", fontsize=9)
    cb = fig.colorbar(im, ax=ax, shrink=0.35, location="right")
    cb.set_label("signed luminance error (radiance units; signed fourth-root scale)", fontsize=9)
    fig.suptitle("Scene error maps with fixed historical sampling (mean of 12 replicates; columns 2–3 = first training start)",
                 x=0.01, ha="left", weight="bold")
    tag(fig, MEASURED + " — WL104I Stage E scenes")
    fr = footnote(fig, f"Red = brighter than the reference, blue = darker. Shared scale ±{M:.3f} for every panel (same value as the original figure). Reference: layered-BSDF "
                  "random-walk render of the same scene (WL104G). Columns 2–3 show one training start;\nthe last column averages the three starts. Source: "
                  "results/ecro_wl84_104i/primary/scenes/*/maps.npz.")
    reserve(fig, fr, top=0.965)
    f = out / "fig09_scene_error_maps_original_vs_gf.png"
    fig.savefig(f, dpi=110); plt.close(fig)
    return f, {"shared_scale": float(M)}


# ================================================================ figure 9: training variability
def fig_training(ev, out):
    fit = read(ev / "fit.json"); ang = read(ev / "angular.json")["summary"]; dec = read(ev / "decision.json")["practical"]
    rows = read(MATERIALS)
    keys = fit["start_keys"]; steps = np.asarray(fit["arms"]["A1"]["history_steps"])
    fig = plt.figure(figsize=(15, 9.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], left=0.06, right=0.99, top=0.88, bottom=0.12, hspace=0.42, wspace=0.28)
    show = [("test83_iid_159", "iid_159 (held out)"), ("test83_block_186", "block_186 (held out)"), ("test83_iid_1", "iid_1 (development)")]
    tags = [r["tag"] for r in rows]
    for n, (m, lab) in enumerate(show):
        a = fig.add_subplot(gs[0, n]); mi = tags.index(m)
        for arm, col in (("A1", C_ORIG), ("B1", C_GF)):
            H = np.asarray(fit["arms"][arm]["history"])
            for i, k in enumerate(keys):
                a.plot(steps, H[:, i, mi], color=col, lw=1.6 if k == "generic" else 1.0, ls="--" if k == "generic" else "-",
                       alpha=0.85, label=(f"{ARM_SHORT[arm]}, {START[k] if k == 'generic' else 'starts 1–3'}" if k in ("generic", "104911") else None) if n == 0 else None)
        a.set_yscale("log"); a.set_xlim(0, 12000)
        lo_ = min(np.asarray(fit["arms"][x]["history"])[-1, :, mi].min() for x in ("A1", "B1"))
        a.set_ylim(lo_ * 0.92, lo_ * 1.6)
        a.set_title(f"A{n + 1}. Training objective — material {lab}", loc="left", fontsize=9.5)
        a.set_xlabel("optimiser step (Adam)"); a.set_ylabel("training objective (el1 + albedo error)")
    fig.axes[0].legend(fontsize=7, loc="upper right")
    # B: final objective ratios
    b = fig.add_subplot(gs[1, :2])
    fa, fb = np.asarray(fit["arms"]["A1"]["final_objective"]), np.asarray(fit["arms"]["B1"]["final_objective"])
    x = np.arange(len(tags))
    for k in range(1, 4):
        b.scatter(x + (k - 2) * 0.12, fb[k] / fa[k], s=18, marker=["o", "s", "^"][k - 1], color="#6b6b6b", label=f"{START[keys[k]]}")
    b.scatter(x, dec["final_objective_ratio_per_material"], s=70, marker="_", color=C_GF, lw=2.5, label="per-material ratio used by the decision rule")
    b.axhline(1, color="k", lw=0.6); b.axhline(0.99, color="#1b7a3a", ls="--", lw=1)
    b.text(len(tags) - 0.6, 0.9905, "predeclared benefit: ≤ 0.99 in ≥ 6 of 9 materials", color="#1b7a3a", fontsize=8, ha="right", va="bottom")
    b.set_xticks(x, [f"{t.replace('test83_', '')}\n({'dev.' if r['cohort'] == 'development' else 'held out'})" for t, r in zip(tags, rows)], fontsize=7.5)
    b.set_ylabel("final training objective\nER-v1-GF / original ER")
    b.set_title(f"B. Final objective after 12,000 steps (median over materials {dec['final_objective_ratio_median']:.4f}; "
                f"wall time ratio {dec['fit_wall_time_ratio']:.2f})", loc="left", fontsize=9.5)
    b.legend(fontsize=7.5, loc="lower left", ncol=2)
    # C: response spread
    c = fig.add_subplot(gs[1, 2])
    for i, coh in enumerate(("development", "held_out")):
        sp = ang[coh]["response_spread"]
        r = np.asarray(sp["B1"]) / np.asarray(sp["A1"])
        c.scatter(np.full(len(r), i) + np.linspace(-0.12, 0.12, len(r)), r, color="#6b6b6b", s=16,
                  label="one material" if i == 0 else None)
        c.scatter([i], [dec["spread_ratio_B_over_A"][coh]], marker="_", s=600, color=C_GF, lw=2.5,
                  label="median (decision value)" if i == 0 else None)
    c.axhline(1, color="k", lw=0.6); c.axhline(0.8, color="#1b7a3a", ls="--", lw=1)
    c.text(-0.45, 0.72, "predeclared benefit: ≤ 0.8 in both cohorts", color="#1b7a3a", fontsize=8, ha="left")
    c.set_yscale("log"); c.set_ylim(0.08, 10); c.set_xticks([0, 1], ["development\n(3 materials)", "held out\n(6 materials)"]); c.set_xlim(-0.5, 1.5)
    c.set_ylabel("spread across the 3 starts\nER-v1-GF / original ER")
    c.set_title("C. Start-to-start variability of the fitted response", loc="left", fontsize=9.5)
    c.legend(fontsize=7.5, loc="upper center", ncol=2)
    fig.suptitle("Training variability: same physical starts, different coordinates — no demonstrated practical benefit", x=0.01, ha="left", weight="bold")
    tag(fig, MEASURED + " — WL104I Stage D")
    footnote(fig, "Both arms: identical physical starting responses (reproduction start + 3 perturbed starts), Adam lr 0.03 → 3e-4 cosine, 12,000 steps, float32, deterministic. "
                  "Adam is coordinate dependent, so identical settings on different coordinates are not an identical optimiser:\ntrajectories coincide at first and then "
                  "separate on some starts (A1, A2). None of the predeclared benefit criteria (spread ≤ 0.8 in both cohorts, objective ≤ 0.99 in ≥ 6/9 materials, wall time ≤ 0.9) "
                  "was met. The development spread ratio (median 0.18) rests on 3 materials whose ratios are "
                  + " / ".join(f"{q:.2f}" for q in np.asarray(ang["development"]["response_spread"]["B1"]) / np.asarray(ang["development"]["response_spread"]["A1"]))
                  + " — one of them is more variable under ER-v1-GF; with 3 starts this cannot be separated from chance. "
                  "Sources: evaluation/104I/fit.json, angular.json (response_spread), decision.json (practical).")
    f = out / "fig10_training_variability_and_benefit.png"
    fig.savefig(f, dpi=130); plt.close(fig)
    return f, {"objective_ratio_per_start": (fb[1:] / fa[1:]).tolist(), "spread": {c: ang[c]["response_spread"] for c in ("development", "held_out")}}


# ================================================================ consistency checks against frozen files
def checks(ev, values):
    dec = read(ev / "decision.json"); ang = read(ev / "angular.json")["summary"]; man = read(ev / "manifest.json")
    fit = read(ev / "fit.json")
    out = {}
    for c, k in (("development", "angular_el1_development"), ("held_out", "angular_el1_held_out")):
        e, d = ang[c]["all"]["el1"], dec["primary_metrics"][k]
        out[f"angular_all_{c}_equals_decision"] = all(abs(e[x] - d[x]) == 0 for x in ("point", "lo", "hi", "A1_mean", "B1_mean", "A0_mean"))
    for s in SCENES:
        a = read(ev / f"scene_{s}.json")
        for e in ("e1", "e2", "total"):
            v, d = a["primary_fixed"][e], dec["primary_metrics"][f"scene_{s}_{e}"]
            out[f"scene_{s}_{e}_equals_decision"] = all(v[x] == d[x] for x in ("point", "lo", "hi", "A1_rms", "B1_rms"))
            out[f"scene_{s}_{e}_point_is_change_of_start_mean_rms"] = abs(v["B1_rms"] / v["A1_rms"] - 1 - v["point"]) < 1e-12
            out[f"scene_{s}_{e}_upper_bound_le_5pct"] = v["hi"] <= 0.05
    out["angular_scale_equals_original_figure"] = abs(values["fig07"]["shared_scale"] - man["display_scales"]["angular_signed_max"]) < 1e-12
    out["scene_scale_equals_original_figure"] = abs(values["fig09"]["shared_scale"] - man["display_scales"]["scene_signed_max"]) < 1e-12
    for c in ("development", "held_out"):
        sp = ang[c]["response_spread"]
        out[f"spread_median_{c}_equals_decision"] = bool(abs(np.median(np.asarray(sp["B1"]) / np.asarray(sp["A1"])) - dec["practical"]["spread_ratio_B_over_A"][c]) < 1e-12)
    fa, fb = np.asarray(fit["arms"]["A1"]["final_objective"]), np.asarray(fit["arms"]["B1"]["final_objective"])
    r = np.median(fb[1:], 0) / np.median(fa[1:], 0)
    out["objective_ratio_is_ratio_of_start_medians"] = bool(np.allclose(r, dec["practical"]["final_objective_ratio_per_material"], rtol=0, atol=1e-12))
    out["objective_ratio_median_equals_decision"] = abs(np.median(r) - dec["practical"]["final_objective_ratio_median"]) < 1e-12
    idt = read(ev / "ident.json")["results"]
    out["exact_null_counts_equal_decision"] = ([idt[k]["original"]["exact_nulls"] for k in idt] == dec["exact_nulls"]["original"]
                                                and [idt[k]["candidate"]["exact_nulls"] for k in idt] == dec["exact_nulls"]["candidate"])
    S = read(H_EVAL / "summary.json")["response"]
    out["wl104h_term_shares_sum_to_1"] = all(abs(sum(S[f"{c}|{b}|all"]["term_luma_share"].values()) - 1) < 1e-9
                                             for c in ("development", "held_out") for b in BANDS)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("review", type=Path)
    args = ap.parse_args()
    ev = args.review.resolve()
    auth = authenticate(ev)
    values, files = {}, {}
    for key, fn in (("fig01", lambda: fig_model_structure(ev)), ("fig02", lambda: fig_energy_share(ev)),
                    ("fig03", lambda: fig_redundancies(ev, ev)), ("fig04", lambda: fig_dependencies(ev, ev)),
                    ("fig05", lambda: fig_near_duplicates(ev)),
                    ("fig06", lambda: fig_angular(ev, ev)), ("fig07", lambda: fig_angular_maps(ev)),
                    ("fig08", lambda: fig_scene_forest(ev, ev)), ("fig09", lambda: fig_scene_maps(ev)),
                    ("fig10", lambda: fig_training(ev, ev))):
        f, v = fn()
        files[key] = f; values[key] = v
        print("wrote", f.name, flush=True)
    chk = {k: bool(v) for k, v in checks(ev, values).items()}
    meta = {
        "batch": "WL104I", "purpose": "researcher-readable reviewer figures (presentation only; no new experiment, no change of any measurement or verdict)",
        "date": "2026-11-21", "script": "experiments/ecro_wl84_104i/review104i_figures.py", "script_sha256": sha(Path(__file__)),
        "inputs_authenticated_against_frozen_manifests": auth,
        "consistency_checks": chk, "all_checks_pass": all(chk.values()),
        "arm_ids": {"A0": "historical R0 (WL83 direct fit; archived)", "A1": "original 64-parameter matched refit", "B1": "ER-v1-GF 58-parameter matched refit"},
        "start_ids": {"reproduction start": "generic WL82 init", "start 1": "seed 104911", "start 2": "seed 104912", "start 3": "seed 104913"},
        "scene_ids": {SCENE_SHORT[s]: s for s in SCENES},
        "figures": {k: {"file": v.name, "sha256": sha(v)} for k, v in files.items()},
        "values": values,
        "original_figures_kept_for_provenance": ["training_histories.png", "jacobian_spectra.png", "angular_el1_bands.png",
                                                 "angular_residual_maps.png", "scene_relative_rms.png", "scene_error_maps.png"],
    }
    (ev / "figures_readable_manifest.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("checks:", "ALL PASS" if meta["all_checks_pass"] else {k: v for k, v in chk.items() if not v}, flush=True)


if __name__ == "__main__":
    main()
