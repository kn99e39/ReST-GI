"""WL104K-B reviewer package (target evaluation/104K-B): readable figures, index.html and figures_manifest.json.

Usage: python figures104kb.py <run_root> <out_dir> [--stageA <stageA.json>] [--scenes-root <dir>]

Reads only (never writes inside <run_root>):
  <run_root>/test_eval/test.json + arrays.npz      independent final angular test (eval104kb run_test schema)
  <run_root>/dev_eval/dev.json + arrays.npz        dev-only decisions (E1 selection criterion, stage-F eligibility)
  <run_root>/fits/<arm>/fit.json                   training histories, termination, final objectives
  <run_root>/decision.json                         optional (decision104kb); else the table is built from test.json labels
  <run_root>/amendment1/{result.json, test_amended.json, decision_amendment1.json}   Amendment 1 (optional)
  <run_root>/scenes/<scene>/analysis104kb.json + maps.npz   optional (scenes104kb analyze schema); skipped when absent
  <run_root>/../stageA/stageA.json                 Stage A diagnostic on archived WL104K hard_1 P fits (stageA104kb schema)
Every figure is produced from stored numbers; nothing is re-measured. Figures whose inputs are missing are skipped and listed
in figures_manifest.json 'skipped'.
"""
import argparse
import hashlib
import html
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402

SCENES = ("test83_iid_1_L2", "test83_hard_1_L1", "test83_iid_159_L2", "test83_hard_53_L2", "test83_block_186_L2")
INDEPENDENT = ("test83_iid_159_L2", "test83_hard_53_L2", "test83_block_186_L2")
SCENE_NAME = {"test83_iid_1_L2": "iid_1 scene (development camera)",
              "test83_hard_1_L1": "hard_1 scene (development regression case)",
              "test83_iid_159_L2": "iid_159 scene (held-out camera, independent)",
              "test83_hard_53_L2": "hard_53 scene (new N1, independent)",
              "test83_block_186_L2": "block_186 scene (new N2, independent)"}
ARM = {"O0": "original Adam fit (O0)", "O1": "L-BFGS (O1)", "D": "refined angular coverage (D)",
       "E2": "core+shoulder objective (E2)", "E1": "selected start (E1)", "F": "combined mechanisms (F)",
       "A_hist": "certified ER baseline (A, WL104K)", "P_hist": "ER-v1-SDM WL104K fit (P)",
       "S_O0": "synthetic: original Adam fit (S-O0)", "S_O1": "synthetic: L-BFGS (S-O1)",
       "S_D": "synthetic: refined angular coverage (S-D)", "S_E2": "synthetic: core+shoulder objective (S-E2)"}
# dataviz reference categorical palette in fixed slot order (blue, orange, aqua, yellow, magenta, green, violet, red);
# the control O0 is neutral ink. Markers are the secondary (non-colour) encoding.
COL = {"O0": "#3c3c3c", "D": "#2a78d6", "O1": "#eb6834", "E2": "#1baf7a", "A_hist": "#eda100", "E1": "#e87ba4",
       "P_hist": "#6250d6", "F": "#e34948"}
for _s in ("O0", "O1", "D", "E2"):
    COL["S_" + _s] = COL[_s]
MK = {"O0": "o", "O1": "s", "D": "^", "E2": "D", "E1": "*", "P_hist": "v", "A_hist": "P", "F": "X"}
for _s in ("O0", "O1", "D", "E2"):
    MK["S_" + _s] = MK[_s]
TAGS = {"PRE": ("PREDECLARED RULE", "#2b6cb0"), "DESC": ("DESCRIPTIVE", "#6b6b6b"),
        "DIAG": ("DIAGNOSTIC: pre-freeze, archived WL104K states", "#8a5a00"), "SEC": ("SECONDARY: repeatedly inspected", "#6b6b6b")}
MAT_NAME = {"test83_iid_1": "iid_1", "test83_iid_0": "iid_0", "test83_hard_1": "hard_1", "test83_iid_191": "iid_191",
            "test83_iid_159": "iid_159", "test83_block_161": "block_161", "test83_block_186": "block_186",
            "test83_hard_76": "hard_76", "test83_hard_53": "hard_53"}
PCT = 100.0


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None


def tag(fig, kind):
    t, c = TAGS[kind]
    fig.text(0.995, 0.995, t, ha="right", va="top", fontsize=8, color="white", bbox=dict(boxstyle="round,pad=0.25", fc=c, ec="none"))


def foot(fig, t, y=0.006):
    w = int(fig.get_figwidth() * 15.5)
    fig.text(0.01, y, "\n".join(textwrap.fill(p, w) for p in t.split("\n")), fontsize=7.6, va="bottom", color="#222")


def title(fig, t, sub=None, warn=None):
    """Header in fixed inches from the top (independent of figure height); records its depth for Pack.save."""
    H = fig.get_figheight()
    tl = textwrap.wrap(t, max(40, int(fig.get_figwidth() * 9.6) - 16))   # bold 11.5 pt, keep clear of the right-hand tag
    fig.suptitle("\n".join(tl), x=0.01, y=1 - 0.06 / H, ha="left", va="top", weight="bold", fontsize=11.5, linespacing=1.1)
    d = 0.36 + 0.2 * (len(tl) - 1)
    if sub:
        fig.text(0.01, 1 - d / H, sub, fontsize=8.8, color="#333", va="top")
        d += 0.2
    if warn:
        lines = textwrap.wrap(warn, int(fig.get_figwidth() * 12.5))     # bold 8.6 pt: ~12.5 characters per inch
        fig.text(0.01, 1 - d / H, "\n".join(lines), fontsize=8.6, color="#b00020", va="top", weight="bold", linespacing=1.15)
        d += 0.2 + 0.15 * (len(lines) - 1)
    fig._hdr_in = d


def symlog_pct(ax, lin=20):
    ax.set_yscale("symlog", linthresh=lin, linscale=1.0)
    ax.axhline(0, color="k", lw=0.8)


def jitter(k, n, w=0.5):
    return (np.arange(n) - (n - 1) / 2) * (w / max(n, 1)) if n > 1 else np.zeros(1)


class Pack:
    def __init__(self, root, out, stageA=None, scenes_root=None, decision=None, decision_amended=None):
        self.root, self.out = root, out
        self.T = read(root / "test_eval/test.json")
        self.Dv = read(root / "dev_eval/dev.json")
        self.DC = read(decision or (root / "decision.json"))
        self.DCa = read(decision_amended or (root / "amendment1/decision_amendment1.json"))
        self.AM = read(root / "amendment1/result.json")
        self.TA = read(root / "amendment1/test_amended.json")
        self.SA = read(stageA or (root.parent / "stageA/stageA.json"))
        self.z = dict(np.load(root / "test_eval/arrays.npz")) if (root / "test_eval/arrays.npz").exists() else {}
        self.fits = {a: read(root / "fits" / a / "fit.json") for a in ("O0", "O1", "D", "E2", "F", "S_O0", "S_O1", "S_D", "S_E2")}
        self.fits = {a: f for a, f in self.fits.items() if f is not None}
        self.mats = [MAT_NAME.get(m, m) for m in (self.T or self.Dv)["materials"]]
        coh = (self.T or self.Dv)["cohorts"]
        self.cohort = ["development" if i in coh["development"] else "held out" for i in range(len(self.mats))]
        self.n = len(self.mats)
        self.arms = [a for a in ("O0", "O1", "D", "E2", "F") if f"test__{a}__e_prof" in self.z]
        sroot = scenes_root or (root / "scenes")
        self.scenes = {}
        for s in SCENES:
            for d in (sroot / s, root / "scenes_new" / s):
                if (d / "analysis104kb.json").exists():
                    a = read(d / "analysis104kb.json")
                    if a.get("contrasts") is None:          # an identity-failure stub has no contrasts
                        self.scenes[s] = {"_stub": a, "_dir": d}
                    else:
                        self.scenes[s] = {**a, "_dir": d}
                    break
        self.figs, self.skipped, self.inputs = [], [], {}
        for rel in ("test_eval/test.json", "test_eval/arrays.npz", "dev_eval/dev.json", "dev_eval/arrays.npz", "decision.json",
                    "amendment1/result.json", "amendment1/test_amended.json", "amendment1/decision_amendment1.json"):
            if (root / rel).exists():
                self.inputs[rel] = sha(root / rel)
        for nm, q in (("decision (override)", decision), ("decision_amendment1 (override)", decision_amended)):
            if q is not None and Path(q).exists():
                self.inputs[nm] = sha(q)
        for a in self.fits:
            self.inputs[f"fits/{a}/fit.json"] = sha(root / "fits" / a / "fit.json")
        sp = stageA or (root.parent / "stageA/stageA.json")
        if Path(sp).exists():
            self.inputs["stageA/stageA.json"] = sha(sp)
        for s, a in self.scenes.items():
            for f in ("analysis104kb.json", "maps.npz"):
                p = a["_dir"] / f
                if p.exists():
                    self.inputs[f"scenes/{s}/{f}"] = sha(p)
        if self.T is not None:
            rep = self.T.get("O0_reproduction", {})
            self.qual = self.T.get("qualified", None)
            am = ""
            if self.AM is not None:
                am = (" Amendment 1 (same-method nondeterminism test): " + self.AM["verdict"]
                      + (", labels stay UNQUALIFIED." if self.AM["verdict"] != "NONDETERMINISM_CONSISTENT"
                         else ", labels qualified with flag O0_REPRODUCTION_DISCREPANCY."))
            self.warn = (None if self.qual else
                         f"Labels UNQUALIFIED: O0 missed the WL104K P reproduction gate (max {PCT * rep.get('max', np.nan):.2f} % > 1 % "
                         f"limit); values and predeclared rules shown descriptively." + am)
        else:
            self.qual, self.warn = None, None

    def save(self, fig, name, cap, kind, src):
        hdr = getattr(fig, "_hdr_in", None)
        if hdr is not None:                      # keep axes titles clear of the header (fixed inches)
            nl = max([ax.get_title().count(chr(10)) + 1 for ax in fig.axes if ax.get_title()] or [0])
            top = 1 - (hdr + 0.08 + 0.17 * nl) / fig.get_figheight()
            if fig.subplotpars.top > top:
                fig.subplots_adjust(top=top)
        fig.savefig(self.out / name, dpi=110)
        plt.close(fig)
        self.figs.append({"file": name, "caption": cap, "kind": TAGS[kind][0], "sources": src})

    def A(self, arm, key, pre="test"):
        return self.z[f"{pre}__{arm}__{key}"]


# ===================================================================== (1) Stage A
def stageA_peak(pk):
    SA = pk.SA
    g = SA["grid"]
    pol, az = np.array(g["incidence_polar_deg"]), np.array(g["incidence_azimuth_deg"])
    noise = np.array(g["reference_half_noise_relative_at_mirror"]).reshape(len(pol), len(az))
    good, bad = SA["good_starts"], SA["bad_starts"]
    fig, axs = plt.subplots(1, len(az), figsize=(15, 5.6), sharey=True)
    cg, cb = ["#2a78d6", "#1c5cab"], ["#eb6834", "#b5461b"]
    for j, ax in enumerate(axs):
        for r, row in enumerate(SA["profiles"]["12k"]):
            v = PCT * np.array(row["peak_height_rel"]).reshape(len(pol), len(az))[:, j]
            c = cg[good.index(r)] if r in good else cb[bad.index(r)]
            ax.plot(pol, v, "-o", color=c, ms=4, lw=1.6,
                    label=f"P start {r + 1} ({'good' if r in good else 'bad'} in WL104K-A)")
            ve = PCT * np.array(SA["profiles"]["extended"][r]["peak_height_rel"]).reshape(len(pol), len(az))[:, j]
            ax.plot(pol, ve, ":", color=c, lw=1.0,
                    label="dotted, same colour: each start after the\nWL104K-A extension (+12k steps)" if r == 0 else None)
            va = PCT * np.array(row["A_peak_height_rel"]).reshape(len(pol), len(az))[:, j]
            ax.plot(pol, va, "--", color="#a0a0a0", lw=1.0, label=(ARM["A_hist"] + ", starts 1-4") if r == 0 else None)
        ax.fill_between(pol, -PCT * noise[:, j], PCT * noise[:, j], color="#888", alpha=0.35, lw=0,
                        label="reference half-split noise (±|a − b| / 2)")
        ax.axhline(0, color="k", lw=0.8)
        ax.axvspan(20, 36, color="#f3e3b0", alpha=0.35, lw=0, zorder=0, label="hard_1 scene highlight incidences\n(20–36°)")
        ax.set_xlabel("incident polar angle (deg)")
        ax.set_title(f"incident azimuth {az[j]:g}°  (reference noise ≤ {PCT * noise[:, j].max():.3f} %)", fontsize=9.5)
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("mirror-peak height error, model / physical − 1 (%)")
    hs, ls = axs[0].get_legend_handles_labels()
    order = sorted(range(len(ls)), key=lambda i: (not ls[i].startswith("P start") or "extended" in ls[i], ls[i]))
    fig.legend([hs[i] for i in order], [ls[i] for i in order], fontsize=7.6, loc="center left", bbox_to_anchor=(0.8, 0.53),
               frameon=False)
    title(fig, "Stage A — hard_1 material: mirror-peak height vs incidence for each archived WL104K fitting start (12k steps)",
          "ER-v1-SDM WL104K fit (P), first bounce, peak = luminance at the exact mirror direction, 14–42° incidence")
    tag(fig, "DIAG")
    foot(fig, "Physical response from fresh layered random walks (65,536 walks per direction, two independent halves, seed 1041301) on a dense "
              "incidence grid covering the hard_1 scene highlight (20–36°). Each solid line is one of the four archived WL104K P fitting starts "
              "(12k Adam steps); dotted = the same start after the WL104K-A 12k-step extension (nearly on top of the solid line); dashed grey = "
              "the four starts of the certified ER baseline A. The grey band is the reference half-split noise at the mirror direction; the gap "
              "between starts 1-2 and starts 3-4 is larger than this band at every incidence. Diagnostic of archived states, run before the "
              "WL104K-B freeze.")
    fig.set_size_inches(18, 5.8)
    fig.subplots_adjust(left=0.05, right=0.79, top=0.86, bottom=0.2, wspace=0.08)
    pk.save(fig, "stageA_hard1_mirror_peak_height_vs_incidence_per_start.png",
            "Stage A: hard_1 mirror-peak height error vs incidence per archived WL104K P start, with the A baseline and reference noise.",
            "DIAG", ["stageA.json:grid", "stageA.json:profiles.12k[*].peak_height_rel / A_peak_height_rel", "stageA.json:profiles.extended"])


def stageA_regions(pk):
    SA = pk.SA
    g = SA["grid"]
    pol, az = np.array(g["incidence_polar_deg"]), np.array(g["incidence_azimuth_deg"])
    good = SA["good_starts"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (k, nm) in zip(axs, (("core_0_3", "specular core, 0–3° from mirror"), ("shoulder_3_12", "shoulder, 3–12° from mirror"),
                                  ("tail_12_30", "tail, 12–30° from mirror"))):
        for r, row in enumerate(SA["profiles"]["12k"]):
            v = PCT * np.array(row[f"{k}_signed"]).reshape(len(pol), len(az))
            c = ["#2a78d6", "#1c5cab"][good.index(r)] if r in good else ["#eb6834", "#b5461b"][SA["bad_starts"].index(r)]
            ax.plot(pol, v[:, 0], "-o", color=c, ms=3.5, lw=1.4, label=f"start {r + 1} ({'good' if r in good else 'bad'}), azimuth {az[0]:g}°")
            ax.plot(pol, v[:, 1], "--", color=c, lw=1.0, label=f"start {r + 1}, azimuth {az[1]:g}°")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(nm, fontsize=10)
        ax.set_xlabel("incident polar angle (deg)")
        ax.set_ylabel("signed energy error in region (%)\nΣ (F_model − F_phys) / Σ F_phys over grid points")
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=6.6, ncol=1)
    title(fig, "Stage A — hard_1 material: signed error of the specular core, shoulder and tail per archived WL104K P start",
          "ER-v1-SDM WL104K fit (P), first bounce, matched physical directions on rings 0–30° around the mirror direction")
    tag(fig, "DIAG")
    foot(fig, "Same fresh-walk grid as the mirror-peak figure. Region errors are luminance sums over the ring grid points (uniform in grid "
              "points, not solid angle). Solid = azimuth 22.5°, dashed = 112.5°. Read the start-to-start gap per region: in the core the bad starts (1-2) "
              "lie above the good starts (3-4) at every incidence; in the shoulder the bad starts lie above and in the tail below "
              "the good starts, by smaller margins than in the core. No reference-noise band is stored for the region sums.")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.2, wspace=0.3)
    pk.save(fig, "stageA_hard1_core_shoulder_tail_signed_error_per_start.png",
            "Stage A: hard_1 signed core / shoulder / tail error vs incidence per archived start.", "DIAG",
            ["stageA.json:profiles.12k[*].core_0_3_signed / shoulder_3_12_signed / tail_12_30_signed"])


def stageA_path(pk):
    SA = pk.SA
    D = SA["good_vs_bad"]
    fig, axs = plt.subplots(1, 4, figsize=(19, 5.4))
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    for k, d in enumerate(D):
        p = d["path"]
        t = np.array(p["t"])
        lab = f"bad start {d['bad'] + 1} → good start {d['good'] + 1}"
        o = np.array(p["objective"])
        axs[0].plot(t, PCT * (o / o[0] - 1), "-o", ms=3, color=cols[k], label=lab)
        pr = np.array(p["peak_rel"])                                 # absolute mirror luminance at 20 / 27.5 / 35 deg
        rel = PCT * (pr / pr[0] - 1)
        axs[1].plot(t, rel.mean(1), "-o", ms=3, color=cols[k], label=lab)
        axs[1].fill_between(t, rel.min(1), rel.max(1), color=cols[k], alpha=0.15, lw=0)
        e = np.array(p["heldout_el1"])
        axs[2].plot(t, e, "-o", ms=3, color=cols[k], label=lab)
    axs[0].set_ylabel("training objective change vs the bad endpoint (%)\n(WL104K objective el1 + ealb on hard_1)")
    axs[1].set_ylabel("mirror-peak luminance change vs the bad endpoint (%)\nline: mean of 20 / 27.5 / 35° incidence, band: range")
    axs[2].set_ylabel("WL104D held-out angular el1 (relative L1, dimensionless)")
    for ax in axs[:3]:
        ax.set_xlabel("position on the straight parameter path (0 = bad state, 1 = good state)")
        ax.axvline(0, color="#ccc", lw=0.6)
        ax.grid(alpha=0.25)
        ax.axhline(0, color="k", lw=0.6) if ax is not axs[2] else None
    axs[0].legend(fontsize=7)
    axs[0].set_title("training objective along the path\n(barrier between basins = positive bump)", fontsize=9.5)
    axs[1].set_title("mirror-peak height along the path", fontsize=9.5)
    axs[2].set_title("held-out angular error along the path", fontsize=9.5)
    ax = axs[3]
    incs = ["20°", "27.5°", "35°"]
    ic = ["#3c3c3c", "#e87ba4", "#eda100"]          # incidence colours: not the path colours of the other panels
    costs, same_sign = [], []
    for r, s in enumerate(SA["sensitivity"]):
        for i, pr in enumerate(s["five_percent_peak_probe"]):
            signs = []
            for sg, dx in (("plus", -0.06), ("minus", 0.06)):
                q = pr[sg]
                if not q.get("reached"):
                    continue
                up = q["peak_rel_change"] > 0               # marker = ACTUAL sign of the peak change, not the step sign
                signs.append(up)
                costs.append(q["objective_rel_change"])
                ax.plot(r + (i - 1) * 0.22 + dx, PCT * q["objective_rel_change"], "^" if up else "v", color=ic[i], ms=7,
                        mfc=ic[i] if up else "white", mew=1.4)
            if len(signs) == 2 and signs[0] == signs[1]:
                same_sign.append(f"start {r + 1} at {incs[i]}")
    for i in range(3):
        ax.plot([], [], "s", color=ic[i], label=f"incidence {incs[i]}")
    ax.plot([], [], "^", color="#555", label="step that RAISED the peak by 5 %")
    ax.plot([], [], "v", color="#555", mfc="white", label="step that LOWERED the peak by 5 %")
    ax.set_yscale("log")
    ax.set_xticks(range(len(SA["sensitivity"])), [f"start {r + 1}\n({'bad' if r in SA['bad_starts'] else 'good'})"
                                                  for r in range(len(SA["sensitivity"]))])
    ax.set_ylabel("training objective increase needed for a 5 % peak change (%)")
    ax.set_title("cost of a local 5 % mirror-peak change\n(least-curvature direction, actual L1 objective)", fontsize=9.5)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.25, which="both")
    title(fig, "Stage A — hard_1 material: straight-path counterfactual between bad and good archived starts, and the objective cost of fixing the peak",
          "ER-v1-SDM WL104K fit (P); path x(t) = (1 − t) x_bad + t x_good in the 178-parameter chart (gates clipped to [0, 1])")
    tag(fig, "DIAG")
    foot(fig, "Left three panels: actual training objective (not a surrogate), mirror-peak luminance and the WL104D held-out angular el1 at 21 points "
              "on the straight line between a bad (overshooting) and a good start (held-out el1 = mean over the 20 WL104D probes of hard_1). "
              "Right: from each archived start, the step along ± the direction of least response-metric (Gauss-Newton surrogate) curvature that "
              "changes the mirror peak (azimuth 22.5°) by exactly 5 %, and the resulting relative increase of the actual training objective (log "
              f"scale). Every one of the {len(costs)} probes INCREASED the objective ({PCT * min(costs):.0f}–{PCT * max(costs):.0f} %), including "
              "the peak-lowering steps from the overshooting bad starts: locally, the WL104K objective penalizes rather than rewards correcting "
              "the peak along these directions."
              + (f" Both step signs lowered the peak for {', '.join(same_sign)} (non-monotonic response along that direction)." if same_sign
                 else "")
              + " A straight path is one path among many; the barrier height is path dependent.")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.21, wspace=0.35)
    pk.save(fig, "stageA_hard1_good_vs_bad_straight_path_objective_peak_and_5pct_cost.png",
            "Stage A: training objective, peak and held-out el1 along the bad-to-good straight path; objective cost of a 5 % peak change.",
            "DIAG", ["stageA.json:good_vs_bad[*].path", "stageA.json:sensitivity[*].five_percent_peak_probe"])


def stageA_ig(pk):
    SA = pk.SA
    groups = SA["groups"]
    D = SA["good_vs_bad"]
    nice = {"outer_amplitude": "outer-lobe amplitudes", "split": "lobe energy split", "roughness": "roughness (alpha)",
            "anisotropy": "anisotropy", "fresnel": "Fresnel F0", "attenuation": "attenuation knots", "eta": "eta"}
    lab = [nice.get(g, g.replace("gate_", "gate of component ")) for g in groups]
    fig, ax = plt.subplots(figsize=(13, 7.6))
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
    y = np.arange(len(groups))
    h = 0.2
    for k, d in enumerate(D):
        v = PCT * np.array([np.mean(d["integrated_gradient_by_group_rel"][g]) for g in groups])
        tot = PCT * np.mean(d["peak_rel_difference_good_vs_bad"])
        ax.barh(y + (k - 1.5) * h, v, h * 0.9, color=cols[k],
                label=f"bad start {d['bad'] + 1} → good start {d['good'] + 1} (total peak change {tot:+.2f} %)")
    ax.set_yticks(y, lab, fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("integrated-gradient share of the mirror-peak change (% of the bad state's peak, mean of 20 / 27.5 / 35° incidence)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    ax.text(0.01, 0.99, "CAVEAT: path dependent and non-identifiable.\nNear-duplicate lobes, amplitude / Fresnel / attenuation collinearity\n"
                        "and gate / amplitude trade-offs make group shares chart- and path-specific.\nFunctional evidence only; not a unique physical cause.",
            transform=ax.transAxes, va="top", fontsize=8.5, color="#8a0000", bbox=dict(fc="#fff4f4", ec="#d99"))
    ax.texts[-1].set_position((0.01, 0.02)); ax.texts[-1].set_va("bottom")
    ax.set_xlim(min(ax.get_xlim()[0], -16), ax.get_xlim()[1])
    resid = max(abs(x) for d in D for x in d["integrated_gradient_completeness_residual_rel"])
    title(fig, "Stage A — hard_1 material: grouped integrated-gradient attribution of the bad-vs-good mirror-peak difference "
               "(first bounce, incident azimuth 22.5°)",
          "ER-v1-SDM WL104K fit (P); 64-point midpoint rule along the straight path; groups = ER-v1-GF parameter groups + one group per component gate")
    tag(fig, "DIAG")
    foot(fig, "Each bar is the integral of (∂ peak / ∂ x) · Δx over the parameters of one group along the straight path from the bad to the good "
              f"start, relative to the bad start's peak luminance; the group bars of one path sum to the total change (max completeness residual "
              f"{resid:.1e} of the peak). "
              "Opposite-signed large bars on correlated groups (e.g. a gate against an amplitude) are compensation along this particular path. "
              "These attributions are path dependent and not identifiable: they must not be read as the error of a specific lobe.")
    fig.subplots_adjust(left=0.2, right=0.99, top=0.88, bottom=0.14)
    pk.save(fig, "stageA_hard1_grouped_integrated_gradient_attribution_bad_vs_good.png",
            "Stage A: grouped integrated-gradient attribution of the bad-vs-good peak difference (path dependent, non-identifiable).",
            "DIAG", ["stageA.json:good_vs_bad[*].integrated_gradient_by_group_rel", "stageA.json:groups"])


# ===================================================================== (2) optimization
def curves(f, u):
    if "history_steps" in f and f.get("history_steps"):
        return np.array(f["history_steps"]), np.array(f["history_objective"][u])
    return np.array(f["history_evaluations"][u]), np.array(f["history_objective"][u])


def opt_curves(pk):
    units = [m for m in ("hard_1", "block_186", "hard_53", "iid_0") if m in pk.mats]
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    n = pk.n
    for ax, mname in zip(axs.ravel(), units):
        m = pk.mats.index(mname)
        for a in ("O0", "O1", "D", "E2", "F"):
            f = pk.fits.get(a)
            if f is None:
                continue
            R = len(f["start_keys"])
            for r in range(R):
                x, v = curves(f, r * n + m)
                ax.plot(np.maximum(x, 1), v, color=COL[a], lw=0.9, alpha=0.8, ls="--" if a == "E2" else ("-." if a == "D" else "-"),
                        label=ARM[a] + (" — own E2 objective" if a == "E2" else " — on refined-coverage lanes" if a == "D" else "")
                        if r == 0 else None)
            fin = np.array(f["final_terms"]["objective"]).reshape(R, n)[:, m]
            ax.plot(np.full(R, 12500), fin, MK[a], color=COL[a], ms=5, mfc="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("optimizer steps (Adam) or objective evaluations (L-BFGS), log")
        ax.set_ylabel("training objective (dimensionless relative L1)")
        ax.set_title(f"material {mname} ({pk.cohort[m]} cohort): 8 starts per arm; open markers at right = final values", fontsize=9.5)
        ax.grid(alpha=0.25, which="both")
    axs[0, 0].legend(fontsize=7.5)
    title(fig, "Optimization: training-objective curves per arm for representative materials (hard_1, block_186, hard_53, iid_0)",
          "Each line = one (material, start) unit; O0 Adam vs O1 L-BFGS optimize the same function on the same lanes")
    tag(fig, "DESC")
    foot(fig, "O0 (solid dark) and O1 (solid orange) minimize the identical WL104K objective (el1 + ealb) on the original training lanes, so their "
              "curves are directly comparable. D (dash-dot) minimizes the same functional on its own refined-coverage lanes and E2 (dashed) "
              "minimizes el1 + ealb + core + shoulder terms, so their absolute levels are not comparable with O0. L-BFGS curves end early where a "
              "unit was frozen after 3 consecutive failed line searches; the open markers at the right edge are every unit's final value drawn at "
              "a common x position (not at its own stopping point). Training curves are logged every 100 steps / evaluations, so the segment "
              "from 1 to 100 is a straight interpolation. iid_0 is the material with the largest O0-vs-WL104K reproduction miss.")
    fig.set_size_inches(16, 10.8)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.135, hspace=0.32, wspace=0.18)
    pk.save(fig, "optimization_training_objective_curves_per_arm_hard1_block186_hard53_iid0.png",
            "Training-objective curves per arm (O0 Adam, O1 L-BFGS, D, E2) for hard_1, block_186, hard_53, iid_0; 8 starts each.",
            "DESC", ["fits/<arm>/fit.json:history_steps|history_evaluations, history_objective, final_terms.objective"])


def opt_summary(pk):
    f0, f1 = pk.fits.get("O0"), pk.fits.get("O1")
    if f0 is None or f1 is None:
        pk.skipped.append("optimization summary: O0 or O1 fit.json missing")
        return
    n = pk.n
    R = len(f1["start_keys"])
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.6))
    term = np.array(f1["terminated_early"], bool).reshape(R, n)
    ev = np.array(f1["evaluations_per_unit"]).reshape(R, n)
    ax = axs[0]
    ax.bar(range(n), term.sum(0), color=COL["O1"])
    ax.set_xticks(range(n), pk.mats, rotation=40, fontsize=8)
    ax.set_ylabel("units terminated early (of 8 starts)")
    ax.set_ylim(0, R + 0.5)
    ax.set_title(f"L-BFGS (O1): {term.sum()} of {term.size} units frozen early\n(3 consecutive failed line searches)", fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)
    ax = axs[1]
    for m in range(n):
        ax.plot(np.full(R, m) + jitter(0, R, 0.5), ev[:, m], "s", color=COL["O1"], ms=4)
    ax.axhline(f1["budget"], color="k", ls="--", lw=0.9, label=f"budget {f1['budget']:,} evaluations (= O0's 12,000 Adam steps)")
    ax.set_xticks(range(n), pk.mats, rotation=40, fontsize=8)
    ax.set_ylabel("objective evaluations used per unit")
    ax.set_title("L-BFGS (O1): evaluations used per start", fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)
    ax = axs[2]
    o0 = np.array(f0["final_terms"]["objective"]).reshape(-1, n)
    o1 = np.array(f1["final_terms"]["objective"]).reshape(-1, n)
    cm = plt.cm.tab10
    for m in range(n):
        ax.scatter(o0[:, m], o1[:, m], s=22, color=cm(m), label=pk.mats[m], edgecolor="white", lw=0.4)
    lim = [min(o0.min(), o1.min()) * 0.8, max(o0.max(), o1.max()) * 1.25]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("final training objective, original Adam fit (O0)")
    ax.set_ylabel("final training objective, L-BFGS (O1)")
    better = int((o1 < o0).sum())
    ax.set_title(f"same function, same lanes: O1 lower in {better} of {o1.size} units\n(above the diagonal = L-BFGS worse)", fontsize=9.5)
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(alpha=0.25, which="both")
    ax = axs[3]
    op = (pk.T or {}).get("optimization", {})
    arms = [a for a in ("O0", "O1", "D", "E2", "F") if a in op]
    x = np.arange(len(arms))
    ax.bar(x - 0.2, [PCT * op[a]["stationary_fraction_0.5pct"] for a in arms], 0.38, color=[COL[a] for a in arms], label="stationary units (%)")
    ax.bar(x + 0.2, [PCT * op[a]["gate_fraction_at_bounds"] for a in arms], 0.38, color=[COL[a] for a in arms], alpha=0.45,
           hatch="//", label="gate knots at 0 or 1 (%)")
    ax.set_xticks(x, [ARM[a] for a in arms], rotation=20, fontsize=7.5)
    ax.set_ylabel("share (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7.5)
    ax.set_title("convergence reporting (not a gate):\nstationary = objective change ≤ 0.5 % over last 2,000", fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)
    title(fig, "Optimization: L-BFGS (O1) early termination, evaluations used, and final training objective of O1 vs the original Adam fit (O0)",
          "72 units = 9 materials × 8 starts; each unit is an independent 178-parameter problem", pk.warn)
    tag(fig, "DESC")
    short = int((ev < 2000).sum())
    foot(fig, "O1 = batched projected L-BFGS (memory 20, Armijo backtracking, gate knots boxed to [0, 1]) with a budget of 12,000 objective "
              "evaluations per unit. Units frozen after three failed line searches stop at their best point; their training objective is then "
              "compared with O0 at equal or smaller budget. Stationarity window: last 2,000 Adam steps (O0, D, E2) or last 2,000 objective "
              f"evaluations (O1); {short} of the {ev.size} O1 units stopped before 2,000 evaluations and count as not stationary. "
              "Stationarity reaching a threshold does not establish a correct or identifiable response.")
    fig.subplots_adjust(left=0.05, right=0.97, top=0.8, bottom=0.22, wspace=0.3)
    pk.save(fig, "optimization_lbfgs_early_termination_and_final_objective_vs_adam.png",
            "O1 early termination per material, evaluations used, O1 vs O0 final objective per unit, stationarity per arm.", "DESC",
            ["fits/O1/fit.json:terminated_early, evaluations_per_unit, final_terms", "fits/O0/fit.json:final_terms",
             "test.json:optimization"])


def qual_repro(pk):
    rep = (pk.T or {}).get("O0_reproduction")
    if rep is None:
        pk.skipped.append("O0 reproduction: test.json missing")
        return
    v = np.array(rep["relative_per_unit"])                          # [4, n]
    AM = pk.AM
    fig, axs = plt.subplots(1, 2 if AM else 1, figsize=(19 if AM else 13, 5.8), squeeze=False,
                            gridspec_kw={"width_ratios": [1.35, 1]} if AM else None)
    ax = axs[0, 0]
    floor = 1e-7                                                    # exact zeros drawn at a visible floor
    for r in range(v.shape[0]):
        ax.plot(np.arange(pk.n) + (r - 1.5) * 0.12, PCT * np.maximum(v[r], floor), MK["O0"],
                color=["#3c3c3c", "#2a78d6", "#eb6834", "#1baf7a"][r], ms=7, label=f"start {r + 1} ({rep['start_keys'][r]})")
    ax.axhline(PCT * 1e-4, color="#2b6cb0", ls="--", lw=1, label="reproduced if ≤ 0.01 % (1e-4)")
    ax.axhline(PCT * 1e-2, color="#b00020", ls="--", lw=1, label="UNQUALIFIED if any unit > 1 % (1e-2)")
    ax.set_yscale("log")
    ax.set_xticks(range(pk.n), [f"{m}\n({c})" for m, c in zip(pk.mats, pk.cohort)], fontsize=8)
    ax.set_ylabel("|O0 final objective − WL104K P final objective|\n/ WL104K P final objective (%)")
    ax.set_title("original rule: O0 (8-start batch) vs archived WL104K P (4-start batch), per unit", fontsize=9.5)
    ax.legend(fontsize=7.6)
    ax.grid(alpha=0.25, which="both")
    cap = ("Protocol arms.O0: the O0 rerun of the original WL104K Adam method (same data, objective, starts 1-4) must match the archived "
           "WL104K P final training objective within 1e-4 relative per unit; a miss between 1e-4 and 1e-2 is disclosed as a flag; a miss above "
           "1e-2 for any unit makes every angular label UNQUALIFIED. Bitwise identity is not expected (WL104K-A).")
    if AM:
        ax = axs[0, 1]
        comp = AM["components"]
        nice = {"batch4_R4a_vs_archive": "rerun R4a vs archive\n(batch 4, run-to-run)",
                "batch4_R4b_vs_archive": "rerun R4b vs archive\n(batch 4, run-to-run)",
                "batch4_R4a_vs_R4b": "R4a vs R4b\n(batch 4, run-to-run)",
                "batch8_R8_vs_O0": "rerun R8 vs O0\n(batch 8, run-to-run)",
                "batchsize_R8_starts1to4_vs_R4a": "R8 starts 1-4 vs R4a\n(batch size 8 vs 4)",
                "batchsize_R8_starts1to4_vs_R4b": "R8 starts 1-4 vs R4b\n(batch size 8 vs 4)"}
        keys = [k for k in nice if k in comp] + ["target"]
        T = AM["target_O0_vs_archive"]
        y = np.arange(len(keys))
        for i, k in enumerate(keys):
            st = T if k == "target" else comp[k]
            c = "#b00020" if k == "target" else "#555"
            ax.plot([PCT * max(st["median"], floor)], [i], "|", color=c, ms=14, mew=2)
            ax.plot([PCT * max(st["q95"], floor)], [i], "D", color=c, ms=6, mfc="white")
            ax.plot([PCT * max(st["max"], floor)], [i], "s", color=c, ms=7)
            ax.text(PCT * 2.2 * max(st["max"], floor), i, f"n {st['n']}, > 1e-4: {st['n_above_1e-4']}, > 1e-2: {st['n_above_1e-2']}"
                    + ("  (all exactly 0)" if st["max"] == 0 else ""), va="center", fontsize=7.4, color=c)
        Pz = AM["pooled_reruns"]
        ax.axvline(PCT * 2 * Pz["q95"], color="#2a78d6", ls="--", lw=1, label=f"criterion (ii): 2 × pooled-rerun 95th pct = {PCT * 2 * Pz['q95']:.2f} %")
        ax.axvline(PCT * 2 * Pz["max"], color="#1baf7a", ls="--", lw=1, label=f"criterion (i): 2 × pooled-rerun max = {PCT * 2 * Pz['max']:.2f} %")
        ax.plot([], [], "|", color="#555", ms=10, mew=2, label="median")
        ax.plot([], [], "D", color="#555", mfc="white", label="95th percentile")
        ax.plot([], [], "s", color="#555", label="maximum")
        ax.set_xscale("log")
        ax.set_xlim(PCT * floor * 0.5, PCT * 3e2)
        ax.set_yticks(y, [nice.get(k, "TARGET: O0 vs archive\n(starts 1-4, 36 units)") for k in keys], fontsize=7.6)
        ax.invert_yaxis()
        ax.set_xlabel(f"per-unit relative difference of the final training objective (%), log; exact 0 drawn at {PCT * floor:g} %")
        ax.set_title(f"Amendment 1 (owner-approved, after results): same-method reruns → {AM['verdict']}\n"
                     f"(i) max {'met' if AM['criterion_parts']['max'] else 'missed'}, (ii) 95th percentile "
                     f"{'met' if AM['criterion_parts']['q95'] else 'missed'}", fontsize=9.2)
        ax.legend(fontsize=7.2, loc="upper right")
        ax.grid(alpha=0.25, which="both", axis="x")
        cap += (" Right: Amendment 1 reruns of the same code. Runs with the same batch size are bitwise identical to each other and to the "
                "archive / O0 (exact 0); the whole O0-vs-archive difference appears only when the batch size changes from 4 to 8 starts. "
                "The amendment criterion (both (i) and (ii) within 2x the pooled rerun distribution) was not met, so the labels stay UNQUALIFIED.")
    title(fig, "Qualification: did the O0 control reproduce the archived WL104K P fit? (final training objective per unit, starts 1-4)",
          f"original rule status {rep['status']}, max relative difference {PCT * rep['max']:.2f} %", pk.warn)
    tag(fig, "PRE")
    foot(fig, cap)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.8, bottom=0.2 if not AM else 0.24, wspace=0.32)
    pk.save(fig, "qualification_O0_reproduction_of_WL104K_P_final_objective_and_amendment1.png",
            "O0 reproduction of WL104K P per unit vs the predeclared 1e-4 / 1e-2 limits, and the Amendment 1 same-method rerun "
            "distributions (reason for UNQUALIFIED labels).", "PRE",
            ["test.json:O0_reproduction"] + (["amendment1/result.json:components, target_O0_vs_archive, pooled_reruns, verdict"] if AM else []))


# ===================================================================== (3) independent test
def extreme_note(pk, arm="O1", k="e_prof"):
    """Name the start(s) whose pooled error exceeds 10x the arm median, and the material that carries it."""
    if arm not in pk.arms:
        return ""
    v = pk.A(arm, k).sum(-1)                                          # [R, n]
    tot = v.sum(1)
    bad = np.flatnonzero(tot > 10 * np.median(tot))
    if not len(bad):
        return ""
    keys = (pk.Dv or {}).get("start_keys") or [str(r) for r in range(len(tot))]
    parts = [f"start {r + 1} (seed {keys[r]}): pooled {k} {tot[r]:.0f}, {100 * v[r].max() / tot[r]:.0f} % of it from material "
             f"{pk.mats[int(np.argmax(v[r]))]}" for r in bad]
    return f"Extreme {ARM[arm]} start(s), kept visible by the log axis: " + "; ".join(parts) + "."


def test_pooled(pk):
    arms = pk.arms + ["P_hist", "A_hist"]
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.8))
    for ax, k, nm in ((axs[0], "e_prof", "peak-profile error e_prof"), (axs[1], "el1", "hemispherical energy-weighted relative L1 el1")):
        for j, a in enumerate(arms):
            v = pk.A(a, k).sum((1, 2))
            x = j + jitter(0, len(v), 0.5)
            ax.plot(x, v, MK[a], color=COL[a], ms=6, mfc="none" if a.endswith("hist") else COL[a], alpha=0.9)
            ax.hlines(np.median(v), j - 0.32, j + 0.32, color=COL[a], lw=2.2)
            ax.text(j + 0.36, np.median(v), f"{np.median(v):.2f}", ha="left", va="center", fontsize=7.5, color="#222")
        ax.set_yscale("log")
        ax.set_xticks(range(len(arms)), [ARM[a] + (" — 4 starts" if a.endswith("hist") else " — 8 starts") for a in arms],
                      rotation=22, ha="right", fontsize=8)
        ax.set_ylabel(f"pooled {k} (sum over 180 material × probe\npairs, dimensionless), log scale")
        ax.set_xlim(-0.5, len(arms) - 0.2)
        ax.set_title(f"pooled {nm}\none marker per fitting start, bar = median", fontsize=9.5)
        ax.grid(axis="y", alpha=0.25, which="both")
    title(fig, "Independent angular test: pooled peak-profile error and hemispherical el1 per fitting start, all arms",
          "Test set: new incidences (seed 1041330), 2,048 lanes × 32,768 walks; peak grid 0–8° around the mirror, 131,072 walks", pk.warn)
    tag(fig, "DESC")
    foot(fig, "e_prof = Σ|F − R| / Σ R of luminance over the peak grid (radius 0, rings 0.25–4° in 0.25° steps, 5, 6, 8°; 16 azimuths) per "
              "(material, probe), summed over all 180 (material, probe) pairs. el1 = WL82 energy-weighted relative L1 per probe, summed likewise. "
              "Historical WL104K arms (open markers) have 4 starts and are unpaired with the new 8-start arms. " + extreme_note(pk))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.83, bottom=0.3, wspace=0.22)
    pk.save(fig, "test_pooled_peak_profile_error_and_el1_per_start_all_arms.png",
            "Independent test: pooled e_prof and el1 per start per arm (dots) with medians.", "DESC",
            ["test_eval/arrays.npz:test__<arm>__e_prof, test__<arm>__el1"])


def contrast_panel(ax, items, thr=None, keep=150.0):
    """items: list of (x, arm, contrast {'median','lo','hi',['per_start'],['resolved']}, filled, label_prefix).
    Linear % axis scaled to the moderate values (|CI| <= keep %, plus 0 and the threshold); values beyond are drawn at the
    axis edge as arrows with their numeric value (off scale), so small contrasts stay readable next to an extreme arm."""
    vals = [0.0] + ([thr] if thr is not None else [])
    for it in items:
        x, a, c = it[:3]
        v = [PCT * c["median"], PCT * c["lo"], PCT * c["hi"]] + [PCT * p for p in c.get("per_start", []) if abs(PCT * p) <= keep]
        if max(abs(PCT * c["lo"]), abs(PCT * c["hi"])) <= keep:
            vals += v
        else:
            vals.append(np.clip(PCT * c["median"], -keep, keep))
    lo, hi = min(vals), max(vals)
    pad = 0.12 * (hi - lo + 1e-9)
    lo, hi = lo - pad, hi + pad
    n_off = 0
    for it in items:
        x, a, c, filled, pre = it[:5]
        dx = it[5] if len(it) > 5 else -0.14
        m, l, h = PCT * c["median"], PCT * c["lo"], PCT * c["hi"]
        mc, lc, hc = np.clip([m, l, h], lo, hi)
        fs = dict(fillstyle="left", mfc=COL.get(a, "k"), mfcalt="white") if filled is None else \
            dict(mfc=COL.get(a, "k") if filled else "white")      # None = no half-split replication stored (half-filled marker)
        ax.errorbar(x, mc, yerr=[[mc - lc], [hc - mc]], fmt=MK.get(a, "o"), color=COL.get(a, "k"), ms=9, capsize=5, mew=1.6, **fs)
        if "per_start" in c:
            ps = PCT * np.array(c["per_start"])
            xs = x + 0.22 + jitter(0, len(ps), 0.15)
            inr = (ps >= lo) & (ps <= hi)
            ax.plot(xs[inr], ps[inr], ".", color=COL.get(a, "k"), ms=5, alpha=0.75)
            ax.plot(xs[ps > hi], np.full((ps > hi).sum(), hi), "^", color=COL.get(a, "k"), ms=5, alpha=0.75)
            ax.plot(xs[ps < lo], np.full((ps < lo).sum(), lo), "v", color=COL.get(a, "k"), ms=5, alpha=0.75)
        off = not (lo <= m <= hi) or l < lo or h > hi
        txt = f"{pre}{m:+.1f} %" + (f"\n[{l:+.0f}, {h:+.0f}] off scale" if off else "")
        if off:                                  # stack off-scale notes from the edge so several never overprint
            n_off += 1
            step = 0.14 * n_off * (hi - lo)
            ty = lo + step if mc <= 0.5 * (lo + hi) else hi - step
        else:
            ty = mc
        ax.text(x + (0.1 if off else dx), ty, txt, ha="left" if (off or dx > 0) else "right", va="center", fontsize=7.3,
                color="#b00020" if off else "#222")
    if thr is not None:
        ax.axhline(thr, color="#b00020", ls="--", lw=1.1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylim(lo, hi)
    return lo, hi


def test_contrasts(pk):
    con = pk.T["contrasts_vs_O0"]
    arms = [a for a in ("O1", "D", "E2", "F") if a in con]
    spec = [("e_prof", "peak-profile error e_prof", -15, "PEAK_GAIN needs ≤ −15 %, CI upper < 0, resolved"),
            ("instability", "across-start peak instability I", -25, "STABILITY_GAIN needs ≤ −25 %, CI upper < 0, resolved"),
            ("el1", "hemispherical el1", 5, "BROAD_PRESERVED needs ≤ +5 % (and ≤ +5 % vs WL104K P)"),
            ("shoulder_l1", "shoulder (3–12°) region L1", 10, "BROAD_PRESERVED needs ≤ +10 %"),
            ("tail_l1", "tail (≥ 12°) region L1", 10, "BROAD_PRESERVED needs ≤ +10 %"),
            ("el1_grazing", "grazing el1 (polar ≥ 75°)", 10, "BROAD_PRESERVED needs ≤ +10 %")]
    fig, axs = plt.subplots(2, 4, figsize=(21, 10))
    for ax, (k, nm, thr, rule) in zip(axs.ravel(), spec):
        contrast_panel(ax, [(j, a, con[a][k]["all"], None if "resolved" not in con[a][k]["all"] else bool(con[a][k]["all"]["resolved"]),
                             "") for j, a in enumerate(arms)], thr)
        ax.text(-0.7, thr, f"rule {thr:+d} %", color="#b00020", fontsize=7.5, va="bottom", ha="left")
        ax.set_xlim(-0.75, len(arms) - 0.4)
        ax.set_xticks(range(len(arms)), [ARM[a] for a in arms], fontsize=7.8, rotation=12)
        ax.set_ylabel("relative change vs original Adam fit (O0) (%)")
        ax.set_title(f"{nm}\n{rule}", fontsize=8.8)
        ax.grid(axis="y", alpha=0.25, which="both")
    ax = axs[1, 2]
    bf = pk.T["broad_preserved_facts"]
    for j, a in enumerate(arms):
        v = PCT * np.array(bf[a]["material_el1_change"])
        ax.plot(np.arange(pk.n) + (j - (len(arms) - 1) / 2) * 0.22, v, MK[a], color=COL[a], ms=6, ls="none", label=ARM[a])
    ax.axhline(10, color="#b00020", ls="--", lw=1.1)
    ax.axhline(0, color="k", lw=0.8)
    for m in range(pk.n - 1):
        ax.axvline(m + 0.5, color="#eee", lw=0.8)
    ax.legend(fontsize=6.8, loc="upper left")
    ax.set_xticks(range(pk.n), pk.mats, fontsize=7.4, rotation=45)
    ax.set_ylabel("median-start el1 change vs O0, per material (%)")
    ax.set_title("per-material el1 change (one marker per material)\nBROAD_PRESERVED needs every material ≤ +10 %", fontsize=8.8)
    ax.grid(axis="y", alpha=0.25, which="both")
    ax = axs[1, 3]
    ax.axis("off")
    lab = pk.T["labels"]
    lines = ["Predeclared angular labels (test set)", ""]
    for a in arms:
        L = lab[a]
        lines += [f"{ARM[a]}:", f"   measured: {L['measured_label']}   evaluation: {L['label']}",
                  f"   PEAK_GAIN {L['PEAK_GAIN']}, STABILITY_GAIN {L['STABILITY_GAIN']},", f"   BROAD_PRESERVED {L['BROAD_PRESERVED']}, "
                  f"synthetic control {L['control_passed']}", f"   el1 vs WL104K P: {PCT * bf[a]['el1_vs_P_hist']:+.1f} %", ""]
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=8.2, family="monospace")
    title(fig, "Independent angular test: relative change of each intervention vs the original Adam fit (O0), with 95 % material-bootstrap CI "
               "and the predeclared thresholds", "Paired starts; filled marker = RESOLVED by the half-a / half-b replication, open = not resolved, "
               "half-filled = no half replication stored (shoulder / tail region L1; not required by their rule); small dots = per-start values",
          pk.warn)
    tag(fig, "PRE")
    foot(fig, "Contrast = Σ_(material, probe) X / Σ O0 − 1 per start (paired starts), median over the 8 starts; CI = 2.5 / 97.5 % of 10,000 "
              "cluster-bootstrap medians over materials (seed 1041350). Instability I = SD over starts (ddof 1) of the signed peak-height error per "
              "(material, probe), averaged; its contrast has no per-start values. Dashed red = predeclared threshold (protocol angular_verdicts). "
              "TRADE_OFF = a gain without broad preservation (not a success). Labels are UNQUALIFIED here because O0 failed the reproduction gate. "
              "Each axis is scaled to the moderate contrasts; an arm whose CI exceeds ±150 % is pinned to the axis edge with its value in red "
              "('off scale'), per-start values beyond the axis are drawn as edge arrows.")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.12, hspace=0.45, wspace=0.3)
    pk.save(fig, "test_relative_change_vs_O0_with_bootstrap_CI_and_predeclared_thresholds.png",
            "Independent test: e_prof, instability, el1, shoulder, tail, grazing changes vs O0 with bootstrap CI and predeclared thresholds; labels.",
            "PRE", ["test.json:contrasts_vs_O0.<arm>.<metric>.all", "test.json:broad_preserved_facts", "test.json:labels"])


def test_per_material(pk):
    arms = pk.arms + ["P_hist"]
    fig, ax = plt.subplots(figsize=(19, 6.4))
    w = 0.8 / len(arms)
    ref = np.concatenate([pk.A(a, "e_prof").sum(-1).ravel() for a in arms if a != "O1"])
    top = 3 * ref.max()                                              # axis top: 3x the largest non-L-BFGS value
    for j, a in enumerate(arms):
        v = pk.A(a, "e_prof").sum(-1)                                # [R, n]
        for m in range(pk.n):
            x = m + (j - (len(arms) - 1) / 2) * w
            xs = np.full(v.shape[0], x) + jitter(0, v.shape[0], w * 0.7)
            inr = v[:, m] <= top
            ax.plot(xs[inr], v[inr, m], MK[a], color=COL[a], ms=3.5, mfc="none" if a.endswith("hist") else COL[a], alpha=0.85)
            for xx, vv in zip(xs[~inr], v[~inr, m]):
                ax.plot(xx, top / 1.12, "^", color=COL[a], ms=6)
                ax.text(xx + 0.03, top / 1.12, f"{vv:.3g}", fontsize=7, color="#b00020", va="center")
            ax.hlines(min(np.median(v[:, m]), top / 1.12), x - w * 0.45, x + w * 0.45, color=COL[a], lw=2)
        ax.plot([], [], MK[a], color=COL[a], mfc="none" if a.endswith("hist") else COL[a],
                label=ARM[a] + (" (4 starts)" if a.endswith("hist") else " (8 starts)"))
    for m in range(pk.n - 1):
        ax.axvline(m + 0.5, color="#ddd", lw=0.8)
    ax.set_yscale("log")
    ax.set_xticks(range(pk.n), [f"{m}\n({c})" for m, c in zip(pk.mats, pk.cohort)], fontsize=8.5)
    ax.set_ylim(top=top)
    ax.set_ylabel("material peak-profile error e_prof\n(sum over 20 test probes, dimensionless), log")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.25, which="both")
    title(fig, "Independent angular test: peak-profile error per material and fitting start, all arms",
          "Markers = individual starts, bars = median over starts; 'development' / 'held out' = WL104D inspection cohort (every material is fitted)",
          pk.warn)
    tag(fig, "DESC")
    foot(fig, "Per (material, start): Σ over the 20 test probes of e_prof = Σ_grid |F − R| / Σ_grid R (luminance on the 0–8° peak grid). "
              "A held-out-cohort material is not zero-shot generalization: every material is fitted on its own angular data. Values above the axis "
              "top (3x the largest non-L-BFGS value) are drawn as edge triangles labelled with their value in red.")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.17)
    pk.save(fig, "test_peak_profile_error_per_material_and_start_all_arms.png",
            "Independent test: e_prof per material, dots per start, all arms.", "DESC", ["test_eval/arrays.npz:test__<arm>__e_prof"])


def test_regions(pk):
    arms = pk.arms + ["P_hist"]
    keys = [("core_l1", "specular core, < 3° from mirror\nregion relative L1 (luminance)"),
            ("shoulder_l1", "shoulder, 3–12° from mirror\nregion relative L1 (luminance)"),
            ("tail_l1", "tail, ≥ 12° from mirror\nregion relative L1 (luminance)"),
            ("el1_grazing", "grazing incidence (polar ≥ 75°)\nel1 over grazing probes")]
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.8))
    for ax, (k, nm) in zip(axs, keys):
        for j, a in enumerate(arms):
            v = pk.A(a, k).sum((1, 2))
            ax.plot(j + jitter(0, len(v), 0.5), v, MK[a], color=COL[a], ms=5.5, mfc="none" if a.endswith("hist") else COL[a])
            ax.hlines(np.median(v), j - 0.32, j + 0.32, color=COL[a], lw=2)
        ax.set_yscale("log")
        ax.set_xticks(range(len(arms)), [ARM[a] for a in arms], rotation=30, ha="right", fontsize=7.6)
        ax.set_ylabel("pooled over materials × probes (dimensionless), log")
        ax.set_title(nm, fontsize=9.5)
        ax.grid(axis="y", alpha=0.25, which="both")
    title(fig, "Independent angular test: error by angular region per arm (core, shoulder, tail, grazing)",
          "One marker per fitting start, bar = median over starts; region edges 3° and 12° from the mirror direction", pk.warn)
    tag(fig, "DESC")
    foot(fig, "Region relative L1 = Σ w |F − R| / Σ w R of luminance over the test lanes whose outgoing direction lies in the region "
              "(importance weights w estimate the solid-angle integral), per (material, probe), summed over all pairs. Grazing el1 = el1 summed over "
              "probes with incident polar ≥ 75°.")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.27, wspace=0.28)
    pk.save(fig, "test_region_errors_core_shoulder_tail_grazing_per_arm.png",
            "Independent test: core / shoulder / tail / grazing errors per arm, per start.", "DESC",
            ["test_eval/arrays.npz:test__<arm>__core_l1|shoulder_l1|tail_l1|el1_grazing"])


def test_peak_polar(pk):
    arms = [a for a in ("O0", "E2", "D") if a in pk.arms]
    pol = pk.z["polar_deg"]
    noise = 0.5 * np.abs(pk.A("O0", "s_h_a") - pk.A("O0", "s_h_b")).mean(0)    # [n, P]
    fig, axs = plt.subplots(3, 3, figsize=(17, 12), sharex=True)
    for m, ax in enumerate(axs.ravel()):
        o = np.argsort(pol[m])
        x = pol[m][o]
        ax.fill_between(x, -PCT * noise[m][o], PCT * noise[m][o], facecolor="none", edgecolor="#b00020", hatch="....", lw=0,
                        label="reference half-split noise (dotted red)" if m == 0 else None)
        for a in arms:
            s = PCT * pk.A(a, "s_h")[:, m][:, o]                    # [R, P]
            ax.fill_between(x, s.min(0), s.max(0), color=COL[a], alpha=0.15, lw=0)
            ax.plot(x, np.median(s, 0), "-" + MK[a], color=COL[a], ms=3.5, lw=1.4,
                    label=f"{ARM[a]}: median of 8 starts, band = start range" if m == 0 else None)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(f"{pk.mats[m]} ({pk.cohort[m]})", fontsize=9.5)
        ax.grid(alpha=0.25)
        if m % 3 == 0:
            ax.set_ylabel("mirror-peak height error,\nmax F / max R − 1 (%)")
        if m >= 6:
            ax.set_xlabel("incident polar angle of the test probe (deg)")
    axs[0, 0].legend(fontsize=7)
    title(fig, "Independent angular test: signed mirror-peak height error vs incidence per material — original Adam fit (O0) vs "
               "core+shoulder objective (E2) vs refined angular coverage (D)",
          "Peak height = maximum luminance over the 0–8° peak grid; 20 independently drawn test incidences per material", pk.warn)
    tag(fig, "DESC")
    foot(fig, "Line = median over the 8 fitting starts of the signed peak-height error s_h at each test probe; shaded band = range (min–max) "
              "over the starts, i.e. start-to-start instability. Probes of both azimuth halves (0–90° and 90–180°) are joined in polar order. "
              "Dotted red band = reference noise |s_h(half a) − s_h(half b)| / 2 (mean over "
              f"the O0 starts); it is ≤ {PCT * noise.max():.2f} % everywhere and therefore barely visible at these scales. First-bounce angular "
              "response of each fitted material (not a scene).")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.08, hspace=0.28, wspace=0.18)
    pk.save(fig, "test_mirror_peak_height_signed_error_vs_incidence_per_material_O0_E2_D.png",
            "Independent test: signed peak-height error vs incidence per material, median over starts with per-start range (O0, E2, D).",
            "DESC", ["test_eval/arrays.npz:test__<arm>__s_h, s_h_a, s_h_b, polar_deg"])


def test_instability(pk):
    arms = pk.arms + ["P_hist"]
    fig, ax = plt.subplots(figsize=(17, 5.8))
    w = 0.8 / len(arms)
    for j, a in enumerate(arms):
        I = PCT * pk.A(a, "s_h").std(0, ddof=1).mean(-1)
        ax.bar(np.arange(pk.n) + (j - (len(arms) - 1) / 2) * w, I, w * 0.92, color=COL[a],
               hatch="//" if a.endswith("hist") else None, alpha=0.6 if a.endswith("hist") else 1,
               label=f"{ARM[a]} (pooled I = {I.mean():.3g} %)" + (" — 4 starts" if a.endswith("hist") else ""))
    ax.set_yscale("log")
    ax.set_axisbelow(True)
    ax.set_xticks(range(pk.n), [f"{m}\n({c})" for m, c in zip(pk.mats, pk.cohort)], fontsize=8.5)
    ax.set_ylabel("instability I_m: SD over starts of the\nsigned peak-height error, mean over probes (%), log")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25, which="both")
    title(fig, "Independent angular test: across-start instability of the mirror-peak height per material and arm",
          "Protocol metric 'instability' (sample SD over starts, ddof 1); lower = the fitting result depends less on the start", pk.warn)
    tag(fig, "DESC")
    foot(fig, "I_m = mean over the 20 test probes of the standard deviation over fitting starts of s_h = max_grid F / max_grid R − 1. The pooled "
              "value (legend) is the mean over materials; the STABILITY_GAIN rule compares it with O0 (−25 %). WL104K P has only 4 starts, "
              "so its SD is not directly comparable.")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.85, bottom=0.17)
    pk.save(fig, "test_peak_height_instability_across_starts_per_material.png",
            "Independent test: across-start SD of peak height per material per arm.", "DESC", ["test_eval/arrays.npz:test__<arm>__s_h"])


def dev_eligibility(pk):
    el = (pk.Dv or {}).get("stage_F", {}).get("eligibility")
    if not el:
        pk.skipped.append("dev eligibility: dev.json missing")
        return
    arms = [a for a in ("O1", "D", "E2") if el.get(a, {}).get("status") == "EVALUATED"]
    spec = [("e_prof", "dev peak-profile error e_prof\neligible needs ≤ −10 % and CI upper < 0", -10),
            ("el1", "dev hemispherical el1\neligible needs ≤ +5 %", 5),
            ("instability", "dev across-start instability I\neligible needs ≤ +10 %", 10)]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, (k, nm, thr) in zip(axs, spec):
        contrast_panel(ax, [(j, a, el[a][k], True, "") for j, a in enumerate(arms)], thr)
        ax.set_xlim(-0.75, len(arms) - 0.4)
        ax.set_xticks(range(len(arms)), [textwrap.fill(ARM[a], 18) + ("\nELIGIBLE" if el[a]["eligible"] else "\nnot eligible")
                                         for a in arms], fontsize=8)
        ax.set_ylabel("relative change vs original Adam fit (O0) (%)")
        ax.set_title(nm, fontsize=9)
        ax.grid(axis="y", alpha=0.25, which="both")
    run = pk.Dv["stage_F"]["run"]
    title(fig, "Development set: stage-F eligibility of each fitting mechanism (decided before the test set was opened)",
          f"eligible: {pk.Dv['stage_F']['eligible'] or 'NONE'}; combination F {'RUN' if run else 'NOT_RUN (needs ≥ 2 eligible)'}")
    tag(fig, "PRE")
    foot(fig, "Dev lanes (seed 1041320) and dev peak grid; paired starts, median over 8 starts, 95 % material-bootstrap CI. Rules from protocol "
              "stage_F.eligibility; all must hold. The dev set is used only for this decision and for the E1 start selection.")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.2, wspace=0.3)
    pk.save(fig, "dev_stage_F_eligibility_vs_predeclared_rule.png", "Dev-set eligibility of O1 / D / E2 for the combination F.", "PRE",
            ["dev.json:stage_F.eligibility"])


# ===================================================================== (4) ring residual
def ring_profiles(pk):
    mats = [m for m in ("hard_1", "hard_53", "block_186") if m in pk.mats]
    arms = pk.arms + ["P_hist"]
    rad = pk.z["profile_radius_deg"]
    fig, axs = plt.subplots(len(mats), len(arms), figsize=(4 * len(arms), 3.6 * len(mats)), sharex=True)
    for i, mn in enumerate(mats):
        m = pk.mats.index(mn)
        lines = {a: PCT * np.median(pk.z[f"profile__{a}"][:, m], 1) for a in arms}      # [R, rings], median over probes
        base = np.concatenate([lines[a] for a in arms if a != "O1"])
        lo, hi = np.percentile(base, 0.5), np.percentile(base, 99.5)
        pad = 0.15 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        for j, a in enumerate(arms):
            ax = axs[i, j]
            L = lines[a]
            clipped = int(((L < lo) | (L > hi)).any(1).sum())
            for r in range(L.shape[0]):
                ax.plot(rad, np.clip(L[r], lo, hi), color=COL[a], lw=0.7, alpha=0.6)
            ax.plot(rad, np.clip(np.median(L, 0), lo, hi), color=COL[a], lw=2.4, label="median over starts")
            ax.axhline(0, color="k", lw=0.7)
            ax.set_ylim(lo, hi)
            ax.grid(alpha=0.25)
            t = f"{mn}: {ARM[a]}" + (" (4 starts)" if a.endswith("hist") else "")
            if clipped:
                t += f"\n({clipped} start line(s) clipped to the row scale)"
            ax.set_title(t, fontsize=8.6)
            if j == 0:
                ax.set_ylabel("ring residual, mean F / mean R − 1 (%)\n(median over the 20 test probes)")
            if i == len(mats) - 1:
                ax.set_xlabel("angle from the mirror direction (deg)")
    title(fig, "Independent angular test: full peak-profile residual per ring (0–8° from the mirror) — hard_1, hard_53, block_186",
          "Thin lines = individual fitting starts, thick = median over starts; same y scale within each material row", pk.warn)
    tag(fig, "DESC")
    foot(fig, "Ring residual = mean over the 16 azimuths of model luminance / mean of reference luminance − 1, per ring (radius 0, 0.25–4° "
              "in 0.25° steps, 5, 6, 8°), first-bounce angular response; median over the 20 independent test probes of each material. "
              "The row scale is set by the non-L-BFGS arms; L-BFGS start lines beyond it are clipped and counted in the panel title. "
              "No per-ring reference-noise band is stored (the half-split noise of the peak height, |s_h(a) − s_h(b)| / 2 over the O0 "
              f"starts, is ≤ {PCT * (0.5 * np.abs(pk.A('O0', 's_h_a') - pk.A('O0', 's_h_b')).mean(0)).max():.2f} % on this test set).",
         y=0.004)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.09, hspace=0.38, wspace=0.18)
    pk.save(fig, "test_ring_residual_profile_0_to_8deg_hard1_hard53_block186.png",
            "Independent test: ring residual 0–8° per arm for hard_1, hard_53 and block_186, per start and median.", "DESC",
            ["test_eval/arrays.npz:profile__<arm>, profile_radius_deg"])


# ===================================================================== (5) E1 selection
def e1_scatter(pk):
    if pk.Dv is None:
        pk.skipped.append("E1 scatter: dev.json missing")
        return
    S = np.array(pk.Dv["selection"]["S_dev"]["O0"])                 # [8, n]
    e = pk.A("O0", "e_prof").sum(-1)                                 # [8, n]
    e1 = pk.T["selection_E1"]["per_arm"]["O0"]
    sel, tsel = np.array(e1["selected_start"]), np.array(e1["training_selected_start"])
    rho = e1["spearman_S_dev_vs_test_e_prof"]
    fig, axs = plt.subplots(3, 3, figsize=(16, 13))
    for m, ax in enumerate(axs.ravel()):
        ax.plot(S[:, m], e[:, m], "o", color=COL["O0"], ms=6, label="O0 start")
        for r in range(S.shape[0]):
            ax.annotate(str(r + 1), (S[r, m], e[r, m]), fontsize=7, xytext=(4, 2), textcoords="offset points")
        ax.plot(S[sel[m], m], e[sel[m], m], "*", color=COL["E1"], ms=17, mec="k", label="dev-selected start (E1)")
        ax.plot(S[tsel[m], m], e[tsel[m], m], "s", mfc="none", mec="#eb6834", ms=13, mew=1.8, label="training-objective-selected start")
        ax.axhline(np.median(e[:, m]), color="#888", ls="--", lw=0.9, label="median over starts (test)")
        ax.set_title(f"{pk.mats[m]} ({pk.cohort[m]}): Spearman ρ = {rho[m]:+.2f}", fontsize=9.5)
        ax.grid(alpha=0.25)
        if m % 3 == 0:
            ax.set_ylabel("test peak-profile error e_prof\n(sum over 20 probes, dimensionless)")
        if m >= 6:
            ax.set_xlabel("dev selection criterion S_dev (mean el1 + ealb + core + shoulder)")
    axs[0, 0].legend(fontsize=7)
    lab = pk.T["selection_E1"]["primary_O0"]
    title(fig, "Start selection (E1) on the original Adam fit (O0): dev criterion vs independent test peak-profile error, 8 starts per material",
          f"PREDICTIVE rule (ρ > 0 in ≥ 7 of 9 materials and median ρ ≥ 0.5): {lab['n_positive_rho']} of 9 positive, median ρ "
          f"{lab['median_rho']:.2f} → PREDICTIVE = {lab['PREDICTIVE']}", pk.warn)
    tag(fig, "PRE")
    foot(fig, "S_dev = mean over the 20 dev probes of (el1 + ealb + e_core + e_shoulder) on independent dev lanes and incidences; the start with "
              "the lowest S_dev is selected per material (ties → lower start index). Numbers = start index (1 = generic). The selection never "
              "reads test or scene data.")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.07, hspace=0.3, wspace=0.22)
    pk.save(fig, "selection_E1_dev_criterion_vs_test_peak_error_per_material.png",
            "E1: per-material scatter of dev selection criterion vs test e_prof over 8 O0 starts; selected start highlighted.", "PRE",
            ["dev.json:selection.S_dev.O0", "test_eval/arrays.npz:test__O0__e_prof", "test.json:selection_E1.per_arm.O0"])


def e1_bars(pk):
    per = pk.T["selection_E1"]["per_arm"]
    arms = [a for a in ("O0", "O1", "D", "E2", "F") if a in per]
    fig, axs = plt.subplots(1, 2, figsize=(17, 5.8))
    for ax, k, nm in ((axs[0], "e_prof", "peak-profile error e_prof"), (axs[1], "el1", "hemispherical el1")):
        w = 0.26
        for j, a in enumerate(arms):
            d = per[a][k]
            vals = [d["selected"], d["median_start"], d["training_selected"]]
            for q, (v, h, lb) in enumerate(zip(vals, (None, "..", "//"), ("dev-selected start (E1)", "per-material median start",
                                                                          "training-objective-selected start"))):
                ax.bar(j + (q - 1) * w, v, w * 0.92, color=COL[a], alpha=(1, 0.45, 0.7)[q], hatch=h, edgecolor="white",
                       label=lb if j == 0 else None)
            if k == "e_prof":
                ax.hlines(0.9 * d["median_start"], j - 1.5 * w, j + 1.5 * w, color="#b00020", ls="--", lw=1.1,
                          label="rule: selected ≤ 0.9 × median start" if j == 0 else None)
            else:
                ax.hlines(1.05 * d["median_start"], j - 1.5 * w, j + 1.5 * w, color="#b00020", ls="--", lw=1.1,
                          label="rule: selected ≤ 1.05 × median start" if j == 0 else None)
            eff = per[a]["labels"]["SELECTION_EFFECTIVE"]
            rel = PCT * (d["selected"] / d["median_start"] - 1)
            ax.text(j, max(vals + [0.9 * d["median_start"] if k == "e_prof" else 1.05 * d["median_start"]]) * 1.03,
                    f"selected / median − 1 = {rel:+.1f} %\nSELECTION_EFFECTIVE {'yes' if eff else 'no'} ({per[a]['role'].split(' ')[0]})",
                    ha="center", fontsize=7.2)
        ax.set_xticks(range(len(arms)), [ARM[a] for a in arms], fontsize=8, rotation=10)
        ax.set_ylabel(f"pooled test {k}\n(sum over 180 material × probe pairs)")
        ax.set_title(f"{nm}: selected vs median vs training-objective-selected starts", fontsize=9.5)
        ax.legend(fontsize=7.3, loc="upper right", ncol=2)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(max(per[a][k][q] for q in ("selected", "median_start", "training_selected")) for a in arms) * 1.45)
    title(fig, "Start selection (E1): pooled test error of the dev-selected starts vs the median start and the training-objective selection",
          "Primary: original Adam fit (O0); other arms descriptive. SELECTION_EFFECTIVE also needs e_prof ≤ training-selected and RESOLVED",
          pk.warn)
    tag(fig, "PRE")
    foot(fig, "Median start = sum over materials of the per-material median over the 8 starts of the material-pooled metric. Training-objective "
              "selection = the start with the lowest final training objective (WL104J / WL104K rule). The SELECTION_EFFECTIVE label (one per "
              "arm, from the combined e_prof and el1 rule) is repeated in both panels. An arm with extreme starts (L-BFGS) has an inflated median "
              "start, which makes selection look effective. Selection success and fitting success are separate questions.")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.2, wspace=0.2)
    pk.save(fig, "selection_E1_selected_vs_median_vs_training_selected_bars.png",
            "E1: pooled test e_prof / el1 of selected vs median vs training-selected starts per arm, with rule lines.", "PRE",
            ["test.json:selection_E1.per_arm"])


# ===================================================================== (6) synthetic controls
def synthetic(pk):
    sy = pk.T.get("synthetic", {})
    P, C = sy.get("pooled", {}), sy.get("controls", {})
    arms = [a for a in ("S_O0", "S_O1", "S_D", "S_E2") if a in P]
    if not arms:
        pk.skipped.append("synthetic controls: none available")
        return
    fig, axs = plt.subplots(1, 4, figsize=(21, 6))
    med0 = {k: np.median(P["S_O0"][k]) for k in ("e_prof", "el1", "objective")}

    def dots(ax, k):
        for j, a in enumerate(arms):
            v = np.array(P[a][k])
            ax.plot(j + jitter(0, len(v), 0.4), v, MK[a], color=COL[a], ms=7)
            ax.hlines(np.median(v), j - 0.3, j + 0.3, color=COL[a], lw=2.2)
        ax.set_xticks(range(len(arms)), [ARM[a].replace("synthetic: ", "") for a in arms], rotation=18, fontsize=7.8)
        ax.grid(axis="y", alpha=0.25, which="both")

    ax = axs[0]
    dots(ax, "e_prof")
    ax.axhline(0.8 * med0["e_prof"], color="#b00020", ls="--", lw=1.1, label="SC_D / SC_E2 pass: ≤ 0.8 × S-O0 median")
    ax.set_yscale("log")
    ax.set_ylabel("truth-referenced pooled peak-profile error e_prof, log")
    ax.set_title(f"peak error vs known truth\nSC_D {'PASS' if C.get('SC_D', {}).get('passed') else 'FAIL'} "
                 f"(ratio {C.get('SC_D', {}).get('e_prof_ratio', np.nan):.2f}), SC_E2 {'PASS' if C.get('SC_E2', {}).get('passed') else 'FAIL'} "
                 f"(ratio {C.get('SC_E2', {}).get('e_prof_ratio', np.nan):.2f})", fontsize=9)
    ax.legend(fontsize=7.3)
    ax = axs[1]
    dots(ax, "el1")
    ax.axhline(1.05 * med0["el1"], color="#b00020", ls="--", lw=1.1, label="SC_O1 pass: S-O1 ≤ 1.05 × S-O0 median")
    ax.set_ylabel("truth-referenced pooled el1 (dimensionless)")
    ax.set_title(f"hemispherical error vs known truth\nS-O1 / S-O0 = {C.get('SC_O1', {}).get('el1_ratio', np.nan):.2f}", fontsize=9)
    ax.legend(fontsize=7.3)
    ax = axs[2]
    dots(ax, "objective")
    ax.axhline(med0["objective"], color="#b00020", ls="--", lw=1.1, label="SC_O1 pass: S-O1 median ≤ S-O0 median")
    ax.set_ylabel("final training objective (sum over 9 materials)")
    ax.set_title(f"training objective on the noisy synthetic targets\nSC_O1 {'PASS' if C.get('SC_O1', {}).get('passed') else 'FAIL'} "
                 "(S-D and S-E2 objectives are on other lanes / functionals)", fontsize=9)
    ax.legend(fontsize=7.3)
    ax = axs[3]
    e1 = C.get("SC_E1", {})
    if "selected_start" in e1:
        e = pk.z["synthetic__S_O0__e_prof"].sum(-1)                 # [4, n]
        sel = np.array(e1["selected_start"])
        x = np.arange(pk.n)
        ax.bar(x - 0.2, e[sel, x], 0.38, color=COL["E1"], label="dev-selected start")
        ax.bar(x + 0.2, np.median(e, 0), 0.38, color="#bbb", label="median start")
        for m in range(pk.n):
            ax.text(m, max(e[sel[m], m], np.median(e[:, m])) * 1.08, "ok" if e1["material_ok"][m] else "no", ha="center", fontsize=7.5)
        ax.set_xticks(x, pk.mats, rotation=40, fontsize=7.8)
        ax.set_yscale("log")
        ax.set_ylabel("truth-referenced material e_prof (sum over probes), log")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.6)
        ax.legend(fontsize=7.3)
        ax.set_title(f"E1 selection on S-O0 (4 starts)\nSC_E1 {'PASS' if e1.get('passed') else 'FAIL'}: selected ≤ median in "
                     f"{e1.get('n_ok')} of 9 (needs ≥ 6)", fontsize=9)
        ax.grid(axis="y", alpha=0.25, which="both")
    else:
        ax.axis("off")
        ax.text(0, 0.5, "SC_E1 not available", fontsize=10)
    title(fig, "Synthetic detection controls: can each intervention correct its intended mechanism when the truth is known and exactly representable?",
          "Truth = ER-v1-GF(R0) with a sharp isotropic outer lobe (alpha 0.03) and the WL104K SP gates; noisy targets at the physical set's "
          "per-lane noise; 4 starts per arm")
    tag(fig, "PRE")
    pr = "; ".join(f"{ctl.replace('SC_', 'S-')} / S-O0 per start " + " / ".join(f"{v:.2f}" for v in C[ctl]["paired_per_start_ratio"])
                   for ctl in ("SC_D", "SC_E2") if "paired_per_start_ratio" in C.get(ctl, {}))
    foot(fig, "Evaluation is against the noise-free truth on the independent test lanes and test peak grid. A failed control means the "
              "intervention could not demonstrate its mechanism on a known case; a negative physical result is then CONTROL_NOT_DEMONSTRATED "
              "rather than evidence that the mechanism is absent. Controls test detection, not physical validity. With 4 starts the median is the "
              "mean of the middle two, so the SC ratios depend on few starts (descriptive paired ratios, starts 1-4: " + pr + ").")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.2, wspace=0.3)
    pk.save(fig, "synthetic_controls_truth_referenced_errors_and_pass_rules.png",
            "Synthetic controls: truth-referenced e_prof, el1, training objective and E1 selection per synthetic arm, with pass rules.", "PRE",
            ["test.json:synthetic.pooled", "test.json:synthetic.controls", "test_eval/arrays.npz:synthetic__S_O0__e_prof"])


# ===================================================================== (7) scenes
SC_ARMS = ("O1", "D", "E2", "F", "E1")
# scene events, named without the arm codes E1 / E2 (which here mean selected start / core+shoulder objective)
EVN = {"total": "TOTAL (higher orders fixed from R0)", "e1": "first bounce only"}
TOTAL_DEF = ("TOTAL = first bounce + second bounce (both replayed per arm with the same fixed proposal directions) + higher-order bounces "
             "taken from R0 (identical for every arm)")


def scene_list(pk, key=None):
    return [s for s in SCENES if s in pk.scenes and "_stub" not in pk.scenes[s] and (key is None or pk.scenes[s].get(key))]


def scene_grid(n):
    """rows = scenes, cols = (TOTAL, first bounce); horizontal panels with the arm names on the y axis."""
    fig, axs = plt.subplots(n, 2, figsize=(16, 1.2 + 3.3 * n + 1.3), squeeze=False)
    return fig, axs


def scene_finish(fig, n, axs=None, extra_bottom=0.0):
    if axs is not None:                     # the right column repeats the arm labels of the left column
        for ax in axs[:, 1]:
            ax.tick_params(axis="y", labelleft=False)
    H = fig.get_figheight()
    fig.subplots_adjust(left=0.2, right=0.98, top=1 - 1.05 / H, bottom=(1.15 + extra_bottom) / H, hspace=0.34,
                        wspace=0.08)


def scenes_contrast(pk):
    have = scene_list(pk, "contrasts")
    stubs = [s for s in SCENES if s in pk.scenes and "_stub" in pk.scenes[s]]
    if not have:
        pk.skipped.append("scene contrasts: no scenes/<scene>/analysis104kb.json with contrasts yet"
                          + (f" (identity-failure stubs: {stubs})" if stubs else ""))
        return
    fig, axs = scene_grid(len(have))
    for k, s in enumerate(have):
        A = pk.scenes[s]
        for i, ev in enumerate(("total", "e1")):
            ax = axs[k, i]
            con = A["contrasts"]
            items = [(a, con[a][ev], True, ARM[a] + (" vs O0 start-mean" if a == "E1" else "")) for a in SC_ARMS if a in con and ev in con[a]]
            dE1 = A.get("contrasts_descriptive_E1")
            if isinstance(dE1, dict) and ev in dE1:
                items.append(("E1", dE1[ev], False, "selected start (E1) vs other 7 O0 starts"))
            for a in ("P_hist", "A_hist"):
                dh = (A.get("contrasts_descriptive_hist") or {}).get(a)
                if isinstance(dh, dict) and ev in dh:
                    items.append((a, dh[ev], False, ARM[a] + ", 4 paired starts"))
            KEEP = 150.0                       # axis scaled to the moderate contrasts; larger values pinned at the edge with their number
            vals = [0.0, 2.0]
            for a, c, filled, lab in items:
                if c.get("point") is None:
                    continue
                if max(abs(PCT * c["lo"]), abs(PCT * c["hi"])) <= KEEP:
                    vals += [PCT * c["lo"], PCT * c["hi"]]
                    vals += [PCT * v for v in (c.get("per_start_rel") or []) if v is not None and abs(PCT * v) <= KEEP]
                else:
                    vals.append(float(np.clip(PCT * c["point"], -KEEP, KEEP)))
            xlo, xhi = min(vals), max(vals)
            pad = 0.08 * (xhi - xlo)
            xlo, xhi = xlo - pad, xhi + pad
            span = xhi - xlo
            unres = []
            for j, (a, c, filled, lab) in enumerate(items):
                if c.get("point") is None:      # execution fix c8bf685: O0 at the replicate-noise floor, relative change undefined
                    unres.append((j, a, c))
                    continue
                m, l, h = PCT * c["point"], PCT * c["lo"], PCT * c["hi"]
                mc, lc, hc = np.clip([m, l, h], xlo, xhi)
                off = not (xlo <= m <= xhi) or l < xlo or h > xhi
                ax.errorbar(mc, j, xerr=[[mc - lc], [hc - mc]], fmt=MK.get(a, "o"), color=COL.get(a, "k"), ms=8, capsize=4,
                            mfc=COL.get(a, "k") if filled else "white", mew=1.5)
                if m > xhi:
                    ax.plot(xhi, j, ">", color=COL.get(a, "k"), ms=7, clip_on=False)
                ps = c.get("per_start_rel")
                if ps and len(ps) > 1:
                    ps = PCT * np.array([np.nan if v is None else v for v in ps], float)
                    ys = j + 0.28 + jitter(0, len(ps), 0.12)
                    inr = (ps >= xlo) & (ps <= xhi)
                    ax.plot(ps[inr], ys[inr], ".", color=COL.get(a, "k"), ms=5)
                    ax.plot(np.full((ps > xhi).sum(), xhi), ys[ps > xhi], ">", color=COL.get(a, "k"), ms=4, alpha=0.8, clip_on=False)
                    ax.plot(np.full((ps < xlo).sum(), xlo), ys[ps < xlo], "<", color=COL.get(a, "k"), ms=4, alpha=0.8, clip_on=False)
                txt = (f"{m:+.1f} %" + (f" [{l:+.0f}, {h:+.0f}] off scale" if off else "")
                       + (f"  {c.get('label', '')}" if filled else "  (descriptive)"))
                col = ("#b00020" if filled and c.get("label") == "REGRESSED" else "#1b6b3a" if filled and c.get("label") == "IMPROVED"
                       else "#333")
                if hc > xhi - 0.3 * span:           # no room on the right: write the value to the left of the marker
                    ax.annotate(txt, (lc, j), xytext=(-6, -2), textcoords="offset points", fontsize=7.4, va="center", ha="right", color=col)
                else:
                    ax.annotate(txt, (hc, j), xytext=(6, -2), textcoords="offset points", fontsize=7.4, va="center", color=col)
            ax.axvline(0, color="k", lw=0.8)
            ax.axvline(2, color="#b00020", ls="--", lw=1)
            if len(unres) == len(items):
                xlo, xhi = -10.0, 60.0
            for j, a, c in unres:
                ax.plot([0], [j], marker=MK.get(a, "o"), color="#999", mfc="none", ms=8, mew=1.2)
                ax.annotate(f"undefined (noise floor): row RMS {c.get('x_rms', float('nan')):.2g} vs comparison "
                            f"{c.get('o0_rms', float('nan')):.2g}; not labelled",
                            (0, j), xytext=(8, 0), textcoords="offset points", fontsize=7.2, va="center", color="#666", style="italic")
            ax.set_xlim(xlo, xhi)
            ax.set_yticks(range(len(items)), [it[3] for it in items], fontsize=7.6)
            ax.set_ylim(len(items) - 0.4, -0.6)
            ax.set_title(f"{SCENE_NAME[s]} — {EVN[ev]}", fontsize=9)
            ax.set_xlabel("relative change of objects stable RMS vs original Adam fit (O0) (%)")
            ax.grid(axis="x", alpha=0.25)
    title(fig, "Scenes: relative change of objects stable RMS vs the original Adam fit (O0) per arm, replicate-jackknife CI and per-start values",
          "Fixed-proposal replay (same sampled directions for every closure); IMPROVED = CI upper < 0, REGRESSED = CI lower > +2 % (dashed red "
          "line), else NEUTRAL; axis limited to ±150 %, larger values pinned at the edge (arrow) with their number")
    tag(fig, "PRE")
    foot(fig, "Large markers: X vs O0 relative change of the start-mean objects stable RMS (noise-debiased) with a delete-one-replicate jackknife "
              "t(11) 95 % CI over 12 render replicates; filled markers carry the predeclared scene rule (label printed), open markers are "
              "descriptive companions (E1 vs the other 7 O0 starts; historical A / P paired with the O0 starts of equal key). E1 = O0 at its "
              "dev-selected start vs the O0 start-mean (no scene-based selection). Small dots: per-start paired relative change (no per-start CI "
              "is stored). Grey open markers at 0 with italic text: relative change undefined because the O0 noise-debiased RMS (or one of its jackknife "
              "replicates) is clipped at 0, i.e. O0 is at the replicate-noise floor (execution fix c8bf685: reported with absolute RMS, never "
              "labelled). " + TOTAL_DEF + ": not an end-to-end ECRO validation. hard_1 is a development regression case, not independent "
              "evidence." + (f" Scenes without usable analysis: {stubs}." if stubs else "")
              + (f" Scenes not yet analysed: {[x for x in SCENES if x not in pk.scenes]}." if len(pk.scenes) < len(SCENES) else ""))
    scene_finish(fig, len(have), axs, extra_bottom=0.3)
    pk.save(fig, "scenes_relative_change_vs_O0_objects_total_and_first_bounce.png",
            "Scenes: X vs O0 relative change of objects TOTAL and E1 stable RMS with jackknife CI and per-start values.", "PRE",
            [f"scenes/{s}/analysis104kb.json:contrasts, contrasts_descriptive_E1, contrasts_descriptive_hist" for s in have])


def scenes_rms_per_start(pk):
    have = scene_list(pk, "arm_summary")
    if not have:
        pk.skipped.append("scene per-start RMS: no arm_summary in the analyses yet")
        return
    fig, axs = scene_grid(len(have))
    for k, s in enumerate(have):
        S = pk.scenes[s]["arm_summary"]
        arms = [a for a in ("O0", "O1", "D", "E2", "F", "P_hist", "A_hist") if a in S]
        sel = (pk.scenes[s].get("selected_E1") or {}).get("O0", {}).get("start")
        for i, ev in enumerate(("total", "e1")):
            ax = axs[k, i]
            for j, a in enumerate(arms):
                d = S[a]["objects_rms"][ev]
                v = np.array(d["per_start"], float)
                ys = j + jitter(0, len(v), 0.45)
                ax.plot(v, ys, MK.get(a, "o"), color=COL.get(a, "k"), ms=5, mfc="none" if a.endswith("hist") else COL.get(a, "k"))
                ax.vlines(d["start_mean"], j - 0.3, j + 0.3, color=COL.get(a, "k"), lw=2.2)
                sd = d.get("sd")
                ax.annotate(f"mean {d['start_mean']:.4g}" + (f", SD {sd:.2g}" if sd is not None else ""), (np.nanmax(v), j),
                            xytext=(8, 0), textcoords="offset points", fontsize=7.2, va="center", color="#333")
                if a == "O0" and sel is not None and sel < len(v):
                    ax.plot(v[sel], ys[sel], "*", color=COL["E1"], ms=15, mec="k", label=f"O0 start {sel + 1}: selected on dev (E1)")
            ax.set_yticks(range(len(arms)), [ARM[a] + (" (4 starts)" if a.endswith("hist") else "")
                                             + (f"\n(star: start {sel + 1}, dev-selected E1)" if a == "O0" and sel is not None else "")
                                             for a in arms], fontsize=7.6)
            ax.set_ylim(len(arms) - 0.4, -0.6)
            lo, hi = ax.get_xlim()
            ax.set_xlim(lo, hi + 0.3 * (hi - lo))
            ax.set_title(f"{SCENE_NAME[s]} — {EVN[ev]}", fontsize=9)
            ax.set_xlabel("objects stable RMS error, noise-debiased, relative to the reference TOTAL RMS (dimensionless)")
            ax.grid(axis="x", alpha=0.25)
    title(fig, "Scenes: objects stable RMS error per fitting start and arm (TOTAL and first bounce), start-mean and across-start SD",
          "One marker per fitting start (12-replicate fixed-proposal replay); vertical bar = start-mean; SD = across-start standard deviation")
    tag(fig, "DESC")
    foot(fig, "Objects stable RMS of each closure itself (not a change vs O0) against the layered reference (the quantity behind the relative contrasts); lower is "
              "better. The across-start SD is the protocol's descriptive across-start consistency. Historical A / P have 4 starts (open markers). "
              "Stable RMS = sqrt(noise-debiased mean square of the RGB residual over the objects / mean square of the reference TOTAL over the "
              "objects), the same denominator for both events; noise-debiasing clips at 0 (values at 0 = at the replicate-noise floor). "
              + TOTAL_DEF + ".")
    scene_finish(fig, len(have), axs)
    pk.save(fig, "scenes_objects_stable_rms_per_start_per_arm_total_and_first_bounce.png",
            "Scenes: absolute objects stable RMS per start and arm with start-mean and across-start SD.", "DESC",
            [f"scenes/{s}/analysis104kb.json:arm_summary, selected_E1" for s in have])


def tonemap(x, white):
    return np.clip(x / white, 0, 1) ** (1 / 2.2)


def scene_images(pk):
    have = [s for s in scene_list(pk) if (pk.scenes[s]["_dir"] / "maps.npz").exists()]
    if not have:
        pk.skipped.append("scene images: no maps.npz yet")
        return
    for s in have:
        with np.load(pk.scenes[s]["_dir"] / "maps.npz") as z:
            m = {k: z[k] for k in z.files}
        obj = m["objects_mask"].astype(bool)
        arms = [a for a in ("O0", "E2", "D", "O1", "F", "E1") if f"{a}_total_lum" in m]
        errk = [(f"{a}_{ev}_abs_err") for ev in ("total", "e1") for a in arms]
        M = float(max(np.nanmax(np.where(obj, m[k], np.nan)) for k in errk))
        ref = m["reference_total_lum"]
        white = float(np.percentile(ref[obj], 99.5)) if obj.any() else float(ref.max())
        ncol = len(arms) + 1
        fig, axs = plt.subplots(4, ncol, figsize=(2.9 * ncol + 1.4, 14.4))
        iy, ix = np.unravel_index(np.argmax(np.where(obj, ref, -np.inf)), ref.shape)   # reference highlight peak in the objects
        hw = max(8, ref.shape[0] // 12)
        y0, y1 = max(0, iy - hw), min(ref.shape[0], iy + hw)
        x0, x1 = max(0, ix - hw), min(ref.shape[1], ix + hw)
        crop = (slice(y0, y1), slice(x0, x1))
        cw = float(ref[crop].max())
        axs[3, 0].imshow(tonemap(ref[crop], cw), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axs[3, 0].set_title(f"reference TOTAL, highlight zoom\n{y1 - y0}x{x1 - x0} px, exposure / {cw:.3g}", fontsize=8.3)
        axs[0, 0].add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, ec="#2a78d6", lw=1.2))
        axs[0, 0].imshow(tonemap(ref, white), cmap="gray", vmin=0, vmax=1)
        axs[0, 0].set_title("layered reference, TOTAL\n(12-replicate mean)", fontsize=8.3)
        axs[1, 0].imshow(np.where(obj, 1.0, 0.3), cmap="gray", vmin=0, vmax=1)
        axs[1, 0].set_title("objects mask (error maps\nonly inside the mask)", fontsize=8.3)
        axs[2, 0].imshow(tonemap(m["reference_e1_lum"], white), cmap="gray", vmin=0, vmax=1)
        axs[2, 0].set_title("layered reference,\nfirst bounce E1", fontsize=8.3)
        norm = PowerNorm(0.5, vmin=0, vmax=M)
        im = None
        sel = (pk.scenes[s].get("selected_E1") or {}).get("O0", {}).get("start")
        signed = {a: np.where(obj, m[f"{a}_total_lum"] - ref, np.nan)[crop] for a in arms}
        Ms = float(max(np.nanmax(np.abs(v)) for v in signed.values()))   # shared symmetric scale of the zoom row

        def ssqrt(x):
            return np.sign(x) * np.sqrt(np.abs(x) / Ms)
        for j, a in enumerate(arms):
            nm = ARM[a] if a != "E1" else f"selected start (E1)\nO0 start {sel + 1 if sel is not None else '?'}"
            mean_txt = "single start" if a == "E1" else "start-mean"
            axs[0, j + 1].imshow(tonemap(m[f"{a}_total_lum"], white), cmap="gray", vmin=0, vmax=1)
            axs[0, j + 1].set_title(f"{nm}\nTOTAL, {mean_txt}", fontsize=8.3)
            ez = signed[a]
            imz = axs[3, j + 1].imshow(ssqrt(ez), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            axs[3, j + 1].set_title(f"model − reference (signed), TOTAL, zoom\nmax {np.nanmax(ez):+.3g}, min {np.nanmin(ez):+.3g}",
                                    fontsize=7.8)
            for i, ev in ((1, "total"), (2, "e1")):
                e = m[f"{a}_{ev}_abs_err"]
                im = axs[i, j + 1].imshow(np.where(obj, e, np.nan), cmap="magma", norm=norm)
                axs[i, j + 1].set_title(f"|model − reference|, {'TOTAL' if ev == 'total' else 'first bounce'}\n"
                                        f"objects mean {np.nanmean(np.where(obj, e, np.nan)):.3g}", fontsize=7.8)
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#1a1a1a")
        H = fig.get_figheight()
        fig.subplots_adjust(left=0.01, right=0.9, top=1 - 1.0 / H, bottom=1.0 / H, hspace=0.24, wspace=0.04)
        pos1, pos2 = axs[2, 1].get_position(), axs[1, 1].get_position()
        cb = fig.colorbar(im, cax=fig.add_axes([0.915, pos1.y0, 0.012, pos2.y1 - pos1.y0]))
        cb.set_label(f"absolute luminance error (linear HDR radiance)\nshared scale 0 – {M:.3g}, square-root display", fontsize=8)
        pz = axs[3, 1].get_position()
        cz = fig.colorbar(imz, cax=fig.add_axes([0.915, pz.y0, 0.012, pz.y1 - pz.y0]))
        tv = np.array([-1, -0.25, -1 / 16, 0, 1 / 16, 0.25, 1]) * Ms
        cz.set_ticks(ssqrt(tv), labels=[f"{v:+.2g}" for v in tv])
        cz.ax.tick_params(labelsize=7)
        cz.set_label("signed error, model − reference\n(red = model too bright), sqrt display", fontsize=8)
        title(fig, f"{SCENE_NAME[s]}: HDR images and absolute error maps on one shared scale (rows: image, TOTAL error, first-bounce "
                   "error, highlight zoom of the TOTAL error)",
              f"Images: linear luminance / {white:.3g} (99.5th percentile of the reference over objects), gamma 2.2, identical exposure; "
              "start-mean over each arm's 8 starts (E1 column: the single dev-selected O0 start)")
        tag(fig, "DESC")
        foot(fig, "Error maps = |model luminance − layered reference luminance| per pixel inside the objects mask (replicate-mean images), one "
                  "absolute colour scale for every map in this figure (no per-image normalization; each scene has its own scale because "
                  "its radiance level differs), square-root colour mapping. Per-pixel errors are NOT noise-debiased: they include the Monte "
                  "Carlo noise of both 12-replicate means (speckle on diffuse faces), unlike the stable RMS of the scene contrast figures. "
                  + TOTAL_DEF + ". Bottom row: SIGNED TOTAL error (model − reference) cropped around the brightest "
                  "reference pixel inside the objects (blue box in the reference image), on its own symmetric scale shared by all arms "
                  "(signed square-root display; red = overshoot). A start-mean image averages the starts' "
                  "renders, so it can be closer to the reference than any single start.")
        short = s.replace("test83_", "")
        pk.save(fig, f"scene_{short}_start_mean_images_and_absolute_error_maps.png",
                f"{SCENE_NAME[s]}: start-mean images (reference, O0, E2, D, O1, E1 start) and shared-scale absolute error maps.", "DESC",
                [f"scenes/{s}/maps.npz"])


def scene_peak_pixels(pk):
    have = scene_list(pk, "peak_pixels")
    if not have:
        pk.skipped.append("scene peak-pixel concentration: no analyses yet")
        return
    fig, axs = scene_grid(len(have))
    for k, s in enumerate(have):
        pp = pk.scenes[s]["peak_pixels"]
        arms = [a for a in ("O1", "D", "E2", "F", "P_hist", "A_hist") if a in pp]
        for i, (key, nm) in enumerate((("top20_share_of_abs_change", "share of |ΔSE| carried by the 20 largest pixels (%)"),
                                       ("total_change_of_O0_SE", "net squared-error change vs O0, % of O0's squared error"))):
            ax = axs[k, i]
            for j, a in enumerate(arms):
                v = [pp[a][r]["total"][key] for r in sorted(pp[a], key=int)]
                v = PCT * np.array([np.nan if x is None else x for x in v], float)
                ax.plot(v, j + jitter(0, len(v), 0.45), MK.get(a, "o"), color=COL.get(a, "k"), ms=5,
                        mfc="none" if a.endswith("hist") else COL.get(a, "k"))
                ax.vlines(np.nanmedian(v), j - 0.3, j + 0.3, color=COL.get(a, "k"), lw=2)
            ax.set_yticks(range(len(arms)), [ARM[a] for a in arms], fontsize=7.6)
            ax.set_ylim(len(arms) - 0.4, -0.6)
            if i == 1:                          # historical A reaches several hundred %: symmetric log beyond ±50 %
                ax.set_xscale("symlog", linthresh=50, linscale=1.5)
                ax.axvspan(-50, 50, color="#f2f2f2", zorder=0, lw=0)
                xl = ax.get_xlim()
                tk = [t for t in (-300, -50, -25, 0, 25, 50, 300, 1000) if xl[0] <= t <= xl[1]]      # no +100: overprints +50 on symlog
                ax.set_xticks(tk, [f"{t:+d}" if t else "0" for t in tk])
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlabel(nm + (" (symlog: linear inside the shaded ±50 %)" if i == 1 else ""))
            ax.axvline(0, color="k", lw=0.6)
            ax.grid(axis="x", alpha=0.25)
            ax.set_title(f"{SCENE_NAME[s]} — TOTAL, paired with the O0 start of equal index", fontsize=9)
    floor = [x for x in have if any((c.get("total") or {}).get("label") == "UNRESOLVED_NOISE_FLOOR"
                                    for c in (pk.scenes[x].get("contrasts") or {}).values())]
    title(fig, "Scenes: is the change vs O0 concentrated in a few highlight peak pixels? (TOTAL luminance, objects, per start)",
          "Left: share of the absolute squared-error change carried by the 20 largest-change pixels; right: net squared-error change vs O0")
    tag(fig, "DESC")
    foot(fig, "Squared error of the replicate-mean luminance vs the layered reference, model X vs the original Adam fit (O0) at the same start "
              "index (A / P paired by start key). Markers = starts, bar = median. A high top-20 share means the scene contrast is decided by "
              "a handful of mirror-peak pixels; a negative net change means X has less squared error than O0. The squared errors are NOT "
              "noise-debiased (replicate-mean images): where O0 is near the replicate-noise floor both quantities are dominated by Monte Carlo "
              "noise" + (f" (O0 TOTAL at the floor: {', '.join(x.replace('test83_', '') for x in floor)})" if floor else "") + ".")
    scene_finish(fig, len(have), axs)
    pk.save(fig, "scenes_peak_pixel_concentration_per_start.png", "Scenes: concentration of the squared-error change in peak pixels.",
            "DESC", [f"scenes/{s}/analysis104kb.json:peak_pixels" for s in have])


# ===================================================================== (8) historical secondary test
def historical(pk):
    H = pk.T.get("secondary_historical", {}).get("contrasts_vs_O0", {})
    C = pk.T["contrasts_vs_O0"]
    arms = [a for a in ("O1", "D", "E2", "F") if a in H]
    keys = [("e_prof", "peak-profile error e_prof"), ("e_h", "peak-height error |s_h|"), ("el1", "hemispherical el1"),
            ("instability", "across-start instability I")]
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.8))
    for ax, (k, nm) in zip(axs, keys):
        items = []
        for j, a in enumerate(arms):
            items.append((1.6 * j - 0.2, a, {q: v for q, v in C[a][k]["all"].items() if q != "per_start"}, True, ""))
            items.append((1.6 * j + 0.25, a, {q: v for q, v in H[a][k]["all"].items() if q != "per_start"}, False, "", 0.12))
        contrast_panel(ax, items)
        ax.set_xlim(-1.0, 1.6 * (len(arms) - 1) + 1.0)
        ax.set_xticks(1.6 * np.arange(len(arms)), [ARM[a] for a in arms], rotation=15, fontsize=8)
        ax.set_ylabel("relative change vs original Adam fit (O0) (%)")
        ax.set_title(nm, fontsize=9.5)
        ax.grid(axis="y", alpha=0.25, which="both")
    axs[0].legend(handles=[plt.Line2D([], [], marker="o", color="#555", ls="none", ms=8, label="filled: independent test (primary)"),
                           plt.Line2D([], [], marker="o", color="#555", mfc="white", ls="none", ms=8, label="open: WL104D Amendment-1 (secondary)")],
                  fontsize=7.5, loc="lower left")
    title(fig, "Historical secondary test (WL104D Amendment-1 reference) vs the independent test: relative change vs O0 per arm",
          "Secondary reference: fixed WL104D incidence grid (peak grid 0–4°), inspected repeatedly in earlier batches — never used for fitting "
          "or selection here", pk.warn)
    tag(fig, "SEC")
    foot(fig, "Filled = independent WL104K-B test set (primary, peak grid 0–8°); open = WL104D Amendment-1 reference (secondary). Same contrast "
              "definition (paired starts, median, material bootstrap CI). Agreement between the two reference sets indicates that a contrast is "
              "not specific to one incidence sample.")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.83, bottom=0.17, wspace=0.3)
    pk.save(fig, "historical_WL104D_secondary_test_vs_independent_test_contrasts.png",
            "Secondary WL104D reference vs primary test: relative changes vs O0 (e_prof, e_h, el1, instability).", "SEC",
            ["test.json:secondary_historical.contrasts_vs_O0", "test.json:contrasts_vs_O0"])


# ===================================================================== index / manifest
def rows_from_decision(DC):
    out = []
    for X, r in DC["rows"].items():
        if X == "E1":
            peak, stab = f"PREDICTIVE {r.get('predictive', '')}", f"SELECTION_EFFECTIVE {r.get('selection_effective', '')}"
            ep = r.get("e_prof_change_vs_median_start")
            ep = {"median": ep} if isinstance(ep, (int, float)) else {}
        else:
            peak, stab, ep = r.get("peak_gain", ""), r.get("stability_gain", ""), r.get("e_prof_change", {})
        sc = r.get("scenes") or {}
        out.append({"row": f"{r['mechanism']} ({X})", "measured": r.get("angular_label_measured", r.get("angular_label", "")),
                    "label": r.get("angular_label", ""), "peak": peak, "stab": stab, "broad": r.get("broad_preserved", ""),
                    "control": r.get("control", ""), "eprof": ep, "dev": r.get("stage_F_eligible_dev", ""),
                    "scenes": "; ".join(f"{k.replace('test83_', '')}: {v}" for k, v in sc.items()),
                    "scene": r.get("scene_verdict", "")})
    return out


def rows_from_test(T, Dv, note):
    out = []
    names = {"O1": "Optimization", "D": "Coverage", "E2": "Objective"}
    el = (Dv or {}).get("stage_F", {}).get("eligibility", {})
    for X in ("O1", "D", "E2"):
        L = T["labels"][X]
        c = T["contrasts_vs_O0"][X]["e_prof"]["all"]
        out.append({"row": f"{names[X]} ({X})", "measured": L["measured_label"], "label": L["label"], "peak": L["PEAK_GAIN"],
                    "stab": L["STABILITY_GAIN"], "broad": L["BROAD_PRESERVED"], "control": L["control_passed"],
                    "eprof": {k: c[k] for k in ("median", "lo", "hi", "resolved")}, "dev": el.get(X, {}).get("eligible"),
                    "scenes": "", "scene": note})
    e = T["selection_E1"]["primary_O0"]
    meas = "SELECTION_EFFECTIVE" if e["SELECTION_EFFECTIVE"] else "SELECTION_NOT_EFFECTIVE"
    lab = "UNQUALIFIED" if e.get("qualified") is False else meas + "".join("+" + f for f in e.get("flags", []))
    out.append({"row": "Selection (E1, on O0)", "measured": meas, "label": lab, "peak": f"PREDICTIVE {e['PREDICTIVE']}",
                "stab": f"SELECTION_EFFECTIVE {e['SELECTION_EFFECTIVE']} (median rho {e['median_rho']:.2f})", "broad": "",
                "control": e.get("control_SC_E1"), "eprof": {"median": e["contrast_vs_median_start"]}, "dev": "", "scenes": "",
                "scene": note})
    fe = T["F"].get("eligible_at_dev")
    out.append({"row": "Combination (F)", "measured": "NOT_RUN" if not fe else "", "label": "NOT_RUN" if not fe else "",
                "peak": "", "stab": "", "broad": "", "control": "", "eprof": {}, "dev": "needs ≥ 2 eligible", "scenes": "",
                "scene": "NOT_RUN" if not fe else note})
    return out


def decision_tables(pk):
    """[(heading, source, rows)]: the original-rule decision and, when present, the Amendment 1 decision."""
    note = "NOT_AVAILABLE (scene analyses / decision.json pending)"
    tabs = []
    if pk.DC is not None:
        tabs.append(("Decision table — original protocol rules", "decision.json", rows_from_decision(pk.DC)))
    elif pk.T is not None:
        src = "test_eval/test.json labels (decision.json not yet written)"
        if not pk.qual:
            src += "; qualification UNQUALIFIED by the O0 reproduction gate"
        tabs.append(("Decision table — original protocol rules", src, rows_from_test(pk.T, pk.Dv, note)))
    if pk.DCa is not None:
        tabs.append(("Decision table — Amendment 1 (owner-approved O0 reproduction rule)", "amendment1/decision_amendment1.json",
                     rows_from_decision(pk.DCa)))
    elif pk.TA is not None:
        tabs.append(("Decision table — Amendment 1 (owner-approved O0 reproduction rule)",
                     f"amendment1/test_amended.json labels (verdict {pk.TA.get('amendment1', {}).get('verdict')}; "
                     "decision_amendment1.json not yet written)", rows_from_test(pk.TA, pk.Dv, note)))
    return tabs


def index(pk):
    esc = html.escape

    def cell(v):
        if isinstance(v, dict):
            if "median" not in v:
                return ""
            s = f"{PCT * v['median']:+.1f} %"
            if "lo" in v:
                s += f" [{PCT * v['lo']:+.1f}, {PCT * v['hi']:+.1f}]"
            if "resolved" in v:
                s += " resolved" if v["resolved"] else " unresolved"
            return esc(s)
        return esc("" if v is None else str(v))
    hdr = ("<tr><th>row</th><th>measured label (predeclared rule)</th><th>evaluation label (gated)</th><th>PEAK_GAIN / PREDICTIVE</th>"
           "<th>STABILITY_GAIN / SELECTION_EFFECTIVE</th><th>BROAD_PRESERVED</th><th>synthetic control</th>"
           "<th>test e_prof change vs O0 (E1: vs median start) [95 % CI]</th><th>dev stage-F eligible</th>"
           "<th>scene labels (TOTAL; :e1 = first bounce)</th><th>scene verdict</th></tr>")
    tables = ""
    for head, src, rows in decision_tables(pk):
        tr = "".join(f"<tr><td>{esc(r['row'])}</td><td>{cell(r['measured'])}</td><td><b>{cell(r['label'])}</b></td><td>{cell(r['peak'])}</td>"
                     f"<td>{cell(r['stab'])}</td><td>{cell(r['broad'])}</td><td>{cell(r['control'])}</td><td>{cell(r['eprof'])}</td>"
                     f"<td>{cell(r['dev'])}</td><td class='small'>{cell(r['scenes'])}</td><td>{cell(r['scene'])}</td></tr>" for r in rows)
        tables += f"<h2>{esc(head)}</h2><p><small>source: {esc(src)}</small></p><table>{hdr}{tr}</table>"
    if pk.DC is not None and pk.DC.get("answers"):
        li = ""
        for k, v in pk.DC["answers"].items():
            if k == "G_remaining_limitations":
                continue
            a = v.get("answer")
            a = json.dumps(a, ensure_ascii=False) if isinstance(a, dict) else str(a)
            li += f"<li><b>{esc(k)}</b>: {esc(a)} <i>[{esc(', '.join(v.get('evidence_class', [])))}]</i></li>"
        tables += f"<h3>Overall answers (decision.json)</h3><ul>{li}</ul>"
    am = ""
    if pk.AM is not None:
        am = (f"<p><b>Amendment 1</b> (owner-approved after the miss had been seen; non-scientific execution correction of the O0 "
              f"reproduction rule): verdict <b>{esc(pk.AM['verdict'])}</b>. Same-batch-size reruns are bitwise identical; the "
              f"O0-vs-archive difference arises only from the batch-size change (4 to 8 starts). The original-rule decision is kept; "
              f"the amended decision is shown separately.</p>")
    figs = "".join(f"<section><h3>{esc(f['file'])} <span class='tag t{esc(f['kind'].split()[0].split(':')[0])}'>{esc(f['kind'])}</span></h3>"
                   f"<p>{esc(f['caption'])}<br><i>Source: {esc('; '.join(f['sources']))}</i></p>"
                   f"<img src='{esc(f['file'])}' loading='lazy' alt='{esc(f['caption'])}'></section>" for f in pk.figs)
    skipped = "".join(f"<li>{esc(s)}</li>" for s in pk.skipped) or "<li>none</li>"
    warn = f"<p class='warn'>{esc(pk.warn)}</p>" if pk.warn else ""
    ko = ("O0 대조군이 WL104K P 를 1 % 이내로 재현하지 못해 원래 규칙의 모든 각도 판정은 UNQUALIFIED 이며, 수치와 사전 규칙은 "
          "기술적으로만 표시됩니다." if not pk.qual else "판정은 사전 규칙에 따른 것입니다.")
    if pk.AM is not None:
        ko += f" Amendment 1 결과: {pk.AM['verdict']}."
    page = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>WL104K-B fitting investigation</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--mut:#555;--bd:#ddd;--warn:#b00020}}
@media (prefers-color-scheme: dark){{:root:not([data-theme=light]){{--bg:#16181d;--fg:#e8e8e8;--mut:#aaa;--bd:#333;--warn:#ff6b7d}}}}
:root[data-theme=dark]{{--bg:#16181d;--fg:#e8e8e8;--mut:#aaa;--bd:#333;--warn:#ff6b7d}}
body{{background:var(--bg);color:var(--fg);font-family:system-ui,sans-serif;max-width:1400px;margin:0 auto;padding:16px}}
img{{max-width:100%;background:#fff;border:1px solid var(--bd)}} .tag{{font-size:12px;padding:2px 6px;border-radius:4px;color:#fff;background:#6b6b6b}}
.tPREDECLARED{{background:#2b6cb0}} .tDIAGNOSTIC{{background:#8a5a00}} .warn{{color:var(--warn);font-weight:600}}
td,th{{border:1px solid var(--bd);padding:3px 6px;vertical-align:top}} table{{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}}
td.small{{font-size:11px}} p{{color:var(--mut)}}</style></head><body>
<h1>WL104K-B — exhaustive bounded fitting investigation for ER-v1-SDM</h1>
<p>Arms: original Adam fit (O0, control), L-BFGS (O1), refined angular coverage (D), core+shoulder objective (E2), selected start (E1),
combination (F, only if ≥ 2 mechanisms are dev-eligible); historical: certified ER baseline (A, WL104K) and ER-v1-SDM WL104K fit (P).
The ER-v1-SDM representation (178 parameters per material) is unchanged; no candidate is adopted.</p>
<p>(한국어 요약: {esc(ko)})</p>
{warn}{am}
{tables}
<p>Tags: <span class='tag tPREDECLARED'>PREDECLARED RULE</span> figure shows a frozen protocol rule;
<span class='tag'>DESCRIPTIVE</span> no rule attached; <span class='tag tDIAGNOSTIC'>DIAGNOSTIC</span> Stage A on archived states before the freeze;
<span class='tag'>SECONDARY</span> repeatedly inspected historical reference.</p>
<h2>Not plotted</h2><ul>{skipped}</ul>
<h2>Figures</h2>{figs}</body></html>"""
    (pk.out / "index.html").write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--stageA", type=Path, default=None)
    ap.add_argument("--scenes-root", type=Path, default=None, help="directory holding <scene>/analysis104kb.json (default <root>/scenes)")
    ap.add_argument("--decision", type=Path, default=None, help="decision json (default <root>/decision.json)")
    ap.add_argument("--decision-amended", type=Path, default=None,
                    help="Amendment 1 decision json (default <root>/amendment1/decision_amendment1.json)")
    a = ap.parse_args()
    root = a.root.resolve()
    out = a.out.resolve()
    if out == root or root in out.parents:
        raise SystemExit("refusing to write inside the run root")
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "figure.facecolor": "white"})
    pk = Pack(root, out, a.stageA, a.scenes_root, a.decision, a.decision_amended)
    steps = []
    if pk.SA is not None:
        steps += [stageA_peak, stageA_regions, stageA_path, stageA_ig]
    else:
        pk.skipped.append("Stage A: stageA.json missing")
    steps += [opt_curves, opt_summary]
    if pk.T is not None:
        steps += [qual_repro, test_pooled, test_contrasts, test_per_material, test_regions, test_peak_polar, test_instability, dev_eligibility,
                  ring_profiles, e1_scatter, e1_bars, synthetic, scenes_contrast, scenes_rms_per_start, scene_images, scene_peak_pixels,
                  historical]
    else:
        pk.skipped.append("independent test figures: test_eval/test.json missing")
    for f in steps:
        f(pk)
    index(pk)
    man = {"figures": [{**f, "sha256": sha(out / f["file"])} for f in pk.figs], "skipped": pk.skipped,
           "inputs_sha256": pk.inputs, "script_sha256": sha(Path(__file__)), "index_sha256": sha(out / "index.html"),
           "qualified": pk.qual, "qualification_note": pk.warn}
    (out / "figures_manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print("figures", len(pk.figs), "skipped", pk.skipped, flush=True)


if __name__ == "__main__":
    main()
