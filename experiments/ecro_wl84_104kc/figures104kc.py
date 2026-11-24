"""WL104K-C reviewer figures -> evaluation/104K-C/ (readable file names, titles with scene / what vs what / plot kind; absolute
error maps on one shared absolute scale per event over all scenes of this worklog, sqrt colour map with a real-unit colour bar).

  python figures104kc.py <root> <out_dir>

Figures whose inputs do not exist yet are skipped (listed in figures_manifest.json).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
NAMES = {"test83_iid_1": "iid_1", "test83_iid_0": "iid_0", "test83_hard_1": "hard_1", "test83_iid_191": "iid_191",
         "test83_iid_159": "iid_159", "test83_block_161": "block_161", "test83_block_186": "block_186",
         "test83_hard_76": "hard_76", "test83_hard_53": "hard_53"}
MATS = list(NAMES.values())
ARM = {"O0": "O0 (original WL104K fit, reproduced)", "CR": "CR (constrained core refinement)", "O0C": "O0C (same continuation, original objective)",
       "E2W": "E2W (same continuation, E2 weighted sum)", "E2_B": "E2_B (WL104K-B peak-weighted fit)", "P_hist": "P_hist (WL104K archive)",
       "CR_R4": "CR from batch-4 warm start", "R4": "O0 with WL104K batch of 4 starts"}
COL = {"O0": "#3c3c3c", "CR": "#2a78d6", "O0C": "#eda100", "E2W": "#1baf7a", "E2_B": "#eb6834", "P_hist": "#6250d6",
       "CR_R4": "#e87ba4", "R4": "#8a8a8a", "CR_S": "#2a78d6", "O0C_S": "#eda100", "E2W_S": "#1baf7a", "CR_SD": "#6250d6",
       "O0C_SD": "#e34948"}
METRIC = {"e_prof": "peak-profile error (0-8 deg grid)", "e_h": "peak-height error |max F/max R - 1|", "core_l1": "core L1 (< 3 deg)",
          "shoulder_l1": "shoulder L1 (3-12 deg)", "tail_l1": "tail L1 (>= 12 deg)", "el1_grazing": "grazing el1 (polar >= 75 deg)",
          "el1": "broad el1 (whole hemisphere)", "ealb": "albedo error ealb", "eshape": "shape error eshape",
          "instability": "across-start peak-height SD"}
SCENES_NEW = ("test83_hard_76_L1", "test83_block_161_L1", "test83_iid_191_L1", "test83_iid_0_L2")
SCENES_REG = ("test83_hard_1_L1", "test83_iid_159_L2", "test83_hard_53_L2", "test83_block_186_L2", "test83_iid_1_L2")
PCT = 100.0
made, skipped = [], []


def rd(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save(fig, out, name, caption):
    fig.savefig(out / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    made.append({"file": name, "caption": caption})


# ------------------------------------------------------------------ Stage A
def fig_stageA(root, out):
    sa, sh = rd(root / "stageA/stageA.json"), rd(root / "stageA/shape.json")
    if sa is None or sh is None:
        skipped.append("stage A figures (stageA.json missing)")
        return
    comps = ("decoded", "pred", "e1", "e2", "objective", "grad")
    lab = {"decoded": "decoded\nparameters", "pred": "lane\nresponse", "e1": "group el1", "e2": "group ealb", "objective": "objective",
           "grad": "gradient"}
    lays = {"L72swap": "same shape 72,\nother units' content changed", "L72perm": "same shape 72,\nstart order reversed",
            "L36": "batch of 4 starts\n(shape 36, WL104K)", "L9": "single start\n(shape 9)"}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, where, title in zip(axs, ("starts", "O0_final"), ("at the 8 starts", "at the converged O0 states")):
        for j, (L, ln) in enumerate(lays.items()):
            frac = [sh[where][L][c]["bitwise_equal_units"] / sh[where][L][c]["units"] for c in comps]
            ax.bar(np.arange(len(comps)) + (j - 1.5) * 0.2, frac, 0.18, label=ln, color=["#2a78d6", "#1baf7a", "#eb6834", "#6250d6"][j])
        ax.set_xticks(range(len(comps)), [lab[c] for c in comps], fontsize=8)
        ax.set_title(f"Per-unit bitwise identity vs the 8-start layout, {title}", fontsize=10)
        ax.set_ylim(0, 1.05)
    axs[0].set_ylabel("fraction of units bitwise identical")
    axs[1].legend(fontsize=7.5, loc="lower left")
    save(fig, out, "stageA_batch_layout_bitwise_identity_by_computation_bar.png",
         "Stage A: which computation depends on the batch layout. Bars = fraction of (material, start) units whose decoded "
         "parameters, lane responses, group terms, objective and gradient are bitwise identical to the 8-start (72-unit) layout. "
         "Only the gradient depends on the batch shape (36 or 9 units); content and order changes at fixed shape 72 change nothing.")
    # first divergence
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, L in enumerate(("L36", "L9")):
        fd = [v for v in sh["trajectory"][L]["per_unit_first_divergence"] if v is not None]
        ax.hist(fd, bins=np.arange(0, 41, 1), alpha=0.6, label=lays[L].replace("\n", " "), color=["#eb6834", "#6250d6"][j])
    ax.set_xlabel("first optimisation step at which the unit's objective differs from the 72-unit run")
    ax.set_ylabel("units")
    ax.set_title("Stage A: trajectory divergence under a different batch shape (original O0 schedule, first 500 steps)", fontsize=10)
    ax.legend(fontsize=8)
    save(fig, out, "stageA_batch_shape_first_divergence_step_histogram.png",
         "Stage A: step at which each unit's per-step training objective first differs from the 8-start run when the batch holds 4 "
         "starts (WL104K) or 1 start. Same code, data, starts and learning-rate schedule.")
    # batch-size difference vs across-start SD
    ns = sa["numerical_sensitivity_dev"]
    keys = [k for k in ("e_prof", "e_h", "core_l1", "shoulder_l1", "tail_l1", "el1", "ealb") if k in ns]
    fig, axs = plt.subplots(1, len(keys), figsize=(3.1 * len(keys), 3.4))
    for ax, k in zip(axs, keys):
        d = np.abs(np.asarray(ns[k]["batch_diff_per_unit"]))
        sd = np.asarray(ns[k]["across_start_sd_per_material"])
        x = np.broadcast_to(sd[None], d.shape).ravel()
        ax.loglog(x, d.ravel(), "o", ms=4, color="#2a78d6", alpha=0.75)
        lo, hi = min(x.min(), d[d > 0].min() if (d > 0).any() else x.min()), max(x.max(), d.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#8a8a8a", lw=1)
        ax.set_title(METRIC.get(k, k), fontsize=8)
        ax.set_xlabel("across-start SD (8 starts)", fontsize=7.5)
    axs[0].set_ylabel("|batch-4 - batch-8| per unit", fontsize=8)
    fig.suptitle("Stage A (WL104K-B DEV set): batch-size numerical difference per (material, start 1-4) vs the across-start spread; "
                 "dashed = equal", fontsize=9.5)
    save(fig, out, "stageA_dev_batch4_vs_batch8_difference_vs_across_start_sd_scatter.png",
         "Stage A: numerical batch-size sensitivity (|metric(R4) - metric(R8)|, mean over the 20 DEV probes) per (material, start "
         "1-4) against the across-start SD of the same metric (8 starts) per material. Points near the dashed line mean the batch "
         "size alone moves a unit about as much as changing the random start.")


# ------------------------------------------------------------------ training
def fig_training(root, out):
    fd = rd(root / "fit_decision/fit_decision.json")
    if fd is None:
        skipped.append("training figures (fit decision missing)")
        return
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for a in ("CR", "CR_R4", "CR_S", "CR_SD"):
        fj = rd(root / ("fits/" + a) / "fit.json")
        if fj is None:
            continue
        h = fj["checkpoint_history"]
        st = [r["step"] for r in h]
        axs[0].plot(st, [np.mean(r["feasible"]) for r in h], marker="o", ms=3, label=a, color=COL[a],
                    ls="--" if a in ("CR_S", "CR_SD") else "-")
        axs[1].plot(st, [np.median(r["F"]) for r in h], marker="o", ms=3, label=a, color=COL[a], ls="--" if a in ("CR_S", "CR_SD") else "-")
    axs[0].set_ylabel("fraction of units satisfying every constraint")
    axs[1].set_ylabel("median training core error relative to the warm start (F)")
    for ax in axs:
        ax.set_xlabel("continuation step")
        ax.legend(fontsize=8)
    axs[0].set_title("CR augmented Lagrangian: feasibility per checkpoint (training data)", fontsize=9.5)
    axs[1].set_title("CR: specular-core training error per checkpoint (1 = O0)", fontsize=9.5)
    save(fig, out, "training_CR_feasibility_and_core_error_by_checkpoint_line.png",
         "CR training (train_original; synthetic arms on their noisy targets): fraction of units whose per-probe el1 / albedo / "
         "shoulder / tail constraints all hold within 0.1 % (left) and median core-error ratio F to the warm start (right) at "
         "every 500-step checkpoint. The returned state is the best feasible checkpoint (or the warm start).")
    cr = fd["training"]["CR"]
    F = np.asarray(cr["F_returned"]).reshape(8, 9)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(F, cmap="viridis", vmin=min(0.5, F.min()), vmax=1.0, aspect="auto")
    ax.set_xticks(range(9), MATS, rotation=30, fontsize=8)
    ax.set_yticks(range(8), [f"start {r + 1}" for r in range(8)], fontsize=8)
    for r in range(8):
        for m in range(9):
            ax.text(m, r, f"{F[r, m]:.2f}", ha="center", va="center", fontsize=6.5, color="w" if F[r, m] < 0.8 else "k")
    fig.colorbar(im, ax=ax, label="returned core error / O0 core error (training)")
    ax.set_title("CR: training core-error ratio of the returned feasible state per material and start (heatmap)", fontsize=9.5)
    save(fig, out, "training_CR_returned_core_error_ratio_per_material_start_heatmap.png",
         "CR: per (material, start) ratio of the returned state's training core error (< 3 deg) to the O0 warm start, all "
         "constraints satisfied. 1.00 = no feasible improvement found (warm start returned).")


# ------------------------------------------------------------------ independent angular
def fig_test(root, out):
    t = rd(root / "test_eval/test.json")
    if t is None:
        skipped.append("independent test figures (test.json missing)")
        return
    arr = np.load(root / "test_eval/arrays.npz")
    keys = ["e_prof", "e_h", "core_l1", "shoulder_l1", "tail_l1", "el1_grazing", "el1", "ealb", "eshape", "instability"]
    arms = ["CR", "O0C", "E2W", "E2_B"]
    null = t["numerical_null"]
    fig, axs = plt.subplots(1, len(arms), figsize=(4.2 * len(arms), 5.4), sharey=True)
    thr = {"e_prof": -15, "instability": -25, "el1": 5, "ealb": 10, "eshape": 10, "shoulder_l1": 10, "tail_l1": 10, "el1_grazing": 10}
    for ax, a in zip(axs, arms):
        for i, k in enumerate(keys):
            c = t["contrasts"][a][k]["all"]
            y = len(keys) - 1 - i
            ax.plot([PCT * c["lo"], PCT * c["hi"]], [y, y], color=COL[a], lw=2)
            ax.plot(PCT * c["median"], y, "o", color=COL[a], ms=6)
            nl = max([abs(null[s][k]) for s in null if k in null[s]] or [0])
            ax.fill_betweenx([y - 0.3, y + 0.3], -PCT * nl, PCT * nl, color="#d0d0d0", lw=0)
            if k in thr:
                ax.plot([thr[k]] * 2, [y - 0.35, y + 0.35], color="#b00020", lw=1.2)
        ax.axvline(0, color="#8a8a8a", lw=0.8)
        ax.set_title(ARM[a], fontsize=9)
    fig.supxlabel("pooled change vs O0 (%): dot = median over starts, bar = 95 % material-cluster bootstrap", fontsize=9)
    axs[0].set_yticks(range(len(keys)), [METRIC[k] for k in keys[::-1]], fontsize=8)
    fig.suptitle("Independent angular test (WL104K-C test_c): change vs O0 per metric; grey band = batch-size numerical null; "
                 "red tick = protocol rule threshold", fontsize=10)
    save(fig, out, "test_c_pooled_change_vs_O0_per_metric_CR_O0C_E2W_E2B_dotplot.png",
         "Independent test_c: paired-start pooled relative change vs O0 (dot = median over 8 starts, bar = 95 % material-cluster "
         "bootstrap) for peak / core / shoulder / tail / grazing / broad el1 / albedo / shape and across-start instability. Grey "
         "band = batch-size numerical null; red tick = PEAK_GAIN (-15 %), STABILITY_GAIN (-25 %) or BROAD_PRESERVED limits.")
    # per material per start
    for k, unit in (("e_prof", "peak-profile error"), ("el1", "broad el1"), ("ealb", "albedo error ealb")):
        fig, ax = plt.subplots(figsize=(11, 3.8))
        for j, a in enumerate(("O0", "CR", "E2_B", "P_hist")):
            v = arr[f"{a}__{k}"].mean(-1)                                       # [R, n]
            for m in range(9):
                ax.plot(m + (j - 1.5) * 0.17 + np.zeros(v.shape[0]), v[:, m], "o", ms=3.5, color=COL[a],
                        label=ARM[a] if m == 0 else None, alpha=0.85)
        ax.set_xticks(range(9), MATS)
        ax.set_yscale("log")
        ax.set_ylabel(f"{unit}, mean over 20 probes")
        ax.set_title(f"Independent test_c: {unit} per material and start (each dot = one start)", fontsize=10)
        ax.legend(fontsize=7.5, ncol=4)
        save(fig, out, f"test_c_{k}_per_material_per_start_O0_CR_E2B_Phist_dots.png",
             f"Independent test_c: {unit} (mean over the 20 probes) for every material and start; O0, CR, E2_B 8 starts, P_hist 4.")
    # ring residual profiles
    rad = arr["ring_radius_deg"]
    fig, axs = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for m, ax in enumerate(axs.ravel()):
        for a in ("O0", "CR", "E2_B", "P_hist"):
            pr = arr[f"{a}__ring_residual"][:, m]                               # [R, P, ring]
            ax.plot(rad, PCT * np.median(pr.mean(1), 0), "o-", ms=2.5, lw=1.4, color=COL[a], label=ARM[a])
        ax.axhline(0, color="#8a8a8a", lw=0.8)
        ax.set_title(MATS[m], fontsize=9)
        ax.set_xscale("symlog", linthresh=0.25)
    for ax in axs[-1]:
        ax.set_xlabel("offset from the mirror direction (deg)")
    for ax in axs[:, 0]:
        ax.set_ylabel("ring residual F/R - 1 (%)")
    axs[0, 0].legend(fontsize=7)
    fig.suptitle("Independent test_c: peak-grid ring residual (mean over 20 probes, median over starts) per material", fontsize=10)
    save(fig, out, "test_c_peak_ring_residual_profile_per_material_O0_CR_E2B_Phist_line.png",
         "Independent test_c peak grid: per ring (0-8 deg from the mirror direction) mean prediction / mean reference - 1, averaged "
         "over the 20 probes and the median over starts. Positive = overshoot.")
    # synthetic
    s = t["synthetic"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ks = ["e_prof", "e_h", "core_l1", "shoulder_l1", "tail_l1", "el1", "ealb", "eshape"]
    for j, a in enumerate(("CR_S", "O0C_S", "E2W_S", "CR_SD", "O0C_SD")):
        r = s["controls"][a]["ratio"]
        ax.bar(np.arange(len(ks)) + (j - 2) * 0.16, [r[k] for k in ks], 0.15, color=COL[a],
               label=f"{a} / {s['controls'][a]['vs']}", hatch="//" if a.endswith("SD") else None)
    ax.axhline(1, color="#8a8a8a", lw=0.8)
    ax.axhline(0.8, color="#b00020", lw=1, ls="--", label="SYN_RECOVERY limit 0.8 (e_prof)")
    ax.set_xticks(range(len(ks)), [METRIC[k] for k in ks], rotation=20, fontsize=8)
    ax.set_ylabel("pooled error vs its warm-start arm (ratio)")
    ax.set_title("Synthetic exactly representable truth: error against the noise-free truth on test_c, relative to the warm start",
                 fontsize=10)
    ax.legend(fontsize=7.5, ncol=3)
    save(fig, out, "synthetic_truth_recovery_ratio_vs_warm_start_bar.png",
         "Synthetic controls (truth exactly representable): pooled test_c error against the noise-free truth of each continuation "
         "arm divided by its warm start (S_O0r for *_S on train_original lanes, S_Dr for *_SD on the denser train_coverage lanes).")


# ------------------------------------------------------------------ scenes
def scene_list(root):
    return [s for s in SCENES_NEW + SCENES_REG if (root / "scenes" / s / "analysis104kc.json").exists()]


def fig_scenes(root, out):
    sc = scene_list(root)
    if not sc:
        skipped.append("scene figures (no analysis)")
        return
    A = {s: rd(root / "scenes" / s / "analysis104kc.json") for s in sc}
    pairs = [("CR", "CR vs O0"), ("O0C", "O0C vs O0"), ("CR_vs_O0C", "CR vs O0C"), ("E2_B", "E2_B vs O0")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 0.55 * len(sc) * len(pairs) / 2 + 2.5), sharey=True)
    for ax, ev in zip(axs, ("e1", "e2", "total")):
        ticks = []
        y = 0
        for s in sc:
            for key, lab in pairs:
                c = A[s]["contrasts"].get(key, {}).get(ev)
                if c is None:
                    continue
                if c.get("point") is not None:
                    ax.plot([PCT * c["lo"], PCT * c["hi"]], [y, y], color=COL[key.split("_vs_")[0]], lw=2)
                    ax.plot(PCT * c["point"], y, "o", color=COL[key.split("_vs_")[0]], ms=5)
                else:
                    ax.text(0, y, "unresolved (noise floor)", fontsize=6.5, va="center")
                ticks.append((y, f"{s.replace('test83_', '')}{' [NEW]' if A[s]['new_independent'] else ''}: {lab}"))
                y -= 1
            y -= 0.6
        ax.axvline(0, color="#8a8a8a", lw=0.8)
        ax.axvline(2, color="#b00020", lw=0.8, ls="--")
        ax.set_title({"e1": "first bounce (e1)", "e2": "second bounce, fixed proposal (e2)", "total": "TOTAL (e1 + e2 + fixed R0 HIGH)"}[ev],
                     fontsize=9.5)
    fig.supxlabel("relative change of the start-mean objects stable RMS (%), replicate jackknife t(11) 95 % CI", fontsize=9)
    axs[0].set_yticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=7)
    fig.suptitle("Scenes (fixed-proposal replay; NOT end-to-end ECRO): change of the start-mean objects RMS error; dashed = REGRESSED "
                 "limit +2 %; [NEW] = new independent scene", fontsize=10)
    save(fig, out, "scenes_objects_rms_change_e1_e2_total_all_scenes_dotplot.png",
         "Every scene: relative change of the start-mean objects stable RMS error (noise-debiased) with replicate jackknife t(11) "
         "95 % CI for CR vs O0, O0C vs O0, CR vs O0C and E2_B vs O0, per event. IMPROVED = upper bound < 0; REGRESSED = lower "
         "bound > +2 %.")
    # absolute error maps, one shared absolute scale per event over all scenes and arms
    for ev in ("e1", "total"):
        maps = {s: np.load(root / "scenes" / s / "maps.npz") for s in sc}
        arms = [a for a in ("O0", "CR", "E2_B") if all(f"{a}_{ev}_abs_err" in maps[s] for s in sc)]
        vmax = max(float(maps[s][f"{a}_{ev}_abs_err"][maps[s]["objects_mask"]].max()) for s in sc for a in arms)
        fig, axs = plt.subplots(len(sc), len(arms) + 1, figsize=(3.2 * (len(arms) + 1), 3.0 * len(sc)))
        axs = np.atleast_2d(axs)
        for i, s in enumerate(sc):
            ref = maps[s][f"reference_{'e1' if ev == 'e1' else 'total'}_lum"]
            axs[i, 0].imshow(np.sqrt(np.clip(ref, 0, None) / max(ref.max(), 1e-30)), cmap="gray")
            axs[i, 0].set_title(f"{s.replace('test83_', '')}: reference\n(sqrt, own scale)", fontsize=8)
            for j, a in enumerate(arms):
                im = axs[i, j + 1].imshow(np.sqrt(maps[s][f"{a}_{ev}_abs_err"]), cmap="magma", vmin=0, vmax=np.sqrt(vmax))
                axs[i, j + 1].set_title(f"{s.replace('test83_', '')}:\n|{a} - reference| {ev}", fontsize=8)
        for ax in axs.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=axs, shrink=0.6)
        tv = np.linspace(0, np.sqrt(vmax), 5)
        cb.set_ticks(tv, labels=[f"{v * v:.3g}" for v in tv])
        cb.set_label(f"absolute luminance error, {ev} (shared scale over all scenes and arms; sqrt mapping)")
        save(fig, out, f"scenes_{ev}_absolute_error_maps_O0_CR_E2B_shared_scale.png",
             f"Absolute HDR luminance error maps of the {ev} event (start-mean images) for O0, CR and E2_B in every scene, all on one "
             f"absolute scale (max {vmax:.3g}) with sqrt mapping; the reference image is shown on its own scale for orientation.")
    # relative signed error maps (X - reference) / reference on the objects mask, one shared symmetric scale over all scenes and arms
    for ev in ("e1", "total"):
        maps = {s: np.load(root / "scenes" / s / "maps.npz") for s in sc}
        arms = [a for a in ("O0", "CR", "E2_B") if all(f"{a}_{ev}_lum" in maps[s] for s in sc)]

        def rel(s, a):
            ref = maps[s][f"reference_{ev}_lum"]
            floor = 0.01 * float(np.percentile(ref[maps[s]["objects_mask"]], 99))
            r = (maps[s][f"{a}_{ev}_lum"] - ref) / np.maximum(ref, floor)
            return np.where(maps[s]["objects_mask"], r, np.nan)

        lim = float(np.nanpercentile(np.concatenate([np.abs(rel(s, a)).ravel() for s in sc for a in arms]), 99))
        fig, axs = plt.subplots(len(sc), len(arms), figsize=(3.2 * len(arms), 3.0 * len(sc)))
        axs = np.atleast_2d(axs)
        for i, s in enumerate(sc):
            for j, a in enumerate(arms):
                im = axs[i, j].imshow(PCT * rel(s, a), cmap="RdBu_r", vmin=-PCT * lim, vmax=PCT * lim)
                axs[i, j].set_facecolor("#d9d9d9")
                axs[i, j].set_title(f"{s.replace('test83_', '')}:\n({a} - reference) / reference, {ev}", fontsize=8)
        for ax in axs.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=axs, shrink=0.6)
        cb.set_label(f"relative signed luminance error, {ev} (%; red = too bright, blue = too dark; shared scale = 99th percentile)")
        save(fig, out, f"scenes_{ev}_relative_signed_error_maps_O0_CR_E2B_shared_scale.png",
             f"Relative signed HDR luminance error (X - reference) / reference of the {ev} event (start-mean images) on the objects mask "
             f"(grey = background, not evaluated) for O0, CR and E2_B; one symmetric shared scale +/-{PCT * lim:.3g} % (99th percentile "
             f"over all scenes and arms; larger values saturate); reference floored at 1 % of its objects 99th percentile.")


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_stageA, fig_training, fig_test, fig_scenes):
        try:
            f(root, out)
        except Exception as e:                                          # recorded, never silent
            skipped.append(f"{f.__name__}: {type(e).__name__}: {e}")
    (out / "figures_manifest.json").write_text(json.dumps({"figures": made, "skipped": skipped}, indent=1), encoding="utf-8")
    print("figures", len(made), "skipped", skipped)


if __name__ == "__main__":
    main()
