"""WL104H reviewer package: figures (shared absolute scales per unit system, signed square-root display), compact summary,
label tables, response/scene agreement and manifest. Post-processing only: reads the frozen outputs, adds no rule.

Usage: python package104h.py <primary_dir> <review_dir>
"""
import argparse
import shutil
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import FuncNorm  # noqa: E402

import io104h as IO  # noqa: E402
import comp104h as K  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722])
SCENES = ("test83_iid_1_L2", "test83_hard_1_L1", "test83_iid_159_L2")
SCENE_MAT = {"test83_iid_1_L2": "test83_iid_1", "test83_hard_1_L1": "test83_hard_1", "test83_iid_159_L2": "test83_iid_159"}
ABBR = {"CONDITIONAL_ERROR_LOCALIZATION": "L", "DETECTABLE_COMPONENT_EFFECT": "D", "NO_DETECTABLE_COMPONENT_EFFECT": "N",
        "CONFOUNDED_OR_NON_IDENTIFIABLE": "C", "EVIDENCE_UNQUALIFIED": "U"}
BANDS = ("near_normal", "mid", "band_55_65", "theta_70", "grazing", "all")
SROIS = ("objects", "sphere", "cube", "highlight_core", "highlight_ring", "grazing")
GROUPS = ("LAMBERT_OFF", "OUTER_OFF", "INNER_OFF", "ATTEN_OFF", "ETA_OFF")
LOBES = ("O0_OFF", "O1_OFF", "O2_OFF", "I0_OFF", "I1_OFF", "I2_OFF", "I3_OFF")
TERMS = ("O0", "O1", "O2", "SL", "SI0", "SI1", "SI2", "SI3")


def ssqrt_norm(M):
    f = lambda x: np.sign(x) * np.sqrt(np.abs(x) / M)
    g = lambda y: np.sign(y) * (y ** 2) * M
    return FuncNorm((f, g), vmin=-M, vmax=M)


def sroot4_norm(M):
    f = lambda x: np.sign(x) * (np.abs(x) / M) ** 0.25
    g = lambda y: np.sign(y) * (y ** 4) * M
    return FuncNorm((f, g), vmin=-M, vmax=M)


def sqrt_norm(M):
    return FuncNorm((lambda x: np.sqrt(np.clip(x, 0, None) / M), lambda y: (y ** 2) * M), vmin=0, vmax=M)


def mirror_frame(wi):
    m = np.array([-wi[0], -wi[1], wi[2]])
    e1 = np.cross([0, 0, 1.0], m)
    e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-9 else np.array([1.0, 0, 0])
    return m, e1, np.cross(m, e1)


def bin_map(wi, wo, v, R=45.0, n=41):
    m, e1, e2 = mirror_frame(wi)
    r = np.degrees(np.arccos(np.clip(wo @ m, -1, 1)))
    az = np.arctan2(wo @ e2, wo @ e1)
    x, y = r * np.cos(az), r * np.sin(az)
    edges = np.linspace(-R, R, n + 1)
    s, _, _ = np.histogram2d(y, x, [edges, edges], weights=v)
    c, _, _ = np.histogram2d(y, x, [edges, edges])
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def angular_maps(lanes, rows, out):
    """Per development material: reference / R0 / residual / terms, and residual vs group changes at 20, 60, 86 deg (phi 22.5)."""
    probes = [1 * 2, 5 * 2, 9 * 2]   # theta 20, 60, 86 at phi 22.5
    sets, pos_max, sgn_max = {}, 0.0, 0.0
    for r in rows:
        if r["cohort"] != "development":
            continue
        tag = r["tag"]
        g = lambda k: lanes[f"{tag}|{k}"]
        w = g("w")
        A = (w[..., None] * 0.5 * (g("a") + g("b"))).sum((1, 2))
        for p in probes:
            norm = lambda v: (v @ LUMA) / A[p]
            pos = {"reference": norm(0.5 * (g("a") + g("b"))[p]), "R0": norm(g("F")[p]), **{t: norm(g(t)[p]) for t in TERMS}}
            sgn = {"residual": norm((0.5 * (g("a") + g("b")) - g("F"))[p]), **{d: norm(g(f"D_{d}")[p]) for d in GROUPS}}
            mp = {k: bin_map(g("wi")[p, 0], g("wo")[p], v) for k, v in pos.items()}
            ms = {k: bin_map(g("wi")[p, 0], g("wo")[p], v) for k, v in sgn.items()}
            sets[(tag, p)] = (mp, ms)
            pos_max = max(pos_max, max(np.nanmax(v) for v in mp.values()))
            sgn_max = max(sgn_max, max(np.nanmax(np.abs(v)) for v in ms.values()))
    files = []
    for tag in sorted({k[0] for k in sets}):
        fig, ax = plt.subplots(len(probes), 11, figsize=(22, 6.4), layout="constrained")
        for i, p in enumerate(probes):
            mp, ms = sets[(tag, p)]
            cols = [("reference", mp), ("R0", mp), ("residual", ms), *[(t, mp) for t in TERMS]]
            for j, (k, src) in enumerate(cols):
                signed = k == "residual"
                im = ax[i, j].imshow(src[k], origin="lower", extent=[-45, 45, -45, 45], cmap="RdBu_r" if signed else "magma",
                                     norm=ssqrt_norm(sgn_max) if signed else sqrt_norm(pos_max))
                ax[i, j].set_title(f"{k}  θi={IO.POLAR[p]:g}°", fontsize=8)
                ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
                if j in (2, 10):
                    fig.colorbar(im, ax=ax[i, j], shrink=0.7)
        fig.suptitle(f"{tag}: luma response / probe albedo normaliser, mirror-centred (±45°); terms include S for L and I_k")
        f = out / f"angular_terms_{tag}.png"
        fig.savefig(f, dpi=110); plt.close(fig); files.append(f)
        fig, ax = plt.subplots(len(probes), 6, figsize=(13, 6.4), layout="constrained")
        for i, p in enumerate(probes):
            _, ms = sets[(tag, p)]
            for j, k in enumerate(("residual", *GROUPS)):
                im = ax[i, j].imshow(ms[k], origin="lower", extent=[-45, 45, -45, 45], cmap="RdBu_r", norm=ssqrt_norm(sgn_max))
                ax[i, j].set_title(f"{'Δ ' if k != 'residual' else ''}{k}  θi={IO.POLAR[p]:g}°", fontsize=8)
                ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            fig.colorbar(im, ax=ax[i, -1], shrink=0.7)
        fig.suptitle(f"{tag}: residual (reference − R0) vs group intervention change ΔF_X (same signed scale)")
        f = out / f"angular_group_changes_{tag}.png"
        fig.savefig(f, dpi=110); plt.close(fig); files.append(f)
    return files, {"response_positive_max": float(pos_max), "response_signed_max": float(sgn_max),
                   "response_mapping": "signed sqrt(|x|/M); units = luma response / probe albedo normaliser A_p"}


def contributions_fig(resp, rows, out):
    pp = resp["per_probe"]
    fig, ax = plt.subplots(3, 3, figsize=(12, 9), layout="constrained", sharex=True)
    for i, r in enumerate(rows):
        st = None
        a = ax.flat[i]
        S = resp["strata"]
        # per-material term shares are not stored per material; use the lanes-free per-probe peak shares at the reference peak
        pk = resp["peak"][r["tag"]]["term_share_at_ref_peak"]
        th = IO.POLAR
        for t in TERMS:
            v = np.asarray(pk[t])
            a.plot(th[::2], 0.5 * (v[::2] + v[1::2]), marker="o", ms=3, label=t)
        a.set_title(f"{r['tag']} ({r['cohort']})", fontsize=9)
        a.set_ylim(-0.05, 1.05)
    ax.flat[0].legend(fontsize=7, ncol=2)
    fig.supxlabel("incident polar angle (deg)")
    fig.supylabel("term share of R0 luma at the reference peak direction")
    f = out / "term_shares_at_peak.png"
    fig.savefig(f, dpi=110); plt.close(fig)
    return f


def stratum_contrib_fig(resp, out):
    S = resp["strata"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained")
    for a, cohort in zip(ax, ("development", "held_out")):
        x = np.arange(len(BANDS))
        bottom = np.zeros(len(BANDS))
        for t in TERMS:
            v = np.array([S[f"{cohort}|{b}|all"]["term_luma_share"][t] for b in BANDS])
            a.bar(x, v, bottom=bottom, label=t)
            bottom += v
        a.set_xticks(x, BANDS, rotation=20)
        a.set_title(f"{cohort}: term share of R0 directional luma albedo")
    ax[0].legend(fontsize=7, ncol=4)
    f = out / "term_albedo_shares.png"
    fig.savefig(f, dpi=110); plt.close(fig)
    return f


def rho_heatmaps(resp, out):
    S = resp["strata"]
    names = K.NAMES
    cols = [(c, b) for c in ("development", "held_out") for b in BANDS]
    files = []
    for region in ("all", "core", "shoulder", "tail"):
        fig, ax = plt.subplots(1, 3, figsize=(20, 6.5), layout="constrained")
        for a, (title, getter) in zip(ax, (("rho (alignment with residual)", lambda st: st["rho"]["point"]),
                                           ("error change (fraction of residual energy removed)", lambda st: st["error_change"]["point"]),
                                           ("albedo-matched rho (DIAGNOSTIC_ONLY)", lambda st: st["matched"]["rho"]["point"]))):
            M = np.array([[getter(S[f"{c}|{b}|{region}"]["interventions"][n]) for c, b in cols] for n in names], float)
            lim = 1.0 if "rho" in title else max(1.0, np.nanmax(np.abs(M)))
            if "error" in title:
                im = a.imshow(M, cmap="RdBu_r", aspect="auto", norm=ssqrt_norm(lim))
            else:
                im = a.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            for i, n in enumerate(names):
                for j, (c, b) in enumerate(cols):
                    st = S[f"{c}|{b}|{region}"]["interventions"][n]
                    lab = st["matched"]["label"] if "matched" in title else st["label"]
                    a.text(j, i, f"{M[i, j]:.2f}\n{ABBR[lab]}", ha="center", va="center", fontsize=6)
            a.set_xticks(range(len(cols)), [f"{c[:3]}|{b}" for c, b in cols], rotation=60, fontsize=7)
            a.set_yticks(range(len(names)), names, fontsize=7)
            a.set_title(title, fontsize=9)
            fig.colorbar(im, ax=a, shrink=0.6)
        fig.suptitle(f"Response-space interventions, region {region} (L localized, D detectable, N none, C confounded, U unqualified)")
        f = out / f"response_rho_{region}.png"
        fig.savefig(f, dpi=110); plt.close(fig); files.append(f)
    return files


def identifiability_fig(resp, out):
    S = resp["strata"]
    fig, ax = plt.subplots(2, 2, figsize=(14, 12), layout="constrained")
    for i, cohort in enumerate(("development", "held_out")):
        blk = S[f"{cohort}|all|all"]
        names = blk["names"]
        C = np.array(blk["cos"])
        a = ax[i, 0]
        a.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        for x in range(len(names)):
            for y in range(len(names)):
                a.text(y, x, f"{C[x, y]:.2f}", ha="center", va="center", fontsize=5)
        a.set_xticks(range(len(names)), names, rotation=90, fontsize=6); a.set_yticks(range(len(names)), names, fontsize=6)
        a.set_title(f"{cohort}: cos between intervention changes ΔF_X (all probes)")
        T = np.array(blk["term_cos"])
        a = ax[i, 1]
        a.imshow(T, cmap="RdBu_r", vmin=-1, vmax=1)
        for x in range(len(TERMS)):
            for y in range(len(TERMS)):
                a.text(y, x, f"{T[x, y]:.2f}", ha="center", va="center", fontsize=7)
        a.set_xticks(range(len(TERMS)), TERMS); a.set_yticks(range(len(TERMS)), TERMS)
        a.set_title(f"{cohort}: cos between base response terms")
    f = out / "response_identifiability.png"
    fig.savefig(f, dpi=110); plt.close(fig)
    return f


def params_fig(par, rows, out):
    groups = list(par["groups"])
    fig, ax = plt.subplots(2, 1, figsize=(14, 8), layout="constrained")
    x = np.arange(len(rows))
    wdt = 0.8 / (len(groups) + 1)
    for k, key in enumerate(("R2", "unique_R2")):
        a = ax[k]
        sets = (["all"] if key == "R2" else []) + groups
        for j, g in enumerate(sets):
            v = [par["results"][r["tag"]]["strata"]["all|all"][f"{key}_{g}"] for r in rows]
            pt = np.array([q["point"] for q in v]); lo = np.array([q["lo"] for q in v]); hi = np.array([q["hi"] for q in v])
            a.bar(x + j * wdt, pt, wdt, yerr=[pt - lo, hi - pt], label=g, capsize=2)
        a.set_xticks(x + 0.4, [r["tag"].replace("test83_", "") for r in rows])
        a.axhline(0, color="k", lw=0.5)
        a.set_title("first-order explainable residual fraction (cross-half, all probes)" if key == "R2" else
                    "unique fraction (R2_all − R2_all-without-group)")
        a.legend(fontsize=7, ncol=7)
    f = out / "parameter_r2.png"
    fig.savefig(f, dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    for a, r in zip(ax, [r for r in rows if r["cohort"] == "development"]):
        pc = par["results"][r["tag"]]["principal_cosines"]
        M = np.eye(len(groups))
        for i, g in enumerate(groups):
            for j, h in enumerate(groups):
                key = f"{g}|{h}" if f"{g}|{h}" in pc else f"{h}|{g}"
                if i != j and key in pc:
                    M[i, j] = max(pc[key])
        a.imshow(M, cmap="viridis", vmin=0, vmax=1)
        for i in range(len(groups)):
            for j in range(len(groups)):
                a.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7, color="w" if M[i, j] < .6 else "k")
        a.set_xticks(range(len(groups)), groups, rotation=45, fontsize=7); a.set_yticks(range(len(groups)), groups, fontsize=7)
        a.set_title(f"{r['tag']}: largest principal cosine", fontsize=9)
    f2 = out / "parameter_coupling.png"
    fig.savefig(f2, dpi=110); plt.close(fig)
    return [f, f2]


def ca_fig(resp, par, scenes, rows, out):
    S = resp["strata"]
    parts = ("O0", "O1", "O2", "atten", "L_content", "I0_content", "I1_content", "I2_content", "I3_content")
    cols, vals = [], []
    for c in ("development", "held_out"):
        for b in BANDS:
            cols.append(f"resp {c[:3]}|{b}")
            vals.append([S[f"{c}|{b}|all"]["ca_shares"][p]["point"] for p in parts])
    for sc, an in scenes.items():
        for ev in ("e1", "e2"):
            sh = an["rois"]["objects"][ev]["ca_shares"]
            cols.append(f"{sc.replace('test83_', '')} {ev}")
            vals.append([sh[p if p in ("O0", "O1", "O2", "atten") else p.replace("_content", "") + "_content"]["point"] for p in parts])
    V = np.array(vals)
    fig, ax = plt.subplots(figsize=(15, 5), layout="constrained")
    pos = np.zeros(len(cols)); neg = np.zeros(len(cols))
    for j, p in enumerate(parts):
        v = V[:, j]
        base = np.where(v >= 0, pos, neg)
        ax.bar(range(len(cols)), v, bottom=base, label=p)
        pos += np.clip(v, 0, None); neg += np.clip(v, None, 0)
    ax.set_xticks(range(len(cols)), cols, rotation=60, fontsize=7)
    ax.axhline(1, color="k", lw=0.5, ls="--"); ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=7, ncol=9)
    ax.set_title("Exact decomposition of the WL104D C − A change: share <part, Δ_CA>/|Δ_CA|² (sums to 1); scene E2 uses the common R0 proposal")
    f = out / "ca_decomposition.png"
    fig.savefig(f, dpi=110); plt.close(fig)
    return f


def scene_rho_fig(scenes, out):
    names = K.NAMES
    cols = [(s, ev, roi) for s in SCENES for ev in ("e1", "e2") for roi in SROIS]
    M = np.full((len(names), len(cols)), np.nan)
    L = [["" for _ in cols] for _ in names]
    for j, (s, ev, roi) in enumerate(cols):
        blk = scenes[s]["rois"].get(roi, {}).get(ev)
        if not blk:
            continue
        for i, n in enumerate(names):
            st = blk["interventions"][n]
            M[i, j] = st["rho"]["point"]
            L[i][j] = ABBR[st["label"]]
    fig, ax = plt.subplots(figsize=(22, 7), layout="constrained")
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(names)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}\n{L[i][j]}", ha="center", va="center", fontsize=5.5)
    ax.set_xticks(range(len(cols)), [f"{s.replace('test83_', '')}|{ev}|{roi}" for s, ev, roi in cols], rotation=75, fontsize=6)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.6)
    ax.set_title("Scene-conditioned rho (ΔE_X vs reference − R0), common R0 proposal; L localized, D detectable, N none, C confounded, U unqualified")
    f = out / "scene_rho.png"
    fig.savefig(f, dpi=110); plt.close(fig)
    return f


def scene_maps(primary, out):
    maps, boxes, M = {}, {}, 0.0
    for s in SCENES:
        with np.load(primary / "scenes" / s / "maps.npz") as z:
            maps[s] = {k: z[k] @ LUMA for k in z.files}
        with np.load(IO.G_RAW / "scenes" / s / "region_masks.npz") as z:
            obj = z["objects"]
        ys, xs = np.nonzero(obj)
        boxes[s] = (max(ys.min() - 6, 0), min(ys.max() + 7, obj.shape[0]), max(xs.min() - 6, 0), min(xs.max() + 7, obj.shape[1]))
        for k, v in maps[s].items():
            if k.startswith(("resid_", "ca_", "d_")):
                M = max(M, float(np.abs(v).max()))
    files = []
    for s in SCENES:
        y0, y1, x0, x1 = boxes[s]
        for kind, cols in (("groups", ("resid", "ca", *GROUPS)), ("lobes", ("resid", *LOBES))):
            fig, ax = plt.subplots(2, len(cols), figsize=(2.3 * len(cols) + 1, 5), layout="constrained")
            for i, ev in enumerate(("e1", "e2")):
                for j, c in enumerate(cols):
                    key = f"{c}_{ev}" if c in ("resid", "ca") else f"d_{c}_{ev}"
                    im = ax[i, j].imshow(maps[s][key][y0:y1, x0:x1], cmap="RdBu_r", norm=sroot4_norm(M))
                    ttl = {"resid": "reference − R0", "ca": "C − A (fixed R0 proposal)"}.get(c, f"Δ {c}")
                    ax[i, j].set_title(f"{ev.upper()} {ttl}", fontsize=8)
                    ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
                fig.colorbar(im, ax=ax[i, -1], shrink=0.8)
            fig.suptitle(f"{s}: mean of 12 replicates, luma radiance, shared signed scale ±{M:.3g} (signed fourth root)")
            f = out / f"scene_{kind}_{s}.png"
            fig.savefig(f, dpi=110); plt.close(fig); files.append(f)
    return files, {"scene_signed_max": M, "scene_mapping": "signed (|x|/M)^(1/4), luma radiance, all scenes and panels"}


def compact(resp, par, scenes, rows):
    S = resp["strata"]
    res = {"response": {}, "scenes": {}, "parameters": {}, "c_minus_a": {}}
    for c in ("development", "held_out"):
        for b in BANDS:
            for region in ("all", "core", "shoulder", "tail"):
                blk = S[f"{c}|{b}|{region}"]
                res["response"][f"{c}|{b}|{region}"] = {
                    "residual_energy": blk["residual_energy"], "reference_noise_energy": blk["reference_noise_energy"],
                    "interventions": {n: {"label": st["label"], "rho": st["rho"], "error_change": st["error_change"],
                                          "change_over_noise": st["change_over_noise"], "relative_change_l2": st["relative_change_l2"],
                                          "energy_change": st["energy_change"], "ca_projection": st["ca_projection"],
                                          "ca_cos": st["ca_cos"], "confounded_with": st.get("confounded_with"),
                                          "matched": {k: st["matched"][k] for k in ("label", "rho", "error_change",
                                                                                     "qualified_probes", "shape_fraction_l2")}}
                                      for n, st in blk["interventions"].items() if n in K.INTERVENTIONS},
                    "interactions": {n: {"rho": blk["interventions"][n]["rho"], "relative_change_l2": blk["interventions"][n]["relative_change_l2"],
                                         "error_change": blk["interventions"][n]["error_change"]}
                                     for n in ("ATTEN_INTERACTION", "ETA_ATTEN_INTERACTION")},
                    "hierarchical_share": blk["hierarchical_share"], "term_luma_share": blk["term_luma_share"],
                    "ca_shares": blk["ca_shares"], "ca_energy": blk["ca_energy"]}
    for s, an in scenes.items():
        res["scenes"][s] = {"identity_relative_l1_max": {k: max(v) for k, v in an["identity_relative_l1"].items()},
                            "ca_closure": an["ca_closure"], "rois": {}}
        for roi, blk in an["rois"].items():
            res["scenes"][s]["rois"][roi] = {ev: {"residual_rms": b["residual_rms"], "ca_rms": b["ca_rms"],
                                                  "fixed_R0_proposal_vs_native_CA": b.get("fixed_R0_proposal_vs_native_CA"),
                                                  "ca_shares": b["ca_shares"],
                                                  "interventions": {n: {k: st.get(k) for k in ("label", "rho", "error_change", "rms_after",
                                                                                                "energy", "ca_projection", "ca_cos",
                                                                                                "confounded_with")}
                                                                    for n, st in b["interventions"].items()}}
                                             for ev, b in blk.items()}
    for r in rows:
        pr = par["results"][r["tag"]]
        res["parameters"][r["tag"]] = {
            "fd_max": pr.get("fd_max_column_relative_error"), "null_directions": len(pr["null_directions"]),
            "weak_directions": len(pr["weak_directions"]), "weak_loadings": pr["weak_directions"],
            "null_loadings": pr["null_directions"],
            "strata": {k: {q: v for q, v in st.items() if q.startswith(("R2_", "unique_R2_"))} | {"lobe_R2": st["lobe_R2"]}
                       for k, st in pr["strata"].items()},
            "max_principal_cosine": {k: max(v) for k, v in pr["principal_cosines"].items()},
            "ig_completeness": pr["ig_completeness_relative"], "ig_ca_shares": pr["ig_ca_shares"]}
    # response / scene agreement on E1 objects (per material rho over all probes vs scene rho)
    agree = {}
    for s, mat in SCENE_MAT.items():
        i = next(k for k, r in enumerate(rows) if r["tag"] == mat)
        cohort = rows[i]["cohort"]
        members = [k for k, r in enumerate(rows) if r["cohort"] == cohort]
        pos = members.index(i)
        blk = S[f"{cohort}|all|all"]["interventions"]
        rr = np.array([blk[n]["per_material_rho"][pos]["point"] for n in K.NAMES])
        sc = {ev: np.array([scenes[s]["rois"]["objects"][ev]["interventions"][n]["rho"]["point"] for n in K.NAMES])
              for ev in ("e1", "e2")}
        from scipy.stats import spearmanr
        agree[s] = {ev: {"spearman": float(spearmanr(rr, v).statistic), "sign_agreement": int((np.sign(rr) == np.sign(v)).sum()),
                         "n": len(K.NAMES), "response_rho": rr.tolist(), "scene_rho": v.tolist()} for ev, v in sc.items()}
    res["agreement_response_scene"] = agree
    return res


def decision(cmp, rows):
    """Collect the labels into the brief's completion question (no new rule)."""
    out = {"localized": [], "confounded": [], "detectable_only": {}, "no_detectable": [], "unqualified": []}
    for key, blk in cmp["response"].items():
        for n, st in blk["interventions"].items():
            lab = st["label"]
            tagged = f"response {key} {n}"
            if lab == "CONDITIONAL_ERROR_LOCALIZATION":
                out["localized"].append(tagged)
            elif lab == "CONFOUNDED_OR_NON_IDENTIFIABLE":
                out["confounded"].append(tagged)
            elif lab == "NO_DETECTABLE_COMPONENT_EFFECT":
                out["no_detectable"].append(tagged)
            elif lab == "EVIDENCE_UNQUALIFIED":
                out["unqualified"].append(tagged)
    for s, sc in cmp["scenes"].items():
        for roi, blk in sc["rois"].items():
            for ev, b in blk.items():
                for n, st in b["interventions"].items():
                    if n not in K.INTERVENTIONS:
                        continue
                    lab = st["label"]
                    tagged = f"scene {s} {roi} {ev} {n}"
                    if lab == "CONDITIONAL_ERROR_LOCALIZATION":
                        out["localized"].append(tagged)
                    elif lab == "CONFOUNDED_OR_NON_IDENTIFIABLE":
                        out["confounded"].append(tagged)
                    elif lab == "NO_DETECTABLE_COMPONENT_EFFECT":
                        out["no_detectable"].append(tagged)
                    elif lab == "EVIDENCE_UNQUALIFIED":
                        out["unqualified"].append(tagged)
    counts = {}
    for key, blk in cmp["response"].items():
        for n, st in blk["interventions"].items():
            counts.setdefault(n, {}).setdefault(st["label"], 0)
            counts[n][st["label"]] += 1
    out["response_label_counts_per_intervention"] = counts
    out["note"] = "Diagnostic only. CONDITIONAL_ERROR_LOCALIZATION is conditional on the frozen decomposition and fixed conditions; it is not a unique root cause. No adoption, replacement or retention decision is made."
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("primary", type=Path)
    ap.add_argument("review", type=Path)
    args = ap.parse_args()
    args.review.mkdir(parents=True, exist_ok=False)
    resp = IO.read(args.primary / "response/response.json")
    par = IO.read(args.primary / "params/params.json")
    scenes = {s: IO.read(args.primary / "scenes" / s / "analysis.json") for s in SCENES}
    rows = resp["materials"]
    with np.load(args.primary / "response/lanes.npz") as z:
        lanes = {k: z[k] for k in z.files}
    files = []
    f, scale_r = angular_maps(lanes, rows, args.review); files += f
    files.append(contributions_fig(resp, rows, args.review))
    files.append(stratum_contrib_fig(resp, args.review))
    files += rho_heatmaps(resp, args.review)
    files.append(identifiability_fig(resp, args.review))
    files += params_fig(par, rows, args.review)
    files.append(ca_fig(resp, par, scenes, rows, args.review))
    files.append(scene_rho_fig(scenes, args.review))
    f, scale_s = scene_maps(args.primary, args.review); files += f
    cmp = compact(resp, par, scenes, rows)
    IO.write(args.review / "summary.json", cmp)
    IO.write(args.review / "decision.json", decision(cmp, rows))
    shutil.copyfile(args.primary / "tests.json", args.review / "tests.json")
    for s in SCENES:
        shutil.copyfile(args.primary / "scenes" / s / "analysis.json", args.review / f"scene_{s}.json")
    shutil.copyfile(args.primary / "params/params.json", args.review / "params.json")
    posthoc = args.primary.parent / "posthoc"
    for name in ("coupling.json", "term_collinearity.json"):
        if (posthoc / name).exists():
            shutil.copyfile(posthoc / name, args.review / f"posthoc_{name}")
    raw = {}
    for p in sorted(args.primary.rglob("*")):
        if p.is_file() and p.suffix in (".json", ".npz", ".log", ".sh"):
            raw[p.relative_to(args.primary).as_posix()] = IO.sha(p)
    review = {p.name: IO.sha(p) for p in sorted(args.review.iterdir()) if p.is_file()}
    IO.write(args.review / "manifest.json", {
        "batch": "WL104H", "branch": "ecro_lr3_adapter", "freeze_commit": "fbdef22b09a8d65810f3e3cfa3ce1878dc66833e",
        "package_commit": IO.commit(), "protocol_sha256": IO.sha(IO.HERE / "protocol.json"),
        "code_sha256": {p.name: IO.sha(p) for p in sorted(IO.HERE.glob("*.py"))},
        "inputs_sha256": resp["inputs_sha256"], "scene_inputs_sha256": {s: scenes[s]["inputs_sha256"] for s in SCENES},
        "display_scales": {**scale_r, **scale_s}, "raw_root": "results/ecro_wl84_104h/primary", "raw_sha256": raw,
        "review_sha256": review})
    print("package done", len(files), flush=True)


if __name__ == "__main__":
    main()
