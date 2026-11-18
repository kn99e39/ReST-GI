"""WL101H offline analysis: F6 effect (stage D), qualification gates (stage F), frozen verdict, figures.

python analyze101h.py RESULTS_DIR WL101G_PRIMARY_A2 WL101_REFERENCES REVIEW_DIR
RESULTS_DIR holds replay_native, replay_candidate, D_construct_{native,candidate},
D_camera_{native,candidate}, F_preflight_{native,candidate}.
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
from scipy.stats import t as student

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_sss101g"))
sys.path.insert(0, str(HERE.parent / "ecro86"))
sys.path.insert(0, str(HERE))
from regions101g import A, EDGES, view_regions  # noqa: E402
import tables101g as T  # noqa: E402
import instrument101h as I  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722])
PROTOCOL = HERE / "protocol/qualification.json"
P = json.loads(PROTOCOL.read_text())
THETA_ABS, THETA_REL = 7.0e-4, 2.5e-4
PATCH = 155


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def interval(grid):
    grid = np.asarray(grid, np.float64)
    if grid.ndim == 1:
        n = len(grid)
        se = grid.std(ddof=1) / math.sqrt(n)
        half = float(student.ppf(.975, n - 1) * se) if se > 0 else 0.
        m = float(grid.mean())
        return {"mean": m, "ci95": [m - half, m + half], "half_width95": half, "values": grid.tolist()}
    n, m = grid.shape
    u, v = grid.mean(1).var(ddof=1) / n, grid.mean(0).var(ddof=1) / m
    se = math.sqrt(u + v)
    df = (u + v) ** 2 / (u * u / (n - 1) + v * v / (m - 1)) if se > 0 else None
    half = float(student.ppf(.975, df) * se) if df else 0.
    c = float(grid.mean())
    return {"mean": c, "ci95": [c - half, c + half], "half_width95": half, "table_se": math.sqrt(u), "camera_se": math.sqrt(v)}


def excludes_zero(ci):
    return ci[0] > 0 or ci[1] < 0


def load_dir(d):
    man = json.loads((d / "run_manifest.json").read_text())
    for name, rec in man["files"].items():
        assert sha(d / name) == rec["sha256"], name
    return man


def packets(d, man):
    return [dict(np.load(d / f"construction_{s}.npz")) for s in man["seeds"]]


def lum_tables(pk):
    tabs = T.independent_tables(pk, names=("Un", "V", "G", "S_G"))
    return {k: v[0].astype(np.float64) @ LUMA for k, v in tabs.items()}


# ---------------------------------------------------------------- stage D, table level
def reproduction(pkN, wl101g):
    rows = []
    for i in range(3):
        old = np.load(wl101g / f"construction_{i}.npz")
        exact = all(np.array_equal(pkN[i][k], old[k]) for k in ("n", "v"))
        rel = max(float(np.max(np.abs(pkN[i][k] - old[k])) / max(float(np.max(np.abs(old[k]))), 1e-30))
                  for k in ("c", "g", "L", "LL", "vL", "gL"))
        rows.append({"replica": i, "n_v_identical": exact, "max_relative_real_difference": rel})
    return {"rows": rows, "pass": all(r["n_v_identical"] and r["max_relative_real_difference"] <= 1e-6 for r in rows)}


def packet_differences(pkN, pkC):
    out = []
    for a, b in zip(pkN, pkC):
        La, Lb = (a["L"].sum((1, 2)) @ LUMA), (b["L"].sum((1, 2)) @ LUMA)
        rel = np.abs(La - Lb) / np.maximum(np.abs(La), 1e-9)
        contact = (a["k_ring"].sum((1, 2)) + a["k_cap"].sum((1, 2)) + b["k_singular"].sum((1, 2))) > 0
        changed = rel > 1e-5
        out.append({"slots_changed_gt_1e-5": int(changed.sum()), "changed_slots_with_contact_classes": int((changed & contact).sum()),
                    "changed_slots_without_contact_classes": int((changed & ~contact).sum()),
                    "max_relative_change_without_contact_classes": float(rel[~contact].max()),
                    "n_identical_outside_contact": bool(np.array_equal(a["n"][~contact], b["n"][~contact])),
                    "native_ring": int(a["k_ring"].sum()), "native_cap": int(a["k_cap"].sum()),
                    "candidate_ring": int(b["k_ring"].sum()), "candidate_singular": int(b["k_singular"].sum())})
    return out


def table_effect(pkN, pkC, pids):
    tN = [lum_tables(p) for p in pkN]
    tC = [lum_tables(p) for p in pkC]
    res = {}
    for pid in pids:
        sl = slice(pid * 64, pid * 64 + 64)
        row = {}
        for fam in ("Un", "V", "G"):
            d = np.stack([c[fam][sl] - n[fam][sl] for n, c in zip(tN, tC)])
            mean, se = d.mean(0), d.std(0, ddof=1) / 2
            material = np.abs(mean) > np.maximum(3 * se, THETA_ABS)
            row[fam] = {"native_mean": float(np.mean([n[fam][sl].mean() for n in tN])),
                        "candidate_mean": float(np.mean([c[fam][sl].mean() for c in tC])),
                        "delta_patch_mean": float(mean.mean()), "delta_max_abs": float(np.abs(mean).max()),
                        "material_bins": int(material.sum()), "delta_per_bin": mean.tolist(), "se_per_bin": se.tolist()}
        dug = np.stack([(c["Un"][sl] - c["G"][sl]) - (n["Un"][sl] - n["G"][sl]) for n, c in zip(tN, tC)])
        m, se = dug.mean(0), dug.std(0, ddof=1) / 2
        row["U_minus_G_change"] = {"patch_mean": float(m.mean()), "max_abs": float(np.abs(m).max()),
                                   "material_bins": int((np.abs(m) > np.maximum(3 * se, THETA_ABS)).sum()),
                                   "native_U_minus_G_patch_mean": float(np.mean([(n["Un"][sl] - n["G"][sl]).mean() for n in tN])),
                                   "candidate_U_minus_G_patch_mean": float(np.mean([(c["Un"][sl] - c["G"][sl]).mean() for c in tC]))}
        sg = np.stack([c["S_G"][sl, 3] - n["S_G"][sl, 3] for n, c in zip(tN, tC)])
        sun = np.stack([(c["Un"][sl] - n["Un"][sl]) for n, c in zip(tN, tC)])
        row["S_G_subcell3"] = {"delta_patch_mean": float(sg.mean()), "delta_max_abs": float(np.abs(sg.mean(0)).max()),
                               "material_bins": int((np.abs(sg.mean(0)) > np.maximum(3 * sg.std(0, ddof=1) / 2, THETA_ABS)).sum())}
        res[str(pid)] = row
    return res


# ---------------------------------------------------------------- stage D, image level
class Cached:
    """npz access with one decompression per key."""
    def __init__(self, path):
        self.z = np.load(path)
        self.files = self.z.files
        self.cache = {}

    def __getitem__(self, key):
        if key not in self.cache:
            self.cache[key] = self.z[key]
        return self.cache[key]


def load_camera(d):
    man = load_dir(d)
    for s in man["streams"]:
        assert sha(d / s["file"]) == s["sha256"], s["file"]
    files = [Cached(d / s["file"]) for s in man["streams"]]
    return man, files


def regions_v0(refs):
    reg, bands = view_regions(refs / "MAP_0.npz", True)
    with np.load(refs / "MAP_0.npz") as m:
        pos, ok = m["p"], m["ok"]
    dist = np.linalg.norm(pos - np.array([0.9, 0.4, 0.0]), axis=-1)
    reg["far_control"] = ok & (dist >= 1.0)
    return reg, bands


def luma_mean(img, mask):
    return float((img[mask].astype(np.float64) @ LUMA).mean())


def grids(files, fam, mask):
    reps = sorted({int(k.rsplit("_", 1)[1]) for k in files[0].files if k.rsplit("_", 1)[0] == fam and ":" not in k})
    return np.array([[luma_mean(f[f"{fam}_{i}"], mask) for f in files] for i in reps])


def image_effect(camN, camC, refs):
    manN, fN = camN
    manC, fC = camC
    reg, bands = regions_v0(refs)
    masks = dict(reg)
    for (a, b), m in zip(zip(EDGES, EDGES[1:]), bands):
        masks[f"band_{a:.2f}_{b:.2f}"] = m
    refN = {k: np.array([luma_mean(f["query"], m) for f in fN]) for k, m in masks.items()}
    refC = {k: np.array([luma_mean(f["query"], m) for f in fC]) for k, m in masks.items()}
    out = {"reference_shift_C_minus_N": {}, "residuals": {}, "delta_residual_C_minus_N": {}, "delta_U_minus_G": {},
           "table_only_change_under_N_pass": {}, "rmse": {}}
    for k in masks:
        d = refC[k] - refN[k]
        out["reference_shift_C_minus_N"][k] = {**interval(d), "native_reference": float(refN[k].mean())}
    for cond, files, ref in (("N", fN, refN), ("C", fC, refC)):
        out["residuals"][cond] = {}
        for fam in ("U_E", "UnN", "VN", "GN", "S_GN", "UnC", "VC", "GC", "S_GC"):
            res = {}
            for k, m in masks.items():
                if m.sum() == 0:
                    continue
                res[k] = interval(grids(files, fam, m) - ref[k][None])
            res["contrast"] = interval((grids(files, fam, masks["near"]) - grids(files, fam, masks["flank"])) -
                                       (ref["near"] - ref["flank"])[None])
            out["residuals"][cond][fam] = res
    for base in ("Un", "V", "G", "S_G"):
        res = {}
        for k, m in list(masks.items()) + [("contrast", None)]:
            if k == "contrast":
                gN = (grids(fN, base + "N", masks["near"]) - grids(fN, base + "N", masks["flank"])) - (refN["near"] - refN["flank"])[None]
                gC = (grids(fC, base + "C", masks["near"]) - grids(fC, base + "C", masks["flank"])) - (refC["near"] - refC["flank"])[None]
            else:
                gN = grids(fN, base + "N", m) - refN[k][None]
                gC = grids(fC, base + "C", m) - refC[k][None]
            iv = interval(gC - gN)
            band = 6.26e-4 if k == "contrast" else THETA_ABS
            iv["material"] = bool(excludes_zero(iv["ci95"]) and abs(iv["mean"]) > band)
            res[k] = iv
        out["delta_residual_C_minus_N"][base] = res
    for k, m in list(masks.items()) + [("contrast", None)]:
        if k == "contrast":
            f = lambda files, fam: grids(files, fam, masks["near"]) - grids(files, fam, masks["flank"])
        else:
            f = lambda files, fam, m=m: grids(files, fam, m)
        dC = f(fC, "UnC") - f(fC, "GC")
        dN = f(fN, "UnN") - f(fN, "GN")
        iv = interval(dC - dN)
        iv["native_U_minus_G"] = float(dN.mean())
        iv["candidate_U_minus_G"] = float(dC.mean())
        iv["material"] = bool(excludes_zero(iv["ci95"]) and abs(iv["mean"]) > (6.26e-4 if k == "contrast" else THETA_ABS))
        out["delta_U_minus_G"][k] = iv
        out["table_only_change_under_N_pass"][k] = interval(f(fN, "UnC") - f(fN, "UnN"))
    # Noise-corrected RMSE against each condition's own reference.
    def stack(files, fam):
        reps = sorted({int(k.rsplit("_", 1)[1]) for k in files[0].files if k.rsplit("_", 1)[0] == fam and ":" not in k})
        return np.stack([np.stack([f[f"{fam}_{i}"] for f in files]) for i in reps])
    qN = np.stack([f["query"] for f in fN]).astype(np.float64)
    qC = np.stack([f["query"] for f in fC]).astype(np.float64)
    for region in ("near", "flank", "contact_shadow", "sphere", "sphere_boundary", "floor", "full", "far_control"):
        m = masks[region]
        row = {}
        for base in ("Un", "G"):
            a = T.cross_rmse(stack(fN, base + "N"), qN, m)["relative"]
            b = T.cross_rmse(stack(fC, base + "C"), qC, m)["relative"]
            row[base] = {"native": a, "candidate": b, "delta": b - a, "material": bool(abs(b - a) > 0.0025)}
        out["rmse"][region] = row
    return out, masks


# ---------------------------------------------------------------- stage F
def patch_lengths():
    import scene86 as S
    g = S.Geom(A["geometry"])
    return np.linalg.norm(g.PE1, axis=1), np.linalg.norm(g.PE2, axis=1), g.PSID


def gates(man, pks, semantics, b_m):
    l1, l2, psid = patch_lengths()
    planar = np.repeat(psid != 6, 64)
    L1, L2 = np.repeat(l1, 64), np.repeat(l2, 64)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_band = 6.4e-5 + np.where(planar, 2 * b_m * (1 / L1 + 1 / L2), 0.0)
    res = {"S1": [], "S2": [], "S3": [], "S4": [], "labels": [], "classes": []}
    d4 = []
    for meta, pk in zip(man["replicas"], pks):
        m = meta["membership"]
        res["S1"].append({"seed": meta["seed"], "invalid": m["invalid"], "support_conflict": m.get("support_conflict", 0),
                          "pass": m["invalid"] == 0 and m.get("support_conflict", 0) == 0})
        cl = meta["classes"]
        res["S2"].append({"seed": meta["seed"], "ring": cl["ring"], "inside_box": cl["inside_box"],
                          "pass": cl["ring"] == 0 and cl["inside_box"] == 0})
        excl_slot = (pk["slot_tie"] + pk["slot_boundary_miss"]) / P["construction"]["proposals_per_slot"]
        sing = pk["k_singular"].sum(2)
        sub_frac = np.divide(sing, pk["n"].sum(2) + sing, out=np.zeros_like(sing), where=(pk["n"].sum(2) + sing) > 0)
        res["S3"].append({"seed": meta["seed"], "max_slot_excluded_fraction": float(excl_slot.max()),
                          "max_subcell_singular_fraction": float(sub_frac.max()),
                          "pass": bool(excl_slot.max() <= 1e-3 and sub_frac.max() <= 1e-3)})
        n = pk["n"]
        Lst = pk["L"] @ LUMA
        mean_st = np.divide(Lst, n, out=np.zeros_like(Lst), where=n >= 16)
        Lslot = np.divide(Lst.sum((1, 2)), n.sum((1, 2)), out=np.zeros(len(n)), where=n.sum((1, 2)) > 0)
        dev = np.where(n >= 16, np.abs(mean_st - Lslot[:, None, None]), 0).reshape(len(n), -1).max(1)
        Ldev = 2 * dev
        gsum, gL = pk["g"].sum((1, 2)), pk["gL"].sum((1, 2)) @ LUMA
        G = np.divide(gL, gsum, out=np.zeros_like(gL), where=gsum > 0)
        f_sing = np.divide(sing.sum(1), n.sum((1, 2)) + sing.sum(1), out=np.zeros(len(n)), where=(n.sum((1, 2)) + sing.sum(1)) > 0)
        kj, vsum = pk["k_junction"].sum((1, 2)), pk["v"].sum((1, 2))
        Lj = np.divide(pk["L_junction"].sum((1, 2)) @ LUMA, kj, out=np.zeros(len(n)), where=kj > 0)
        b_junction = np.divide(kj, vsum, out=np.zeros(len(n)), where=vsum > 0) * np.abs(Lj - G)
        bound = f_band * Ldev + f_sing * Ldev + b_junction
        active = gsum > 0
        fail = active & ((bound > THETA_ABS) | (bound > THETA_REL * np.maximum(np.abs(G), 1e-3)))
        worst = np.argsort(-(bound * active))[:5]
        res["S4"].append({"seed": meta["seed"], "max_bound": float(bound[active].max()), "failing_slots": int(fail.sum()),
                          "failing_slot_ids": np.flatnonzero(fail)[:50].tolist(),
                          "worst": [{"slot": int(s), "bound": float(bound[s]), "G": float(G[s]),
                                     "parts": [float(f_band[s] * Ldev[s]), float(f_sing[s] * Ldev[s]), float(b_junction[s])]} for s in worst],
                          "pass": bool(fail.sum() == 0)})
        res["labels"].append(meta["labels"])
        res["classes"].append(cl)
        gs, gLs = pk["g_small"].sum((1, 2)), pk["gL_small"].sum((1, 2)) @ LUMA
        Gs = np.divide(gLs, gs, out=np.zeros_like(gLs), where=gs > 0)
        d4.append(np.where(active & (gs > 0), Gs - G, 0.0))
        res.setdefault("G", []).append(G)
    d4 = np.array(d4)
    seeds_intended = P["construction"]["intended_frozen_seeds"]
    sel = [i for i, meta in enumerate(man["replicas"]) if meta["seed"] in seeds_intended]
    mean, se = d4[sel].mean(0), d4[sel].std(0, ddof=1) / math.sqrt(len(sel))
    Gm = np.array(res.pop("G"))[sel].mean(0)
    flagged = ((np.abs(mean) > THETA_ABS) | (np.abs(mean) > THETA_REL * np.maximum(np.abs(Gm), 1e-3))) & (np.abs(mean) > 3 * se)
    res["D4"] = {"flagged_slots": int(flagged.sum()), "flagged_slot_ids": np.flatnonzero(flagged)[:200].tolist(),
                 "max_abs_mean_delta": float(np.abs(mean).max()),
                 "flagged_by_shape": {int(s): int(flagged.reshape(-1, 64)[psid == s].sum()) for s in np.unique(psid)}}
    res["D3_unregistered_events"] = man.get("unregistered_events", [])
    res["camera_independence"] = man["camera_independence"]
    ci = man["camera_independence"]
    res["camera_independence_pass"] = all(v["integer_columns_and_counts_identical"] and v["max_relative_real_difference"] <= 1e-5
                                         for v in ci.values())
    # Planar cos normalisation on the intended replicas.
    rows = []
    sb = int((psid != 6).sum())
    for k in range(4):
        c = sum(pks[i]["c"][:sb * 64].reshape(sb, 64, 16)[:, 16 * k:16 * (k + 1)].sum() for i in sel)
        nn = sum(pks[i]["n"][:sb * 64].reshape(sb, 64, 16)[:, 16 * k:16 * (k + 1)].sum() for i in sel)
        se_k = 0.25 / math.sqrt(12) / math.sqrt(nn)
        rows.append({"ring": k, "standard_errors": float((c / nn - (2 * k + 1) / 8) / se_k)})
    res["cos_normalization"] = {"rows": rows, "pass": all(abs(r["standard_errors"]) <= 5 for r in rows)}
    for key in ("S1", "S2", "S3", "S4"):
        res[f"{key}_pass"] = all(r["pass"] for r in res[key])
    return res


# ---------------------------------------------------------------- replay comparison (X checks)
def replay_compare(rN, rC):
    out = {}
    for name, d in (("native", rN), ("candidate", rC)):
        z = np.load(d / "part2_radial_grid.npz")
        meta, radii = z["meta"], z["radii_mm"]
        L = (z["f2_mean"] + z["fh_mean"]) @ LUMA
        se = np.sqrt(((z["f2_sd"] ** 2 + z["fh_sd"] ** 2) @ LUMA ** 2) / 256)
        att = (z["nee_through"] @ LUMA) / np.maximum((z["nee_f"] @ LUMA) * 27.634931507169394 / z["nee_light_d2"], 1e-30)
        phys = np.exp(-12 * z["chord_physical"])
        rows = []
        for ri, r in enumerate(radii):
            m = meta[:, 0] == ri
            shadowed = m & (z["nee_cross"] > 0)
            rows.append({"r_mm": float(r), "L": float(L[m].mean()), "L_se": float(np.sqrt(np.mean(se[m] ** 2) / m.sum())),
                         "E2": float((z["f2_mean"][m] @ LUMA).mean()), "E3plus": float((z["fh_mean"][m] @ LUMA).mean()),
                         "attenuation_ratio_to_physical": float(np.median(att[shadowed] / phys[shadowed])) if shadowed.any() else None,
                         "origin_inside_sphere": float(z["origin_inside_sphere_nee"][m].mean()),
                         "visibility_front": float(z["vis_front"][m].mean())})
        out[name] = rows
    p1 = np.load(rC / "part1_construction_samples.npz")
    out["candidate_part1"] = {"samples": int(len(p1["bin"]))}
    return out


# ---------------------------------------------------------------- figures and page
def figures(out_dir, replay, tables_eff, image, f_cand):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for name, color, marker in (("native", "#2a78d6", "o"), ("candidate", "#eb6834", "s")):
        rows = replay[name]
        r = [x["r_mm"] for x in rows]
        ax[0].errorbar(r, [x["L"] for x in rows], yerr=[1.96 * x["L_se"] for x in rows], fmt=marker + "-", color=color, ms=4, label=name)
        ax[1].plot(r, [x["E3plus"] for x in rows], marker + "-", color=color, ms=4, label=f"{name} E3+")
        ax[1].plot(r, [x["E2"] for x in rows], marker + "--", color=color, ms=4, alpha=.6, label=f"{name} E2")
    for a in ax:
        a.set_xscale("log"); a.axvline(11.58, color="#58616b", lw=.8, ls=":"); a.grid(alpha=.2)
        a.set_xlabel("floor radius from the tangency (mm)")
    ax[0].set_ylabel("mean outgoing E2-E5 luminance"); ax[1].set_ylabel("component luminance")
    ax[0].legend(); ax[1].legend(fontsize=8)
    fig.suptitle("WL101H · radiance across the native ring boundary (dotted: native r* = 1.158 cm)")
    fig.savefig(out_dir / "radial_contact.png", dpi=140); plt.close(fig)

    row = tables_eff[str(PATCH)]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), layout="constrained", sharey=True)
    for a, fam in zip(ax, ("Un", "V", "G")):
        d = np.array(row[fam]["delta_per_bin"]); se = np.array(row[fam]["se_per_bin"])
        a.bar(np.arange(64), d, yerr=1.96 * se, color="#2a78d6", width=.8)
        a.axhline(0, color="#58616b", lw=.8); a.axhspan(-THETA_ABS, THETA_ABS, color="#7a7a7a", alpha=.15)
        a.set_title(f"{fam}: candidate - native, floor patch {PATCH}", fontsize=10); a.set_xlabel("direction bin")
    ax[0].set_ylabel("table luminance difference")
    fig.savefig(out_dir / "contact_slot_tables.png", dpi=140); plt.close(fig)

    keys = ["near", "flank", "contrast", "contact_shadow", "sphere", "sphere_boundary", "floor", "far_control", "band_0.10_0.14"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained")
    for a, (title, src) in zip(ax, (("C - N residual change (U arm)", image["delta_residual_C_minus_N"]["Un"]),
                                    ("C - N change of (U - G)", image["delta_U_minus_G"]))):
        vals = [src[k]["mean"] for k in keys]
        err = [[src[k]["mean"] - src[k]["ci95"][0] for k in keys], [src[k]["ci95"][1] - src[k]["mean"] for k in keys]]
        a.errorbar(range(len(keys)), vals, yerr=err, fmt="o", color="#eb6834")
        a.axhline(0, color="#58616b", lw=.8); a.axhspan(-THETA_ABS, THETA_ABS, color="#7a7a7a", alpha=.15)
        a.set_xticks(range(len(keys)), keys, rotation=35, ha="right"); a.set_title(title, fontsize=10); a.grid(alpha=.2, axis="y")
    fig.suptitle("WL101H · image-level effect at V0 (each condition against its own reference; band = materiality)")
    fig.savefig(out_dir / "image_effect.png", dpi=140); plt.close(fig)


def main(results, wl101g, refs, out):
    out.mkdir(parents=True, exist_ok=True)
    R = {k: results / k for k in ("replay_native", "replay_candidate", "D_construct_native", "D_construct_candidate",
                                  "D_camera_native", "D_camera_candidate", "F_preflight_native", "F_preflight_candidate")}
    manN, manC = load_dir(R["D_construct_native"]), load_dir(R["D_construct_candidate"])
    pkN, pkC = packets(R["D_construct_native"], manN), packets(R["D_construct_candidate"], manC)
    result = {"protocol_sha256": sha(PROTOCOL)}
    result["reproduction"] = reproduction(pkN, wl101g)
    result["packet_differences"] = packet_differences(pkN, pkC)
    result["table_effect"] = table_effect(pkN, pkC, [PATCH, 154, 156, 139, 171])
    del pkN, pkC
    camN, camC = load_camera(R["D_camera_native"]), load_camera(R["D_camera_candidate"])
    result["camera_consistency"] = {"native": camN[0]["consistency"], "candidate": camC[0]["consistency"],
                                    "native_replay": [s.get("replay") for s in camN[0]["streams"]]}
    image, masks = image_effect(camN, camC, refs)
    result["image_effect"] = image
    result["replay"] = replay_compare(R["replay_native"], R["replay_candidate"])
    b_m = P["membership"]["b_m_m"]
    F = {}
    for sem in ("native", "candidate"):
        man = load_dir(R[f"F_preflight_{sem}"])
        F[sem] = gates(man, packets(R[f"F_preflight_{sem}"], man), sem, b_m)
    result["stage_F"] = F
    # S5 instrument consistency for the candidate.
    cc = camC[0]["consistency"]
    s5 = {"split_consistency_all_views": all(v["smoke"]["event_errors"]["h"]["relative_l1"] <= P["gates"]["smoke_event_relative_l1"]
                                             for v in cc.values()),
          "low_invariance_all_views": all(v["low"]["low_max_abs_zero_vs_learned_table"] == 0 for v in cc.values()),
          "cos_normalization": F["candidate"]["cos_normalization"]["pass"],
          "camera_independence": F["candidate"]["camera_independence_pass"],
          "native_reproduction_of_WL101G": result["reproduction"]["pass"]}
    s5["pass"] = all(s5.values())
    fc = F["candidate"]
    discrimination = (not F["native"]["S2_pass"]) and F["native"]["S1_pass"]
    sem_pass = fc["S1_pass"] and fc["S2_pass"] and fc["S3_pass"] and fc["S4_pass"] and s5["pass"]
    d3_empty = len(fc["D3_unregistered_events"]) == 0
    if not discrimination:
        verdict = "STOP_GATE_CANNOT_DISTINGUISH"
    elif sem_pass and d3_empty and fc["D4"]["flagged_slots"] == 0:
        verdict = "QUALIFIED"
    elif sem_pass and d3_empty:
        verdict = "PARTIALLY_QUALIFIED"
    else:
        verdict = "NOT_QUALIFIED"
    t155 = result["table_effect"][str(PATCH)]
    di = image["delta_U_minus_G"]
    answers = {
        "contact_slot_material": {fam: t155[fam]["material_bins"] > 0 for fam in ("Un", "V", "G")},
        "U_G_comparison_material": {"slot_bins": t155["U_minus_G_change"]["material_bins"] > 0,
                                    **{k: di[k]["material"] for k in ("near", "contrast", "contact_shadow", "sphere", "sphere_boundary")}},
        "lobes_material": {"reference_shift": bool(excludes_zero(image["reference_shift_C_minus_N"]["band_0.10_0.14"]["ci95"]) and
                                                   abs(image["reference_shift_C_minus_N"]["band_0.10_0.14"]["mean"]) > THETA_ABS),
                           "U_residual_change": image["delta_residual_C_minus_N"]["Un"]["band_0.10_0.14"]["material"]}}
    result["answers"] = answers
    result["S5"] = s5
    result["discrimination_check"] = discrimination
    result["verdict"] = verdict
    out_json = out / "metrics.json"
    out_json.write_text(json.dumps(result, indent=1, default=float) + "\n")
    figures(out, result["replay"], result["table_effect"], image, fc)
    rows = "".join(f"<tr><td>{k}</td><td>{html.escape(json.dumps(v))}</td></tr>" for k, v in answers.items())
    (out / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WL101H contact semantics</title><style>:root{{--bg:#fcfcfb;--ink:#1f1f1f;--line:#d9d8d2}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--ink:#f0efec;--line:#3b3b38}}}}
body{{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;max-width:1200px;margin:0 auto;padding:16px}}
img{{max-width:100%;height:auto;background:#fff}} td{{border-bottom:1px solid var(--line);padding:3px 8px;vertical-align:top}}</style></head><body>
<h1>WL101H · estimator/reference contact semantics and qualification</h1><p><b>Verdict:</b> {verdict}</p>
<table>{rows}</table><p><img src="radial_contact.png"></p><p><img src="contact_slot_tables.png"></p><p><img src="image_effect.png"></p>
<p><a href="metrics.json">metrics.json</a></p></body></html>""")
    manifest = {"review_files": {p.name: sha(p) for p in sorted(out.iterdir()) if p.name != "manifest101h.json"}}
    (out / "manifest101h.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(verdict, json.dumps(answers), flush=True)


if __name__ == "__main__":
    main(*[Path(a).resolve() for a in sys.argv[1:5]])
