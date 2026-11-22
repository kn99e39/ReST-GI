"""WL104K-A Stage C: scene-conditioned angular attribution. For every visible first-bounce lane on the ER objects of a frozen scene,
record (pixel, view direction, light direction, light weight), evaluate the physical layered response (fresh independent walks)
and the A / T / P responses (12k and extended states, every start) at exactly those directions, check the correspondence with the
rendered fixed-pass E1 images, and attribute the image residual by region and direction.

Usage: python lanes104ka.py <scene> <ext_dir> <out_dir>
"""
import sys
import time
from pathlib import Path

import numpy as np

import common104ka as KA

CM, IO = KA.CM, KA.IO
sys.path.insert(0, str(KA.HERE.parent / "ecro_wl84_104k"))
import scenes104k as SK  # noqa: E402

S8, C, RT, mi, dr = SK.S8, SK.C, SK.RT, SK.mi, SK.dr
from common104d import DA  # noqa: E402

LUMA = CM.LUMA
K_SPP = 16
WALKS = 8192
REGIONS = ("objects", "sphere", "cube", "highlight_core", "highlight_ring", "grazing", "sphere_lower")
MIRROR_BINS = np.array([0, 0.5, 1, 2, 3, 5, 8, 12, 20, 40, 180.0])


def extract(ctx, light, seed, k=K_SPP):
    n = C.RES * C.RES * k
    rng = mi.PCG32(size=n, initstate=seed, initseq=mi.UInt64(0))
    ray, pix = ctx.rays(k, rng)
    si, sid, _, ok = ctx.surface(ray, mi.Bool(True))
    active = ok & (si.wi.z > 1e-6)
    er = ctx.uses_er(sid)
    wo, vis, lf = ctx.light_term(si, light, active)
    m = vis & er
    dr.eval(m, pix, si.wi, wo, lf, sid)
    msk = np.asarray(m)
    v = lambda x: np.stack([np.asarray(x[c]) for c in range(3)], 1)[msk].astype(np.float64)
    return {"pix": np.asarray(pix)[msk].astype(np.int64), "wi": v(si.wi), "wo": v(wo), "lf": v(lf),
            "obj": np.asarray(sid)[msk].astype(np.int64), "n_lanes_total": n}


def region_stats(sel, c_ref_a, c_ref_b, c_mod):
    """sel bool [L]; c_* luminance contributions [L] -> signed excess, L1, noise-based SE."""
    cref = 0.5 * (c_ref_a + c_ref_b)
    S = cref[sel].sum()
    se_S = np.sqrt((((c_ref_a - c_ref_b) / 2)[sel] ** 2).sum())
    out = {}
    for name, c in c_mod.items():
        Sm = c[sel].sum()
        ex = Sm / S - 1
        out[name] = {"signed_excess": float(ex), "se": float(se_S * Sm / S ** 2), "l1": float(np.abs(c - cref)[sel].sum() / S)}
    out["_reference"] = {"sum": float(S), "se_rel": float(se_S / S), "l1_noise_floor": float((np.abs(c_ref_a - c_ref_b) / 2)[sel].sum() / S),
                         "lanes": int(sel.sum())}
    return out


def main():
    scene, ext, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rows, raw, compiled, desc, ref, seen = CM.inputs()
    n = len(rows)
    si_ = KA.SCENES.index(scene)
    i, row = SK.setup(scene)
    ctx = SK.Fixed104K(row["tag"] + "_" + scene + "_lanes", compiled["R0"], {}, i)
    light = RT.point_lights(row["light"])
    L = extract(ctx, light, KA.P["seeds"]["camera_lanes"] + si_)
    nl = len(L["pix"])
    # physical response at the lanes (fresh walks, two halves)
    tw = time.monotonic()
    a, b = DA.ref_eval(desc, np.full(nl, i), L["wi"], L["wo"], WALKS, KA.P["seeds"]["lane_walks"] + si_,
                       pair0=KA.P["seeds"]["lane_walk_pair0"] + si_ * 4_000_000)
    walk_s = time.monotonic() - tw
    lfl = L["lf"]
    c_ref_a, c_ref_b = (a * lfl) @ LUMA, (b * lfl) @ LUMA
    # model responses: 12k and extended, every start
    s12, R, _ = KA.states(KA.K_ROOT / "fit_primary", n)
    s24, _, _ = KA.states(ext, n)
    c_mod, F_mod = {}, {}
    for tag, st in (("12k", s12), ("ext", s24)):
        for arm in CM.ARMS:
            for r in range(R):
                s1 = {k: v[i:i + 1] for k, v in st[arm][r].items()}
                F = CM.predict(arm, s1, L["wi"][None, None], L["wo"][None, None])[0, 0]
                c_mod[f"{tag}|{arm}|{r}"] = (F * lfl) @ LUMA
    # ---- correspondence with the rendered fixed-pass E1 (12k states) and the archived reference
    arch, masks, _ = S8.archived(scene, None)
    rd = KA.K_ROOT / "posthoc_scenes" / scene
    fx = {}
    for rr in range(12):
        with np.load(rd / f"fixed_r{rr:03d}.npz") as z:
            for q in z.files:
                if q.endswith("|e1"):
                    fx.setdefault(q, []).append(z[q] @ LUMA)
    fxm = {q: np.mean(v, 0) for q, v in fx.items()}
    fxs = {q: np.std(v, 0, ddof=1) / np.sqrt(len(v)) for q, v in fx.items()}
    ref_er = (arch["reference_e1"] @ LUMA) - np.stack(fx["N|e1"])            # [12, H, W]
    ref_er_m, ref_er_se = ref_er.mean(0), ref_er.std(0, ddof=1) / np.sqrt(12)
    npx = C.RES * C.RES
    pixmap = lambda c: (np.bincount(L["pix"], c, npx) / K_SPP).reshape(C.RES, C.RES)
    corr = {}
    for reg in REGIONS:
        if reg not in masks or not masks[reg].any():
            continue
        mk = masks[reg]
        row_ = {}
        for arm in CM.ARMS:
            pred = np.mean([pixmap(c_mod[f"12k|{arm}|{r}"])[mk].sum() for r in range(R)])
            rend = np.mean([fxm[f"{arm}_{r}|e1"][mk].sum() for r in range(R)])
            row_[arm] = {"lane_pred": float(pred), "rendered": float(rend), "rel": float(pred / rend - 1)}
        pr = pixmap(0.5 * (c_ref_a + c_ref_b))[mk].sum()
        se_lane = np.sqrt((pixmap(np.abs(c_ref_a - c_ref_b) / 2)[mk] ** 2).sum())    # crude (per-pixel aggregated)
        rr_ = ref_er_m[mk].sum(); rse = np.sqrt((ref_er_se[mk] ** 2).sum())
        row_["reference"] = {"lane_pred": float(pr), "rendered": float(rr_), "rel": float(pr / rr_ - 1),
                             "combined_se_rel": float(np.hypot(se_lane, rse) / rr_)}
        row_["passed"] = bool(all(abs(row_[a]["rel"]) <= 0.02 for a in CM.ARMS) and
                              (abs(row_["reference"]["rel"]) <= 0.03 or abs(row_["reference"]["rel"]) <= 3 * row_["reference"]["combined_se_rel"]))
        corr[reg] = row_
    corr_ok = all(v["passed"] for v in corr.values())
    # ---- attribution by region (lane level)
    lane_region = {reg: masks[reg].reshape(-1)[L["pix"]] for reg in corr}
    attr = {}
    for reg, sel in lane_region.items():
        if sel.sum() == 0:
            continue
        attr[reg] = region_stats(sel, c_ref_a, c_ref_b, c_mod)
    # rendered (P - A) signed difference per region (12k, start-mean) vs lane-level
    for reg in attr:
        mk = masks[reg]
        rend_diff = np.mean([(fxm[f"P_{r}|e1"] - fxm[f"A_{r}|e1"])[mk].sum() for r in range(R)]) / ref_er_m[mk].sum()
        lane_diff = np.mean([attr[reg][f"12k|P|{r}"]["signed_excess"] - attr[reg][f"12k|A|{r}"]["signed_excess"] for r in range(R)])
        attr[reg]["_rendered_vs_lane_P_minus_A"] = {"rendered": float(rend_diff), "lane": float(lane_diff)}
    # ---- direction features and binned attribution (objects and every region), 12k start-mean and per start
    mrad = CM.mirror_radius(L["wi"], L["wo"])
    th_v, th_l = np.degrees(np.arccos(np.clip(L["wi"][:, 2], -1, 1))), np.degrees(np.arccos(np.clip(L["wo"][:, 2], -1, 1)))
    cref = 0.5 * (c_ref_a + c_ref_b)
    bins = {}
    for reg, sel in lane_region.items():
        if sel.sum() == 0:
            continue
        b_ = np.digitize(mrad[sel], MIRROR_BINS) - 1
        S = cref[sel].sum()
        row_ = {"ref_share": np.bincount(b_, cref[sel], len(MIRROR_BINS) - 1).tolist()}
        for tag in ("12k", "ext"):
            for arm in CM.ARMS:
                row_[f"{tag}|{arm}"] = [(np.bincount(b_, c_mod[f"{tag}|{arm}|{r}"][sel] - cref[sel], len(MIRROR_BINS) - 1) / S).tolist() for r in range(R)]
        row_["ref_share"] = (np.array(row_["ref_share"]) / S).tolist()
        wq = lambda x: [float(np.interp(q, np.cumsum(cref[sel][np.argsort(x[sel])]) / S, np.sort(x[sel]))) for q in (0.05, 0.5, 0.95)]
        row_["view_polar_q05_50_95"] = wq(th_v)
        row_["light_polar_q05_50_95"] = wq(th_l)
        row_["mirror_offset_q05_50_95"] = wq(mrad)
        probes = np.array(CM.ANGLES)
        near = (np.abs(th_v[sel][:, None] - probes[None]).min(1) <= 2.5) | (np.abs(th_l[sel][:, None] - probes[None]).min(1) <= 2.5)
        row_["share_within_2.5deg_of_a_probe_polar"] = float(cref[sel][near].sum() / S)
        row_["share_mirror_offset_lt_4deg"] = float(cref[sel][mrad[sel] < 4].sum() / S)
        bins[reg] = row_
    # ---- attribution label for hard_1 highlight core (and the same quantities everywhere)
    def label(reg):
        if reg not in attr:
            return None
        ex = [attr[reg][f"12k|P|{r}"] for r in range(R)]
        pos = sum(e["signed_excess"] - 1.96 * e["se"] > 0 for e in ex)
        sgn = sum(e["signed_excess"] > 0 for e in ex)
        inc0 = sum(abs(e["signed_excess"]) <= 1.96 * e["se"] for e in ex)
        rv = attr[reg]["_rendered_vs_lane_P_minus_A"]
        explains = rv["rendered"] != 0 and rv["lane"] / rv["rendered"] >= 0.8
        if pos >= 3 and explains:
            lab = "ANGULAR_RESPONSE_OVERSHOOT"
        elif inc0 >= 2:
            lab = "NOISE_COMPATIBLE"
        elif max(sgn, R - sgn) < 3:
            lab = "START_DEPENDENT"
        else:
            lab = "MIXED"
        return {"label": lab, "starts_positive_lb": int(pos), "starts_positive": int(sgn), "explains_rendered_P_minus_A": bool(explains)}
    labels = {reg: label(reg) for reg in attr}
    # ---- per-pixel maps for figures (12k start 0 and start-mean, ext start-mean)
    maps = {"ref": pixmap(cref), "ref_noise": pixmap(np.abs(c_ref_a - c_ref_b) / 2)}
    for tag in ("12k", "ext"):
        for arm in CM.ARMS:
            maps[f"{tag}_{arm}_mean"] = np.mean([pixmap(c_mod[f"{tag}|{arm}|{r}"]) for r in range(R)], 0)
            for r in range(R):
                maps[f"{tag}_{arm}_{r}"] = pixmap(c_mod[f"{tag}|{arm}|{r}"])
    for arm in CM.ARMS:
        maps[f"rendered_{arm}_mean"] = np.mean([fxm[f"{arm}_{r}|e1"] for r in range(R)], 0)
    maps["rendered_reference_er"] = ref_er_m
    np.savez_compressed(out / "maps.npz", **{k: v.astype(np.float32) for k, v in maps.items()},
                        **{f"mask_{k}": v for k, v in masks.items()})
    np.savez_compressed(out / "lanes.npz", pix=L["pix"], wi=L["wi"].astype(np.float32), wo=L["wo"].astype(np.float32),
                        lf=L["lf"].astype(np.float32), ref_a=a.astype(np.float32), ref_b=b.astype(np.float32),
                        **{k.replace("|", "_"): v.astype(np.float32) for k, v in c_mod.items()})
    res = {"scene": scene, "material": row["tag"], "lanes": nl, "camera_samples_per_pixel": K_SPP, "walks_per_lane": WALKS,
           "walk_seconds": walk_s, "correspondence": corr, "correspondence_passed": bool(corr_ok), "attribution": attr,
           "labels": labels, "mirror_bins_deg": MIRROR_BINS.tolist(), "binned": bins, "commit": CM.commit(), "inputs_sha256": seen,
           "seconds": time.monotonic() - t0}
    KA.write(out / "attribution.json", res)
    print("lanes", scene, nl, "corr", corr_ok, {k: v["label"] for k, v in labels.items() if v}, flush=True)
    if not corr_ok:
        raise IO.Stop(f"LANE_CORRESPONDENCE {scene}")


if __name__ == "__main__":
    main()
