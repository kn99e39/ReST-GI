"""WL55 Stage B analysis (protocol fnto_k55, stage_B.rules): group biases of every arm against the Cycles references (amended and
declared units), the decomposition T = B(arm vs Cycles), R = B(COR vs Cycles), I = B(arm vs COR), and the predeclared rules.

    P53_MI_VARIANT=cuda_ad_rgb python k55_analyze.py stageB <out json>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

import k55_common as KCm
import k55_fixtures as KF
import c54_fixtures as CF
import p53_geom as G

ROOT = Path(os.environ.get("WL55", Path.home() / "rest_wl55")); SB = ROOT / "out" / "stageB"; CYC = ROOT / "out" / os.environ.get("WL55_CYC", "cycles_cuda")
WL53 = Path.home() / "rest_wl53" / "results"; WL54 = Path.home() / "rest_wl54" / "out"
FULL = ("gap8", "gap4", "floor", "bend90")
PAIRS = {"P10": ("P10_s1", "P10_s2"), "P05": ("P05_s1", "P05_s2"), "P025": ("P025_s1", "P025_s2"), "P10d": ("P10d_s1", "P10d_s2"), "COR": ("COR_5571", "COR_5572")}
SINGLE = ("P10_c2", "P05d_s1", "P10_din64", "P10_wl53")
NEAR = {"gap8": ("slab_lo:top", "slab_hi:bot"), "gap4": ("slab_lo:top", "slab_hi:bot"), "floor": ("dom:bot", "floor"), "bend90": ("dom:bot", "floor")}


def cycles_of(scene):
    sd = KF.scene_def(scene); geo = json.load(open(CYC / f"ccam_{scene}.json")); ss = KCm.classify_sceneset(sd); out = {}
    for cam, g in geo.items():
        files = sorted(CYC.glob(f"cimg_{scene}_{cam}_s*.npy"))
        if files:
            out[cam] = KCm.cycles_ports(ss, g, [np.load(f) for f in files])
    return sd, out


def load_arm(scene, arm, key):
    """(nee, nee + ball) predictions (P, 3) for camera direction key, or None."""
    d = SB / scene
    if arm.startswith("COR_"):
        p = d / f"{arm}.npz"
        if not p.exists():
            return None
        z = np.load(p); keys = list(z["keys"]); k = keys.index(key)
        return z["L"][k], z["L"][k] + z["ball"][k]
    if arm == "P10_wl53":
        p = WL53 / "oracle" / "pred" / f"{scene}__{'current' if scene in ('bend90', 'dent') else 'part_s1'}.npz"
        z = np.load(p) if p.exists() else None
        if z is None or f"{key}__nee" not in z.files:
            return None
        return z[f"{key}__nee"], None
    p = d / f"{arm}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    if f"{key}|nee" not in z.files:
        return None
    return z[f"{key}|nee"], z[f"{key}|nee"] + z[f"{key}|ball"]


def gb(pred, ref, se, sel, lab, share_mask=None):
    return KCm.group_bias(pred, ref, sel, lab, se, share_mask=share_mask)


def pair_stats(scene, sd, cyc, lab, share_mask=None):
    """For every camera / unit / arm / readout: group bias vs Cycles (mean of the two seeds where they exist, SE_oracle from the
    seed difference) and vs the COR mean (interface I)."""
    res = {}
    for cam, units in cyc.items():
        key = KCm.dirkey(sd["cams"][cam]["dir"]); res[cam] = {}
        cor = [load_arm(scene, a, key) for a in PAIRS["COR"]]
        for unit in ("amended", "declared"):
            u = units[unit]; R = {}
            arms = {**{k: v for k, v in PAIRS.items()}, **{k: (k,) for k in SINGLE}}
            for name, members in arms.items():
                preds = [load_arm(scene, a, key) for a in members]
                if any(p is None for p in preds):
                    continue
                for ri, ro in enumerate(("nee", "ball")):
                    if any(p[ri] is None for p in preds):
                        continue
                    bs = [gb(p[ri], u["mean"], u["se"], u["sel"], lab, share_mask) for p in preds]
                    mean_pred = np.mean([p[ri] for p in preds], 0)
                    g = gb(mean_pred, u["mean"], u["se"], u["sel"], lab, share_mask)
                    for gn in g:
                        if len(bs) == 2 and gn in bs[0] and gn in bs[1]:
                            so = [abs(a - b) / 2 for a, b in zip(bs[0][gn]["bias"], bs[1][gn]["bias"])]
                            g[gn]["se_oracle"] = so; g[gn]["se_total"] = [float(np.hypot(a, b)) for a, b in zip(g[gn]["se"], so)]
                            g[gn]["bias_seeds"] = [bs[0][gn]["bias"], bs[1][gn]["bias"]]
                        else:
                            g[gn]["se_total"] = g[gn]["se"]
                        g[gn]["passed_total"] = [bool(abs(b) <= KCm.TOL + 3 * s) if r else None for b, s, r in zip(g[gn]["bias"], g[gn]["se_total"], g[gn]["ruled"])]
                    R[f"{name}|{ro}|cyc"] = g
                    if all(c is not None for c in cor) and name != "COR":
                        cm = np.mean([c[ri] for c in cor], 0); cse = np.abs(cor[0][ri] - cor[1][ri]) / 2
                        R[f"{name}|{ro}|cor"] = gb(mean_pred, cm, cse, u["sel"], lab, share_mask)
            res[cam][unit] = R
    return res


def bias(R, key, gn, c):
    try:
        return R[key][gn]["bias"][c]
    except KeyError:
        return None


def rules_full(scene, S):
    """B-E, B-R, B-I per ruled group (amended unit)."""
    out = []
    for cam, units in S.items():
        R = units["amended"]
        if "P10|nee|cyc" not in R:
            continue
        for gn, v in R["P10|nee|cyc"].items():
            for c in range(3):
                if not v["ruled"][c]:
                    continue
                Tn, Tb = bias(R, "P10|nee|cyc", gn, c), bias(R, "P10|ball|cyc", gn, c)
                Rn, Rb = bias(R, "COR|nee|cyc", gn, c), bias(R, "COR|ball|cyc", gn, c)
                rec = dict(scene=scene, cam=cam, group=gn, ch=c, T_nee=Tn, T_ball=Tb, R_nee=Rn, R_ball=Rb, se_total=R["P10|nee|cyc"][gn].get("se_total", [0, 0, 0])[c])
                rec["E_reduction"] = None if Tb is None or not Tn else 1 - abs(Tb) / abs(Tn)
                rec["E_reduction_COR"] = None if Rb is None or not Rn else 1 - abs(Rb) / abs(Rn)
                I = {a: bias(R, f"{a}|ball|cor", gn, c) for a in ("P10", "P05", "P025", "P10d", "P10_c2", "P10_din64")}
                Is = R.get("P10|ball|cyc", {}).get(gn, {}).get("bias_seeds")
                rec.update({f"I_{a}": v_ for a, v_ in I.items()})
                T = Tb if Tb is not None else Tn; Rr = Rb if Rb is not None else Rn
                se = rec["se_total"] or 0.0
                rec["B_R"] = bool(Rr is not None and abs(Rr) > max(0.01, 3 * se) and abs(Rr) >= 0.5 * abs(T))
                i10, i05, i025 = I["P10"], I["P05"], I["P025"]
                rec["B_I"] = bool(i10 is not None and abs(i10) >= max(0.01, 0.5 * abs(T)))
                rec["spatial"] = bool(rec["B_I"] and i05 is not None and i025 is not None and abs(i10) > abs(i05) > abs(i025) and abs(i025) <= 0.5 * abs(i10))
                rec["directional"] = bool(rec["B_I"] and I["P10_din64"] is not None and I["P10d"] is not None and abs(I["P10_din64"] - I["P10d"]) >= abs(I["P10d"]) / 3)
                rec["oracle_noise"] = bool(rec["B_I"] and Is is not None and abs(Is[0][c] - Is[1][c]) >= abs(i10) / 2)
                rec["C_noise"] = bool(rec["B_I"] and I["P10_c2"] is not None and abs(I["P10_c2"] - i10) >= abs(i10) / 2)
                rec["near_contact"] = gn in NEAR.get(scene, ())
                out.append(rec)
    return out


def stage_full():
    res = {}
    for scene in FULL:
        if not (CYC / f"ccam_{scene}.json").exists():
            continue
        sd, cyc = cycles_of(scene); lab = KCm.labels_of(sd); S = pair_stats(scene, sd, cyc, lab)
        res[scene] = dict(stats=S, rules=rules_full(scene, S))
    return res


def stage_kf():
    res = {}
    for g in ("gap8", "gap4"):
        for f in ("U", "X", "S"):
            scene = f"kf_{g}_{f}"
            if not (CYC / f"ccam_{scene}.json").exists():
                continue
            sd, cyc = cycles_of(scene); lab = KCm.labels_of(sd); med = KF.medium_ports(sd); rec = {}
            for cam, units in cyc.items():
                key = KCm.dirkey(sd["cams"][cam]["dir"]); u = units["amended"]; R = {}
                cor = [load_arm(scene, f"COR_{s}", key) for s in (5573, 5574)]
                if any(c is None for c in cor):
                    continue
                for ri, ro in enumerate(("nee", "ball")):
                    cm = np.mean([c[ri] for c in cor], 0); cse = np.abs(cor[0][ri] - cor[1][ri]) / 2
                    R[f"COR|{ro}|cyc"] = gb(cm, u["mean"], u["se"], u["sel"], lab, med)
                    for arm in ("K10_s1", "K10_s2", "K05_s1", "K05_s2", "K025_s1", "K025_s2", "K10d_s1"):
                        p = load_arm(scene, arm, key)
                        if p is None:
                            continue
                        R[f"{arm}|{ro}|cyc"] = gb(p[ri], u["mean"], u["se"], u["sel"], lab, med)
                        R[f"{arm}|{ro}|cor"] = gb(p[ri], cm, cse, u["sel"], lab, med)
                if f == "S" and (SB / scene / "sun.pt").exists():
                    import torch
                    sun = torch.load(SB / scene / "sun.pt"); L0 = sun[0]["L0"][sun[0]["cam_keys"].index(key)].numpy()
                    full = np.zeros((len(lab), 3)); full[:len(L0)] = L0
                    R["L0exact|nee|cyc"] = gb(full, u["mean"], u["se"], u["sel"], lab, med)
                    cm = np.mean([c[1] for c in cor], 0); cse = np.abs(cor[0][1] - cor[1][1]) / 2
                    R["L0exact|nee|cor"] = gb(full, cm, cse, u["sel"], lab, med)
                rec[cam] = R
            res[scene] = rec
    return res


def kf_rules(K):
    out = []
    for scene, cams in K.items():
        if scene.endswith("_S"):
            continue
        for cam, R in cams.items():
            if "K10_s1|ball|cor" not in R:
                continue
            for gn, v in R["COR|ball|cyc"].items():
                for c in range(3):
                    if not v["ruled"][c]:
                        continue
                    a = {s: np.mean([R[f"{s}_s{i}|ball|cor"][gn]["bias"][c] for i in (1, 2) if f"{s}_s{i}|ball|cor" in R and gn in R[f"{s}_s{i}|ball|cor"]]) for s in ("K10", "K05", "K025")}
                    se = R["K10_s1|ball|cor"][gn]["se"][c]
                    so = abs(R["K10_s1|ball|cor"][gn]["bias"][c] - R["K10_s2|ball|cor"][gn]["bias"][c]) / 2 if "K10_s2|ball|cor" in R else 0
                    st = float(np.hypot(se, so))
                    out.append(dict(scene=scene, cam=cam, group=gn, ch=c, A10=a["K10"], A05=a["K05"], A025=a["K025"], se=st, R_COR=v["bias"][c],
                                    A_loss=bool(abs(a["K10"]) >= 0.01 and abs(a["K10"]) > 3 * st and abs(a["K025"]) <= 0.5 * abs(a["K10"]))))
    return out


def btest():
    res = {}
    for g in ("gap8", "gap4"):
        scene = f"kf_{g}_S"; p = SB / scene / "btest.npz"
        if not p.exists():
            continue
        z = np.load(p); sd = KF.scene_def(scene); n0 = len(sd["comps"][0][2]["F"]); m, V = sd["comps"][1][2], sd["comps"][1][3]
        fr = G.tri_frames(m["V"], V, m["F"], m["tri_port"], m["port_tangent"]); cen = fr["centroid"]; nt = len(m["F"])
        cors = [np.load(SB / scene / f"COR_{s}.npz")["arr_tri"][:, n0:n0 + nt] for s in (5573, 5574) if (SB / scene / f"COR_{s}.npz").exists()]
        if not cors:
            continue
        cell_c = z["L4|cen"]; par = z["L4|parent"]
        tri_cell = np.argmin(((cen[:, None, :] - cell_c[None]) ** 2).sum(-1), 1)
        cor_cell = np.stack([np.stack([np.bincount(tri_cell, weights=a[c], minlength=len(cell_c)) for c in range(3)], 1) for a in cors])
        cm = cor_cell.mean(0); cse = np.abs(cor_cell[0] - cor_cell[-1]) / 2 if len(cors) > 1 else np.zeros_like(cm); rec = {}
        for lev in (1, 2, 4):
            a = z[f"L{lev}|arr"]
            pool = (a.sum(0) - cm.sum(0)) / cm.sum(0)
            ap = np.stack([np.bincount(par, weights=a[:, c]) for c in range(3)], 1); cp = np.stack([np.bincount(par, weights=cm[:, c]) for c in range(3)], 1)
            sig = cp[:, 0] > 0.01 * cp[:, 0].max()
            rel_p = np.abs(ap - cp)[sig] / cp[sig]; sigc = cm[:, 0] > 0.01 * cm[:, 0].max(); rel_c = np.abs(a - cm)[sigc] / cm[sigc]
            rec[f"L{lev}"] = dict(pooled=pool.tolist(), port_p90=np.percentile(rel_p, 90, 0).tolist(), port_p50=np.percentile(rel_p, 50, 0).tolist(),
                                  cell_p90=np.percentile(rel_c, 90, 0).tolist(), n_ports=int(sig.sum()), n_cells=int(sigc.sum()))
        rec["cor_seed_rel_noise_pooled"] = (np.abs(cor_cell[0].sum(0) - cor_cell[-1].sum(0)) / cm.sum(0) / 2).tolist()
        rec["B_loss"] = [bool((abs(rec["L1"]["pooled"][c]) >= 0.01 or rec["L1"]["port_p90"][c] >= 0.05) and
                              max(abs(rec["L4"]["pooled"][c]), 0) <= 0.5 * max(abs(rec["L1"]["pooled"][c]), 1e-9) and rec["L4"]["port_p90"][c] <= 0.5 * rec["L1"]["port_p90"][c]) for c in range(3)]
        res[g] = rec
    return res


def cor_valid():
    out = {}
    # (ii) floor_lam: COR vs WL54 F16 (Lambert, grey -> channel 0) on the amended ruled groups of WL54's 0.5 mm renders
    p = SB / "floor_lam" / "COR_5571.npz"
    if p.exists():
        z = np.load(p); keys = list(z["keys"]); arr = np.load(WL54 / "stageB" / "floor_lam_coarse_arrays.npz")["F16"]
        sd = CF.scene_def("floor_lam"); import c54_common as CC
        ss = CC.sceneset_of(sd); geo = json.load(open(WL54 / "cycles" / "ccam_floor_lam_L.json")); lab = CF.coarse_labels("floor_lam"); rec = {}
        for cam, g in geo.items():
            u = KCm.cycles_ports(ss, g, [np.load(f)[..., :3] for f in sorted((WL54 / "cycles").glob(f"cimg_floor_lam_L_{cam}_*.npy"))])["amended"]
            key = KCm.dirkey(sd["cams"][cam]["dir"]); Lc = z["L"][keys.index(key)][:, 0]
            gg = KCm.group_bias(Lc, arr, u["sel"], lab)
            rec[cam] = {gn: dict(bias=v["bias"][0], ruled=v["ruled"][0]) for gn, v in gg.items()}
        out["floor_lam_COR_vs_F16"] = rec
        out["floor_lam_pass"] = bool(all(abs(v["bias"]) <= 0.01 for r in rec.values() for v in r.values() if v["ruled"]))
    return out


def main():
    res = dict(cor_valid=cor_valid(), full=stage_full(), kf=stage_kf(), btest=btest())
    res["kf_rules"] = kf_rules(res["kf"])
    # (iii) COR vs Cycles on directly lit faces
    lit = []
    for scene, d in res["full"].items():
        for cam, units in d["stats"].items():
            R = units["amended"].get("COR|ball|cyc", {})
            for gn, v in R.items():
                if (cam == "top" and gn.endswith(":top")) or (cam == "obotx" and gn.endswith(":-x")):
                    lit.append(dict(scene=scene, cam=cam, group=gn, bias=v["bias"], ruled=v["ruled"]))
    res["cor_valid"]["lit_faces"] = lit
    res["cor_valid"]["lit_pass"] = bool(all(abs(b) <= 0.01 for x in lit for b, r in zip(x["bias"], x["ruled"]) if r))
    # gate C
    rules = [r for d in res["full"].values() for r in d["rules"]]
    sp = {}
    for r in rules:
        if r["near_contact"] and r["spatial"]:
            sp.setdefault((r["scene"], r["cam"], r["group"]), set()).add(r["ch"])
    spatial_groups = [k for k, v in sp.items() if len(v) >= 2]
    a_loss = any(r["A_loss"] for r in res["kf_rules"]); b_loss = any(any(v["B_loss"]) for v in res["btest"].values())
    res["gate_C"] = dict(spatial_groups=[list(k) for k in spatial_groups], A_loss=a_loss, B_loss=b_loss, demonstrated=bool(spatial_groups and (a_loss or b_loss)))
    KCm.dump(res, Path(sys.argv[2])); print(json.dumps(res["gate_C"], indent=1)); print(json.dumps(res["cor_valid"].get("floor_lam_pass")), res["cor_valid"]["lit_pass"])


if __name__ == "__main__":
    main()
