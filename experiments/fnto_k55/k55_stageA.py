"""WL55 Stage A: independent audit of WL54 (protocol fnto_k55, stage_A). Replays use k55_common.group_bias (new code), WL54's stored
arrays (results/fnto_c54 == ~/rest_wl54/out) and the stored Cycles images; reported values are read from WL54's own JSON outputs.

    P53_MI_VARIANT=cuda_ad_rgb python k55_stageA.py replay <out json>
    P53_MI_VARIANT=cuda_ad_rgb python k55_stageA.py code <out json>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

import k55_common as KCm
import c54_casp as CA
import c54_common as CC
import c54_fixtures as CF
import c54_lam as LM
import p53_compose as CP
import p53_fixtures as FX
import p53_oracle as OR
import p53_report as RP

H54 = Path.home() / "rest_wl54" / "out"; H53 = Path.home() / "rest_wl53" / "results"
LAM = ("gap4", "gap8", "gap30", "floor_lam", "lblock_lam", "rot_lam")


def lam_ports(name, cdir, geo_file, pattern, cams=None):
    sd = CF.scene_def(name); ss = CC.sceneset_of(sd); geo = json.load(open(cdir / geo_file)); out = {}
    for key, g in geo.items():
        cam = key.split("@")[0]
        if cams and cam not in cams:
            continue
        files = sorted(cdir.glob(pattern.format(name=name, cam=cam, key=key, px=key.split("@")[-1])))
        if not files:
            continue
        imgs = [np.load(f)[..., :3] for f in files]
        out[key] = KCm.cycles_ports(ss, g, imgs)
    return out


def replay(outp):
    rep = dict(lambert_05={}, lambert_025={}, stageE={}, kfine={}, gap8_lowerside={})
    A1 = json.load(open(H54 / "amend1_tables.json")); A3 = json.load(open(H54 / "amend3_lam.json"))
    diffs = []
    # (a) 0.5 mm amended Lambert table (C0prod, F16, C1), red channel (grey scenes)
    for name in LAM:
        arr = dict(np.load(H54 / "stageB" / f"{name}_coarse_arrays.npz")); arr.update({k: v for k, v in np.load(H54 / "stageC" / f"{name}_casp_arrays.npz").items() if k.startswith("C1")})
        lab = CF.coarse_labels(name); P = lam_ports(name, H54 / "cycles", f"ccam_{name}_L.json", "cimg_{name}_L_{cam}_*.npy"); rep["lambert_05"][name] = {}
        for cam, u in P.items():
            a = u["amended"]; rec = {}
            for arm in ("C0prod", "F16", "C1"):
                gb = KCm.group_bias(arr[arm], a["mean"][:, 0], a["sel"], lab, a["se"][:, 0])
                rec[arm] = {g: dict(bias=v["bias"][0], se=v["se"][0], ruled=v["ruled"][0], n=v["n"]) for g, v in gb.items()}
                try:
                    reported = A1[name][cam]["amended"][arm]
                    for g, v in rec[arm].items():
                        if g in reported:
                            diffs.append(dict(table="lambert_05", scene=name, cam=cam, arm=arm, group=g, replay=v["bias"], reported=reported[g]["bias"], d_pp=100 * abs(v["bias"] - reported[g]["bias"])))
                except KeyError:
                    pass
            rep["lambert_05"][name][cam] = rec
    # ruled-group count and CASP failures (rule 3 % + 3 SE) on the 0.5 mm amended table
    ruled = [(n, c, g, v) for n, d in rep["lambert_05"].items() for c, r in d.items() for g, v in r["C1"].items() if v["ruled"]]
    rep["lambert_05_summary"] = dict(n_ruled=len(ruled), C1_fail=[(n, c, g, round(100 * v["bias"], 2)) for n, c, g, v in ruled if abs(v["bias"]) > 0.03 + 3 * v["se"]],
                                     C0_fail=[(n, c, g, round(100 * rep["lambert_05"][n][c]["C0prod"][g]["bias"], 2)) for n, c, g, v in ruled
                                              if abs(rep["lambert_05"][n][c]["C0prod"][g]["bias"]) > 0.03 + 3 * rep["lambert_05"][n][c]["C0prod"][g]["se"]])
    # (b) 0.25 mm addendum
    for name in LAM:
        if not (H54 / "cycles_hi" / f"ccam_{name}.json").exists():
            continue
        arr = dict(np.load(H54 / "stageB" / f"{name}_coarse_arrays.npz")); arr.update({k: v for k, v in np.load(H54 / "stageC" / f"{name}_casp_arrays.npz").items() if k.startswith("C1")})
        lab = CF.coarse_labels(name); P = lam_ports(name, H54 / "cycles_hi", f"ccam_{name}.json", "cimg_{name}_{cam}_px{px}_s*.npy"); rep["lambert_025"][name] = {}
        for key, u in P.items():
            a = u["amended"]; rec = {}
            for arm in ("C0prod", "F16", "C1", "C1_c4"):
                gb = KCm.group_bias(arr[arm], a["mean"][:, 0], a["sel"], lab, a["se"][:, 0])
                rec[arm] = {g: dict(bias=v["bias"][0], ruled=v["ruled"][0], n=v["n"]) for g, v in gb.items()}
                cam = key.split("@")[0]
                rp = A3.get(name, {}).get(key, {}).get("groups", {}).get(arm, {})
                for g, v in rec[arm].items():
                    if g in rp:
                        diffs.append(dict(table="lambert_025", scene=name, cam=key, arm=arm, group=g, replay=v["bias"], reported=rp[g]["bias"], d_pp=100 * abs(v["bias"] - rp[g]["bias"])))
            rep["lambert_025"][name][key] = rec
    # (c) Stage E medium: CASP (C1) vs C0rebuilt vs stored WL53, against the WL53 Cycles volume renders (1 mm), both units
    for scene in ("gap4", "gap8"):
        sd = FX.scene_def(scene); ss = OR.scene_set(sd); lab = RP.port_groups(scene); P = np.load(H54 / "stageE" / f"{scene}_preds.npz")
        W = np.load(H53 / "oracle" / "pred" / f"{scene}__part_s1.npz"); geo = json.load(open(H53 / "cycles" / f"ccam_{scene}.json")); rep["stageE"][scene] = {}
        maxd = 0.0
        for cam, g in geo.items():
            imgs = [np.load(f) for f in sorted((H53 / "cycles").glob(f"cimg_{scene}_{cam}_*.npy"))]; u = KCm.cycles_ports(ss, g, imgs)["amended"]
            key = f"{RP.dirkey(sd['cams'][cam]['dir'])}__nee"; rec = {}
            for arm, pr in (("C1", P[f"C1|{key}"]), ("C0rebuilt", P[f"C0rebuilt|{key}"]), ("C0stored", W[key])):
                rec[arm] = KCm.group_bias(pr, u["mean"], u["sel"], lab, u["se"])
            for gname in rec["C1"]:
                if gname in rec["C0rebuilt"]:
                    maxd = max(maxd, 100 * max(abs(a - b) for a, b in zip(rec["C1"][gname]["bias"], rec["C0rebuilt"][gname]["bias"])))
            rep["stageE"][scene][cam] = {a: {gg: dict(bias=v["bias"], ruled=v["ruled"], n=v["n"]) for gg, v in r.items()} for a, r in rec.items()}
        rep["stageE"][scene]["max_abs_C1_minus_C0rebuilt_pp"] = maxd
    # (d) matched-resolution K control (WL54 kfine), amended unit, against the WL53 renders
    KF54 = json.load(open(H54 / "kfine" / "kfine_control.json"))
    for scene in ("gap4", "gap8"):
        sd = FX.scene_def(scene); ss = OR.scene_set(sd); lab = RP.port_groups(scene); P = np.load(H54 / "kfine" / f"{scene}_kfine_preds.npz")
        geo = json.load(open(H53 / "cycles" / f"ccam_{scene}.json")); rep["kfine"][scene] = {}
        for cam, g in geo.items():
            imgs = [np.load(f) for f in sorted((H53 / "cycles").glob(f"cimg_{scene}_{cam}_*.npy"))]; u = KCm.cycles_ports(ss, g, imgs)["amended"]
            key = f"{RP.dirkey(sd['cams'][cam]['dir'])}__nee"; rec = {}
            for s in (1, 2):
                gb = KCm.group_bias(P[f"K{s}|{key}"], u["mean"], u["sel"], lab, u["se"]); rec[f"K{s}"] = {gg: dict(bias=v["bias"], n=v["n"]) for gg, v in gb.items()}
                rp = KF54[scene]["cams"][cam].get(f"K{s}_amended", {})
                for gg, v in rec[f"K{s}"].items():
                    for c in range(3):
                        if str(c) in rp and gg in rp[str(c)]:
                            diffs.append(dict(table="kfine", scene=scene, cam=cam, arm=f"K{s}", group=f"{gg}/{c}", replay=v["bias"][c], reported=rp[str(c)][gg]["bias"], d_pp=100 * abs(v["bias"][c] - rp[str(c)][gg]["bias"])))
            rep["kfine"][scene][cam] = rec
    # (e) gap8 obot slab_lo:-y at 1 / 0.5 / 0.25 mm
    scene = "gap8"; sd = FX.scene_def(scene); ss = OR.scene_set(sd); lab = RP.port_groups(scene); pred = np.load(H53 / "oracle" / "pred" / f"{scene}__part_s1.npz")["obot__nee"]
    A3v = json.load(open(H54 / "amend3_vol.json")); g1 = json.load(open(H53 / "cycles" / f"ccam_{scene}.json"))["obot"]; gv = json.load(open(H54 / "cycles_hi_vol" / f"ccam_{scene}.json"))
    for tag, g, files in (("1mm_wl53", g1, sorted((H53 / "cycles").glob("cimg_gap8_obot_*.npy"))), ("0.5mm", gv["obot@0.5"], [H54 / "cycles_hi_vol" / "cimg_gap8_obot_px0.5_s5499.npy"]),
                          ("0.25mm", gv["obot@0.25"], [H54 / "cycles_hi_vol" / "cimg_gap8_obot_px0.25_s5499.npy"])):
        u = KCm.cycles_ports(ss, g, [np.load(f) for f in files])["amended"]; gb = KCm.group_bias(pred, u["mean"], u["sel"], lab, u["se"])
        v = gb.get("slab_lo:-y"); rep["gap8_lowerside"][tag] = v
        if v:
            for c in range(3):
                diffs.append(dict(table="gap8_lowerside", scene=tag, cam="obot", arm="wl53", group=f"slab_lo:-y/{c}", replay=v["bias"][c], reported=A3v[tag]["slab_lo:-y"]["wl53_pred_bias"][c],
                                  d_pp=100 * abs(v["bias"][c] - A3v[tag]["slab_lo:-y"]["wl53_pred_bias"][c])))
    rep["diffs"] = diffs
    rep["summary"] = {t: dict(n=len([d for d in diffs if d["table"] == t]), max_d_pp=max([d["d_pp"] for d in diffs if d["table"] == t] or [None]),
                              n_over_0p1=len([d for d in diffs if d["table"] == t and d["d_pp"] > 0.1])) for t in ("lambert_05", "lambert_025", "kfine", "gap8_lowerside")}
    KCm.dump(rep, outp); print(json.dumps(rep["summary"], indent=1)); print(json.dumps(rep["lambert_05_summary"], indent=1))
    print("stageE max |C1-C0rebuilt| pp:", {s: rep["stageE"][s]["max_abs_C1_minus_C0rebuilt_pp"] for s in ("gap4", "gap8")})


def code(outp):
    res = {}
    # refine_mesh area conservation on deformed meshes (rest and current)
    import p53_geom as G
    for geo in ("bend90", "dent"):
        m, V = FX.domain_geometry(geo); lev = np.random.default_rng(0).choice([1, 2, 4, 8], len(m["port_area"]))
        fm, VC, par = CA.refine_mesh(m, V, lev)
        fr0 = G.tri_frames(m["V"], V, m["F"], m["tri_port"], m["port_tangent"]); a0 = G.port_areas(fr0["area"], m["tri_port"], len(m["port_area"]))
        fr1 = G.tri_frames(fm["V"], VC, fm["F"], fm["tri_port"], fm["port_tangent"]); a1 = np.bincount(par, weights=G.port_areas(fr1["area"], fm["tri_port"], len(par)))
        res[f"area_{geo}"] = float(np.max(np.abs(a1 - a0) / a0))
    # C closure and padded-port sampling on a CASP Lambert scene with mixed levels (gap4)
    lev, nu, d = CA.levels_for("gap4"); lam = CA.LamCASP("gap4", lev).build()
    C = lam.C; col = torch.zeros(lam.ss.Ptot, device="cuda", dtype=torch.float64).index_add_(0, C._indices()[1], C._values().double())
    tot = col + lam.esc.double(); res["closure_casp_gap4"] = dict(min=float(tot.min()), max=float(tot.max()), back_hits=int(lam.stats["back_hits"]))
    gen = torch.Generator(device="cuda"); gen.manual_seed(1); ss = lam.ss; nbad = 0; N = 0
    for _ in range(20):
        ports = torch.randint(0, ss.Ptot, (4_000_000,), device="cuda", generator=gen)
        u = torch.rand(len(ports), device="cuda", generator=gen); k = (u[:, None] > ss.port_cum[ports]).sum(1).clamp(max=ss.port_tris.shape[1] - 1)
        nbad += int((ss.port_tris[ports, k] < 0).sum()); N += len(ports)
    res["sceneset_padded_sampling"] = dict(samples=N, padded_hits=nbad, rate=nbad / N, levels=np.unique(lev).tolist(), note="p53_compose.SceneSet.sample_ports on ports with unequal triangle counts")
    # A^2 spectral-radius estimator vs Arnoldi on the WL53 gap4 medium operator (coarse) and the floor_lam CASP operator
    import scipy.sparse.linalg as sla
    sd = FX.scene_def("gap4"); ss = OR.scene_set(sd); T = OR.load_T(H53 / "oracle", "slab_s1")["T"]; Tm = {0: T, 1: T}
    C, _, _ = CP.cast_C(ss, 128, 5340)
    for c in range(3):
        n = ss.Ptot * ss.Dout
        def mv(x, c=c):
            v = torch.zeros(n, 3, device="cuda"); v[:, c] = torch.tensor(x.real, dtype=torch.float32, device="cuda")
            return CP.apply_K(ss, Tm, CP.apply_C(C, v))[:, c].double().cpu().numpy()
        ev = sla.eigs(sla.LinearOperator((n, n), matvec=mv, dtype=np.float64), k=4, which="LM", return_eigenvectors=False, tol=1e-9)
        res.setdefault("rho_gap4", []).append(dict(arnoldi=float(np.max(np.abs(ev))), eig=[complex(e).__repr__() for e in ev]))
    rr = CA.rho_robust(ss, Tm, C); w53 = CP.spectral_radius(ss, Tm, C)
    for c in range(3):
        res["rho_gap4"][c].update(a2=rr[c], wl53=w53[c], rel_a2=abs(rr[c] - res["rho_gap4"][c]["arnoldi"]) / res["rho_gap4"][c]["arnoldi"])
    lev, nu, d = CA.levels_for("floor_lam"); lam = CA.LamCASP("floor_lam", lev, bins=(64, 16)).build(); Cl = lam.C; ssl = lam.ss
    n = ssl.Ptot * ssl.Dout
    def mvl(x):
        v = torch.zeros(n, 3, device="cuda"); v[:, 0] = torch.tensor(x.real, dtype=torch.float32, device="cuda")
        return CP.apply_K(ssl, {}, CP.apply_C(Cl, v))[:, 0].double().cpu().numpy()
    ev = sla.eigs(sla.LinearOperator((n, n), matvec=mvl, dtype=np.float64), k=4, which="LM", return_eigenvectors=False, tol=1e-9)
    rr = CA.rho_robust(ssl, {}, Cl); w53 = CP.spectral_radius(ssl, {}, Cl)
    res["rho_floor_lam_casp"] = dict(arnoldi=float(np.max(np.abs(ev))), a2=rr[0], wl53=w53[0], eig=[complex(e).__repr__() for e in ev])
    KCm.dump(res, outp); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"replay": replay, "code": code}[sys.argv[1]](Path(sys.argv[2]))
