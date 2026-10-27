"""WL53 oracle K / C runs and compositions (protocol phys_ref53). Mitsuba cuda_ad_rgb is used ONLY as a ray intersector.

    P53_MI_VARIANT=cuda_ad_rgb python p53_oracle.py <stage> <out dir> [smoke]
stages: tr (TR cell runs), g1a_iw, g2 (isolated slab: TR sun / binned K / IW predictions + operator properties),
        compose (every composition arm), ops (rigid / non-rigid operator differences), all
"""
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("P53_MI_VARIANT", "cuda_ad_rgb")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import p53_compose as CP  # noqa: E402
import p53_fixtures as FX  # noqa: E402
import p53_geom as G  # noqa: E402
import p53_mi as M  # noqa: E402
import p53_tracer as TR  # noqa: E402

DEV = "cuda"
if os.environ.get("P53_GPU_FRAC"):                 # shared GPU (LabServer69 with WL52): cap this process's torch memory
    torch.cuda.set_per_process_memory_fraction(float(os.environ["P53_GPU_FRAC"]))
CFG = dict(n_per_hi=32768, n_per=2048, n_per_alpha1=256, n_per_pair=1024, n_sun=16_000_000, M_rays=128, K_ext=256, iw_N=16_000_000, iw_N_h=8_000_000)
SMOKE = dict(n_per_hi=64, n_per=32, n_per_alpha1=16, n_per_pair=16, n_sun=400_000, M_rays=16, K_ext=32, iw_N=400_000, iw_N_h=200_000)
TR_RUNS = {   # name: (geometry, seed, cam keys | None, continue_air, n_per key)
    "slab_s1": ("slab", 5311, ("top", "bot", "otop", "obot"), False, "n_per"),
    "slab_s2": ("slab", 5312, ("top", "bot", "otop", "obot"), False, "n_per"),
    "slab_alpha1": ("slab", 5313, None, False, "n_per_alpha1"),
    "lblock_s1": ("lblock", 5311, "lblock_reuse", False, "n_per"),
    "lblock_s2": ("lblock", 5312, "lblock_reuse", False, "n_per"),
    "lblock_pose1_s3": ("lblock_pose1", 5313, ("top", "bot", "otop", "ocorner"), False, "n_per"),
    "lblock_far_s3": ("lblock_far", 5314, None, False, "n_per"),
    "bend30_s3": ("bend30", 5313, ("top", "bot", "otop"), False, "n_per"),
    "bend90_s3": ("bend90", 5313, ("top", "bot", "otop"), False, "n_per"),
    "dent_s3": ("dent", 5313, ("top", "bot", "otop"), False, "n_per"),
    "pair4_obj": ("pair4", 5315, None, True, "n_per_pair"),
    "pair8_obj": ("pair8", 5316, None, True, "n_per_pair"),
    "pair4_dup": ("pair4", 5317, None, "dup", "n_per_pair"),
}
TR_RUNS_HI = {   # amendment 1: sample size only (rules unchanged); T only, no camera operators
    "slab_hi_s1": ("slab", 5321, None, False, "n_per_hi"),
    "slab_hi_s2": ("slab", 5322, None, False, "n_per_hi"),
    "bend30_hi": ("bend30", 5323, None, False, "n_per_hi"),
    "bend90_hi": ("bend90", 5323, None, False, "n_per_hi"),
    "dent_hi": ("dent", 5323, None, False, "n_per_hi"),
}
LB_CAMS = ("top", "bot", "otop", "ocorner")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cam_dirs(keys):
    if keys == "lblock_reuse":                    # pose-0 run queried at the pose-1 cameras' LOCAL directions too
        d = [FX.DIRS[k] for k in LB_CAMS]
        return np.array(d + [FX.R_POSE1.T @ np.asarray(x) for x in d])
    return None if keys is None else np.array([FX.DIRS[k] for k in keys])


def make_domain(geo, alb=None):
    m, V = FX.domain_geometry(geo)
    return TR.Domain(m, V, FX.SIGMA_T, FX.ALBEDO if alb is None else alb, FX.G_PHASE, name=geo)


def stage_tr_hi(out, cfg, only=None):
    stage_tr(out, cfg, only, runs=TR_RUNS_HI)


def stage_tr(out, cfg, only=None, runs=None):
    (out / "T").mkdir(exist_ok=True)
    for name, (geo, seed, cams, cair, npk) in (runs or TR_RUNS).items():
        if only and name not in only:
            continue
        if (out / "T" / f"{name}.pt").exists():
            log("skip existing", name); continue
        dom = make_domain(geo, (1.0, 1.0, 1.0) if name == "slab_alpha1" else None)
        r = TR.run_cells(dom, FX.DOUT, FX.DIN, cfg[npk], seed, cams=cam_dirs(cams), continue_air=cair, log=log)
        torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in r.items()} | dict(geo=geo, seed=seed, cams=None if cams is None else cam_dirs(cams).tolist()),
                   out / "T" / f"{name}.pt")
        log(name, "done", f"{r['seconds']:.0f}s", json.dumps(r["stats"]))
        del r; torch.cuda.empty_cache()


def load_T(out, name):
    r = torch.load(out / "T" / f"{name}.pt")
    return {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in r.items()}


# ------------------------------------------------------------------ G1a independent walk


def stage_g1a_iw(out, cfg):
    res = []
    for alb in (0.5, 0.9, 0.99, 0.999):
        for th0 in (0.0, 60.0):
            s = (math.sin(math.radians(th0)), 0.0, -math.cos(math.radians(th0)))
            cams = [(0.0, math.sin(math.radians(th)), math.cos(math.radians(th))) for th in (0.0, 40.0, 70.0)]
            r = TR.iw_box((-1, -1, -0.5), (1, 1, 0.0), FX.SIGMA_T, alb, 0.0, s, 1.0, (-1, 1, -1, 1), cfg["iw_N_h"], cams, 5320, port_h=0.05,
                          count_region=(-0.8, 0.8, -0.8, 0.8))
            for k, th in enumerate((0.0, 40.0, 70.0)):
                exp = G.semi_infinite_radiance(alb, math.cos(math.radians(th)), math.cos(math.radians(th0)), 1.0)
                res.append(dict(albedo=alb, theta0=th0, theta=th, iw=float(r["center_radiance_top"][k, 0]), analytic=exp,
                                plane_albedo_iw=float(r["R"][0]), plane_albedo_analytic=G.plane_albedo(alb, math.cos(math.radians(th0)))))
            log("g1a_iw", alb, th0, [round(x["iw"] / x["analytic"], 4) for x in res[-3:]], r["R"][0], res[-1]["plane_albedo_analytic"])
    json.dump(res, open(out / "g1a_iw.json", "w"), indent=1)


# ------------------------------------------------------------------ G2 isolated slab


def stage_g2(out, cfg):
    dom = make_domain("slab"); cams = cam_dirs(("top", "bot", "otop", "obot"))
    T1 = load_T(out, "slab_s1"); res = {}
    for scene in ("iso_L1", "iso_L2", "iso_L3"):
        sd = FX.scene_def(scene); s, E = sd["lights"][0]
        mask = sd.get("sun_mask", {}).get("slab")
        r = TR.run_sun(dom, s, E, cfg["n_sun"], 5330, FX.DOUT, cams=cams, tri_mask=mask)
        L_exact = r["KC"][:, :, :, 0].permute(0, 2, 1).cpu().numpy()                  # (cam, P, 3)
        # binned input: E cos A into (port, bin(-s)) of every lit triangle (mask), K_cells + KC_cells
        ss = CP.SceneSet([CP.Comp("slab", "medium", dom.mesh, dom.Vcur)], FX.DOUT, FX.DIN)
        Ep, phi_ext = CP.sun_external(ss, s, E, cfg["K_ext"], 5331)
        if mask is not None:
            cen = (dom.mesh["port_center"]); keep = torch.tensor((np.abs(cen[:, 0]) < 0.01) & (np.abs(cen[:, 1]) < 0.01) & (dom.mesh["port_normal"][:, 2] > 0.5), device=DEV)
            phi_ext = phi_ext.view(dom.P, FX.DIN) * keep[:, None].float(); phi_ext = phi_ext.reshape(-1)
        phi3 = phi_ext[:, None].expand(-1, 3).contiguous()
        L_bin_nee = torch.stack([torch.stack([T1["KC"][c, k] @ phi3[:, k] for k in range(3)], 1) for c in range(len(cams))]).cpu().numpy()
        phi_out = torch.stack([T1["T"][k] @ phi3[:, k] for k in range(3)], 1)
        sol = dict(phi_out=phi_out, phi_in_ind=torch.zeros_like(phi3))
        L_bin_out = np.stack([CP.readout(ss, c, sol, None, mode="bin").cpu().numpy() for c in cams])
        res[scene] = dict(L_exact=L_exact, L_bin_nee=L_bin_nee, L_bin_out=L_bin_out, flux_out_exact=r["T"].sum(1)[:, 0].cpu().numpy(), stats=r["stats"])
        # IW
        region = (-0.01, 0.01, -0.01, 0.01) if scene == "iso_L3" else (-0.05, 0.05, -0.05, 0.05)
        iw = TR.iw_box((-0.05, -0.05, -0.005), (0.05, 0.05, 0.005), FX.SIGMA_T, FX.ALBEDO, FX.G_PHASE, s, E, region, cfg["iw_N"], cams, 5332, port_h=0.01)
        Liw = np.full((len(cams), dom.P, 3), np.nan)
        pc = dom.mesh["port_center"]; pn = dom.mesh["port_normal"]
        for f, sgn in (("top", 1), ("bot", -1)):
            sel = np.nonzero(pn[:, 2] == sgn)[0]; ix = np.floor((pc[sel, 0] + 0.05) / 0.01).astype(int); iy = np.floor((pc[sel, 1] + 0.05) / 0.01).astype(int)
            Liw[:, sel] = iw["radiance"][f][:, ix, iy]
        res[scene]["L_iw"] = Liw; res[scene]["iw_flux"] = dict(R=iw["R"].tolist(), T=iw["T"].tolist(), side=iw["side"].tolist())
        log("g2", scene, "TR exact top centre", L_exact[0, 45], "IW", Liw[0, 45], "bin", L_bin_nee[0, 45])
    np.savez(out / "g2_pred.npz", **{f"{sc}__{k}": v for sc, d in res.items() for k, v in d.items() if isinstance(v, np.ndarray)})
    json.dump({sc: dict(stats=d["stats"], iw_flux=d["iw_flux"], flux_out_exact=d["flux_out_exact"].tolist()) for sc, d in res.items()}, open(out / "g2_meta.json", "w"), indent=1)
    # ---------------- operator properties
    props = {}
    T2 = load_T(out, "slab_s2"); Ta = load_T(out, "slab_alpha1")
    for tag, R in (("s1", T1), ("s2", T2), ("alpha1", Ta)):
        cs = R["T"].sum(1)                                                              # (3, nsrc) column sums
        props[tag] = dict(colsum_max=cs.max(1).values.tolist(), colsum_min=cs.min(1).values.tolist(), colsum_mean=cs.mean(1).tolist(), stats=R["stats"])
    # reciprocity at port-pair level: A_j T_ij  vs  A_i T_ji   (T_ij = flux from port j to port i, all bins)
    A = dom.port_area.double()
    rec = {}
    for tag, R in (("s1", T1), ("s2", T2)):
        Tp = R["T"].view(3, dom.P, FX.DOUT, dom.P, FX.DIN).sum((2, 4)).double()    # (3, P_out, P_in)
        V2 = R["S2"].double()                                                            # sum of squared exit weights = variance estimate of Tp
        X = Tp * A[None, None, :]; Y = X.transpose(1, 2)
        sx = torch.sqrt(V2) * A[None, None, :]; sy = sx.transpose(1, 2)
        z = (X - Y) / torch.sqrt(sx ** 2 + sy ** 2).clamp(min=1e-30)
        good = (sx < 0.05 * X) & (sy < 0.05 * Y) & (X > 0)
        iu = torch.triu_indices(dom.P, dom.P, 1, device=DEV)
        rec[tag] = dict(pooled_asym=[float((X[k] - Y[k]).abs()[iu[0], iu[1]][good[k][iu[0], iu[1]]].sum() / X[k][iu[0], iu[1]][good[k][iu[0], iu[1]]].sum()) for k in range(3)],
                        frac_absz_gt4=[float((z[k][iu[0], iu[1]][good[k][iu[0], iu[1]]].abs() > 4).float().mean()) for k in range(3)],
                        n_pairs=[int(good[k][iu[0], iu[1]].sum()) for k in range(3)])
        # direction-resolved reciprocity on adjacent top-face pairs (16 x 16 bins, out bins aggregated 64 -> 16)
        par = torch.tensor(G.bin_parent(8, 4), device=DEV)
        T16 = torch.zeros(3, dom.P, FX.DIN, dom.P * FX.DIN, device=DEV).index_add_(2, par, R["T"].view(3, dom.P, FX.DOUT, dom.P * FX.DIN))
        T5 = T16.view(3, dom.P, FX.DIN, dom.P, FX.DIN).permute(1, 3, 0, 2, 4)            # (P_out, P_in, 3, out16, in16)
        top = np.nonzero(dom.mesh["port_normal"][:, 2] > 0.5)[0]; pc = dom.mesh["port_center"]
        pairs = [(i, j) for i in top for j in top if 0 < np.linalg.norm(pc[i] - pc[j]) < 0.0101]
        I = torch.tensor([p[0] for p in pairs], device=DEV); J = torch.tensor([p[1] for p in pairs], device=DEV)
        Xd = T5[I, J] * A[J].float()[:, None, None, None]                                # X[(i,o) <- (j,b)] A_j
        Yd = T5[J, I].transpose(2, 3) * A[I].float()[:, None, None, None]                # Y[(j,b) <- (i,o)] A_i, as [o, b]
        rec[tag]["dir_asym_adjacent"] = [float((Xd[:, k] - Yd[:, k]).abs().sum() / Xd[:, k].sum()) for k in range(3)]
        rec[tag]["_Xd"] = Xd
    # noise scale of the same statistic: |X(s1) - X(s2)| (two independent estimates of the same entries)
    for tag in ("s1", "s2"):
        rec[tag]["dir_noise_adjacent"] = [float((rec["s1"]["_Xd"][:, k] - rec["s2"]["_Xd"][:, k]).abs().sum() / rec["s1"]["_Xd"][:, k].sum()) for k in range(3)]
    for tag in ("s1", "s2"):
        del rec[tag]["_Xd"]
    props["reciprocity"] = rec
    # seed-noise reference for the direction-resolved asymmetry: same statistic between s1 and s2 (no physics difference)
    json.dump(props, open(out / "g2_props.json", "w"), indent=1)
    log("g2 props", json.dumps(props)[:600])


# ------------------------------------------------------------------ compositions


def scene_set(sd, override=None):
    comps = []
    for n, k, m, V in (override or sd["comps"]):
        comps.append(CP.Comp(n, k, m, V, rho=FX.FLOOR_RHO if k == "lambert" else None))
    return CP.SceneSet(comps, FX.DOUT, FX.DIN)


def compose_arm(sd, ss, Tm, KCm, cams_keys, cfg, kc_index, continue_air_sun=False, Cmask=None, order0=False, sun_doms=None, tag=""):
    """sun_doms: comp index -> TR.Domain used for the exact first-order term (domain geometry at its scene pose)."""
    s, E = sd["lights"][0]
    C, esc, cst = CP.cast_C(ss, cfg["M_rays"], 5340, mask=Cmask)
    Ep, _ = CP.sun_external(ss, s, E, cfg["K_ext"], 5341)
    phi0 = torch.zeros(ss.Ptot * ss.Dout, 3, device=DEV); L0 = {}
    dirs = np.array([FX.DIRS[k] for k in cams_keys])
    for ci, comp in enumerate(ss.comps):
        sl = ss.comp_ports(ci)
        if comp.kind == "medium":
            dom = sun_doms[ci]
            r = TR.run_sun(dom, s, E, cfg["n_sun"], 5342 + ci, FX.DOUT, cams=None if continue_air_sun else dirs, shadow=ss.isect)
            phi0[sl.start * ss.Dout:sl.stop * ss.Dout] = r["T"][:, :, 0].T
            if not continue_air_sun:
                L0[ci] = r["KC"][:, :, :, 0].permute(0, 2, 1)                                      # (ncam, P, 3)
        elif comp.kind == "lambert":
            rho = torch.tensor(comp.rho, dtype=torch.float32, device=DEV)
            phi0[sl.start * ss.Dout:sl.stop * ss.Dout] = ((Ep[sl] * ss.port_area[sl])[:, None] * rho / ss.Dout).repeat_interleave(ss.Dout, 0)
    if order0:
        C = torch.sparse_coo_tensor(torch.zeros(2, 0, dtype=torch.long, device=DEV), torch.zeros(0, device=DEV), C.shape)
    sol = CP.solve(ss, Tm, C, phi0)
    rho = CP.spectral_radius(ss, Tm, C) if not order0 else [0, 0, 0]
    pred = {}
    for k, key in enumerate(cams_keys):
        d = FX.DIRS[key]
        pred[f"{key}__bin"] = CP.readout(ss, d, sol, None, E_ext=Ep, mode="bin").cpu().numpy()
        if KCm is not None and not continue_air_sun:
            L0k = {ci: L0[ci][k] for ci in L0}; KCk = {ci: KCm[ci][kc_index[key]] for ci in KCm}
            pred[f"{key}__nee"] = CP.readout(ss, d, sol, L0k, KC=KCk, E_ext=Ep, mode="nee").cpu().numpy()
    meta = dict(rho=rho, iters=sol["iters"], converged=bool(sol["converged"]), last_rel=sol["last_rel"], C_stats=cst,
                escape_mean=float(esc.mean()), hist=sol["hist"][:20])
    log("compose", tag, json.dumps({k: v for k, v in meta.items() if k != "hist"}))
    return pred, meta


def stage_compose(out, cfg, only=None):
    (out / "pred").mkdir(exist_ok=True)
    S1 = load_T(out, "slab_s1"); S2 = load_T(out, "slab_s2")
    slab_kc = {k: i for i, k in enumerate(("top", "bot", "otop", "obot"))}
    arms = []
    # gap scenes: partition with the slab oracle (both slabs) -- seeds 1 and 2 for oracle noise
    for scene in ("gap4", "gap8"):
        sd = FX.scene_def(scene); cams = ("top", "bot", "otop", "obot")
        for tag, R in (("part_s1", S1), ("part_s2", S2)):
            arms.append((scene, tag, sd, None, {0: R["T"], 1: R["T"]}, {0: R["KC"], 1: R["KC"]}, slab_kc, cams, {}))
    # floor
    sd = FX.scene_def("floor")
    for tag, R in (("part_s1", S1), ("part_s2", S2)):
        arms.append(("floor", tag, sd, None, {0: R["T"]}, {0: R["KC"]}, slab_kc, ("top", "otop", "bot"), {}))
    arms.append(("floor", "order0", sd, None, {0: S1["T"]}, {0: S1["KC"]}, slab_kc, ("top", "otop", "bot"), dict(order0=True)))
    for scene in ("gap4", "gap8"):
        sd = FX.scene_def(scene)
        arms.append((scene, "order0", sd, None, {0: S1["T"], 1: S1["T"]}, {0: S1["KC"], 1: S1["KC"]}, slab_kc, ("top", "bot", "otop", "obot"), dict(order0=True)))
    for a in arms:
        if only and a[0] not in only:
            continue
        scene, tag, sd, ovr, Tm, KCm, kci, cams, kw = a
        ss = scene_set(sd, ovr); doms = {ci: TR.Domain(c.mesh, c.Vcur, FX.SIGMA_T, FX.ALBEDO, FX.G_PHASE, intersector=None, name=c.name) for ci, c in enumerate(ss.comps) if c.kind == "medium"}
        pred, meta = compose_arm(sd, ss, Tm, KCm, cams, cfg, kci, sun_doms=doms, tag=f"{scene}/{tag}", **kw)
        np.savez(out / "pred" / f"{scene}__{tag}.npz", **pred); json.dump(meta, open(out / "pred" / f"{scene}__{tag}.json", "w"), indent=1)
        torch.cuda.empty_cache()
    del S2
    # object-scoped (pair) controls on gap4 / gap8 (bin readout only)
    for scene, pname, tag_suffix in (("gap4", "pair4_obj", "pair4"), ("gap8", "pair8_obj", "pair8"), ("gap8", "pair4_obj", "pair4stale"), ("gap4", "pair4_dup", "pair4dup")):
        if only and scene not in only:
            continue
        if not (out / "T" / f"{pname}.pt").exists():
            log("skip pair arm (no T)", pname); continue
        sd = FX.scene_def(scene); P = load_T(out, pname)
        m, V = FX.domain_geometry("pair4" if scene == "gap4" else "pair8")
        ovr = [("pair", "medium", m, V)]
        ss = scene_set(sd, ovr)
        dom = TR.Domain(m, V, FX.SIGMA_T, FX.ALBEDO, FX.G_PHASE, name="pair")
        mode = "dup" if tag_suffix == "pair4dup" else True
        for ctag, mask in (("full", None), ("masked", lambda a, b: ~((a == 0) & (b == 0)))):
            if tag_suffix in ("pair4stale", "pair4dup") and ctag == "masked":
                continue
            pred, meta = _pair_arm(sd, ss, P["T"], dom, cfg, mask, f"{scene}/{tag_suffix}_{ctag}", mode)
            np.savez(out / "pred" / f"{scene}__{tag_suffix}_{ctag}.npz", **pred); json.dump(meta, open(out / "pred" / f"{scene}__{tag_suffix}_{ctag}.json", "w"), indent=1)
        del P; torch.cuda.empty_cache()
    # L-block pose 0 / pose 1 (rigid reuse)
    L1 = load_T(out, "lblock_s1"); lb_kc = {k: i for i, k in enumerate(LB_CAMS)}; lb_kc_reuse = {k: i + 4 for i, k in enumerate(LB_CAMS)}
    cams = ("top", "otop", "ocorner", "bot")
    for scene, tag, Tsrc, kci in (("lblock", "part_s1", L1, lb_kc), ("lblock_pose1", "reuse_pose0", L1, lb_kc_reuse)):
        if only and scene not in only:
            continue
        sd = FX.scene_def(scene); ss = scene_set(sd); doms = {0: TR.Domain(ss.comps[0].mesh, ss.comps[0].Vcur, FX.SIGMA_T, FX.ALBEDO, FX.G_PHASE, name="dom")}
        pred, meta = compose_arm(sd, ss, {0: Tsrc["T"]}, {0: Tsrc["KC"]}, cams, cfg, kci, sun_doms=doms, tag=f"{scene}/{tag}")
        np.savez(out / "pred" / f"{scene}__{tag}.npz", **pred); json.dump(meta, open(out / "pred" / f"{scene}__{tag}.json", "w"), indent=1)
    del L1; torch.cuda.empty_cache()
    if not only or "lblock_pose1" in only:
        Lp = load_T(out, "lblock_pose1_s3"); sd = FX.scene_def("lblock_pose1"); ss = scene_set(sd)
        doms = {0: TR.Domain(ss.comps[0].mesh, ss.comps[0].Vcur, FX.SIGMA_T, FX.ALBEDO, FX.G_PHASE, name="dom")}
        pred, meta = compose_arm(sd, ss, {0: Lp["T"]}, {0: Lp["KC"]}, cams, cfg, {"top": 0, "bot": 1, "otop": 2, "ocorner": 3}, sun_doms=doms, tag="lblock_pose1/recomputed")
        np.savez(out / "pred" / "lblock_pose1__recomputed.npz", **pred); json.dump(meta, open(out / "pred" / "lblock_pose1__recomputed.json", "w"), indent=1)
        del Lp; torch.cuda.empty_cache()
    # non-rigid: current K (nee + bin) and stale rest K (bin)
    for scene in ("bend30", "bend90", "dent"):
        if only and scene not in only:
            continue
        sd = FX.scene_def(scene); ss = scene_set(sd); cams = ("top", "otop", "bot")
        doms = {0: TR.Domain(ss.comps[0].mesh, ss.comps[0].Vcur, FX.SIGMA_T, FX.ALBEDO, FX.G_PHASE, name="dom")}
        Rc = load_T(out, f"{scene}_s3")
        for tag, Tsrc, KC in (("current", Rc["T"], Rc["KC"]), ("stale_rest", S1["T"], None)):
            pred, meta = compose_arm(sd, ss, {0: Tsrc}, None if KC is None else {0: KC}, cams, cfg, {"top": 0, "bot": 1, "otop": 2}, sun_doms=doms, tag=f"{scene}/{tag}")
            np.savez(out / "pred" / f"{scene}__{tag}.npz", **pred); json.dump(meta, open(out / "pred" / f"{scene}__{tag}.json", "w"), indent=1)
        del Rc; torch.cuda.empty_cache()


def _pair_arm(sd, ss, Tpair, dom, cfg, mask, tag, mode=True):
    s, E = sd["lights"][0]
    C, esc, cst = CP.cast_C(ss, cfg["M_rays"], 5340, mask=mask)
    r = TR.run_sun(dom, s, E, cfg["n_sun"], 5342, FX.DOUT, shadow=ss.isect, continue_air=mode)
    phi0 = r["T"][:, :, 0].T.contiguous()
    sol = CP.solve(ss, {0: Tpair}, C, phi0); rho = CP.spectral_radius(ss, {0: Tpair}, C)
    pred = {f"{k}__bin": CP.readout(ss, FX.DIRS[k], sol, None, mode="bin").cpu().numpy() for k in ("top", "bot", "otop", "obot")}
    meta = dict(rho=rho, iters=sol["iters"], converged=bool(sol["converged"]), C_stats=cst)
    log("compose", tag, json.dumps(meta))
    return pred, meta


# ------------------------------------------------------------------ operator differences (rigid / non-rigid)


def port_pair(R, P):
    return R["T"].view(3, P, FX.DOUT, P, FX.DIN).sum((2, 4)).double()


def stage_ops_hi(out, cfg):
    stage_ops(out, cfg, hi=True)


def stage_ops(out, cfg, hi=False):
    res = {}
    m = FX.slab_mesh(); P = len(m["port_area"])
    S1, S2 = (load_T(out, "slab_hi_s1"), load_T(out, "slab_hi_s2")) if hi else (load_T(out, "slab_s1"), load_T(out, "slab_s2"))
    A1, A2 = port_pair(S1, P), port_pair(S2, P); noise = A2 - A1
    var = S1["S2"].double()                                                                        # entry variance estimate (3, P, P)
    pc = m["port_center"]; dist = torch.tensor(np.linalg.norm(pc[:, None] - pc[None], axis=2), device=DEV)
    face = m["port_face"]; same_face = torch.tensor(face[:, None] == face[None], device=DEV)
    bins = [(0, 1e-9), (1e-9, 0.0151), (0.0151, 0.0301), (0.0301, 0.0501), (0.0501, 1.0)]

    def summarize(D, Abase, var_d, area_new=None, area_old=None):
        out_ = {}
        tot = (D ** 2).sum((1, 2)); out_["fro_rel"] = (D.norm(dim=(1, 2)) / Abase.norm(dim=(1, 2))).tolist()
        out_["fro_vs_noise"] = (D.norm(dim=(1, 2)) / noise.norm(dim=(1, 2))).tolist()
        r_ = D.norm(dim=(1, 2)) / noise.norm(dim=(1, 2))                                  # E||D||^2 = ||signal||^2 + ||noise||^2
        out_["signal_vs_noise_debiased"] = torch.sqrt((r_ ** 2 - 1).clamp(min=0)).tolist()
        z = D / torch.sqrt(var_d).clamp(min=1e-30); sig = z.abs() > 4
        out_["frac_entries_sig"] = [float(sig[k][Abase[k] > 0].float().mean()) for k in range(3)]
        out_["energy_by_distance"] = {f"{lo:.3f}-{hi:.3f}": [float((D[k] ** 2)[(dist >= lo) & (dist < hi)].sum() / tot[k]) for k in range(3)] for lo, hi in bins}
        out_["relchange_by_distance"] = {f"{lo:.3f}-{hi:.3f}": [float(D[k][(dist >= lo) & (dist < hi)].abs().sum() / Abase[k][(dist >= lo) & (dist < hi)].abs().sum().clamp(min=1e-30)) for k in range(3)] for lo, hi in bins}
        out_["energy_same_face"] = [float((D[k] ** 2)[same_face].sum() / tot[k]) for k in range(3)]
        cs_old = Abase.sum(1); cs_new = (Abase + D).sum(1)
        out_["colsum_change_mean"] = ((cs_new - cs_old) / cs_old).mean(1).tolist(); out_["colsum_change_absmax"] = ((cs_new - cs_old) / cs_old).abs().max(1).values.tolist()
        sv = [torch.linalg.svdvals(D[k]) for k in range(3)]; svn = [torch.linalg.svdvals(noise[k]) for k in range(3)]
        out_["svd_top10"] = [s[:10].tolist() for s in sv]; out_["noise_svd_top"] = [float(s[0]) for s in svn]
        out_["rank_above_noise"] = [int((sv[k] > svn[k][0]).sum()) for k in range(3)]
        e = [torch.cumsum(s ** 2, 0) / (s ** 2).sum() for s in sv]
        out_["rank90"] = [int((ek < 0.9).sum()) + 1 for ek in e]
        if area_new is not None:
            Dar = Abase * ((area_new / area_old)[None, :, None] - 1)                     # area-only correspondence model on the receiving port
            out_["area_model_explained"] = [float(1 - (D[k] - Dar[k]).norm() ** 2 / D[k].norm() ** 2) for k in range(3)]
        return out_

    res["seed_noise"] = dict(fro_rel=(noise.norm(dim=(1, 2)) / A1.norm(dim=(1, 2))).tolist(),
                             frac_entries_sig=[float(((noise[k] / torch.sqrt(2 * var[k]).clamp(min=1e-30)).abs() > 4)[A1[k] > 0].float().mean()) for k in range(3)])
    rig = None
    # rigid: L-block pose 0 (s1, s2) vs pose 1 (s3) and far (s3)
    lm = FX.lblock_mesh(); PL = len(lm["port_area"])
    L1, L2 = load_T(out, "lblock_s1"), load_T(out, "lblock_s2"); B1, B2 = port_pair(L1, PL), port_pair(L2, PL)
    rig = dict(seed_fro=(B2 - B1).norm(dim=(1, 2)).tolist())
    for nm in (() if hi else ("lblock_pose1_s3", "lblock_far_s3")):
        Bp = port_pair(load_T(out, nm), PL); D = Bp - B1
        rig[nm] = dict(fro_vs_seed=(D.norm(dim=(1, 2)) / (B2 - B1).norm(dim=(1, 2))).tolist(), fro_rel=(D.norm(dim=(1, 2)) / B1.norm(dim=(1, 2))).tolist(),
                       mean_rel_diff=((Bp.sum((1, 2)) - B1.sum((1, 2))) / B1.sum((1, 2))).tolist())
        # direction-resolved: full T matrices (bins in material frames)
        Tfull = load_T(out, nm)["T"]
        rig[nm]["fro_full_vs_seed"] = ((Tfull - L1["T"]).norm(dim=(1, 2)) / (L2["T"] - L1["T"]).norm(dim=(1, 2))).tolist()
    if not hi:
        res["rigid"] = rig
    del L1, L2
    # non-rigid
    for nm in (("bend30_hi", "bend90_hi", "dent_hi") if hi else ("bend30_s3", "bend90_s3", "dent_s3")):
        R = load_T(out, nm); geo = nm.split("_")[0]; _, Vc = FX.domain_geometry(geo)
        fr = G.tri_frames(m["V"], Vc, m["F"], m["tri_port"], m["port_tangent"])
        an = torch.tensor(G.port_areas(fr["area"], m["tri_port"], P), device=DEV); ao = torch.tensor(m["port_area"], device=DEV)
        D = port_pair(R, P) - A1
        res[nm] = summarize(D, A1, 2 * var, an, ao)
        res[nm]["area_ratio_range"] = [float((an / ao).min()), float((an / ao).max())]
        if geo == "dent":
            r = np.hypot(pc[:, 0], pc[:, 1]); top = m["port_normal"][:, 2] > 0.5
            for k, Ld in enumerate((0.0181, 0.0058, 0.0018)):
                sup = torch.tensor(r < 0.02 + 3 * Ld, device=DEV)
                both = sup[:, None] & sup[None, :]
                res[nm].setdefault("energy_within_support_3Ld", []).append(float((D[k] ** 2)[both].sum() / (D[k] ** 2).sum()))
            res[nm]["support_def"] = "port centres with r < 2 cm + 3 L_d(channel), L_d = 1/sqrt(3 sigma_a sigma_t) = 18.1 / 5.8 / 1.8 mm"
        # visibility change (C): same-domain re-entry fraction of uniform Lambertian exitance
        ss = CP.SceneSet([CP.Comp("d", "medium", m, Vc)], FX.DOUT, FX.DIN)
        C, esc, _ = CP.cast_C(ss, cfg["M_rays"], 5350)
        res[nm]["self_reentry_fraction_uniform"] = float(1 - esc.mean())
        np.save(out / f"ops_D_{nm}.npy", D.float().cpu().numpy())
        log("ops", nm, json.dumps({k: res[nm][k] for k in ("fro_rel", "fro_vs_noise", "rank90", "colsum_change_mean", "self_reentry_fraction_uniform")}))
    ss = CP.SceneSet([CP.Comp("d", "medium", m, m["V"])], FX.DOUT, FX.DIN); C, esc, _ = CP.cast_C(ss, cfg["M_rays"], 5350)
    res["slab_rest_self_reentry_fraction_uniform"] = float(1 - esc.mean())
    sfx = "_hi" if hi else ""
    np.save(out / f"ops{sfx}_noise.npy", noise.float().cpu().numpy()); np.save(out / f"ops{sfx}_A1.npy", A1.float().cpu().numpy())
    json.dump(res, open(out / f"ops{sfx}.json", "w"), indent=1)


def main():
    stage, out = sys.argv[1], Path(sys.argv[2]); smoke = "smoke" in sys.argv[3:]
    cfg = SMOKE if smoke else CFG; out.mkdir(parents=True, exist_ok=True)
    only = [a.split("=", 1)[1].split(",") for a in sys.argv[3:] if a.startswith("only=")]
    only = only[0] if only else None
    t0 = time.time()
    for st in ([stage] if stage != "all" else ["tr", "g1a_iw", "g2", "compose", "ops"]):
        fn = globals()[f"stage_{st}"]
        fn(out, cfg, only) if st in ("tr", "tr_hi", "compose") else fn(out, cfg)
    json.dump(dict(cfg=cfg, variant=M.VARIANT, seconds=time.time() - t0, smoke=smoke, torch=torch.__version__,
                   peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9), open(out / f"oracle_meta_{stage}.json", "w"))
    log("done", stage, time.time() - t0)
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
