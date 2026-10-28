"""WL54 Stage E (medium K, coarse K interface): coupled evaluation of C0 vs CASP on the WL53 scenes with the stored WL53 oracle K
(T matrices, 1 cm ports, 8x8 out / 4x4 in bins) and the stored Cycles VOLUME references (protocol fnto_c54).

    P53_MI_VARIANT=cuda_ad_rgb python c54_stageE.py <out dir> <wl53 results> <scene,...>
Interface (conservative mapping, K unchanged): medium domains keep their coarse port unknowns (K_coarse = WL53 TR operator): the cells that
refinement would create on a medium port collapse by SUM on the receiving side, and its emission stays uniform per (port, out-bin).
CASP therefore refines the Lambert domains (floor) only; a medium-to-medium link is unchanged in expectation (predeclared dependency).
Arms: C0stored (WL53 realization, read from the stored predictions), C1 = CASP (c = 2, l_max = 8, equal ray budget, seeds 5810 / 5811),
C1_rep (3 replicates), C0rebuilt (this module's caster at all-ones levels; a statistical equivalence check of the caster), Cfine
(uniform s = 4 floor). Readout: nee (primary) for the sun-lit scenes, bin for the stale-vs-current K comparison (stale K has no KC).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phys_ref53"))
import c54_casp as CA  # noqa: E402
import c54_common as CC  # noqa: E402
import c54_stageB as SB  # noqa: E402
import p53_compose as CP  # noqa: E402
import p53_fixtures as FX  # noqa: E402
import p53_geom as G  # noqa: E402
import p53_oracle as OR  # noqa: E402
import p53_report as RP  # noqa: E402
import p53_tracer as TR  # noqa: E402

DEV = "cuda"
LB_CAMS = ("top", "bot", "otop", "ocorner")
SCENES = {   # scene: (T name current/partition, cams for the compose, kc index, readout modes)
    "gap4": dict(T="slab_s1", cams=("top", "bot", "otop", "obot"), kci={k: i for i, k in enumerate(("top", "bot", "otop", "obot"))}, modes=("nee",)),
    "gap8": dict(T="slab_s1", cams=("top", "bot", "otop", "obot"), kci={k: i for i, k in enumerate(("top", "bot", "otop", "obot"))}, modes=("nee",)),
    "floor": dict(T="slab_s1", cams=("top", "otop", "bot"), kci={k: i for i, k in enumerate(("top", "bot", "otop", "obot"))}, modes=("nee",)),
    "lblock": dict(T="lblock_s1", cams=("top", "otop", "ocorner", "bot"), kci={k: i for i, k in enumerate(LB_CAMS)}, modes=("nee",)),
    "bend90": dict(T="bend90_s3", cams=("top", "otop", "bot"), kci={"top": 0, "bot": 1, "otop": 2}, modes=("nee", "bin"), stale="slab_s1"),
    "dent": dict(T="dent_s3", cams=("top", "otop", "bot"), kci={"top": 0, "bot": 1, "otop": 2}, modes=("nee", "bin"), stale="slab_s1"),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def comps_with_levels(sd, levels_by_comp, s_uniform=None):
    """Override list for OR.scene_set: Lambert comps refined (CASP levels or uniform s), medium comps coarse. Returns (override, parents list)."""
    ov, parents, off = [], [], 0
    for ci, (n, k, m, V) in enumerate(sd["comps"]):
        P = len(m["port_area"])
        if k == "lambert" and (levels_by_comp is not None or s_uniform):
            lev = levels_by_comp[ci] if s_uniform is None else np.full(P, s_uniform)
            fm, VC, par = CA.refine_mesh(m, V, lev); ov.append((n, k, fm, VC)); parents.append(par + off)
        else:
            ov.append((n, k, m, V)); parents.append(np.arange(P) + off)
        off += P
    return ov, parents


def prepare_sun(sd, ss0, cams_keys):
    """Exact first-order sun term of every medium domain (deterministic, seeds as WL53) -- independent of C, shared by all arms."""
    s, E = sd["lights"][0]; dirs = np.array([FX.DIRS[k] for k in cams_keys]); cache = {}
    for ci, comp in enumerate(ss0.comps):
        if comp.kind != "medium":
            continue
        dom = TR.Domain(comp.mesh, comp.Vcur, FX.SIGMA_T, FX.ALBEDO, FX.G_PHASE, name=comp.name)
        r = TR.run_sun(dom, s, E, OR.CFG["n_sun"], 5342 + ci, FX.DOUT, cams=dirs, shadow=ss0.isect)
        cache[ci] = (r["T"][:, :, 0].T.contiguous(), r["KC"][:, :, :, 0].permute(0, 2, 1).contiguous()); del r, dom; torch.cuda.empty_cache()
    return cache


def run_arm(sd, ov, parents, sun, Tm, KCm, cams_keys, kci, modes, seeds, K_ext=256, label=""):
    ss = OR.scene_set(sd, ov); s, E = sd["lights"][0]; t0 = time.time()
    # ray budget: coarse / medium ports 128 per out-cell; refined cells 128 / l^2 (equal rays per coarse port)
    par_all = np.concatenate(parents); cells_per_port = np.bincount(par_all)[par_all]
    m_port = np.maximum(1, np.round(128 / cells_per_port)).astype(int)
    torch.cuda.synchronize(); tc = time.time()
    C, esc, st = CA.cast_adaptive(ss, m_port, seeds[0]); torch.cuda.synchronize(); t_C = time.time() - tc
    Ep, _ = CP.sun_external(ss, s, E, K_ext, seeds[1])
    phi0 = torch.zeros(ss.Ptot * ss.Dout, 3, device=DEV); L0 = {}
    for ci, comp in enumerate(ss.comps):
        sl = ss.comp_ports(ci)
        if comp.kind == "medium":
            phi0[sl.start * ss.Dout:sl.stop * ss.Dout] = sun[ci][0]; L0[ci] = sun[ci][1]
        else:
            rho = torch.tensor(comp.rho, dtype=torch.float32, device=DEV)
            phi0[sl.start * ss.Dout:sl.stop * ss.Dout] = ((Ep[sl] * ss.port_area[sl])[:, None] * rho / ss.Dout).repeat_interleave(ss.Dout, 0)
    sol = CP.solve(ss, Tm, C, phi0); rho_s = CP.spectral_radius(ss, Tm, C)
    # coarse-port arrays (area-weighted for refined Lambert comps)
    par = np.concatenate(parents); nc = int(par.max()) + 1; area = ss.port_area.double().cpu().numpy(); wsum = np.bincount(par, weights=area, minlength=nc)
    pred = {}
    for k, key in enumerate(cams_keys):
        d = FX.DIRS[key]
        for mode in modes:
            if mode == "nee":
                KCk = {ci: KCm[ci][kci[key]] for ci in KCm}; L0k = {ci: L0[ci][k] for ci in L0}
                L = CP.readout(ss, d, sol, L0k, KC=KCk, E_ext=Ep, mode="nee")
            else:
                L = CP.readout(ss, d, sol, None, E_ext=Ep, mode="bin")
            L = L.double().cpu().numpy(); pred[f"{key}__{mode}"] = np.stack([np.bincount(par, weights=area * L[:, c], minlength=nc) / wsum for c in range(3)], 1)
    meta = dict(label=label, cells=int(ss.Ptot), coarse_ports=nc, rho=rho_s, iters=sol["iters"], converged=bool(sol["converged"]), nnz=int(C._nnz()), t_C=t_C,
                wall=time.time() - t0, back_hits=st["back_hits"], gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    log("arm", scene_name_global, label, json.dumps(meta)); del C, ss, sol; torch.cuda.empty_cache()
    return pred, meta


scene_name_global = ""


def compare_rgb(ref_seeds, pred, cnt, groups, tol=0.03):
    """Per channel group stats; ref_seeds (S, P, 3), pred (P, 3)."""
    ok = cnt >= SB.MINPIX; out = {}
    for c in range(3):
        se = ref_seeds[:, :, c].std(0, ddof=1) / np.sqrt(len(ref_seeds)); out[c] = SB.group_stats(pred[:, c], ref_seeds[:, :, c].mean(0), ok, groups, se, tol)
    return out


def run_scene(scene, out, R, nrep=3):
    global scene_name_global
    scene_name_global = scene; cfg = SCENES[scene]; t_all = time.time()
    sd = FX.scene_def(scene); ss0 = OR.scene_set(sd); groups = RP.port_groups(scene); odir = R / "oracle"; cdir = R / "cycles"
    Tc = OR.load_T(odir, cfg["T"]); Tm = {ci: Tc["T"] for ci, c in enumerate(ss0.comps) if c.kind == "medium"}; KCm = {ci: Tc["KC"] for ci in Tm}
    sun = prepare_sun(sd, ss0, cfg["cams"]); res = dict(scene=scene, arms={}, cams={}); preds = {}
    lev_all, nu_all, _ = CA.assign_levels(ss0)                       # geometry-only levels on the coarse scene (prepass seed 5801)
    off = 0; levels_by_comp = []
    for ci, comp in enumerate(ss0.comps):
        P = comp.P; levels_by_comp.append(lev_all[off:off + P] if comp.kind == "lambert" else np.ones(P, int)); off += P
    near = nu_all >= CA.NU_MIN
    arms = {}
    ov, par = comps_with_levels(sd, None); arms["C0rebuilt"] = (ov, par, False, (5340, 5341))
    ov, par = comps_with_levels(sd, levels_by_comp); arms["C1"] = (ov, par, True, (5810, 5811))
    for r in range(nrep):
        arms[f"C1_r{r}"] = (ov, par, True, (5820 + r, 5830 + r))
    if any(c.kind == "lambert" for c in ss0.comps):
        ov4, par4 = comps_with_levels(sd, None, s_uniform=4); arms["Cfine_s4"] = (ov4, par4, True, (5840, 5841))
    for name, (ov, par, lp, seeds) in arms.items():
        preds[name], res["arms"][name] = run_arm(sd, ov, par, sun, Tm, KCm, cfg["cams"], cfg["kci"], cfg["modes"][:1], seeds, label=name)
    # stale (rest) K arms for the differential evaluation (bin readout only)
    if "stale" in cfg:
        S1 = OR.load_T(odir, cfg["stale"]); Ts = {ci: S1["T"] for ci in Tm}
        for name in ("C0rebuilt", "C1"):
            ov, par, lp, seeds = arms[name]
            pc, _ = run_arm(sd, ov, par, sun, Tm, KCm, cfg["cams"], cfg["kci"], ("bin",), seeds, label=name + "_cur_bin")
            pst, m = run_arm(sd, ov, par, sun, Ts, None, cfg["cams"], cfg["kci"], ("bin",), seeds, label=name + "_stale_bin")
            preds[name + "|cur_bin"], preds[name + "|stale_bin"] = pc, pst
        del S1
    # references: Cycles volume images, V1 (centre) primary and V0 (WL53 interior), per camera
    ss_c = ss0; geo = json.load(open(cdir / f"ccam_{scene}.json")); cyc = {}
    for nm, off_ in (("V1", SB.V1_OFF), ("V0", SB.V0_OFF)):
        cyc[nm] = {}
        for cam, g in geo.items():
            files = sorted(cdir.glob(f"cimg_{scene}_{cam}_*.npy")); pid = CP.classify(ss_c, CC.blender_ortho_rays(g, off_))
            means = [CP.port_means(pid, np.load(f).reshape(-1, 3), ss_c.Ptot) for f in files]
            cyc[nm][cam] = (np.stack([m[0] for m in means]), means[0][1])
    for cam in sd["cams"]:
        key = f"{RP.dirkey(sd['cams'][cam]['dir'])}__{cfg['modes'][0]}"; rc = {}
        for nm in ("V1", "V0"):
            ref, cnt = cyc[nm][cam]
            for arm in preds:
                if "|" in arm or key not in preds[arm]:
                    continue
                rc[f"{arm}_vs_cycles_{nm}"] = compare_rgb(ref, preds[arm][key], cnt, groups)
                rc[f"{arm}_vs_cycles_{nm}_nearfar"] = compare_rgb(ref, preds[arm][key], cnt, np.where(near, "near", "far"))
        # stored WL53 C0 realization (nee): reuse the stored prediction file
        P = np.load(odir / "pred" / f"{scene}__{'current' if scene in ('bend90', 'dent') else 'part_s1'}.npz")
        if key in P.files:
            rc["C0stored_vs_cycles_V1"] = compare_rgb(cyc["V1"][cam][0], P[key], cyc["V1"][cam][1], groups); rc["C0stored_vs_cycles_V0"] = compare_rgb(cyc["V0"][cam][0], P[key], cyc["V0"][cam][1], groups)
        for arm in preds:
            kb = f"{RP.dirkey(sd['cams'][cam]['dir'])}__bin"
            if "|" in arm and kb in preds[arm]:
                rc[f"{arm}_vs_cycles_V1"] = compare_rgb(cyc["V1"][cam][0], preds[arm][kb], cyc["V1"][cam][1], groups)
        res["cams"][cam] = rc
    if "stale" in cfg:                                                  # paired differential D = stale - current (same C, reference cancels)
        res["differential"] = {}
        for name in ("C0rebuilt", "C1"):
            res["differential"][name] = {}
            for cam in sd["cams"]:
                key = f"{RP.dirkey(sd['cams'][cam]['dir'])}__bin"
                if key not in preds[name + "|cur_bin"]:
                    continue
                D = preds[name + "|stale_bin"][key] - preds[name + "|cur_bin"][key]; ref, cnt = cyc["V1"][cam]; ok = cnt >= SB.MINPIX
                res["differential"][name][cam] = {c: {g: dict(D_rel=float(D[ok & (groups == g), c].sum() / max(ref[:, :, c].mean(0)[ok & (groups == g)].sum(), 1e-30)),
                                                          signal=float(ref[:, :, c].mean(0)[ok & (groups == g)].sum()))
                                                      for g in np.unique(groups[ok])} for c in range(3)}
    np.savez(out / f"{scene}_preds.npz", **{f"{a}|{k}": v for a, d in preds.items() for k, v in d.items()})
    res["wall_s"] = time.time() - t_all
    CC.dump(res, out / f"{scene}_stageE.json"); log("scene done", scene, f"{time.time() - t_all:.0f}s")


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); R = Path(sys.argv[2])
    for s in sys.argv[3].split(","):
        run_scene(s, out, R)


if __name__ == "__main__":
    main()
