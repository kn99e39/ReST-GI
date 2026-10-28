"""WL54 Stage B: isolate the spatial approximation error with exact local Lambert K (protocol fnto_c54).

    python c54_stageB.py <out dir> <cycles dir> <fixture,...> <levels,...> [key=value ...]     (levels: refinement s, e.g. 2,4,8)
Arms per fixture (all coarse numbers are per 1 cm WL53 port, radiance W m^-2 sr^-1):
  C0prod   production C0: coarse mesh, Dout 64 / Din 16, M = 128 rays per out-cell, K_ext 256, seeds 5340 / 5341     (the object under test)
  C0inf    same interface, Dout = Din = 1, M = 65536 (sampling-noise-free coarse C; binning is irrelevant for Lambert)
  F_s      fine solve with s x s cells per port (cell = one unknown), projected onto the coarse ports (area weighted)
  G_s      coarse solve with the Galerkin operator aggregated from F_s's C (exact coarse operator for piecewise-constant ports)
Decomposition (all vs the same port groups / cameras as WL53):
  A  representation   G_s   vs F_s   exact transfer integrals, coarse unknowns        (spatial discretization of the interface)
  B  sampled C        C0prod vs G_s  production sampler on that interface              (MC noise, directional bins)
  C  readout          F_s    vs Cycles   fine reference vs the independent renderer, centre-ray and WL53 interior comparison units
  T  total            C0prod vs Cycles   what WL53 measured
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import c54_common as CC
import c54_fixtures as CF
import c54_lam as LM
import p53_compose as CP

MINPIX, MINPORTS, MINSHARE = 30, 5, 0.02
V0_OFF = ((0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))
V1_OFF = ((0.5, 0.5),)
MS = {1: 4096, 2: 1024, 4: 512, 8: 256, 16: 128, 32: 64}               # rays per fine cell (declared in the protocol)
KEXT = {1: 256, 2: 256, 4: 256, 8: 128, 16: 64, 32: 32}


def cycles_ports(cdir, name, sd, ss, offsets):
    """Cycles port means per seed (S, P) and pixel counts (P) for one comparison unit; per camera."""
    geo = json.load(open(cdir / f"ccam_{name}_L.json")); out = {}
    for cam, g in geo.items():
        files = sorted(cdir.glob(f"cimg_{name}_L_{cam}_*.npy"))
        pid = CP.classify(ss, CC.blender_ortho_rays(g, offsets))
        means = [CP.port_means(pid, np.load(f).reshape(-1, 3)[:, :1], ss.Ptot) for f in files]
        out[cam] = (np.stack([m[0][:, 0] for m in means]), means[0][1])
    return geo, out


def group_stats(pred, ref, ok, groups, se=None, tol=0.03):
    """Pooled bias per port group + distribution of per-port relative differences; ruled as in WL53 (>= 5 ports, >= 2 % of signal)."""
    tot = ref[ok].sum(); res = {}
    for g in np.unique(groups[ok]):
        m = ok & (groups == g); s = ref[m].sum(); d = pred[m] - ref[m]
        if s <= 0:
            continue
        B = float(d.sum() / s); rel = d / np.maximum(ref[m], 1e-30)
        SE = float(np.sqrt((se[m] ** 2).sum()) / s) if se is not None else 0.0
        ruled = bool(m.sum() >= MINPORTS and s / tot >= MINSHARE)
        res[g] = dict(n=int(m.sum()), share=float(s / tot), bias=B, se=SE, ruled=ruled, passed=(abs(B) <= tol + 3 * SE) if ruled else None,
                      port_rel=dict(p50=float(np.percentile(np.abs(rel), 50)), p90=float(np.percentile(np.abs(rel), 90)), p99=float(np.percentile(np.abs(rel), 99)),
                                    max=float(np.abs(rel).max())), signed_p5=float(np.percentile(rel, 5)), signed_p95=float(np.percentile(rel, 95)))
    return res


def run_fixture(name, levels, out, cdir, seeds_off=0, with_cycles=True, extra_levels_noise=True, kv=None):
    kv = kv or {}
    t_all = time.time(); sd = CF.scene_def(name); groups = CF.coarse_labels(name); res = dict(fixture=name, levels=levels, arms={})
    LM.log("fixture", name, "levels", levels)
    # ---- coarse arms
    c0 = LM.Lam(name, 1, (64, 16), coarse_mesh=True).build(**LM.PROD); r0 = c0.solve()
    prod_sol = CP.solve(c0.ss, {}, c0.C, torch.tensor(np.repeat((c0.Ep * c0.area * LM.RHO / 64)[:, None], 64, 0).repeat(3, 1), dtype=torch.float32, device="cuda"))
    # production solver (CP.solve, 3-channel float32) vs the reduced patch-level solve: a check of the reduction used by every arm
    Eind = prod_sol["phi_in_ind"].view(c0.ss.Ptot, 16, 3).sum(1)[:, 0].double().cpu().numpy() / c0.area
    L_prod = LM.RHO / np.pi * (c0.Ep + Eind)
    res["prod_vs_reduced_solve_max_rel"] = float(np.max(np.abs(L_prod - r0["L"]) / np.maximum(r0["L"], 1e-30)))
    res["C0prod"] = dict(meta=dict(iters=r0["iters"], rho=r0["rho"], closure=c0.closure(), t_C=c0.t_C, t_ext=c0.t_ext, nnz=int(c0.C._nnz())))
    arrays = dict(C0prod=L_prod, Ep_C0=c0.Ep)
    ci = LM.Lam(name, 1, (1, 1), coarse_mesh=True).build(M=65536, seed_C=5400 + seeds_off, K_ext=4096, seed_E=5401 + seeds_off); ri = ci.solve()
    arrays["C0inf"] = ri["L"]; res["C0inf"] = dict(iters=ri["iters"], rho=ri["rho"], closure=ci.closure()); ci.C = None; torch.cuda.empty_cache()
    # ---- production-estimator noise: replicate seeds and one-factor variants (same interface, same estimator definition)
    NREP = int(kv.get("nrep", 8))
    def prod_L(**kw):
        c = LM.Lam(name, 1, (64, 16), coarse_mesh=True).build(**kw); L = c.solve()["L"]; del c; torch.cuda.empty_cache(); return L
    for r in range(NREP):
        arrays[f"C0prod_r{r}"] = prod_L(M=128, K_ext=256, seed_C=5600 + r, seed_E=5700 + r)
    arrays["C0_Eonly"] = prod_L(M=128, K_ext=4096, seed_C=5340, seed_E=5341)            # sun-visibility estimator upgraded, production C rays and seed
    arrays["C0_Conly"] = prod_L(M=2048, K_ext=256, seed_C=5340, seed_E=5341)            # C rays upgraded (16x), production sun-visibility realization
    # ---- fine arms
    fine_lam = None
    for s in levels:
        t0 = time.time(); f = LM.Lam(name, s, (1, 1)).build(M=MS[s], seed_C=5410 + s + seeds_off, K_ext=KEXT[s], seed_E=5420 + s + seeds_off)
        rf = f.solve(); Lc = f.project(rf["L"]); rg = LM.solve_coarse_from_fine(f)
        arrays[f"F{s}"] = Lc; arrays[f"G{s}"] = rg["L"]
        res[f"F{s}"] = dict(patches=int(f.ss.Ptot), iters=rf["iters"], rho=rf["rho"], converged=rf["converged"], nnz=int(f.C._nnz()), closure=f.closure(),
                            t_C=f.t_C, t_ext=f.t_ext, wall=time.time() - t0, Gal_rho=rg["rho"], Gal_iters=rg["iters"],
                            gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9)
        LM.log(name, "s", s, json.dumps({k: v for k, v in res[f"F{s}"].items() if k != "closure"}))
        if s == levels[-1]:
            fine_lam = (f, rf["L"])
            if extra_levels_noise:                                                    # replicate at the finest level (noise estimate)
                f2 = LM.Lam(name, s, (1, 1)).build(M=MS[s], seed_C=5510 + s + seeds_off, K_ext=KEXT[s], seed_E=5520 + s + seeds_off)
                r2 = f2.solve(); arrays[f"F{s}_rep"] = f2.project(r2["L"]); arrays[f"G{s}_rep"] = LM.solve_coarse_from_fine(f2)["L"]; del f2
        else:
            del f
        torch.cuda.empty_cache()
    np.savez(out / f"{name}_coarse_arrays.npz", **arrays)
    # ---- comparisons per camera
    ss_c = CC.sceneset_of(sd)
    cyc = {}
    if with_cycles and (cdir / f"ccam_{name}_L.json").exists():
        geo, ports1 = cycles_ports(cdir, name, sd, ss_c, V1_OFF); _, ports0 = cycles_ports(cdir, name, sd, ss_c, V0_OFF); cyc = dict(geo=geo, V1=ports1, V0=ports0)
    cams = list(sd["cams"])
    res["cams"] = {}
    for cam in cams:
        rc = {}
        if cyc:
            ref1, cnt1 = cyc["V1"][cam]; ok = cnt1 >= MINPIX; refm = ref1.mean(0); se = ref1.std(0, ddof=1) / np.sqrt(len(ref1))
            ref0, cnt0 = cyc["V0"][cam]; ok0 = cnt0 >= MINPIX
        else:
            ok = np.ones(len(groups), bool)
        for arm in [k for k in arrays if k != "Ep_C0" and "_r" not in k.split("prod")[-1]]:
            if cyc:
                rc[f"T:{arm}_vs_cycles_V1"] = group_stats(arrays[arm], refm, ok, groups, se)
                rc[f"T:{arm}_vs_cycles_V0"] = group_stats(arrays[arm], ref0.mean(0), ok0, groups, ref0.std(0, ddof=1) / np.sqrt(len(ref0)))
        m = ok if cyc else np.ones(len(groups), bool)
        sm = f"{levels[-1]}"
        rc["A:G_vs_F"] = group_stats(arrays[f"G{sm}"], arrays[f"F{sm}"], m, groups)
        rc["B:C0prod_vs_G"] = group_stats(arrays["C0prod"], arrays[f"G{sm}"], m, groups)
        rc["B:C0inf_vs_G"] = group_stats(arrays["C0inf"], arrays[f"G{sm}"], m, groups)
        rc["noise:F_rep_vs_F"] = group_stats(arrays.get(f"F{sm}_rep", arrays[f"F{sm}"]), arrays[f"F{sm}"], m, groups)
        rc["T:C0prod_vs_F"] = group_stats(arrays["C0prod"], arrays[f"F{sm}"], m, groups)
        for nm, refk in (("G", f"G{sm}"), ("F", f"F{sm}")):
            reps = [group_stats(arrays[f"C0prod_r{r}"], arrays[refk], m, groups) for r in range(NREP)]
            rc[f"rep_bias_vs_{nm}"] = {g: dict(mean=float(np.mean([x[g]["bias"] for x in reps])), sd=float(np.std([x[g]["bias"] for x in reps], ddof=1)))
                                       for g in reps[0] if all(g in x for x in reps)}
        rc["B:Eonly_vs_G"] = group_stats(arrays["C0_Eonly"], arrays[f"G{sm}"], m, groups)
        rc["B:Conly_vs_G"] = group_stats(arrays["C0_Conly"], arrays[f"G{sm}"], m, groups)
        res["cams"][cam] = rc
    res["wall_s"] = time.time() - t_all
    CC.dump(res, out / f"{name}_stageB.json")
    if fine_lam is not None and cyc:
        f, Lf = fine_lam; pix = {}
        for cam, g in cyc["geo"].items():
            files = sorted(cdir.glob(f"cimg_{name}_L_{cam}_*.npy")); imgs = np.stack([np.load(p)[..., 0] for p in files])
            pred, pid = LM.pixel_prediction(f, Lf, g); ref = imgs.mean(0); var = imgs.var(0, ddof=1) / len(imgs); valid = np.isfinite(pred)
            coarse_pid = np.where(pid >= 0, f.parent[np.clip(pid, 0, None)], -1)
            per = {}
            for gname in np.unique(groups[coarse_pid[valid]]):
                mm = valid & (groups[np.clip(coarse_pid, 0, None)] == gname)
                if mm.sum() < 50:
                    continue
                d = pred[mm] - ref[mm]
                per[gname] = dict(pixels=int(mm.sum()), bias=float(d.sum() / ref[mm].sum()), nrmse=float(np.sqrt((d ** 2).sum() / (ref[mm] ** 2).sum())),
                                  noise_nrmse=float(np.sqrt(var[mm].sum() / (ref[mm] ** 2).sum())))
            pix[cam] = per
            np.save(out / f"{name}_{cam}_pix_pred.npy", pred.astype(np.float32)); np.save(out / f"{name}_{cam}_pix_ref.npy", ref.astype(np.float32))
        CC.dump(pix, out / f"{name}_pixelwise.json")
    LM.log("fixture done", name, f"{time.time() - t_all:.0f}s")
    return res


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); cdir = Path(sys.argv[2])
    fixtures = sys.argv[3].split(","); levels = [int(x) for x in sys.argv[4].split(",")]
    kv = dict(a.split("=") for a in sys.argv[5:])
    for n in fixtures:
        run_fixture(n, levels, out, cdir, with_cycles=kv.get("cycles", "1") == "1", kv=kv)


if __name__ == "__main__":
    main()
