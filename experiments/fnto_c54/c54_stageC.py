"""WL54 Stage C/E (Lambert, exact K): evaluate the candidate CASP (protocol fnto_c54) against C0, uniform refinement, the fine solution
and the Cycles reference.

    python c54_stageC.py <out dir> <cycles dir> <stageB dir> <fixture,...>
Arms: C0prod (from Stage B), C1 = CASP (c = 2, l_max = 8, budget 8192 rays / coarse port, K_ext 256 per cell; seeds 5810 / 5811),
C1_rep (8 replicates, seeds 5820+r / 5830+r), sensitivity arms C1_c1, C1_c4 (levels c = 1 / 4) and C1_b4 (4x ray budget) -- reported,
never selected --, uniform F_s / G_s from Stage B for the accuracy-vs-unknowns comparison.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import c54_casp as CA
import c54_common as CC
import c54_fixtures as CF
import c54_lam as LM
import c54_stageB as SB


def vis_edge_ports(name, s=8):
    """Coarse ports whose fine-cell sun irradiance varies inside the port (sub-port visibility transition): (max - min) / mean > 0.2."""
    f = LM.Lam(name, s).build(M=8, seed_C=1, K_ext=128, seed_E=5)
    ep = f.Ep; lo = np.full(f.nc, np.inf); hi = np.zeros(f.nc)
    np.minimum.at(lo, f.parent, ep); np.maximum.at(hi, f.parent, ep)
    mean = np.bincount(f.parent, weights=f.area * ep, minlength=f.nc) / np.bincount(f.parent, weights=f.area, minlength=f.nc)
    return (hi - lo) > 0.2 * np.maximum(mean, 1e-12) * (mean > 0)


def casp_arm(name, c, budget, seeds, lmax=8):
    lev, nu, d = CA.levels_for(name, c=c, lmax=lmax)
    lam = CA.LamCASP(name, lev); torch.cuda.reset_peak_memory_stats()
    lam.build(budget=budget, seed_C=seeds[0], K_ext=256, seed_E=seeds[1]); r = lam.solve(); L = lam.project(r["L"])
    meta = dict(cells=int(len(lam.area)), level_hist={int(k): int((lev == k).sum()) for k in np.unique(lev)}, nnz=int(lam.C._nnz()), iters=r["iters"], rho=r["rho"], closure=lam.closure(),
                t_C=lam.t_C, t_ext=lam.t_ext, gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9, c=c, budget=budget)
    del lam; torch.cuda.empty_cache()
    return L, meta, lev, nu


def run(name, out, cdir, bdir, nrep=8):
    t_all = time.time(); sd = CF.scene_def(name); groups = CF.coarse_labels(name); arrB = np.load(bdir / f"{name}_coarse_arrays.npz")
    F16, G16 = arrB["F16"], arrB["G16"]; res = dict(fixture=name, arms={}, cams={})
    arrays = {k: arrB[k] for k in ("C0prod", "F2", "F4", "F8", "F16", "G2", "G4", "G8", "G16") if k in arrB.files}
    L, meta, lev, nu = casp_arm(name, 2.0, 8192, (5810, 5811)); arrays["C1"] = L; res["arms"]["C1"] = meta
    LM.log(name, "C1", json.dumps({k: v for k, v in meta.items() if k != "closure"}))
    for r in range(nrep):
        arrays[f"C1_r{r}"] = casp_arm(name, 2.0, 8192, (5820 + r, 5830 + r))[0]
    for tag, c, bud in (("C1_c1", 1.0, 8192), ("C1_c4", 4.0, 8192), ("C1_b4", 2.0, 32768)):
        L, m, _, _ = casp_arm(name, c, bud, (5810, 5811)); arrays[tag] = L; res["arms"][tag] = m
        LM.log(name, tag, json.dumps({k: v for k, v in m.items() if k not in ("closure", "level_hist")}))
    res["uniform_cells"] = {f"F{s}": int(CF_cells(name, s)) for s in (1, 2, 4, 8, 16)}
    near = nu >= CA.NU_MIN; vis = vis_edge_ports(name)
    extra = {"near": np.where(near, "near", "far"), "vis": np.where(vis, "vis_edge", "vis_other")}
    cyc = {}
    ss_c = CC.sceneset_of(sd)
    if (cdir / f"ccam_{name}_L.json").exists():
        geo, p1 = SB.cycles_ports(cdir, name, sd, ss_c, SB.V1_OFF); _, p0 = SB.cycles_ports(cdir, name, sd, ss_c, SB.V0_OFF); cyc = dict(V1=p1, V0=p0)
    sel = ["C0prod", "C1", "C1_c1", "C1_c4", "C1_b4", "F2", "F4", "F8", "G16", "F16"]
    for cam in sd["cams"]:
        rc = {}
        if cyc:
            ref1, cnt1 = cyc["V1"][cam]; ok = cnt1 >= SB.MINPIX; refm = ref1.mean(0); se = ref1.std(0, ddof=1) / np.sqrt(len(ref1))
            ref0, cnt0 = cyc["V0"][cam]; ok0 = cnt0 >= SB.MINPIX
        else:
            ok = np.ones(len(groups), bool)
        for arm in sel:
            a = arrays[arm]
            if cyc:
                rc[f"{arm}_vs_cycles_V1"] = SB.group_stats(a, refm, ok, groups, se); rc[f"{arm}_vs_cycles_V0"] = SB.group_stats(a, ref0.mean(0), ok0, groups, ref0.std(0, ddof=1) / np.sqrt(len(ref0)))
                for gk, lab in extra.items():
                    rc[f"{arm}_vs_cycles_V1_{gk}"] = SB.group_stats(a, refm, ok, lab, se)
            rc[f"{arm}_vs_F16"] = SB.group_stats(a, F16, ok, groups)
            for gk, lab in extra.items():
                rc[f"{arm}_vs_F16_{gk}"] = SB.group_stats(a, F16, ok, lab)
        reps = [SB.group_stats(arrays[f"C1_r{r}"], F16, ok, groups) for r in range(nrep)]
        rc["C1_rep_bias_vs_F16"] = {g: dict(mean=float(np.mean([x[g]["bias"] for x in reps])), sd=float(np.std([x[g]["bias"] for x in reps], ddof=1))) for g in reps[0] if all(g in x for x in reps)}
        res["cams"][cam] = rc
    res["wall_s"] = time.time() - t_all
    np.savez(out / f"{name}_casp_arrays.npz", levels=lev, nu=nu, vis=vis, **{k: v for k, v in arrays.items() if k.startswith("C1")})
    CC.dump(res, out / f"{name}_stageC.json"); LM.log("done", name, f"{time.time() - t_all:.0f}s")


def CF_cells(name, s):
    return sum(len(m["port_area"]) * (s ** 2 if s > 1 else 1) for _, m, _ in CF.comps_at(name, 1, coarse=True))


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); cdir, bdir = Path(sys.argv[2]), Path(sys.argv[3])
    for n in sys.argv[4].split(","):
        run(n, out, cdir, bdir)


if __name__ == "__main__":
    main()
