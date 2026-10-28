"""WL54 LABELLED CONTROL (not a candidate): matched-resolution intrinsic-K oracle (protocol fnto_c54, stage_E arms.Kfine_control).

Question: with the medium K's port resolution matched to the explicit C's, does the medium-to-medium composition error of the WL53 gap scenes
shrink? CASP cannot change a medium-to-medium link (K is port-resolved), so this isolates how much of that error lives on the K side of the
interface. The WL53 oracle TR (p53_tracer, unchanged) is run on the slab at 1 cm ports (240) and at 5 mm ports (960) with IDENTICAL bins
(Dout = Din = 16, i.e. 4 x 4 / 4 x 4), 2048 photons per input cell; both are composed with uniform C at the same resolution (equal ray density)
and the same exact sun term, and compared with the stored Cycles volume images.   K design is not changed: this is the WL53 operator at a finer port.

    P53_MI_VARIANT=cuda_ad_rgb python c54_kfine.py <out dir> <wl53 results> [scene,...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phys_ref53"))
import c54_amend1 as AM  # noqa: E402
import c54_casp as CA  # noqa: E402
import c54_common as CC  # noqa: E402
import c54_stageB as SB  # noqa: E402
import c54_stageE as SE  # noqa: E402
import p53_compose as CP  # noqa: E402
import p53_fixtures as FX  # noqa: E402
import p53_geom as G  # noqa: E402
import p53_oracle as OR  # noqa: E402
import p53_report as RP  # noqa: E402
import p53_tracer as TR  # noqa: E402

DEV = "cuda"; DB = 16
CAMS = ("top", "bot", "otop", "obot"); DIRS = np.array([FX.DIRS[k] for k in CAMS])


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def slab(s):
    if s == 1:
        return FX.slab_mesh()
    return G.voxel_boundary(np.ones((20, 20, 2), bool), (FX.H / 2,) * 3, (-0.05, -0.05, -0.005), sub=2)


def tr_operator(s, n_per, seed):
    m = slab(s); dom = TR.Domain(m, m["V"].copy(), FX.SIGMA_T, FX.ALBEDO, FX.G_PHASE, name=f"slab_s{s}")
    r = TR.run_cells(dom, DB, DB, n_per, seed, cams=DIRS, log=None); log("TR", s, f"{r['seconds']:.0f}s", json.dumps({k: v for k, v in r["stats"].items() if k in ("photons", "leaks", "wrong_side")}))
    return m, dict(T=r["T"], KC=r["KC"]), r["seconds"]


def coarse_parent(m, s):
    """fine port -> coarse (1 cm) port id of the slab (voxel // s, face)."""
    mc = slab(1); key = {(int(v[0]), int(v[1]), int(v[2]), int(f)): i for i, (v, f) in enumerate(zip(mc["port_voxel"], mc["port_face"]))}
    return np.array([key[(int(v[0]) // s, int(v[1]) // s, int(v[2]) // s, int(f))] for v, f in zip(m["port_voxel"], m["port_face"])])


def compose(scene, s, m, op, Ts, seed_off=0):
    sd = FX.scene_def(scene); off = (FX.H + sd["gap"]) / 2; V = m["V"]
    comps = [("slab_lo", "medium", m, V + [0, 0, -off]), ("slab_hi", "medium", m, V + [0, 0, off])]
    ss = CP.SceneSet([CP.Comp(n, k, mm, VV) for n, k, mm, VV in comps], DB, DB); sun, E = sd["lights"][0]
    m_port = np.full(ss.Ptot, 128 if s == 1 else 32)                                   # equal ray density per area
    C, esc, st = CA.cast_adaptive(ss, m_port, 5340 + seed_off)
    phi0 = torch.zeros(ss.Ptot * DB, 3, device=DEV); L0 = {}
    for ci, comp in enumerate(ss.comps):
        dom = TR.Domain(comp.mesh, comp.Vcur, FX.SIGMA_T, FX.ALBEDO, FX.G_PHASE, name=comp.name)
        r = TR.run_sun(dom, sun, E, OR.CFG["n_sun"], 5342 + ci + seed_off, DB, cams=DIRS, shadow=ss.isect); sl = ss.comp_ports(ci)
        phi0[sl.start * DB:sl.stop * DB] = r["T"][:, :, 0].T; L0[ci] = r["KC"][:, :, :, 0].permute(0, 2, 1).contiguous(); del r, dom; torch.cuda.empty_cache()
    Tm = {0: op["T"], 1: op["T"]}; sol = CP.solve(ss, Tm, C, phi0); rho = CA.rho_robust(ss, Tm, C); rho_wl53 = CP.spectral_radius(ss, Tm, C)
    pred = {}
    for k, key in enumerate(CAMS):
        d = np.asarray(FX.DIRS[key]); KCk = {ci: op["KC"][k] for ci in (0, 1)}; L0k = {ci: L0[ci][k] for ci in (0, 1)}
        L = CP.readout(ss, d, sol, L0k, KC=KCk, E_ext=None, mode="nee").double().cpu().numpy()
        ct = torch.tensor(d, dtype=torch.float32, device=DEV)
        wproj = torch.zeros(ss.Ptot, device=DEV).index_add_(0, ss.tri_port, ss.area * (ss.n * ct).sum(1).clamp(min=0)).double().cpu().numpy()
        par = np.concatenate([coarse_parent(m, s) + 240 * ci for ci in (0, 1)]); w = np.maximum(wproj, 1e-30)
        pred[f"{key}__nee"] = np.stack([np.bincount(par, weights=w * L[:, c], minlength=480) / np.bincount(par, weights=w, minlength=480) for c in range(3)], 1)
    meta = dict(ports=int(ss.Ptot), rho=rho, rho_wl53_power300=rho_wl53, iters=sol["iters"], converged=bool(sol["converged"]), back_hits=int(st["back_hits"]))
    log("compose", scene, s, json.dumps(meta)); del C, ss, sol; torch.cuda.empty_cache()
    return pred, meta


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); R = Path(sys.argv[2]); scenes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["gap4", "gap8"]
    ops = {}
    for s in (1, 2):
        ops[s] = tr_operator(s, 2048, 5851 + s - 1)
    res = {}
    for scene in scenes:
        sd = FX.scene_def(scene); groups = RP.port_groups(scene); ss0 = OR.scene_set(sd); cdir = R / "cycles"; geo = json.load(open(cdir / f"ccam_{scene}.json")); res[scene] = {"arms": {}, "cams": {}}
        preds = {}
        for s in (1, 2):
            m, op, t = ops[s]; preds[s], meta = compose(scene, s, m, op, t); meta["tr_seconds"] = t; res[scene]["arms"][f"K{s}"] = meta
        for cam, g in geo.items():
            files = sorted(cdir.glob(f"cimg_{scene}_{cam}_*.npy")); pid = CP.classify(ss0, CC.blender_ortho_rays(g, SB.V1_OFF))
            means = [CP.port_means(pid, np.load(f).reshape(-1, 3), ss0.Ptot) for f in files]; ref = np.stack([x[0] for x in means]); cnt = means[0][1]
            exp = AM.expected_pixels(ss0, g); cov = cnt / np.maximum(exp, 1e-9); ok_a = (cnt >= SB.MINPIX) & (cov >= AM.COV)
            key = f"{RP.dirkey(sd['cams'][cam]['dir'])}__nee"; rc = {}
            for s in (1, 2):
                rc[f"K{s}_declared"] = SE.compare_rgb(ref, preds[s][key], cnt, groups)
                rc[f"K{s}_amended"] = {c: SB.group_stats(preds[s][key][:, c], ref[:, :, c].mean(0), ok_a, groups, ref[:, :, c].std(0, ddof=1) / np.sqrt(len(ref))) for c in range(3)}
            res[scene]["cams"][cam] = rc
        np.savez(out / f"{scene}_kfine_preds.npz", **{f"K{s}|{k}": v for s in (1, 2) for k, v in preds[s].items()})
    CC.dump(res, out / "kfine_control.json"); log("done")


if __name__ == "__main__":
    main()
