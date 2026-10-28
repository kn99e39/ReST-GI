"""WL54 Stage F: cost and empirical scaling of C0 vs CASP vs uniform fine C (protocol fnto_c54, stage_F_cost).

    P53_MI_VARIANT=cuda_ad_rgb python c54_cost.py <out json> [repeats=5]
Interface matched to the K-C pipeline: Dout 64 / Din 16, equal ray budget (128 rays per out-cell of a coarse port; CASP cells 128 / l^2).
Per arm and repeat (CUDA-synchronised wall clock, median over repeats):
  scene     SceneSet construction (frames, port tables, Mitsuba BVH build)
  levels    CASP only: coarse SceneSet + 256-ray prepass + level rule (the coarse SceneSet is also what C0 needs; it is reported separately)
  refine    CASP only: numpy mesh refinement
  C         relation build (ray casting, sparse assembly)           sun       sun-visibility term
  apply20   20 iterations of  phi_in = C phi_out ; phi_out = phi0 + K phi_in  (3 channels, Lambert K)
  total     scene + (levels + refine) + C + sun + apply20 = a geometry update followed by a 20-iteration solve
Also: unknowns (cells), nnz(C), bytes(C), peak GPU memory. Scaling: floor edge 10 / 20 / 30 / 60 cm (ports), gap 4 / 8 / 30 mm
(near-contact density), and a rigid box move (rot_lam vs floor_lam geometry; a full rebuild is the update).
The historical Python-looped hit mapping of WL53 is NOT the comparator: C0 here is the same GPU caster cast_C.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phys_ref53"))
import c54_casp as CA  # noqa: E402
import c54_common as CC  # noqa: E402
import c54_fixtures as CF  # noqa: E402
import c54_lam as LM  # noqa: E402
import p53_compose as CP  # noqa: E402

DEV = "cuda"
BINS = (64, 16)


def sync():
    torch.cuda.synchronize(); return time.perf_counter()


def apply20(ss, C, area, ep):
    phi0 = torch.tensor(np.repeat((ep * area * LM.RHO / ss.Dout)[:, None], ss.Dout, 0).repeat(3, 1), dtype=torch.float32, device=DEV); phi = phi0.clone()
    t0 = sync()
    for _ in range(20):
        phi = phi0 + CP.apply_K(ss, {}, CP.apply_C(C, phi))
    return sync() - t0


def one(name, arm, nf=None):
    if nf is not None:
        CF.NF = nf
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); T = {}
    t0 = sync()
    if arm == "C0":
        lam = LM.Lam(name, 1, BINS, coarse_mesh=True); T["scene"] = sync() - t0
        lam.build(**LM.PROD); T["C"], T["sun"] = lam.t_C, lam.t_ext
    elif arm == "CASP":
        lam0 = LM.Lam(name, 1, BINS, coarse_mesh=True); T["coarse_scene"] = sync() - t0; t1 = sync()
        lev, nu, d = CA.assign_levels(lam0.ss); T["levels"] = sync() - t1; del lam0
        t1 = sync()
        lam = CA.LamCASP.__new__(CA.LamCASP); lam.name, lam.s, lam.bins = name, 0, BINS
        coarse = CF.comps_at(name, 1, coarse=True); off = 0; comps, pars = [], []
        for (n, m, V) in coarse:
            P = len(m["port_area"]); fm, Vc, pp = CA.refine_mesh(m, V, lev[off:off + P]); comps.append((n, fm, Vc)); pars.append(pp + off); off += P
        T["refine"] = sync() - t1; t1 = sync()
        lam.levels = lev; lam.ss = CP.SceneSet([CP.Comp(n, "lambert", m, V, rho=LM.RHO) for n, m, V in comps], BINS[0], BINS[1]); T["scene"] = sync() - t1
        lam.parent = np.concatenate(pars); lam.nc = int(lam.parent.max()) + 1; lam.area = lam.ss.port_area.double().cpu().numpy(); lam.sun = np.asarray(CF.light_of(name), float)
        lam.build(budget=8192, seed_C=5810, K_ext=256, seed_E=5811); T["C"], T["sun"] = lam.t_C, lam.t_ext
    elif arm.startswith("uniform"):
        s = int(arm[7:]); lam = LM.Lam(name, s, BINS); T["scene"] = sync() - t0
        lam.build(M=max(1, 128 // s ** 2), seed_C=5410, K_ext=256, seed_E=5420); T["C"], T["sun"] = lam.t_C, lam.t_ext
    T["apply20"] = apply20(lam.ss, lam.C, lam.area, lam.Ep)
    T["total"] = T.get("coarse_scene", 0.0) * (arm == "CASP") + T.get("levels", 0.0) + T.get("refine", 0.0) + T["scene"] + T["C"] + T["sun"] + T["apply20"]
    mem = torch.cuda.max_memory_allocated() / 1e9
    nnz = int(lam.C._nnz()); out = dict(T=T, cells=int(lam.ss.Ptot), coarse_ports=int(lam.nc), nnz=nnz, bytes_C=nnz * (16 + 4), gpu_peak_gb=mem)
    del lam; torch.cuda.empty_cache(); return out


def bench(configs, repeats):
    res = []
    for (name, nf, tag) in configs:
        for arm in ("C0", "CASP", "uniform4"):
            runs = [one(name, arm, nf) for _ in range(repeats + 1)][1:]                     # first run discarded (warm-up / allocator)
            med = {k: float(np.median([r["T"][k] for r in runs])) for k in runs[0]["T"]}; mad = {k: float(np.max([r["T"][k] for r in runs]) - np.min([r["T"][k] for r in runs])) for k in med}
            res.append(dict(tag=tag, fixture=name, floor_n=nf, arm=arm, median=med, range=mad, cells=runs[0]["cells"], coarse_ports=runs[0]["coarse_ports"], nnz=runs[0]["nnz"],
                            bytes_C=runs[0]["bytes_C"], gpu_peak_gb=max(r["gpu_peak_gb"] for r in runs)))
            print(tag, arm, json.dumps(dict(total=med["total"], C=med["C"], cells=runs[0]["cells"], nnz=runs[0]["nnz"], gb=res[-1]["gpu_peak_gb"])), flush=True)
    return res


def main():
    out = Path(sys.argv[1]); repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    load = open("/proc/loadavg").read().strip(); other = subprocess.run("ps -eo pcpu,comm --sort=-pcpu | head -4", shell=True, capture_output=True, text=True).stdout
    meta = dict(gpu_before=smi, loadavg=load, top_processes=other, repeats=repeats, device=torch.cuda.get_device_name(0), torch=torch.__version__)
    print(json.dumps(meta), flush=True)
    configs = [("floor_lam", 10, "floor10"), ("floor_lam", 20, "floor20"), ("floor_lam", 30, "floor30"), ("floor_lam", 60, "floor60"),
               ("gap4", None, "gap4"), ("gap8", None, "gap8"), ("gap30", None, "gap30"), ("rot_lam", 30, "rot_lam30"), ("lblock_lam", 30, "lblock30")]
    res = bench(configs, repeats); CF.NF = 30
    CC.dump(dict(meta=meta, results=res), out)


if __name__ == "__main__":
    main()
