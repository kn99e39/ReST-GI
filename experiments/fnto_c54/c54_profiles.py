"""WL54: dump the geometry-dependent profile / accuracy-vs-unknowns data for the figures (needs torch for the fixtures).
    python c54_profiles.py <out json> <stageB dir> <stageC dir>"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phys_ref53"))
import c54_common as CC  # noqa: E402
import c54_fixtures as CF  # noqa: E402

out, bdir, sdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); res = {}
# x profile (mean over y) of the gap4 lower-slab top face: fine solutions, coarse interface, CASP
name = "gap4"; comps = CF.comps_at(name, 1, coarse=True); m = comps[0][1]; sel = m["port_face"] == 5; xs = np.round(m["port_center"][sel][:, 0], 4); ux = np.unique(xs)
lab = CF.coarse_labels(name); g = lab == "slab_lo:top"; B = np.load(bdir / f"{name}_coarse_arrays.npz"); S = np.load(sdir / f"{name}_casp_arrays.npz")
prof = lambda a: [float(a[g][xs == u].mean()) for u in ux]  # noqa: E731
res["gap4_profile"] = dict(x=ux.tolist(), **{k: prof(B[k]) for k in ("F4", "F8", "F16", "F32", "G16", "C0inf", "C0prod")}, C1=prof(S["C1"]), C1_c4=prof(S["C1_c4"]))
# accuracy vs unknowns: pooled group bias vs F16 (amended full-coverage groups = whole face groups here) for uniform levels and CASP arms
par = {}
for name, grp, recv in (("gap4", "slab_hi:bot", None), ("gap4", "slab_lo:top", None), ("floor_lam", "dom:bot", None), ("rot_lam", "dom:bot", None), ("lblock_lam", "dom:bot", None), ("gap8", "slab_hi:bot", None)):
    B = np.load(bdir / f"{name}_coarse_arrays.npz"); S = np.load(sdir / f"{name}_casp_arrays.npz"); lab = CF.coarse_labels(name); g = lab == grp; ref = B["F16"][g].sum()
    cells = {f"F{s}": sum(len(mm["port_area"]) * (s ** 2 if s > 1 else 1) for _, mm, _ in CF.comps_at(name, 1, coarse=True)) for s in (1, 2, 4, 8, 16)}
    entry = {"uniform": {s: dict(cells=cells[f"F{s}"], bias=float((B[f"F{s}"][g].sum() - ref) / ref)) for s in (2, 4, 8)}, "coarse_C0inf": dict(cells=cells["F1"], bias=float((B["C0inf"][g].sum() - ref) / ref)),
             "coarse_G": dict(cells=cells["F1"], bias=float((B["G16"][g].sum() - ref) / ref))}
    import json as _j
    sc = _j.load(open(sdir / f"{name}_stageC.json")); entry["casp"] = {k: dict(cells=sc["arms"][k]["cells"], bias=float((S[k][g].sum() - ref) / ref)) for k in ("C1", "C1_c1", "C1_c4", "C1_b4")}
    par[f"{name}:{grp}"] = entry
res["pareto"] = par
CC.dump(res, out)
print("ok")
