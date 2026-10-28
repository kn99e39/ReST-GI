"""WL55 Stage D operator tests of the CAK K interface (protocol fnto_k55, stage_D.operator_tests; amendment A1.4) on a 2 x 2 x 1 cm block
(16 coarse ports, 256 fine 2.5 mm cells), K_fine = TR at level 4, K_cells = R_out K_fine P_in on mixed levels {1, 2, 4}.

    P53_MI_VARIANT=cuda_ad_rgb python k55_stageD.py ops <out json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import k55_common as KCm
import k55_compose as KP
import k55_ops as KO
import p53_fixtures as FX
import p53_geom as G

DEV = "cuda"


def block():
    return G.voxel_boundary(np.ones((2, 2, 1), bool), (FX.H,) * 3, (-FX.H, -FX.H, -FX.H / 2), sub=4)


def op(m, V, seed, albedo=FX.ALBEDO, n_per=2048, cams=("top",)):
    fm, VC, par = KO.refined(m, V, 4); dom = KO.DomainX(fm, VC, albedo=albedo)
    r = KO.tr_sparse(dom, 64, 16, n_per, seed, np.array([FX.DIRS[k] for k in cams]), par, len(m["port_area"]), s2=False)
    r["cam_keys"] = list(cams); r["level"] = 4; return r


def colsum(r):
    return torch.zeros(3, r["shape"][1], dtype=torch.float64).index_add_(1, r["idx"] % r["shape"][1], r["val"].double())


def portpair(r, P, cof_cells=None):
    """(3, P_out, P_in) coarse port-pair totals of a level-4 operator."""
    nrow, ncol = r["shape"]; row = r["idx"] // ncol; col = r["idx"] % ncol
    po = (row // 64) // 16; pi = (col // 16) // 16
    M = torch.zeros(3, P * P, dtype=torch.float64).index_add_(1, po * P + pi, r["val"].double()); return M.view(3, P, P) / 16 / 16


def main(out):
    res = {}; m = block(); V = m["V"].copy(); P = len(m["port_area"]); lev = np.random.default_rng(55).choice([1, 2, 4], P)
    af = KP.cell_areas(m, V, 4); cof = KO.fine_to_cells(lev); ncell = int((lev ** 2).sum())
    # closure at albedo 1 (fine and aggregated)
    a1 = op(m, V, 9101, albedo=(1.0, 1.0, 1.0)); cs = colsum(a1); ag = KO.aggregate(a1, cof, af, ncell); csa = colsum(ag)
    leak = (a1["stats"]["leaks"] + a1["stats"]["wrong_side"]) / a1["stats"]["photons"]
    res["closure_albedo1"] = dict(fine_min=float(cs.min()), fine_max=float(cs.max()), cells_min=float(csa.min()), cells_max=float(csa.max()), leak_rate=leak,
                                  passed=bool(abs(float(cs.min()) - 1) < 1e-5 + leak * 10 and abs(float(csa.max()) - 1) < 1e-5 + leak * 10 and leak <= 1e-5))
    # passivity + flux preservation across resolutions (normal albedo)
    r1 = op(m, V, 9102); cs1 = colsum(r1); g1 = KO.aggregate(r1, cof, af, ncell); csg = colsum(g1)
    w = torch.tensor(af / np.bincount(cof, weights=af)[cof]).repeat_interleave(16)
    exp = torch.zeros(3, g1["shape"][1], dtype=torch.float64).index_add_(1, torch.tensor(cof).repeat_interleave(16) * 16 + torch.arange(16).repeat(len(cof)), cs1 * w[None])
    se = float(np.sqrt(1.0 / 2048))
    res["passivity"] = dict(max_colsum_fine=cs1.max(1).values.tolist(), max_colsum_cells=csg.max(1).values.tolist(), passed=bool(float(csg.max()) <= 1 + 3 * se))
    res["flux_across_resolutions"] = dict(max_abs=float((csg - exp).abs().max()), passed=bool(float((csg - exp).abs().max()) < 1e-6))
    # reciprocity at the coarse port-pair level vs cross-seed noise
    r2 = op(m, V, 9103); A = torch.tensor(m["port_area"], dtype=torch.float64)
    X1 = portpair(r1, P) * A[None, None, :]; X2 = portpair(r2, P) * A[None, None, :]
    asym = float((X1 - X1.transpose(1, 2)).abs().sum() / X1.sum()); noise = float((X1 - X2).abs().sum() / X1.sum())
    res["reciprocity"] = dict(asym=asym, cross_seed=noise, ratio=asym / noise, passed=bool(asym <= 1.2 * noise))
    # rigid pose (material coordinates): K_fine at a rotated + translated pose vs rest, port-pair, vs seed noise
    R = G.rot((1.0, 2.0, 3.0), 37.0); rp = op(m, G.map_rigid(V, R, (0.03, -0.02, 0.01)), 9104)
    Xp = portpair(rp, P) * A[None, None, :]
    ratio = (torch.linalg.norm((Xp - X1).reshape(3, -1), dim=1) / torch.linalg.norm((X2 - X1).reshape(3, -1), dim=1)).tolist()
    res["rigid_pose"] = dict(ratio=ratio, passed=bool(all(0.85 <= x <= 1.15 for x in ratio)))
    # zero / linearity of the aggregated operator
    K = KO.KOp(g1["idx"], g1["val"], g1["shape"]); z = torch.zeros(g1["shape"][1], device=DEV)
    x1, x2 = torch.rand(g1["shape"][1], device=DEV), torch.rand(g1["shape"][1], device=DEV)
    lin = float(((K[1] @ (2 * x1 - 0.5 * x2)) - (2 * (K[1] @ x1) - 0.5 * (K[1] @ x2))).abs().max() / (K[1] @ (2 * x1)).abs().max())
    res["zero_linearity"] = dict(zero=float((K[1] @ z).abs().max()), linearity=lin, passed=bool(float((K[1] @ z).abs().max()) == 0 and lin < 1e-5))
    res["boundary_correspondence"] = dict(cells=ncell, area_rel=float(np.max(np.abs(np.bincount(cof, weights=af) - KP.cell_areas(m, V, lev)) / KP.cell_areas(m, V, lev))),
                                          passed=bool(np.max(np.abs(np.bincount(cof, weights=af) - KP.cell_areas(m, V, lev)) / KP.cell_areas(m, V, lev)) < 1e-12))
    res["n_passed"] = sum(v["passed"] for v in res.values() if isinstance(v, dict)); res["n"] = sum(1 for v in res.values() if isinstance(v, dict))
    KCm.dump(res, out); print(json.dumps(res, indent=1))


if __name__ == "__main__" and sys.argv[1] == "ops":
    main(Path(sys.argv[2]))


def diag(out):
    """Amendment A2 diagnostics (rules fixed before the run):
    closure: at albedo 1 the column-sum deficit, in photons (sum(1 - colsum) x n_per), must equal the leaked + wrong-side photon count;
    rigid: 4 rest seeds, 4 pose seeds; r = ||mean(pose) - mean(rest)|| / (0.5 rms_{i<j} ||rest_i - rest_j||) (expectation 1 under invariance);
    invariance holds iff r in [0.85, 1.15] in every channel."""
    res = {}; m = block(); V = m["V"].copy(); P = len(m["port_area"]); A = torch.tensor(m["port_area"], dtype=torch.float64)
    a1 = op(m, V, 9101, albedo=(1.0, 1.0, 1.0)); cs = colsum(a1)
    deficit = float(((1 - cs) * 2048).sum(1).max()); lost = a1["stats"]["leaks"] + a1["stats"]["wrong_side"]
    res["closure_accounting"] = dict(deficit_photons_max_channel=deficit, leaked_plus_wrong=lost, passed=bool(abs(deficit - lost) < 0.01 and lost / a1["stats"]["photons"] <= 1e-5))
    R = G.rot((1.0, 2.0, 3.0), 37.0); Vp = G.map_rigid(V, R, (0.03, -0.02, 0.01))
    rest = [portpair(op(m, V, s), P) * A[None, None, :] for s in (9102, 9103, 9105, 9106)]
    pose = [portpair(op(m, Vp, s), P) * A[None, None, :] for s in (9104, 9107, 9108, 9109)]
    rm, pm = torch.stack(rest).mean(0), torch.stack(pose).mean(0)
    pair = [torch.linalg.norm((rest[i] - rest[j]).reshape(3, -1), dim=1) ** 2 for i in range(4) for j in range(i + 1, 4)]
    rms = torch.sqrt(torch.stack(pair).mean(0)); r = (torch.linalg.norm((pm - rm).reshape(3, -1), dim=1) / (0.5 * rms)).tolist()
    res["rigid_replicates"] = dict(r=r, passed=bool(all(0.85 <= x <= 1.15 for x in r)))
    KCm.dump(res, out); print(json.dumps(res, indent=1))


if __name__ == "__main__" and sys.argv[1] == "diag":
    diag(Path(sys.argv[2]))
