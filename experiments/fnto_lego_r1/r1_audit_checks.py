"""Worklog 42: independent verification of the Batch 6 validity-audit findings on the ORIGINAL Batch 6 implementation
(commit 7ba0848; read-only on its code, data and results). 8dna26 .venv-cu128 (Mitsuba, torch, scipy).

    python r1_audit_checks.py <out dir>

F1 boundary ports: is the port origin on the surface? (ray from the origin along +n / -n: distance to the nearest
   surface; first hit along +n being the inside face of a closed solid = buried origin); port sun visibility vs the
   renderer's own per-pixel visibility of the pixels that map to the port (Stage A holdout: independent ground truth).
F2 excitation: effective rank of the 1,400 training incident fields in the coarse-port basis; share of an indirect-like
   field (one Lambertian bounce, albedo 0.5, over the T0 coupling graph, held-out T0 view 6 sun) outside their span.
F3 kernel support: actual radius reached by the N_LOCAL / N_MID nearest-neighbour lists vs the declared R_LOCAL / R_MID.
F6 solver / coupling: does solve() flag a slowly diverging operator (closed cube, rho = 1.05)? Material / normal agreement
   of the 615 incremental-vs-full 'tie' relation entries.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_lego"))
import fnto_common as C  # noqa: E402
import fnto_relations as R  # noqa: E402
import fnto_solve as SV  # noqa: E402


def f1(A, rec):
    geo = R.Geo(*C.offsets("T0"))
    n = A["nrm"].astype(np.float64); p = A["pos"].astype(np.float64)
    sel = np.arange(0, int(A["n_A"]), 7)                      # 1/7 of the Lego ports
    out = {}
    for sgn, name in ((1, "plus_n"), (-1, "minus_n")):
        valid, hp, prim, sid, t = geo.intersect(p[sel], sgn * n[sel])
        out[name] = (valid, prim, t)
    V, F = R.mesh_a()
    vp, prim_p, tp = out["plus_n"]; vm, prim_m, tm = out["minus_n"]
    cr = np.cross(V[F[prim_p, 1]] - V[F[prim_p, 0]], V[F[prim_p, 2]] - V[F[prim_p, 0]]) * A["tri_sign"][prim_p][:, None]
    inside_face = vp & ~A["tri_two"][prim_p] & ((cr * n[sel]).sum(1) > 0)      # first hit along +n seen from behind
    d_surf = np.minimum(np.where(vp, tp, np.inf), np.where(vm, tm, np.inf))
    rec["F1_port_origin"] = {
        "ports_checked": int(len(sel)),
        "dist_to_nearest_surface_along_normal_q50_q90_q99_mm": (np.quantile(d_surf[np.isfinite(d_surf)], [0.5, 0.9, 0.99]) * 1000).tolist(),
        "frac_origin_farther_than_1mm_from_surface": float((d_surf > 1e-3).mean()),
        "frac_first_hit_along_plus_n_is_inside_face": float(inside_face.mean()),
        "frac_plus_n_hit_within_eps_1mm": float((vp & (tp < C.EPS)).mean()),
    }
    # port sun visibility vs per-pixel renderer visibility (Stage A holdout)
    D = C.OUT / "stagea/prep_holdout"
    st = np.load(D / "start.npy"); port = np.load(D / "px_port.npy", mmap_mode="r"); vis = np.load(D / "px_vis.npy", mmap_mode="r")
    nrm = np.load(D / "px_nrm.npy", mmap_mode="r"); dom = np.load(D / "px_dom.npy", mmap_mode="r")
    pE = np.load(D / "portE.npy", mmap_mode="r"); suns = np.load(D / "suns.npy")
    agree, n_pairs, port_dark_pix_lit, port_lit_pix_dark, bias = 0, 0, 0, 0, []
    for j in range(len(st) - 1):
        s, e = st[j], st[j + 1]
        pp = np.asarray(port[s:e]).astype(np.int64); vv = np.asarray(vis[s:e]).astype(bool); nn = np.asarray(nrm[s:e]).astype(np.float32)
        w = suns[j]; cpx = np.clip(nn @ w, 0, None); cpo = np.clip(A["nrm"][pp] @ w, 0, None)
        m = (cpx > 0.1) & (cpo > 0.1) & (np.asarray(dom[s:e]) == 0)
        Vp = np.asarray(pE[j])[pp] > 0
        agree += int((Vp[m] == vv[m]).sum()); n_pairs += int(m.sum())
        port_dark_pix_lit += int((~Vp[m] & vv[m]).sum()); port_lit_pix_dark += int((Vp[m] & ~vv[m]).sum())
        Epix = C.E_SUN * cpx * vv; Epo = np.asarray(pE[j]).astype(np.float32)[pp]
        if m.any():
            bias.append((Epo[m].sum() - Epix[m].sum()) / max(Epix[m].sum(), 1e-9))
    rec["F1_port_vs_pixel_sun"] = {"pairs": n_pairs, "visibility_agreement": agree / n_pairs,
                                   "port_dark_where_pixel_lit": port_dark_pix_lit / n_pairs, "port_lit_where_pixel_dark": port_lit_pix_dark / n_pairs,
                                   "port_minus_pixel_irradiance_relative_bias_mean": float(np.mean(bias)), "per_image_bias_q10_q90": np.quantile(bias, [0.1, 0.9]).tolist()}


def f2(A, rec):
    pe = np.load(C.OUT / "stagea/prep_train/portE.npy", mmap_mode="r")
    f2c = A["f2c"].astype(np.int64); Nc = len(A["cobj"]); area = A["area"].astype(np.float64)
    X = np.zeros((pe.shape[0], Nc), np.float32)
    for s in range(0, pe.shape[0], 50):
        E = np.asarray(pe[s:s + 50]).astype(np.float32)
        for k in range(len(E)):
            X[s + k] = np.bincount(f2c, E[k] * area, Nc)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    U, S, Vt = np.linalg.svd(X.astype(np.float64), full_matrices=False)
    en = np.cumsum(S ** 2) / (S ** 2).sum()
    rec["F2_excitation"] = {"fields": int(X.shape[0]), "coarse_basis_dim": Nc, "effective_rank_90_99_999": [int(np.searchsorted(en, q) + 1) for q in (0.9, 0.99, 0.999)],
                            "coarse_ports_never_excited": int((np.abs(X).sum(0) == 0).sum())}
    # indirect-like field: one Lambertian bounce (albedo 0.5) of the held-out T0 view-6 sun over the T0 coupling graph
    D = np.load(C.OUT / "eval_px/heldout_T0.npz"); w = D["v6_light"]
    Vp = np.unpackbits(np.load(C.OUT / "eval_vis.npz")["heldout_T0_v6_T0"])[:len(A["obj"])].astype(bool)
    Es = C.E_SUN * np.clip(A["nrm"] @ w, 0, None) * Vp
    H = np.load(C.OUT / "rel_T0.npz")["hit"]
    rad = 0.5 / np.pi * Es
    Eind = np.zeros(len(Es))
    for s in range(0, len(H), 100000):
        h = H[s:s + 100000]; Eind[s:s + 100000] = (np.where(h >= 0, rad[np.clip(h, 0, None)], 0)).sum(1) * np.pi / C.K_GATHER
    y = np.bincount(f2c, Eind * area, Nc); y /= np.linalg.norm(y)
    ys = np.bincount(f2c, Es * area, Nc); ys /= np.linalg.norm(ys)
    k99 = rec["F2_excitation"]["effective_rank_90_99_999"][1]
    for name, v in (("indirect_one_bounce", y), ("direct_sun_same_view", ys)):
        proj = Vt[:k99].T @ (Vt[:k99] @ v)
        rec["F2_excitation"][f"{name}_energy_outside_span_rank99"] = float(np.linalg.norm(v - proj) ** 2)
        proj_all = Vt.T @ (Vt @ v)
        rec["F2_excitation"][f"{name}_energy_outside_full_span"] = float(np.linalg.norm(v - proj_all) ** 2)


def f3(A, rec):
    nb, nd = A["nbr"], A["nbr_d"]
    last = np.where((nb >= 0).all(1), nd[:, -1], C.R_LOCAL)
    cpos, mid = A["cpos"], A["mid"]
    sel = np.arange(0, len(mid), 11)
    dm = np.linalg.norm(cpos[np.clip(mid[sel], 0, None)] - A["pos"][sel][:, None], axis=-1)
    dmax = np.where(mid[sel] >= 0, dm, 0).max(1)
    rec["F3_support"] = {"band_declared_R_m": C.R_LOCAL, "band_kth_neighbour_dist_q10_q50_q90_m": np.quantile(last, [0.1, 0.5, 0.9]).tolist(),
                         "band_frac_lists_full_before_R": float((nb >= 0).all(1).mean()),
                         "mid_declared_R_m": C.R_MID, "mid_kth_neighbour_dist_q10_q50_q90_m": np.quantile(dmax, [0.1, 0.5, 0.9]).tolist(),
                         "mid_frac_reaching_declared_R_within_10pct": float((dmax > 0.9 * C.R_MID).mean())}


def f6(A, rec):
    # 615 'tie' entries: material / normal agreement of the two ports
    z0 = np.load(C.OUT / "rel_T1.npz")["hit"]; zi = np.load(C.OUT / "rel_T1_inc.npz")["hit"]
    mm = np.argwhere(z0 != zi); a = z0[mm[:, 0], mm[:, 1]]; b = zi[mm[:, 0], mm[:, 1]]
    ok = (a >= 0) & (b >= 0)
    rec["F6_ties"] = {"entries": int(len(mm)), "both_ports": int(ok.sum()), "same_material": float((A["mat"][a[ok]] == A["mat"][b[ok]]).mean()),
                      "normal_dot_q10_q50": np.quantile((A["nrm"][a[ok]] * A["nrm"][b[ok]]).sum(1), [0.1, 0.5]).tolist(),
                      "one_side_blocked_or_miss": int((~ok).sum())}
    # solver divergence detection on a slowly diverging operator (Batch 6 solve(), unchanged)
    sys.path.insert(0, str(HERE.parent / "fnto_lego" / "tests"))
    import test_fnto_contracts as TC
    faces = [([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]), ([0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, -1]),
             ([0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]), ([0, 1, 0], [1, 0, 0], [0, 0, 1], [0, -1, 0]),
             ([0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]), ([1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0])]
    An = TC.make_anchors([TC.rect_ports(o_, u, v, n_, 16, 16, 0) for o_, u, v, n_ in faces])
    geo = R.Geo(np.zeros(3), np.zeros(3), meshes=[(f"S{i}", *TC.rect_mesh(o_, u, v)) for i, (o_, u, v, n_) in enumerate(faces)])
    H, _ = R.gather(An, geo, mapper=lambda va, p_, pr, sid, d: R.map_hits_nearest(An, geo, va, p_, pr, sid, d, {f"S{i}": 0 for i in range(6)}))
    res = {}
    for rho in (0.8, 0.999, 1.05):
        K = TC.kernel([[-0.1, -0.1, -0.1], [1.1, 1.1, 1.1]])
        with torch.no_grad():                                  # rho > 1: bypass the sigmoid cap by scaling the diffuse via kappa lobes
            TC.set_analytic(K, rho=(min(rho, 0.999),) * 3)
        T = SV.Tables(K, TC.kernel([[-1, -1, -1], [1, 1, 1]]), torch.device("cuda"), anchors=An)
        if rho > 1:
            T.rho = T.rho * (rho / 0.999)
        S = SV.State(T, H, np.zeros(len(An["obj"]), bool), np.array([0, 0, 1.0])); S.Esun = torch.full_like(S.Esun, 1.0); S.spec.zero_()
        S.solve()
        res[str(rho)] = {"iters": S.iters, "diverged_flag": S.diverged, "last_resid": S.hist[-1], "flagged_unconverged": S.hist[-1] >= SV.TOL}
    rec["F6_solver_detection"] = res


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    A = R.load_anchors(); rec = {"batch6_commit_checked": "7ba0848 (Batch 6 code unchanged)"}
    for f in (f3, f1, f2, f6):
        try:
            f(A, rec)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc(); rec[f.__name__ + "_error"] = repr(e)
        (out / "audit_checks.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
