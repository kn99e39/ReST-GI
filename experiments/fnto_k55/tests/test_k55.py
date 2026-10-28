"""WL55 focused tests (protocol fnto_k55, section 7 of the brief). Every test records measured values; it passes only if every stated
tolerance holds.      P53_MI_VARIANT=cuda_ad_rgb python tests/test_k55.py <out json>
"""
import json
import math
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import k55_common as KCm  # noqa: E402
import k55_compose as KP  # noqa: E402
import k55_cor as COR  # noqa: E402
import k55_fixtures as KF  # noqa: E402
import k55_ops as KO  # noqa: E402
import c54_casp as CA  # noqa: E402
import p53_compose as CP  # noqa: E402
import p53_fixtures as FX  # noqa: E402
import p53_geom as G  # noqa: E402
import p53_tracer as TR  # noqa: E402

RES = []
DEV = "cuda"


def test(fn):
    def run(*a, **k):
        t0 = time.time()
        try:
            d = fn(*a, **k); ok = bool(d.pop("passed"))
        except Exception:
            d = dict(error=traceback.format_exc()); ok = False
        d.update(name=fn.__name__, passed=ok, seconds=round(time.time() - t0, 1)); RES.append(d)
        print(("PASS " if ok else "FAIL ") + fn.__name__, {k: v for k, v in d.items() if k not in ("name", "error")}, flush=True)
        if "error" in d:
            print(d["error"])
        return ok
    return run


def block(nx=2, ny=2, nz=1):
    return G.voxel_boundary(np.ones((nx, ny, nz), bool), (FX.H,) * 3, (-nx * FX.H / 2, -ny * FX.H / 2, -nz * FX.H / 2), sub=4)


@test
def boundary_correspondence_and_area():
    out = {}; ok = True
    for geo in ("slab", "bend90", "dent"):
        m, V = FX.domain_geometry(geo); lev = np.random.default_rng(1).choice([1, 2, 4], len(m["port_area"]))
        fm, VC, par = KO.refined(m, V, lev)
        fr0 = G.tri_frames(m["V"], V, m["F"], m["tri_port"], m["port_tangent"]); a0 = G.port_areas(fr0["area"], m["tri_port"], len(m["port_area"]))
        fr1 = G.tri_frames(fm["V"], VC, fm["F"], fm["tri_port"], fm["port_tangent"]); ac = G.port_areas(fr1["area"], fm["tri_port"], len(par))
        err = float(np.max(np.abs(np.bincount(par, weights=ac) - a0) / a0))
        # every cell's centroid lies on its parent port: distance to the parent's nearest triangle centroid < port diagonal
        cc = np.stack([np.bincount(fm["tri_port"], weights=fr1["centroid"][:, k] * fr1["area"]) for k in range(3)], 1) / np.bincount(fm["tri_port"], weights=fr1["area"])[:, None]
        pc = np.stack([np.bincount(m["tri_port"], weights=fr0["centroid"][:, k] * fr0["area"]) for k in range(3)], 1) / a0[:, None]
        dist = float(np.max(np.linalg.norm(cc - pc[par], axis=1)))
        cnt_ok = bool(np.all(np.bincount(par, minlength=len(lev)) == lev ** 2))
        out[geo] = dict(area_rel_err=err, max_cell_to_port_centroid=dist, counts_ok=cnt_ok, welded_vertices=int(len(fm["V"])))
        ok &= err < 1e-12 and dist < 0.0075 and cnt_ok
    return dict(passed=ok, **out)


@test
def fine_to_cells_nesting():
    m, V = FX.domain_geometry("dent"); lev = np.random.default_rng(2).choice([1, 2, 4], len(m["port_area"]))
    ff, VF, pf = KO.refined(m, V, 4); fc, VCc, pc = KO.refined(m, V, lev)
    cof = KO.fine_to_cells(lev)
    af = KP.cell_areas(m, V, 4); ac = KP.cell_areas(m, V, lev)
    agg = np.bincount(cof, weights=af, minlength=len(ac))
    frf = G.tri_frames(ff["V"], VF, ff["F"], ff["tri_port"], ff["port_tangent"]); frc = G.tri_frames(fc["V"], VCc, fc["F"], fc["tri_port"], fc["port_tangent"])
    cf = np.stack([np.bincount(ff["tri_port"], weights=frf["centroid"][:, k] * frf["area"]) for k in range(3)], 1) / np.bincount(ff["tri_port"], weights=frf["area"])[:, None]
    cc = np.stack([np.bincount(fc["tri_port"], weights=frc["centroid"][:, k] * frc["area"]) for k in range(3)], 1) / np.bincount(fc["tri_port"], weights=frc["area"])[:, None]
    d = np.linalg.norm(cf - cc[cof], axis=1); half_diag = 0.01 / lev[pf] * math.sqrt(2) / 2 + 1e-6
    return dict(passed=bool(np.max(np.abs(agg - ac) / ac) < 1e-12 and np.all(d <= half_diag)), area_rel=float(np.max(np.abs(agg - ac) / ac)), max_d_over_halfdiag=float(np.max(d / half_diag)))


@test
def sparse_tr_equals_dense_tr():
    m = block(); cams = np.array([FX.DIRS[k] for k in ("top", "obot")])
    fm, VC, par = KO.refined(m, m["V"].copy(), 1); dx = KO.DomainX(fm, VC, name="blkx")
    rd = TR.run_cells(dx, 64, 16, 256, 9001, cams=cams)                       # WL53 dense accumulator, same domain and seed
    rs = KO.tr_sparse(dx, 64, 16, 256, 9001, cams, par, len(m["port_area"]))
    Ts = torch.zeros(3, rs["shape"][0] * rs["shape"][1]); Ts[:, rs["idx"]] = rs["val"]; Ts = Ts.view(3, *rs["shape"])
    dT = float((Ts - rd["T"].cpu()).abs().max() / rd["T"].abs().max()); dK = float((rs["KC"] - rd["KC"].cpu()).abs().max() / rd["KC"].abs().max())
    return dict(passed=dT < 1e-5 and dK < 1e-5, max_rel_T=dT, max_rel_KC=dK, nnz=len(rs["idx"]), stats_equal=rs["stats"]["exits"] == rd["stats"]["exits"])


@test
def aggregation_flux_preservation_and_expectation():
    m = block(); V = m["V"].copy(); cams = np.array([FX.DIRS["top"]])
    fm, VC, par = KO.refined(m, V, 4); dx = KO.DomainX(fm, VC); r4 = KO.tr_sparse(dx, 64, 16, 512, 9002, cams, par, len(m["port_area"]))
    lev = np.ones(len(m["port_area"]), int); cof = KO.fine_to_cells(lev); af = KP.cell_areas(m, V, 4)
    agg = KO.aggregate(r4, cof, af, len(lev))
    cs4 = torch.zeros(3, r4["shape"][1], dtype=torch.float64).index_add_(1, r4["idx"] % r4["shape"][1], r4["val"].double())
    w = torch.tensor(af / np.bincount(cof, weights=af)[cof]).repeat_interleave(16)
    exp_cs = torch.zeros(3, agg["shape"][1], dtype=torch.float64).index_add_(1, torch.tensor(cof).repeat_interleave(16) * 16 + torch.arange(16).repeat(len(cof)), cs4 * w[None])
    csa = torch.zeros(3, agg["shape"][1], dtype=torch.float64).index_add_(1, agg["idx"] % agg["shape"][1], agg["val"].double())
    d_flux = float((csa - exp_cs).abs().max())
    # expectation: a direct level-1 run (16x fewer photons per coarse cell) agrees in port-pair totals within MC noise
    fm1, VC1, par1 = KO.refined(m, V, 1); r1 = KO.tr_sparse(KO.DomainX(fm1, VC1), 64, 16, 8192, 9003, cams, par1, len(m["port_area"]))
    def pp(r):
        T = torch.zeros(3, r["shape"][0] * r["shape"][1]); T[:, r["idx"]] = r["val"]; P = len(m["port_area"])
        return T.view(3, P, 64, P, 16).sum((2, 4))
    a, b = pp(agg), pp(r1); rel = float((a - b).norm() / b.norm())
    return dict(passed=d_flux < 1e-6 and rel < 0.015, colsum_dev=d_flux, portpair_rel_fro=rel)


@test
def zero_linearity_and_fixed_point():
    bm = block(); Vb = FX.place_above(bm["V"], 0.0, 0.01); fl = FX.floor_mesh(0.0, 6)
    sdx = dict(comps=[("dom", "medium", bm, Vb), ("floor", "lambert", fl, fl["V"].copy())], lights=[(KF.sun_s(40, 20), 1.0)], cams={}, emit={})
    fm, VC, par = KO.refined(bm, bm["V"].copy(), 1); r = KO.tr_sparse(KO.DomainX(fm, VC), 64, 16, 256, 9004, np.array([FX.DIRS["top"]]), par, len(bm["port_area"]))
    K = KO.KOp(r["idx"], r["val"], r["shape"])
    z = torch.zeros(r["shape"][1], device=DEV); zero = float((K[0] @ z).abs().max())
    x1 = torch.rand(r["shape"][1], device=DEV); x2 = torch.rand(r["shape"][1], device=DEV)
    lin = float(((K[0] @ (2 * x1 + 3 * x2)) - (2 * (K[0] @ x1) + 3 * (K[0] @ x2))).abs().max() / (K[0] @ (2 * x1 + 3 * x2)).abs().max())
    ss, parent, _ = KP.build_ss(sdx, [np.ones(len(bm["port_area"]), int), np.ones(len(fl["port_area"]), int)])
    C, esc, st = CA.cast_adaptive(ss, np.full(ss.Ptot, 32), 9005)
    phi0 = torch.rand(ss.Ptot * 64, 3, device=DEV) * 1e-3
    sol = CP.solve(ss, {0: K}, C, phi0, tol=1e-10)
    n = ss.Ptot * 64; Cd = C.to_dense().double(); Kd = torch.zeros(n, ss.Ptot * 16, dtype=torch.float64, device=DEV)
    sl = ss.comp_ports(0); Kd[sl.start * 64:sl.stop * 64, sl.start * 16:sl.stop * 16] = K[0].csr.to_dense().double()
    fl_sl = ss.comp_ports(1)
    for p in range(fl_sl.start, fl_sl.stop):
        Kd[p * 64:(p + 1) * 64, p * 16:(p + 1) * 16] = FX.FLOOR_RHO / 64
    x = torch.linalg.solve(torch.eye(n, dtype=torch.float64, device=DEV) - Kd @ Cd, phi0[:, 0].double())
    fp = float((sol["phi_out"][:, 0].double() - x).abs().max() / x.abs().max())
    return dict(passed=zero == 0 and lin < 1e-5 and fp < 1e-5 and st["back_hits"] == 0, zero=zero, linearity=lin, fixed_point_vs_dense=fp, back_hits=st["back_hits"])


@test
def ballistic_analytic_side_face():
    """KB on the slab's -y face toward obot with a uniform unit radiance arriving on the top cells from the backward direction:
    analytic port mean of exp(-sigma_t s) = (cz / (sigma h)) (1 - exp(-sigma h / cz)), cz = cos50 (no rays reach the +y face)."""
    m, V = FX.domain_geometry("slab"); sd = dict(comps=[("slab", "medium", m, V)], lights=[], cams={}, emit={}); out = {}; ok = True
    for lev in (1, 2):
        ss, parent, _ = KP.build_ss(sd, [np.full(len(m["port_area"]), lev)])
        c = np.asarray(FX.DIRS["obot"]); KB = KP.ballistic_KB(ss, 0, parent, c, n_per_port=200000, seed=3 + lev)
        phi = torch.zeros(ss.Ptot * 16, dtype=torch.float64, device=DEV); top = (ss.n[:, 2] > 0.5)
        cells = torch.unique(ss.tri_port[top]); tri = torch.stack([(ss.tri_port == cc).nonzero()[0, 0] for cc in cells])
        b = G.local_to_bin(G.to_local(torch.tensor(-c, dtype=torch.float32, device=DEV).expand(len(tri), 3), ss.n[tri], ss.t[tri], ss.b[tri]), 4)
        phi[cells * 16 + b] = ss.port_area[cells].double() * math.pi / 16
        L = torch.sparse.mm(KB, phi[:, None]).squeeze(1).cpu().numpy()
        lab = KCm.labels_of(dict(comps=[("slab", "medium", m, V)])); my = lab == "slab:-y"
        cz = math.cos(math.radians(50)); ana = cz / (FX.SIGMA_T * 0.01) * (1 - math.exp(-FX.SIGMA_T * 0.01 / cz))
        e = float(np.max(np.abs(L[my] - ana) / ana)); out[f"L{lev}"] = dict(mean=float(L[my].mean()), analytic=ana, max_rel=e); ok &= e < 0.02
    return dict(passed=ok, **out)


@test
def cor_zero_lambert_plane_and_isolated_slab():
    fl = FX.floor_mesh(0.0, 4); s = KF.sun_s(40, 20)
    sd = dict(comps=[("floor", "lambert", fl, fl["V"].copy())], lights=[(s, 1.0)], cams={}, emit={})
    cor = COR.COR(sd, ["top"]); r = cor.run(16_000_000, 9006)
    exact = FX.FLOOR_RHO / math.pi * (-s[2]); lam = float(np.max(np.abs(r["L"][0][:, 0] - exact) / exact))
    sd0 = dict(sd, lights=[(s, 0.0)]); r0 = COR.COR(sd0, ["top"]).run(200_000, 9007); zero = float(np.abs(r0["L"]).max())
    sdi = KF.scene_def("iso_L2"); keys = ["top", "obot"]; cs = COR.COR(sdi, keys); rc = cs.run(16_000_000, 9008)
    m, V = sdi["comps"][0][2], sdi["comps"][0][3]; dom = TR.Domain(m, V, FX.SIGMA_T, FX.ALBEDO, FX.G_PHASE)
    rt = TR.run_sun(dom, sdi["lights"][0][0], 1.0, 16_000_000, 9009, 64, cams=np.array([FX.DIRS[k] for k in keys]))
    Lt = rt["KC"][:, :, :, 0].permute(0, 2, 1).cpu().numpy(); lab = KCm.labels_of(sdi); top = lab == "slab:top"
    b = float(np.max(np.abs(rc["L"][0][top].sum(0) - Lt[0][top].sum(0)) / Lt[0][top].sum(0)))
    lk = (rc["stats"]["leaks"] + rc["stats"]["wrong"]) / rc["stats"]["launched"]                 # protocol stage_D: leaks + wrong side <= 1e-5
    return dict(passed=lam < 5e-3 and zero == 0 and b < 3e-3 and lk <= 1e-5, lambert_plane_max_rel=lam, zero=zero, iso_top_pooled_rel=b, leak_rate=lk,
                leaks=rc["stats"]["leaks"], wrong=rc["stats"]["wrong"], back_hits=rc["stats"]["back_hits"])


@test
def ownership_no_back_hits_refined_media():
    sd = KF.scene_def("gap4"); ss, parent, _ = KP.build_ss(sd, [np.full(240, 4), np.full(240, 2)])
    C, esc, st = CA.cast_adaptive(ss, np.full(ss.Ptot, 2), 9010)
    col = torch.zeros(ss.Ptot * 64, device=DEV, dtype=torch.float64).index_add_(0, C._indices()[1], C._values().double()) + esc.double()
    return dict(passed=st["back_hits"] == 0 and float((col - 1).abs().max()) < 1e-5, back_hits=st["back_hits"], closure_dev=float((col - 1).abs().max()))


@test
def rigid_frames_refined():
    m, V = FX.domain_geometry("slab"); R = G.rot((1.0, 2.0, 3.0), 37.0)
    f0, V0, _ = KO.refined(m, V, 2); f1, V1, _ = KO.refined(m, G.map_rigid(V, R, (0.1, -0.2, 0.05)), 2)
    a = G.tri_frames(f0["V"], V0, f0["F"], f0["tri_port"], f0["port_tangent"]); b = G.tri_frames(f1["V"], V1, f1["F"], f1["tri_port"], f1["port_tangent"])
    e = max(float(np.abs(a[k] @ R.T - b[k]).max()) for k in ("n", "t", "b")); ea = float(np.abs(a["area"] - b["area"]).max() / a["area"].max())
    return dict(passed=e < 1e-9 and ea < 1e-9, frame_err=e, area_err=ea)


@test
def deterministic_group_stats_json():
    rng = np.random.default_rng(0); P = 50; lab = np.array(["a"] * 25 + ["b"] * 25); ref = rng.random((P, 3)) + 0.1; pred = ref * 1.02
    g1 = KCm.group_bias(pred, ref, np.ones(P, bool), lab, ref * 0.01); g2 = KCm.group_bias(pred, ref, np.ones(P, bool), lab, ref * 0.01)
    s = json.dumps(KCm.jclean(g1), allow_nan=False)
    return dict(passed=g1 == g2 and abs(g1["a"]["bias"][0] - 0.02) < 1e-12 and len(s) > 0, bias=g1["a"]["bias"])


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for fn in (boundary_correspondence_and_area, fine_to_cells_nesting, sparse_tr_equals_dense_tr, aggregation_flux_preservation_and_expectation,
               zero_linearity_and_fixed_point, ballistic_analytic_side_face, cor_zero_lambert_plane_and_isolated_slab, ownership_no_back_hits_refined_media,
               rigid_frames_refined, deterministic_group_stats_json):
        fn()
    KCm.dump(dict(tests=RES, passed=sum(r["passed"] for r in RES), total=len(RES)), out)
    print(f"{sum(r['passed'] for r in RES)}/{len(RES)} passed")
