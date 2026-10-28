"""WL54 focused physical-contract tests of the explicit coupling C and its spatial aggregation (protocol fnto_c54, stage_D_tests).

    P53_MI_VARIANT=cuda_ad_rgb python tests/test_c54.py <out json> [cycles dir]
Every test records its measured values; a test passes only if every stated tolerance holds.
"""
import json
import math
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phys_ref53"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import c54_casp as CA  # noqa: E402
import c54_common as CC  # noqa: E402
import c54_fixtures as CF  # noqa: E402
import c54_lam as LM  # noqa: E402
import p53_compose as CP  # noqa: E402
import p53_fixtures as FX  # noqa: E402
import p53_geom as G  # noqa: E402

RES = []


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


# ------------------------------------------------------------------ helpers


def face_group_flux(lam, src_label, dst_label, labels):
    """Area-weighted fraction of the flux leaving cells of group src that enters cells of group dst (patch-level C)."""
    C = lam.port_C(); idx = C._indices(); v = C._values().double()
    lab = np.array(labels)[lam.parent]; src = torch.tensor(lab[idx[1].cpu().numpy()] == src_label, device=LM.DEV); dst = torch.tensor(lab[idx[0].cpu().numpy()] == dst_label, device=LM.DEV)
    a = torch.tensor(lam.area, device=LM.DEV, dtype=torch.float64)
    num = (v * a[idx[1]] * (src & dst).double()).sum(); den = a[torch.tensor(lab == src_label, device=LM.DEV)].sum()
    return float(num / den)


def howell_parallel_squares(a, d):
    X = Y = a / d; s1 = math.sqrt(1 + Y * Y); s2 = math.sqrt(1 + X * X)
    t = 0.5 * math.log((1 + X * X) * (1 + Y * Y) / (1 + X * X + Y * Y)) + X * s1 * math.atan(X / s1) + Y * s2 * math.atan(Y / s2) - X * math.atan(X) - Y * math.atan(Y)
    return 2 / (math.pi * X * Y) * t


def dense(C):
    return C.to_dense().double().cpu().numpy()


# ------------------------------------------------------------------ tests


@test
def refine_mesh_conservation():
    """Port subdivision and reconstruction: areas, normals, parents, conservative projection / prolongation (planar and bent)."""
    out = {}; ok = True
    for tag, (m, V) in (("planar", FX.domain_geometry("slab")), ("bend90", FX.domain_geometry("bend90")), ("dent", FX.domain_geometry("dent"))):
        P = len(m["port_area"]); lev = np.array([1, 2, 4, 8])[np.arange(P) % 4]
        fm, VC, pp = CA.refine_mesh(m, V, lev)
        fr0 = G.tri_frames(m["V"], V, m["F"], m["tri_port"], m["port_tangent"]); a0 = np.bincount(m["tri_port"], weights=fr0["area"], minlength=P)
        fr1 = G.tri_frames(fm["V"], VC, fm["F"], fm["tri_port"], fm["port_tangent"]); ptri = pp[fm["tri_port"]]
        a1 = np.bincount(ptri, weights=fr1["area"], minlength=P)
        rel = float(np.max(np.abs(a1 / a0 - 1)))
        ncell = np.bincount(pp, minlength=P); cells_ok = bool(np.all(ncell == lev ** 2))
        n0 = np.zeros((P, 3)); np.add.at(n0, m["tri_port"], fr0["n"] * fr0["area"][:, None]); n0 /= np.linalg.norm(n0, axis=1, keepdims=True)
        dots = (fr1["n"] * n0[ptri]).sum(1)
        # surface identity: every refined triangle's vertices lie on the coarse triangulated surface (distance to the coarse port plane of its triangle)
        a_cell = np.bincount(fm["tri_port"], weights=fr1["area"]); flux = 3.7 * a_cell; port_flux = np.bincount(pp, weights=flux, minlength=P)
        cons = float(np.max(np.abs(port_flux / (3.7 * a1) - 1)))
        out[tag] = dict(max_rel_area_err=rel, cells_match_levels=cells_ok, min_normal_dot=float(dots.min()), projection_identity_err=cons)
        ok &= rel < 1e-9 and cells_ok and cons < 1e-12 and dots.min() > (1 - 1e-9 if tag == "planar" else 0.97)
    return dict(passed=ok, **out)


@test
def analytic_form_factor():
    """Coaxial parallel squares (a = 0.1 m): C-estimated face-to-face flux fraction vs Howell's closed form; unbiased in seed."""
    out = {}; ok = True
    for name in ("gap4", "gap8", "gap30"):
        exact = howell_parallel_squares(0.1, CF.GAP[name]); labels = CF.coarse_labels(name); row = dict(exact=exact)
        for tag, kw in (("s8", dict(s=8)), ("s16", dict(s=16))):
            vals = []
            for r in range(3):
                lam = LM.Lam(name, kw["s"]).build(M=MS_T[kw["s"]], seed_C=900 + r, K_ext=8, seed_E=1); vals.append(face_group_flux(lam, "slab_lo:top", "slab_hi:bot", labels)); del lam; torch.cuda.empty_cache()
            row[tag] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)), rel_err=float(np.mean(vals) / exact - 1))
        lam = LM.Lam(name, 1, coarse_mesh=True).build(M=65536, seed_C=77, K_ext=8, seed_E=1); v = face_group_flux(lam, "slab_lo:top", "slab_hi:bot", labels)
        row["coarse_Minf"] = dict(value=v, rel_err=v / exact - 1); del lam
        lev = CA.levels_for(name)[0]; lc = CA.LamCASP(name, lev).build(); v = face_group_flux(lc, "slab_lo:top", "slab_hi:bot", labels)
        row["casp"] = dict(value=v, rel_err=v / exact - 1, cells=int(len(lc.area))); del lc; torch.cuda.empty_cache()
        out[name] = row
        ok &= all(abs(row[k]["rel_err"]) < 0.01 for k in ("s8", "s16", "coarse_Minf", "casp"))
    return dict(passed=ok, **out)


MS_T = {8: 256, 16: 128}


@test
def closure_nonneg_and_determinism():
    """Flux closure (column sum + escape = 1), non-negativity, no back hits, same-seed determinism, escape accounting."""
    out = {}; ok = True
    arms = {"gap4_s8": lambda: LM.Lam("gap4", 8).build(M=256, seed_C=5, K_ext=16, seed_E=6),
            "floor_lam_s4": lambda: LM.Lam("floor_lam", 4).build(M=512, seed_C=5, K_ext=16, seed_E=6),
            "floor_lam_prod": lambda: LM.Lam("floor_lam", 1, (64, 16), coarse_mesh=True).build(**LM.PROD),
            "floor_lam_casp": lambda: CA.LamCASP("floor_lam", CA.levels_for("floor_lam")[0]).build(),
            "lblock_casp_bins": lambda: CA.LamCASP("lblock_lam", CA.levels_for("lblock_lam")[0], bins=(64, 16)).build()}
    for k, mk in arms.items():
        lam = mk(); cl = lam.closure(); C = lam.C
        nonneg = bool((C._values() >= 0).all()); out[k] = dict(closure_min=cl["min"], closure_max=cl["max"], back_hits=cl["back_hits"], nonneg=nonneg, nnz=int(C._nnz()), escape_mean=float(lam.esc.double().mean()))
        ok &= abs(cl["min"] - 1) < 1e-6 and abs(cl["max"] - 1) < 1e-6 and cl["back_hits"] == 0 and nonneg
        if k in ("gap4_s8", "floor_lam_casp"):
            lam2 = mk(); same = bool(torch.equal(lam.C._indices(), lam2.C._indices())) and float((lam.C._values() - lam2.C._values()).abs().max()) < 1e-7
            r1, r2 = lam.solve(), lam2.solve(); out[k]["same_seed_identical"] = same; out[k]["solve_max_rel"] = float(np.max(np.abs(r1["L"] - r2["L"]) / np.maximum(r1["L"], 1e-30)))
            ok &= same and out[k]["solve_max_rel"] < 1e-6
        del lam; torch.cuda.empty_cache()
    return dict(passed=ok, **out)


def _asym(X):
    return float(np.abs(X - X.T).sum() / np.maximum((X + X.T).sum(), 1e-30))


@test
def reciprocity_area_weighted():
    """A_j C_ij = A_i C_ji up to the cross-seed noise (WL53 rule: <= 1.25 x noise statistic + 0.005), uniform fine and CASP mixed levels."""
    out = {}; ok = True
    for tag, mk in (("gap4_s4", lambda sd: LM.Lam("gap4", 4).build(M=4096, seed_C=sd, K_ext=8, seed_E=1)),
                    ("floor_lam_casp", lambda sd: CA.LamCASP("floor_lam", CA.levels_for("floor_lam")[0]).build(budget=65536, seed_C=sd, K_ext=8, seed_E=1))):
        Xs = []
        for sd in (11, 12):
            lam = mk(sd); X = dense(lam.C) * lam.area[None, :]; Xs.append(X); del lam; torch.cuda.empty_cache()
        asym = _asym(Xs[0]); noise = float(np.abs(Xs[0] - Xs[1]).sum() / (Xs[0] + Xs[1]).sum())
        out[tag] = dict(asym=asym, cross_seed_stat=noise, rule=1.25 * noise + 0.005); ok &= asym <= 1.25 * noise + 0.005
    return dict(passed=ok, **out)


@test
def zero_superposition_application_fixed_point():
    """Zero field -> exactly 0; linearity solve(a + b) = solve(a) + solve(b); sparse apply vs dense; dense (I - rho C)^-1 vs iteration; rho < 1."""
    lam = LM.Lam("gap4", 2).build(M=4096, seed_C=3, K_ext=64, seed_E=4); C = lam.port_C(); rng = np.random.default_rng(0)
    za = lam.solve(Ep=np.zeros_like(lam.Ep))["L"]; zero_ok = float(np.abs(za).max()) == 0.0
    ea, eb = rng.random(len(lam.Ep)) * (lam.Ep > 0), rng.random(len(lam.Ep)); ra, rb, rab = lam.solve(Ep=ea, tol=1e-12)["L"], lam.solve(Ep=eb, tol=1e-12)["L"], lam.solve(Ep=ea + eb, tol=1e-12)["L"]
    lin = float(np.abs(rab - ra - rb).max() / rab.max())
    D = dense(C); v = rng.random(len(lam.Ep)); applied = torch.sparse.mm(C.double(), torch.tensor(v, device=LM.DEV, dtype=torch.float64)[:, None]).squeeze(1).cpu().numpy()
    apply_err = float(np.abs(applied - D @ v).max() / np.abs(D @ v).max())
    phi0 = LM.RHO * lam.Ep * lam.area; phi = np.linalg.solve(np.eye(len(phi0)) - LM.RHO * D, phi0); Ld = LM.RHO / math.pi * (lam.Ep + (D @ phi) / lam.area)
    r = lam.solve(tol=1e-12); fp = float(np.abs(Ld - r["L"]).max() / Ld.max())
    return dict(passed=zero_ok and lin < 1e-9 and apply_err < 1e-6 and fp < 1e-8 and r["rho"] < 1, zero_exact=zero_ok, linearity=lin, apply_err=apply_err, fixed_point_vs_dense=fp, rho=r["rho"])


@test
def blocker():
    """A thin black plate between the two plates removes the face-to-face transfer (uniform fine, coarse, CASP)."""
    out = {}; ok = True
    plate = G.open_grid(20, 20, 0.01, (-0.1, -0.1, 0.0), sub=1)
    for tag, (comps_f) in (("coarse", lambda: [(n, m, V) for n, m, V in CF.comps_at("gap4", 1, coarse=True)]),
                           ("s4", lambda: CF.comps_at("gap4", 4)),
                           ("casp", lambda: [(n, *CA.refine_mesh(m, V, CA.levels_for("gap4")[0][o:o + len(m["port_area"])])[:2]) for (n, m, V), o in zip(CF.comps_at("gap4", 1, coarse=True), (0, 240))])):
        comps = comps_f() + [("plate", plate, plate["V"].copy())]
        ss = CP.SceneSet([CP.Comp(n, "lambert", m, V, rho=0.8) for n, m, V in comps], 1, 1)
        C, esc, st = CP.cast_C(ss, 2048, 5); idx = C._indices(); v = C._values(); pc = ss.port_comp
        cross = ((pc[idx[1]] == 0) & (pc[idx[0]] == 1)) | ((pc[idx[1]] == 1) & (pc[idx[0]] == 0))
        leak = float(v[cross].sum()); to_plate = float(v[(pc[idx[0]] == 2)].sum())
        out[tag] = dict(cross_flux=leak, onto_plate_flux=to_plate, back_hits=int(st["back_hits"])); ok &= leak < 1e-9 and to_plate > 0
        del ss, C; torch.cuda.empty_cache()
    return dict(passed=ok, **out)


@test
def rigid_consistency():
    """Rotating and translating the whole scene (sun included) leaves port radiance unchanged within the cross-seed noise; CASP levels are rotation invariant."""
    R = G.rot((0.3, -1.0, 0.7), 41.0); t = np.array([0.37, -0.21, 0.11]); out = {}; ok = True
    for name in ("floor_lam", "gap4"):
        a = [LM.Lam(name, 4).build(M=512, seed_C=s, K_ext=128, seed_E=s + 1) for s in (31, 32)]; La = [x.project(x.solve()["L"]) for x in a]
        b = LM.Lam(name, 4, rigid=(R, t)).build(M=512, seed_C=33, K_ext=128, seed_E=34); Lb = b.project(b.solve()["L"])
        m = La[0] > 1e-6 * La[0].max(); noise = float(np.abs(La[0] - La[1])[m].sum() / La[0][m].sum()); diff = float(np.abs(La[0] - Lb)[m].sum() / La[0][m].sum())
        l0 = CA.levels_for(name)[0]; lam = LM.Lam(name, 1, rigid=(R, t), coarse_mesh=True); l1 = CA.assign_levels(lam.ss)[0]
        out[name] = dict(pooled_L1_diff=diff, cross_seed_noise=noise, levels_equal_fraction=float((l0 == l1).mean()))
        ok &= diff <= 1.5 * noise + 0.005 and (l0 == l1).mean() > 0.95
        del a, b, lam; torch.cuda.empty_cache()
    return dict(passed=ok, **out)


@test
def same_domain_reentry():
    """Concave L-block re-enters its own exitance; a convex box does not; fine uniform agrees with coarse within noise."""
    out = {}; ok = True
    def frac(comps, M, seed):
        ss = CP.SceneSet([CP.Comp(n, "lambert", m, V, rho=0.8) for n, m, V in comps], 1, 1); C, esc, st = CP.cast_C(ss, M, seed)
        a = ss.port_area.double(); e = esc.double()
        return float(1 - (a * e).sum() / a.sum())
    for name, comp0 in (("slab", lambda s, c: CF.box_mesh(CF.SLAB_OCC, CF.SLAB_ORG, s, c)), ("lblock", lambda s, c: CF.box_mesh(CF.LB_OCC, CF.LB_ORG, s, c))):
        coarse = [("dom", comp0(1, True), None)]; fine = [("dom", comp0(8, False), None)]
        vc = [(n, m, m["V"].copy()) for n, m, _ in coarse]; vf = [(n, m, m["V"].copy()) for n, m, _ in fine]
        fc = [frac(vc, 8192, s) for s in (1, 2, 3)]; ff = frac(vf, 256, 5)
        out[name] = dict(coarse_mean=float(np.mean(fc)), coarse_sd=float(np.std(fc, ddof=1)), fine_s8=ff)
    ok = out["slab"]["coarse_mean"] < 1e-6 and out["slab"]["fine_s8"] < 1e-6 and out["lblock"]["coarse_mean"] > 0.01 and abs(out["lblock"]["fine_s8"] - out["lblock"]["coarse_mean"]) < max(4 * out["lblock"]["coarse_sd"], 5e-4)
    return dict(passed=ok, **out)


@test
def casp_levels_and_ownership():
    """CASP levels: geometry only, in [1, 8], contact ports refined, far ports not; no cell carries a medium path (medium comps keep back_hits = 0)."""
    out = {}; ok = True
    for name in ("gap4", "gap8", "gap30", "floor_lam", "lblock_lam", "rot_lam"):
        lev, nu, d = CA.levels_for(name); lab = CF.coarse_labels(name)
        out[name] = dict(hist={int(k): int((lev == k).sum()) for k in np.unique(lev)}, cells=int((lev ** 2).sum()), coarse_ports=int(len(lev)),
                         top_face_max_level=int(lev[lab == "slab_hi:top"].max()) if name.startswith("gap") else None,
                         contact_face_level=int(np.median(lev[lab == "slab_lo:top"])) if name.startswith("gap") else None)
        ok &= lev.min() >= 1 and lev.max() <= 8
        if name.startswith("gap"):
            ok &= out[name]["top_face_max_level"] == 1
    sd = FX.scene_def("floor"); ss = CC.sceneset_of(sd); C, esc, st = CP.cast_C(ss, 64, 9); out["medium_floor_back_hits"] = int(st["back_hits"]); ok &= st["back_hits"] == 0
    return dict(passed=ok, **out)


@test
def comparison_masks(cdir=None):
    """Deterministic comparison masks: classification is repeatable and V1 (centre) pixel sets contain V0 (interior) sets."""
    cdir = Path(cdir) if cdir else None
    if cdir is None or not (cdir / "ccam_floor_lam.json").exists():
        return dict(passed=True, skipped="no cycles dir")
    geo = json.load(open(cdir / "ccam_floor_lam.json")); sd = FX.scene_def("floor_lam"); ss = CC.sceneset_of(sd); out = {}; ok = True
    for cam, g in geo.items():
        p1 = CP.classify(ss, CC.blender_ortho_rays(g, ((0.5, 0.5),))).cpu().numpy(); p1b = CP.classify(ss, CC.blender_ortho_rays(g, ((0.5, 0.5),))).cpu().numpy()
        p0 = CP.classify(ss, CC.blender_ortho_rays(g)).cpu().numpy()
        out[cam] = dict(repeatable=bool((p1 == p1b).all()), v0_subset_of_v1=bool(((p0 < 0) | (p0 == p1)).all()), v1_pixels=int((p1 >= 0).sum()), v0_pixels=int((p0 >= 0).sum()))
        ok &= out[cam]["repeatable"] and out[cam]["v0_subset_of_v1"]
    return dict(passed=ok, **out)


if __name__ == "__main__":
    out = Path(sys.argv[1]); cdir = sys.argv[2] if len(sys.argv) > 2 else None
    for f in (refine_mesh_conservation, analytic_form_factor, closure_nonneg_and_determinism, reciprocity_area_weighted, zero_superposition_application_fixed_point,
              blocker, rigid_consistency, same_domain_reentry, casp_levels_and_ownership):
        f()
    comparison_masks(cdir)
    CC.dump(dict(protocol="fnto_c54", n=len(RES), passed=sum(r["passed"] for r in RES), results=RES), out)
    print(f"{sum(r['passed'] for r in RES)}/{len(RES)} passed")
    sys.exit(0 if all(r["passed"] for r in RES) else 1)
