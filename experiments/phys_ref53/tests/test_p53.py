"""WL53 focused tests (protocol phys_ref53). Run on the server:
    P53_MI_VARIANT=cuda_ad_rgb python tests/test_p53.py <out json>
The radiometric-unit test renders with llvm_ad_rgb in a subprocess (renders never use LabServer63's cuda film)."""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("P53_MI_VARIANT", "cuda_ad_rgb")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import p53_compose as CP  # noqa: E402
import p53_fixtures as FX  # noqa: E402
import p53_geom as G  # noqa: E402
import p53_tracer as TR  # noqa: E402

DEV = "cuda"
RESULTS = {}


def test(fn):
    def run():
        t0 = time.time()
        try:
            info = fn() or {}
            RESULTS[fn.__name__] = dict(ok=True, seconds=round(time.time() - t0, 2), **info)
        except AssertionError as e:
            RESULTS[fn.__name__] = dict(ok=False, error=str(e))
        print(fn.__name__, RESULTS[fn.__name__], flush=True)
    run.__name__ = fn.__name__
    return run


def ff_parallel(a, b, c):
    X, Y = a / c, b / c
    return 2 / (math.pi * X * Y) * (math.log(math.sqrt((1 + X * X) * (1 + Y * Y) / (1 + X * X + Y * Y)))
                                    + X * math.sqrt(1 + Y * Y) * math.atan(X / math.sqrt(1 + Y * Y)) + Y * math.sqrt(1 + X * X) * math.atan(Y / math.sqrt(1 + X * X))
                                    - X * math.atan(X) - Y * math.atan(Y))


@test
def test_radiometric_units():
    """Lambert plane under a sun: Mitsuba (llvm) render and the composition readout both give rho E cos / pi."""
    code = r'''
import os, sys, math, numpy as np; os.environ["P53_MI_VARIANT"]="llvm_ad_rgb"; sys.path.insert(0, sys.argv[1])
import p53_mi as M, p53_fixtures as FX
fl = FX.floor_mesh(0.0, 10); s = (math.sin(math.radians(30)), 0.0, -math.cos(math.radians(30)))
sc = M.scene({"f": M.lambert_shape(fl["V"], fl["F"], 0.8)}, [M.sun(s, 1.0)], {"c": M.ortho((0,0,0.1),(0,0,0),(0,1,0),0.05,0.05,16,16)})
print(float(M.render(sc, "c", 64, 1).mean())); sys.stdout.flush(); os._exit(0)
'''
    out = subprocess.run([sys.executable, "-c", code, str(HERE.parent)], capture_output=True, text=True, env=dict(os.environ, P53_MI_VARIANT="llvm_ad_rgb"))
    L_mi = float(out.stdout.strip().splitlines()[-1])
    exp = 0.8 * 1.0 * math.cos(math.radians(30)) / math.pi
    fl = FX.floor_mesh(0.0, 10); ss = CP.SceneSet([CP.Comp("f", "lambert", fl, fl["V"], rho=0.8)])
    Ep, phi = CP.sun_external(ss, (math.sin(math.radians(30)), 0.0, -math.cos(math.radians(30))), 1.0, 16, 0)
    sol = dict(phi_out=torch.zeros(ss.Ptot * ss.Dout, 3, device=DEV), phi_in_ind=torch.zeros(ss.Ptot * ss.Din, 3, device=DEV))
    L = CP.readout(ss, (0, 0, 1), sol, None, E_ext=Ep, mode="bin")
    assert abs(L_mi / exp - 1) < 1e-4, (L_mi, exp)
    assert torch.allclose(L, torch.full_like(L, exp), rtol=1e-5), (L.min(), L.max(), exp)
    assert abs(float(phi.sum()) - math.cos(math.radians(30)) * 0.01) < 1e-7                      # flux = E cos A (10 x 10 cm)
    return dict(L_mitsuba=L_mi, analytic=exp)


@test
def test_coordinate_transforms():
    u = np.random.rand(10000); v = np.random.rand(10000)
    x, y = G.square_to_disk(u, v); u2, v2 = G.disk_to_square(x, y)
    assert np.abs(u - u2).max() < 1e-12 and np.abs(v - v2).max() < 1e-12
    r = np.sqrt(np.random.rand(2 * 10 ** 6)); ph = 2 * np.pi * np.random.rand(2 * 10 ** 6)
    d = np.stack([r * np.cos(ph), r * np.sin(ph), np.sqrt(1 - r * r)], 1)
    occ = np.bincount(G.local_to_bin(d, 8), minlength=64) / len(d) * 64
    assert np.abs(occ - 1).max() < 0.02, occ                                                       # equal projected solid angle
    m = FX.lblock_mesh(); R = G.rot((1, 2, 3), 37.0); Vr = G.map_rigid(m["V"], R, (3.0, -1.0, 0.5))
    f0 = G.tri_frames(m["V"], m["V"], m["F"], m["tri_port"], m["port_tangent"]); f1 = G.tri_frames(m["V"], Vr, m["F"], m["tri_port"], m["port_tangent"])
    for k in ("n", "t", "b"):
        assert np.abs(f0[k] @ R.T - f1[k]).max() < 1e-12, k                                     # frames rotate with the body
    w = np.random.randn(1000, 3); w /= np.linalg.norm(w, axis=1, keepdims=True); i = np.random.randint(0, len(m["F"]), 1000)
    l0 = G.to_local(w, f0["n"][i], f0["t"][i], f0["b"][i]); l1 = G.to_local(w @ R.T, f1["n"][i], f1["t"][i], f1["b"][i])
    assert np.abs(l0 - l1).max() < 1e-12                                                          # local directions invariant
    fb = G.tri_frames(FX.slab_mesh()["V"], G.map_bend(FX.slab_mesh()["V"], 90, 0.1), FX.slab_mesh()["F"], FX.slab_mesh()["tri_port"], FX.slab_mesh()["port_tangent"])
    assert np.abs(np.einsum("ij,ij->i", fb["n"], fb["t"])).max() < 1e-12 and np.abs(np.linalg.norm(fb["b"], axis=1) - 1).max() < 1e-12


@test
def test_boundary_correspondence():
    m = FX.slab_mesh(); P = len(m["port_area"])
    a_rest = G.port_areas(G.tri_frames(m["V"], m["V"], m["F"], m["tri_port"], m["port_tangent"])["area"], m["tri_port"], P)
    assert np.allclose(a_rest, m["port_area"])
    Vb = G.map_bend(m["V"], 90.0, 0.1); a_b = G.port_areas(G.tri_frames(m["V"], Vb, m["F"], m["tri_port"], m["port_tangent"])["area"], m["tri_port"], P)
    R = 0.1 / (math.pi / 2); top = m["port_normal"][:, 2] > 0.5; bot = m["port_normal"][:, 2] < -0.5
    rt = a_b[top] / a_rest[top]; rb = a_b[bot] / a_rest[bot]
    assert abs(rt.mean() - (R - 0.005) / R) < 2e-3 and abs(rb.mean() - (R + 0.005) / R) < 2e-3, (rt.mean(), rb.mean())   # chordal tessellation
    Vd = G.map_dent(m["V"], 0.005, 0.02, 0.01); assert np.all(Vd[:, 2] >= m["V"][:, 2] - 0.005 - 1e-12)
    # vertices shared by ports keep their ids; voxel faces are welded (closed manifold)
    E = np.sort(np.concatenate([m["F"][:, [0, 1]], m["F"][:, [1, 2]], m["F"][:, [2, 0]]]), 1); _, c = np.unique(E, axis=0, return_counts=True)
    assert np.all(c == 2)
    return dict(top_ratio=float(rt.mean()), bottom_ratio=float(rb.mean()), analytic=((R - 0.005) / R, (R + 0.005) / R))


@test
def test_zero_and_superposition():
    m, V = FX.domain_geometry("slab"); dom = TR.Domain(m, V, FX.SIGMA_T, FX.ALBEDO)
    r0 = TR.run_sun(dom, (0, 0, -1), 0.0, 20000, 1, FX.DOUT)
    assert float(r0["T"].abs().sum()) == 0.0
    Ta = TR.run_cells(dom, FX.DOUT, FX.DIN, 8, 2, cells=torch.arange(0, 64, device=DEV))["T"]
    g = torch.Generator(device=DEV); g.manual_seed(0)
    a = torch.rand(Ta.shape[2], device=DEV, generator=g); b = torch.rand(Ta.shape[2], device=DEV, generator=g)
    lhs = Ta[0] @ (2 * a + 3 * b); rhs = 2 * (Ta[0] @ a) + 3 * (Ta[0] @ b)
    assert torch.allclose(lhs, rhs, rtol=1e-4, atol=1e-7)
    ss = CP.SceneSet([CP.Comp("s", "medium", m, V)]); C, _, _ = CP.cast_C(ss, 4, 0)
    z = CP.solve(ss, {0: Ta.new_zeros(3, ss.Ptot * ss.Dout, ss.Ptot * ss.Din)}, C, torch.zeros(ss.Ptot * ss.Dout, 3, device=DEV))
    assert float(z["phi_out"].abs().sum()) == 0.0


@test
def test_conservation_and_reciprocity():
    m, V = FX.domain_geometry("lblock"); dom = TR.Domain(m, V, FX.SIGMA_T, (1.0, 1.0, 1.0))
    r = TR.run_cells(dom, FX.DOUT, FX.DIN, 16, 3)
    st = r["stats"]; cs = r["T"].sum(1)
    assert st["leaks"] == 0 and st["wrong_side"] == 0 and st["truncated"] == 0, st
    assert float((cs - 1).abs().max()) < 1e-5, float((cs - 1).abs().max())                         # albedo 1: every photon exits
    # C: two directly opposed 10 x 10 cm Lambert plates 2 cm apart, analytic form factor
    up = FX.floor_mesh(0.0, 10); dn = FX.floor_mesh(0.02, 10); dn["F"] = dn["F"][:, ::-1].copy(); dn["port_normal"] = -dn["port_normal"]
    ss = CP.SceneSet([CP.Comp("a", "lambert", up, up["V"], rho=0.5), CP.Comp("b", "lambert", dn, dn["V"], rho=0.5)])
    C, esc, cst = CP.cast_C(ss, 256, 4)
    uni = torch.ones(ss.Ptot * ss.Dout, device=DEV); uni[ss.comps[0].P * ss.Dout:] = 0                   # uniform radiance from plate a
    w = (ss.port_area.repeat_interleave(ss.Dout) * math.pi / ss.Dout) * uni                              # flux of uniform unit radiance
    got = float(torch.sparse.mm(C, w[:, None]).sum() / w.sum()); exp = ff_parallel(0.1, 0.1, 0.02)
    assert abs(got / exp - 1) < 0.01 and cst["back_hits"] == 0, (got, exp, cst)
    # reciprocity of C: A_i F_ij = A_j F_ji at port level (ports of a vs b), against the cross-seed noise of the same statistic
    A = ss.port_area; Pa = ss.comps[0].P

    def F_of(Cx):
        Cd = Cx.to_dense().view(ss.Ptot, ss.Din, ss.Ptot, ss.Dout).sum((1, 3)) / ss.Dout             # Cd[j, i]: fraction of port i's uniform exitance reaching j
        return Cd[Pa:, :Pa] * A[None, :Pa], (Cd[:Pa, Pa:] * A[None, Pa:]).T
    Fab, Fba = F_of(C); Fab2, _ = F_of(CP.cast_C(ss, 256, 44)[0])
    asym = float((Fab - Fba).abs().sum() / Fab.sum()); noise = float((Fab - Fab2).abs().sum() / Fab.sum())
    assert asym < 1.25 * noise + 0.005, (asym, noise)
    return dict(C_form_factor=got, analytic=exp, C_port_reciprocity_asym=asym, C_cross_seed_noise=noise, alpha1_colsum_maxdev=float((cs - 1).abs().max()))


@test
def test_optical_domain_ownership():
    """TR stops at the first exit: a concave domain's same-domain air re-entry is NOT in K; C carries it."""
    m, V = FX.domain_geometry("lblock"); dom = TR.Domain(m, V, FX.SIGMA_T, FX.ALBEDO)
    ss = CP.SceneSet([CP.Comp("L", "medium", m, V)]); C, esc, _ = CP.cast_C(ss, 32, 5)
    reentry = float(1 - esc.mean()); assert reentry > 0.01, reentry
    m2, V2 = FX.domain_geometry("slab"); ss2 = CP.SceneSet([CP.Comp("s", "medium", m2, V2)]); C2, esc2, _ = CP.cast_C(ss2, 32, 5)
    assert float(1 - esc2.mean()) == 0.0                                                          # convex: no same-domain air path
    return dict(lblock_reentry=reentry)


@test
def test_air_gap_separation():
    """Medium-partition K of one slab has no transfer to the other slab; the object-scoped control (continue_air) does."""
    m, V = FX.domain_geometry("pair4"); P = len(m["port_area"]); half = P // 2
    dom_obj = TR.Domain(m, V, FX.SIGMA_T, FX.ALBEDO)
    cells = torch.arange(0, half * FX.DIN, 7, device=DEV)                                             # sources on the lower slab
    r_obj = TR.run_cells(dom_obj, FX.DOUT, FX.DIN, 64, 6, cells=cells, continue_air=True)
    Tp = r_obj["T"].view(3, P, FX.DOUT, P * FX.DIN)[:, :, :, cells].sum((2, 3))
    cross_obj = float(Tp[:, half:].sum() / Tp.sum())
    ms, Vs = FX.slab_mesh(), FX.slab_mesh()["V"] - [0, 0, 0.007]
    dom_part = TR.Domain(ms, Vs, FX.SIGMA_T, FX.ALBEDO)                                              # isolated: other slab does not exist
    r_p = TR.run_cells(dom_part, FX.DOUT, FX.DIN, 64, 6, cells=cells)
    assert r_p["stats"]["leaks"] == 0 and cross_obj > 0.005, (cross_obj, r_p["stats"])
    return dict(object_scoped_cross_fraction=cross_obj)


@test
def test_path_history_independence():
    """The composition applies the same K at every order: solve() == explicit Neumann series with one operator."""
    m, V = FX.domain_geometry("lblock"); fl = FX.floor_mesh(-0.02, 14)
    ss = CP.SceneSet([CP.Comp("L", "medium", m, V), CP.Comp("f", "lambert", fl, fl["V"], rho=0.8)])
    dom = TR.Domain(m, V, FX.SIGMA_T, FX.ALBEDO)
    T = TR.run_cells(dom, FX.DOUT, FX.DIN, 4, 7)["T"]
    C, _, _ = CP.cast_C(ss, 8, 8)
    g = torch.Generator(device=DEV); g.manual_seed(1); phi0 = torch.rand(ss.Ptot * ss.Dout, 3, device=DEV, generator=g)
    sol = CP.solve(ss, {0: T}, C, phi0, tol=1e-9)
    term = phi0.clone(); acc = phi0.clone()
    for _ in range(sol["iters"] + 50):
        term = CP.apply_K(ss, {0: T}, CP.apply_C(C, term)); acc += term
    rel = float((acc - sol["phi_out"]).abs().sum() / acc.abs().sum())
    assert sol["converged"] and rel < 1e-5, (rel, sol["iters"])
    return dict(neumann_vs_solve=rel, iters=sol["iters"])


@test
def test_oracle_composition():
    """Two-plate Lambert system: composed first bounce = rho F E_1 (form factor) and linear in the source."""
    up = FX.floor_mesh(0.0, 10); dn = FX.floor_mesh(0.02, 10); dn["F"] = dn["F"][:, ::-1].copy(); dn["port_normal"] = -dn["port_normal"]
    ss = CP.SceneSet([CP.Comp("a", "lambert", up, up["V"], rho=0.5), CP.Comp("b", "lambert", dn, dn["V"], rho=0.5)])
    C, _, _ = CP.cast_C(ss, 256, 9)
    phi0 = torch.zeros(ss.Ptot * ss.Dout, 3, device=DEV); Pa = ss.comps[0].P
    phi0[:Pa * ss.Dout] = (ss.port_area[:Pa] * 1.0 / ss.Dout).repeat_interleave(ss.Dout)[:, None]      # plate a: exitance 1 W/m^2
    sol = CP.solve(ss, {}, C, phi0)
    Eb = float(sol["phi_in_ind"][Pa * ss.Din:, 0].sum() / ss.port_area[Pa:].sum())
    first = float(CP.apply_C(C, phi0)[Pa * ss.Din:, 0].sum() / ss.port_area[Pa:].sum())
    F = ff_parallel(0.1, 0.1, 0.02)
    assert abs(first / F - 1) < 0.01, (first, F)
    sol2 = CP.solve(ss, {}, C, 2 * phi0)
    assert torch.allclose(sol2["phi_out"], 2 * sol["phi_out"], rtol=1e-5, atol=1e-9)
    return dict(first_bounce=first, form_factor=F, all_bounce_E_b=Eb, iters=sol["iters"])


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "test_p53.json"
    for fn in (test_radiometric_units, test_coordinate_transforms, test_boundary_correspondence, test_zero_and_superposition,
               test_conservation_and_reciprocity, test_optical_domain_ownership, test_air_gap_separation, test_path_history_independence,
               test_oracle_composition):
        fn()
    n_ok = sum(r["ok"] for r in RESULTS.values())
    json.dump(dict(passed=n_ok, total=len(RESULTS), results=RESULTS), open(out, "w"), indent=1)
    print(f"{n_ok}/{len(RESULTS)} passed", flush=True)
    sys.stdout.flush(); os._exit(0 if n_ok == len(RESULTS) else 1)
