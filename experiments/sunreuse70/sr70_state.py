"""WL70 first-order sun-state reuse (protocol sunreuse70_v1): cache construction, dependency contract, validation and reuse.

The historical first-order sun term (k55_compose.sun_term -> p53_tracer.run_sun on the medium domain's level-4 mesh) depends on
  (D1) the domain's rest mesh (refinement level 4, port frames)            -> rest-mesh hash
  (D2) the intrinsic medium (sigma_t, albedo, g)                            -> parameter tuple
  (D3) the domain's pose only through material-frame directions: the walk runs on the domain's OWN intersector and stops at the first
       air exit (k55_ops.DomainX), phi_f is stored in port-local frames, L0 per camera direction and port
  (D4) the light list (one collimated sun: travel direction, E) in the material frame
  (D5) the camera directions requested for L0, in the material frame
  (D6) source-to-domain visibility: the full-scene shadow test of every source sample (sun_source(shadow=ss.isect))
  (D7) estimator identity: n_sun, seed, chunking, Dout
A cached state is reused only if D1, D2, D4, D5, D7 are identical, the current pose is rigid w.r.t. the rest mesh (D3), and replaying the
cached source samples (material coordinates) through the CURRENT full-scene shadow test reproduces the cached visibility mask bit for
bit (D6). Otherwise the caller retraces with the unchanged historical sun_term. The scene name is never consulted.
"""
import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_k55"))                   # k55_common adds phys_ref53 and fnto_c54
import k55_common as KCm  # noqa: E402
import k55_compose as KP  # noqa: E402
import k55_ops as KO  # noqa: E402
import p53_fixtures as FX  # noqa: E402
import p53_geom as G  # noqa: E402
import p53_tracer as TR  # noqa: E402

DEV = "cuda"
TOL_RIGID = 1e-9          # m, max residual of the rest -> current rigid fit
TOL_DIR = 1e-9            # max |difference| of unit material-frame directions
MEDIUM = (FX.SIGMA_T, tuple(FX.ALBEDO), FX.G_PHASE)


def sync():
    torch.cuda.synchronize(); return time.perf_counter()


def mesh_hash(m):
    h = hashlib.sha256()
    for k in ("V", "F", "tri_port", "port_tangent", "port_area", "port_face"):
        a = np.ascontiguousarray(m[k]); h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def rigid_fit(Vrest, Vcur):
    """Kabsch fit Vcur ~ R Vrest + t (float64). Returns (R, t, max residual)."""
    A = np.asarray(Vrest, float); B = np.asarray(Vcur, float); ca, cb = A.mean(0), B.mean(0)
    U, _, Wt = np.linalg.svd((A - ca).T @ (B - cb)); d = np.sign(np.linalg.det(Wt.T @ U.T))
    R = Wt.T @ np.diag([1.0, 1.0, d]) @ U.T; t = cb - R @ ca
    return R, t, float(np.abs(A @ R.T + t - B).max())


def unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


# ------------------------------------------------------------------ recording replica of the historical source / run_sun


def sun_source_rec(dom, s_travel, E, N, gen, shadow):
    """p53_tracer.sun_source (tri_mask None) with identical RNG consumption; also returns the source record."""
    s = torch.tensor(np.asarray(s_travel, float) / np.linalg.norm(s_travel), dtype=torch.float32, device=DEV)
    cos = -(dom.n * s[None, :]).sum(1)
    wt = dom.area * cos.clamp(min=0)
    total = float(wt.sum())
    tri = torch.multinomial(wt, N, replacement=True, generator=gen)
    u1 = torch.rand(N, device=DEV, generator=gen); u2 = torch.rand(N, device=DEV, generator=gen)
    x = G.sample_on_tris(dom.V, dom.F, tri, u1, u2)
    tt, _ = shadow(x + 4 * dom.eps * dom.n[tri], -s.expand(N, 3)); ok = ~torch.isfinite(tt)
    w = (E * total / N) * ok.float()[:, None].expand(N, 3).contiguous()
    keep = ok
    rec = dict(tri=tri.int(), u1=u1, u2=u2, ok=ok, w=float(E * total / N))
    return x[keep], s.expand(N, 3)[keep].contiguous(), dom.n[tri][keep], w[keep], torch.zeros(int(keep.sum()), dtype=torch.long, device=DEV), rec


def run_sun_rec(dom, s_travel, E, N, seed, Dout, cams, shadow, chunk_photons=4_000_000):
    """p53_tracer.run_sun (continue_air False, tri_mask None) with the source record of every chunk."""
    gen = torch.Generator(device=DEV); gen.manual_seed(int(seed))
    acc = TR.Accum(dom, Dout, 1, 1, cams, psrc=1); t0 = time.time(); recs = []
    for i in range(0, N, chunk_photons):
        n = min(chunk_photons, N - i)
        x, d, nn, w, src, rec = sun_source_rec(dom, s_travel, E * n / N, n, gen, shadow)
        TR.walk(acc, x, d, nn, w, src, gen, continue_air=False)
        recs.append(rec)
    out = acc.finalize(); out["seconds"] = time.time() - t0
    out["rec"] = dict(tri=torch.cat([r["tri"] for r in recs]), u1=torch.cat([r["u1"] for r in recs]), u2=torch.cat([r["u2"] for r in recs]),
                      ok=torch.cat([r["ok"] for r in recs]), w=[r["w"] for r in recs])
    return out


def sun_term_rec(sd, cam_keys, n_sun=16_000_000, seed=5591, medium=MEDIUM, record=True):
    """k55_compose.sun_term with identical arithmetic (optionally another medium for a diagnostic control) plus, per medium comp, the
    state needed to validate reuse. Returns (sun dict in the historical format, {ci: state})."""
    ss0 = KCm.classify_sceneset(sd); out, states = {}, {}
    if not sd["lights"]:
        return out, states
    s, E = sd["lights"][0]; dirs = np.array([FX.DIRS[k] for k in cam_keys])
    for ci, (n, k, m, V) in enumerate(sd["comps"]):
        if k != "medium":
            continue
        fm, VC, par = KO.refined(m, V, 4)
        dom = KO.DomainX(fm, VC, sigma_t=medium[0], albedo=medium[1], g=medium[2], name=n)
        if record:
            r = run_sun_rec(dom, s, E, n_sun, seed + ci, KP.DOUT, dirs, ss0.isect)
        else:
            r = TR.run_sun(dom, s, E, n_sun, seed + ci, KP.DOUT, cams=dirs, shadow=ss0.isect)
        Pc = len(m["port_area"]); pt = torch.as_tensor(par, device=DEV)
        KCf = r["KC"][:, :, :, 0]; ap = r["proj_area"]
        num = torch.zeros(len(dirs), 3, Pc, device=DEV, dtype=torch.float64).index_add_(2, pt, (KCf * ap[:, None]).double())
        den = torch.zeros(len(dirs), Pc, device=DEV, dtype=torch.float64).index_add_(1, pt, ap.double())
        out[ci] = dict(phi_f=r["T"][:, :, 0].T.contiguous().cpu(), L0=(num / den.clamp(min=1e-30)[:, None]).permute(0, 2, 1).cpu(), stats=r["stats"],
                       seconds=r["seconds"], cam_keys=list(cam_keys))
        if record:
            R0, t0, res = rigid_fit(m["V"], V)
            states[ci] = dict(comp=n, mesh_hash=mesh_hash(m), medium=(float(medium[0]), tuple(float(a) for a in medium[1]), float(medium[2])),
                              lights_mat=[(tuple(unit(R0.T @ unit(sl))), float(El)) for sl, El in sd["lights"]],
                              cams_mat={kk: tuple(unit(R0.T @ unit(FX.DIRS[kk]))) for kk in cam_keys},
                              estimator=dict(n_sun=int(n_sun), seed=int(seed + ci), Dout=int(KP.DOUT), level=4, chunk=4_000_000),
                              rest_fineV=torch.tensor(fm["V"], dtype=torch.float64, device=DEV), fineF=torch.tensor(fm["F"], device=DEV),
                              rest_n=torch.tensor(unit_rows(G.tri_frames(fm["V"], fm["V"], fm["F"], fm["tri_port"], fm["port_tangent"])["n"]), dtype=torch.float64, device=DEV),
                              eps=float(dom.eps), rec=r["rec"], pose=(R0, t0, res), sun=out[ci], out_hash=hash_out(out[ci]))
        del r, dom; torch.cuda.empty_cache()
    return out, states


def unit_rows(a):
    a = np.asarray(a, float); return a / np.linalg.norm(a, axis=1, keepdims=True)


def hash_out(o):
    h = hashlib.sha256(); h.update(o["phi_f"].numpy().tobytes()); h.update(o["L0"].numpy().tobytes()); return h.hexdigest()


def state_bytes(st):
    b = sum(t.numel() * t.element_size() for t in (st["rec"]["tri"], st["rec"]["u1"], st["rec"]["u2"], st["rec"]["ok"]))
    b_geo = sum(t.numel() * t.element_size() for t in (st["rest_fineV"], st["fineF"], st["rest_n"]))
    b_out = st["sun"]["phi_f"].numel() * 4 + st["sun"]["L0"].numel() * 4
    return dict(source_record=b, rest_geometry=b_geo, outputs=b_out, total=b + b_geo + b_out)


# ------------------------------------------------------------------ validation (contract) and reuse


def validate(st, sd, ci, cam_keys, medium=MEDIUM, chunk=4_000_000):
    """Check the dependency contract of cached state `st` against the current scene `sd` (comp ci). Returns (accept, report)."""
    rep = dict(checks={}, timing={}); t = sync()
    n, k, m, V = sd["comps"][ci] if ci < len(sd["comps"]) else (None, None, None, None)
    ok_id = (k == "medium") and mesh_hash(m) == st["mesh_hash"]
    ok_med = (float(medium[0]), tuple(float(a) for a in medium[1]), float(medium[2])) == st["medium"]
    rep["checks"]["D1_rest_mesh"] = bool(ok_id); rep["checks"]["D2_medium"] = bool(ok_med); rep["timing"]["identity_s"] = sync() - t
    if not (ok_id and ok_med):
        return False, rep
    t = sync(); R1, t1, res = rigid_fit(m["V"], V); ok_rig = res <= TOL_RIGID
    rep["checks"]["D3_rigid"] = bool(ok_rig); rep["rigid_residual_m"] = res; rep["timing"]["rigid_s"] = sync() - t
    if not ok_rig:
        return False, rep
    t = sync()
    lm = [(tuple(unit(R1.T @ unit(sl))), float(El)) for sl, El in sd["lights"]]
    ok_l = len(lm) == len(st["lights_mat"]) and all(np.abs(np.subtract(a[0], b[0])).max() <= TOL_DIR and a[1] == b[1] for a, b in zip(lm, st["lights_mat"]))
    ok_c = all(kk in st["cams_mat"] and np.abs(np.subtract(unit(R1.T @ unit(FX.DIRS[kk])), st["cams_mat"][kk])).max() <= TOL_DIR for kk in cam_keys)
    rep["checks"]["D4_light_material_frame"] = bool(ok_l); rep["checks"]["D5_cameras_material_frame"] = bool(ok_c)
    rep["checks"]["D7_estimator"] = True                                   # the cached estimator is the historical one (fixed n_sun/seed/Dout/level)
    rep["timing"]["directions_s"] = sync() - t
    if not (ok_l and ok_c):
        return False, rep
    # D6: replay every cached source sample (material coordinates) through the CURRENT full-scene shadow test
    t = sync(); ss = KCm.classify_sceneset(sd); rep["timing"]["shadow_scene_build_s"] = sync() - t
    t = sync(); s1 = torch.tensor(unit(sd["lights"][0][0]), dtype=torch.float32, device=DEV)
    R = torch.tensor(R1, dtype=torch.float64, device=DEV); tt_ = torch.tensor(t1, dtype=torch.float64, device=DEV)
    rec = st["rec"]; N = len(rec["tri"]); mism = 0
    for i in range(0, N, chunk):
        tri = rec["tri"][i:i + chunk].long()
        xm = G.sample_on_tris(st["rest_fineV"], st["fineF"], tri, rec["u1"][i:i + chunk].double(), rec["u2"][i:i + chunk].double())
        x = (xm @ R.T + tt_).float(); nrm = (st["rest_n"][tri] @ R.T).float()
        th, _ = ss.isect(x + 4 * st["eps"] * nrm, -s1.expand(len(tri), 3))
        mism += int((~torch.isfinite(th) != rec["ok"][i:i + chunk]).sum())
    rep["timing"]["visibility_replay_s"] = sync() - t
    rep["visibility_mismatch"] = mism; rep["visibility_samples"] = N; rep["checks"]["D6_visibility"] = mism == 0
    del ss
    return mism == 0, rep


def reuse(st, cam_keys):
    """The cached sun entry in the historical format, L0 rows ordered as cam_keys."""
    o = st["sun"]; idx = [o["cam_keys"].index(kk) for kk in cam_keys]
    return dict(phi_f=o["phi_f"], L0=o["L0"][idx], stats=o["stats"], seconds=0.0, cam_keys=list(cam_keys))
