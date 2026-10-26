"""WL46 MS geometry (Mitsuba, LabServer63 8dna26 venv + mi_env): relation tables, sun fractions and pixel lists of the
specular scene (Lego + box + glossy plane), T0 and T1. Lego / box ports, their gathers (r1_relations.ray_setup) and the hit
-> port mapping (r1_relations.map_hits) are the frozen r1 ones; the plane is an extra analytic domain:
  plane ports  a 2.5 cm grid on the plane (76 x 42), normal -x, global ids N + m
  H            [N, 128] int32: r1 semantics (-1 miss, -2 inside face, >= 0 port) + plane hits N + m
  Hm, dm       [M, 128]: the plane ports' own 128 cosine gathers (origin 0.1 mm off the plane, per-port random rotation),
               hits -> Lego / box ports (r1 mapping)
  sun          Lego / box port sun fractions with worklog 44's 0.3 mm rule (k44_eval_vis.sun_vis_t); plane ports: n . w < 0 -> 0
  pixels       from the AOV renders: alpha > 0 and object index 1 / 2 / 3; Lego -> r1_map.pixels_to_ports, box ->
               map_hits, plane -> grid cell; sun visibility = the renderer's diffuse-direct > 0 (as r1); camera direction
  IS rays      plane pixels: 128 GGX(alpha 0.04) importance samples of the reflection of the camera direction (seed 4646),
               traced from 0.1 mm off the pixel point: hit port (r1 mapping, -1 miss), direction, weight cos / pdf / 128

    source mi_env.sh; <8dna26 python> ms46_prep.py <refs dir> <out dir>
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_lego_k44")); sys.path.insert(0, str(HERE.parent / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parent / "fnto_lego"))
import r1_common as C  # noqa: E402
import fnto_relations as R  # noqa: E402
import r1_relations as RR  # noqa: E402
import r1_map as Mp  # noqa: E402
from k44_eval_vis import sun_vis_t  # noqa: E402

PLANE = {"x": 0.85, "y": (-1.0, 0.9), "z": (-0.25, 0.8)}; CELL = 0.025; ALPHA = 0.04
SUN = np.array([0.2, -0.5, 0.84]) / np.linalg.norm([0.2, -0.5, 0.84]); NP_ = np.array([-1.0, 0, 0])
NY = int(round((PLANE["y"][1] - PLANE["y"][0]) / CELL)); NZ = int(round((PLANE["z"][1] - PLANE["z"][0]) / CELL))


def plane_mesh():
    x = PLANE["x"]; (y0, y1), (z0, z1) = PLANE["y"], PLANE["z"]
    V = np.array([(x, y0, z0), (x, y0, z1), (x, y1, z1), (x, y1, z0)], np.float64); F = np.array([(0, 1, 2), (0, 2, 3)])
    return V, F


def plane_cell(p):
    iy = np.clip(((p[:, 1] - PLANE["y"][0]) / CELL).astype(np.int64), 0, NY - 1); iz = np.clip(((p[:, 2] - PLANE["z"][0]) / CELL).astype(np.int64), 0, NZ - 1)
    return iy * NZ + iz


def geo_of(state):
    oa, ob = C.offsets(state); m = np.load(C.MESH_A); b = np.load(C.BOX_T0); Vp, Fp = plane_mesh()
    return R.Geo(oa, ob, meshes=[("A", m["V"] + oa, m["F"]), ("B", b["V"] + ob, b["F"]), ("P", Vp, Fp)])


def map46(A, geo, valid, p, prim, sid, d, N):
    h = RR.map_hits(A, geo, valid, p, prim, sid, d)
    mp = valid & (sid == geo.names.index("P"))
    if mp.any():
        front = (d[mp] @ NP_) < 0
        h[mp] = np.where(front, N + plane_cell(p[mp]), -2)
    return h


def frames(n, rot):
    a = np.where(np.abs(n[:, 2:3]) < 0.9, [[0.0, 0, 1]], [[1.0, 0, 0]])
    t = np.cross(a, n); t /= np.linalg.norm(t, axis=1, keepdims=True); b = np.cross(n, t)
    c, s = np.cos(rot)[:, None], np.sin(rot)[:, None]
    return c * t + s * b, -s * t + c * b


def ggx_is(n, wo, k, rng):
    u1, u2 = rng.random((len(n), k)), rng.random((len(n), k)); a2 = ALPHA * ALPHA
    ct = np.sqrt((1 - u1) / (1 + (a2 - 1) * u1)); st = np.sqrt(1 - ct * ct); ph = 2 * np.pi * u2
    t, b = frames(n, np.zeros(len(n)))
    hv = st[..., None] * np.cos(ph)[..., None] * t[:, None] + st[..., None] * np.sin(ph)[..., None] * b[:, None] + ct[..., None] * n[:, None]
    hw = (hv * wo[:, None]).sum(-1); wi = 2 * hw[..., None] * hv - wo[:, None]
    D = a2 / (np.pi * (ct * ct * (a2 - 1) + 1) ** 2); pdf = D * ct / (4 * np.abs(hw).clip(1e-12))
    ci = (wi * n[:, None]).sum(-1); w = np.where(ci > 0, ci / pdf.clip(1e-30) / k, 0.0)
    return wi, w


def main(refs, out):
    refs, out = Path(refs), Path(out); out.mkdir(parents=True, exist_ok=True)
    A = RR.load(); N = len(A["obj"]); M = NY * NZ; K = C.K_GATHER
    yc = PLANE["y"][0] + (np.arange(NY) + 0.5) * CELL; zc = PLANE["z"][0] + (np.arange(NZ) + 0.5) * CELL
    pm = np.stack([np.full(M, PLANE["x"]), np.repeat(yc, NZ), np.tile(zc, NY)], 1)
    rec = {"N": N, "M": M, "cell": CELL, "plane": PLANE, "sun": SUN.tolist()}
    for state in ("T0", "T1"):
        geo = geo_of(state); off = RR.offs(A, geo); H = np.full((N, K), -1, np.int32)
        for s in range(0, N, 40000):
            ii = np.arange(s, min(s + 40000, N)); o, d = RR.ray_setup(A, ii); o = o + off[ii][:, None]
            o = o.reshape(-1, 3); d = d.reshape(-1, 3); valid, p, prim, sid, t = geo.intersect(o, d)
            H[ii] = map46(A, geo, valid, p, prim, sid, d, N).reshape(len(ii), -1)
        rng = np.random.default_rng(4647); rot = rng.uniform(0, 2 * np.pi, M)
        t_, b_ = frames(np.repeat(NP_[None], M, 0), rot); Lc = C.cosine_dirs_local()
        dm = Lc[None, :, 0:1] * t_[:, None] + Lc[None, :, 1:2] * b_[:, None] + Lc[None, :, 2:3] * NP_[None, None]
        om = np.repeat((pm + C.EPS1 * NP_)[:, None], K, 1)
        valid, p, prim, sid, t = geo.intersect(om.reshape(-1, 3), dm.reshape(-1, 3))
        Hm = map46(A, geo, valid, p, prim, sid, dm.reshape(-1, 3), N).reshape(M, K).astype(np.int32)
        sun = sun_vis_t(A, geo, SUN).astype(np.float32)
        pix = {}
        for v in range(3):
            z = np.load(refs / f"aov_{state}_v{v}.npz"); obj = np.round(z["obj"]).astype(np.int64); mk = (z["alpha"] > 0) & (obj >= 1)
            P = z["pos"][mk].astype(np.float64); Nn = z["nrm"][mk].astype(np.float64); Nn /= np.linalg.norm(Nn, axis=1, keepdims=True).clip(1e-12)
            Cd = z["cam_pos"][None] - P; Cd /= np.linalg.norm(Cd, axis=1, keepdims=True); Nn = np.where(((Nn * Cd).sum(1) < 0)[:, None], -Nn, Nn)
            ob = obj[mk]; dom = np.where(ob == 1, 0, np.where(ob == 2, 1, 2)); port = np.full(len(P), -1, np.int64)
            la = dom == 0
            if la.any():
                pa, ok = Mp.pixels_to_ports(A, geo, P[la], Cd[la]); port[np.flatnonzero(la)[ok]] = pa[ok]
            lb = dom == 1
            if lb.any():
                port[lb] = RR.map_hits(A, geo, np.ones(int(lb.sum()), bool), P[lb], np.zeros(int(lb.sum()), np.int64), np.full(int(lb.sum()), geo.names.index("B")), -Cd[lb])
            lp = dom == 2; port[lp] = N + plane_cell(P[lp])
            offs = np.stack(C.offsets(state) + (np.zeros(3),)); canon = P - offs[dom]
            mat = np.where(dom == 1, 8, np.where(dom == 2, 9, A["mat"][np.clip(port, 0, N - 1)]))
            vis = (z["diffdir"][mk].sum(1) > 0).astype(np.uint8)
            # plane pixels: GGX importance-sampled rays
            ip = np.flatnonzero(lp); rngp = np.random.default_rng(4646 + 10 * v + (state == "T1"))
            wi, ww = ggx_is(np.repeat(NP_[None], len(ip), 0), Cd[ip], K, rngp)
            oo = np.repeat((P[ip] + C.EPS1 * NP_)[:, None], K, 1)
            vv, pp, pr, sd, tt = geo.intersect(oo.reshape(-1, 3), wi.reshape(-1, 3))
            his = map46(A, geo, vv, pp, pr, sd, wi.reshape(-1, 3), N).reshape(len(ip), K)
            his = np.where(ww > 0, his, -1)
            pix[v] = dict(mk=np.flatnonzero(mk).astype(np.int32), pos=canon.astype(np.float32), nrm=Nn.astype(np.float32), cam=Cd.astype(np.float32),
                          vis=vis, dom=dom.astype(np.int8), port=port.astype(np.int32), valid=(port >= 0).astype(np.uint8), mat=mat.astype(np.int8),
                          is_dir=wi.astype(np.float32), is_w=ww.astype(np.float32), is_hit=his.astype(np.int32))
            rec[f"{state}_v{v}"] = {"pixels": int(mk.sum()), "lego": int(la.sum()), "box": int(lb.sum()), "plane": int(lp.sum()), "valid": float((port >= 0).mean()),
                                    "plane_is_ray_hit_frac": float((his >= 0).mean()), "plane_is_ray_plane_hits": int((his >= N).sum())}
        np.savez_compressed(out / f"geom_{state}.npz", H=H, Hm=Hm, dm=dm.astype(np.float32), pm=pm.astype(np.float32), sun=sun,
                            **{f"v{v}_{k}": a for v, d_ in pix.items() for k, a in d_.items()})
        rec[f"{state}_rays"] = {"lego_box_to_plane_frac": float((H >= N).mean()), "miss": float((H == -1).mean()), "inside": float((H == -2).mean()),
                                "plane_gathers_hit_frac": float((Hm >= 0).mean()), "plane_gathers_miss": float((Hm == -1).mean())}
        print(state, json.dumps({k: v for k, v in rec.items() if k.startswith(state)}), flush=True)
    (out / "ms46_prep.json").write_text(json.dumps(rec, indent=1))
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
