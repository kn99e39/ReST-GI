"""FNTO boundary ports (surface anchors) of the two optical domains, built from the meshes (full coverage).

    python fnto_anchors.py            # (8dna26 .venv-cu128: numpy + scipy) -> results/fnto_lego/anchors.npz

Unlike SPT-1 (observed T0 surface cells only), ports cover every mesh surface: Burley SSS in the reference enters
anywhere on the Lego object (fnto_semantics), including surfaces no training camera sees.

Domain A (official Lego, one Blender object, 4.2 M triangles, 1,872 components):
  * vertices welded at 1e-6 m; a component is closed if every edge has exactly two faces; closed components are oriented
    outward (positive signed volume), open components are two-sided (a port on each side);
  * area-proportional surface samples (>= 1 per triangle, one per (H_VOX/4)^2) are clustered by (voxel H_VOX, dominant signed normal axis, material): one port
    per cluster with the area-weighted mean position and normal, its area and its (pure) material.
Domain B (box): an analytic H_VOX grid on its six faces (canonical = T0 placement).
Per port: a random tangent-frame rotation (gather directions = cosine pattern in that frame), the N_LOCAL nearest ports
of the same domain within R_LOCAL (any material/orientation, itself included) for the band intrinsic kernel, and the N_MID
nearest coarse source ports (H_MID clusters) within R_MID for the mid-range intrinsic kernel.
Hit mapping tables: per triangle orientation sign, two-sidedness and material; the sorted cluster keys (a hit maps to the
cluster of the hit point on the triangle side facing the ray origin; area-proportional samples cover large triangles).
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.spatial import cKDTree

import fnto_common as C


def weld(V, F, tol=1e-6):
    q = np.round(V / tol).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    return inv.ravel()[F]


def orient(V, F, comp):
    """(sign per triangle so that sign * winding-normal is outward, two-sided flag per triangle)."""
    Fw = weld(V, F)
    e = np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]])
    e.sort(1)
    tri = np.tile(np.arange(len(F)), 3)
    key = e[:, 0] * (Fw.max() + 1) + e[:, 1]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    bad = cnt[inv] != 2                                        # boundary or non-manifold edge
    open_tri = np.zeros(len(F), bool); np.logical_or.at(open_tri, tri[bad], True)
    ncomp = comp.max() + 1
    open_comp = np.zeros(ncomp, bool); np.logical_or.at(open_comp, comp[open_tri], True)
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0
    cvol = np.bincount(comp, vol, ncomp)
    sign = np.where(cvol[comp] < 0, -1.0, 1.0)
    two = open_comp[comp]
    return sign, two, {"components": int(ncomp), "closed": int((~open_comp).sum()), "open": int(open_comp.sum()),
                       "closed_negative_volume": int(((cvol < 0) & ~open_comp).sum())}


def axis_key(n):
    ax = np.argmax(np.abs(n), 1)
    sg = (n[np.arange(len(n)), ax] > 0).astype(np.int64)
    return ax * 2 + sg


def cluster_key(p, n, mat):
    ijk = np.floor(p / C.H_VOX).astype(np.int64) + 4096
    return (((ijk[:, 0] * 8192 + ijk[:, 1]) * 8192 + ijk[:, 2]) * 6 + axis_key(n)) * 16 + mat


def build_A():
    m = np.load(C.MESH_A)
    V, F, mat, comp = m["V"].astype(np.float64), m["F"], m["material"], m["comp"]
    sign, two, rec = orient(V, F, comp)
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(v1 - v0, v2 - v0); area = 0.5 * np.linalg.norm(cr, axis=1)
    ok = area > 0
    nw = np.zeros_like(cr); nw[ok] = cr[ok] / (2 * area[ok, None])
    n = nw * sign[:, None]
    idx = np.flatnonzero(ok)
    # area-proportional samples (>= 1 per triangle, ~ (H/4)^2 per sample) so that large triangles span all their clusters
    ns = np.maximum(1, np.ceil(area[idx] / (C.H_VOX / 4) ** 2)).astype(np.int64)
    tri = np.repeat(idx, ns)
    rng = np.random.default_rng(0)
    r1, r2 = rng.random(len(tri)), rng.random(len(tri))
    sq = np.sqrt(r1)
    pts = (1 - sq)[:, None] * v0[tri] + (sq * (1 - r2))[:, None] * v1[tri] + (sq * r2)[:, None] * v2[tri]
    pts[np.cumsum(ns) - ns] = ((v0 + v1 + v2) / 3)[idx]                     # first sample of a triangle = its centroid
    sa = area[tri] / ns[np.searchsorted(idx, tri)]
    P = [pts]; N = [n[tri]]; A = [sa]; M = [mat[tri]]
    tw = two[tri]
    P.append(pts[tw]); N.append(-n[tri][tw]); A.append(sa[tw]); M.append(mat[tri][tw])
    P, N, A, M = (np.concatenate(x) for x in (P, N, A, M))
    key = cluster_key(P, N, M)
    u, inv = np.unique(key, return_inverse=True)
    w = np.bincount(inv, A, len(u))
    pos = np.stack([np.bincount(inv, A * P[:, j], len(u)) for j in range(3)], 1) / w[:, None]
    nrm = np.stack([np.bincount(inv, A * N[:, j], len(u)) for j in range(3)], 1)
    nn = np.linalg.norm(nrm, axis=1); bad = nn < 1e-8
    nrm[bad] = N[np.unique(inv, return_index=True)[1]][bad]
    del P, N, A, M
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    matu = (u % 16).astype(np.int8)
    rec.update({"triangles": int(len(F)), "two_sided_triangles": int(two.sum()), "ports": int(len(u)), "area": float(area.sum()),
                "area_two_sided_extra": float(area[two].sum()), "surface_samples": int(len(tri) + tw.sum())})
    tri = {"tri_sign": sign.astype(np.int8), "tri_two": two, "tri_mat": mat.astype(np.int8)}
    return pos, nrm, w, matu, u, tri, rec


def build_B():
    bV = np.load(C.BOX_T0)["V"]; lo, hi = bV.min(0), bV.max(0)
    P, N, A, K = [], [], [], []
    for ax in range(3):
        u_, v_ = [i for i in range(3) if i != ax]
        nu = int(np.ceil((hi[u_] - lo[u_]) / C.H_VOX)); nv = int(np.ceil((hi[v_] - lo[v_]) / C.H_VOX))
        eu = np.linspace(lo[u_], hi[u_], nu + 1); ev = np.linspace(lo[v_], hi[v_], nv + 1)
        cu = 0.5 * (eu[1:] + eu[:-1]); cv = 0.5 * (ev[1:] + ev[:-1])
        U, W = np.meshgrid(cu, cv, indexing="ij")
        for side, val in ((-1, lo[ax]), (1, hi[ax])):
            q = np.zeros((U.size, 3)); q[:, u_] = U.ravel(); q[:, v_] = W.ravel(); q[:, ax] = val
            nn = np.zeros((U.size, 3)); nn[:, ax] = side
            P.append(q); N.append(nn); A.append(np.full(U.size, (eu[1] - eu[0]) * (ev[1] - ev[0])))
            K.append({"axis": ax, "side": side, "nu": nu, "nv": nv, "lo_u": float(lo[u_]), "lo_v": float(lo[v_]),
                      "du": float(eu[1] - eu[0]), "dv": float(ev[1] - ev[0]), "start": int(sum(len(p) for p in P[:-1]))})
    return np.concatenate(P), np.concatenate(N), np.concatenate(A), K, {"lo": lo.tolist(), "hi": hi.tolist()}


def coarse_ports(pos, nrm, area, mat, obj):
    """Mid-range source ports: per domain, clusters of fine ports by (H_MID voxel, dominant signed normal axis, material).
    Returns (fine -> coarse index, coarse pos, nrm, area, mat, obj) and per fine port its N_MID nearest coarse ports of the
    same domain within R_MID (-1 padded)."""
    ijk = np.floor(pos / C.H_MID).astype(np.int64) + 1024
    key = ((((ijk[:, 0] * 2048 + ijk[:, 1]) * 2048 + ijk[:, 2]) * 6 + axis_key(nrm)) * 16 + mat.astype(np.int64)) * 2 + obj.astype(np.int64)
    u, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel(); a = np.bincount(inv, area, len(u))
    cp = np.stack([np.bincount(inv, area * pos[:, j], len(u)) for j in range(3)], 1) / a[:, None]
    cn = np.stack([np.bincount(inv, area * nrm[:, j], len(u)) for j in range(3)], 1)
    cn /= np.maximum(np.linalg.norm(cn, axis=1, keepdims=True), 1e-12)
    cm = (u // 2 % 16).astype(np.int8); co = (u % 2).astype(np.int8)
    mid = np.full((len(pos), C.N_MID), -1, np.int32)
    for o in np.unique(obj):
        g = np.flatnonzero(co == o); f = np.flatnonzero(obj == o)
        k = min(C.N_MID, len(g))
        d, i = cKDTree(cp[g]).query(pos[f], k=k, distance_upper_bound=C.R_MID, workers=-1)
        d = d.reshape(len(f), k); i = i.reshape(len(f), k)
        mid[f, :k] = np.where(np.isfinite(d), g[np.minimum(i, len(g) - 1)], -1)
    return inv.astype(np.int32), cp.astype(np.float32), cn.astype(np.float32), a.astype(np.float32), cm, co, mid


def neighbours(pos):
    tree = cKDTree(pos)
    d, i = tree.query(pos, k=C.N_LOCAL, distance_upper_bound=C.R_LOCAL, workers=-1)
    i = np.where(np.isfinite(d), i, -1).astype(np.int32)
    return i, np.where(np.isfinite(d), d, -1).astype(np.float32)


def main():
    t0 = time.time()
    pA, nA, aA, mA, keyA, tri, recA = build_A()
    pB, nB, aB, faces, boxrec = build_B()
    nbA, dA = neighbours(pA); nbB, dB = neighbours(pB)
    NA, NB = len(pA), len(pB)
    rot = np.random.default_rng(0).uniform(0, 2 * np.pi, NA + NB)
    P_, N_, A_ = np.concatenate([pA, pB]), np.concatenate([nA, nB]), np.concatenate([aA, aB])
    M_, O_ = np.concatenate([mA, np.full(NB, C.MAT_BOX, np.int8)]), np.concatenate([np.zeros(NA, np.int8), np.ones(NB, np.int8)])
    f2c, cpos, cnrm, carea, cmat, cobj, mid = coarse_ports(P_, N_, A_, M_, O_)
    C.OUT.mkdir(parents=True, exist_ok=True)
    np.savez(C.OUT / "anchors.npz",
             pos=np.concatenate([pA, pB]).astype(np.float32), nrm=np.concatenate([nA, nB]).astype(np.float32),
             area=np.concatenate([aA, aB]).astype(np.float32), mat=np.concatenate([mA, np.full(NB, C.MAT_BOX, np.int8)]),
             obj=np.concatenate([np.zeros(NA, np.int8), np.ones(NB, np.int8)]), rot=rot.astype(np.float32),
             nbr=np.concatenate([nbA, np.where(nbB >= 0, nbB + NA, -1)]).astype(np.int32), nbr_d=np.concatenate([dA, dB]),
             keyA=keyA, n_A=NA, n_B=NB, f2c=f2c, cpos=cpos, cnrm=cnrm, carea=carea, cmat=cmat, cobj=cobj, mid=mid, **tri)
    rec = {"A": recA, "B": {"ports": NB, "faces": faces, **boxrec}, "h_vox": C.H_VOX, "n_local": C.N_LOCAL, "r_local": C.R_LOCAL,
           "neighbours_used_mean_A": float((nbA >= 0).sum(1).mean()), "area_by_material_A": {C.MATERIALS[k]: float(aA[mA == k].sum()) for k in range(8)},
           "ports_by_material_A": {C.MATERIALS[k]: int((mA == k).sum()) for k in range(8)},
           "coarse_ports": int(len(cpos)), "mid_neighbours_used_mean": float((mid >= 0).sum(1).mean()), "seconds": time.time() - t0}
    (C.OUT / "anchors.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
