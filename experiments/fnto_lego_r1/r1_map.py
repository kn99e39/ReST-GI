"""WL42 r1 surface identity -> port mapping (PC, Mitsuba).

A point on a Lego triangle (a gather hit, or the surface a pixel shows) maps to the r1 port whose cluster key it belongs
to: (1.2 cm tangential voxel, 3 mm plane bin along the dominant signed axis, material, connected component), using the
triangle's oriented normal on the side facing the observer. Pixels are identified by tracing from 1 cm on the camera side
of the pixel's AOV position back along the camera direction (the triangle Cycles shaded). Missing keys (a hit point in a
cluster cell that no surface sample populated) fall back to the nearest port of the same material and component.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
import r1_common as C
import fnto_relations as R
from r1_anchors import plane_key

_FB = {}


def lookup(A, p, n, mat, comp):
    k1, k2 = plane_key(p, n, mat, comp)
    comb = A["keys"][:, 0].astype(np.int64) * 65536 + A["keys"][:, 1].astype(np.int64)   # k2 < 2^16: lexicographic order kept
    q = k1.astype(np.int64) * 65536 + k2.astype(np.int64)
    j = np.clip(np.searchsorted(comb, q), 0, len(comb) - 1)
    port = np.where(comb[j] == q, j, -1).astype(np.int64)
    miss = port < 0
    if miss.any():
        nA = int(A["n_A"])
        for (mt, cp) in set(zip(mat[miss].tolist(), comp[miss].tolist())):
            key = (mt, cp)
            if key not in _FB:
                g = np.flatnonzero((A["mat"][:nA] == mt) & (A["comp"][:nA] == cp))
                _FB[key] = (cKDTree(A["pos"][g]) if len(g) else None, g)
            tree, g = _FB[key]
            sel = np.flatnonzero(miss & (mat == mt) & (comp == cp))
            if tree is None:
                continue
            _, ii = tree.query(p[sel], k=1)
            port[sel] = g[ii]
    return port


def hits_to_ports(A, prim, p, d):
    """Points p on Lego triangles prim, seen along direction d (ray direction) -> r1 port (side facing the ray origin)."""
    V, F = R.mesh_a()
    cr = np.cross(V[F[prim, 1]] - V[F[prim, 0]], V[F[prim, 2]] - V[F[prim, 0]]) * A["tri_sign"][prim][:, None]
    cr /= np.maximum(np.linalg.norm(cr, axis=1, keepdims=True), 1e-20)
    back = (cr * -d).sum(1) < 0
    n = np.where((A["tri_two"][prim] & back)[:, None], -cr, cr)
    return lookup(A, p, n, A["tri_mat"][prim].astype(np.int64), A["tri_comp"][prim].astype(np.int64)), back & ~A["tri_two"][prim]


def pixels_to_ports(A, geo, pos, cam):
    """Pixel AOV position + camera direction -> (port, traced?) via the triangle the camera ray hits near the pixel."""
    o = pos + 0.01 * cam; d = -cam
    valid, hp, prim, sid, t = geo.intersect(o, d)
    ok = valid & (sid == geo.names.index("A")) & (np.abs(t - 0.01) < 0.005)
    port = np.full(len(pos), -1, np.int64)
    if ok.any():
        port[ok], _ = hits_to_ports(A, prim[ok], hp[ok], d[ok])
    return port, ok
