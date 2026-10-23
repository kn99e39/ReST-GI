"""FNTO r1 boundary ports (worklog 42; corrects audit finding #1 and #4). 8dna26 .venv-cu128 (numpy, scipy).

    python r1_anchors.py          -> results/fnto_lego_r1/anchors.npz, anchors.json

Ports of domain A (official Lego):
  * area-proportional surface samples (>= 1 per triangle, one per (H_VOX/4)^2), oriented outward on closed components,
    duplicated on both sides of open components (as Batch 6);
  * cluster key = (1.2 cm tangential voxel, 3 mm plane bin along the dominant signed normal axis, material, connected
    component). Batch 6 keyed only (voxel, axis, material), which merged a visible face with a parallel hidden sheet a
    few mm below it, so the area-weighted centroid sank under the visible surface (audit #1);
  * port ORIGIN = the medoid surface sample of its cluster (a real surface point with its own triangle normal), plus up
    to S_PORT surface samples for visibility averaging and gather origins. Rays start EPS1 = 0.1 mm off a real surface.
Domain B (box): analytic grid as Batch 6, S_PORT samples per cell.
Kernel support (audit #4): for every port, ALL same-domain band-source ports (H_BAND1 = 1.5 cm aggregates of fine
ports, area-mean incident) within R_LOCAL and ALL coarse ports (H_MID1 = 8 cm) within R_MID are found; they are split into K_BAND / K_MID distance strata of equal count and one seeded representative
per stratum carries the stratum's total area (represented-area weight), so the quadrature covers the declared radius.
Coarse ports: H_MID1 = 8 cm clusters of fine ports (voxel, signed axis, material).
"""

from __future__ import annotations

import json
import time

import numpy as np
from scipy.spatial import cKDTree

import r1_common as C
import fnto_anchors as A0                      # Batch 6 module, reused read-only (orient, axis_key)


def plane_key(p, n, mat, comp):
    ax = np.argmax(np.abs(n), 1); sg = (n[np.arange(len(n)), ax] > 0).astype(np.int64)
    ijk = np.floor(p / C.H_VOX).astype(np.int64) + 4096
    pb = np.floor(p[np.arange(len(p)), ax] / C.H_PLANE).astype(np.int64) + 1000
    ijk[np.arange(len(p)), ax] = pb
    k1 = ((ijk[:, 0] * 8192 + ijk[:, 1]) * 8192 + ijk[:, 2]) * 6 + ax * 2 + sg
    k2 = mat.astype(np.int64) * 4096 + comp.astype(np.int64)
    return k1, k2


def build_A(rng):
    m = np.load(C.MESH_A)
    V, F, mat, comp = m["V"].astype(np.float64), m["F"], m["material"], m["comp"]
    sign, two, rec = A0.orient(V, F, comp)
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(v1 - v0, v2 - v0); area = 0.5 * np.linalg.norm(cr, axis=1); ok = area > 0
    nw = np.zeros_like(cr); nw[ok] = cr[ok] / (2 * area[ok, None]); n = nw * sign[:, None]
    idx = np.flatnonzero(ok)
    ns = np.maximum(1, np.ceil(area[idx] / (C.H_VOX / 4) ** 2)).astype(np.int64)
    tri = np.repeat(idx, ns)
    r1, r2 = rng.random(len(tri)), rng.random(len(tri)); sq = np.sqrt(r1)
    pts = (1 - sq)[:, None] * v0[tri] + (sq * (1 - r2))[:, None] * v1[tri] + (sq * r2)[:, None] * v2[tri]
    sa = area[tri] / np.repeat(ns, ns)
    tw = two[tri]
    P = np.concatenate([pts, pts[tw]]); N = np.concatenate([n[tri], -n[tri][tw]]); A = np.concatenate([sa, sa[tw]])
    T = np.concatenate([tri, tri[tw]]); Sd = np.concatenate([np.ones(len(tri), np.int8), -np.ones(int(tw.sum()), np.int8)])
    M = mat[T]; Cp = comp[T]
    k1, k2 = plane_key(P, N, M, Cp)
    keys = np.stack([k1, k2], 1)
    u, inv = np.unique(keys, axis=0, return_inverse=True); inv = inv.ravel()
    nc = len(u)
    w = np.bincount(inv, A, nc)
    mean = np.stack([np.bincount(inv, A * P[:, j], nc) for j in range(3)], 1) / w[:, None]
    nm = np.stack([np.bincount(inv, A * N[:, j], nc) for j in range(3)], 1)
    nm /= np.maximum(np.linalg.norm(nm, axis=1, keepdims=True), 1e-12)
    # medoid (on-surface origin) and S_PORT samples per cluster
    order = np.argsort(inv, kind="stable"); invs = inv[order]
    start = np.searchsorted(invs, np.arange(nc)); cnt = np.bincount(inv, minlength=nc)
    d2 = ((P[order] - mean[invs]) ** 2).sum(1)
    big = np.full(nc, np.inf); np.minimum.at(big, invs, d2)
    is_med = d2 <= big[invs] + 1e-18
    first_med = np.full(nc, -1, np.int64); pos_in = np.arange(len(order))
    cand = pos_in[is_med]; first_med[invs[cand][::-1]] = cand[::-1]
    samp = np.full((nc, C.S_PORT), -1, np.int64)
    samp[:, 0] = order[first_med]
    for k in range(1, C.S_PORT):
        off = (rng.random(nc) * cnt).astype(np.int64)
        samp[:, k] = np.where(cnt > k, order[start + off], -1)
    samp[:, 1:] = np.where(samp[:, 1:] == samp[:, :1], -1, samp[:, 1:])
    sp = np.where(samp[..., None] >= 0, P[np.clip(samp, 0, None)], np.nan).astype(np.float32)
    sn = np.where(samp[..., None] >= 0, N[np.clip(samp, 0, None)], np.nan).astype(np.float32)
    origin = P[samp[:, 0]]; onrm = N[samp[:, 0]]
    matu = M[order[first_med]].astype(np.int8); compu = Cp[order[first_med]].astype(np.int32)
    rec.update({"ports": int(nc), "surface_samples": int(len(P)), "area": float(area.sum()),
                "ports_with_full_S": float((samp >= 0).all(1).mean())})
    return dict(pos=origin, nrm=onrm, cnrm=nm, area=w, mat=matu, comp=compu, spos=sp, snrm=sn, keys=u), \
        dict(tri_sign=sign.astype(np.int8), tri_two=two, tri_mat=mat.astype(np.int8), tri_comp=comp.astype(np.int32)), rec


def build_B(rng):
    bV = np.load(C.BOX_T0)["V"]; lo, hi = bV.min(0), bV.max(0)
    P, N, A, SP, faces = [], [], [], [], []
    for ax in range(3):
        u_, v_ = [i for i in range(3) if i != ax]
        nu = int(np.ceil((hi[u_] - lo[u_]) / C.H_VOX)); nv = int(np.ceil((hi[v_] - lo[v_]) / C.H_VOX))
        eu = np.linspace(lo[u_], hi[u_], nu + 1); ev = np.linspace(lo[v_], hi[v_], nv + 1)
        cu = 0.5 * (eu[1:] + eu[:-1]); cv = 0.5 * (ev[1:] + ev[:-1])
        U, W = np.meshgrid(cu, cv, indexing="ij")
        for side, val in ((-1, lo[ax]), (1, hi[ax])):
            q = np.zeros((U.size, 3)); q[:, u_] = U.ravel(); q[:, v_] = W.ravel(); q[:, ax] = val
            nn = np.zeros((U.size, 3)); nn[:, ax] = side
            jit = (rng.random((U.size, C.S_PORT, 2)) - 0.5)
            s = np.repeat(q[:, None], C.S_PORT, 1).copy(); s[..., u_] += jit[..., 0] * (eu[1] - eu[0]); s[..., v_] += jit[..., 1] * (ev[1] - ev[0])
            s[:, 0] = q
            faces.append({"axis": ax, "side": side, "nu": nu, "nv": nv, "lo_u": float(lo[u_]), "lo_v": float(lo[v_]),
                          "du": float(eu[1] - eu[0]), "dv": float(ev[1] - ev[0]), "start": int(sum(len(p) for p in P))})
            P.append(q); N.append(nn); A.append(np.full(U.size, (eu[1] - eu[0]) * (ev[1] - ev[0]))); SP.append(s)
    P = np.concatenate(P); N = np.concatenate(N)
    return P, N, np.concatenate(A), np.concatenate(SP), faces


def stratified(query_pos, src_pos, src_area, R, K, rng, self_offset=None):
    """For each query point: all sources within R, split into K equal-count distance strata; one representative per
    stratum (seeded) carrying the stratum's total area. Returns (rep [Q, K] int32, Arep [Q, K] float32, n_in_ball [Q],
    reach [Q] = max distance among the ball members)."""
    tree = cKDTree(src_pos)
    Q = len(query_pos)
    rep = np.full((Q, K), -1, np.int32); Arep = np.zeros((Q, K), np.float32); nball = np.zeros(Q, np.int32); reach = np.zeros(Q, np.float32)
    kq = 256
    todo = np.arange(Q)
    while len(todo):
        d, i = tree.query(query_pos[todo], k=min(kq, len(src_pos)), distance_upper_bound=R, workers=-1)
        d = d.reshape(len(todo), -1); i = i.reshape(len(todo), -1)
        full = np.isfinite(d[:, -1]) & (kq < len(src_pos))           # cap reached before R: re-query with more
        done = todo[~full]; dd = d[~full]; ii = i[~full]
        cnt = np.isfinite(dd).sum(1)
        nball[done] = cnt; reach[done] = np.where(cnt > 0, dd[np.arange(len(dd)), np.maximum(cnt - 1, 0)], 0)
        for c in np.unique(cnt):
            sel = cnt == c
            if c == 0:
                continue
            q_ = done[sel]; ii_ = ii[sel][:, :c]
            if c <= K:
                rep[q_, :c] = ii_; Arep[q_, :c] = src_area[ii_]
            else:
                edges = np.linspace(0, c, K + 1).astype(np.int64)
                a = src_area[ii_]; csum = np.concatenate([np.zeros((len(q_), 1)), np.cumsum(a, 1)], 1)
                pick = edges[:-1][None] + (rng.random((len(q_), K)) * (edges[1:] - edges[:-1])[None]).astype(np.int64)
                rep[q_] = np.take_along_axis(ii_, pick, 1)
                Arep[q_] = (csum[:, edges[1:]] - csum[:, edges[:-1]]).astype(np.float32)
        todo = todo[full]; kq *= 4
    return rep, Arep, nball, reach


def coarse(pos, nrm, area, mat, obj, h=None):
    h = C.H_MID1 if h is None else h
    ijk = np.floor(pos / h).astype(np.int64) + 1024
    key = ((((ijk[:, 0] * 2048 + ijk[:, 1]) * 2048 + ijk[:, 2]) * 6 + A0.axis_key(nrm)) * 16 + mat.astype(np.int64)) * 2 + obj.astype(np.int64)
    u, inv = np.unique(key, return_inverse=True); inv = inv.ravel(); a = np.bincount(inv, area, len(u))
    cp = np.stack([np.bincount(inv, area * pos[:, j], len(u)) for j in range(3)], 1) / a[:, None]
    cn = np.stack([np.bincount(inv, area * nrm[:, j], len(u)) for j in range(3)], 1); cn /= np.maximum(np.linalg.norm(cn, axis=1, keepdims=True), 1e-12)
    return inv.astype(np.int32), cp.astype(np.float32), cn.astype(np.float32), a.astype(np.float32), (u // 2 % 16).astype(np.int8), (u % 2).astype(np.int8)


def main():
    t0 = time.time(); rng = np.random.default_rng(0)
    a, tri, recA = build_A(rng)
    pB, nB, aB, spB, faces = build_B(rng)
    NA, NB = len(a["pos"]), len(pB)
    pos = np.concatenate([a["pos"], pB]); nrm = np.concatenate([a["nrm"], nB]); area = np.concatenate([a["area"], aB])
    mat = np.concatenate([a["mat"], np.full(NB, C.MAT_BOX, np.int8)]); obj = np.concatenate([np.zeros(NA, np.int8), np.ones(NB, np.int8)])
    spos = np.concatenate([a["spos"], spB.astype(np.float32)]); snrm = np.concatenate([a["snrm"], np.repeat(nB[:, None], C.S_PORT, 1).astype(np.float32)])
    rot = rng.uniform(0, 2 * np.pi, NA + NB)
    # band representatives (fine ports, same domain), mid representatives (coarse ports, same domain)
    f2c, cpos, cnrm, carea, cmat, cobj = coarse(pos, nrm, area, mat, obj)
    f2b, bpos, bnrm, barea, bmat, bobj = coarse(pos, nrm, area, mat, obj, C.H_BAND1)       # band source aggregates
    nbr = np.full((NA + NB, C.K_BAND_A0), -1, np.int32); nbrA = np.zeros((NA + NB, C.K_BAND_A0), np.float32)
    mid = np.full((NA + NB, C.K_MID_A0), -1, np.int32); midA = np.zeros((NA + NB, C.K_MID_A0), np.float32)
    sup = {}
    for o in (0, 1):
        g = np.flatnonzero(obj == o); gc = np.flatnonzero(cobj == o); gb = np.flatnonzero(bobj == o)
        r_, A_, n_, reach = stratified(pos[g], bpos[gb], barea[gb], C.R_LOCAL, C.K_BAND_A0, rng)
        nbr[g] = np.where(r_ >= 0, gb[np.clip(r_, 0, None)], -1); nbrA[g] = A_
        rm, Am, nm_, reachm = stratified(pos[g], cpos[gc], carea[gc], C.R_MID, C.K_MID_A0, rng)
        mid[g] = np.where(rm >= 0, gc[np.clip(rm, 0, None)], -1); midA[g] = Am
        sup[o] = {"band_ball_count_q10_q50_q90": np.quantile(n_, [0.1, 0.5, 0.9]).tolist(), "band_reach_q10_q50_q90_m": np.quantile(reach, [0.1, 0.5, 0.9]).tolist(),
                  "mid_ball_count_q10_q50_q90": np.quantile(nm_, [0.1, 0.5, 0.9]).tolist(), "mid_reach_q10_q50_q90_m": np.quantile(reachm, [0.1, 0.5, 0.9]).tolist(),
                  "band_represented_area_over_ball_area_q50": float(np.median(nbrA[g].sum(1) / (np.pi * C.R_LOCAL ** 2))),
                  "mid_represented_area_over_ball_area_q50": float(np.median(midA[g].sum(1) / (np.pi * C.R_MID ** 2)))}
    C.OUT1.mkdir(parents=True, exist_ok=True)
    np.savez(C.OUT1 / "anchors.npz", pos=pos.astype(np.float32), nrm=nrm.astype(np.float32), area=area.astype(np.float32), mat=mat, obj=obj,
             comp=np.concatenate([a["comp"], np.full(NB, -1, np.int32)]), spos=spos, snrm=snrm, rot=rot.astype(np.float32),
             nbr=nbr, nbrA=nbrA, mid=mid, midA=midA, f2c=f2c, cpos=cpos, cnrm=cnrm, carea=carea, cmat=cmat, cobj=cobj,
             f2b=f2b, bpos=bpos, bnrm=bnrm, barea=barea, bmat=bmat, bobj=bobj,
             keys=a["keys"], n_A=NA, n_B=NB, **tri)
    rec = {"A": recA, "B": {"ports": NB, "faces": faces}, "coarse_ports": int(len(cpos)), "band_source_ports": int(len(bpos)), "support": sup,
           "h_vox": C.H_VOX, "h_plane": C.H_PLANE, "s_port": C.S_PORT, "seconds": time.time() - t0}
    (C.OUT1 / "anchors.json").write_text(json.dumps(rec, indent=1)); print(json.dumps({k: v for k, v in rec.items() if k != "B"}, indent=1))


if __name__ == "__main__":
    main()
