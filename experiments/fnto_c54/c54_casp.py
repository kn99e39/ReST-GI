"""WL54 candidate CASP: contact-adaptive sub-port explicit coupling (protocol fnto_c54, section candidate_CASP).

One mechanism: conservative adaptive refinement of the explicit-C unknowns. A coarse 1 cm port J is subdivided into l(J) x l(J)
cells (one unknown per cell) where its near-field transport is spatially resolved only by cells smaller than the distance to the
surfaces it exchanges light with. The level rule is geometry only:

    cast 256 cosine-weighted rays from uniform positions of port J;  nu(J) = fraction with first-hit distance t < W h  (W = 4)
    nu(J) < nu_min (0.05)  ->  l(J) = 1
    else d(J) = median{t : t < W h},  l(J) = clamp(2^ceil(log2(c h / d(J))), 1, l_max)

Refined meshes are built from the port's current grid (bilinear in its (g+1)^2 grid points, g = 4 for WL53 voxel ports, 1 for the
floor), so rigid and non-rigid maps of the coarse mesh carry through. The ray caster generalises p53_compose.cast_C to a per-port
ray count (weights 1 / count): still an unbiased estimator of the cell-to-cell flux fraction, with exact closure (every ray hits a
front face or escapes).
"""
import math
import time

import numpy as np
import torch

import c54_common as CC
import c54_fixtures as CF
import c54_lam as LM
import p53_compose as CP
import p53_geom as G

DEV = "cuda"
W_NEAR, NU_MIN, NPRE = 4.0, 0.05, 256


# ------------------------------------------------------------------ port grids and refinement


def port_grids(mesh, Vcur):
    """Rest and current grid points of every port, (P, g+1, g+1, 3), recovered from the (p, q) cell / two-triangle ordering that
    voxel_boundary and open_grid use."""
    P = len(mesh["port_area"]); F = mesh["F"]; tp = mesh["tri_port"]
    order = np.argsort(tp, kind="stable"); k = len(F) // P
    assert k * P == len(F) and np.all(tp[order].reshape(P, k) == np.arange(P)[:, None]), "ports must have equal triangle counts"
    tri = F[order].reshape(P, k // 2, 2, 3); g = int(round(math.sqrt(k // 2))); assert g * g == k // 2
    ids = np.zeros((P, g + 1, g + 1), np.int64)
    for p in range(g):
        for q in range(g):
            t0, t1 = tri[:, p * g + q, 0], tri[:, p * g + q, 1]
            ids[:, p, q] = t0[:, 0]; ids[:, p + 1, q] = t0[:, 1]; ids[:, p + 1, q + 1] = t0[:, 2]; ids[:, p, q + 1] = t1[:, 2]
    return mesh["V"][ids], np.asarray(Vcur)[ids], g


def _tri_affine(Gr, u, v):
    """Point of the TRIANGULATED port surface at parameter (u, v) in [0, 1]^2. Gr (n, g+1, g+1, 3). Each grid cell is the two
    triangles (P00, P10, P11) for u >= v and (P00, P11, P01) for u <= v (the voxel_boundary / open_grid tessellation), so the
    returned points lie exactly on the surface the reference renders (bilinear interpolation would not for non-planar cells)."""
    g = Gr.shape[1] - 1; n = len(u); r = np.arange(n)
    gu, gv = np.clip(u * g, 0, g), np.clip(v * g, 0, g)
    i, j = np.minimum(gu.astype(int), g - 1), np.minimum(gv.astype(int), g - 1); fu, fv = (gu - i)[:, None], (gv - j)[:, None]
    P00, P10, P11, P01 = Gr[r, i, j], Gr[r, i + 1, j], Gr[r, i + 1, j + 1], Gr[r, i, j + 1]
    return np.where(fu >= fv, P00 + fu * (P10 - P00) + fv * (P11 - P10), P00 + fu * (P11 - P01) + fv * (P01 - P00))


def refine_mesh(mesh, Vcur, levels):
    """Subdivide port k into levels[k] x levels[k] cells (one unknown each). The surface is reproduced EXACTLY: triangles are cut on a
    grid of L = max(level, g) cells per edge by triangle-wise affine interpolation, and a cell owns the triangles inside it (2 (L / l)^2
    of them). Returns (fine mesh dict, fine Vcur, cell -> local coarse port (int array)). Unknown id = cell; the rest geometry is
    refined in the same way as the current geometry."""
    levels = np.asarray(levels, int); GR, GC, g = port_grids(mesh, Vcur); P = len(levels)
    ncell_port = levels ** 2; cell_off = np.concatenate([[0], np.cumsum(ncell_port)[:-1]]); ncell = int(ncell_port.sum())
    parent = np.repeat(np.arange(P), ncell_port)
    VR, VC, tri_cell = [], [], []
    for l in np.unique(levels):
        ports = np.nonzero(levels == l)[0]; L = max(int(l), g)
        a, b = np.meshgrid(np.arange(L), np.arange(L), indexing="ij"); a, b = a.ravel(), b.ravel()
        pp = np.repeat(ports, L * L); aa = np.tile(a, len(ports)); bb = np.tile(b, len(ports))
        cid = cell_off[pp] + (aa * l // L) * l + (bb * l // L)
        for tri_corners in (((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 1), (0, 1))):
            for Gs, out in ((GR, VR), (GC, VC)):
                out.append(np.stack([_tri_affine(Gs[pp], (aa + ci) / L, (bb + cj) / L) for ci, cj in tri_corners], 1))     # (n, 3 corners, 3)
            tri_cell.append(cid)
    VRa = np.concatenate(VR).reshape(-1, 3); VCa = np.concatenate(VC).reshape(-1, 3); tc = np.concatenate(tri_cell)
    nt = len(tc); F = np.arange(3 * nt).reshape(nt, 3)
    fm = dict(V=VRa, F=F, tri_port=tc, port_tangent=mesh["port_tangent"][parent], port_area=mesh["port_area"][parent] / ncell_port[parent],
              port_face=mesh["port_face"][parent], port_voxel=mesh["port_voxel"][parent], port_center=mesh["port_center"][parent], port_normal=mesh["port_normal"][parent])
    return fm, VCa, parent


# ------------------------------------------------------------------ level assignment


def assign_levels(ss, c=2.0, lmax=8, nu_min=NU_MIN, W=W_NEAR, nrays=NPRE, seed=5801):
    """Geometry-only level per coarse port of SceneSet ss (coarse mesh). Returns (levels (P,), nu (P,), d (P,))."""
    gen = torch.Generator(device=DEV); gen.manual_seed(int(seed)); P = ss.Ptot
    h = torch.sqrt(ss.port_area.double()).float()
    port = torch.arange(P, device=DEV).repeat_interleave(nrays)
    tri, x = ss.sample_ports(port, gen)
    wl = G.bin_sample(torch.zeros(len(port), dtype=torch.long, device=DEV), 1, torch.rand(len(port), device=DEV, generator=gen), torch.rand(len(port), device=DEV, generator=gen)).float()
    d = torch.nn.functional.normalize(G.to_world(wl, ss.n[tri], ss.t[tri], ss.b[tri]), dim=1)
    t, prim = ss.isect(x + ss.eps * ss.n[tri], d)
    t = t.view(P, nrays); near = torch.isfinite(t) & (t < W * h[:, None])
    nu = near.float().mean(1)
    dm = torch.where(near, t, torch.full_like(t, float("nan"))).nanmedian(1).values
    ratio = (c * h / dm).cpu().numpy(); nu_np = nu.cpu().numpy()
    lev = np.ones(P, int); ok = (nu_np >= nu_min) & np.isfinite(ratio)
    lev[ok] = np.clip(2 ** np.ceil(np.log2(np.maximum(ratio[ok], 1.0))), 1, lmax).astype(int)
    return lev, nu_np, dm.cpu().numpy()


# ------------------------------------------------------------------ adaptive caster


def cast_adaptive(ss, m_port, seed, chunk_rays=4_000_000, mask=None):
    """p53_compose.cast_C with a per-port ray count m_port (int array (Ptot,), rays per OUT-CELL of that port). Weight of a ray =
    1 / (rays of its out-cell). Same sparse layout, escape fraction and stats as cast_C."""
    gen = torch.Generator(device=DEV); gen.manual_seed(int(seed))
    nout = ss.Ptot * ss.Dout; rows, cols, vals = [], [], []
    esc = torch.zeros(nout, device=DEV); stats = dict(rays=0, back_hits=0, escapes=0)
    mp = torch.as_tensor(np.asarray(m_port), device=DEV, dtype=torch.long)
    cnt_all = mp[torch.arange(nout, device=DEV) // ss.Dout]; cum = torch.cumsum(cnt_all, 0); start = 0
    while start < nout:
        base = int(cum[start - 1]) if start > 0 else 0
        stop = int(torch.searchsorted(cum, torch.tensor(base + chunk_rays, device=DEV))) + 1; stop = min(max(stop, start + 1), nout)
        cells = torch.arange(start, stop, device=DEV); cnt = cnt_all[start:stop]
        rep = cells.repeat_interleave(cnt); w = (1.0 / cnt.double()).float().repeat_interleave(cnt)
        port = rep // ss.Dout; ob = rep % ss.Dout
        tri, x = ss.sample_ports(port, gen)
        wl = G.bin_sample(ob, ss.Nout, torch.rand(len(rep), device=DEV, generator=gen), torch.rand(len(rep), device=DEV, generator=gen)).float()
        d = torch.nn.functional.normalize(G.to_world(wl, ss.n[tri], ss.t[tri], ss.b[tri]), dim=1)
        t, prim = ss.isect(x + ss.eps * ss.n[tri], d)
        hit = torch.isfinite(t); front = hit.clone(); front[hit] = (d[hit] * ss.n[prim[hit]]).sum(1) < 0
        stats["rays"] += len(rep); stats["back_hits"] += int((hit & ~front).sum()); stats["escapes"] += int((~hit).sum())
        esc.index_add_(0, rep[~hit], w[~hit])
        j = front.nonzero().squeeze(1)
        if mask is not None and len(j):
            j = j[mask(ss.port_comp[port[j]], ss.tri_comp[prim[j]])]
        pr = prim[j]; ib = G.local_to_bin(G.to_local(-d[j], ss.n[pr], ss.t[pr], ss.b[pr]), ss.Nin)
        rows.append(ss.tri_port[pr] * ss.Din + ib); cols.append(rep[j]); vals.append(w[j])
        start = stop
    idx = torch.stack([torch.cat(rows), torch.cat(cols)])
    C = torch.sparse_coo_tensor(idx, torch.cat(vals), (ss.Ptot * ss.Din, nout)).coalesce()
    return C, esc, stats


# ------------------------------------------------------------------ Lambert CASP scene


class LamCASP(LM.Lam):
    """Lambert scene whose unknowns are the CASP cells. levels: per coarse port (global WL53 order)."""

    def __init__(self, name, levels, bins=(1, 1)):
        self.name, self.s, self.bins = name, 0, bins
        coarse = CF.comps_at(name, 1, coarse=True); off = 0; comps, pars = [], []
        for (n, m, V) in coarse:
            P = len(m["port_area"]); fm, Vc, pp = refine_mesh(m, V, levels[off:off + P]); comps.append((n, fm, Vc)); pars.append(pp + off); off += P
        self.levels = np.asarray(levels, int)
        self.ss = CP.SceneSet([CP.Comp(n, "lambert", m, V, rho=LM.RHO) for n, m, V in comps], bins[0], bins[1])
        self.parent = np.concatenate(pars); self.nc = int(self.parent.max()) + 1
        self.area = self.ss.port_area.double().cpu().numpy(); self.sun = np.asarray(CF.light_of(name), float)
        self.C = self.esc = self.stats = self.Ep = None

    def build(self, budget=8192, seed_C=5810, K_ext=256, seed_E=5811):
        """budget: rays per coarse port (production Lambert: 128 rays x 64 out-bins = 8192)."""
        l2 = self.levels[self.parent] ** 2
        if self.bins == (1, 1):
            m = np.maximum(1, np.round(budget / l2)).astype(int)
        else:
            m = np.maximum(1, np.round(budget / self.bins[0] / l2)).astype(int)
        torch.cuda.synchronize(); t0 = time.time()
        self.C, self.esc, self.stats = cast_adaptive(self.ss, m, seed_C)
        torch.cuda.synchronize(); self.t_C = time.time() - t0; t0 = time.time()
        Ep, _ = CP.sun_external(self.ss, self.sun, 1.0, K_ext, seed_E)
        torch.cuda.synchronize(); self.t_ext = time.time() - t0
        self.Ep = Ep.double().cpu().numpy(); self.M, self.K_ext = budget, K_ext
        return self


def levels_for(name, c=2.0, lmax=8, seed=5801, bins=(1, 1)):
    """Geometry-only levels of the coarse WL53 ports of fixture `name`."""
    lam = LM.Lam(name, 1, bins, coarse_mesh=True)
    return assign_levels(lam.ss, c=c, lmax=lmax, seed=seed)
