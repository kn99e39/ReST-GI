"""WL53 explicit free-space coupling C, external field, K/C composition, readout and reference port averages
(protocol phys_ref53).

Path ownership (protocol "ownership"): a light path is split at every crossing of a medium / opaque boundary. Every segment
in air (vacuum) belongs to C, and every sub-path that enters a domain and travels inside the medium until the next exit
belongs to K. The two families are disjoint by construction; neither depends on how components are grouped into authoring
objects. C is a pure ray cast: radiance is constant along an air segment, so with equal projected-solid-angle bins a flux
fraction is carried from an out-cell (port, out-bin) to the in-cell (port', in-bin) its rays first hit.

  phi_in  = phi_ext + C phi_out                (phi_ext: unoccluded external light)
  phi_out = K phi_in                           (block-diagonal over domains: TR matrices, Lambert, black)
The external term is applied with the exact light direction (TR sun runs; Lambert uses its exact irradiance); only the
C-coupled terms use binned directions.
"""
import math

import numpy as np
import torch

import p53_geom as G

DEV = "cuda"


class Comp:
    def __init__(self, name, kind, mesh, Vcur, medium=None, rho=None, port_ids=True):
        assert kind in ("medium", "lambert", "black")
        self.name, self.kind, self.mesh, self.Vcur = name, kind, mesh, np.asarray(Vcur, float)
        self.medium = medium; self.rho = None if rho is None else np.broadcast_to(np.asarray(rho, float), (3,)).copy()
        self.port_ids = port_ids
        self.P = len(mesh["port_area"])


class SceneSet:
    def __init__(self, comps, Dout=64, Din=16):
        import p53_mi as M
        self.comps = comps; self.Dout, self.Din = Dout, Din
        self.Nout, self.Nin = int(round(math.sqrt(Dout))), int(round(math.sqrt(Din)))
        V, F, tc, tp, n, t, b, ar = [], [], [], [], [], [], [], []
        self.port_off = []; off_v = 0; off_p = 0
        for ci, c in enumerate(comps):
            fr = G.tri_frames(c.mesh["V"], c.Vcur, c.mesh["F"], c.mesh["tri_port"], c.mesh["port_tangent"])
            V.append(c.Vcur); F.append(c.mesh["F"] + off_v); tc.append(np.full(len(c.mesh["F"]), ci)); tp.append(c.mesh["tri_port"] + off_p)
            n.append(fr["n"]); t.append(fr["t"]); b.append(fr["b"]); ar.append(fr["area"])
            self.port_off.append(off_p); off_v += len(c.Vcur); off_p += c.P
        self.Ptot = off_p
        f32 = lambda a: torch.tensor(np.concatenate(a), dtype=torch.float32, device=DEV)  # noqa: E731
        self.Vnp = np.concatenate(V); self.Fnp = np.concatenate(F)
        self.V = torch.tensor(self.Vnp, dtype=torch.float32, device=DEV); self.F = torch.tensor(self.Fnp, device=DEV)
        self.n, self.t, self.b, self.area = f32(n), f32(t), f32(b), f32(ar)
        self.tri_comp = torch.tensor(np.concatenate(tc), device=DEV); self.tri_port = torch.tensor(np.concatenate(tp), device=DEV)
        self.port_comp = torch.tensor(np.concatenate([np.full(c.P, i) for i, c in enumerate(comps)]), device=DEV)
        self.port_area = torch.zeros(self.Ptot, device=DEV, dtype=torch.float64).index_add_(0, self.tri_port, self.area.double()).float()
        order = np.argsort(self.tri_port.cpu().numpy(), kind="stable"); cnt = np.bincount(self.tri_port.cpu().numpy(), minlength=self.Ptot)
        kmax = cnt.max(); pt = np.full((self.Ptot, kmax), -1); start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        for p in range(self.Ptot):
            pt[p, :cnt[p]] = order[start[p]:start[p] + cnt[p]]
        self.port_tris = torch.tensor(pt, device=DEV)
        a = torch.where(self.port_tris >= 0, self.area[self.port_tris.clamp(min=0)], torch.zeros(1, device=DEV))
        self.port_cum = torch.cumsum(a, 1) / a.sum(1, keepdim=True)
        self.isect = M.Intersector(self.Vnp, self.Fnp)
        self.eps = 2e-6 * max(1.0, float(np.abs(self.Vnp).max()))

    def comp_ports(self, ci):
        return slice(self.port_off[ci], self.port_off[ci] + self.comps[ci].P)

    def sample_ports(self, ports, gen):
        N = len(ports); u = torch.rand(N, device=DEV, generator=gen)
        k = (u[:, None] > self.port_cum[ports]).sum(1).clamp(max=self.port_tris.shape[1] - 1)
        tri = self.port_tris[ports, k]
        x = G.sample_on_tris(self.V, self.F, tri, torch.rand(N, device=DEV, generator=gen), torch.rand(N, device=DEV, generator=gen))
        return tri, x


# ------------------------------------------------------------------ C


def cast_C(ss, M_rays, seed, mask=None, chunk_rays=4_000_000):
    """Sparse C (n_in_cells x n_out_cells), escape fraction per out-cell, error counts.
    mask(src_comp, dst_comp) -> bool tensor: False drops that link (only for declared control arms)."""
    gen = torch.Generator(device=DEV); gen.manual_seed(int(seed))
    nout = ss.Ptot * ss.Dout; rows, cols, vals = [], [], []
    esc = torch.zeros(nout, device=DEV); stats = dict(rays=0, back_hits=0, escapes=0)
    per = max(1, chunk_rays // M_rays)
    cells_all = torch.arange(nout, device=DEV)
    for i in range(0, nout, per):
        cells = cells_all[i:i + per].repeat_interleave(M_rays); port = cells // ss.Dout; ob = cells % ss.Dout
        tri, x = ss.sample_ports(port, gen)
        wl = G.bin_sample(ob, ss.Nout, torch.rand(len(cells), device=DEV, generator=gen), torch.rand(len(cells), device=DEV, generator=gen)).float()
        d = torch.nn.functional.normalize(G.to_world(wl, ss.n[tri], ss.t[tri], ss.b[tri]), dim=1)
        t, prim = ss.isect(x + ss.eps * ss.n[tri], d)
        hit = torch.isfinite(t)
        front = hit.clone(); front[hit] = (d[hit] * ss.n[prim[hit]]).sum(1) < 0
        stats["rays"] += len(cells); stats["back_hits"] += int((hit & ~front).sum()); stats["escapes"] += int((~hit).sum())
        esc.index_add_(0, cells[~hit], torch.full((int((~hit).sum()),), 1.0 / M_rays, device=DEV))
        j = front.nonzero().squeeze(1)
        if mask is not None and len(j):
            keep = mask(ss.port_comp[port[j]], ss.tri_comp[prim[j]]); j = j[keep]
        pr = prim[j]
        ib = G.local_to_bin(G.to_local(-d[j], ss.n[pr], ss.t[pr], ss.b[pr]), ss.Nin)
        rows.append(ss.tri_port[pr] * ss.Din + ib); cols.append(cells[j]); vals.append(torch.full((len(j),), 1.0 / M_rays, device=DEV))
    idx = torch.stack([torch.cat(rows), torch.cat(cols)])
    C = torch.sparse_coo_tensor(idx, torch.cat(vals), (ss.Ptot * ss.Din, nout)).coalesce()
    return C, esc, stats


def sun_external(ss, s_travel, E, K, seed):
    """Unoccluded direct irradiance per port (port average, W/m^2) and binned external in-cell flux."""
    gen = torch.Generator(device=DEV); gen.manual_seed(int(seed))
    s = torch.tensor(np.asarray(s_travel, float) / np.linalg.norm(s_travel), dtype=torch.float32, device=DEV)
    ports = torch.arange(ss.Ptot, device=DEV).repeat_interleave(K)
    tri, x = ss.sample_ports(ports, gen)
    cos = -(ss.n[tri] * s).sum(1); lit = cos > 0
    t, _ = ss.isect(x + 4 * ss.eps * ss.n[tri], (-s).expand(len(x), 3).contiguous())
    vis = lit & ~torch.isfinite(t)
    e = E * cos.clamp(min=0) * vis.float()
    E_port = torch.zeros(ss.Ptot, device=DEV).index_add_(0, ports, e) / K
    ib = G.local_to_bin(G.to_local((-s).expand(len(x), 3), ss.n[tri], ss.t[tri], ss.b[tri]), ss.Nin)
    phi = torch.zeros(ss.Ptot * ss.Din, device=DEV).index_add_(0, ports * ss.Din + ib, e * ss.port_area[ports] / K)
    return E_port, phi


# ------------------------------------------------------------------ K application and solve


def apply_K(ss, Tm, phi_in):
    """phi_in (n_in, 3) -> phi_out (n_out, 3). Tm: dict comp index -> T (3, P*Dout, P*Din)."""
    out = torch.zeros(ss.Ptot * ss.Dout, 3, device=DEV)
    for ci, c in enumerate(ss.comps):
        sl = ss.comp_ports(ci); i0, i1 = sl.start * ss.Din, sl.stop * ss.Din; o0, o1 = sl.start * ss.Dout, sl.stop * ss.Dout
        if c.kind == "medium":
            T = Tm[ci]
            out[o0:o1] = torch.stack([T[k] @ phi_in[i0:i1, k] for k in range(3)], 1)
        elif c.kind == "lambert":
            tot = phi_in[i0:i1].view(c.P, ss.Din, 3).sum(1)
            rho = torch.tensor(c.rho, dtype=torch.float32, device=DEV)
            out[o0:o1] = (tot * rho / ss.Dout)[:, None, :].expand(c.P, ss.Dout, 3).reshape(-1, 3)
    return out


def apply_C(C, phi_out):
    return torch.stack([torch.sparse.mm(C, phi_out[:, k:k + 1]).squeeze(1) for k in range(3)], 1)


def solve(ss, Tm, C, phi0_out, tol=1e-7, max_iter=2000):
    phi = phi0_out.clone(); hist = []
    for it in range(1, max_iter + 1):
        phi_in = apply_C(C, phi); new = phi0_out + apply_K(ss, Tm, phi_in)
        rel = float((new - phi).abs().sum() / new.abs().sum().clamp(min=1e-30)); hist.append(rel); phi = new
        if rel < tol:
            break
    phi_in = apply_C(C, phi)
    return dict(phi_out=phi, phi_in_ind=phi_in, iters=it, converged=rel < tol, last_rel=rel, hist=hist[:50])


def spectral_radius(ss, Tm, C, iters=300, seed=0):
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    v = torch.rand(ss.Ptot * ss.Dout, 3, device=DEV, generator=g); rho = None
    for _ in range(iters):
        w = apply_K(ss, Tm, apply_C(C, v)); nv = w.norm(dim=0); rho = (nv / v.norm(dim=0).clamp(min=1e-30)).cpu().numpy(); v = w / nv.clamp(min=1e-30)
        if float(nv.max()) == 0.0:
            return [0.0, 0.0, 0.0]
    return rho.tolist()


# ------------------------------------------------------------------ readout


def readout(ss, cam_dir, sol, L0, KC=None, E_ext=None, mode="nee"):
    """Port-average radiance toward cam_dir (unit, pointing to the camera) for every port (Ptot, 3).
    L0: dict comp -> first-order port radiance (P, 3) (medium, exact sun via TR) ; KC: dict comp -> (3, P, P*Din)."""
    c = torch.tensor(np.asarray(cam_dir, float), dtype=torch.float32, device=DEV)
    L = torch.zeros(ss.Ptot, 3, device=DEV)
    phi_out = sol["phi_out"]; phi_in = sol["phi_in_ind"]
    for ci, comp in enumerate(ss.comps):
        sl = ss.comp_ports(ci); i0, i1 = sl.start * ss.Din, sl.stop * ss.Din
        if comp.kind == "medium":
            if mode == "nee":
                K = KC[ci]
                L[sl] = L0[ci] + torch.stack([K[k] @ phi_in[i0:i1, k] for k in range(3)], 1)
            else:
                tris = (ss.tri_comp == ci).nonzero().squeeze(1)
                cs = (ss.n[tris] * c).sum(1).clamp(min=0)
                ob = G.local_to_bin(G.to_local(c.expand(len(tris), 3), ss.n[tris], ss.t[tris], ss.b[tris]), ss.Nout)
                port = ss.tri_port[tris]
                Lb = phi_out[port * ss.Dout + ob] / (ss.port_area[port] * math.pi / ss.Dout)[:, None]
                wsum = torch.zeros(ss.Ptot, device=DEV).index_add_(0, port, ss.area[tris] * cs)
                acc = torch.zeros(ss.Ptot, 3, device=DEV).index_add_(0, port, Lb * (ss.area[tris] * cs)[:, None])
                L[sl] = (acc / wsum.clamp(min=1e-30)[:, None])[sl]
        elif comp.kind == "lambert":
            Eind = phi_in[i0:i1].view(comp.P, ss.Din, 3).sum(1) / ss.port_area[sl][:, None]
            rho = torch.tensor(comp.rho, dtype=torch.float32, device=DEV)
            L[sl] = rho / math.pi * (E_ext[sl][:, None] + Eind)
    return L


# ------------------------------------------------------------------ reference port averages


def classify(ss, rays, valid_comps=None):
    """Per pixel global port id if the centre and every corner ray hit the same port first, else -1."""
    ids = []
    for o, d in rays:
        t, prim = ss.isect(o, d)
        pid = torch.where(torch.isfinite(t) & (prim >= 0), ss.tri_port[prim.clamp(min=0)], torch.full_like(prim, -1))
        ids.append(pid)
    ids = torch.stack(ids); pid = ids[0]
    ok = (ids == pid[None]).all(0) & (pid >= 0)
    if valid_comps is not None:
        ok &= torch.isin(ss.port_comp[pid.clamp(min=0)], torch.tensor(valid_comps, device=DEV))
    return torch.where(ok, pid, torch.full_like(pid, -1))


def port_means(pid, img, Ptot):
    """img (H*W, 3) numpy -> mean per port (Ptot, 3), pixel counts (Ptot,)."""
    pid = pid.cpu().numpy(); m = pid >= 0
    cnt = np.bincount(pid[m], minlength=Ptot).astype(float)
    s = np.stack([np.bincount(pid[m], weights=img[m, k], minlength=Ptot) for k in range(3)], 1)
    return s / np.maximum(cnt, 1)[:, None], cnt
