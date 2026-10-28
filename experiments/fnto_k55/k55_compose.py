"""WL55 matched-resolution K/C composition on arbitrary per-component cell levels (protocol fnto_k55, stage_B.matched_composition,
stage_C), the interface-consistent ballistic readout KB, the known-field (single-flight) arms and the outgoing-aggregation test.

Arm = (levels per coarse port of every component, medium operator per medium component). Cells: k55_ops.refined (exact surface).
  C        c54_casp.cast_adaptive, equal ray budget per coarse port (128 rays per out-cell at level 1, 128 / l^2 at level l; 64 out-bins)
  sun      ONE TR.run_sun per medium domain on its level-4 (2.5 mm) mesh per scene (seed 5591 + comp), aggregated to every arm's cells:
           arms differ only by their interface, not by first-order sun noise; Lambert: p53_compose.sun_external on the arm's cells
  K        aggregated from the 2.5 mm oracle (k55_ops.aggregate) or a direct dense / sparse TR operator at the arm's level
  solve    p53_compose.solve (unchanged); spectral radius c54_casp.rho_robust (A^2)
  readout  medium coarse port: L0 + KC phi_in_ind (nee, = WL53) and + KB phi_in_ind (ballistic); Lambert: rho/pi (E_ext + E_ind),
           area-weighted onto the coarse port
KB[i <- (j, b)]: projected-area mean over port i of exp(-sigma_t s) / (A_j pi / Din) for the backward camera ray that leaves the medium
through cell j with outward direction in bin b (the in-cell radiance is uniform over cell and bin, the interface's own assumption).
"""
import math
import time

import numpy as np
import torch

import k55_common as KCm
import k55_fixtures as KF
import k55_ops as KO
import c54_casp as CA
import p53_compose as CP
import p53_fixtures as FX
import p53_geom as G
import p53_tracer as TR

DEV = "cuda"
DOUT, DIN = 64, 16


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def comp_ss_kind(k):
    return "black" if k == "emitter" else k


def build_ss(sd, levels, din=DIN):
    """levels: list (per comp) of int arrays (per coarse port). Returns (SceneSet, parents (cell -> global coarse port), cell levels)."""
    comps, parents, lev_cell, off = [], [], [], 0
    for ci, (n, k, m, V) in enumerate(sd["comps"]):
        lv = np.broadcast_to(np.asarray(levels[ci], int), (len(m["port_area"]),)).copy()
        fm, VC, par = KO.refined(m, V, lv)
        comps.append(CP.Comp(n, comp_ss_kind(k), fm, VC, rho=FX.FLOOR_RHO if k == "lambert" else None))
        parents.append(par + off); lev_cell.append(lv[par]); off += len(m["port_area"])
    ss = CP.SceneSet(comps, DOUT, din)
    return ss, np.concatenate(parents), np.concatenate(lev_cell)


def fine_domain(m, V, name=""):
    fm, VC, par = KO.refined(m, V, 4)
    return KO.DomainX(fm, VC, name=name), par


def cell_areas(m, V, levels):
    fm, VC, par = KO.refined(m, V, levels)
    fr = G.tri_frames(fm["V"], VC, fm["F"], fm["tri_port"], fm["port_tangent"])
    return G.port_areas(fr["area"], fm["tri_port"], len(par))


# ------------------------------------------------------------------ first-order sun term (shared by every arm of a scene)


def sun_term(sd, cam_keys, n_sun=16_000_000, seed=5591):
    ss0 = KCm.classify_sceneset(sd); out = {}
    if not sd["lights"]:
        return out
    s, E = sd["lights"][0]; dirs = np.array([FX.DIRS[k] for k in cam_keys])
    for ci, (n, k, m, V) in enumerate(sd["comps"]):
        if k != "medium":
            continue
        dom, par = fine_domain(m, V, n)
        r = TR.run_sun(dom, s, E, n_sun, seed + ci, DOUT, cams=dirs, shadow=ss0.isect)
        Pc = len(m["port_area"]); pt = torch.as_tensor(par, device=DEV)
        KCf = r["KC"][:, :, :, 0]; ap = r["proj_area"]                      # (ncam, 3, Pf), (ncam, Pf)
        num = torch.zeros(len(dirs), 3, Pc, device=DEV, dtype=torch.float64).index_add_(2, pt, (KCf * ap[:, None]).double())
        den = torch.zeros(len(dirs), Pc, device=DEV, dtype=torch.float64).index_add_(1, pt, ap.double())
        out[ci] = dict(phi_f=r["T"][:, :, 0].T.contiguous().cpu(), L0=(num / den.clamp(min=1e-30)[:, None]).permute(0, 2, 1).cpu(), stats=r["stats"], seconds=r["seconds"],
                       cam_keys=list(cam_keys))
        del r, dom; torch.cuda.empty_cache()
    return out


def aggregate_phi(phi_f, levels):
    cof = torch.as_tensor(KO.fine_to_cells(levels), device=DEV); ncell = int((np.asarray(levels) ** 2).sum())
    rows = (cof[:, None] * DOUT + torch.arange(DOUT, device=DEV)[None]).reshape(-1)
    return torch.zeros(ncell * DOUT, 3, device=DEV).index_add_(0, rows, phi_f.to(DEV))


# ------------------------------------------------------------------ medium operators per arm


def arm_operator(Kf, m, V, levels):
    """Aggregate a raw 2.5 mm operator (k55_ops.tr_sparse output) onto `levels` cells of the domain (m, V)."""
    lv = np.broadcast_to(np.asarray(levels, int), (len(m["port_area"]),)).copy()
    af = cell_areas(m, V, 4); ncell = int((lv ** 2).sum())
    if np.all(lv == 4):
        agg = Kf
    else:
        agg = KO.aggregate(Kf, KO.fine_to_cells(lv), af, ncell)
    return dict(K=KO.KOp(agg["idx"], agg["val"], agg["shape"]), KC=agg["KC"].to(DEV), cam_keys=list(Kf["cam_keys"]), nnz=len(agg["idx"]))


# ------------------------------------------------------------------ ballistic readout KB


def ballistic_KB(ss, ci, parent, cam_dir, n_per_port=4096, seed=0, sigma_t=FX.SIGMA_T):
    """Sparse (Pc_comp, n_in_cells_comp) ballistic camera operator of medium comp ci."""
    gen = torch.Generator(device=DEV); gen.manual_seed(int(seed))
    c = torch.tensor(np.asarray(cam_dir, float), dtype=torch.float32, device=DEV)
    sl = ss.comp_ports(ci); tris = (ss.tri_comp == ci).nonzero().squeeze(1)
    par = torch.as_tensor(parent, device=DEV)
    cp_of_tri = par[ss.tri_port[tris]]; c0 = int(par[sl.start]); Pc = int(par[sl.stop - 1]) - c0 + 1
    w = (ss.area[tris] * (ss.n[tris] * c).sum(1).clamp(min=0)).double()
    pw = torch.zeros(Pc, device=DEV, dtype=torch.float64).index_add_(0, cp_of_tri - c0, w)
    vis = (pw > 0).nonzero().squeeze(1)
    order = torch.argsort(cp_of_tri - c0, stable=True); ts = tris[order]; ps = (cp_of_tri - c0)[order]; cum = torch.cumsum(w[order], 0)
    rows, cols, vals = [], [], []
    for i in range(0, len(vis), 64):
        pv = vis[i:i + 64]; rep = pv.repeat_interleave(n_per_port)
        first = torch.searchsorted(ps, rep); base = torch.where(first > 0, cum[(first - 1).clamp(min=0)], torch.zeros(len(rep), device=DEV, dtype=torch.float64))
        u = torch.rand(len(rep), device=DEV, generator=gen, dtype=torch.float64)
        idx = torch.searchsorted(cum, base + u * pw[rep]).clamp(max=len(cum) - 1); tri = ts[idx]
        y = G.sample_on_tris(ss.V, ss.F, tri, torch.rand(len(rep), device=DEV, generator=gen), torch.rand(len(rep), device=DEV, generator=gen))
        o = y - ss.eps * ss.n[tri]; dm = (-c).expand(len(rep), 3).contiguous()
        t1, p1 = ss.isect(o, dm); ok = torch.isfinite(t1)
        okk = ok.clone(); okk[ok] = (ss.tri_comp[p1[ok]] == ci) & ((ss.n[p1[ok]] * (-c)).sum(1) > 0)
        j = okk.nonzero().squeeze(1); pj = p1[j]
        b = G.local_to_bin(G.to_local(dm[j], ss.n[pj], ss.t[pj], ss.b[pj]), ss.Nin)
        cell = ss.tri_port[pj]
        v = torch.exp(-sigma_t * t1[j].double()) / (ss.port_area[cell].double() * math.pi / ss.Din) / n_per_port
        rows.append(rep[j]); cols.append((cell - sl.start) * ss.Din + b); vals.append(v)
    nin = (sl.stop - sl.start) * ss.Din
    if not rows:
        return torch.sparse_coo_tensor(torch.zeros(2, 0, dtype=torch.long, device=DEV), torch.zeros(0, device=DEV, dtype=torch.float64), (Pc, nin)).coalesce()
    return torch.sparse_coo_tensor(torch.stack([torch.cat(rows), torch.cat(cols)]), torch.cat(vals), (Pc, nin)).coalesce()


def kb_apply(KB, phi):
    return torch.stack([torch.sparse.mm(KB, phi[:, k:k + 1].double()).squeeze(1) for k in range(3)], 1)


# ------------------------------------------------------------------ composition arm


def compose(sd, levels, Kops, sun, cam_keys, seed_C=5581, K_ext=256, seed_E=5585, kb=True, rho=True, label="", din=DIN):
    """Kops: comp index -> dict(K (KOp or dense T), KC (ncam, 3, Pc, n_in) on the coarse output ports, cam_keys of KC's first axis). Returns dict(pred={cam: {'nee': (Pc_tot, 3), 'ball': (Pc_tot, 3)}}, meta)."""
    t0 = time.time(); ss, parent, lev_cell = build_ss(sd, levels, din)
    s, E = sd["lights"][0] if sd["lights"] else ((0.0, 0.0, -1.0), 0.0)
    cnt = np.bincount(parent); m_port = np.maximum(1, np.round(128 / cnt[parent])).astype(int)
    torch.cuda.synchronize(); tc = time.time(); C, esc, cst = CA.cast_adaptive(ss, m_port, seed_C); torch.cuda.synchronize(); t_C = time.time() - tc
    Ep, _ = CP.sun_external(ss, s, E, K_ext, seed_E)
    phi0 = torch.zeros(ss.Ptot * DOUT, 3, device=DEV); Tm = {}
    for ci, comp in enumerate(ss.comps):
        sl = ss.comp_ports(ci)
        if comp.kind == "medium":
            phi0[sl.start * DOUT:sl.stop * DOUT] = aggregate_phi(sun[ci]["phi_f"], levels[ci]) if ci in sun else 0
            Tm[ci] = Kops[ci]["K"]
        elif comp.kind == "lambert":
            rr = torch.tensor(comp.rho, dtype=torch.float32, device=DEV)
            phi0[sl.start * DOUT:sl.stop * DOUT] = ((Ep[sl] * ss.port_area[sl])[:, None] * rr / DOUT).repeat_interleave(DOUT, 0)
    torch.cuda.synchronize(); ts = time.time(); sol = CP.solve(ss, Tm, C, phi0); torch.cuda.synchronize(); t_solve = time.time() - ts
    rho_v = CA.rho_robust(ss, Tm, C) if rho else None
    phi_in = sol["phi_in_ind"]; Pc_tot = int(parent.max()) + 1; area = ss.port_area.double()
    par_t = torch.as_tensor(parent, device=DEV)
    pred = {}
    for k, key in enumerate(cam_keys):
        d = np.asarray(FX.DIRS[key], float); Ln = torch.zeros(Pc_tot, 3, device=DEV, dtype=torch.float64); Lb = torch.zeros_like(Ln)
        for ci, comp in enumerate(ss.comps):
            sl = ss.comp_ports(ci); i0, i1 = sl.start * din, sl.stop * din; c0 = int(parent[sl.start]); c1 = int(parent[sl.stop - 1]) + 1
            if comp.kind == "medium":
                KCk = Kops[ci]["KC"][Kops[ci]["cam_keys"].index(key)]
                ph = phi_in[i0:i1]
                Ln[c0:c1] = torch.stack([KCk[cc].double() @ ph[:, cc].double() for cc in range(3)], 1)
                if ci in sun:
                    Ln[c0:c1] += sun[ci]["L0"][sun[ci]["cam_keys"].index(key)].to(DEV).double()
                if kb:
                    KB = ballistic_KB(ss, ci, parent, d, seed=7000 + 13 * ci + k)
                    Lb[c0:c1] = kb_apply(KB, ph)
            elif comp.kind == "lambert":
                Eind = phi_in[i0:i1].view(comp.P, din, 3).sum(1).double() / area[sl][:, None]
                Lc = float(comp.rho[0]) / math.pi * (Ep[sl].double()[:, None] + Eind)
                pc = par_t[sl]; num = torch.zeros(Pc_tot, 3, device=DEV, dtype=torch.float64).index_add_(0, pc, Lc * area[sl][:, None])
                den = torch.zeros(Pc_tot, device=DEV, dtype=torch.float64).index_add_(0, pc, area[sl])
                Ln[c0:c1] = (num / den.clamp(min=1e-30)[:, None])[c0:c1]
        pred[key] = dict(nee=Ln.cpu().numpy(), ball=Lb.cpu().numpy())
    meta = dict(label=label, cells=int(ss.Ptot), coarse=Pc_tot, rho=rho_v, iters=sol["iters"], converged=bool(sol["converged"]), t_C=t_C, t_solve=t_solve,
                C_nnz=int(C._nnz()), back_hits=int(cst["back_hits"]), wall=time.time() - t0, gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                levels_hist={int(l): int(c) for l, c in zip(*np.unique(lev_cell, return_counts=True))})
    log("compose", label, {k: v for k, v in meta.items() if k not in ("levels_hist",)})
    del C, sol, ss; torch.cuda.empty_cache()
    return dict(pred=pred, meta=meta)


# ------------------------------------------------------------------ known-field single-flight arms


def emitter_rays(ss, sd, N, gen):
    """Cosine rays from the scene's emitter components (profile-importance sampled); returns (x, d, w)."""
    sel = torch.zeros(len(ss.tri_comp), dtype=torch.bool, device=DEV); Lc = torch.zeros(len(ss.tri_comp), device=DEV)
    cen = (ss.V[ss.F[:, 0]] + ss.V[ss.F[:, 1]] + ss.V[ss.F[:, 2]]) / 3; fns = {}
    for ci, (n, k, m, V) in enumerate(sd["comps"]):
        if k == "emitter":
            f = KF.profile(sd["emit"][n]); fns[ci] = f; mm = ss.tri_comp == ci; sel |= mm; Lc[mm] = f(cen[mm]).float()
    wt = ss.area * Lc * sel.float(); tot = float(wt.sum())
    tri = torch.multinomial(wt, N, replacement=True, generator=gen)
    x = G.sample_on_tris(ss.V, ss.F, tri, torch.rand(N, device=DEV, generator=gen), torch.rand(N, device=DEV, generator=gen))
    Lx = torch.zeros(N, device=DEV)
    for ci, f in fns.items():
        mm = ss.tri_comp[tri] == ci; Lx[mm] = f(x[mm]).float()
    import k55_cor as COR
    d = COR._cos_dir(ss.n[tri], gen)
    return x + ss.eps * ss.n[tri], d, (math.pi * Lx * tot / (Lc[tri] * N))


def kf_field(sd, levels, N=64_000_000, seed=5601, chunk=8_000_000):
    """Exactly cast incident in-cell flux (n_cells * Din, 3) of every medium comp from the scene's emitters (single flight)."""
    ss, parent, _ = build_ss(sd, levels); gen = torch.Generator(device=DEV); gen.manual_seed(int(seed))
    phi = torch.zeros(ss.Ptot * DIN, device=DEV, dtype=torch.float64)
    for i in range(0, N, chunk):
        x, d, w = emitter_rays(ss, sd, min(chunk, N - i), gen); w = w * min(chunk, N - i) / N
        t, prim = ss.isect(x, d); hit = torch.isfinite(t); hit[hit.clone()] &= (d[hit] * ss.n[prim[hit]]).sum(1) < 0
        j = hit.nonzero().squeeze(1); pj = prim[j]
        b = G.local_to_bin(G.to_local(-d[j], ss.n[pj], ss.t[pj], ss.b[pj]), ss.Nin)
        phi.index_add_(0, ss.tri_port[pj] * ss.Din + b, w[j].double())
    return ss, parent, phi


def kf_readout(ss, parent, phi, Kops, cam_keys, kb=True):
    """Single-flight medium readout from an in-cell flux field (grey emitter: the same flux in R, G, B)."""
    pred = {}; Pc_tot = int(parent.max()) + 1; ph3 = phi[:, None].expand(-1, 3).float()
    for k, key in enumerate(cam_keys):
        d = np.asarray(FX.DIRS[key], float); Ln = np.zeros((Pc_tot, 3)); Lb = np.zeros((Pc_tot, 3))
        for ci, comp in enumerate(ss.comps):
            if comp.kind != "medium" or ci not in Kops:
                continue
            sl = ss.comp_ports(ci); i0, i1 = sl.start * DIN, sl.stop * DIN; c0 = int(parent[sl.start]); c1 = int(parent[sl.stop - 1]) + 1
            KCk = Kops[ci]["KC"][Kops[ci]["cam_keys"].index(key)]; ph = ph3[i0:i1]
            Ln[c0:c1] = torch.stack([KCk[cc] @ ph[:, cc] for cc in range(3)], 1).double().cpu().numpy()
            if kb:
                Lb[c0:c1] = kb_apply(ballistic_KB(ss, ci, parent, d, seed=7100 + k), ph).cpu().numpy()
        pred[key] = dict(nee=Ln, ball=Lb)
    return pred


def outgoing_test(sd, lev_lo, sun, seed_C=5587):
    """Arrival flux on the receiver comp 'hi_bot' (level-4 cells) from slab_lo's binned first-order exit field at level lev_lo."""
    levels = []
    for n, k, m, V in sd["comps"]:
        levels.append(lev_lo if n == "slab_lo" else (4 if n == "hi_bot" else 1))
    ss, parent, _ = build_ss(sd, levels)
    ci_lo = [i for i, c in enumerate(ss.comps) if c.name == "slab_lo"][0]; ci_hi = [i for i, c in enumerate(ss.comps) if c.name == "hi_bot"][0]
    cnt = np.bincount(parent); m_port = np.maximum(1, np.round(128 / cnt[parent])).astype(int)
    lo_sl = ss.comp_ports(ci_lo); m_port[:lo_sl.start] = 1; m_port[lo_sl.stop:] = 1
    C, esc, cst = CA.cast_adaptive(ss, m_port, seed_C)
    phi_out = torch.zeros(ss.Ptot * DOUT, 3, device=DEV)
    phi_out[lo_sl.start * DOUT:lo_sl.stop * DOUT] = aggregate_phi(sun[ci_lo]["phi_f"], lev_lo)
    phi_in = CP.apply_C(C, phi_out); hs = ss.comp_ports(ci_hi)
    arr = phi_in[hs.start * DIN:hs.stop * DIN].view(hs.stop - hs.start, DIN, 3).sum(1)
    tris = (ss.tri_comp == ci_hi).nonzero().squeeze(1)
    cen = torch.zeros(hs.stop - hs.start, 3, device=DEV).index_add_(0, ss.tri_port[tris] - hs.start, ((ss.V[ss.F[tris, 0]] + ss.V[ss.F[tris, 1]] + ss.V[ss.F[tris, 2]]) / 3) * ss.area[tris][:, None])
    cen = cen / torch.zeros(hs.stop - hs.start, device=DEV).index_add_(0, ss.tri_port[tris] - hs.start, ss.area[tris])[:, None]
    return dict(arr=arr.cpu().numpy(), cen=cen.cpu().numpy(), parent=parent[hs.start:hs.stop] - parent[hs.start], back_hits=int(cst["back_hits"]))
