"""WL55 multi-resolution intrinsic-K oracle operators (protocol fnto_k55, stage_B.instruments.OPS and stage_C.K_on_cells).

The WL53 oracle TR (p53_tracer.cell_source / walk, unchanged) is run on REFINED cell meshes of the WL53 ports: c54_casp.refine_mesh (exact
triangle-wise subdivision of the current surface, rest geometry refined identically), welded here. A level-l port has l x l cells.
Accumulation is duck-typed so that walk() needs no change:
  T   sparse: every photon exit adds one (out-cell, in-cell) entry; entries are coalesced per source chunk in float64 and stored float32
  KC  next-event radiance toward each camera direction, accumulated on the COARSE 1 cm output ports (the compared quantity; summing NEE
      flux over a port's cells is exact) and normalised by the coarse projected area
  S2  per (fine out-port, fine in-port) second moments (WL53 semantics, psrc = P)
Aggregation (stage_C K_on_cells): K_cells = R_out K_fine P_in, P_in distributing a cell's in-flux over its fine sub-cells by current area
(same in-bin), R_out summing fine out-cells (same out-bin). This equals a TR run on the cells in expectation.
"""
import math
import time

import numpy as np
import torch

import k55_common as KC  # noqa: F401
import c54_casp as CA
import p53_fixtures as FX
import p53_geom as G
import p53_tracer as TR

DEV = "cuda"


# ------------------------------------------------------------------ meshes


def refined(mesh, Vcur, levels):
    """c54_casp.refine_mesh + vertex welding (rest and current coordinates rounded to 1 nm jointly). Returns (fine mesh, Vcur, parent)."""
    fm, VC, par = CA.refine_mesh(mesh, Vcur, np.broadcast_to(np.asarray(levels, int), (len(mesh["port_area"]),)).copy())
    key = np.round(np.concatenate([fm["V"], VC], 1) / 1e-9).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    fm = dict(fm); fm["V"] = fm["V"][first]; fm["F"] = np.asarray(inv).reshape(-1)[fm["F"]]
    return fm, VC[first], par


def fine_to_cells(levels, lf=4):
    """Map of the uniform level-lf cells (refine_mesh numbering, g = 4 ports) to the cells of a level array (per coarse port)."""
    levels = np.asarray(levels, int); P = len(levels)
    off = np.concatenate([[0], np.cumsum(levels ** 2)[:-1]])
    f = np.arange(P * lf * lf); p = f // (lf * lf); aa, bb = np.divmod(f % (lf * lf), lf); l = levels[p]
    return off[p] + (aa * l // lf) * l + (bb * l // lf)


class DomainX(TR.Domain):
    """TR.Domain for meshes whose ports have unequal triangle counts (adaptive cells): padded port_tris; the cumulative area table
    is set to 2 from each port's last real triangle on, so a uniform draw can never select a padding entry."""

    def __init__(self, mesh, Vcur, sigma_t=FX.SIGMA_T, albedo=FX.ALBEDO, g=FX.G_PHASE, name=""):
        import p53_mi as M
        self.name = name; self.mesh = mesh; self.sigma_t = float(sigma_t); self.g = float(g)
        self.albedo = torch.tensor(np.broadcast_to(np.asarray(albedo, float), (3,)).copy(), dtype=torch.float32, device=DEV)
        F = mesh["F"]; fr = G.tri_frames(mesh["V"], Vcur, F, mesh["tri_port"], mesh["port_tangent"])
        f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=DEV)  # noqa: E731
        self.Vcur = np.asarray(Vcur); self.V = f32(Vcur); self.F = torch.tensor(F, device=DEV)
        self.n, self.t, self.b, self.area = f32(fr["n"]), f32(fr["t"]), f32(fr["b"]), f32(fr["area"])
        tp = np.asarray(mesh["tri_port"]); self.tri_port = torch.tensor(tp, device=DEV); self.P = len(mesh["port_area"])
        self.port_area = torch.zeros(self.P, device=DEV, dtype=torch.float64).index_add_(0, self.tri_port, self.area.double()).float()
        order = np.argsort(tp, kind="stable"); cnt = np.bincount(tp, minlength=self.P); kmax = int(cnt.max())
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]]); pt = np.full((self.P, kmax), -1)
        for k in range(kmax):
            has = cnt > k; pt[has, k] = order[start[has] + k]
        self.port_tris = torch.tensor(pt, device=DEV)
        a = torch.where(self.port_tris >= 0, self.area[self.port_tris.clamp(min=0)], torch.zeros(1, device=DEV))
        cum = torch.cumsum(a, 1) / a.sum(1, keepdim=True)
        last = torch.tensor(cnt - 1, device=DEV)[:, None]; cols = torch.arange(kmax, device=DEV)[None, :]
        self.port_cum = torch.where(cols >= last, torch.full_like(cum, 2.0), cum)
        self.isect = M.Intersector(Vcur, F)
        self.eps = 2e-6 * max(1.0, float(np.abs(Vcur).max()))


# ------------------------------------------------------------------ duck-typed accumulator for p53_tracer.walk


class _TSink:
    def __init__(self):
        self.idx, self.val = [], []

    def index_add_(self, dim, idx, src):
        self.idx.append(idx.clone()); self.val.append(src.clone())

    def flush(self):
        if not self.idx:
            return torch.empty(0, dtype=torch.long), torch.empty(3, 0)
        i = torch.cat(self.idx); v = torch.cat(self.val, 1); self.idx, self.val = [], []
        u, inv = torch.unique(i, return_inverse=True)
        acc = torch.zeros(3, len(u), dtype=torch.float64, device=DEV).index_add_(1, inv, v)
        return u.cpu(), acc.float().cpu()


class _KCView:
    def __init__(self, sink, k):
        self.s, self.k = sink, k

    def index_add_(self, dim, idx, src):
        s = self.s; fp = torch.div(idx, s.nsrc, rounding_mode="floor"); col = idx - fp * s.nsrc
        s.buf[self.k].index_add_(1, s.parent[fp] * s.nsrc + col, src)


class _KCSink:
    def __init__(self, ncam, nsrc, parent, Pc):
        self.nsrc, self.Pc = nsrc, Pc; self.parent = torch.as_tensor(parent, device=DEV, dtype=torch.long)
        self.buf = torch.zeros(ncam, 3, Pc * nsrc, dtype=torch.float64, device=DEV)

    def __getitem__(self, k):
        return _KCView(self, k)


class SparseAccum:
    def __init__(self, dom, Dout, Din, cams, parent, Pc, s2=True):
        self.dom, self.Dout, self.Din = dom, Dout, Din; self.Nout = int(round(math.sqrt(Dout))); self.nsrc = dom.P * Din
        self.T = _TSink(); self.psrc = dom.P
        self.S2 = torch.zeros(3, dom.P * dom.P if s2 else dom.P, device=DEV, dtype=torch.float64)
        if not s2:
            self.psrc = 1
        self.cams = None if cams is None else torch.tensor(np.asarray(cams, float), dtype=torch.float32, device=DEV).reshape(-1, 3)
        self.KC = None if cams is None else _KCSink(len(self.cams), self.nsrc, parent, Pc)
        self.stats = dict(photons=0, exits=0, leaks=0, wrong_side=0, truncated=0, nee_invalid=0, steps=0, collisions=0,
                          flux_in=np.zeros(3), flux_out=np.zeros(3), flux_leak=np.zeros(3))


def coarse_proj_area(dom, cams, parent, Pc):
    par = torch.as_tensor(parent, device=DEV)[dom.tri_port]
    return torch.stack([torch.zeros(Pc, device=DEV, dtype=torch.float64).index_add_(0, par, (dom.area * (dom.n * c).sum(1).clamp(min=0)).double())
                        for c in torch.tensor(np.asarray(cams, float), dtype=torch.float32, device=DEV)])


def tr_sparse(dom, Dout, Din, n_per, seed, cams, parent, Pc, chunk_photons=4_000_000, log=None, s2=True):
    """Sparse TR operator on DomainX/Domain dom. parent: fine port -> coarse port (local). Returns dict(idx, val, shape, KC, S2, stats)."""
    gen = torch.Generator(device=DEV); gen.manual_seed(int(seed)); nsrc = dom.P * Din
    acc = SparseAccum(dom, Dout, Din, cams, parent, Pc, s2=s2)
    per = max(1, chunk_photons // n_per); I, Vv = [], []; t0 = time.time()
    for i in range(0, nsrc, per):
        cells = torch.arange(i, min(i + per, nsrc), device=DEV)
        x, d, n, w, src = TR.cell_source(dom, cells, n_per, Din, gen)
        TR.walk(acc, x, d, n, w, src, gen)
        a, b = acc.T.flush(); I.append(a); Vv.append(b)
        if log and (i // per) % 10 == 0:
            log(f"{dom.name} cells {min(i + per, nsrc)}/{nsrc} {time.time() - t0:.0f}s steps {acc.stats['steps']}")
    out = dict(idx=torch.cat(I), val=torch.cat(Vv, 1), shape=(dom.P * Dout, nsrc), Dout=Dout, Din=Din, n_per=n_per, seed=int(seed),
               stats={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in acc.stats.items()}, seconds=time.time() - t0)
    if cams is not None:
        ap = coarse_proj_area(dom, cams, parent, Pc)
        KCm = acc.KC.buf.view(len(cams), 3, Pc, nsrc) / ap[:, None, :, None].clamp(min=1e-30)
        KCm[(ap <= 0)[:, None, :, None].expand_as(KCm)] = 0
        out["KC"] = KCm.float().cpu(); out["cams"] = np.asarray(cams, float).tolist(); acc.KC = None
    out["S2"] = acc.S2.view(3, dom.P, acc.psrc).float().cpu() if s2 else None
    out["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    del acc; torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------ operator object (works inside p53_compose.apply_K / solve)


class _Chan:
    def __init__(self, csr):
        self.csr = csr

    def __matmul__(self, v):
        return (self.csr @ v.contiguous().unsqueeze(1)).squeeze(1)


class KOp:
    """Sparse T as three CSR matrices (rows out-cells, cols in-cells); T[k] @ v works in p53_compose.apply_K."""

    def __init__(self, idx, val, shape):
        idx = idx.to(DEV); val = val.to(DEV)
        order = torch.argsort(idx); idx = idx[order]; val = val[:, order]
        rows = torch.div(idx, shape[1], rounding_mode="floor"); cols = (idx - rows * shape[1]).int()
        crow = torch.zeros(shape[0] + 1, dtype=torch.int64, device=DEV); crow[1:] = torch.cumsum(torch.bincount(rows, minlength=shape[0]), 0)
        crow = crow.int(); self.shape = shape; self.nnz = len(idx)
        self.ch = [_Chan(torch.sparse_csr_tensor(crow, cols, val[k].contiguous(), shape)) for k in range(3)]

    def __getitem__(self, k):
        return self.ch[k]

    def colsum(self):
        ones = torch.ones(self.shape[0], device=DEV)
        return torch.stack([(c.csr.t() @ ones[:, None]).squeeze(1) if False else _colsum(c.csr, self.shape) for c in self.ch])


def _colsum(csr, shape):
    col = csr.col_indices().long(); v = csr.values()
    return torch.zeros(shape[1], device=DEV, dtype=torch.float64).index_add_(0, col, v.double())


def aggregate(op, cell_of_fine, area_f, ncell, chunk=40_000_000):
    """R_out K P_in of a raw operator dict (idx, val, shape, Dout, Din, KC) onto cells. cell_of_fine (Pf,), area_f (Pf,) current areas."""
    Dout, Din = op["Dout"], op["Din"]; nrow_f, nsrc_f = op["shape"]; nsrc_c = ncell * Din
    cof = torch.as_tensor(cell_of_fine, device=DEV, dtype=torch.long); af = torch.as_tensor(area_f, device=DEV, dtype=torch.float64)
    ac = torch.zeros(ncell, device=DEV, dtype=torch.float64).index_add_(0, cof, af); wf = (af / ac[cof])
    I, V = [], []
    for i in range(0, len(op["idx"]), chunk):
        idx = op["idx"][i:i + chunk].to(DEV); val = op["val"][:, i:i + chunk].to(DEV).double()
        row = torch.div(idx, nsrc_f, rounding_mode="floor"); col = idx - row * nsrc_f
        fo = torch.div(row, Dout, rounding_mode="floor"); fi = torch.div(col, Din, rounding_mode="floor")
        nidx = (cof[fo] * Dout + (row - fo * Dout)) * nsrc_c + cof[fi] * Din + (col - fi * Din)
        I.append(nidx); V.append(val * wf[fi][None])
    idx = torch.cat(I); val = torch.cat(V, 1); u, inv = torch.unique(idx, return_inverse=True)
    acc = torch.zeros(3, len(u), dtype=torch.float64, device=DEV).index_add_(1, inv, val)
    out = dict(idx=u.cpu(), val=acc.float().cpu(), shape=(ncell * Dout, nsrc_c), Dout=Dout, Din=Din)
    if op.get("KC") is not None:
        KCf = op["KC"]; ncam, _, Pc, _ = KCf.shape
        colmap = (cof[:, None] * Din + torch.arange(Din, device=DEV)[None]).reshape(-1)
        wcol = wf.repeat_interleave(Din).float()
        KCc = torch.zeros(ncam, 3, Pc, nsrc_c, device=DEV)
        for k in range(ncam):
            KCc[k].index_add_(2, colmap, KCf[k].to(DEV) * wcol[None, None, :])
        out["KC"] = KCc.cpu(); out["cams"] = op.get("cams")
    del idx, val, acc; torch.cuda.empty_cache()
    return out


def save_op(op, path):
    torch.save(op, path)


def load_op(path):
    return torch.load(path)
