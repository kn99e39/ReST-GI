"""WL53 oracle intrinsic-K tracer TR and independent box walk IW (protocol phys_ref53).

TR solves the radiative transfer equation inside ONE optical domain in isolation by Monte Carlo:
  * homogeneous medium (achromatic sigma_t, RGB single-scattering albedo, isotropic or Henyey-Greenstein phase);
  * index-matched boundary (no reflection or refraction at the interface);
  * photons enter through the boundary, free-fly with pdf sigma_t exp(-sigma_t s), scatter with albedo weight, and are
    recorded at their FIRST exit into air.
Nothing outside the domain exists for TR, so K holds exactly the in-medium path family and every air segment is left to C
(path-ownership rule of the protocol). Mitsuba is used only as a ray-triangle intersector.

Run outputs (all per unit source flux):
  T    (3, P*Dout, nsrc)  flux leaving through (port, out-bin)
  KC   (ncam, 3, P, nsrc) port-average radiance toward each camera direction, by next-event estimation at every collision
                          (excludes the unscattered term: e^-10 for the 1 cm fixtures)
  S2   (3, P, Psrc)       per-port-pair second moment of the exit weights (entry-level variance)
  stats                   photons, exits, leaks (ray miss inside a closed domain), wrong-side exits, truncations, steps

IW is an independent estimator for an axis-aligned box: analytic exits, no Mitsuba, no mesh, its own sampling code. It is
used for the semi-infinite H-function control and the slab cross-check.
"""
import math
import time

import numpy as np
import torch

import p53_geom as G

DEV = "cuda"
INV4PI = 1.0 / (4 * math.pi)


def _basis(d):
    s = torch.where(d[:, 2] >= 0, 1.0, -1.0)
    a = -1.0 / (s + d[:, 2]); b = d[:, 0] * d[:, 1] * a
    t1 = torch.stack([1 + s * d[:, 0] ** 2 * a, s * b, -s * d[:, 0]], 1)
    t2 = torch.stack([b, s + d[:, 1] ** 2 * a, -d[:, 1]], 1)
    return t1, t2


def phase_sample(d, g, gen):
    n = len(d)
    u1 = torch.rand(n, device=DEV, generator=gen); u2 = torch.rand(n, device=DEV, generator=gen)
    if g == 0:
        c = 1 - 2 * u1
    else:
        q = (1 - g * g) / (1 - g + 2 * g * u1); c = (1 + g * g - q * q) / (2 * g)
    c = c.clamp(-1, 1); sn = torch.sqrt(torch.clamp(1 - c * c, min=0)); ph = 2 * math.pi * u2
    t1, t2 = _basis(d)
    out = sn[:, None] * torch.cos(ph)[:, None] * t1 + sn[:, None] * torch.sin(ph)[:, None] * t2 + c[:, None] * d
    return torch.nn.functional.normalize(out, dim=1)


def phase_eval(d, w, g):
    c = (d * w).sum(-1)
    if g == 0:
        return torch.full_like(c, INV4PI)
    return INV4PI * (1 - g * g) / (1 + g * g - 2 * g * c).clamp(min=1e-12) ** 1.5


class Domain:
    """One optical domain: rest mesh (ports, rest tangents), current vertices, medium, intersector."""

    def __init__(self, mesh, Vcur, sigma_t, albedo, g=0.0, intersector=None, name=""):
        import p53_mi as M
        self.name = name; self.mesh = mesh; self.sigma_t = float(sigma_t); self.g = float(g)
        self.albedo = torch.tensor(np.broadcast_to(np.asarray(albedo, float), (3,)).copy(), dtype=torch.float32, device=DEV)
        F = mesh["F"]; fr = G.tri_frames(mesh["V"], Vcur, F, mesh["tri_port"], mesh["port_tangent"])
        f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=DEV)  # noqa: E731
        self.Vcur = np.asarray(Vcur); self.V = f32(Vcur); self.F = torch.tensor(F, device=DEV)
        self.n, self.t, self.b, self.area = f32(fr["n"]), f32(fr["t"]), f32(fr["b"]), f32(fr["area"])
        self.tri_port = torch.tensor(mesh["tri_port"], device=DEV); self.P = len(mesh["port_area"])
        self.port_area = torch.zeros(self.P, device=DEV, dtype=torch.float64).index_add_(0, self.tri_port, self.area.double()).float()
        order = np.argsort(mesh["tri_port"], kind="stable"); k = len(F) // self.P
        assert k * self.P == len(F) and np.all(mesh["tri_port"][order].reshape(self.P, k) == np.arange(self.P)[:, None])
        self.port_tris = torch.tensor(order.reshape(self.P, k), device=DEV)
        a = self.area[self.port_tris]; self.port_cum = torch.cumsum(a, 1) / a.sum(1, keepdim=True)
        self.isect = intersector or M.Intersector(Vcur, F)
        self.eps = 2e-6 * max(1.0, float(np.abs(Vcur).max()))

    def proj_area(self, c):
        """Projected area of every port toward world direction c (torch (3,))."""
        cs = (self.n * c[None, :]).sum(1).clamp(min=0)
        return torch.zeros(self.P, device=DEV).index_add_(0, self.tri_port, self.area * cs)


# ------------------------------------------------------------------ sources


def cell_source(dom, cells, n_per, Din, gen):
    """Photons for input cells (port, in-bin): uniform on the port, uniform in projected solid angle inside the bin; the
    bin labels the outward direction the light comes FROM, so the photon travels along -w. Unit flux per cell."""
    port = (cells // Din).repeat_interleave(n_per); ib = (cells % Din).repeat_interleave(n_per)
    N = len(port); r = lambda: torch.rand(N, device=DEV, generator=gen)  # noqa: E731
    k = (r()[:, None] > dom.port_cum[port]).sum(1).clamp(max=dom.port_tris.shape[1] - 1)
    tri = dom.port_tris[port, k]
    x = G.sample_on_tris(dom.V, dom.F, tri, r(), r())
    wl = G.bin_sample(ib, int(round(math.sqrt(Din))), r(), r()).float()
    d = -G.to_world(wl, dom.n[tri], dom.t[tri], dom.b[tri])
    w = torch.full((N, 3), 1.0 / n_per, device=DEV)
    return x, torch.nn.functional.normalize(d, dim=1), dom.n[tri], w, cells.repeat_interleave(n_per)


def sun_source(dom, s_travel, E, N, gen, shadow=None, tri_mask=None):
    """Collimated beam (travel direction s, perpendicular irradiance E) entering the domain; positions on lit, unshadowed
    boundary triangles with density proportional to projected area. shadow = Intersector of the FULL scene (light that
    first hits any other boundary is that domain's input, not this one's)."""
    s = torch.tensor(np.asarray(s_travel, float) / np.linalg.norm(s_travel), dtype=torch.float32, device=DEV)
    cos = -(dom.n * s[None, :]).sum(1)
    wt = dom.area * cos.clamp(min=0)
    if tri_mask is not None:
        wt = wt * torch.as_tensor(tri_mask, device=DEV, dtype=torch.float32)
    total = float(wt.sum())
    tri = torch.multinomial(wt, N, replacement=True, generator=gen)
    x = G.sample_on_tris(dom.V, dom.F, tri, torch.rand(N, device=DEV, generator=gen), torch.rand(N, device=DEV, generator=gen))
    ok = torch.ones(N, dtype=torch.bool, device=DEV)
    if shadow is not None:
        tt, _ = shadow(x + 4 * dom.eps * dom.n[tri], -s.expand(N, 3)); ok = ~torch.isfinite(tt)
    w = (E * total / N) * ok.float()[:, None].expand(N, 3).contiguous()
    keep = ok
    return x[keep], s.expand(N, 3)[keep].contiguous(), dom.n[tri][keep], w[keep], torch.zeros(int(keep.sum()), dtype=torch.long, device=DEV)


# ------------------------------------------------------------------ walk


class Accum:
    def __init__(self, dom, Dout, Din, nsrc, cams=None, psrc=None):
        self.dom, self.Dout, self.Din, self.nsrc = dom, Dout, Din, nsrc
        self.Nout = int(round(math.sqrt(Dout)))
        # float64 accumulators (amendment 2): float32 atomics lost tiny NEE increments once a bin held ~2^22 of them
        self.T = torch.zeros(3, dom.P * Dout * nsrc, device=DEV, dtype=torch.float64)
        self.psrc = psrc if psrc is not None else max(1, nsrc // Din)
        self.S2 = torch.zeros(3, dom.P * self.psrc, device=DEV, dtype=torch.float64)
        self.cams = None if cams is None else torch.tensor(np.asarray(cams, float), dtype=torch.float32, device=DEV).reshape(-1, 3)
        self.KC = None if cams is None else torch.zeros(len(self.cams), 3, dom.P * nsrc, device=DEV, dtype=torch.float64)
        self.stats = dict(photons=0, exits=0, leaks=0, wrong_side=0, truncated=0, nee_invalid=0, steps=0, collisions=0,
                          flux_in=np.zeros(3), flux_out=np.zeros(3), flux_leak=np.zeros(3))

    def finalize(self):
        P, Dout, nsrc = self.dom.P, self.Dout, self.nsrc
        T = self.T.view(3, P * Dout, nsrc).float(); self.T = None; torch.cuda.empty_cache()
        out = dict(T=T, S2=self.S2.view(3, P, self.psrc).float(),
                   stats={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.stats.items()})
        if self.KC is not None:
            KC = self.KC.view(len(self.cams), 3, P, nsrc).float(); self.KC = None
            ap = torch.stack([self.dom.proj_area(c) for c in self.cams])                       # (ncam, P)
            KC /= ap[:, None, :, None].clamp(min=1e-30); KC[(ap <= 0)[:, None, :, None].expand_as(KC)] = 0; out["KC"] = KC
            out["proj_area"] = ap
        return out


def walk(acc, x, d, n_entry, w, src, gen, rr_rel=0.05, rr_min_steps=4, continue_air=False, max_steps=200000):
    dom = acc.dom; sig = dom.sigma_t; alb = dom.albedo
    x = x - dom.eps * n_entry; w0 = w.max(1).values.clamp(min=1e-30)
    st = acc.stats; st["photons"] += len(x); st["flux_in"] += w.double().sum(0).cpu().numpy()
    step = 0
    while len(x):
        step += 1
        if step > max_steps:
            st["truncated"] += len(x); break
        t, prim = dom.isect(x, d)
        s = -torch.log1p(-torch.rand(len(x), device=DEV, generator=gen)) / sig
        miss = ~torch.isfinite(t)
        coll = (s < t) & ~miss
        ex = ~coll & ~miss
        if miss.any():
            st["leaks"] += int(miss.sum()); st["flux_leak"] += w[miss].double().sum(0).cpu().numpy()
        keep_x, keep_d, keep_w, keep_s, keep_w0 = [], [], [], [], []
        if ex.any():
            ie = ex.nonzero().squeeze(1); tri = prim[ie]; dd = d[ie]; nn = dom.n[tri]
            ok = (dd * nn).sum(1) > 0
            st["wrong_side"] += int((~ok).sum())
            ie, tri, dd, nn = ie[ok], tri[ok], dd[ok], nn[ok]
            xe = x[ie] + t[ie, None] * dd
            if continue_air and len(ie):                                   # object-scoped controls: cross air inside the group
                t2, p2 = dom.isect(xe + dom.eps * nn, dd)
                re = torch.isfinite(t2)
                re_ok = re.clone(); re_ok[re] = (dd[re] * dom.n[p2[re]]).sum(1) < 0
                if re_ok.any():
                    j = re_ok.nonzero().squeeze(1)
                    xr = xe[j] + (dom.eps + t2[j, None]) * dd[j] - dom.eps * dom.n[p2[j]]
                    keep_x.append(xr); keep_d.append(dd[j]); keep_w.append(w[ie[j]]); keep_s.append(src[ie[j]]); keep_w0.append(w0[ie[j]])
                rec = ~re_ok if continue_air != "dup" else torch.ones_like(re_ok)    # "dup": ALSO record the exit (double-count control)
                ie, tri, dd, nn = ie[rec], tri[rec], dd[rec], nn[rec]
            if len(ie):
                port = dom.tri_port[tri]
                wl = G.to_local(dd, nn, dom.t[tri], dom.b[tri])
                ob = G.local_to_bin(wl, acc.Nout)
                we = w[ie]; se = src[ie]
                acc.T.index_add_(1, (port * acc.Dout + ob) * acc.nsrc + se, we.T.double())
                acc.S2.index_add_(1, port * acc.psrc + torch.div(se, acc.Din, rounding_mode="floor").clamp(max=acc.psrc - 1), (we.double() ** 2).T.contiguous())
                st["exits"] += len(ie); st["flux_out"] += we.double().sum(0).cpu().numpy()
        if coll.any():
            ic = coll.nonzero().squeeze(1)
            xc = x[ic] + s[ic, None] * d[ic]; dc = d[ic]; wc = w[ic] * alb[None, :]; sc = src[ic]; w0c = w0[ic]
            st["collisions"] += len(ic)
            if acc.KC is not None:
                for k, c in enumerate(acc.cams):
                    ce = c.expand(len(xc), 3).contiguous()
                    te, pe = dom.isect(xc, ce)
                    val = torch.isfinite(te)
                    vi = val.nonzero().squeeze(1)
                    val[vi[(dom.n[pe[vi]] * c).sum(1) <= 0]] = False
                    st["nee_invalid"] += int((~val).sum())
                    if val.any():
                        iv = val.nonzero().squeeze(1)
                        contrib = wc[iv] * (phase_eval(dc[iv], ce[iv], dom.g) * torch.exp(-sig * te[iv]))[:, None]
                        acc.KC[k].index_add_(1, dom.tri_port[pe[iv]] * acc.nsrc + sc[iv], contrib.T.double())
            dn = phase_sample(dc, dom.g, gen)
            if step >= rr_min_steps:
                r = wc.max(1).values / w0c; low = r < rr_rel
                if low.any():
                    p = (r / rr_rel).clamp(max=1); u = torch.rand(len(r), device=DEV, generator=gen)
                    surv = ~low | (u < p)
                    wc = torch.where(low[:, None], wc / p[:, None].clamp(min=1e-30), wc)
                    xc, dn, wc, sc, w0c = xc[surv], dn[surv], wc[surv], sc[surv], w0c[surv]
            keep_x.append(xc); keep_d.append(dn); keep_w.append(wc); keep_s.append(sc); keep_w0.append(w0c)
        if keep_x:
            x = torch.cat(keep_x); d = torch.cat(keep_d); w = torch.cat(keep_w); src = torch.cat(keep_s); w0 = torch.cat(keep_w0)
        else:
            break
    st["steps"] = max(st["steps"], step)


def run_cells(dom, Dout, Din, n_per, seed, cams=None, cells=None, chunk_photons=4_000_000, continue_air=False, log=None):
    gen = torch.Generator(device=DEV); gen.manual_seed(int(seed))
    nsrc = dom.P * Din
    acc = Accum(dom, Dout, Din, nsrc, cams, psrc=dom.P)
    cells = torch.arange(nsrc, device=DEV) if cells is None else cells
    per = max(1, chunk_photons // n_per); t0 = time.time()
    for i in range(0, len(cells), per):
        x, d, n, w, src = cell_source(dom, cells[i:i + per], n_per, Din, gen)
        walk(acc, x, d, n, w, src, gen, continue_air=continue_air)
        if log:
            log(f"{dom.name} cells {min(i + per, len(cells))}/{len(cells)} {time.time() - t0:.1f}s steps {acc.stats['steps']}")
    out = acc.finalize(); out["seconds"] = time.time() - t0; out["n_per"] = n_per
    return out


def run_sun(dom, s_travel, E, N, seed, Dout, cams=None, shadow=None, tri_mask=None, chunk_photons=4_000_000, continue_air=False):
    gen = torch.Generator(device=DEV); gen.manual_seed(int(seed))
    acc = Accum(dom, Dout, 1, 1, cams, psrc=1); t0 = time.time()
    for i in range(0, N, chunk_photons):
        n = min(chunk_photons, N - i)
        x, d, nn, w, src = sun_source(dom, s_travel, E * n / N, n, gen, shadow=shadow, tri_mask=tri_mask)
        walk(acc, x, d, nn, w, src, gen, continue_air=continue_air)
    out = acc.finalize(); out["seconds"] = time.time() - t0
    return out


# ------------------------------------------------------------------ IW: independent analytic box walk


def iw_box(lo, hi, sigma_t, albedo, g, s_travel, E, region, N, cams, seed, port_h=0.01, count_region=None, chunk=4_000_000, max_steps=200000):
    """Box [lo, hi] (np 3), beam of travel direction s through the top face z = hi over region (x0, x1, y0, y1).
    Returns per camera direction the next-event radiant intensity on the top and bottom faces binned on a port_h grid
    (dict face -> (nx, ny, 3)), plus reflected / transmitted / absorbed / leaked flux fractions. Independent of Mitsuba."""
    lo = torch.tensor(lo, dtype=torch.float64, device=DEV); hi = torch.tensor(hi, dtype=torch.float64, device=DEV)
    alb = torch.tensor(np.broadcast_to(np.asarray(albedo, float), (3,)).copy(), dtype=torch.float64, device=DEV)
    s = torch.tensor(np.asarray(s_travel, float) / np.linalg.norm(s_travel), dtype=torch.float64, device=DEV)
    cams = torch.tensor(np.asarray(cams, float), dtype=torch.float64, device=DEV).reshape(-1, 3)
    gen = torch.Generator(device=DEV); gen.manual_seed(int(seed))
    nx = int(round(float(hi[0] - lo[0]) / port_h)); ny = int(round(float(hi[1] - lo[1]) / port_h))
    I = {f: torch.zeros(len(cams), nx, ny, 3, dtype=torch.float64, device=DEV) for f in ("top", "bot")}
    flux = dict(R=torch.zeros(3, dtype=torch.float64, device=DEV), T=torch.zeros(3, dtype=torch.float64, device=DEV), side=torch.zeros(3, dtype=torch.float64, device=DEV))
    x0, x1, y0, y1 = region; area = (x1 - x0) * (y1 - y0); cos0 = float(-s[2]); P_in = E * cos0 * area

    def exit_t(x, d):
        tp = torch.where(d > 0, (hi - x) / d, torch.where(d < 0, (lo - x) / d, torch.full_like(x, float("inf"))))
        tm, ax = tp.min(1)
        return tm, ax

    def tally(face_mask, xe, ce_idx, val):
        for f, z in (("top", hi[2]), ("bot", lo[2])):
            m = face_mask & (torch.abs(xe[:, 2] - z) < 1e-9)
            if m.any():
                ix = ((xe[m, 0] - lo[0]) / port_h).long().clamp(0, nx - 1); iy = ((xe[m, 1] - lo[1]) / port_h).long().clamp(0, ny - 1)
                I[f][ce_idx].view(-1, 3).index_add_(0, ix * ny + iy, val[m])

    for i0 in range(0, N, chunk):
        n = min(chunk, N - i0)
        u = torch.rand(n, 2, dtype=torch.float64, device=DEV, generator=gen)
        x = torch.stack([x0 + (x1 - x0) * u[:, 0], y0 + (y1 - y0) * u[:, 1], hi[2].expand(n)], 1)
        d = s.expand(n, 3).clone(); w = torch.full((n, 3), P_in / N, dtype=torch.float64, device=DEV)
        step = 0
        while len(x) and step < max_steps:
            step += 1
            te, ax = exit_t(x, d)
            ss = -torch.log1p(-torch.rand(len(x), dtype=torch.float64, device=DEV, generator=gen)) / sigma_t
            coll = ss < te
            ex = ~coll
            if ex.any():
                xe = x[ex] + te[ex, None] * d[ex]; we = w[ex]; a = ax[ex]
                top = (a == 2) & (d[ex, 2] > 0); bot = (a == 2) & (d[ex, 2] < 0)
                flux["R"] += we[top].sum(0); flux["T"] += we[bot].sum(0); flux["side"] += we[~top & ~bot].sum(0)
            xc = x[coll] + ss[coll, None] * d[coll]; wc = w[coll] * alb; dc = d[coll]
            for k, c in enumerate(cams):
                ce = c.expand(len(xc), 3)
                t2, a2 = exit_t(xc, ce)
                xe = xc + t2[:, None] * ce
                pv = (phase_eval(dc, ce, g) * torch.exp(-sigma_t * t2))[:, None] * wc
                tally((a2 == 2), xe, k, pv)
            if g == 0:
                cz = 1 - 2 * torch.rand(len(xc), dtype=torch.float64, device=DEV, generator=gen)
                ph = 2 * math.pi * torch.rand(len(xc), dtype=torch.float64, device=DEV, generator=gen)
                sn = torch.sqrt((1 - cz * cz).clamp(min=0))
                dn = torch.stack([sn * torch.cos(ph), sn * torch.sin(ph), cz], 1)
            else:
                dn = phase_sample(dc.float(), g, gen).double()
            r = wc.max(1).values / (P_in / N); low = (r < 0.05) & (step >= 4)
            p = (r / 0.05).clamp(max=1); surv = ~low | (torch.rand(len(r), dtype=torch.float64, device=DEV, generator=gen) < p)
            wc = torch.where(low[:, None], wc / p[:, None].clamp(min=1e-30), wc)
            x, d, w = xc[surv], dn[surv], wc[surv]
    res = {}
    for f, sgn in (("top", 1.0), ("bot", -1.0)):
        cosf = (cams[:, 2] * sgn).clamp(min=0)                                      # projected-area factor per camera
        res[f] = (I[f] / (port_h * port_h * cosf.clamp(min=1e-30))[:, None, None, None]).cpu().numpy()
    out = dict(radiance=res, P_in=P_in, R=(flux["R"] / P_in).cpu().numpy(), T=(flux["T"] / P_in).cpu().numpy(), side=(flux["side"] / P_in).cpu().numpy())
    if count_region is not None:
        cx0, cx1, cy0, cy1 = count_region
        ix0 = int(round((cx0 - float(lo[0])) / port_h)); ix1 = int(round((cx1 - float(lo[0])) / port_h))
        iy0 = int(round((cy0 - float(lo[1])) / port_h)); iy1 = int(round((cy1 - float(lo[1])) / port_h))
        out["center_radiance_top"] = res["top"][:, ix0:ix1, iy0:iy1].reshape(len(cams), -1, 3).mean(1)
    return out
