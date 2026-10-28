"""WL55 continuous oracle COR (protocol fnto_k55, stage_B.instruments.COR): a multi-domain path tracer with the WL53 TR medium physics
and NO boundary discretisation, used to split a composition error into interface (ports / bins / MC of K and C) and everything else.

Physics, identical to p53_tracer.walk inside a medium: free flight pdf sigma_t exp(-sigma_t s); at a collision the weight is multiplied
by the RGB albedo, a next-event estimate toward every camera direction is recorded at the port where that ray leaves the domain
(phase x exp(-sigma_t t)), a new direction is drawn from the phase function, and Russian roulette applies after 4 events when the
weight falls below 5 % of the initial weight. Index-matched interfaces: a photon leaving a medium continues in a straight line through
air (scene intersector) to the next boundary: a medium is entered, a Lambert surface records the arriving flux and reflects (cosine,
weight x rho), black / emitter surfaces record the arriving flux and absorb; misses escape. Photons are processed as a refilled pool
(per-photon event counters) to avoid long single-chunk tails. float64 accumulators (WL53 A2).

Sources: the collimated sun (positions on every lit triangle of every component with density proportional to projected area, shadow
test toward the sun) or Lambertian emitters (radiance profile L_e(x), cosine directions).
Readouts (per camera direction c):
  medium port   L = sum of NEE flux at the port / projected area (same estimator as KC; unscattered light excluded)
  Lambert port  L = rho / pi * arriving flux / area
  ballistic     the unscattered term: backward ray from a port sample along -c through the medium (distance s) and on through air;
                L_ball = exp(-sigma_t s) x radiance of the seen surface toward c (COR's own per-triangle readout; emitter: analytic)
"""
import math
import time

import numpy as np
import torch

import k55_common as KCm  # noqa: F401
import k55_fixtures as KF
import p53_fixtures as FX
import p53_geom as G
import p53_tracer as TR

DEV = "cuda"
KIND = {"medium": 0, "lambert": 1, "black": 2, "emitter": 3}


def _cos_dir(n, gen):
    N = len(n); u1 = torch.rand(N, device=DEV, generator=gen); u2 = torch.rand(N, device=DEV, generator=gen)
    r = torch.sqrt(u1); ph = 2 * math.pi * u2; z = torch.sqrt((1 - u1).clamp(min=0))
    t1, t2 = TR._basis(n)
    return torch.nn.functional.normalize(r[:, None] * torch.cos(ph)[:, None] * t1 + r[:, None] * torch.sin(ph)[:, None] * t2 + z[:, None] * n, dim=1)


def _fine_floor(m, sub=4):
    """Re-tessellate a WL53 open floor grid with sub x sub cells per port (same plane, same port ids)."""
    P = len(m["port_area"]); n = int(round(math.sqrt(P))); h = math.sqrt(float(m["port_area"][0])); o = m["V"].min(0)
    return G.open_grid(n, n, h, (float(o[0]), float(o[1]), float(o[2])), sub=sub)


class COR:
    def __init__(self, sd, cam_keys, floor_sub=4, sigma_t=FX.SIGMA_T, albedo=FX.ALBEDO, g=FX.G_PHASE, rho=FX.FLOOR_RHO):
        import p53_mi as M
        self.sd = sd; self.cam_keys = list(cam_keys)
        V, F, tc, tp, n, t, b, ar, cen, kinds = [], [], [], [], [], [], [], [], [], []
        ov = op = 0; self.port_off = []; self.Pc = []
        for ci, (nm, k, m, Vc) in enumerate(sd["comps"]):
            if k == "lambert" and len(m["F"]) == 2 * len(m["port_area"]) and floor_sub > 1:
                m = _fine_floor(m, floor_sub); Vc = m["V"].copy()
            fr = G.tri_frames(m["V"], Vc, m["F"], m["tri_port"], m["port_tangent"])
            V.append(Vc); F.append(m["F"] + ov); tc.append(np.full(len(m["F"]), ci)); tp.append(m["tri_port"] + op)
            n.append(fr["n"]); t.append(fr["t"]); b.append(fr["b"]); ar.append(fr["area"]); cen.append(fr["centroid"]); kinds.append(KIND[k])
            self.port_off.append(op); self.Pc.append(len(m["port_area"])); ov += len(Vc); op += len(m["port_area"])
        self.Ptot = op
        f32 = lambda a: torch.tensor(np.concatenate(a), dtype=torch.float32, device=DEV)  # noqa: E731
        self.Vnp, self.Fnp = np.concatenate(V), np.concatenate(F)
        self.V = torch.tensor(self.Vnp, dtype=torch.float32, device=DEV); self.F = torch.tensor(self.Fnp, device=DEV)
        self.n, self.t, self.b, self.area, self.cen = f32(n), f32(t), f32(b), f32(ar), f32(cen)
        self.tri_comp = torch.tensor(np.concatenate(tc), device=DEV); self.tri_port = torch.tensor(np.concatenate(tp), device=DEV)
        self.ntri = len(self.Fnp)
        self.kind = torch.tensor(kinds, device=DEV)
        nc = len(kinds)
        self.alb = torch.tensor(np.broadcast_to(np.asarray(albedo, float), (3,)), dtype=torch.float32, device=DEV)[None].expand(nc, 3).contiguous()
        self.rho = torch.full((nc, 3), float(rho), device=DEV); self.sig = float(sigma_t); self.g = float(g)
        self.port_area = torch.zeros(self.Ptot, device=DEV, dtype=torch.float64).index_add_(0, self.tri_port, self.area.double())
        self.cams = torch.tensor(np.array([FX.DIRS[k] for k in self.cam_keys], float), dtype=torch.float32, device=DEV)
        self.ap_tri = (self.n[None] * self.cams[:, None]).sum(-1).clamp(min=0) * self.area[None]                         # (ncam, ntri)
        self.ap_port = torch.stack([torch.zeros(self.Ptot, device=DEV, dtype=torch.float64).index_add_(0, self.tri_port, a.double()) for a in self.ap_tri])
        self.emit = {ci: KF.profile(sd["emit"][nm]) for ci, (nm, k, m, Vc) in enumerate(sd["comps"]) if k == "emitter"}
        self.isect = M.Intersector(self.Vnp, self.Fnp)
        self.eps = 2e-6 * max(1.0, float(np.abs(self.Vnp).max()))

    # ------------------------------------------------------------------ sources
    def _sun(self, N, gen, s_travel, E):
        s = torch.tensor(np.asarray(s_travel, float) / np.linalg.norm(s_travel), dtype=torch.float32, device=DEV)
        wt = self.area * (-(self.n * s).sum(1)).clamp(min=0); total = float(wt.sum())
        tri = torch.multinomial(wt, N, replacement=True, generator=gen)
        x = G.sample_on_tris(self.V, self.F, tri, torch.rand(N, device=DEV, generator=gen), torch.rand(N, device=DEV, generator=gen))
        tt, _ = self.isect(x + 4 * self.eps * self.n[tri], (-s).expand(N, 3).contiguous()); ok = ~torch.isfinite(tt)
        w = torch.full((N, 3), E * total / N, device=DEV)
        return tri[ok], x[ok], s.expand(int(ok.sum()), 3).contiguous(), w[ok]

    def _emitter(self, N, gen):
        sel = torch.zeros(self.ntri, dtype=torch.bool, device=DEV); Lc = torch.zeros(self.ntri, device=DEV)
        for ci, f in self.emit.items():
            m = self.tri_comp == ci; sel |= m; Lc[m] = f(self.cen[m]).float()
        wt = self.area * Lc * sel.float(); tot = float(wt.sum())
        tri = torch.multinomial(wt, N, replacement=True, generator=gen)
        x = G.sample_on_tris(self.V, self.F, tri, torch.rand(N, device=DEV, generator=gen), torch.rand(N, device=DEV, generator=gen))
        Lx = torch.zeros(N, device=DEV)
        for ci, f in self.emit.items():
            m = self.tri_comp[tri] == ci; Lx[m] = f(x[m]).float()
        w = (math.pi * Lx * tot / (Lc[tri] * N))[:, None].expand(N, 3).contiguous()
        d = _cos_dir(self.n[tri], gen)
        return x + self.eps * self.n[tri], d, w, math.pi * tot

    # ------------------------------------------------------------------ transport
    def run(self, N, seed, source="sun", pool=4_000_000, log=None, max_iter=400000):
        gen = torch.Generator(device=DEV); gen.manual_seed(int(seed)); ncam = len(self.cams)
        A = dict(nee_port=torch.zeros(ncam, 3, self.Ptot, dtype=torch.float64, device=DEV), nee_tri=torch.zeros(ncam, 3, self.ntri, dtype=torch.float64, device=DEV),
                 arr_tri=torch.zeros(3, self.ntri, dtype=torch.float64, device=DEV), esc=torch.zeros(3, dtype=torch.float64, device=DEV))
        st = dict(launched=0, source_flux=np.zeros(3), leaks=0, wrong=0, back_hits=0, nee_invalid=0, collisions=0, iters=0, entries=0, lambert_bounces=0)
        S = dict(x=torch.empty(0, 3, device=DEV), d=torch.empty(0, 3, device=DEV), w=torch.empty(0, 3, device=DEV), w0=torch.empty(0, device=DEV),
                 med=torch.empty(0, dtype=torch.bool, device=DEV), comp=torch.empty(0, dtype=torch.long, device=DEV), ns=torch.empty(0, dtype=torch.long, device=DEV))
        remaining = int(N); t0 = time.time(); src_flux = 0.0
        s_travel, E = (self.sd["lights"][0] if source == "sun" else (None, None))
        while remaining > 0 or len(S["x"]):
            st["iters"] += 1
            if st["iters"] > max_iter:
                break
            if remaining > 0 and len(S["x"]) < pool // 2:
                n_new = min(pool - len(S["x"]), remaining); remaining -= n_new; st["launched"] += n_new
                if source == "sun":                                   # every batch carries n_new / N of the source flux
                    tri, xh, d, w = self._sun(n_new, gen, s_travel, E); w = w * (n_new / N); st["source_flux"] += w.double().sum(0).cpu().numpy()
                    new = self._arrive(A, st, tri, xh, d, w, w.max(1).values, torch.zeros(len(tri), dtype=torch.long, device=DEV), gen)
                else:
                    x, d, w, ftot = self._emitter(n_new, gen); w = w * (n_new / N); st["source_flux"] += w.double().sum(0).cpu().numpy()
                    new = dict(x=x, d=d, w=w, w0=w.max(1).values, med=torch.zeros(len(x), dtype=torch.bool, device=DEV),
                               comp=torch.full((len(x),), -1, dtype=torch.long, device=DEV), ns=torch.zeros(len(x), dtype=torch.long, device=DEV))
                S = {k: torch.cat([S[k], new[k]]) for k in S}
            if not len(S["x"]):
                continue
            S = self._step(A, st, S, gen)
            if log and st["iters"] % 2000 == 0:
                log(f"COR it {st['iters']} alive {len(S['x'])} remaining {remaining} {time.time() - t0:.0f}s")
        st["seconds"] = time.time() - t0; st["truncated_alive"] = int(len(S["x"]))
        return self._finish(A, st)

    def _arrive(self, A, st, tri, xh, d, w, w0, ns, gen):
        """Photons arriving through air at triangles tri (points xh, direction d)."""
        nn = self.n[tri]; front = (d * nn).sum(1) < 0; st["back_hits"] += int((~front).sum())
        tri, xh, d, w, w0, ns, nn = tri[front], xh[front], d[front], w[front], w0[front], ns[front], nn[front]
        cp = self.tri_comp[tri]; kd = self.kind[cp]
        nonmed = kd != 0
        if nonmed.any():
            A["arr_tri"].index_add_(1, tri[nonmed], w[nonmed].double().T.contiguous())
        out = []
        m = kd == 0
        if m.any():
            st["entries"] += int(m.sum())
            out.append(dict(x=xh[m] - self.eps * nn[m], d=d[m], w=w[m], w0=w0[m], med=torch.ones(int(m.sum()), dtype=torch.bool, device=DEV), comp=cp[m], ns=ns[m]))
        m = kd == 1
        if m.any():
            st["lambert_bounces"] += int(m.sum())
            wl = w[m] * self.rho[cp[m]]; nl = ns[m] + 1; w0l = w0[m]
            wl, keep = self._rr(wl, w0l, nl, gen)
            nd = _cos_dir(nn[m][keep], gen)
            out.append(dict(x=xh[m][keep] + self.eps * nn[m][keep], d=nd, w=wl[keep], w0=w0l[keep], med=torch.zeros(int(keep.sum()), dtype=torch.bool, device=DEV),
                            comp=torch.full((int(keep.sum()),), -1, dtype=torch.long, device=DEV), ns=nl[keep]))
        if not out:
            return dict(x=torch.empty(0, 3, device=DEV), d=torch.empty(0, 3, device=DEV), w=torch.empty(0, 3, device=DEV), w0=torch.empty(0, device=DEV),
                        med=torch.empty(0, dtype=torch.bool, device=DEV), comp=torch.empty(0, dtype=torch.long, device=DEV), ns=torch.empty(0, dtype=torch.long, device=DEV))
        return {k: torch.cat([o[k] for o in out]) for k in out[0]}

    def _rr(self, w, w0, ns, gen, rr_rel=0.05, rr_min=4):
        r = w.max(1).values / w0.clamp(min=1e-30); low = (r < rr_rel) & (ns >= rr_min)
        p = (r / rr_rel).clamp(max=1); u = torch.rand(len(r), device=DEV, generator=gen)
        keep = ~low | (u < p)
        w = torch.where(low[:, None], w / p[:, None].clamp(min=1e-30), w)
        return w, keep

    def _step(self, A, st, S, gen):
        x, d, w, w0, med, comp, ns = S["x"], S["d"], S["w"], S["w0"], S["med"], S["comp"], S["ns"] + 1
        t, prim = self.isect(x, d); miss = ~torch.isfinite(t); out = []
        # ---- air
        air = ~med
        esc = air & miss
        if esc.any():
            A["esc"] += w[esc].double().sum(0)
        hit = air & ~miss
        if hit.any():
            j = hit.nonzero().squeeze(1); xh = x[j] + t[j, None] * d[j]
            out.append(self._arrive(A, st, prim[j], xh, d[j], w[j], w0[j], ns[j], gen))
        # ---- medium
        leak = med & miss; st["leaks"] += int(leak.sum())
        s = -torch.log1p(-torch.rand(len(x), device=DEV, generator=gen)) / self.sig
        coll = med & ~miss & (s < t); ex = med & ~miss & ~coll
        if ex.any():
            j = ex.nonzero().squeeze(1); pj = prim[j]; ok = (self.tri_comp[pj] == comp[j]) & ((d[j] * self.n[pj]).sum(1) > 0)
            st["wrong"] += int((~ok).sum()); j, pj = j[ok], pj[ok]
            xe = x[j] + t[j, None] * d[j] + self.eps * self.n[pj]
            out.append(dict(x=xe, d=d[j], w=w[j], w0=w0[j], med=torch.zeros(len(j), dtype=torch.bool, device=DEV), comp=torch.full((len(j),), -1, dtype=torch.long, device=DEV), ns=ns[j]))
        if coll.any():
            j = coll.nonzero().squeeze(1); st["collisions"] += len(j)
            xc = x[j] + s[j, None] * d[j]; dc = d[j]; cj = comp[j]; wc = w[j] * self.alb[cj]
            for k, c in enumerate(self.cams):
                ce = c.expand(len(xc), 3).contiguous(); te, pe = self.isect(xc, ce)
                val = torch.isfinite(te); vi = val.nonzero().squeeze(1)
                bad = (self.tri_comp[pe[vi]] != cj[vi]) | ((self.n[pe[vi]] * c).sum(1) <= 0); val[vi[bad]] = False
                st["nee_invalid"] += int((~val).sum())
                if val.any():
                    iv = val.nonzero().squeeze(1)
                    contrib = (wc[iv] * (TR.phase_eval(dc[iv], ce[iv], self.g) * torch.exp(-self.sig * te[iv]))[:, None]).double().T.contiguous()
                    A["nee_port"][k].index_add_(1, self.tri_port[pe[iv]], contrib); A["nee_tri"][k].index_add_(1, pe[iv], contrib)
            dn = TR.phase_sample(dc, self.g, gen)
            wc, keep = self._rr(wc, w0[j], ns[j], gen)
            out.append(dict(x=xc[keep], d=dn[keep], w=wc[keep], w0=w0[j][keep], med=torch.ones(int(keep.sum()), dtype=torch.bool, device=DEV), comp=cj[keep], ns=ns[j][keep]))
        if not out:
            return {k: v[:0] for k, v in S.items()}
        return {k: torch.cat([o[k] for o in out]) for k in out[0]}

    def _finish(self, A, st):
        ncam = len(self.cams); L = torch.zeros(ncam, self.Ptot, 3, dtype=torch.float64, device=DEV)
        port_comp = torch.zeros(self.Ptot, dtype=torch.long, device=DEV).index_put_((self.tri_port,), self.tri_comp)
        pk = self.kind[port_comp]
        arr_port = torch.zeros(3, self.Ptot, dtype=torch.float64, device=DEV).index_add_(1, self.tri_port, A["arr_tri"])
        for k in range(ncam):
            Lm = (A["nee_port"][k] / self.ap_port[k].clamp(min=1e-30)).T
            Ll = (arr_port / self.port_area).T * self.rho[port_comp].double() / math.pi
            L[k] = torch.where((pk == 0)[:, None], Lm, torch.where((pk == 1)[:, None], Ll, torch.zeros_like(Lm)))
            L[k][(pk == 0) & (self.ap_port[k] <= 0)] = 0
        self.A = A
        st = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
        return dict(L=L.cpu().numpy(), arr_tri=A["arr_tri"].cpu().numpy(), arr_port=arr_port.cpu().numpy(), esc=A["esc"].cpu().numpy(), stats=st)

    # ------------------------------------------------------------------ ballistic readout
    def tri_radiance(self, k):
        """Per-triangle radiance toward camera k from this run (medium: NEE; Lambert: rho/pi E; others 0)."""
        A = self.A; kd = self.kind[self.tri_comp]
        Lm = (A["nee_tri"][k] / self.ap_tri[k].double().clamp(min=1e-30)).T
        Ll = (A["arr_tri"] / self.area.double()).T * self.rho[self.tri_comp].double() / math.pi
        return torch.where((kd == 0)[:, None], Lm, torch.where((kd == 1)[:, None], Ll, torch.zeros_like(Lm)))

    def ballistic(self, k, n_per_port=4096, seed=0):
        """Unscattered camera term per medium port (Ptot, 3) toward camera k."""
        gen = torch.Generator(device=DEV); gen.manual_seed(int(seed)); c = self.cams[k]; Lt = self.tri_radiance(k)
        out = torch.zeros(self.Ptot, 3, dtype=torch.float64, device=DEV)
        med_ports = torch.unique(self.tri_port[self.kind[self.tri_comp] == 0])
        tri_w = self.ap_tri[k]
        for i in range(0, len(med_ports), 256):
            ports = med_ports[i:i + 256]
            sel = torch.isin(self.tri_port, ports) & (tri_w > 0)
            if not sel.any():
                continue
            tris = sel.nonzero().squeeze(1); pw = torch.zeros(self.Ptot, device=DEV).index_add_(0, self.tri_port[tris], tri_w[tris])
            # stratified: n_per_port samples per visible port, triangles proportional to projected area within the port
            vis_ports = ports[pw[ports] > 0]
            if not len(vis_ports):
                continue
            rep = vis_ports.repeat_interleave(n_per_port)
            # choose triangle within port
            u = torch.rand(len(rep), device=DEV, generator=gen)
            tr_port = self.tri_port[tris]; order = torch.argsort(tr_port, stable=True); tris_s = tris[order]; tp_s = tr_port[order]
            cw = tri_w[tris_s]; cum = torch.cumsum(cw, 0)
            first = torch.searchsorted(tp_s, rep); base = torch.where(first > 0, cum[(first - 1).clamp(min=0)], torch.zeros_like(u))
            target = base + u * pw[rep]
            idx = torch.searchsorted(cum, target).clamp(max=len(cum) - 1); tri = tris_s[idx]
            y = G.sample_on_tris(self.V, self.F, tri, torch.rand(len(rep), device=DEV, generator=gen), torch.rand(len(rep), device=DEV, generator=gen))
            o = y - self.eps * self.n[tri]; dm = (-c).expand(len(rep), 3).contiguous()
            t1, p1 = self.isect(o, dm); ok = torch.isfinite(t1)
            ok[ok.clone()] &= (self.tri_comp[p1[ok]] == self.tri_comp[tri[ok]]) & ((self.n[p1[ok]] * (-c)).sum(1) > 0)
            val = torch.zeros(len(rep), 3, dtype=torch.float64, device=DEV)
            j = ok.nonzero().squeeze(1)
            x1 = o[j] + t1[j, None] * dm[j] + self.eps * self.n[p1[j]]
            t2, p2 = self.isect(x1, dm[j]); hit = torch.isfinite(t2)
            jj = j[hit]; p2h = p2[hit]; front = (self.n[p2h] * c).sum(1) > 0
            jj, p2h = jj[front], p2h[front]; xh = (x1[hit][front] + t2[hit][front, None] * dm[jj])
            Ls = Lt[p2h].clone()
            for ci, f in self.emit.items():
                m = self.tri_comp[p2h] == ci
                if m.any():
                    Ls[m] = f(xh[m]).double()[:, None].expand(int(m.sum()), 3)
            val[jj] = Ls * torch.exp(-self.sig * t1[jj].double())[:, None]
            out.index_add_(0, rep, val / n_per_port)
        return out.cpu().numpy()
