"""WL115 tracer: WL114 tracer114 (deep physical suffix estimator, contract v3, C semantics) with three experiment-local additions
for the within-patch spatial representation study. Nothing in tracer112 / tracer113 / tracer114 or a shared file is edited.

  (i)   continuous within-patch coordinates (a, b) in [0, 1)^2 of a hit (the patch's own parameters: rect e1 / e2, sphere phi / z;
        identical formula to tracer113.subcell before flooring) and a K x K address for any K (K = 4 reproduces tracer113.subcell),
  (ii)  construct_k: the deep U-measure construction (tracer114.construct_deep, same proposal sampling, pinned first vertex,
        exclusion bookkeeping, deep probe) on a K x K sub-cell grid for any K and an optional patch subset, with the per-cell sums
        split by launch parity (two independent half tables for the table-noise estimate),
  (iii) readout_multi: physical LOW (camera events 1-2, the unchanged trace112 prefix) plus the event-2 lookup of any number of
        representations, each linear in its table and non-negative for non-negative tables:
          'pc'  piecewise constant on a per-patch K_p x K_p grid (flat table with per-patch offsets),
          'bl'  bilinear interpolation of K x K cell-centre values over the surface-global cell lattice of each rectangle (clamped
                at its edges) and of the sphere (periodic in phi, clamped at the poles); every neighbour value is read in ITS patch's
                direction bin of the query direction (exact per-neighbour rebinning),
  (iv)  deep_capture: a diagnostic deep camera pass that also accumulates, per event-2 cell, the prefix throughput and the camera
        path's full-depth HIGH (camera events 3..K*) - the camera-conditioned (query-weighted) projection used ONLY as a diagnostic
        upper bound (never a candidate: it depends on the camera).
"""
import math, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / 'ecro_fd114'))
import tracer114 as T114  # noqa: E402

T113, T112 = T114.T113, T114.T112
PT, S, C100 = T114.PT, T114.S, T114.C100
mi, dr, F, U = T114.mi, T114.dr, T114.F, T114.U
NB, NS = T114.NB, T114.NS
LUM = T114.LUM
SID_SPHERE = S.SID_SPHERE
HIGH_BUCKETS = ('h', 'c7_12', 'c13_24')          # camera events 3..24 (K* = 24, WL114)


def lattice_tables(G):
    """Per patch: surface id (rect index 0..17, or 18 for the sphere), patch column / row in its surface, surface columns / rows,
    surface base patch id. numpy int arrays."""
    N = G.N
    surf = np.zeros(N, np.int64); col = np.zeros(N, np.int64); row = np.zeros(N, np.int64)
    ncol = np.zeros(N, np.int64); nrow = np.zeros(N, np.int64); base = np.zeros(N, np.int64)
    for r in range(S.N_RECT):
        nu, nv = G.RG[r]
        for iv in range(nv):
            for iu in range(nu):
                p = G.RBASE[r] + iu + nu * iv
                surf[p], col[p], row[p], ncol[p], nrow[p], base[p] = r, iu, iv, nu, nv, G.RBASE[r]
    for iz in range(S.SPH_NZ):
        for ip in range(S.SPH_NPHI):
            p = G.SBASE + ip + S.SPH_NPHI * iz
            surf[p], col[p], row[p], ncol[p], nrow[p], base[p] = S.N_RECT, ip, iz, S.SPH_NPHI, S.SPH_NZ, G.SBASE
    return {'surf': surf, 'col': col, 'row': row, 'ncol': ncol, 'nrow': nrow, 'base': base}


class Ctx115(T114.Ctx114):
    def __init__(self, geom, semantics='C', h_min=T112.H_MIN):
        super().__init__(geom, semantics, h_min)
        lt = lattice_tables(self.G)
        self.LT = lt
        self.L_SPH = U(np.ascontiguousarray((lt['surf'] == S.N_RECT).astype(np.uint32)))
        self.L_COL = U(np.ascontiguousarray(lt['col'].astype(np.uint32))); self.L_ROW = U(np.ascontiguousarray(lt['row'].astype(np.uint32)))
        self.L_NCOL = U(np.ascontiguousarray(lt['ncol'].astype(np.uint32))); self.L_NROW = U(np.ascontiguousarray(lt['nrow'].astype(np.uint32)))
        self.L_BASE = U(np.ascontiguousarray(lt['base'].astype(np.uint32)))

    # ------------------------------------------------------------------ continuous within-patch coordinates and K x K address
    def subcoord(self, si, pid):
        """(a, b) of a hit inside patch pid, clamped to [0, 1) (tracer113.subcell before flooring)."""
        g = lambda k: dr.gather(F, self.PTB, pid * 21 + k)
        o = mi.Point3f(g(0), g(1), g(2)); e1 = mi.Vector3f(g(3), g(4), g(5)); e2 = mi.Vector3f(g(6), g(7), g(8))
        rel = si.p - o
        a = dr.dot(rel, e1) / dr.maximum(dr.squared_norm(e1), 1e-30); b = dr.dot(rel, e2) / dr.maximum(dr.squared_norm(e2), 1e-30)
        q = (si.p - self.cvec) * (1.0 / self.SR)
        zf = (q.z + 1.0) * (0.5 * S.SPH_NZ) - g(20); pf = (dr.atan2(q.y, q.x) + math.pi) * (S.SPH_NPHI / (2 * math.pi)) - g(19)
        sph = g(18) > 0.5
        a = dr.select(sph, pf, a); b = dr.select(sph, zf, b)
        return a, b

    @staticmethod
    def sub_k(a, b, Kk):
        """K x K address (u + K v) from continuous coordinates; Kk a python int or a UInt32 per lane."""
        Kf = F(Kk) if not isinstance(Kk, int) else float(Kk)
        su = U(dr.clamp(dr.floor(a * Kf), 0, Kf - 1)); sv = U(dr.clamp(dr.floor(b * Kf), 0, Kf - 1))
        return su + (U(Kk) if not isinstance(Kk, int) else Kk) * sv

    def subcell_k(self, si, pid, Kk):
        a, b = self.subcoord(si, pid)
        return self.sub_k(a, b, Kk)

    # ------------------------------------------------------------------ deep K x K construction (U measure), optional patch subset
    def construct_k(self, lights, n_sub, seed, kmax_probe, Kk, pids=None, b=2, verbose=False):
        """tracer114.construct_deep on a Kk x Kk grid of the patches `pids` (default all). Slot = (local patch, sub, bin).
        Returns per-cell float64 sums split by launch parity p in {0, 1}: w{p}, D6{p}{c} (f2 + fh), HF{p}{c} (f2 + fh + deep
        buckets to camera event 24), plus diagnostics."""
        pids = np.arange(self.N) if pids is None else np.asarray(pids, np.int64)
        P_ = len(pids); NSK = Kk * Kk
        ncs = P_ * NSK * NB
        PIDS = U(np.ascontiguousarray(pids.astype(np.uint32)))
        acc = {}
        diag = {'compactions': 0, 'cap_alive': 0, 'cap_throughput_lum': 0., 'proposals': ncs * n_sub, 'mismatch': 0., 'counted': 0.,
                'buried': 0., 'singular': 0., 'back': 0., 'bad_origin': 0., 'bad_floor': 0., 'launches': 0}
        deep = T114.deep_through(24)
        import time
        t0 = time.perf_counter()
        for launch, done in enumerate(range(0, n_sub, b)):
            bb = min(b, n_sub - done); Nl = ncs * bb
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            rng = mi.PCG32(size=Nl, initstate=sd, initseq=mi.UInt64(0))
            slot = dr.arange(U, Nl) % ncs
            bn = slot % NB; ps = slot // NB; sub = ps % NSK; pl = ps // NSK
            pid = dr.gather(U, PIDS, pl)
            u0, u1, u2, u3 = (rng.next_float32() for _ in range(4))
            a = (F(sub % Kk) + u0) * (1.0 / Kk); bpar = (F(sub // Kk) + u1) * (1.0 / Kk)
            sid = dr.gather(U, self.PSIDU, pid)
            t, bv, n = self.patch_frame(pid)
            local = self.dir_in_bin(bn, u2, u3)
            z = 1.25 * u2 - 1.
            ph = (F(bn % S.NPHI) + u3) * (2 * math.pi / S.NPHI) - math.pi
            rr_ = dr.sqrt(dr.maximum(1 - z * z, 0.))
            local = dr.select(dr.eq(sid, S.SID_SPHERE) & (bn < S.NPHI), mi.Vector3f(rr_ * dr.cos(ph), rr_ * dr.sin(ph), z), local)
            w = t * local.x + bv * local.y + n * local.z
            si, ssid, nn = self.first_si(pid, a, bpar, w)
            bur = self.buried(si.p, sid, nn); sing = ~bur & self.singular_point(si.p, sid, nn)
            ray = mi.Ray3f(si.p + w * 2e-3, -w)
            pre = ~bur & ~sing
            apid, abn = self.patch_bin(si, sid, w)
            asub = self.subcell_k(si, apid, Kk)
            wi0 = si.to_local(w)
            cnt = pre & si.is_valid() & (wi0.z > 1e-6)
            wt = dr.select(cnt, F(1.), F(0.))
            mism = cnt & (dr.neq(apid, pid) | dr.neq(abn, bn) | dr.neq(asub, sub))
            dr.eval(wt, mism, slot, si, ray)
            out, cnt2, d = self.trace_deep(ray, Nl, rng, sd, lights, 'probe', kmax_probe, slot, ncs, wt=wt, first=si, first_ok=pre)
            vals = {'w': wt, 'buried': dr.select(bur, F(1.), F(0.)), 'singular': dr.select(sing, F(1.), F(0.)),
                    'back': dr.select(pre & ~cnt, F(1.), F(0.)), 'mismatch': dr.select(mism, F(1.), F(0.)),
                    'cnt_disagree': dr.select(dr.neq(cnt, cnt2), F(1.), F(0.))}
            keys = sorted(vals); vv = [vals[k_] for k_ in keys]
            dr.eval(vv, slot); dr.sync_thread()
            par = launch % 2
            for k_, v in zip(keys, vv):
                if k_ == 'w':
                    bufv = dr.zeros(mi.Float64, ncs)
                    dr.scatter_reduce(dr.ReduceOp.Add, bufv, mi.Float64(v), slot)
                    acc[f'w{par}'] = acc.get(f'w{par}', 0.) + np.asarray(bufv, np.float64)
                else:
                    diag[k_] = diag.get(k_, 0.) + float(dr.sum(mi.Float64(v))[0])
            for c in range(3):
                d6 = out.get(f'f2{c}', 0.) + out.get(f'fh{c}', 0.)
                hf = d6 + sum((out[f'{nm}{c}'] for nm in deep if f'{nm}{c}' in out), 0.)
                acc[f'D6{par}{c}'] = acc.get(f'D6{par}{c}', 0.) + d6
                acc[f'HF{par}{c}'] = acc.get(f'HF{par}{c}', 0.) + hf
            for k_ in ('compactions', 'cap_alive', 'cap_throughput_lum', 'bad_origin', 'bad_floor'):
                diag[k_] += d[k_]
            diag['counted'] += float(dr.sum(mi.Float64(wt))[0]); diag['launches'] += 1
            if verbose and launch % 32 == 0:
                print('  launch', launch, round(time.perf_counter() - t0, 1), flush=True)
            del out, rng, si, ray
        diag['seconds'] = time.perf_counter() - t0
        dr.flush_malloc_cache()
        acc = {k_: np.asarray(v, np.float64).reshape(P_, NSK, NB) for k_, v in acc.items()}
        return acc, diag

    # ------------------------------------------------------------------ 2-event prefix with continuous event-2 capture
    def prefix2(self, ray, rng, lights):
        """trace112(kmax = 2, mode 'ecro') statement-identical prefix (LOW = e1 + e2), capturing at event 2: pid, bin, (a, b),
        hit point, outgoing world direction (toward the event-1 vertex), throughput and activity."""
        thr = mi.Color3f(1.)
        active = mi.Bool(True)
        out = {}; cap = {}

        def add(name, color):
            for c, ch in enumerate((color.x, color.y, color.z)):
                key = f'{name}{c}'
                out[key] = out.get(key, F(0.)) + ch
        for k in (1, 2):
            si = self.scene.ray_intersect(ray, active)
            active &= si.is_valid()
            sid = self.shape_id(si); pid, bn = self.patch_bin(si, sid, -ray.d)
            mid = dr.gather(U, self.PMATU, pid, active)
            wi = si.to_local(-ray.d)
            active &= wi.z > 1e-6; wi = dr.select(active, wi, mi.Vector3f(0, 0, 1))
            if k == 2:
                a, b = self.subcoord(si, pid)
                cap.update(pid=pid, bin=bn, a=a, b=b, p=mi.Point3f(si.p), w=mi.Vector3f(-ray.d), thr=mi.Color3f(thr), act=mi.Bool(active))
            cl = PT.CD.DrClosure83('ER', self.mats.states, mid)
            is_sss = active & dr.eq(mid, 2)
            qs, ep, ew, eb = self.sss_walk(si, ray, rng, lights, is_sss)
            d, ballistic = self.direct(si, wi, cl, lights, active, ~is_sss)
            add('e1' if k == 1 else 'e2', thr * (d + qs))
            if k == 1:
                add('e2', thr * ballistic)
            if k == 2:
                break
            wo, pdf, weight, valid = cl.sample(wi, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()), active & ~is_sss)
            newray, hs = self.spawn(si, si.to_world(wo))
            eo, es = self.exit_origin(ep, ew)
            newray.o = dr.select(is_sss, eo, newray.o)
            newray.d = dr.select(is_sss, ew, newray.d)
            weight = dr.select(is_sss, eb, weight)
            valid = dr.select(is_sss, dr.max(eb) > 0., valid)
            valid = valid & ~(hs & ~is_sss) & ~(es & is_sss)
            thr *= dr.select(valid, weight, 0.); active &= valid; ray = newray
        return out, cap

    def bin_in(self, pid, w):
        t, bv, nn = self.patch_frame(pid)
        return self.bin_of(dr.dot(w, t), dr.dot(w, bv), dr.dot(w, nn))

    def lookup(self, rep, cap):
        """Per-lane RGB value of one representation at the captured event-2 queries (before the throughput product)."""
        kind = rep['kind']; Tt = rep['T']
        pid, bn, a, b = cap['pid'], cap['bin'], cap['a'], cap['b']
        if kind == 'pc':
            Kp = dr.gather(U, rep['Kp'], pid)
            off = dr.gather(U, rep['off'], pid)
            idx = off + self.sub_k(a, b, Kp) * NB + bn
            return [dr.gather(F, Tt, idx * 3 + c) for c in range(3)]
        if kind == 'bl':
            Kk = rep['K']
            sph = dr.gather(U, self.L_SPH, pid) > 0
            col = dr.gather(U, self.L_COL, pid); row = dr.gather(U, self.L_ROW, pid)
            ncol = dr.gather(U, self.L_NCOL, pid); nrow = dr.gather(U, self.L_NROW, pid); base = dr.gather(U, self.L_BASE, pid)
            NU = F(ncol * Kk); NV = F(nrow * Kk)
            x = (F(col) + dr.clamp(a, 0., 1.)) * Kk - 0.5; y = (F(row) + dr.clamp(b, 0., 1.)) * Kk - 0.5
            x0 = dr.floor(x); y0 = dr.floor(y); fx = x - x0; fy = y - y0
            acc = [F(0.), F(0.), F(0.)]
            for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                xi = x0 + dx; yi = y0 + dy
                xi = dr.select(sph, xi - NU * dr.floor(xi / NU), dr.clamp(xi, 0., NU - 1))       # sphere: periodic in phi
                yi = dr.clamp(yi, 0., NV - 1)
                ix = U(xi); iy = U(yi)
                pn = base + ix // Kk + ncol * (iy // Kk)
                sub = (ix % Kk) + Kk * (iy % Kk)
                bnn = self.bin_in(pn, cap['w'])
                idx = (pn * (Kk * Kk) + sub) * NB + bnn
                for c in range(3):
                    acc[c] = acc[c] + wgt * dr.gather(F, Tt, idx * 3 + c)
            return acc
        raise KeyError(kind)

    def readout_multi(self, lights, cam, res, spp, seed, reps, b=64, group=8):
        """LOW (e1, e2) + event-2 readout of every representation in `reps` (dict name -> rep). Same per-launch seeding, camera
        rays and first two events as tracer112.cam_pass / tracer114.deep_pass (paired with the WL114 physical references)."""
        n = res * res
        img = {}
        for launch, done in enumerate(range(0, spp, b)):
            bb = min(b, spp - done); Nl = n * bb
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            rng = mi.PCG32(size=Nl, initstate=sd, initseq=mi.UInt64(0))
            ray, pix = PT.gen_rays(cam, res, bb, rng)
            out, cap = self.prefix2(ray, rng, lights)
            dr.eval(cap['pid'], cap['bin'], cap['a'], cap['b'], cap['p'], cap['w'], cap['thr'], cap['act'], pix, list(out.values()))
            dr.sync_thread()
            a = self.accum(out, pix, n)
            for k_, v in a.items():
                img[k_] = img.get(k_, 0.) + v
            th = cap['thr']; act = cap['act']
            names = list(reps)
            for g0 in range(0, len(names), group):            # one evaluation / scatter / host transfer per group of representations
                vals = {}
                for name in names[g0:g0 + group]:
                    val = self.lookup(reps[name], cap)
                    vals.update({f'{name}|{c}': dr.select(act, tc * vc, 0.) for c, (tc, vc) in enumerate(zip((th.x, th.y, th.z), val))})
                a2 = self.accum(vals, pix, n)
                for k_, v in a2.items():
                    img[k_] = img.get(k_, 0.) + v
        res_img = {}
        for nm in ('e1', 'e2'):
            res_img[nm] = np.stack([img[f'{nm}{c}'] for c in range(3)], -1).reshape(res, res, 3) / spp
        for name in reps:
            res_img[name] = np.stack([img[f'{name}|{c}'] for c in range(3)], -1).reshape(res, res, 3) / spp
        dr.flush_malloc_cache()
        return res_img

    # ------------------------------------------------------------------ diagnostic deep capture (camera-conditioned projection)
    def deep_capture(self, lights, cam, res, spp, seed, cells, b=48, kmax=24):
        """Physical deep camera pass to camera event kmax (tracer114.trace_deep statements, no roulette) that additionally
        accumulates, for every camera path active at event 2, into each cell system in `cells` (dict name -> (Kk, pid -> local
        index array or None for all)): n (query count), t{c} (event-2 prefix throughput) and hq{c} (the path's camera events
        3..kmax). Returns (images {bucket: [res,res,3]}, cells {name: {key: [P, Kk^2, NB]}})."""
        n = res * res
        img = {}
        cbuf = {}
        meta = {}
        for name, (Kk, loc) in cells.items():
            P_ = self.N if loc is None else int((np.asarray(loc) >= 0).sum())
            LOC = U(np.ascontiguousarray((np.arange(self.N) if loc is None else np.where(np.asarray(loc) >= 0, np.asarray(loc), 2 ** 31)).astype(np.uint32)))
            meta[name] = (Kk, P_, LOC, P_ * Kk * Kk * NB)
            cbuf[name] = {k_: dr.zeros(mi.Float64, P_ * Kk * Kk * NB) for k_ in ('n', 't0', 't1', 't2', 'h0', 'h1', 'h2')}
        for launch, done in enumerate(range(0, spp, b)):
            bb = min(b, spp - done); Nl = n * bb
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            rng = mi.PCG32(size=Nl, initstate=sd, initseq=mi.UInt64(0))
            ray, pix = PT.gen_rays(cam, res, bb, rng)
            o, cb = self._trace_capture(ray, Nl, rng, sd, lights, kmax, pix, n, meta)
            for k_, v in o.items():
                img[k_] = img.get(k_, 0.) + v
            for name in cells:
                for k_, v in cb[name].items():
                    cbuf[name][k_] += v
            dr.eval([v for d_ in cbuf.values() for v in d_.values()]); dr.sync_thread()
        names = sorted({k_[:-1] for k_ in img})
        res_img = {nm: np.stack([img.get(f'{nm}{c}', np.zeros(n)) for c in range(3)], -1).reshape(res, res, 3) / spp for nm in names}
        out_cells = {}
        for name, (Kk, P_, LOC, nc) in meta.items():
            out_cells[name] = {k_: np.asarray(v, np.float64).reshape(P_, Kk * Kk, NB) for k_, v in cbuf[name].items()}
        dr.flush_malloc_cache()
        return res_img, out_cells

    def _trace_capture(self, ray, N, rng, sd, lights, kmax, idx, nslots, meta):
        """tracer114.trace_deep (mode 'cam', deterministic, no roulette) + per-lane event-2 cell indices, carried through compaction."""
        thr = mi.Color3f(1.)
        active = mi.Bool(True)
        acc = {}; bufs = {}
        cidx = {}; cok = {}
        cb = {name: {} for name in meta}
        hsum = None

        def add(name, color):
            for c, ch in enumerate((color.x, color.y, color.z)):
                key = f'{name}{c}'
                acc[key] = acc.get(key, F(0.)) + ch

        def flush(mask):
            keys = sorted(acc)
            vals = [dr.select(mask, acc[k_], 0.) for k_ in keys]
            hq = [dr.select(mask, sum((acc[f'{b_}{c}'] for b_ in HIGH_BUCKETS if f'{b_}{c}' in acc), F(0.)), 0.) for c in range(3)]
            dr.eval(vals, hq, idx, list(cidx.values()), list(cok.values())); dr.sync_thread()
            for n_, v in zip(keys, vals):
                if n_ not in bufs:
                    bufs[n_] = dr.zeros(mi.Float64, nslots)
                dr.scatter_reduce(dr.ReduceOp.Add, bufs[n_], mi.Float64(v), idx)
            for name in meta:
                ok = cok[name] & mask
                for c in range(3):
                    cb[name].setdefault(f'h{c}', dr.zeros(mi.Float64, meta[name][3]))
                    dr.scatter_reduce(dr.ReduceOp.Add, cb[name][f'h{c}'], mi.Float64(dr.select(ok, hq[c], 0.)), cidx[name], ok)
            dr.eval(list(bufs.values()), [v for d_ in cb.values() for v in d_.values()]); dr.sync_thread()

        for k in range(1, kmax + 1):
            si = self.scene.ray_intersect(ray, active)
            active &= si.is_valid()
            sid = self.shape_id(si); pid, bn = self.patch_bin(si, sid, -ray.d)
            mid = dr.gather(U, self.PMATU, pid, active)
            wi = si.to_local(-ray.d)
            active &= wi.z > 1e-6; wi = dr.select(active, wi, mi.Vector3f(0, 0, 1))
            if k == 2:
                a, b = self.subcoord(si, pid)
                for name, (Kk, P_, LOC, nc) in meta.items():
                    loc = dr.gather(U, LOC, pid)
                    ok = active & (loc < 2 ** 31)
                    ci = dr.select(ok, (loc * (Kk * Kk) + self.sub_k(a, b, Kk)) * NB + bn, U(0))
                    cidx[name] = ci; cok[name] = ok
                    for key, v in (('n', F(1.)), ('t0', thr.x), ('t1', thr.y), ('t2', thr.z)):
                        buf = dr.zeros(mi.Float64, nc)
                        dr.scatter_reduce(dr.ReduceOp.Add, buf, mi.Float64(dr.select(ok, v, 0.)), ci, ok)
                        cb[name][key] = buf
                dr.eval(list(cidx.values()), list(cok.values()), [v for d_ in cb.values() for v in d_.values()])
            cl = PT.CD.DrClosure83('ER', self.mats.states, mid)
            is_sss = active & dr.eq(mid, 2)
            qs, ep, ew, eb = self.sss_walk(si, ray, rng, lights, is_sss)
            d, ballistic = self.direct(si, wi, cl, lights, active, ~is_sss)
            add(T114.bucket_cam(k), thr * (d + qs))
            if k < kmax:
                add(T114.bucket_cam(k + 1), thr * ballistic)
            if k == kmax:
                break
            wo, pdf, weight, valid = cl.sample(wi, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()), active & ~is_sss)
            newray, hs = self.spawn(si, si.to_world(wo))
            eo, es = self.exit_origin(ep, ew)
            newray.o = dr.select(is_sss, eo, newray.o)
            newray.d = dr.select(is_sss, ew, newray.d)
            weight = dr.select(is_sss, eb, weight)
            valid = dr.select(is_sss, dr.max(eb) > 0., valid)
            valid = valid & ~(hs & ~is_sss) & ~(es & is_sss)
            thr *= dr.select(valid, weight, 0.); active &= valid; ray = newray
            dr.eval(thr, active, ray.o, ray.d, rng.state, rng.inc, list(acc.values()))
            if k >= 6 and k % T114.COMPACT_EVERY == 0:
                alive = int(dr.count(active)); width = dr.width(active)
                if alive == 0:
                    break
                if alive < 0.5 * width:
                    flush(~active)
                    sel = U(np.flatnonzero(np.asarray(active)).astype(np.uint32))
                    g = lambda x: dr.gather(type(x), x, sel)
                    mt = g(ray.maxt); thr = g(thr); ray = mi.Ray3f(g(ray.o), g(ray.d)); ray.maxt = mt; idx = g(idx)
                    r2 = mi.PCG32(dr.width(sel)); r2.state = g(rng.state); r2.inc = g(rng.inc); rng = r2
                    acc = {k_: g(v) for k_, v in acc.items()}
                    cidx = {k_: g(v) for k_, v in cidx.items()}; cok = {k_: g(v) for k_, v in cok.items()}
                    active = dr.full(mi.Bool, True, dr.width(sel))
                    dr.eval(thr, ray.o, ray.d, idx, rng.state, rng.inc, list(acc.values()), list(cidx.values()), list(cok.values()), active)
        flush(dr.full(mi.Bool, True, dr.width(thr.x)))
        out = {k_: np.asarray(v, np.float64) for k_, v in bufs.items()}
        return out, cb


# ------------------------------------------------------------------------------------------------ tables from construction sums
def pool(x, Kfrom, Kto):
    """Sum-pool a [P, Kfrom^2, NB, ...] per-cell sum array to Kto x Kto (Kto divides Kfrom). Sub index = u + K v."""
    P_ = x.shape[0]; r = Kfrom // Kto
    t = x.reshape((P_, Kto, r, Kto, r) + x.shape[2:])                      # [P, v, rv, u, ru, ...]
    return t.sum((2, 4)).reshape((P_, Kto * Kto) + x.shape[2:])


def means(acc, which='HF', parity=None):
    """(w [P, K2, NB], L [P, K2, NB, 3]) sums of one or both parities."""
    ps = (0, 1) if parity is None else (parity,)
    w = sum(acc[f'w{p}'] for p in ps)
    L = np.stack([sum(acc[f'{which}{p}{c}'] for p in ps) for c in range(3)], -1)
    return w, L


def table_from_sums(w, L, Kk, parent=None):
    """U-mean table [P, Kk^2, NB, 3]; an empty cell takes its parent (next coarser nested level) value - recursively to coarse."""
    T = L / np.maximum(w, 1e-30)[..., None]
    empty = w <= 0
    if empty.any():
        if Kk > 1:
            par = table_from_sums(pool(w, Kk, Kk // 2), pool(L, Kk, Kk // 2), Kk // 2)
            up = np.repeat(np.repeat(par.reshape((par.shape[0], Kk // 2, Kk // 2) + par.shape[2:]), 2, 1), 2, 2).reshape(T.shape)
        else:
            up = np.zeros_like(T)
        T = np.where(empty[..., None], up, T)
    return T.astype(np.float32)


def upsample(T, Kfrom, Kto):
    """Exact broadcast of a [P, Kfrom^2, NB, 3] table to Kto x Kto."""
    P_ = T.shape[0]; r = Kto // Kfrom
    t = T.reshape((P_, Kfrom, Kfrom) + T.shape[2:])
    t = np.repeat(np.repeat(t, r, 1), r, 2)
    return t.reshape((P_, Kto * Kto) + T.shape[2:])


def pc_rep(tables_by_patch, N=1270):
    """Flat 'pc' representation from a per-patch list of (Kp, table [Kp^2, NB, 3])."""
    Kp = np.array([k for k, _ in tables_by_patch], np.uint32)
    sizes = np.array([k * k * NB for k, _ in tables_by_patch], np.int64)
    off = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.uint32)
    flat = np.concatenate([np.asarray(t, np.float32).reshape(-1) for _, t in tables_by_patch])
    return {'kind': 'pc', 'Kp': U(Kp), 'off': U(off), 'T': F(np.ascontiguousarray(flat))}


def pc_uniform(T, Kk):
    """'pc' representation of a dense [N, Kk^2, NB, 3] table."""
    N = T.shape[0]
    return {'kind': 'pc', 'Kp': U(np.full(N, Kk, np.uint32)), 'off': U((np.arange(N) * Kk * Kk * NB).astype(np.uint32)),
            'T': F(np.ascontiguousarray(np.asarray(T, np.float32).ravel()))}


def pc_hybrid(T_base, Kb, fine, Kf, pids):
    """Dense Kb x Kb everywhere, Kf x Kf on the patches `pids` (fine [len(pids), Kf^2, NB, 3])."""
    lst = [(Kb, T_base[p]) for p in range(T_base.shape[0])]
    for i, p in enumerate(pids):
        lst[p] = (Kf, fine[i])
    return pc_rep(lst, T_base.shape[0])


def bl_rep(T, Kk):
    return {'kind': 'bl', 'K': int(Kk), 'T': F(np.ascontiguousarray(np.asarray(T, np.float32).ravel()))}


def contact_patches(G, centre_xy, r_floor=0.30):
    """Diagnostic contact neighbourhood (ring-restricted control, never a candidate support): the 8 lowest-ring sphere patches and
    the floor patches whose closest point lies within r_floor (horizontal) of the support point."""
    sph = (G.PSID == S.SID_SPHERE) & (G.PSPH[:, 1] == 0)
    fl = np.zeros(G.N, bool)
    cx, cy = centre_xy
    for p in np.flatnonzero(G.PSID == 0):
        lo = G.PO[p, :2]; hi = G.PO[p, :2] + G.PE1[p, :2] + G.PE2[p, :2]
        qx = min(max(cx, min(lo[0], hi[0])), max(lo[0], hi[0])); qy = min(max(cy, min(lo[1], hi[1])), max(lo[1], hi[1]))
        fl[p] = math.hypot(qx - cx, qy - cy) < r_floor
    return np.flatnonzero(sph | fl)


def lum(x):
    return x[..., 0] * LUM[0] + x[..., 1] * LUM[1] + x[..., 2] * LUM[2]
