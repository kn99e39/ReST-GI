"""WL114 deep physical tracer for the SSS family: WL113 tracer113 (contract v3, C semantics, 4 x 4 addresses) with one
experiment-local addition, a long-path suffix estimator:

  * event-bucketed accumulation (camera event convention; probe event j = camera event j + 1):
        camera:   e1 | e2 | h = 3..6 | c7_12 | c13_24 | c25_48 | c49_96 | c97_192 | rr (> rr_from, Russian roulette)
        probe:    f2 = camera 3 | fh = camera 4..6 | c7_12 | ... (same buckets, camera numbering)
    Note: in the WL86-WL113 field convention the probe's own first vertex is the table patch (camera event 2, LOW), so
    'f2' is camera event 3 and 'fh' camera events 4..6; the D6 HIGH target is f2 + fh and a depth-K HIGH target is
    f2 + fh + all buckets up to camera event K.
  * events up to the first bucket boundary beyond 6 are statement- and RNG-identical to tracer113.trace_full / tracer112
    (focused tests: test114.py),
  * stream compaction of the surviving paths (host-side index list; Dr.Jit 0.4.6's dr.compress hangs on the RTX 5080 for
    arrays larger than one block) (exact: per-lane state incl. RNG and accumulators is gathered; dead lanes are
    flushed first) every COMPACT_EVERY events once fewer than half of the lanes are alive,
  * optional unbiased Russian roulette after camera event rr_from: survival q = min(max_c thr_c / max_c thr_c(start), 0.95)
    (relative to the lane's throughput when roulette starts; lower variance than absolute throughput), thr /= q, one extra
    RNG draw per roulette event; a hard cap (rr_cap camera events) whose residual throughput is reported, never hidden.
Nothing in a shared production file or in tracer112 / tracer113 is edited.
"""
import math, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / 'ecro_sss113'))
import tracer113 as T113  # noqa: E402

T112 = T113.T112
PT, S, C100 = T113.PT, T113.S, T113.C100
mi, dr, F, U = T113.mi, T113.dr, T113.F, T113.U
NB, K, NS = T113.NB, T113.K, T113.NS
LUM = T113.LUM
EDGES = (1, 2, 6, 12, 24, 48, 96, 192)            # bucket upper camera events: e1, e2, h, c7_12, c13_24, c25_48, c49_96, c97_192
CAM_NAMES = ('e1', 'e2', 'h', 'c7_12', 'c13_24', 'c25_48', 'c49_96', 'c97_192')
PROBE_NAMES = ('-', 'lowe2', 'f2', 'fh', 'c7_12', 'c13_24', 'c25_48', 'c49_96', 'c97_192')   # by probe event (see bucket_probe)
DEEP = ('c7_12', 'c13_24', 'c25_48', 'c49_96', 'c97_192')
COMPACT_EVERY = 4
Q_MAX = 0.95


def bucket_cam(k):
    """camera event k -> bucket name (or 'rr' handled by the caller)."""
    for e, n in zip(EDGES, CAM_NAMES):
        if k <= e:
            return n
    return None


def bucket_probe(j):
    """probe event j (j = 1 is the table patch itself, not counted) -> bucket name in camera numbering."""
    if j == 2:
        return 'f2'
    if j <= 5:
        return 'fh'
    return bucket_cam(j + 1)


def deep_through(K_star):
    """Buckets (beyond D6) that make up camera events 7..K_star; K_star must be a bucket edge."""
    assert K_star in EDGES and K_star >= 6
    return [n for e, n in zip(EDGES, CAM_NAMES) if 6 < e <= K_star]


class Ctx114(T113.Ctx113):
    # ------------------------------------------------------------------ the deep suffix estimator
    def trace_deep(self, ray, N, rng, sd, lights, mode, kmax, idx, nslots, wt=None, first=None, first_ok=None,
                   rr_from=None, compact=True, sq=False):
        """mode 'cam' (camera rays) or 'probe' (pinned first vertex = table patch). Returns float64 numpy buffers
        {bucket+channel: [nslots]} accumulated by slot `idx` (pixel / construction slot) with lane weight `wt`, plus
        diagnostics. kmax: last camera event (cam) / last probe event (probe) of the deterministic or roulette loop."""
        thr = mi.Color3f(1.)
        active = mi.Bool(True) if first_ok is None else mi.Bool(first_ok)
        wt = F(1.) if wt is None else wt
        acc = {}; bufs = {}
        diag = {'bad_origin': 0., 'bad_floor': 0., 'compactions': 0, 'lanes_at_end': 0, 'cap_throughput_lum': 0.,
                'cap_alive': 0, 'rr_events': 0}
        bad = mi.Float(0.); badf = mi.Float(0.)
        cnt = None
        thr_ref = None
        names_all = set()

        def add(name, color):
            names_all.add(name)
            for c, ch in enumerate((color.x, color.y, color.z)):
                key = f'{name}{c}'
                acc[key] = acc.get(key, F(0.)) + ch

        def contribution(k, color):
            if mode == 'probe':
                if k >= 2:
                    add(bucket_probe(k) if not (rr_from is not None and k + 1 > rr_from) else 'rr', color)
            else:
                add(bucket_cam(k) if not (rr_from is not None and k > rr_from) else 'rr', color)

        def flush(mask):
            """scatter the per-lane accumulators of the lanes in mask into the float64 slot buffers."""
            keys = sorted(acc)
            extra = {}
            if sq:
                for c in range(3):
                    d6 = sum((acc[f'{n}{c}'] for n in ('f2', 'fh') if f'{n}{c}' in acc), F(0.))
                    allv = sum((acc[f'{n}{c}'] for n in ('f2', 'fh') + DEEP if f'{n}{c}' in acc), F(0.))
                    extra[f'sq6{c}'] = d6 * d6; extra[f'sqA{c}'] = allv * allv
            vals = [dr.select(mask, wt * acc[k_], 0.) for k_ in keys] + [dr.select(mask, wt * v, 0.) for v in extra.values()]
            names = keys + list(extra)
            dr.eval(vals, idx); dr.sync_thread()
            for n_, v in zip(names, vals):
                if n_ not in bufs:
                    bufs[n_] = dr.zeros(mi.Float64, nslots)
                dr.scatter_reduce(dr.ReduceOp.Add, bufs[n_], mi.Float64(v), idx)
            dr.eval(list(bufs.values())); dr.sync_thread()

        k_last = kmax
        for k in range(1, kmax + 1):
            pin = mode == 'probe' and k == 1 and first is not None
            si = first if pin else self.scene.ray_intersect(ray, active)
            active &= si.is_valid()
            sid = self.shape_id(si); pid, bn = self.patch_bin(si, sid, -ray.d)
            mid = dr.gather(U, self.PMATU, pid, active)
            wi = si.to_local(-ray.d)
            active &= wi.z > 1e-6; wi = dr.select(active, wi, mi.Vector3f(0, 0, 1))
            if mode == 'probe' and k == 1:
                cnt = mi.Bool(active)
            cl = PT.CD.DrClosure83('ER', self.mats.states, mid)
            is_sss = active & dr.eq(mid, 2)
            qs, ep, ew, eb = self.sss_walk(si, ray, rng, lights, is_sss)
            d, ballistic = self.direct(si, wi, cl, lights, active, ~is_sss)
            bad += dr.select(self._bad_direct[0], 1., 0.); badf += dr.select(self._bad_direct[1], 1., 0.)
            if not (mode == 'probe' and k == 1):
                contribution(k, thr * (d + qs))
            if k < kmax:
                contribution(k + 1, thr * ballistic)
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
            ins, blw = self.origin_inside(newray.o)
            bad += dr.select(valid & ins, 1., 0.); badf += dr.select(valid & blw, 1., 0.)
            thr *= dr.select(valid, weight, 0.); active &= valid; ray = newray
            cam_next = k + 1 if mode == 'cam' else k + 2          # camera index of the next event
            if rr_from is not None and cam_next > rr_from:
                if thr_ref is None:                       # throughput at roulette start (per lane): survival relative to it
                    thr_ref = dr.maximum(dr.max(thr), 1e-30)
                q = dr.minimum(dr.max(thr) / thr_ref, Q_MAX)
                live = active & (rng.next_float32() < q)
                thr = dr.select(live, thr / dr.maximum(q, 1e-30), 0.); active = live
                diag['rr_events'] += 1
            dr.eval(thr, active, ray.o, ray.d, rng.state, rng.inc, list(acc.values()), bad, badf, *([thr_ref] if thr_ref is not None else []))
            if mode == 'probe' and k == 1:
                dr.eval(cnt)
            if compact and k >= 6 and k % COMPACT_EVERY == 0:
                alive = int(dr.count(active))
                width = dr.width(active)
                if alive == 0:
                    k_last = k
                    break
                if alive < 0.5 * width:
                    flush(~active)
                    sel = U(np.flatnonzero(np.asarray(active)).astype(np.uint32))   # dr.compress hangs on this GPU for n > 4096 (Dr.Jit 0.4.6, sm_120)
                    g = lambda x: dr.gather(type(x), x, sel)
                    mt = g(ray.maxt); thr_ref = g(thr_ref) if thr_ref is not None else None; thr = g(thr); ray = mi.Ray3f(g(ray.o), g(ray.d)); ray.maxt = mt; idx = g(idx); wt = g(wt)
                    r2 = mi.PCG32(dr.width(sel)); r2.state = g(rng.state); r2.inc = g(rng.inc); rng = r2
                    acc = {k_: g(v) for k_, v in acc.items()}
                    diag['bad_origin'] += float(dr.sum(bad)[0]); diag['bad_floor'] += float(dr.sum(badf)[0])
                    bad = mi.Float(0.); badf = mi.Float(0.)
                    active = dr.full(mi.Bool, True, dr.width(sel))
                    dr.eval(thr, ray.o, ray.d, idx, wt, rng.state, rng.inc, list(acc.values()), active)
                    diag['compactions'] += 1
        # residual mass still alive at the hard cap (deterministic truncation or roulette cap): reported, never added
        alive_m = active
        diag['cap_alive'] = int(dr.count(alive_m))
        diag['cap_throughput_lum'] = float(dr.sum(dr.select(alive_m, wt * (LUM[0] * thr.x + LUM[1] * thr.y + LUM[2] * thr.z), 0.))[0])
        diag['lanes_at_end'] = int(dr.width(thr.x))
        diag['k_last'] = k_last
        flush(dr.full(mi.Bool, True, dr.width(thr.x)))
        diag['bad_origin'] += float(dr.sum(bad)[0]); diag['bad_floor'] += float(dr.sum(badf)[0])
        out = {k_: np.asarray(v, np.float64) for k_, v in bufs.items()}
        return out, cnt, diag

    # ------------------------------------------------------------------ deep camera pass
    def deep_pass(self, lights, cam, res, spp, seed, kmax, b=32, rr_from=None, compact=True):
        """Physical camera pass to camera event kmax; per-bucket images [res,res,3] / spp. Same per-launch seeding and camera
        rays as tracer113.full_pass and tracer112.cam_pass (events 1..6 paired with the readout)."""
        n = res * res
        img = {}; dg = {'bad_origin': 0., 'bad_floor': 0., 'compactions': 0, 'cap_alive': 0, 'cap_throughput_lum': 0., 'paths': n * spp}
        for launch, done in enumerate(range(0, spp, b)):
            bb = min(b, spp - done); Nl = n * bb
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            rng = mi.PCG32(size=Nl, initstate=sd, initseq=mi.UInt64(0))
            ray, pix = PT.gen_rays(cam, res, bb, rng)
            out, _, d = self.trace_deep(ray, Nl, rng, sd, lights, 'cam', kmax, pix, n, rr_from=rr_from, compact=compact)
            for k_, v in out.items():
                img[k_] = img.get(k_, 0.) + v
            for k_ in ('bad_origin', 'bad_floor', 'compactions', 'cap_alive', 'cap_throughput_lum'):
                dg[k_] += d[k_]
        names = sorted({k_[:-1] for k_ in img})
        res_img = {nm: np.stack([img.get(f'{nm}{c}', np.zeros(n)) for c in range(3)], -1).reshape(res, res, 3) / spp for nm in names}
        dr.flush_malloc_cache()
        return res_img, dg

    # ------------------------------------------------------------------ deep 4 x 4 construction (U measure)
    def construct_deep(self, lights, n_sub, seed, kmax_probe, b=8, rr_from=None):
        """construct4 of tracer113 with the deep probe: identical proposal sampling, pinned first vertex, address and
        exclusion bookkeeping; per-bucket U sums f2 / fh / c7_12 ... (+ rr), second moments sq6 (D6) and sqA (all
        deterministic buckets)."""
        ncs = self.N * NS * NB
        acc = {}
        diag = {'compactions': 0, 'cap_alive': 0, 'cap_throughput_lum': 0., 'proposals': ncs * n_sub}
        for launch, done in enumerate(range(0, n_sub, b)):
            bb = min(b, n_sub - done); Nl = ncs * bb
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            rng = mi.PCG32(size=Nl, initstate=sd, initseq=mi.UInt64(0))
            slot = dr.arange(U, Nl) % ncs
            bn = slot % NB; ps = slot // NB; sub = ps % NS; pid = ps // NS
            u0, u1, u2, u3 = (rng.next_float32() for _ in range(4))
            a = (F(sub % K) + u0) * (1.0 / K); bpar = (F(sub // K) + u1) * (1.0 / K)
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
            asub = self.subcell(si, apid)
            # the first-vertex activity (count) is decided before any transport (k = 1), identical to trace112's [CNT]
            wi0 = si.to_local(w)
            cnt = pre & si.is_valid() & (wi0.z > 1e-6)
            wt = dr.select(cnt, F(1.), F(0.))
            mism = cnt & (dr.neq(apid, pid) | dr.neq(abn, bn) | dr.neq(asub, sub))
            dr.eval(wt, mism, slot, si, ray)
            out, cnt2, d = self.trace_deep(ray, Nl, rng, sd, lights, 'probe', kmax_probe, slot, ncs, wt=wt, first=si, first_ok=pre,
                                           rr_from=rr_from, sq=True)
            vals = {'U_w': wt, 'buried': dr.select(bur, F(1.), F(0.)), 'singular': dr.select(sing, F(1.), F(0.)),
                    'back': dr.select(pre & ~cnt, F(1.), F(0.)), 'mismatch': dr.select(mism, F(1.), F(0.)),
                    'cnt_disagree': dr.select(dr.neq(cnt, cnt2), F(1.), F(0.))}
            keys = sorted(vals); vv = [vals[k_] for k_ in keys]
            dr.eval(vv, slot); dr.sync_thread()
            for k_, v in zip(keys, vv):
                bufv = dr.zeros(mi.Float64, ncs)
                dr.scatter_reduce(dr.ReduceOp.Add, bufv, mi.Float64(v), slot)
                acc[k_] = acc.get(k_, 0.) + np.asarray(bufv, np.float64)
            for k_, v in out.items():
                acc['U_' + k_] = acc.get('U_' + k_, 0.) + v
            for k_ in ('compactions', 'cap_alive', 'cap_throughput_lum'):
                diag[k_] += d[k_]
            diag['bad_origin'] = diag.get('bad_origin', 0.) + d['bad_origin']; diag['bad_floor'] = diag.get('bad_floor', 0.) + d['bad_floor']
        acc['proposals'] = np.full(ncs, float(n_sub))
        dr.flush_malloc_cache()
        return acc, diag


def high_tables(acc, K_star, N=1270):
    """D6 (f2 + fh) and depth-K* (f2 + fh + deep buckets <= K*) HIGH tables from the SAME samples, through the unchanged WL113
    table contract (tracer113.tables4: nested 4x4 / 2x2 / coarse U means, empty sub-cell -> parent). Returns
    {'U_D6', 'U_F' (coarse [N,64,3]), 'S4_D6', 'S4_F' (dense [N,16,64,3]), 'native_c', 'native_4'}."""
    out = {}
    deep = deep_through(K_star)
    for tag, extra in (('D6', ()), ('F', deep)):
        a = {'U_w': acc['U_w']}
        for c in range(3):
            v = acc[f'U_f2{c}'] + acc[f'U_fh{c}']
            for nm in extra:
                if f'U_{nm}{c}' in acc:
                    v = v + acc[f'U_{nm}{c}']
            a[f'U_f2{c}'] = v; a[f'U_fh{c}'] = np.zeros_like(v)
        Tc, T2, T4, nat = T113.tables4(a, N)
        out[f'U_{tag}'] = Tc; out[f'S4_{tag}'] = T4
    out['native_c'] = nat['c']; out['native_4'] = nat['4']
    return out


def ring0(G, centre_xy, r=0.10):
    """Construction-level contact set: the 8 lowest-ring sphere patches and the floor patches whose centre lies within r of the
    sphere's support point."""
    fl = (G.PSID == 0) & (np.linalg.norm(G.PC[:, :2] - np.asarray(centre_xy)[None], axis=1) < r)
    return T113.support_ring0(G) | fl


def lum(x):
    return T113.lum(x)
