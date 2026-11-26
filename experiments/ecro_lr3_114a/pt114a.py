"""WL114A: the WL102 LR3 ECRO transport module (pt102) ported to Mitsuba 3.6.4 / Dr.Jit 1.0.5 (cuda_ad_rgb), plus the WL114 deep
event-bucketed suffix estimator.

Why: Mitsuba 3.5.1 / Dr.Jit 0.4.6 (the ECRO runtime) never returns from large compress / scan kernels on the local RTX 5080, which
Mesh::build_pmf needs when loading LR3 (WL114 §3). pt102 inherited pt86 (which initialises 3.5.1 through layered81); this module
removes that dependency.

Unchanged from pt102 / pt86 (copied statement by statement, only Dr.Jit 1.x spelling changed: dr.eq -> ==, dr.cuda.ad.Float64 ->
mi.Float64, dr.count -> host count):
  patch lookup by (object, face normal class, 25 cm cell) and side; mesh patch sampling; two-sided handling; scene emitters via
  scene.sample_emitter_direction or pt86 point / spot lights; NEE at every vertex, BSDF sampling, no emission at hit points, no
  roulette in the estimator; D = 6; probe / field definition (f2 = probe event 2 = camera event 3, fh = probe events 3..5);
  bins (bin_of, dir_in_bin), state rays, source encoding (point lights exactly as pt86.source; scene emitters as pt102.source);
  per-launch seeding seed * 1000003 + launch, PCG32 initseq 0; accumulation (float32 scatter per launch, float64 host sum).
New (experiment-local, WL114 semantics): trace_deep with buckets e1 | e2 | h(3-6) | c7_12 | c13_24 | c25_48 | c49_96 (camera
numbering; probe event j = camera j + 1), exact compaction with a host index list, per-event evaluation.
The ECRO model / training (model86, train102) are torch-only and unchanged.
"""
import math, os, sys, time
import numpy as np
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')
import drjit as dr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro86'))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_lr3_102'))
import scene86 as S  # noqa: E402  (numpy only)
import mesh102 as MG  # noqa: E402  (numpy only)

F, U, I32, F64 = mi.Float, mi.UInt32, mi.Int32, mi.Float64
NB, K, KAPPA, PSRC = S.NB, S.K, S.KAPPA, S.PSRC
D = 6
TWO_PI = 2 * math.pi
EDGES = (1, 2, 6, 12, 24, 48, 96)
CAM_NAMES = ('e1', 'e2', 'h', 'c7_12', 'c13_24', 'c25_48', 'c49_96')
DEEP = CAM_NAMES[3:]
COMPACT_EVERY = 4


def fl(a):
    return F(np.ascontiguousarray(np.asarray(a, np.float32).ravel()))


def bucket_cam(k):
    for e, n in zip(EDGES, CAM_NAMES):
        if k <= e:
            return n
    return None


def bucket_probe(j):
    if j == 2:
        return 'f2'
    if j <= 5:
        return 'fh'
    return bucket_cam(j + 1)


def deep_through(K_star):
    assert K_star in EDGES and K_star >= 6
    return [n for e, n in zip(EDGES, CAM_NAMES) if 6 < e <= K_star]


class Lights:
    """pt86.Lights (point / spot lights; cos_cut = -2 for point lights)."""
    def __init__(self, lights):
        pos, rgb, axis, cc = S.light_arrays(lights)
        self.n = len(lights)
        op = lambda v: dr.opaque(F, float(v))
        self.pos = [mi.Point3f(op(pos[i, 0]), op(pos[i, 1]), op(pos[i, 2])) for i in range(self.n)]
        self.rgb = [mi.Color3f(op(rgb[i, 0]), op(rgb[i, 1]), op(rgb[i, 2])) for i in range(self.n)]
        self.axis = [mi.Vector3f(op(axis[i, 0]), op(axis[i, 1]), op(axis[i, 2])) for i in range(self.n)]
        self.cc = [op(cc[i]) for i in range(self.n)]


class SceneLights:
    """The scene's own emitters, sampled by Mitsuba (area lights). m: emitter samples per source point in the encoding."""
    n = 0

    def __init__(self, m=64):
        self.m = m


def load_scene(path, res=(640, 360), **kw):
    return mi.load_file(path, resx=res[0], resy=res[1], **kw)


class Ctx:
    def __init__(self, scene, geom=None, table_seed=None):
        self.scene = scene
        self.meshes = MG.mesh_arrays(scene, mi)
        self.ptrs = [mi.ShapePtr(m[0]) for m in self.meshes]
        g = geom or MG.MeshGeom(self.meshes)
        rs = np.random.default_rng(table_seed or S.P['representation'].get('table_seed', 86001))
        self.u_state_np = rs.random((4, 1, K), np.float32)
        self.u_src_np = rs.random((2, 1, PSRC), np.float32)
        self.bctx = mi.BSDFContext()
        self.set_tables(g)

    # ---------------------------------------------------------------- tables
    def set_tables(self, g):
        self.G = g; self.N = g.N
        self.PTB = fl(np.concatenate([g.PT, g.PB, g.PN], 1))
        self.TV0, self.TE1, self.TE2 = fl(g.T_V0), fl(g.T_E1), fl(g.T_E2)
        self.TCDF = fl(g.T_CDF)
        piece_pid = np.repeat(np.arange(g.N), g.P_TCNT)
        self.TKEY = F64(np.ascontiguousarray(piece_pid + g.T_CDF, np.float64)); self.NPIECE = len(g.T_CDF)
        self.TOFF = U(g.P_TOFF.astype(np.uint32)); self.TCNT = U(g.P_TCNT.astype(np.uint32))
        foff = np.concatenate([[0], np.cumsum([len(m[3]) for m in self.meshes])[:-1]])
        self.FOFF = U(foff.astype(np.uint32)); self.FCLS = U(g.FACE_CLS.astype(np.uint32))
        self.KEYS = U(g.KEYS.astype(np.uint32)); self.NKEYS = len(g.KEYS)
        self.KFRONT = I32(g.KFRONT.astype(np.int32)); self.KBACK = I32(g.KBACK.astype(np.int32))
        self.CMIN = [int(v) for v in g.CMIN]; self.CDIM = [int(v) for v in g.CDIM]; self.CELL = float(g.cell)
        N = g.N
        self.US = [fl(np.broadcast_to(self.u_state_np[k], (N, K))) for k in range(4)]
        self.UP = [fl(np.broadcast_to(self.u_src_np[k], (N, PSRC))) for k in range(2)]
        dr.eval(self.PTB, self.TV0, self.TE1, self.TE2, self.TCDF, self.TKEY, self.TOFF, self.TCNT, self.FOFF, self.FCLS, self.KEYS,
                self.KFRONT, self.KBACK, self.US, self.UP)

    # ---------------------------------------------------------------- bins (pt86)
    @staticmethod
    def bin_of(x, y, z):
        iz = dr.clamp(dr.floor(z * S.NZ), 0, S.NZ - 1)
        ip = dr.clamp(dr.floor((dr.atan2(y, x) + math.pi) * (S.NPHI / TWO_PI)), 0, S.NPHI - 1)
        return U(iz) * S.NPHI + U(ip)

    @staticmethod
    def dir_in_bin(bn, a, b):
        z = (F(bn // S.NPHI) + a) * (1.0 / S.NZ)
        phi = (F(bn % S.NPHI) + b) * (TWO_PI / S.NPHI) - math.pi
        r = dr.sqrt(dr.maximum(1.0 - z * z, 0.0))
        return mi.Vector3f(r * dr.cos(phi), r * dr.sin(phi), z)

    # ---------------------------------------------------------------- geometry lookups (pt102)
    def shape_id(self, si):
        sid = U(0)
        for i, p in enumerate(self.ptrs):
            sid = dr.select(si.shape == p, U(i), sid)
        return sid

    def patch_frame(self, pid):
        g = lambda k: dr.gather(F, self.PTB, pid * 9 + k)
        return mi.Vector3f(g(0), g(1), g(2)), mi.Vector3f(g(3), g(4), g(5)), mi.Vector3f(g(6), g(7), g(8))

    def patch_of(self, si, sid, w):
        hv = si.is_valid()
        cls = dr.gather(U, self.FCLS, dr.gather(U, self.FOFF, sid, hv) + si.prim_index, hv)
        c0 = [I32(dr.floor(si.p[a] * (1.0 / self.CELL))) - self.CMIN[a] for a in range(3)]
        ax = cls // 2
        found = mi.Bool(False); idx = U(0)
        for dz in (0, -1, 1):
            c = [c0[a] + dr.select(ax == a, I32(dz), I32(0)) for a in range(3)]
            inside = (c[0] >= 0) & (c[1] >= 0) & (c[2] >= 0) & (c[0] < self.CDIM[0]) & (c[1] < self.CDIM[1]) & (c[2] < self.CDIM[2])
            code = ((sid * 6 + cls) * self.CDIM[0] + U(dr.maximum(c[0], 0))) * (self.CDIM[1] * self.CDIM[2]) \
                + U(dr.maximum(c[1], 0)) * self.CDIM[2] + U(dr.maximum(c[2], 0))
            i_ = dr.binary_search(0, self.NKEYS - 1, lambda i: dr.gather(U, self.KEYS, i) < code)
            hit_ = hv & inside & (dr.gather(U, self.KEYS, i_) == code) & ~found
            idx = dr.select(hit_, i_, idx); found |= hit_
        front = dr.dot(si.n, w) > 0
        pid = dr.select(front, dr.gather(I32, self.KFRONT, idx, found), dr.gather(I32, self.KBACK, idx, found))
        return dr.select(found, pid, I32(-1)), front

    def patch_bin(self, si, sid, w):
        pid, _ = self.patch_of(si, sid, w)
        p = U(dr.maximum(pid, 0))
        t, bv, nn = self.patch_frame(p)
        return pid, self.bin_of(dr.dot(w, t), dr.dot(w, bv), dr.dot(w, nn))

    def patch_point(self, pid, a, b):
        off = dr.gather(U, self.TOFF, pid)
        q = F64(pid) + F64(dr.minimum(a, 0.99999994))
        ti = dr.binary_search(0, self.NPIECE - 1, lambda i: dr.gather(F64, self.TKEY, i) < q)
        first = ti <= off
        hi = dr.gather(F, self.TCDF, ti); lo = dr.select(first, F(0.0), dr.gather(F, self.TCDF, dr.select(first, ti, ti - 1)))
        a2 = dr.clamp((a - lo) * dr.rcp(dr.maximum(hi - lo, 1e-30)), 0.0, 1.0)
        bc = mi.warp.square_to_uniform_triangle(mi.Point2f(a2, b))
        g = lambda tab, k: dr.gather(F, tab, ti * 3 + k)
        v0 = mi.Point3f(g(self.TV0, 0), g(self.TV0, 1), g(self.TV0, 2))
        e1 = mi.Vector3f(g(self.TE1, 0), g(self.TE1, 1), g(self.TE1, 2)); e2 = mi.Vector3f(g(self.TE2, 0), g(self.TE2, 1), g(self.TE2, 2))
        t, bv, n = self.patch_frame(pid)
        return v0 + e1 * bc.x + e2 * bc.y, n, (t, bv, n)

    def surface(self, ray, active):
        si = self.scene.ray_intersect(ray, active)
        sid = self.shape_id(si)
        pid, front = self.patch_of(si, sid, -ray.d)
        ok = mi.Bool(active) & si.is_valid() & (pid >= 0)
        flip = ~front
        si.n = dr.select(flip, -si.n, si.n)
        si.sh_frame = mi.Frame3f(dr.select(flip, -si.sh_frame.n, si.sh_frame.n))
        si.wi = si.to_local(-ray.d)
        return si, sid, pid, ok

    # ---------------------------------------------------------------- lights (pt102)
    def nee(self, si, wi_ok, lights, rng, active):
        c = mi.Color3f(0.0)
        bsdf = si.bsdf()
        if isinstance(lights, SceneLights):
            ds, ew = self.scene.sample_emitter_direction(si, mi.Point2f(rng.next_float32(), rng.next_float32()), True, active)
            wo = si.to_local(ds.d)
            on = active & (ds.pdf > 0) & (wo.z > 1e-6)
            f = bsdf.eval(self.bctx, si, dr.select(on, wo, mi.Vector3f(0, 0, 1)), on)
            c += dr.select(on, f * ew, 0.0)
        else:
            for l in range(lights.n):
                dl = lights.pos[l] - si.p
                d2 = dr.squared_norm(dl)
                wl = dl * dr.rsqrt(d2)
                wo = si.to_local(wl)
                on = active & (wo.z > 1e-6) & (dr.dot(-wl, lights.axis[l]) > lights.cc[l])
                vis = on & ~self.scene.ray_test(si.spawn_ray_to(lights.pos[l]), on)
                f = bsdf.eval(self.bctx, si, dr.select(vis, wo, mi.Vector3f(0, 0, 1)), vis)
                c += dr.select(vis, f * lights.rgb[l] * dr.rcp(d2), 0.0)
        return c

    # ---------------------------------------------------------------- transport (pt102, D = 6)
    def trace(self, ray, N, rng, sd, ref, kmax, lights, mode, B=None, probe_y=None, probe_ok=None, plain=False):
        assert not plain
        thr = mi.Color3f(1.0)
        active = mi.Bool(True) if probe_ok is None else mi.Bool(probe_ok)
        out = {}
        okp = None

        def add(name, col):
            for i, ch in enumerate((col.x, col.y, col.z)):
                key = f'{name}{i}'
                out[key] = out[key] + ch if key in out else ch
        for k in range(1, kmax + 1):
            si, sid, pid, ok = self.surface(ray, active)
            active &= ok
            if mode == 'probe' and k == 1:
                active &= dr.norm(si.p - probe_y) < 1e-3
                okp = mi.Bool(active)
            active &= si.wi.z > 1e-6
            if mode == 'ecro' and k == 2:
                bn = self.bin_of(*[dr.dot(-ray.d, v) for v in self.patch_frame(U(dr.maximum(pid, 0)))])
                i3 = (U(dr.maximum(pid, 0)) * NB + bn) * 3
                hl = thr * mi.Color3f(dr.gather(F, B, i3, active), dr.gather(F, B, i3 + 1, active), dr.gather(F, B, i3 + 2, active))
                add('hl', dr.select(active, hl, 0.0))
            if not (mode == 'probe' and k == 1):
                c = thr * self.nee(si, None, lights, rng, active)
                if mode == 'probe':
                    add('f2' if k == 2 else 'fh', c)
                else:
                    add('e1' if k == 1 else ('e2' if k == 2 else 'h'), c)
            if k == kmax:
                break
            bs, w = si.bsdf().sample(self.bctx, si, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()), active)
            valid = active & (bs.pdf > 0) & (dr.max(w) > 0)
            thr = thr * dr.select(valid, w, 0.0)
            active &= valid
            ray = si.spawn_ray(si.to_world(bs.wo))
        return out, okp

    # ---------------------------------------------------------------- deep suffix estimator (WL114 semantics)
    def trace_deep(self, ray, rng, lights, mode, kmax, idx, nslots, probe_y=None, probe_ok=None, compact=True):
        """Bucketed accumulation by camera event (probe: probe event j = camera j + 1; probe event 1 is the table patch, not
        counted). Events identical to trace() up to its depth. Returns float64 numpy buffers {bucket+channel: [nslots]} and
        the per-slot probe count (probe mode)."""
        thr = mi.Color3f(1.0)
        active = mi.Bool(True) if probe_ok is None else mi.Bool(probe_ok)
        acc = {}; bufs = {}
        okp = None

        def add(name, col):
            for i, ch in enumerate((col.x, col.y, col.z)):
                key = f'{name}{i}'
                acc[key] = acc[key] + ch if key in acc else ch

        def flush(mask):
            keys = sorted(acc)
            vals = [dr.select(mask, acc[k_], 0.) for k_ in keys]
            dr.eval(vals, idx); dr.sync_thread()
            for k_, v in zip(keys, vals):
                if k_ not in bufs:
                    bufs[k_] = dr.zeros(F64, nslots)
                dr.scatter_reduce(dr.ReduceOp.Add, bufs[k_], F64(v), idx)
            dr.eval(list(bufs.values())); dr.sync_thread()
        for k in range(1, kmax + 1):
            si, sid, pid, ok = self.surface(ray, active)
            active &= ok
            if mode == 'probe' and k == 1:
                active &= dr.norm(si.p - probe_y) < 1e-3
                okp = mi.Bool(active)
                dr.eval(okp)
            active &= si.wi.z > 1e-6
            if not (mode == 'probe' and k == 1):
                c = thr * self.nee(si, None, lights, rng, active)
                add(bucket_probe(k) if mode == 'probe' else bucket_cam(k), c)
            if k == kmax:
                break
            bs, w = si.bsdf().sample(self.bctx, si, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()), active)
            valid = active & (bs.pdf > 0) & (dr.max(w) > 0)
            thr = thr * dr.select(valid, w, 0.0)
            active &= valid
            ray = si.spawn_ray(si.to_world(bs.wo))
            dr.eval(thr, active, ray.o, ray.d, ray.maxt, rng.state, rng.inc, list(acc.values()))
            if compact and k >= 6 and k % COMPACT_EVERY == 0:
                am = np.asarray(active)
                alive = int(am.sum())
                if alive == 0:
                    break
                if alive < 0.5 * am.size:
                    flush(~active)
                    sel = U(np.flatnonzero(am).astype(np.uint32))
                    g = lambda x: dr.gather(type(x), x, sel)
                    mt = g(ray.maxt); thr = g(thr); ray = mi.Ray3f(g(ray.o), g(ray.d)); ray.maxt = mt; idx = g(idx)
                    r2 = mi.PCG32(dr.width(sel)); r2.state = g(rng.state); r2.inc = g(rng.inc); rng = r2
                    acc = {k_: g(v) for k_, v in acc.items()}
                    active = dr.full(mi.Bool, True, dr.width(sel))
                    dr.eval(thr, ray.o, ray.d, ray.maxt, idx, rng.state, rng.inc, list(acc.values()), active)
        flush(dr.full(mi.Bool, True, dr.width(thr.x)))
        return {k_: np.asarray(v, np.float64) for k_, v in bufs.items()}, okp

    # ---------------------------------------------------------------- camera (pt102)
    def gen_rays(self, b, rng):
        sensor = self.scene.sensors()[0]
        W, H = sensor.film().crop_size()
        n = W * H
        lane = dr.arange(U, n * b)
        pix = lane % n
        jx, jy = (rng.next_float32(), rng.next_float32()) if rng is not None else (F(0.5), F(0.5))
        pos = mi.Point2f((F(pix % W) + jx) / W, (F(pix // W) + jy) / H)
        ray, _ = sensor.sample_ray(0.0, 0.5, pos, mi.Point2f(0.5, 0.5))
        return mi.Ray3f(ray), pix, (W, H)

    @staticmethod
    def accum(out, index, n):
        keys = sorted(out)
        vals = [out[k] for k in keys]
        dr.eval(vals, index); dr.sync_thread()
        bufs = []
        for v in vals:
            buf = dr.zeros(F, n)
            dr.scatter_reduce(dr.ReduceOp.Add, buf, v, index)
            bufs.append(buf)
        dr.eval(bufs); dr.sync_thread()
        return {k: np.asarray(b, np.float64) for k, b in zip(keys, bufs)}

    def render(self, lights, spp, seed, mode='cam', B=None, b=8, kmax=None):
        kmax = kmax or (2 if mode == 'ecro' else D)
        acc = {}
        T_ = {'trace_s': 0.0, 'eval_s': 0.0, 'launches': 0}
        t_all = time.perf_counter()
        done, launch = 0, 0
        while done < spp:
            bb = min(b, spp - done)
            t0 = time.perf_counter()
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            W, H = self.scene.sensors()[0].film().crop_size()
            N = W * H * bb
            rng = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(0))
            ray, pix, _ = self.gen_rays(bb, rng)
            out, _ = self.trace(ray, N, rng, sd, 'MI', kmax, lights, mode, B=B)
            t1 = time.perf_counter()
            a = self.accum(out, pix, W * H)
            for k_, v in a.items():
                acc[k_] = acc[k_] + v if k_ in acc else v
            T_['trace_s'] += t1 - t0; T_['eval_s'] += time.perf_counter() - t1; T_['launches'] += 1
            done += bb; launch += 1
        T_['wall_s'] = time.perf_counter() - t_all; T_['spp'] = spp; T_['res'] = [W, H]
        return acc, T_

    def render_deep(self, lights, spp, seed, kmax, b=8):
        """Deep physical camera pass to camera event kmax (same seeding / camera rays as render)."""
        W, H = self.scene.sensors()[0].film().crop_size(); n = W * H
        acc = {}; t0 = time.perf_counter()
        for launch, done in enumerate(range(0, spp, b)):
            bb = min(b, spp - done)
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            rng = mi.PCG32(size=n * bb, initstate=sd, initseq=mi.UInt64(0))
            ray, pix, _ = self.gen_rays(bb, rng)
            out, _ = self.trace_deep(ray, rng, lights, 'cam', kmax, pix, n)
            for k_, v in out.items():
                acc[k_] = acc.get(k_, 0.) + v
        dr.flush_malloc_cache()
        return acc, {'wall_s': time.perf_counter() - t0, 'spp': spp, 'res': [W, H], 'kmax': kmax}

    # ---------------------------------------------------------------- field targets, state, source (pt102)
    def field_components(self, lights, spp, seed, b=16):
        ns = self.N * NB; acc = {}; launch = 0
        for done in range(0, spp, b):
            bb = min(b, spp - done); N = ns * bb
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch); launch += 1
            rng = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(0))
            slot = dr.arange(U, N) % ns; pid = slot // NB; bn = slot % NB
            p, nn, (t, bv, n) = self.patch_point(pid, rng.next_float32(), rng.next_float32())
            local = self.dir_in_bin(bn, rng.next_float32(), rng.next_float32())
            w = t * local.x + bv * local.y + n * local.z
            ok = dr.dot(w, nn) > 1e-3
            ray = mi.Ray3f(p + w * 2e-3, -w)
            a, okp = self.trace(ray, N, rng, sd, 'MI', D - 1, lights, 'probe', probe_y=p, probe_ok=ok)
            a['count'] = dr.select(okp, F(1.0), F(0.0))
            red = self.accum(a, slot, ns)
            for key, val in red.items():
                acc[key] = acc.get(key, 0.0) + val
        cnt = acc['count'].reshape(self.N, NB)

        def pack(name):
            return (np.stack([acc.get(name + str(c), np.zeros(ns)) for c in range(3)], -1).reshape(self.N, NB, 3)
                    / np.maximum(cnt, 1)[..., None]).astype(np.float32)
        return pack('f2'), pack('fh'), cnt.astype(np.float32)

    def field_deep(self, lights, spp, seed, kmax_probe, b=16):
        """field_components with the deep probe: bucket means per (patch, bin) (f2, fh, c7_12, ...) and the count."""
        ns = self.N * NB; acc = {}; cnt = np.zeros(ns)
        for launch, done in enumerate(range(0, spp, b)):
            bb = min(b, spp - done); N = ns * bb
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            rng = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(0))
            slot = dr.arange(U, N) % ns; pid = slot // NB; bn = slot % NB
            p, nn, (t, bv, n) = self.patch_point(pid, rng.next_float32(), rng.next_float32())
            local = self.dir_in_bin(bn, rng.next_float32(), rng.next_float32())
            w = t * local.x + bv * local.y + n * local.z
            ok = dr.dot(w, nn) > 1e-3
            ray = mi.Ray3f(p + w * 2e-3, -w)
            out, okp = self.trace_deep(ray, rng, lights, 'probe', kmax_probe, slot, ns, probe_y=p, probe_ok=ok)
            c = dr.select(okp, F(1.0), F(0.0)); dr.eval(c, slot); dr.sync_thread()
            cnt += np.bincount(np.asarray(slot), weights=np.asarray(c, np.float64), minlength=ns)
            for k_, v in out.items():
                acc[k_] = acc.get(k_, 0.) + v
        dr.flush_malloc_cache()
        names = sorted({k_[:-1] for k_ in acc})
        res = {nm: (np.stack([acc.get(f'{nm}{c}', np.zeros(ns)) for c in range(3)], -1).reshape(self.N, NB, 3)
                    / np.maximum(cnt.reshape(self.N, NB), 1)[..., None]).astype(np.float32) for nm in names}
        return res, cnt.reshape(self.N, NB).astype(np.float32)

    def state_rays(self, idx=None):
        ri = dr.arange(U, self.N * K) if idx is None else U(np.ascontiguousarray(idx.astype(np.uint32)))
        pid = ri // K; k = ri % K
        ug = lambda j: dr.gather(F, self.US[j], ri)
        p, nn, (t, bv, n) = self.patch_point(pid, ug(0), ug(1))
        wl = self.dir_in_bin(k // KAPPA, ug(2), ug(3))
        w = t * wl.x + bv * wl.y + n * wl.z
        ok = dr.dot(w, nn) > 1e-3
        o = p + nn * 1e-4
        si = self.scene.ray_intersect(mi.Ray3f(o, w), ok)
        sid = self.shape_id(si)
        ph, bh = self.patch_bin(si, sid, -w)
        valid = ok & si.is_valid()
        hit = dr.select(valid & (ph >= 0), ph, I32(-1))
        tt = dr.select(valid, si.t, 30.0)
        e = o + w * tt
        vals = [hit, bh, o.x, o.y, o.z, e.x, e.y, e.z]
        dr.eval(vals); dr.sync_thread()
        hit = np.asarray(hit).astype(np.int16); bh = np.asarray(bh).astype(np.uint8)
        seg = np.stack([np.asarray(v) for v in vals[2:]], 1).astype(np.float32)
        return hit, bh, seg

    def source(self, lights, seed=7):
        if not isinstance(lights, SceneLights):
            return self._source_points(lights)
        m = lights.m
        Np = self.N * PSRC * m
        lane = dr.arange(U, Np)
        pt_ = lane // m; pid = pt_ // PSRC
        p, nn, (t, bv, n) = self.patch_point(pid, dr.gather(F, self.UP[0], pt_), dr.gather(F, self.UP[1], pt_))
        rng = mi.PCG32(size=Np, initstate=mi.UInt64(seed), initseq=mi.UInt64(0))
        it = dr.zeros(mi.Interaction3f, Np)
        it.p = p + nn * 1e-4; it.n = nn
        ds, ew = self.scene.sample_emitter_direction(it, mi.Point2f(rng.next_float32(), rng.next_float32()), True)
        cs = dr.dot(ds.d, nn)
        on = (ds.pdf > 0) & (cs > 1e-6)
        e = [dr.select(on, ch * cs, 0.0) for ch in (ew.x, ew.y, ew.z)]
        bn = self.bin_of(dr.dot(ds.d, t), dr.dot(ds.d, bv), dr.dot(ds.d, n))
        slot = pid * NB + bn
        dr.eval(e, slot); dr.sync_thread()
        sl = np.asarray(slot)
        s = np.stack([np.bincount(sl, weights=np.asarray(e[c], np.float64), minlength=self.N * NB) for c in range(3)], -1)
        return (s / (PSRC * m)).reshape(self.N, NB, 3).astype(np.float32)

    def _source_points(self, lights):
        """pt86.Ctx.source (point / spot lights)."""
        Np = self.N * PSRC
        lane = dr.arange(U, Np)
        pid = lane // PSRC
        p, nn, (t, bv, n) = self.patch_point(pid, dr.gather(F, self.UP[0], lane), dr.gather(F, self.UP[1], lane))
        s = np.zeros((self.N * NB, 3))
        for l in range(lights.n):
            dl = lights.pos[l] - p
            d2 = dr.squared_norm(dl); dist = dr.sqrt(d2)
            wl = dl * dr.rcp(dist)
            cs = dr.dot(wl, nn)
            on = (cs > 1e-6) & (dr.dot(-wl, lights.axis[l]) > lights.cc[l])
            r = mi.Ray3f(p + nn * 1e-4, wl)
            r.maxt = dist * (1.0 - 1e-4)
            occ = self.scene.ray_test(r, on)
            e = dr.select(on & ~occ, cs * dr.rcp(d2), 0.0)
            bn = self.bin_of(dr.dot(wl, t), dr.dot(wl, bv), dr.dot(wl, n))
            ch = [e * lights.rgb[l].x, e * lights.rgb[l].y, e * lights.rgb[l].z]
            slot = pid * NB + bn
            dr.eval(ch, slot); dr.sync_thread()
            sl = np.asarray(slot)
            for c in range(3):
                s[:, c] += np.bincount(sl, weights=np.asarray(ch[c], np.float64), minlength=self.N * NB)
        return (s / PSRC).reshape(self.N, NB, 3).astype(np.float32)


def pack(acc, W, H, spp):
    g = lambda k: (acc[k] / spp).reshape(H, W).astype(np.float32)
    o = {}
    names = sorted({k[:-1] for k in acc})
    for name in names:
        if f'{name}0' in acc:
            o[name] = np.stack([g(f'{name}{c}') for c in range(3)], -1)
    return o
