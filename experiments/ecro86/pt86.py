"""WL86 transport module (Dr.Jit on Mitsuba 3.5.1 ray intersection): reference path tracer, LOW / ECRO renderer, field probes,
geometry state rays, source encoding, geometric masks.

Estimator (as WL85): at each surface vertex next-event estimation to every light, then BSDF sampling; no Russian roulette.
Event convention: a contribution made by next-event estimation at the k-th vertex of a camera path has k scattering events.
  camera modes  : e1 (k = 1) = q read by the camera, e2 (k = 2) = T q, h (3 <= k <= D) = H[q] = sum_{j=2}^{D-1} T^j q, D = 6
  ecro mode     : e1, e2 as above (identical estimator) + hl = throughput(x1) * B[patch(x2), bin(x2 -> x1)], where the table B holds
                  the outgoing radiance of x2 made of 2..D-1 scattering events (so that hl estimates h; no extra ray)
  probe mode    : radiance leaving a surface point y in direction w made of 2..kmax scattering events (kmax = D - 1 = 5)
ER: frozen DrClosure83 (per-lane material table). P: layered81 random walk / exact Lambert.
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ['fnto_compiled83', 'fnto_compiled82', 'fnto_compiled81']:
    sys.path.insert(0, os.path.join(HERE, '..', d))
sys.path.insert(0, HERE)
import layered81 as L  # noqa: E402
mi = L.init('cuda_ad_rgb')
import drjit as dr  # noqa: E402
import closure83_dr as CD  # noqa: E402
import scene86 as S  # noqa: E402

P = S.P; FX = S.FX
D = 6
T = mi.ScalarTransform4f
F = mi.Float; U = mi.UInt32
NB, K, KAPPA, PSRC = S.NB, S.K, S.KAPPA, S.PSRC
SHAPE_NAMES = ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'sph', 'boxA', 'boxB']
SHAPE_MAT = [0, 3, 3, 3, 3, 1, 2, 4, 5]
TWO_PI = 2 * math.pi


def fl(a):
    return F(np.ascontiguousarray(np.asarray(a, np.float32).ravel()))


def mean3(c):
    return (c.x + c.y + c.z) * (1.0 / 3.0)


class Mats:
    def __init__(self):
        z = np.load(os.path.join(HERE, 'data', 'mats86.npz'))
        self.states = z['states']; self.desc = z['desc']; self.lambert = z['lambert']; self.albedo = z['albedo']
        self.desc_cols = [fl(self.desc[:, j]) for j in range(10)]
        self.lam_f = fl(self.lambert); self.alb_f = fl(self.albedo)
        self.cls = U(np.array(S.MAT_CLASS, np.uint32))


class Lights:
    def __init__(self, lights):
        pos, rgb, axis, cc = S.light_arrays(lights)
        self.n = len(lights)
        op = lambda v: dr.opaque(F, float(v))
        self.pos = [mi.Point3f(op(pos[i, 0]), op(pos[i, 1]), op(pos[i, 2])) for i in range(self.n)]
        self.rgb = [mi.Color3f(op(rgb[i, 0]), op(rgb[i, 1]), op(rgb[i, 2])) for i in range(self.n)]
        self.axis = [mi.Vector3f(op(axis[i, 0]), op(axis[i, 1]), op(axis[i, 2])) for i in range(self.n)]
        self.cc = [op(cc[i]) for i in range(self.n)]


def cam_basis(cam):
    o = np.array(cam['origin'], np.float64); t = np.array(cam['target'], np.float64)
    fwd = t - o; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return o, fwd, right, up, math.tan(math.radians(cam['fov_deg']) / 2)


def gen_rays(cam, res, b, rng):
    o, fwd, right, up, s = cam_basis(cam)
    n = res * res
    lane = dr.arange(U, n * b)
    pix = lane % n
    ii, jj = pix // res, pix % res
    jx, jy = (rng.next_float32(), rng.next_float32()) if rng is not None else (F(0.5), F(0.5))
    x = (F(jj) + jx) * (2.0 / res) - 1.0
    y = 1.0 - (F(ii) + jy) * (2.0 / res)
    d = dr.normalize(mi.Vector3f(*[float(fwd[k]) + s * x * float(right[k]) + s * y * float(up[k]) for k in range(3)]))
    return mi.Ray3f(mi.Point3f(*[F(float(o[k])) for k in range(3)]), d), pix


class Ctx:
    """One Mitsuba scene of the family + Dr.Jit tables of the current geometry; set_geom() edits the boxes in place."""

    def __init__(self, geom, bsdfs=None, emitters=None):
        self.mats = Mats()
        g = S.Geom(geom)
        df = {'type': 'diffuse'}
        bs = bsdfs or [df] * 9
        d = {'type': 'scene'}
        for i, r in enumerate(S.stationary_rects()):
            d[f'r{i}'] = {'type': 'rectangle', 'to_world': T(S.rect_matrix(*r[:5]).tolist()), 'bsdf': bs[i]}
        d['sph'] = {'type': 'sphere', 'center': FX['sphere']['center'], 'radius': FX['sphere']['radius'], 'bsdf': bs[6]}
        d['boxA'] = {'type': 'cube', 'to_world': T(g.box_matrix('A').tolist()), 'bsdf': bs[7]}
        d['boxB'] = {'type': 'cube', 'to_world': T(g.box_matrix('B').tolist()), 'bsdf': bs[8]}
        for i, e in enumerate(emitters or []):
            d[f'em{i}'] = e
        self.scene = mi.load_dict(d)
        by = {s.id(): mi.ShapePtr(s) for s in self.scene.shapes()}
        self.ptrs = [by[n] for n in SHAPE_NAMES]
        self.params = mi.traverse(self.scene)
        self.V0 = {}; self.N0 = {}
        for b, name in (('A', 'boxA'), ('B', 'boxB')):
            M = g.box_matrix(b)
            V = np.asarray(self.params[f'{name}.vertex_positions'], np.float64).reshape(-1, 3)
            self.V0[name] = (V - M[:3, 3]) @ np.linalg.inv(M[:3, :3]).T
            Nn = np.asarray(self.params[f'{name}.vertex_normals'], np.float64).reshape(-1, 3)
            if Nn.size:
                nl = Nn @ M[:3, :3]
                self.N0[name] = nl / np.maximum(np.linalg.norm(nl, axis=1, keepdims=True), 1e-30)
        self.SMAT = U(np.array(SHAPE_MAT, np.uint32))
        rs = np.random.default_rng(P['representation'].get('table_seed', 86001))
        us = rs.random((4, 1, K), np.float32)                   # the same K sample offsets for every patch
        self.u_state_np = us
        self.u_src_np = rs.random((2, 1, PSRC), np.float32)
        self.set_tables(g)

    def set_tables(self, g):
        self.G = g
        self.N = g.N
        self.RT = fl(g.rect_table()); self.PTB = fl(g.patch_table()); self.BX = fl(g.BX)
        self.SC = [float(v) for v in g.SC]; self.SR = g.SR; self.SBASE = int(g.SBASE)
        N = g.N
        self.US = [fl(np.broadcast_to(self.u_state_np[k], (N, K))) for k in range(4)]
        self.UP = [fl(np.broadcast_to(self.u_src_np[k], (N, PSRC))) for k in range(2)]
        dr.eval(self.RT, self.PTB, self.BX, self.US, self.UP)

    def set_geom(self, geom):
        g = S.Geom(geom)
        for b, name in (('A', 'boxA'), ('B', 'boxB')):
            M = g.box_matrix(b)
            self.params[f'{name}.vertex_positions'] = fl(self.V0[name] @ M[:3, :3].T + M[:3, 3])
            if name in self.N0:
                R = np.stack(S.box_axes(geom[b]['yaw']), 1)
                self.params[f'{name}.vertex_normals'] = fl(self.N0[name] @ R.T)
        self.params.update()
        self.set_tables(g)

    # ---------------------------------------------------------------- geometry lookups
    def shape_id(self, si):
        sid = U(0)
        for i, p in enumerate(self.ptrs):
            sid = dr.select(dr.eq(si.shape, p), U(i), sid)
        return sid

    def patch_frame(self, pid):
        g = lambda k: dr.gather(F, self.PTB, pid * 21 + k)
        return mi.Vector3f(g(9), g(10), g(11)), mi.Vector3f(g(12), g(13), g(14)), mi.Vector3f(g(15), g(16), g(17))

    @staticmethod
    def bin_of(x, y, z):
        iz = dr.clamp(dr.floor(z * S.NZ), 0, S.NZ - 1)
        ip = dr.clamp(dr.floor((dr.atan2(y, x) + math.pi) * (S.NPHI / TWO_PI)), 0, S.NPHI - 1)
        return U(iz) * S.NPHI + U(ip)

    def patch_bin(self, si, sid, w):
        """Patch id of the hit and direction bin of the world direction w in that patch's frame."""
        isb = sid >= S.SID_BOXA
        b = dr.select(isb, sid - S.SID_BOXA, U(0))
        gb = lambda k: dr.gather(F, self.BX, b * 9 + k)
        n = si.n
        dd = [n.x * gb(3 * a) + n.y * gb(3 * a + 1) + n.z * gb(3 * a + 2) for a in range(3)]
        aa = [dr.abs(v) for v in dd]
        ax = dr.select((aa[0] >= aa[1]) & (aa[0] >= aa[2]), U(0), dr.select(aa[1] >= aa[2], U(1), U(2)))
        val = dr.select(dr.eq(ax, 0), dd[0], dr.select(dr.eq(ax, 1), dd[1], dd[2]))
        rid = dr.select(isb, 6 + 6 * b + 2 * ax + dr.select(val > 0, U(1), U(0)), dr.minimum(sid, U(5)))
        gr = lambda k: dr.gather(F, self.RT, rid * 14 + k)
        rel = si.p - mi.Point3f(gr(0), gr(1), gr(2))
        u = (dr.dot(rel, mi.Vector3f(gr(3), gr(4), gr(5))) * dr.rcp(gr(9)) + 1.0) * 0.5
        v = (dr.dot(rel, mi.Vector3f(gr(6), gr(7), gr(8))) * dr.rcp(gr(10)) + 1.0) * 0.5
        nu, nv = gr(11), gr(12)
        iu = dr.clamp(dr.floor(u * nu), 0, nu - 1); iv = dr.clamp(dr.floor(v * nv), 0, nv - 1)
        pid_r = U(gr(13) + iu + nu * iv)
        q = (si.p - mi.Point3f(*self.SC)) * (1.0 / self.SR)
        izs = dr.clamp(dr.floor((q.z + 1.0) * (0.5 * S.SPH_NZ)), 0, S.SPH_NZ - 1)
        ips = dr.clamp(dr.floor((dr.atan2(q.y, q.x) + math.pi) * (S.SPH_NPHI / TWO_PI)), 0, S.SPH_NPHI - 1)
        pid = dr.select(dr.eq(sid, S.SID_SPHERE), U(self.SBASE + ips + S.SPH_NPHI * izs), pid_r)
        t, bv, nn = self.patch_frame(pid)
        return pid, self.bin_of(dr.dot(w, t), dr.dot(w, bv), dr.dot(w, nn))

    def patch_point(self, pid, a, b):
        g = lambda k: dr.gather(F, self.PTB, pid * 21 + k)
        pr = mi.Point3f(g(0), g(1), g(2)) + a * mi.Vector3f(g(3), g(4), g(5)) + b * mi.Vector3f(g(6), g(7), g(8))
        z = (g(20) + b) * (2.0 / S.SPH_NZ) - 1.0
        phi = (g(19) + a) * (TWO_PI / S.SPH_NPHI) - math.pi
        r = dr.sqrt(dr.maximum(1.0 - z * z, 0.0))
        ns = mi.Vector3f(r * dr.cos(phi), r * dr.sin(phi), z)
        sph = g(18) > 0.5
        t, bv, n = self.patch_frame(pid)
        return dr.select(sph, mi.Point3f(*self.SC) + self.SR * ns, pr), dr.select(sph, ns, n), (t, bv, n)

    @staticmethod
    def dir_in_bin(bn, a, b):
        z = (F(bn // S.NPHI) + a) * (1.0 / S.NZ)
        phi = (F(bn % S.NPHI) + b) * (TWO_PI / S.NPHI) - math.pi
        r = dr.sqrt(dr.maximum(1.0 - z * z, 0.0))
        return mi.Vector3f(r * dr.cos(phi), r * dr.sin(phi), z)

    # ---------------------------------------------------------------- transport
    def trace(self, ray, N, rng, sd, ref, kmax, lights, mode, B=None, probe_y=None, probe_ok=None, plain=False):
        scene, mats = self.scene, self.mats
        thr = mi.Color3f(1.0)
        active = mi.Bool(True) if probe_ok is None else mi.Bool(probe_ok)
        out = {}

        def add(name, c):
            for i, ch in enumerate((c.x, c.y, c.z)):
                key = f'{name}{i}'
                out[key] = out[key] + ch if key in out else ch

        def adds(key, v):
            out[key] = out[key] + v if key in out else v
        cls2 = None
        okp = None
        for k in range(1, kmax + 1):
            si = scene.ray_intersect(ray, active)
            active &= si.is_valid()
            sid = self.shape_id(si)
            mid = dr.gather(U, self.SMAT, sid)
            wi = si.to_local(-ray.d)
            if mode == 'probe' and k == 1:
                active &= dr.norm(si.p - probe_y) < 1e-3
                okp = mi.Bool(active)
            active &= wi.z > 1e-6
            wi = dr.select(active, wi, mi.Vector3f(0, 0, 1))
            if mode == 'ecro' and k == 2:
                pid, bn = self.patch_bin(si, sid, -ray.d)
                i3 = (pid * NB + bn) * 3
                hl = thr * mi.Color3f(dr.gather(F, B, i3, active), dr.gather(F, B, i3 + 1, active), dr.gather(F, B, i3 + 2, active))
                hl = dr.select(active, hl, 0.0)
                add('tot' if plain else 'hl', hl)
            if k == 2 and not plain and mode == 'cam':
                cls2 = dr.gather(U, mats.cls, mid)
            if ref == 'ER':
                cl = CD.DrClosure83('ER', mats.states, mid)
            else:
                lam = dr.gather(F, mats.lam_f, mid) > 0.5
                alb = dr.gather(F, mats.alb_f, mid)
                gd = lambda j: dr.gather(F, mats.desc_cols[j], mid)
                lm = L.Mat(gd(0), gd(1), gd(2), mi.Color3f(gd(3), gd(4), gd(5)), mi.Color3f(gd(6), gd(7), gd(8)), gd(9))
            if not (mode == 'probe' and k == 1):
                c = mi.Color3f(0.0)
                for l in range(lights.n):
                    dl = lights.pos[l] - si.p
                    d2 = dr.squared_norm(dl)
                    wl = dl * dr.rsqrt(d2)
                    wo_l = si.to_local(wl)
                    on = active & (wo_l.z > 1e-6) & (dr.dot(-wl, lights.axis[l]) > lights.cc[l])
                    occ = scene.ray_test(si.spawn_ray_to(lights.pos[l]), on)
                    vis = on & ~occ
                    wo_s = dr.select(vis, wo_l, mi.Vector3f(0, 0, 1))
                    if ref == 'ER':
                        f = cl.eval(wi, wo_s, vis)
                    else:
                        r1 = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64((4 * k + l) << 32))
                        f = dr.select(lam, mi.Color3f(alb * wo_s.z * (1.0 / math.pi)), L.layered_eval(lm, wi, wo_s, r1, 1024))
                    c += dr.select(vis, thr * f * lights.rgb[l] * dr.rcp(d2), 0.0)
                if plain:
                    add('tot', c)
                elif mode == 'probe':
                    add('f', c)
                else:
                    add('e1' if k == 1 else ('e2' if k == 2 else 'h'), c)
                    if mode == 'cam' and k >= 2:
                        cm = mean3(c)
                        for q in range(3):
                            adds(f'{"c2" if k == 2 else "ch"}_{q}', dr.select(dr.eq(cls2, q), cm, 0.0))
            if k == kmax:
                break
            u1 = rng.next_float32(); u2 = mi.Point2f(rng.next_float32(), rng.next_float32())
            if ref == 'ER':
                wo, pdf, w, valid = cl.sample(wi, u1, u2, active)
            else:
                r2 = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64((4 * k + 3) << 32))
                wo_y, w_y, ok_y = L.layered_sample(lm, wi, r2, 1024)
                wo_d = mi.warp.square_to_cosine_hemisphere(u2)
                wo = dr.select(lam, wo_d, wo_y)
                w = dr.select(lam, mi.Color3f(alb), w_y)
                valid = active & dr.select(lam, wo_d.z > 1e-6, ok_y)
            thr = thr * dr.select(valid, w, 0.0)
            active &= valid
            ray = si.spawn_ray(si.to_world(wo))
        return out, okp

    @staticmethod
    def accum(out, index, n):
        keys = sorted(out)
        vals = [out[k] for k in keys]
        dr.eval(vals, index); dr.sync_thread()                  # evaluate BEFORE any scatter (Dr.Jit 0.4.6 lazy-scatter defect, WL84)
        bufs = []
        for v in vals:
            buf = dr.zeros(F, n)
            dr.scatter_reduce(dr.ReduceOp.Add, buf, v, index)
            bufs.append(buf)
        dr.eval(bufs); dr.sync_thread()
        return {k: np.asarray(b, np.float64) for k, b in zip(keys, bufs)}

    def render(self, lights, ref, cam, res, spp, seed, mode='cam', B=None, plain=False, b=32, kmax=None):
        """Per-pixel channel sums (float64) + timing. mode 'cam' (kmax D) or 'ecro' (kmax 2 + table lookup)."""
        kmax = kmax or (2 if mode == 'ecro' else D)
        n = res * res
        acc = {}
        T_ = {'trace_s': 0.0, 'eval_s': 0.0, 'gpu_ms': 0.0, 'launches': 0}
        t_all = time.perf_counter()
        done, launch = 0, 0
        while done < spp:
            bb = min(b, spp - done)
            t0 = time.perf_counter()
            dr.kernel_history()
            N = n * bb
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            rng = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(0))
            ray, pix = gen_rays(cam, res, bb, rng)
            out, _ = self.trace(ray, N, rng, sd, ref, kmax, lights, mode, B=B, plain=plain)
            t1 = time.perf_counter()
            a = self.accum(out, pix, n)
            T_['gpu_ms'] += sum(float(h.get('execution_time', 0.0)) for h in dr.kernel_history())
            for k_, v in a.items():
                acc[k_] = acc[k_] + v if k_ in acc else v
            T_['trace_s'] += t1 - t0; T_['eval_s'] += time.perf_counter() - t1; T_['launches'] += 1
            done += bb; launch += 1
        T_['wall_s'] = time.perf_counter() - t_all; T_['spp'] = spp
        return acc, T_

    def field(self, lights, spp, seed, kmax=D - 1, b=128):
        """Bin-averaged outgoing radiance of 2..kmax scattering events for every (patch, bin): [N, NB, 3] and the valid-sample count."""
        ns = self.N * NB
        acc = None
        done, launch = 0, 0
        while done < spp:
            bb = min(b, spp - done)
            N = ns * bb
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            rng = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(0))
            slot = dr.arange(U, N) % ns
            pid = slot // NB; bn = slot % NB
            p, nn, (t, bv, n) = self.patch_point(pid, rng.next_float32(), rng.next_float32())
            wl = self.dir_in_bin(bn, rng.next_float32(), rng.next_float32())
            w = t * wl.x + bv * wl.y + n * wl.z
            ok = dr.dot(w, nn) > 1e-3
            ray = mi.Ray3f(p + w * 2e-3, -w)
            out, okp = self.trace(ray, N, rng, sd, 'ER', kmax, lights, 'probe', probe_y=p, probe_ok=ok)
            out['cnt'] = dr.select(okp, F(1.0), F(0.0))
            a = self.accum(out, slot, ns)
            acc = a if acc is None else {k_: acc[k_] + v for k_, v in a.items()}
            done += bb; launch += 1
        cnt = acc['cnt'].reshape(self.N, NB)
        f = np.stack([acc[f'f{c}'] for c in range(3)], -1).reshape(self.N, NB, 3) / np.maximum(cnt, 1)[..., None]
        return f.astype(np.float32), cnt.astype(np.float32)

    def state_rays(self, idx=None):
        """Geometry state: for ray r = patch * K + k (k -> incident bin k // KAPPA), hit patch (-1: none / back face), outgoing bin at the
        hit patch, origin and end point (for the dirty test). idx: subset of ray ids (selective update)."""
        ri = dr.arange(U, self.N * K) if idx is None else U(np.ascontiguousarray(idx.astype(np.uint32)))
        pid = ri // K; k = ri % K
        ug = lambda j: dr.gather(F, self.US[j], ri)
        p, nn, (t, bv, n) = self.patch_point(pid, ug(0), ug(1))
        wl = self.dir_in_bin(k // KAPPA, ug(2), ug(3))
        w = t * wl.x + bv * wl.y + n * wl.z
        ok = dr.dot(w, nn) > 1e-3
        o = p + nn * 1e-4
        si = self.scene.ray_intersect(mi.Ray3f(o, w), ok)
        valid = ok & si.is_valid()
        sid = self.shape_id(si)
        front = si.to_local(-w).z > 1e-6
        ph, bh = self.patch_bin(si, sid, -w)
        hit = dr.select(valid & front, mi.Int32(ph), mi.Int32(-1))
        tt = dr.select(valid, si.t, 30.0)
        e = o + w * tt
        vals = [hit, bh, o.x, o.y, o.z, e.x, e.y, e.z]
        dr.eval(vals); dr.sync_thread()
        hit = np.asarray(hit).astype(np.int16); bh = np.asarray(bh).astype(np.uint8)
        seg = np.stack([np.asarray(v) for v in vals[2:]], 1).astype(np.float32)
        return hit, bh, seg

    def source(self, lights):
        """Source-field encoding s [N, NB, 3]: patch-averaged direct irradiance by incident-direction bin (patch frame)."""
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

    def gbuffer(self, cam, res):
        ray, pix = gen_rays(cam, res, 1, None)
        si = self.scene.ray_intersect(ray)
        sid = self.shape_id(si)
        front = si.to_local(-ray.d).z > 1e-6
        pid, _ = self.patch_bin(si, sid, -ray.d)
        ok = si.is_valid() & front
        g = lambda v: np.asarray(v)
        return {'ok': g(ok).reshape(res, res), 'sid': np.where(g(ok), g(sid), -1).reshape(res, res), 'pid': g(pid).reshape(res, res),
                'p': np.stack([g(si.p[k]) for k in range(3)], 1).reshape(res, res, 3).astype(np.float64),
                'n': np.stack([g(si.n[k]) for k in range(3)], 1).reshape(res, res, 3).astype(np.float64),
                'd': np.stack([g(ray.d[k]) for k in range(3)], 1).reshape(res, res, 3).astype(np.float64)}

    # ---------------------------------------------------------------- geometric queries for the region masks
    def light_vis(self, p, n, lights):
        """Bool per light: p (normal n) receives direct light (front side, inside the cone, unoccluded)."""
        out = []
        for l in range(lights.n):
            dl = lights.pos[l] - p
            dist = dr.norm(dl); wl = dl * dr.rcp(dist)
            on = (dr.dot(wl, n) > 1e-6) & (dr.dot(-wl, lights.axis[l]) > lights.cc[l])
            r = mi.Ray3f(p + n * 1e-4, wl); r.maxt = dist * (1.0 - 1e-4)
            out.append(on & ~self.scene.ray_test(r, on))
        return out

    def hit_info(self, o, d, active):
        si = self.scene.ray_intersect(mi.Ray3f(o, d), active)
        v = active & si.is_valid()
        return v, self.shape_id(si), si.p, si.n, v & (si.to_local(-d).z > 1e-6)

    def seg_clear(self, a, b, active):
        dl = b - a
        dist = dr.norm(dl); w = dl * dr.rcp(dist)
        r = mi.Ray3f(a + w * 1e-4, w); r.maxt = dist - 2e-4
        return active & ~self.scene.ray_test(r, active)


def pack(acc, res, spp):
    g = lambda k: (acc[k] / spp).reshape(res, res).astype(np.float32)
    o = {}
    for name in ('tot', 'e1', 'e2', 'h', 'hl'):
        if f'{name}0' in acc:
            o[name] = np.stack([g(f'{name}{c}') for c in range(3)], -1)
    for name in ('c2', 'ch'):
        if f'{name}_0' in acc:
            o[name] = np.stack([g(f'{name}_{q}') for q in range(3)], -1)
    return o


def config(name):
    """'E1_T0' -> (geometry, light-set name)."""
    e, t = name.split('_')
    return FX[e][t], FX[e]['light_set']


def gpu_mem():
    return os.popen(f'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^{os.getpid()}," ').read().strip()


def main():
    cmd, root = sys.argv[1], sys.argv[2]
    opt = {a.split('=')[0]: a.split('=')[1] for a in sys.argv[3:] if a.startswith('--') and '=' in a}
    args = [a for a in sys.argv[3:] if not a.startswith('--')]
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    if cmd == 'render':                       # render <root> <cfg> <ER|P> <seed> <spp> <tag> [--ls=] [--plain] [--kmax=] [--res=] [--cam=2]
        cfg, ref, seed, spp, tag = args[0], args[1], int(args[2]), int(args[3]), args[4]
        geom, ls = config(cfg)
        ls = opt.get('--ls', ls)
        res = int(opt.get('--res', FX['camera']['res']))
        cam = FX['camera2'] if opt.get('--cam') == '2' else FX['camera']
        ctx = Ctx(geom)
        acc, T_ = ctx.render(Lights(FX['light_sets'][ls]), ref, cam, res, spp, seed, 'cam', plain='--plain' in sys.argv,
                             kmax=int(opt.get('--kmax', D)), b=int(opt.get('--b', 32)))
        out = os.path.join(root, 'results', 'renders'); os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, f'{tag}.npz'), **pack(acc, res, spp))
        json.dump({'cfg': cfg, 'ls': ls, 'ref': ref, 'seed': seed, 'spp': spp, 'res': res, 'timing': T_, 'gpu_mem_mib': gpu_mem()},
                  open(os.path.join(out, f'{tag}.json'), 'w'), indent=1)
        print(tag, json.dumps(T_), flush=True)
    elif cmd == 'field':                      # field <root> <cfg> <seed> <spp> <tag> [--ls=]  (oracle projection target, two halves)
        cfg, seed, spp, tag = args[0], int(args[1]), int(args[2]), args[3]
        geom, ls = config(cfg)
        ls = opt.get('--ls', ls)
        ctx = Ctx(geom)
        lt = Lights(FX['light_sets'][ls])
        t0 = time.perf_counter()
        fa, ca = ctx.field(lt, spp // 2, seed)
        fb, cb = ctx.field(lt, spp // 2, seed + 500)
        out = os.path.join(root, 'results', 'fields'); os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, f'{tag}.npz'), fa=fa, fb=fb, ca=ca, cb=cb)
        print(tag, 'field', round(time.perf_counter() - t0, 2), 's', 'valid', float((ca > 0).mean()), flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
