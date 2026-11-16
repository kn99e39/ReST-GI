"""WL99 transport module: path-tagged ER reference path tracer (Dr.Jit on Mitsuba 3.5.1 intersection), exact-direction probes and the
oracle-self frame estimator used by the learning-free utility ceiling.

Estimator (as WL85 / WL86 pt86): next-event estimation to every point light at each vertex, then ER BSDF sampling; no Russian roulette;
max depth D vertices. A contribution made by NEE at the k-th vertex of a camera path is tagged by the path's vertex classes
(body / external):
  B1 : v1 on the body, k = 1                                   (first event: exact ER, not owned by N_b)
  S2 : v1, v2 on the body, k = 2                               (N_b output, direct-light input, one self-hit)
  S3 : v1..vk all on the body, k >= 3                          (N_b output, direct-light input, >= 2 self-hits)
  SX : v1..vr on the body with r >= 2, then an external vertex (N_b output, external-indirect input)
  X  : v1 on the body, v2 external                             (external indirect on the body, no self-hit)
  ES : v1 external, and vertices fb..k all on the body with k - fb >= 1   (N_b exits, direct-light input, delivered to external receivers)
  E  : every other contribution on an external pixel
Sum of all tags = the untagged estimator (test). Body pixel mask channel 'mb' = number of samples whose v1 is on the body.
oracle_self mode: identical except that a body-pixel path whose v2 is on the body is terminated after v1's NEE (N_b's whole output S2 + S3 + SX
is then supplied by a representation; here: the reference), i.e. the frame cost / variance of everything that is not N_b at body pixels.
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ['fnto_compiled83', 'fnto_compiled82', 'fnto_compiled81', 'ecro86']:
    sys.path.insert(0, os.path.join(HERE, '..', d))
sys.path.insert(0, HERE)
import layered81 as L  # noqa: E402
mi = L.init(os.environ.get('C80_MI_VARIANT', 'cuda_ad_rgb'))
import drjit as dr  # noqa: E402
import closure83_dr as CD  # noqa: E402
import scene99 as S  # noqa: E402

T = mi.ScalarTransform4f
F = mi.Float; U = mi.UInt32
TAGS = ['B1', 'S2', 'S3', 'SX', 'X', 'ES', 'E']
ROOM = [  # (centre, axis1, axis2, half1, half2, material)  normal = a1 x a2 (into the room); open towards -y
    (np.array([0, 0, 0.0]), S.X, S.Y, 2.0, 2.0, 'B'),
    (np.array([0, 0, 3.0]), S.Y, S.X, 2.0, 2.0, 'LW'),
    (np.array([0, 2.0, 1.5]), S.X, S.Z, 2.0, 1.5, 'LW'),
    (np.array([-2.0, 0, 1.5]), S.Y, S.Z, 2.0, 1.5, 'LW'),
    (np.array([2.0, 0, 1.5]), S.Z, S.Y, 1.5, 2.0, 'LW'),
]


def rect_matrix(c, a1, a2, h1, h2):
    M = np.eye(4)
    M[:3, 0] = a1 * h1; M[:3, 1] = a2 * h2; M[:3, 2] = np.cross(a1, a2); M[:3, 3] = c
    return M


def fl(a):
    return F(np.ascontiguousarray(np.asarray(a, np.float32).ravel()))


class Lights:
    def __init__(self, lights):
        op = lambda v: dr.opaque(F, float(v))
        self.n = len(lights)
        self.pos = [mi.Point3f(*[op(v) for v in l['pos']]) for l in lights]
        self.rgb = [mi.Color3f(*[op(v) for v in l['rgb']]) for l in lights]


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
    x = (F(jj) + rng.next_float32()) * (2.0 / res) - 1.0
    y = 1.0 - (F(ii) + rng.next_float32()) * (2.0 / res)
    op = lambda v: dr.opaque(F, float(v))
    fw = [op(fwd[k]) for k in range(3)]; rt = [op(s * right[k]) for k in range(3)]; uv = [op(s * up[k]) for k in range(3)]
    d = dr.normalize(mi.Vector3f(*[fw[k] + x * rt[k] + y * uv[k] for k in range(3)]))
    return mi.Ray3f(mi.Point3f(*[op(o[k]) for k in range(3)]), d), pix


class Ctx:
    """One Mitsuba scene with a FIXED shape layout (so kernels compile once per process): 5 room rectangles, NPL body-plate cube slots,
    NBX Lambert-box cube slots. set_cfg(cfg) moves the slots (unused ones parked 50 m below the floor) and rewrites the material /
    body tables; the light set (count fixed per process) and the camera are opaque (no recompilation)."""
    NPL, NBX = 3, 2
    PARK = np.diag([1e-3, 1e-3, 1e-3, 1.0]); PARK[:3, 3] = [0, 0, -50.0]

    def __init__(self, cfg, nlights=None):
        z = np.load(os.path.join(HERE, '..', 'ecro86', 'data', 'mats86.npz'))
        st = z['states'].astype(np.float32)
        black = st[3].copy(); black[[0, 8, 16]] = 0.0                     # Lambert albedo 0 (walls of the 'dark' room)
        self.states = np.concatenate([st, black[None]], 0)             # ids 0..5 as mats86, 6 = black
        d = {'type': 'scene'}
        names = []
        for i, (c, a1, a2, h1, h2, m) in enumerate(ROOM):
            d[f'r{i}'] = {'type': 'rectangle', 'to_world': T(rect_matrix(c, a1, a2, h1, h2).tolist())}
            names.append(f'r{i}')
        for j in range(self.NPL):
            d[f'b{j}'] = {'type': 'cube', 'to_world': T(self.PARK.tolist())}
            names.append(f'b{j}')
        for j in range(self.NBX):
            d[f'x{j}'] = {'type': 'cube', 'to_world': T(self.PARK.tolist())}
            names.append(f'x{j}')
        self.scene = mi.load_dict(d)
        by = {s.id(): mi.ShapePtr(s) for s in self.scene.shapes()}
        self.ptrs = [by[n] for n in names]
        self.names = names
        self.params = mi.traverse(self.scene)
        self.V0 = {}
        for n in names[5:]:
            V = np.asarray(self.params[f'{n}.vertex_positions'], np.float64).reshape(-1, 3)
            self.V0[n] = (V - self.PARK[:3, 3]) / 1e-3                 # unit-cube vertices
        self.N0 = {}
        for n in names[5:]:
            if f'{n}.vertex_normals' in self.params:
                Nn = np.asarray(self.params[f'{n}.vertex_normals'], np.float64).reshape(-1, 3)
                if Nn.size:
                    self.N0[n] = Nn                                          # PARK is a uniform scale: unit-cube normals
        self.nl = nlights or len(cfg['lights'])
        self.set_cfg(cfg)

    def set_cfg(self, cfg):
        self.cfg = cfg
        bd = cfg['body']
        Mp = S.pose_matrix(bd['pose'])
        self.plates = S.body_plates(bd['kind'], bd['prm'])
        assert len(self.plates) <= self.NPL and len(cfg.get('boxes', [])) <= self.NBX and len(cfg['lights']) == self.nl
        dark = cfg.get('room', 'gm1') == 'dark'                         # 'dark': floor B + black walls / ceiling (open-air-like)
        mats = [S.MAT_ID[m] if (i == 0 or not dark) else 6 for i, (*_, m) in enumerate(ROOM)]
        isb = [0] * 5
        for j in range(self.NPL):
            used = j < len(self.plates)
            M = Mp @ self.plates[j]['M'] if used else self.PARK
            self._place(f'b{j}', M)
            mats.append(S.MAT_ID[bd['mat']]); isb.append(1 if used else 0)
        boxes = cfg.get('boxes', [])
        for j in range(self.NBX):
            M = S.box_matrix(boxes[j]) if j < len(boxes) else self.PARK
            self._place(f'x{j}', M)
            mats.append(S.MAT_ID['LB']); isb.append(0)
        self.params.update()
        self.nbody = len(self.plates)
        self.SMAT = U(np.array(mats, np.uint32)); self.SBODY = U(np.array(isb, np.uint32))
        self.lights = Lights(cfg['lights'])
        self.Mpose = Mp
        dr.eval(self.SMAT, self.SBODY)

    def _place(self, name, M):
        A = M[:3, :3]
        self.params[f'{name}.vertex_positions'] = fl(self.V0[name] @ A.T + M[:3, 3])
        if name in self.N0:
            nn = self.N0[name] @ np.linalg.inv(A)                       # normals transform with inv(A)^T (row form)
            self.params[f'{name}.vertex_normals'] = fl(nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-30))

    def shape_id(self, si):
        sid = U(0)
        for i, p in enumerate(self.ptrs):
            sid = dr.select(dr.eq(si.shape, p), U(i), sid)
        return sid

    # ------------------------------------------------------------------------------------------------ transport
    def trace(self, ray, rng, kmax, mode='cam', probe_ok=None):
        """mode 'cam' (tagged), 'oracle_self' (terminate body-pixel paths whose v2 is on the body), 'probe' (no NEE at k = 1;
        everything is reported as leaving v1 towards -ray.d; tags as for 'cam' but the B1 term is absent)."""
        scene, lights = self.scene, self.lights
        thr = mi.Color3f(1.0)
        active = mi.Bool(True) if probe_ok is None else mi.Bool(probe_ok)
        out = {}

        def add(name, c, m):
            for i, ch in enumerate((c.x, c.y, c.z)):
                key = f'{name}{i}'
                v = dr.select(m, ch, 0.0)
                out[key] = out[key] + v if key in out else v

        onb1 = mi.Bool(False); runall = mi.Bool(False); run = U(0)
        fb = U(0); allb = mi.Bool(False)
        for k in range(1, kmax + 1):
            si = scene.ray_intersect(ray, active)
            active &= si.is_valid()
            sid = self.shape_id(si)
            mid = dr.gather(U, self.SMAT, sid)
            isb = dr.eq(dr.gather(U, self.SBODY, sid), 1) & active
            wi = si.to_local(-ray.d)
            active &= wi.z > 1e-6
            isb &= active
            if k == 1:
                onb1 = mi.Bool(isb); runall = mi.Bool(isb); run = dr.select(isb, U(1), U(0))
                fb = dr.select(isb, U(1), U(0)); allb = mi.Bool(isb)
                out['mb'] = dr.select(isb, F(1.0), F(0.0))
                out['mv'] = dr.select(active, F(1.0), F(0.0))
            else:
                runall &= isb
                run = dr.select(runall, run + 1, run)
                newb = dr.eq(fb, 0) & isb
                allb = dr.select(newb, mi.Bool(True), allb & isb)
                fb = dr.select(newb, U(k), fb)
                if mode == 'oracle_self' and k == 2:
                    active &= ~(onb1 & isb)                     # N_b's output at body pixels comes from the representation
            wi = dr.select(active, wi, mi.Vector3f(0, 0, 1))
            cl = CD.DrClosure83('ER', self.states, mid)
            if not (mode == 'probe' and k == 1):
                c = mi.Color3f(0.0)
                for l in range(lights.n):
                    dl = lights.pos[l] - si.p
                    d2 = dr.squared_norm(dl)
                    wl = dl * dr.rsqrt(d2)
                    wo_l = si.to_local(wl)
                    on = active & (wo_l.z > 1e-6)
                    occ = scene.ray_test(si.spawn_ray_to(lights.pos[l]), on)
                    vis = on & ~occ
                    f = cl.eval(wi, dr.select(vis, wo_l, mi.Vector3f(0, 0, 1)), vis)
                    c += dr.select(vis, thr * f * lights.rgb[l] * dr.rcp(d2), 0.0)
                if k == 1:
                    add('B1', c, onb1)
                    add('E', c, ~onb1)
                else:
                    add('S2' if k == 2 else 'S3', c, onb1 & runall)
                    add('SX', c, onb1 & ~runall & (run >= 2))
                    add('X', c, onb1 & (run < 2))
                    es = ~onb1 & (fb > 0) & allb & (U(k) >= fb + 1)
                    add('ES', c, es)
                    add('E', c, ~onb1 & ~es)
            if k == kmax:
                break
            u1 = rng.next_float32(); u2 = mi.Point2f(rng.next_float32(), rng.next_float32())
            wo, pdf, w, valid = cl.sample(wi, u1, u2, active)
            thr = thr * dr.select(valid, w, 0.0)
            active &= valid
            ray = si.spawn_ray(si.to_world(wo))
        return out

    @staticmethod
    def accum(out, index, n):
        keys = sorted(out)
        vals = [out[k] for k in keys]
        dr.eval(vals, index); dr.sync_thread()                  # evaluate BEFORE any scatter (Dr.Jit lazy-scatter defect, WL84)
        bufs = []
        for v in vals:
            buf = dr.zeros(F, n)
            dr.scatter_reduce(dr.ReduceOp.Add, buf, v, index)
            bufs.append(buf)
        dr.eval(bufs); dr.sync_thread()
        return {k: np.asarray(b, np.float64) for k, b in zip(keys, bufs)}

    def render(self, cam, res, spp, seed, kmax=8, mode='cam', b=16):
        """Per-pixel channel sums (float64) over spp samples + timing (GPU kernel ms from Dr.Jit's kernel history, wall s)."""
        n = res * res
        acc = {}
        dr.set_flag(dr.JitFlag.KernelHistory, True)
        Tm = {'gpu_ms': 0.0, 'wall_s': 0.0, 'launches': 0, 'spp': spp}
        t_all = time.perf_counter()
        done, launch = 0, 0
        while done < spp:
            bb = min(b, spp - done)
            dr.kernel_history()
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            rng = mi.PCG32(size=n * bb, initstate=sd, initseq=mi.UInt64(0))
            ray, pix = gen_rays(cam, res, bb, rng)
            out = self.trace(ray, rng, kmax, mode)
            a = self.accum(out, pix, n)
            Tm['gpu_ms'] += sum(float(h.get('execution_time', 0.0)) for h in dr.kernel_history())
            for k_, v in a.items():
                acc[k_] = acc[k_] + v if k_ in acc else v
            Tm['launches'] += 1
            done += bb; launch += 1
        Tm['wall_s'] = time.perf_counter() - t_all
        return acc, Tm

    def pixel_hits(self, cam, res, pix):
        """Pixel-centre camera rays for pixel ids pix: hit point (world), unit direction towards the camera, is-body flag."""
        o, fwd, right, up, s = cam_basis(cam)
        pix = np.asarray(pix)
        ii, jj = pix // res, pix % res
        x = (jj + 0.5) * (2.0 / res) - 1.0; yv = 1.0 - (ii + 0.5) * (2.0 / res)
        d = fwd[None] + s * x[:, None] * right[None] + s * yv[:, None] * up[None]
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        ray = mi.Ray3f(mi.Point3f(*[fl(np.full(len(pix), o[k])) for k in range(3)]), mi.Vector3f(*[fl(d[:, k]) for k in range(3)]))
        si = self.scene.ray_intersect(ray)
        isb = dr.eq(dr.gather(U, self.SBODY, self.shape_id(si)), 1) & si.is_valid()
        p = np.stack([np.asarray(si.p.x), np.asarray(si.p.y), np.asarray(si.p.z)], 1)
        return p, -d, np.asarray(isb)

    def probe(self, y, w, spp, seed, kmax=8, b=256):
        """Exact-direction probes: radiance leaving body-surface points y [P,3] (world) along unit w [P,3] (world), tagged.
        Rays start 2 mm outside along w and point back to y; a probe is valid only if its first hit is within 1 mm of y."""
        P = len(y)
        acc = {}
        done, launch = 0, 0
        o_np = np.asarray(y, np.float64) + 2e-3 * np.asarray(w, np.float64)
        while done < spp:
            bb = min(b, spp - done)
            N = P * bb
            lane = dr.arange(U, N); pid = lane % P
            g = lambda a, j: dr.gather(F, fl(a[:, j]), pid)
            o = mi.Point3f(g(o_np, 0), g(o_np, 1), g(o_np, 2))
            yy = mi.Point3f(g(np.asarray(y), 0), g(np.asarray(y), 1), g(np.asarray(y), 2))
            ww = mi.Vector3f(g(np.asarray(w), 0), g(np.asarray(w), 1), g(np.asarray(w), 2))
            ray = mi.Ray3f(o, -ww)
            si0 = self.scene.ray_intersect(ray)
            ok = si0.is_valid() & (dr.norm(si0.p - yy) < 1e-3)
            sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)
            rng = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(0))
            out = self.trace(ray, rng, kmax, 'probe', probe_ok=ok)
            out['ok'] = dr.select(ok, F(1.0), F(0.0))
            a = self.accum(out, pid, P)
            for k_, v in a.items():
                acc[k_] = acc[k_] + v if k_ in acc else v
            done += bb; launch += 1
        return acc


def pack(acc, n_samples):
    """-> dict tag -> [n, 3] mean radiance (per pixel / probe), plus 'tot', 'self' (= S2 + S3 + SX) and 'mb' fraction."""
    out = {}
    for t in TAGS:
        if f'{t}0' in acc:
            out[t] = np.stack([acc[f'{t}{i}'] for i in range(3)], 1) / n_samples
    out['tot'] = sum(out[t] for t in out)
    out['self'] = sum(out[t] for t in ('S2', 'S3', 'SX') if t in out)
    out['Sd'] = sum(out[t] for t in ('S2', 'S3') if t in out)
    for k in ('mb', 'mv', 'ok'):
        if k in acc:
            out[k] = acc[k] / n_samples
    return out
