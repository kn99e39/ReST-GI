"""WL85 characterisation path tracer (Dr.Jit on Mitsuba 3.5.1 ray intersection). Not a candidate mechanism: a reference renderer with
geometric and lobe-class instrumentation.

  python pt85.py render <root> <cfg> <ER|P> <seed> <spp> <tag> [--plain] [--res=N] [--cam=2] [--frac=f]
Estimator: at each of at most D = 6 surface vertices, next-event estimation to the point light, then BSDF sampling. No Russian roulette.
ER: frozen DrClosure83 (per-lane material table). P: layered81 random walk (one walk per query) / exact Lambert on Lambert surfaces.
Instrumented outputs per pixel (sums over samples; divide by spp):
  tot[3], dirty[3], dmov (channel mean, contributions with a vertex on the baffle), depth[D], depth_dirty[D],
  ER only: dir1[3] / dir1_dirty[3] (k = 1 by receiver class), mat[9] / mat_dirty[9] (k >= 2: receiver class a x sender class b, index 3a + b).
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ['fnto_compiled83', 'fnto_compiled82', 'fnto_compiled81']:
    sys.path.insert(0, os.path.join(HERE, '..', d))
import layered81 as L  # noqa: E402
mi = L.init('cuda_ad_rgb')
import drjit as dr  # noqa: E402
import closure83_dr as CD  # noqa: E402

P = json.load(open(os.path.join(HERE, 'protocol', 'char85.json')))
FX = P['stage1_fixture']
D = 6
T = mi.ScalarTransform4f
SHAPES = ['floor', 'wall', 'panel', 'sphere', 'baffle']
MAT_OF = {'floor': 0, 'panel': 1, 'sphere': 2, 'wall': 3, 'baffle': 4}
LPOS = FX['light']['position']; LINT = FX['light']['intensity']


def baffle_center(cfg, frac=None):
    c = np.array(FX['shapes']['baffle']['center_T0'], np.float64); t = np.array(FX['edit']['translation'], np.float64)
    if frac is not None:
        return c + frac * t
    if cfg == 'T1':
        return c + t
    if cfg == 'RET':
        return (c + t) - t
    return c


def box_of(center):
    h = np.array(FX['shapes']['baffle']['half'], np.float64) + 1e-3     # 1 mm guard: grazing segments are flagged dirty in both configurations
    return center - h, center + h


def scene_dict(cfg, frac=None, bsdfs=None):
    S = FX['shapes']
    bc = baffle_center(cfg, frac)
    df = {'type': 'diffuse'}
    b = bsdfs or {k: df for k in SHAPES}
    return {'type': 'scene',
            'floor': {'type': 'rectangle', 'to_world': T.scale([S['floor']['half'][0], S['floor']['half'][1], 1]), 'bsdf': b['floor']},
            'wall': {'type': 'rectangle', 'to_world': T.translate(S['wall']['center']) @ T.rotate([1, 0, 0], 90) @ T.scale([S['wall']['half'][0], S['wall']['half'][1], 1]), 'bsdf': b['wall']},
            'panel': {'type': 'rectangle', 'to_world': T.translate(S['panel']['center']) @ T.rotate([0, 1, 0], 90) @ T.scale([S['panel']['half'][0], S['panel']['half'][1], 1]), 'bsdf': b['panel']},
            'sphere': {'type': 'sphere', 'center': S['sphere']['center'], 'radius': S['sphere']['radius'], 'bsdf': b['sphere']},
            'baffle': {'type': 'cube', 'to_world': T.translate([float(x) for x in bc]) @ T.scale(S['baffle']['half']), 'bsdf': b['baffle']},
            'light': {'type': 'point', 'position': LPOS, 'intensity': {'type': 'rgb', 'value': LINT}}}


def load_scene(cfg, frac=None, bsdfs=None):
    sc = mi.load_dict(scene_dict(cfg, frac, bsdfs))
    ptr = {}
    for s in sc.shapes():
        ptr[s.id()] = mi.ShapePtr(s)
    return sc, ptr


def cam_basis(cam):
    o = np.array(cam['origin'], np.float64); t = np.array(cam['target'], np.float64)
    fwd = t - o; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return o, fwd, right, up, math.tan(math.radians(cam['fov_deg']) / 2)


def gen_rays(cam, res, b, rng):
    o, fwd, right, up, s = cam_basis(cam)
    n = res * res
    lane = dr.arange(mi.UInt32, n * b)
    pix = lane % n
    ii, jj = pix // res, pix % res
    jx, jy = (rng.next_float32(), rng.next_float32()) if rng is not None else (mi.Float(0.5), mi.Float(0.5))
    x = (mi.Float(jj) + jx) * (2.0 / res) - 1.0
    y = 1.0 - (mi.Float(ii) + jy) * (2.0 / res)
    d = dr.normalize(mi.Vector3f(*[float(fwd[k]) + s * x * float(right[k]) + s * y * float(up[k]) for k in range(3)]))
    return mi.Ray3f(mi.Point3f(*[mi.Float(float(o[k])) for k in range(3)]), d), pix


def seg_box(o, p, lo, hi):
    d = p - o
    d = mi.Vector3f(*[dr.select(dr.abs(d[k]) < 1e-9, 1e-9, d[k]) for k in range(3)])
    inv = dr.rcp(d)
    t0 = (mi.Point3f(*[float(v) for v in lo]) - o) * inv
    t1 = (mi.Point3f(*[float(v) for v in hi]) - o) * inv
    tmin = dr.maximum(dr.maximum(dr.minimum(t0.x, t1.x), dr.minimum(t0.y, t1.y)), dr.minimum(t0.z, t1.z))
    tmax = dr.minimum(dr.minimum(dr.maximum(t0.x, t1.x), dr.maximum(t0.y, t1.y)), dr.maximum(t0.z, t1.z))
    return (tmax >= dr.maximum(tmin, 0.0)) & (tmin <= 1.0)


def both_box(o, p, boxes):
    """Segment intersects the (1 mm inflated) baffle volume of EITHER configuration: the same test in T0 and T1, so the clean set is
    identical in both by construction."""
    return seg_box(o, p, *boxes[0]) | seg_box(o, p, *boxes[1])


class Materials:
    def __init__(self, fixture_npz):
        z = np.load(fixture_npz)
        self.states = z['states']; self.cls = z['cls']; self.desc = z['desc']; self.lambert = z['lambert']; self.albedo = z['albedo']
        self.masked = []
        for c in range(3):
            st = self.states.copy()
            for m in range(st.shape[0]):
                for r in range(8):
                    if self.cls[m, r] != c:
                        for ch in range(3):
                            st[m, ch * 8 + r] = 0.0
            self.masked.append(st)
        self.cls_cols = [mi.UInt32(np.ascontiguousarray(self.cls[:, r].astype(np.uint32))) for r in range(8)]
        self.desc_cols = [mi.Float(np.ascontiguousarray(self.desc[:, j])) for j in range(10)]
        self.lam_f = mi.Float(np.ascontiguousarray(self.lambert.astype(np.float32)))
        self.alb_f = mi.Float(np.ascontiguousarray(self.albedo))


def mat_id(si, ptr):
    mid = mi.UInt32(0)
    for name in SHAPES:
        mid = dr.select(dr.eq(si.shape, ptr[name]), mi.UInt32(MAT_OF[name]), mid)
    return mid


def mean3(c):
    return (c.x + c.y + c.z) * (1.0 / 3.0)


def class_eval(cl, cls_l, wi, wo, active):
    """Channel-mean eval of the closure split by lobe class (0 S, 1 M, 2 B): per-lobe terms are computed once with the closure's own
    methods (_lobe, _s, _refr) and assembled exactly as DrClosure83.eval_pdf does; cls_l[r] is the per-lane class of lobe r (0 = Lambert).
    The sum over classes equals mean3(cl.eval) (validated by test V2)."""
    ci, co = wi.z, wo.z
    valid = mi.Bool(active) & (ci > CD.EPS) & (co > CD.EPS) & (dr.norm(wi + wo) > 1e-6)
    co_ = dr.maximum(co, 0.0)
    lamb = co_ * CD.INV_PI
    si_, so_ = cl._s(ci), cl._s(co)
    S = [si_[c] * so_[c] for c in range(3)]
    wi_r, wo_r = cl._refr(wi), cl._refr(wo)
    ie2 = dr.rcp(cl.eta * cl.eta)
    vals = [(cl.lam[0][0] * S[0] + cl.lam[1][0] * S[1] + cl.lam[2][0] * S[2]) * lamb * (1.0 / 3.0)]
    for k in range(7):
        inner = k in CD.INNER
        g, _ = cl._lobe(k, wi_r if inner else wi, wo_r if inner else wo, True, False)
        t = g * co_ * (ie2 if inner else 1.0)
        if inner:
            w = cl.lam[0][k + 1] * S[0] + cl.lam[1][k + 1] * S[1] + cl.lam[2][k + 1] * S[2]
        else:
            w = cl.lam[0][k + 1] + cl.lam[1][k + 1] + cl.lam[2][k + 1]
        vals.append(w * t * (1.0 / 3.0))
    out = []
    for q in range(3):
        acc = mi.Float(0.0)
        for r in range(8):
            acc += dr.select(dr.eq(cls_l[r], q), vals[r], 0.0)
        out.append(dr.select(valid, acc, 0.0))
    return out


def trace(scene, ptr, mats, ref, cam, res, b, seed, launch, other_box, moved, instrument, depth=D):
    """One launch of res^2 * b paths. Returns dict name -> per-lane Float (not yet evaluated), pix, counters."""
    n = res * res; N = n * b
    sd = dr.opaque(mi.UInt64, seed * 1000003 + launch)            # opaque: the kernel is identical for every launch (cache hit)
    rng = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(0))
    ray, pix = gen_rays(cam, res, b, rng)
    lpos = mi.Point3f(*[float(v) for v in LPOS])
    thr = mi.Color3f(1.0)
    active = mi.Bool(True)
    dirty = mi.Bool(False); onm = mi.Bool(False)
    out = {}
    add = lambda k, v: out.__setitem__(k, out[k] + v if k in out else v)
    geo = instrument
    phi1 = phi2 = None
    cnt = {'paths': N}
    seg_tot = mi.Float(0.0); seg_dirty = mi.Float(0.0); path_dirty = mi.Bool(False)
    for k in range(1, depth + 1):
        si = scene.ray_intersect(ray, active)
        active &= si.is_valid()
        mid = mat_id(si, ptr)
        wi = si.to_local(-ray.d)
        active &= wi.z > 1e-6
        wi = dr.select(active, wi, mi.Vector3f(0, 0, 1))
        if geo:
            seg_tot += dr.select(active, 1.0, 0.0)
            if moved:
                hitm = dr.eq(si.shape, ptr['baffle'])
                dirty |= active & (both_box(ray.o, si.p, other_box) | hitm)
                onm |= active & hitm
            seg_dirty += dr.select(active & dirty, 1.0, 0.0)
        # ---- next-event estimation
        dl = lpos - si.p
        d2 = dr.squared_norm(dl)
        wl = dl * dr.rsqrt(d2)
        wo_l = si.to_local(wl)
        occ = scene.ray_test(si.spawn_ray_to(lpos), active)
        vis = active & ~occ & (wo_l.z > 1e-6)
        wo_s = dr.select(vis, wo_l, mi.Vector3f(0, 0, 1))
        if ref == 'ER':
            cl = CD.DrClosure83('ER', mats.states, mid)
            f = cl.eval(wi, wo_s, vis)
        else:
            lam = dr.gather(mi.Float, mats.lam_f, mid) > 0.5
            alb = dr.gather(mi.Float, mats.alb_f, mid)
            g = lambda j: dr.gather(mi.Float, mats.desc_cols[j], mid)
            lm = L.Mat(g(0), g(1), g(2), mi.Color3f(g(3), g(4), g(5)), mi.Color3f(g(6), g(7), g(8)), g(9))
            r1 = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64((2 * k) << 32))
            fl = L.layered_eval(lm, wi, wo_s, r1, 1024)
            f = dr.select(lam, mi.Color3f(alb * wo_s.z * (1.0 / math.pi)), fl)
            f = dr.select(vis, f, 0.0)
        c = thr * f * (LINT * dr.rcp(d2))
        c = dr.select(vis, c, 0.0)
        cm = mean3(c)
        add('tot0', c.x); add('tot1', c.y); add('tot2', c.z)
        if geo:
            add(f'depth{k}', cm)
            if moved:
                dc = dirty | both_box(si.p, lpos, other_box)
                cd = dr.select(dc, c, 0.0)
                add('dirty0', cd.x); add('dirty1', cd.y); add('dirty2', cd.z)
                add(f'ddepth{k}', dr.select(dc, cm, 0.0))
                add('dmov', dr.select(onm, cm, 0.0))
                path_dirty |= vis & dc
                seg_dirty += dr.select(active & dc, 1.0, 0.0)
            seg_tot += dr.select(active, 1.0, 0.0)
        if instrument and ref == 'ER':
            cls_l = [dr.gather(mi.UInt32, mats.cls_cols[r], mid) for r in range(8)]
            fc = class_eval(cl, cls_l, wi, wo_s, vis)
            ft = dr.maximum(fc[0] + fc[1] + fc[2], 1e-30)
            psi = [x / ft for x in fc]
            for a in range(3):
                if k == 1:
                    v = cm * psi[a]
                    add(f'dir{a}', v)
                    if moved:
                        add(f'ddir{a}', dr.select(dc, v, 0.0))
                else:
                    for bb in range(3):
                        v = cm * phi1[a] * (psi[bb] if k == 2 else phi2[bb])
                        add(f'mat{3 * a + bb}', v)
                        if moved:
                            add(f'dmat{3 * a + bb}', dr.select(dc, v, 0.0))
        if k == depth:
            break
        # ---- BSDF sampling
        u1 = rng.next_float32(); u2 = mi.Point2f(rng.next_float32(), rng.next_float32())
        if ref == 'ER':
            wo, pdf, w, valid = cl.sample(wi, u1, u2, active)
        else:
            r2 = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64((2 * k + 1) << 32))
            wo_y, w_y, ok_y = L.layered_sample(lm, wi, r2, 1024)
            wo_d = mi.warp.square_to_cosine_hemisphere(u2)
            wo = dr.select(lam, wo_d, wo_y)
            w = dr.select(lam, mi.Color3f(alb), w_y)
            valid = active & dr.select(lam, wo_d.z > 1e-6, ok_y)
        if instrument and ref == 'ER' and k <= 2:
            wo_c = dr.select(valid, wo, mi.Vector3f(0, 0, 1))
            fc = class_eval(cl, cls_l, wi, wo_c, valid)
            ft = dr.maximum(fc[0] + fc[1] + fc[2], 1e-30)
            ph = [x / ft for x in fc]
            if k == 1:
                phi1 = ph
            else:
                phi2 = ph
        thr = thr * dr.select(valid, w, 0.0)
        active &= valid
        ray = si.spawn_ray(si.to_world(wo))
    if geo:
        cnt['seg_tot'] = dr.sum(seg_tot); cnt['seg_dirty'] = dr.sum(seg_dirty)
        cnt['paths_dirty'] = dr.sum(dr.select(path_dirty | dirty, 1.0, 0.0))
    return out, pix, cnt


def render(scene, ptr, mats, ref, cam, res, spp, seed, other_box, moved, instrument, b=None, depth=D):
    """Accumulated per-pixel sums (float64) for every output channel + timing."""
    b = b or P['stage2_references']['launch_spp']
    n = res * res
    acc = {}
    T_ = {'trace_s': 0.0, 'eval_s': 0.0, 'readback_s': 0.0, 'gpu_ms': 0.0, 'launches': 0}
    C = {}
    t_all = time.perf_counter()
    done, launch = 0, 0
    while done < spp:
        bb = min(b, spp - done)
        t0 = time.perf_counter()
        dr.kernel_history()
        out, pix, cnt = trace(scene, ptr, mats, ref, cam, res, bb, seed, launch, other_box, moved, instrument, depth)
        keys = sorted(out)
        vals = [out[k] for k in keys]
        t1 = time.perf_counter()
        dr.eval(vals, list(v for v in cnt.values() if not isinstance(v, int)))          # evaluate BEFORE any scatter (Dr.Jit 0.4.6 lazy-scatter defect, WL84)
        dr.sync_thread()
        t2 = time.perf_counter()
        bufs = []
        for v in vals:
            buf = dr.zeros(mi.Float, n)
            dr.scatter_reduce(dr.ReduceOp.Add, buf, v, pix)
            bufs.append(buf)
        dr.eval(bufs); dr.sync_thread()
        T_['gpu_ms'] += sum(float(h.get('execution_time', 0.0)) for h in dr.kernel_history())
        for k, buf in zip(keys, bufs):
            a = np.asarray(buf, np.float64)
            acc[k] = acc[k] + a if k in acc else a
        for k, v in cnt.items():
            C[k] = C.get(k, 0.0) + (float(v) if isinstance(v, int) else float(np.asarray(v)[0]))
        t3 = time.perf_counter()
        T_['trace_s'] += t1 - t0; T_['eval_s'] += t2 - t1; T_['readback_s'] += t3 - t2; T_['launches'] += 1
        done += bb; launch += 1
    T_['wall_s'] = time.perf_counter() - t_all; T_['spp'] = spp
    return acc, C, T_


def pack(acc, res, spp):
    """Channel dict -> arrays (means per sample)."""
    g = lambda k: (acc[k] / spp).reshape(res, res).astype(np.float32) if k in acc else np.zeros((res, res), np.float32)
    o = {'tot': np.stack([g(f'tot{c}') for c in range(3)], -1)}
    if 'depth1' in acc:
        o['depth'] = np.stack([g(f'depth{k}') for k in range(1, D + 1)], -1)
        o['dirty'] = np.stack([g(f'dirty{c}') for c in range(3)], -1)
        o['ddepth'] = np.stack([g(f'ddepth{k}') for k in range(1, D + 1)], -1)
        o['dmov'] = g('dmov')
    if 'dir0' in acc:
        o['dir'] = np.stack([g(f'dir{a}') for a in range(3)], -1); o['ddir'] = np.stack([g(f'ddir{a}') for a in range(3)], -1)
        o['mat'] = np.stack([g(f'mat{i}') for i in range(9)], -1); o['dmat'] = np.stack([g(f'dmat{i}') for i in range(9)], -1)
    return o


def setup(root, cfg, frac=None):
    mats = Materials(os.path.join(root, 'results', 'census', 'fixture.npz'))
    scene, ptr = load_scene(cfg, frac)
    moved = cfg in ('T0', 'T1')
    other = (box_of(baffle_center('T0')), box_of(baffle_center('T1')))
    return mats, scene, ptr, other, moved


def gbuffer(scene, ptr, cam, res):
    ray, pix = gen_rays(cam, res, 1, None)
    si = scene.ray_intersect(ray)
    mid = dr.select(si.is_valid(), mi.Int32(mat_id(si, ptr)), mi.Int32(-1))
    p = np.stack([np.asarray(si.p[k], np.float64) for k in range(3)], 1)
    return np.asarray(mid).reshape(res, res), p.reshape(res, res, 3)


def gbuf(root):
    res = FX['camera']['res']
    o = {}
    for i, cfg in enumerate(['T0', 'T1']):
        sc, ptr = load_scene(cfg)
        m, p = gbuffer(sc, ptr, FX['camera'], res)
        o[f'mid{i}'] = m; o[f'p{i}'] = p
    np.savez_compressed(os.path.join(root, 'results', 'gbuffer.npz'), **o)


def main():
    if sys.argv[1] == 'gbuf':
        return gbuf(sys.argv[2])
    root, cfg, ref, seed, spp, tag = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
    opt = {a.split('=')[0]: a.split('=')[1] for a in sys.argv[8:] if '=' in a}
    res = int(opt.get('--res', FX['camera']['res']))
    cam = FX['camera2'] if opt.get('--cam') == '2' else FX['camera']
    frac = float(opt['--frac']) if '--frac' in opt else None
    instrument = '--plain' not in sys.argv
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    t0 = time.perf_counter()
    mats, scene, ptr, other, moved = setup(root, cfg, frac)
    t_scene = time.perf_counter() - t0
    acc, C, T_ = render(scene, ptr, mats, ref, cam, res, spp, seed, other, moved, instrument)
    o = pack(acc, res, spp)
    T_['scene_s'] = t_scene
    out = os.path.join(root, 'results', 'renders')
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f'{tag}.npz'), **o)
    json.dump({'cfg': cfg, 'ref': ref, 'seed': seed, 'spp': spp, 'res': res, 'instrumented': instrument, 'timing': T_, 'counters': C,
               'gpu_mem_mib': os.popen(f'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^{os.getpid()}," ').read().strip()},
              open(os.path.join(out, f'{tag}.json'), 'w'), indent=1)
    print(tag, json.dumps(T_), flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
