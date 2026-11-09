"""WL81 Stage F: limited rendering through Mitsuba 3.5.1's existing 'path' integrator (cuda_ad_rgb), materials as Python BSDF plugins.

  python render81.py validate <out> <states.npy>                                   # renderer validation (a)-(c)
  python render81.py render <data> <evalD_dir> <out> <arm> [<arm> ...] [--dev]     # images, metrics, rule

REF  ('lay81'): layered reference; eval = one random walk, sample = Sample_f weight; per-call random streams are seeded from a hash of
     the query (hit distance, uv, wo / sample bits) and a per-pass seed, so every call is an independent unbiased estimate.
CMP  ('cmp81'): compiled closure, closure81_dr.DrClosure81 (uniform compiled state).
Only point lights and Lambert / material surfaces: BSDF-sampled rays never hit an emitter, so MIS never needs the reference's missing
pdf; reported pdf = 1 for valid reference samples. Accumulation: own float64 accumulation of integrator.sample() (no cuda film).
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import layered81 as L  # noqa: E402
mi = L.init('cuda_ad_rgb')
import drjit as dr  # noqa: E402
import closure81_dr as CD  # noqa: E402
dr.set_flag(dr.JitFlag.KernelHistory, True)

P = json.load(open(os.path.join(HERE, 'protocol', 'compiled81.json')))
SF = P['stage_F']
SEED = P['seeds']['render']
REG = {}
PASS_SEED = {'v': 1}


def _bits(x):
    return dr.reinterpret_array_v(mi.UInt32, x)


def _hash_seq(a, b, c, d):
    h = _bits(a) * mi.UInt32(0x9E3779B1) ^ _bits(b) * mi.UInt32(0x85EBCA77) ^ _bits(c) * mi.UInt32(0xC2B2AE3D) ^ _bits(d) * mi.UInt32(0x27D4EB2F)
    h ^= h >> 15
    h *= mi.UInt32(0x2C1B3C6D)
    h ^= h >> 12
    return mi.UInt64(h) | (mi.UInt64(_bits(a) ^ _bits(d)) << 32)


class Cmp81(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        self.cl = CD.DrClosure81(REG[props['key']])
        self.m_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide
        self.m_components = [self.m_flags]

    def sample(self, ctx, si, s1, s2, active=True):
        wo, pdf, w, valid = self.cl.sample(si.wi, s1, s2, active)
        bs = dr.zeros(mi.BSDFSample3f)
        bs.wo = wo
        bs.pdf = pdf
        bs.eta = 1.0
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)
        bs.sampled_component = mi.UInt32(0)
        return bs, mi.Color3f(dr.select(valid, w, 0.0))

    def eval(self, ctx, si, wo, active=True):
        return mi.Color3f(self.cl.eval(si.wi, wo, active))

    def pdf(self, ctx, si, wo, active=True):
        return self.cl.pdf(si.wi, wo, active)

    def eval_pdf(self, ctx, si, wo, active=True):
        f, p = self.cl.eval_pdf(si.wi, wo, active)
        return mi.Color3f(f), p

    def to_string(self):
        return 'Cmp81'


class Lay81(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        d = REG[props['key']]
        v = []
        for x in d:
            f = mi.Float(float(x)); dr.make_opaque(f); v.append(f)
        self.m = L.Mat(v[0], v[1], v[2], mi.Color3f(v[3], v[4], v[5]), mi.Color3f(v[6], v[7], v[8]), v[9])
        self.m_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide
        self.m_components = [self.m_flags]

    def _rng(self, si, a, b):
        return mi.PCG32(size=dr.width(si.t), initstate=PASS_SEED['v'], initseq=_hash_seq(si.t, si.uv.x + a, si.uv.y + b, a * 1.7 + b))

    def sample(self, ctx, si, s1, s2, active=True):
        rng = self._rng(si, s2.x + s1, s2.y)
        wo, w, valid = L.layered_sample(self.m, si.wi, rng)
        valid &= mi.Bool(active)
        bs = dr.zeros(mi.BSDFSample3f)
        bs.wo = wo
        bs.pdf = dr.select(valid, 1.0, 0.0)
        bs.eta = 1.0
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)
        bs.sampled_component = mi.UInt32(0)
        return bs, mi.Color3f(dr.select(valid, w, 0.0))

    def eval(self, ctx, si, wo, active=True):
        rng = self._rng(si, wo.x, wo.y + wo.z)
        return mi.Color3f(dr.select(mi.Bool(active), L.layered_eval(self.m, si.wi, wo, rng), 0.0))

    def pdf(self, ctx, si, wo, active=True):
        return dr.select(mi.Bool(active) & (si.wi.z > 0) & (wo.z > 0), 1.0, 0.0)

    def eval_pdf(self, ctx, si, wo, active=True):
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

    def to_string(self):
        return 'Lay81'


mi.register_bsdf('cmp81', lambda props: Cmp81(props))
mi.register_bsdf('lay81', lambda props: Lay81(props))
T = mi.ScalarTransform4f
RES = 256


def scene_dict(mat, light, pose):
    rot = T.rotate([0, 0, 1], 30.0 if pose == 1 else 0.0)
    tr = T.translate([0.1, 0, 0]) if pose == 1 else T.translate([0, 0, 0])
    g = tr @ rot
    lights = {1: ([-1.5, -1.5, 2.5], 25.0), 2: ([2.0, -0.6, 0.45], 12.0)}
    pos, I = lights[light]
    white = {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': 0.8}}
    return {'type': 'scene',
            'floor': {'type': 'rectangle', 'to_world': T.scale([3, 3, 1]), 'bsdf': white},
            'wall': {'type': 'rectangle', 'to_world': T.translate([0, 1.5, 1.5]) @ T.rotate([1, 0, 0], 90) @ T.scale([3, 1.5, 1]), 'bsdf': white},
            'sphere': {'type': 'sphere', 'to_world': g @ T.translate([-0.4, 0, 0.35]) @ T.scale(0.35), 'bsdf': mat},
            'cube': {'type': 'cube', 'to_world': g @ T.translate([0.45, 0.1, 0.25]) @ T.rotate([0, 0, 1], 35) @ T.scale(0.25), 'bsdf': mat},
            'light': {'type': 'point', 'position': pos, 'intensity': {'type': 'rgb', 'value': I}}}


def camera_rays(seed, jitter=True, origin=(0.0, -3.0, 1.2), target=(0.0, 0.0, 0.3), fov=40.0):
    n = RES * RES
    o = np.array(origin); t = np.array(target)
    fwd = t - o; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rg = np.random.default_rng(seed)
    j = rg.random((n, 2)) if jitter else np.full((n, 2), 0.5)
    ii, jj = np.meshgrid(np.arange(RES), np.arange(RES), indexing='ij')
    x = (jj.ravel() + j[:, 0]) / RES * 2 - 1
    y = 1 - (ii.ravel() + j[:, 1]) / RES * 2
    s = math.tan(math.radians(fov) / 2)
    d = fwd[None] + s * x[:, None] * right[None] + s * y[:, None] * up[None]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    f = lambda a: mi.Float(np.ascontiguousarray(a, np.float32))
    return mi.Ray3f(mi.Point3f(f(np.full(n, o[0])), f(np.full(n, o[1])), f(np.full(n, o[2]))), mi.Vector3f(f(d[:, 0]), f(d[:, 1]), f(d[:, 2])))


def render(scene, spp, seed, max_depth=4, jitter=True):
    integ = mi.load_dict({'type': 'path', 'max_depth': max_depth, 'rr_depth': 64})
    acc = np.zeros((RES * RES, 3)); acc2 = np.zeros((RES * RES, 3))
    sampler = mi.load_dict({'type': 'independent'})
    gpu = 0.0
    ex = 0.0
    t0 = time.perf_counter()
    for k in range(spp):
        PASS_SEED['v'] = mi.UInt64(seed * 100003 + k + 1)
        dr.make_opaque(PASS_SEED['v'])
        sampler.seed(seed * 7919 + k, RES * RES)
        rays = camera_rays(seed * 104729 + k, jitter)
        dr.kernel_history()
        Lr, valid, _ = integ.sample(scene, sampler, rays, None, True)
        dr.sync_thread(); te = time.perf_counter()
        dr.eval(Lr)
        dr.sync_thread(); ex += time.perf_counter() - te
        x = np.stack([np.asarray(Lr[c], np.float64) for c in range(3)], 1)
        gpu += sum(float(h.get('execution_time', 0.0)) for h in dr.kernel_history())
        acc += x; acc2 += x * x
    wall = time.perf_counter() - t0
    m = acc / spp
    return m.reshape(RES, RES, 3), (acc2 / spp - m * m).reshape(RES, RES, 3), {'wall_s': wall, 'gpu_ms': gpu, 'exec_wall_s': ex, 'spp': spp}


def masks(scene):
    rays = camera_rays(0, False)
    si = scene.ray_intersect(rays)
    shp = scene.shapes()
    names = [s.id() for s in shp]
    obj = mi.Bool(False)
    for s, nme in zip(shp, names):
        if nme in ('sphere', 'cube'):
            obj |= dr.eq(si.shape, mi.ShapePtr(s))
    obj &= si.is_valid()
    ndv = dr.abs(dr.dot(si.n, rays.d))
    return np.asarray(obj).reshape(RES, RES), np.asarray(ndv).reshape(RES, RES)


def lum(x):
    return x @ np.array([0.2126, 0.7152, 0.0722])


def rel_l1(img, ref, mask):
    return float(np.abs(img - ref)[mask].sum() / max(ref[mask].sum(), 1e-30))


# ---------------------------------------------------------------- validation
def validate(out, states_path):
    os.makedirs(out, exist_ok=True)
    res = {}
    # (a) analytic Lambert plane under a point light, direct only, pixel centres
    I, rho = 1.0, 0.5
    sc = mi.load_dict({'type': 'scene', 'floor': {'type': 'rectangle', 'to_world': T.scale([5, 5, 1]), 'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': rho}}},
                       'light': {'type': 'point', 'position': [0.3, -0.2, 1.0], 'intensity': {'type': 'rgb', 'value': I}}})
    img, _, _ = render(sc, 4, 1, max_depth=2, jitter=False)
    rays = camera_rays(0, False)
    si = sc.ray_intersect(rays)
    p = np.stack([np.asarray(si.p[k]) for k in range(3)], 1).astype(np.float64)
    ok = np.asarray(si.is_valid())
    dvec = np.array([0.3, -0.2, 1.0])[None] - p
    dist2 = (dvec ** 2).sum(1)
    exact = rho / math.pi * I * np.maximum(dvec[:, 2], 0) / np.sqrt(dist2) / dist2
    rimg = img.reshape(-1, 3)[:, 0]
    rel = np.abs(rimg[ok] - exact[ok]) / exact[ok]
    res['a_lambert_point'] = {'mean_rel': float(np.mean(rimg[ok] - exact[ok]) / np.mean(exact[ok])), 'max_rel': float(rel.max()), 'pixels': int(ok.sum())}
    res['a_lambert_point']['PASS'] = abs(res['a_lambert_point']['mean_rel']) <= 1e-3 and res['a_lambert_point']['max_rel'] <= 1e-3
    # (b) closure plane, direct only, vs operator-level evaluation at 16 pixels
    st = np.load(states_path)[0]
    REG['val_cmp'] = st
    sc = mi.load_dict({'type': 'scene', 'floor': {'type': 'rectangle', 'to_world': T.scale([5, 5, 1]), 'bsdf': {'type': 'cmp81', 'key': 'val_cmp'}},
                       'light': {'type': 'point', 'position': [0.3, -0.2, 1.0], 'intensity': {'type': 'rgb', 'value': I}}})
    img, _, _ = render(sc, 1, 2, max_depth=2, jitter=False)
    pix = np.nonzero(ok)[0][np.linspace(0, ok.sum() - 1, 16).astype(int)]
    wo_w = dvec[pix] / np.sqrt(dist2[pix])[:, None]
    wi_w = -np.stack([np.asarray(rays.d[k]) for k in range(3)], 1)[pix].astype(np.float64)
    # rectangle frame: n = +z, s = +x, t = +y
    cl = CD.DrClosure81(st)
    V = lambda a: mi.Vector3f(*[mi.Float(np.ascontiguousarray(a[:, k], np.float32)) for k in range(3)])
    f = np.stack([np.asarray(cl.eval(V(wi_w), V(wo_w))[k]) for k in range(3)], 1) * I / dist2[pix][:, None]
    rr = np.abs(img.reshape(-1, 3)[pix] - f) / np.maximum(np.abs(f), 1e-12)
    res['b_closure_plane'] = {'max_rel': float(rr.max()), 'PASS': float(rr.max()) <= 1e-4}
    # (c) layered plane, direct only: pixel mean over passes vs operator-level mean over walks (z-test)
    desc = np.array([0.1, 1.5, 0.5, 0.9, 0.5, 0.1, 0.9, 0.6, 0.3, 0.25])
    REG['val_lay'] = desc
    sc = mi.load_dict({'type': 'scene', 'floor': {'type': 'rectangle', 'to_world': T.scale([5, 5, 1]), 'bsdf': {'type': 'lay81', 'key': 'val_lay'}},
                       'light': {'type': 'point', 'position': [0.3, -0.2, 1.0], 'intensity': {'type': 'rgb', 'value': I}}})
    img, var, _ = render(sc, 256, 3, max_depth=2, jitter=False)
    n16 = len(pix)
    w = 1 << 16
    m = L.Mat(*[mi.Float(np.full(n16 * w, x, np.float32)) for x in desc[:3]], mi.Color3f(*[mi.Float(np.full(n16 * w, x, np.float32)) for x in desc[3:6]]),
              mi.Color3f(*[mi.Float(np.full(n16 * w, x, np.float32)) for x in desc[6:9]]), mi.Float(np.full(n16 * w, desc[9], np.float32)))
    rng = mi.PCG32(size=n16 * w, initstate=SEED + 3, initseq=dr.arange(mi.UInt64, n16 * w))
    e = L.layered_eval(m, V(np.repeat(wi_w, w, 0)), V(np.repeat(wo_w, w, 0)), rng)
    e = np.stack([np.asarray(e[k]) for k in range(3)], 1).astype(np.float64).reshape(n16, w, 3) * (I / dist2[pix])[:, None, None]
    em, es = e.mean(1), e.std(1) / math.sqrt(w)
    im = img.reshape(-1, 3)[pix]; ise = np.sqrt(var.reshape(-1, 3)[pix] / 256)
    z = (im - em) / np.sqrt(es ** 2 + ise ** 2 + 1e-30)
    res['c_layered_plane'] = {'max_abs_z': float(np.abs(z).max()), 'max_rel': float((np.abs(im - em) / np.maximum(em, 1e-12)).max()),
                              'PASS': float(np.abs(z).max()) <= 4.5}
    res['PASS'] = bool(res['a_lambert_point']['PASS'] and res['b_closure_plane']['PASS'] and res['c_layered_plane']['PASS'])
    json.dump(res, open(os.path.join(out, 'validation.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))
    return res['PASS']


# ---------------------------------------------------------------- images
def run_images(data, evalD, out, arms, dev):
    os.makedirs(out, exist_ok=True)
    mats = np.load(os.path.join(data, 'materials.npz'))['test_iid'][:3]
    s0 = P['fitting']['compiler']['seeds'][0]
    states = {a: np.load(os.path.join(evalD, f'states_{a}_s{s0}_test_iid.npy'))[:3] for a in arms}
    spp_t, spp_hq = (8, 64) if dev else (SF['spp']['test'], SF['spp']['reference_hq'])
    rows = []
    for k in range(3):
        REG[f'lay_{k}'] = mats[k]
        for a in arms:
            REG[f'{a}_{k}'] = states[a][k]
        for light in [1, 2]:
            for pose in [0, 1]:
                tag = f'm{k}_L{light}_P{pose}'
                sref = mi.load_dict(scene_dict({'type': 'lay81', 'key': f'lay_{k}'}, light, pose))
                obj, ndv = masks(sref)
                hA, _, tA = render(sref, spp_hq // 2, SEED + 11 * k + 3 * light + pose)
                hB, _, tB = render(sref, spp_hq // 2, SEED + 50000 + 11 * k + 3 * light + pose)
                hq = 0.5 * (hA + hB)
                lr = lum(hq)
                hl = obj & (lr >= 3 * np.median(lr[obj]))
                gz = obj & (ndv <= 0.25)
                ref_unc = 0.5 * rel_l1(hA, hB, obj)
                r_ref, _, t_ref = render(sref, spp_t, SEED + 90000 + 11 * k + 3 * light + pose)
                passes_r = [render(sref, spp_t, SEED + 200000 + 1000 * q + 11 * k + 3 * light + pose)[0] for q in range(2 if dev else 8)]
                row = {'tag': tag, 'ref_hq_uncertainty': ref_unc, 'ref_hq_time': {'A': tA, 'B': tB}, 'ref_spp': t_ref,
                       'REF_err': rel_l1(r_ref, hq, obj), 'REF_hl_err': rel_l1(r_ref, hq, hl), 'REF_gz_err': rel_l1(r_ref, hq, gz),
                       'REF_pixvar': float(np.mean(np.var(np.stack(passes_r), 0)[obj])), 'n_obj': int(obj.sum()), 'n_hl': int(hl.sum()), 'n_gz': int(gz.sum())}
                np.save(os.path.join(out, f'{tag}_refhq.npy'), hq.astype(np.float32))
                np.save(os.path.join(out, f'{tag}_ref{spp_t}.npy'), r_ref.astype(np.float32))
                for a in arms:
                    sc = mi.load_dict(scene_dict({'type': 'cmp81', 'key': f'{a}_{k}'}, light, pose))
                    img, _, tc = render(sc, spp_t, SEED + 90000 + 11 * k + 3 * light + pose)
                    passes_c = [render(sc, spp_t, SEED + 200000 + 1000 * q + 11 * k + 3 * light + pose)[0] for q in range(2 if dev else 8)]
                    spp_match = max(1, int(math.floor(spp_t * tc['gpu_ms'] / max(t_ref['gpu_ms'], 1e-9))))
                    r_m, _, t_m = render(sref, spp_match, SEED + 300000 + 11 * k + 3 * light + pose)
                    e_c = rel_l1(img, hq, obj)
                    row[a] = {'err': e_c, 'hl_err': rel_l1(img, hq, hl), 'gz_err': rel_l1(img, hq, gz), 'time': tc,
                              'pixvar': float(np.mean(np.var(np.stack(passes_c), 0)[obj])),
                              'time_matched_ref_spp': spp_match, 'time_matched_ref_err': rel_l1(r_m, hq, obj), 'time_matched_ref_time': t_m,
                              'ref_limited': ref_unc > 0.25 * e_c}
                    row[a]['pass'] = bool(e_c <= row[a]['time_matched_ref_err'] and row[a]['hl_err'] <= 0.10 and not row[a]['ref_limited'])
                    np.save(os.path.join(out, f'{tag}_{a}{spp_t}.npy'), img.astype(np.float32))
                rows.append(row)
                print(json.dumps(row, default=str), flush=True)
    res = {'rows': rows, 'arms': arms, 'spp_test': spp_t, 'spp_hq': spp_hq,
           'RENDERING_rule_per_arm': {a: bool(all(r[a]['pass'] for r in rows)) for a in arms}}
    json.dump(res, open(os.path.join(out, 'stageF.json'), 'w'), indent=1, default=str)
    print(json.dumps(res['RENDERING_rule_per_arm']))


if __name__ == '__main__':
    if sys.argv[1] == 'validate':
        ok = validate(sys.argv[2], sys.argv[3])
    else:
        run_images(sys.argv[2], sys.argv[3], sys.argv[4], [a for a in sys.argv[5:] if not a.startswith('--')], '--dev' in sys.argv)
    os._exit(0)
