"""WL83 Stage E: focused rendering through Mitsuba 3.5.1's 'path' integrator (cuda_ad_rgb) with own float64 accumulation (WL81 render81
machinery, imported: scene, camera rays, render loop, masks, the layered-reference plugin 'lay81').

  python render83.py validate <out>                                         # (a) Lambert analytic, (b) E / ER closure plane vs operator
  python render83.py render <data83> <w81> <fitdir> <w82fit> <out> [--dev] # images and metrics

Materials (predeclared): test83_iid #0, #1; test83_hard #0, #1; WL82 parameterised finish gold_clearcoat_polished (not a measured BRDF).
Lights: L1 key (render81 light 1), L2 grazing (render81 light 2); pose 0. path max_depth 4 (floor / wall bounce = limited indirect).
Arms: REF_HQ (layered, 2 x 2048 spp halves), REF at 64 spp and time-matched spp, C0_DF, E_DF, ER_DF, E_L, ER_L (compiler seed 8301) at 64 spp.
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled82'))
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled81'))
import render81 as R81  # noqa: E402  (initialises Mitsuba cuda_ad_rgb, registers lay81 / cmp81)
mi, dr = R81.mi, R81.dr
import closure83_dr as CD  # noqa: E402

P83 = json.load(open(os.path.join(HERE, 'protocol', 'compiled83.json')))
DFS = int(os.environ.get('W83_DF_STEPS', '12000'))   # dev smoke only overrides
RP = P83['render']
SEED = P83['seeds']['render']
REG = R81.REG


class Cmp83(mi.BSDF):
    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        arm, st = REG[props['key']]
        self.cl = CD.DrClosure83(arm, st)
        self.m_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide
        self.m_components = [self.m_flags]

    def sample(self, ctx, si, s1, s2, active=True):
        wo, pdf, w, valid = self.cl.sample(si.wi, s1, s2, active)
        bs = dr.zeros(mi.BSDFSample3f)
        bs.wo = wo; bs.pdf = pdf; bs.eta = 1.0
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection); bs.sampled_component = mi.UInt32(0)
        return bs, mi.Color3f(dr.select(valid, w, 0.0))

    def eval(self, ctx, si, wo, active=True):
        return mi.Color3f(self.cl.eval(si.wi, wo, active))

    def pdf(self, ctx, si, wo, active=True):
        return self.cl.pdf(si.wi, wo, active)

    def eval_pdf(self, ctx, si, wo, active=True):
        f, p = self.cl.eval_pdf(si.wi, wo, active)
        return mi.Color3f(f), p

    def to_string(self):
        return 'Cmp83'


mi.register_bsdf('cmp83', lambda props: Cmp83(props))
T = R81.T
RES = R81.RES


def camera_rays_b(seed, b, jitter=True, origin=(0.0, -3.0, 1.2), target=(0.0, 0.0, 0.3), fov=40.0):
    """render81.camera_rays for b samples per pixel in one wavefront (lane = sample * RES^2 + pixel)."""
    n = RES * RES
    o = np.array(origin); t = np.array(target)
    fwd = t - o; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rg = np.random.default_rng(seed)
    j = rg.random((n * b, 2)) if jitter else np.full((n * b, 2), 0.5)
    ii, jj = np.meshgrid(np.arange(RES), np.arange(RES), indexing='ij')
    ii, jj = np.tile(ii.ravel(), b), np.tile(jj.ravel(), b)
    x = (jj + j[:, 0]) / RES * 2 - 1
    y = 1 - (ii + j[:, 1]) / RES * 2
    s = math.tan(math.radians(fov) / 2)
    d = fwd[None] + s * x[:, None] * right[None] + s * y[:, None] * up[None]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    f = lambda a: mi.Float(np.ascontiguousarray(a, np.float32))
    N = n * b
    return mi.Ray3f(mi.Point3f(f(np.full(N, o[0])), f(np.full(N, o[1])), f(np.full(N, o[2]))), mi.Vector3f(f(d[:, 0]), f(d[:, 1]), f(d[:, 2])))


def render_b(scene, spp, seed, max_depth=4, jitter=True):
    """render81.render with RP['spp_per_launch'] samples per pixel traced per launch (GPU-bound instead of per-spp Python tracing)."""
    B = RP['spp_per_launch']
    integ = mi.load_dict({'type': 'path', 'max_depth': max_depth, 'rr_depth': 64})
    n = RES * RES
    acc = np.zeros((n, 3)); acc2 = np.zeros((n, 3))
    sampler = mi.load_dict({'type': 'independent'})
    gpu = 0.0
    t0 = time.perf_counter()
    done, k = 0, 0
    while done < spp:
        b = min(B, spp - done)
        R81.PASS_SEED['v'] = mi.UInt64(seed * 100003 + k + 1)
        dr.make_opaque(R81.PASS_SEED['v'])
        sampler.seed(seed * 7919 + k, n * b)
        rays = camera_rays_b(seed * 104729 + k, b, jitter)
        dr.kernel_history()
        Lr, valid, _ = integ.sample(scene, sampler, rays, None, True)
        dr.eval(Lr); dr.sync_thread()
        x = np.stack([np.asarray(Lr[c], np.float64) for c in range(3)], 1).reshape(b, n, 3)
        gpu += sum(float(h.get('execution_time', 0.0)) for h in dr.kernel_history())
        acc += x.sum(0); acc2 += (x * x).sum(0)
        done += b; k += 1
    m = acc / spp
    return m.reshape(RES, RES, 3), (acc2 / spp - m * m).reshape(RES, RES, 3), {'wall_s': time.perf_counter() - t0, 'gpu_ms': gpu, 'spp': spp, 'launches': k}


def validate(out):
    """(a) Lambert plane under a point light vs analytic (WL81 render81 validation (a)); (b) E / ER closure plane: rendered direct radiance
    at 16 pixels vs operator-level closure83 evaluation (rel <= 1e-4)."""
    os.makedirs(out, exist_ok=True)
    import torch
    import closure82 as CL
    import closure83  # noqa: F401
    res = {}
    I, rho = 1.0, 0.5
    sc = mi.load_dict({'type': 'scene', 'floor': {'type': 'rectangle', 'to_world': T.scale([5, 5, 1]), 'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': rho}}},
                       'light': {'type': 'point', 'position': [0.3, -0.2, 1.0], 'intensity': {'type': 'rgb', 'value': I}}})
    img, _, _ = R81.render(sc, 4, 1, max_depth=2, jitter=False)
    rays = R81.camera_rays(0, False)
    si = sc.ray_intersect(rays)
    p = np.stack([np.asarray(si.p[k]) for k in range(3)], 1).astype(np.float64)
    ok = np.asarray(si.is_valid())
    dvec = np.array([0.3, -0.2, 1.0])[None] - p
    dist2 = (dvec ** 2).sum(1)
    exact = rho / math.pi * I * np.maximum(dvec[:, 2], 0) / np.sqrt(dist2) / dist2
    rimg = img.reshape(-1, 3)[:, 0]
    rel = np.abs(rimg[ok] - exact[ok]) / exact[ok]
    res['a_lambert_point'] = {'mean_rel': float(np.mean(rimg[ok] - exact[ok]) / np.mean(exact[ok])), 'max_rel': float(rel.max())}
    res['a_lambert_point']['PASS'] = abs(res['a_lambert_point']['mean_rel']) <= 1e-3 and res['a_lambert_point']['max_rel'] <= 1e-3
    pix = np.nonzero(ok)[0][np.linspace(0, ok.sum() - 1, 16).astype(int)]
    wo_w = dvec[pix] / np.sqrt(dist2[pix])[:, None]
    wi_w = -np.stack([np.asarray(rays.d[k]) for k in range(3)], 1)[pix].astype(np.float64)
    g = torch.Generator().manual_seed(SEED)
    for arm in ['E', 'ER']:
        z = torch.randn(1, CL.N_RAW, generator=g) * 1.5
        st = CD.compiled_state(arm, CL.decode(arm, 'cert', z))[0]
        REG['val_' + arm] = (arm, st)
        sc = mi.load_dict({'type': 'scene', 'floor': {'type': 'rectangle', 'to_world': T.scale([5, 5, 1]), 'bsdf': {'type': 'cmp83', 'key': 'val_' + arm}},
                           'light': {'type': 'point', 'position': [0.3, -0.2, 1.0], 'intensity': {'type': 'rgb', 'value': I}}})
        img, _, _ = R81.render(sc, 1, 2, max_depth=2, jitter=False)
        V = lambda a: mi.Vector3f(*[mi.Float(np.ascontiguousarray(a[:, k], np.float32)) for k in range(3)])
        f = np.stack([np.asarray(CD.DrClosure83(arm, st).eval(V(wi_w), V(wo_w))[k]) for k in range(3)], 1) * I / dist2[pix][:, None]
        rr = np.abs(img.reshape(-1, 3)[pix] - f) / np.maximum(np.abs(f), 1e-12)
        res[f'b_closure_plane_{arm}'] = {'max_rel': float(rr.max()), 'PASS': float(rr.max()) <= 1e-4}
    res['PASS'] = bool(all(v['PASS'] for v in res.values() if isinstance(v, dict)))
    json.dump(res, open(os.path.join(out, 'validation.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


def materials(data, w81, fitdir, w82fit):
    import torch
    import closure82 as CL
    import closure83  # noqa: F401
    import data83 as D83
    import data82 as D82
    mats = []
    for s, ks in [('test83_iid', [0, 1]), ('test83_hard', [0, 1])]:
        d = D83.descs(data, w81, s)
        for k in ks:
            mats.append((f'{s}_{k}', d[k], s, k))
    names = [m['name'] for m in P82['real_materials']['list']]
    gi = names.index('gold_clearcoat_polished')
    mats.append(('real_gold_clearcoat_polished', D82.real_desc()[gi], None, gi))
    st = {}
    s0 = P83['compiler']['seeds'][0]
    for tag, desc, s, k in mats:
        for arm in ['C0', 'E', 'ER']:
            if s is not None:
                z = np.load(os.path.join(fitdir, f'fit_{s}_{arm}_cert_metric_{DFS}_generic.npz'))['z'][k:k + 1]
            else:
                z = np.load(os.path.join(w82fit, f'fit_real_{arm}_cert_metric_12000_generic.npz'))['z'][k:k + 1]   # stored WL82 DF
            st[(tag, f'{arm}_DF')] = (arm, CD.compiled_state(arm, CL.decode(arm, 'cert', torch.from_numpy(z).cuda()))[0])
        for arm in ['E', 'ER']:
            zl = np.load(os.path.join(fitdir, f'fit_{s or "real83"}_L_{arm}_s{s0}.npz'))['z'][k:k + 1]
            st[(tag, f'{arm}_L')] = (arm, CD.compiled_state(arm, CL.decode(arm, 'cert', torch.from_numpy(zl).cuda()))[0])
    return mats, st


P82 = json.load(open(os.path.join(HERE, '..', 'fnto_compiled82', 'protocol', 'compiled82.json')))


def run(data, w81, fitdir, w82fit, out, dev):
    os.makedirs(out, exist_ok=True)
    mats, st = materials(data, w81, fitdir, w82fit)
    arms = RP['arms']
    spp_t, spp_hq = (8, 64) if dev else (RP['spp_test'], RP['spp_hq'])
    npass = 2 if dev else RP['variance_passes']
    rows = []
    for mi_, (tag, desc, _, _) in enumerate(mats[: 2 if dev else len(mats)]):
        R81.REG[f'lay_{tag}'] = desc
        for a in arms:
            REG[f'{a}_{tag}'] = st[(tag, a)]
        for light in [1, 2]:
            if PART and (2 * mi_ + light - 1) % PART[1] != PART[0]:
                continue                                   # amendment 2: image-parallel rendering (one image list slice per process)
            key = f'{tag}_L{light}'
            seed = SEED + 101 * mi_ + 7 * light
            sref = mi.load_dict(R81.scene_dict({'type': 'lay81', 'key': f'lay_{tag}'}, light, 0))
            obj, ndv = R81.masks(sref)
            hA, _, tA = render_b(sref, spp_hq // 2, seed)
            hB, _, tB = render_b(sref, spp_hq // 2, seed + 50000)
            hq = 0.5 * (hA + hB)
            lr = R81.lum(hq)
            hl = obj & (lr >= 3 * np.median(lr[obj]))
            gz = obj & (ndv <= 0.25)
            ref_unc = 0.5 * R81.rel_l1(hA, hB, obj)
            r_ref, _, t_ref = render_b(sref, spp_t, seed + 90000)
            pr = [render_b(sref, spp_t, seed + 200000 + 1000 * q)[0] for q in range(npass)]
            row = {'image': key, 'material': tag, 'light': light, 'ref_hq_uncertainty': ref_unc, 'ref_hq_time': {'A': tA, 'B': tB}, 'ref_64': t_ref,
                   'REF_err': R81.rel_l1(r_ref, hq, obj), 'REF_hl_err': R81.rel_l1(r_ref, hq, hl), 'REF_gz_err': R81.rel_l1(r_ref, hq, gz),
                   'REF_pixvar': float(np.mean(np.var(np.stack(pr), 0)[obj])), 'n_obj': int(obj.sum()), 'n_hl': int(hl.sum()), 'n_gz': int(gz.sum())}
            np.save(os.path.join(out, f'{key}_refhq.npy'), hq.astype(np.float32))
            np.save(os.path.join(out, f'{key}_ref{spp_t}.npy'), r_ref.astype(np.float32))
            for a in arms:
                sc = mi.load_dict(R81.scene_dict({'type': 'cmp83', 'key': f'{a}_{tag}'}, light, 0))
                img, _, tc = render_b(sc, spp_t, seed + 90000)
                pc = [render_b(sc, spp_t, seed + 200000 + 1000 * q)[0] for q in range(npass)]
                spp_m = max(1, int(math.floor(spp_t * tc['gpu_ms'] / max(t_ref['gpu_ms'], 1e-9))))
                r_m, _, t_m = render_b(sref, spp_m, seed + 300000)
                e = R81.rel_l1(img, hq, obj)
                row[a] = {'err': e, 'hl_err': R81.rel_l1(img, hq, hl), 'gz_err': R81.rel_l1(img, hq, gz), 'time': tc,
                          'pixvar': float(np.mean(np.var(np.stack(pc), 0)[obj])), 'time_matched_ref_spp': spp_m,
                          'time_matched_ref_err': R81.rel_l1(r_m, hq, obj), 'time_matched_ref_time': t_m, 'ref_limited': ref_unc > 0.25 * e}
                row[a]['pass'] = bool(e <= row[a]['time_matched_ref_err'] and row[a]['hl_err'] <= RP['hl_err_max'] and not row[a]['ref_limited'])
                np.save(os.path.join(out, f'{key}_{a}{spp_t}.npy'), img.astype(np.float32))
            rows.append(row)
            print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ('err', 'hl_err', 'time_matched_ref_err', 'pass')})
                              for k, v in row.items() if k not in ('ref_hq_time', 'ref_64')}, default=str), flush=True)
    res = {'rows': rows, 'arms': arms, 'spp_test': spp_t, 'spp_hq': spp_hq,
           'RENDER_rule_per_arm': {a: bool(all(r[a]['pass'] for r in rows)) for a in arms},
           'pass_count_per_arm': {a: int(sum(r[a]['pass'] for r in rows)) for a in arms}, 'images': len(rows)}
    fn = f'render83_part{PART[0]}of{PART[1]}.json' if PART else 'render83.json'
    json.dump(res, open(os.path.join(out, fn), 'w'), indent=1, default=str)
    print(json.dumps(res['RENDER_rule_per_arm']))


def merge(out, K):
    parts = [json.load(open(os.path.join(out, f'render83_part{i}of{K}.json'))) for i in range(K)]
    rows = sorted([r for p in parts for r in p['rows']], key=lambda r: (r['material'], r['light']))
    arms = parts[0]['arms']
    res = {'rows': rows, 'arms': arms, 'spp_test': parts[0]['spp_test'], 'spp_hq': parts[0]['spp_hq'],
           'RENDER_rule_per_arm': {a: bool(all(r[a]['pass'] for r in rows)) for a in arms},
           'pass_count_per_arm': {a: int(sum(r[a]['pass'] for r in rows)) for a in arms}, 'images': len(rows),
           'concurrency': f'amendment 2: {K} concurrent render processes (image slices); kernel GPU times may include overlap with the other processes'}
    json.dump(res, open(os.path.join(out, 'render83.json'), 'w'), indent=1, default=str)
    print(json.dumps(res['RENDER_rule_per_arm']), len(rows))


PART = None
for _a in sys.argv:
    if _a.startswith('--part='):
        _i, _k = _a.split('=')[1].split('/'); PART = (int(_i), int(_k))


if __name__ == '__main__':
    if sys.argv[1] == 'validate':
        validate(sys.argv[2])
    elif sys.argv[1] == 'merge':
        merge(sys.argv[2], int(sys.argv[3]))
    else:
        run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], '--dev' in sys.argv)
    os._exit(0)
