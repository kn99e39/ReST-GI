"""WL84 rendering qualification (supplement to WL83; WL83 code imported unchanged: render83 -> render81 scene / masks / metrics, closure83_dr).

  python render84.py audit <out>                       # A: WL83 artefact check + bit-identical reproduction of the stored 64-spp renders
  python render84.py refgen <image> <out>              # B: independent layered reference, two halves of SPP_REF / 2, new seeds
  python render84.py econ <out>                        # D: uncontended timing (trace / GPU / wall) + time-to-quality curves (run alone)
  python render84.py replicates <image> <out>          # C: candidate Monte Carlo variability (new seeds, unchanged arms / conditions)
  python render84.py evaluate <out>                    # C: WL81 Stage F rule against the new reference
Images (predeclared): test83_iid_1_L2 (the WL83 ER_L failure, reference-limited), test83_iid_0_L2 (passing control, same grazing light),
test83_hard_1_L1 (control: ER_L passes, E_L genuine highlight failure in WL83).
"""
import json, math, os, sys, time, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled83'))
import render83 as RR  # noqa: E402  (registers cmp83 / lay81, imports render81)
R81, mi, dr = RR.R81, RR.mi, RR.dr

P84 = json.load(open(os.path.join(HERE, 'protocol', 'compiled84.json')))
if os.environ.get('W84_REF_DIR'):
    P84['paths']['ref_dir'] = os.environ['W84_REF_DIR']
if os.environ.get('W84_DEV'):                       # dev probe only: reduced budgets
    P84['reference']['chunks_per_half'] = [256, 256, 512]
    P84['replicates']['n'] = 2
    P84['economics']['reps'] = 1
    P84['economics']['ladder_ref'] = [64, 256]
    P84['economics']['ladder_closure'] = [16, 64]
P83 = RR.P83
W83 = os.path.expanduser('~/rest_compiled83/results')
W81 = os.path.expanduser('~/rest_compiled81/results/data')
W82F = os.path.expanduser('~/rest_compiled82/results/fit')
IMAGES = P84['images']
ARMS = P83['render']['arms']
SEED83 = P83['seeds']['render']
MAT_ORDER = ['test83_iid_0', 'test83_iid_1', 'test83_hard_0', 'test83_hard_1', 'real_gold_clearcoat_polished']
_CACHE = {}


def setup():
    if not _CACHE:
        mats, st = RR.materials(os.path.join(W83, 'data'), W81, os.path.join(W83, 'fit'), W82F)
        _CACHE['mats'] = {m[0]: m for m in mats}
        _CACHE['st'] = st
    return _CACHE['mats'], _CACHE['st']


def image_spec(image):
    tag, light = image.rsplit('_L', 1)
    light = int(light)
    mi_ = MAT_ORDER.index(tag)
    return tag, light, SEED83 + 101 * mi_ + 7 * light


def scenes(image):
    mats, st = setup()
    tag, light, seed = image_spec(image)
    R81.REG[f'lay_{tag}'] = mats[tag][1]
    for a in ARMS:
        RR.REG[f'{a}_{tag}'] = st[(tag, a)]
    sref = mi.load_dict(R81.scene_dict({'type': 'lay81', 'key': f'lay_{tag}'}, light, 0))
    sc = {a: mi.load_dict(R81.scene_dict({'type': 'cmp83', 'key': f'{a}_{tag}'}, light, 0)) for a in ARMS}
    return tag, light, seed, sref, sc


def _cam():
    o = np.array((0.0, -3.0, 1.2)); t = np.array((0.0, 0.0, 0.3))
    fwd = t - o; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return o, fwd, right, up, math.tan(math.radians(40.0) / 2)


def rays_gpu(seed, b):
    """Same camera and per-sample uniform pixel jitter as render81 / render83 camera rays, generated on the GPU (Dr.Jit PCG32); sample-major
    lanes (lane = sample * RES^2 + pixel, as the WL83 numpy path). Different random stream than the numpy path. Returns rays, pixel index."""
    o, fwd, right, up, s = _cam()
    n = RR.RES * RR.RES
    N = n * b
    lane = dr.arange(mi.UInt32, N)
    pix = lane % n
    ii, jj = pix // RR.RES, pix % RR.RES
    rg = mi.PCG32(size=N, initstate=mi.UInt64(seed), initseq=mi.UInt64(0))
    jx, jy = rg.next_float32(), rg.next_float32()
    x = (mi.Float(jj) + jx) * (2.0 / RR.RES) - 1.0
    y = 1.0 - (mi.Float(ii) + jy) * (2.0 / RR.RES)
    d = dr.normalize(mi.Vector3f(*[fwd[k] + s * x * right[k] + s * y * up[k] for k in range(3)]))
    return mi.Ray3f(mi.Point3f(*[mi.Float(float(o[k])) for k in range(3)]), d), pix


def render_g(scene, spp, seed, B, max_depth=4):
    """GPU-resident variant of render_t for NEW renders (amendment-free WL84 instrumentation): GPU ray generation + GPU per-pixel
    accumulation (float32 scatter-add per launch, float64 across launches); identical camera, jitter distribution, sampler, integrator and BSDF plugins."""
    integ = mi.load_dict({'type': 'path', 'max_depth': max_depth, 'rr_depth': 64})
    n = RR.RES * RR.RES
    acc = np.zeros((n, 3)); acc2 = np.zeros((n, 3))
    sampler = mi.load_dict({'type': 'independent'})
    T = {'rays_s': 0.0, 'trace_s': 0.0, 'eval_s': 0.0, 'readback_s': 0.0, 'gpu_ms': 0.0, 'launches': 0}
    t_all = time.perf_counter()
    done, k = 0, 0
    while done < spp:
        b = min(B, spp - done)
        R81.PASS_SEED['v'] = mi.UInt64(seed * 100003 + k + 1)
        dr.make_opaque(R81.PASS_SEED['v'])
        sampler.seed(seed * 7919 + k, n * b)
        t0 = time.perf_counter()
        dr.kernel_history()
        rays, pix = rays_gpu(seed * 104729 + k, b)
        t1 = time.perf_counter()
        Lr, valid, _ = integ.sample(scene, sampler, rays, None, True)
        dr.eval(Lr)   # amendment 1: scatter-add of the UNEVALUATED integrator output summed every sample twice (Dr.Jit 0.4.6; run 1 invalid)
        # per-launch per-pixel sums on the GPU (float32 scatter-add, sample-major -> low contention); float64 accumulation across launches on
        # the host (pre-freeze probe: block_sum in float64 with pixel-major lanes cost 5x the path-tracing GPU time)
        s1 = [dr.zeros(mi.Float, n) for _ in range(3)]; s2 = [dr.zeros(mi.Float, n) for _ in range(3)]
        for c in range(3):
            dr.scatter_reduce(dr.ReduceOp.Add, s1[c], Lr[c], pix)
            dr.scatter_reduce(dr.ReduceOp.Add, s2[c], Lr[c] * Lr[c], pix)
        t2 = time.perf_counter()
        dr.eval(s1, s2); dr.sync_thread()
        t3 = time.perf_counter()
        T['gpu_ms'] += sum(float(h.get('execution_time', 0.0)) for h in dr.kernel_history())
        acc += np.stack([np.asarray(v, np.float64) for v in s1], 1); acc2 += np.stack([np.asarray(v, np.float64) for v in s2], 1)
        t4 = time.perf_counter()
        T['rays_s'] += t1 - t0; T['trace_s'] += t2 - t1; T['eval_s'] += t3 - t2; T['readback_s'] += t4 - t3; T['launches'] += 1
        done += b; k += 1
    T['wall_s'] = time.perf_counter() - t_all; T['spp'] = spp
    m = acc / spp
    return m.reshape(RR.RES, RR.RES, 3), acc, acc2, T


def render_t(scene, spp, seed, B, max_depth=4):
    """render83.render_b with per-phase timing: ray generation (numpy), trace (integrator.sample call), GPU execution (dr.eval wall and
    kernel-history GPU time), host readback / accumulation. Used for the bit-identical reproduction of WL83 renders."""
    integ = mi.load_dict({'type': 'path', 'max_depth': max_depth, 'rr_depth': 64})
    n = RR.RES * RR.RES
    acc = np.zeros((n, 3)); acc2 = np.zeros((n, 3))
    sampler = mi.load_dict({'type': 'independent'})
    T = {'rays_s': 0.0, 'trace_s': 0.0, 'eval_s': 0.0, 'readback_s': 0.0, 'gpu_ms': 0.0, 'launches': 0}
    t_all = time.perf_counter()
    done, k = 0, 0
    while done < spp:
        b = min(B, spp - done)
        R81.PASS_SEED['v'] = mi.UInt64(seed * 100003 + k + 1)
        dr.make_opaque(R81.PASS_SEED['v'])
        sampler.seed(seed * 7919 + k, n * b)
        t0 = time.perf_counter()
        rays = RR.camera_rays_b(seed * 104729 + k, b, True)
        dr.eval(rays); dr.sync_thread()
        t1 = time.perf_counter()
        dr.kernel_history()
        Lr, valid, _ = integ.sample(scene, sampler, rays, None, True)
        t2 = time.perf_counter()
        dr.eval(Lr); dr.sync_thread()
        t3 = time.perf_counter()
        T['gpu_ms'] += sum(float(h.get('execution_time', 0.0)) for h in dr.kernel_history())
        x = np.stack([np.asarray(Lr[c], np.float64) for c in range(3)], 1).reshape(b, n, 3)
        acc += x.sum(0); acc2 += (x * x).sum(0)
        t4 = time.perf_counter()
        T['rays_s'] += t1 - t0; T['trace_s'] += t2 - t1; T['eval_s'] += t3 - t2; T['readback_s'] += t4 - t3; T['launches'] += 1
        done += b; k += 1
    T['wall_s'] = time.perf_counter() - t_all; T['spp'] = spp
    m = acc / spp
    return m.reshape(RR.RES, RR.RES, 3), acc, acc2, T


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, np.float32).tobytes()).hexdigest()


# ---------------------------------------------------------------- A: audit and reproduction
def audit(out):
    os.makedirs(out, exist_ok=True)
    rj = json.load(open(os.path.join(W83, 'render', 'render83.json')))
    rows = {r['image']: r for r in rj['rows']}
    res = {'wl83_part_files': rj['part_files'], 'images': {}}
    for image in IMAGES:
        tag, light, seed, sref, sc = scenes(image)
        hq = np.load(os.path.join(W83, 'render', f'{image}_refhq.npy'))
        obj, ndv = R81.masks(sref)
        lr = R81.lum(hq); hl = obj & (lr >= 3 * np.median(lr[obj]))
        r = {'seed': seed, 'n_obj': int(obj.sum()), 'n_hl': int(hl.sum()), 'n_obj_wl83': rows[image]['n_obj'], 'n_hl_wl83': rows[image]['n_hl'], 'arms': {}}
        # recompute WL83 errors from the stored arrays; reproduce each stored 64-spp render with its WL83 seed
        for a in ARMS + ['REF']:
            fn = os.path.join(W83, 'render', f'{image}_{a}64.npy' if a != 'REF' else f'{image}_ref64.npy')
            stored = np.load(fn)
            img, _, _, _ = render_t(sref if a == 'REF' else sc[a], 64, seed + 90000, P84['launch_spp'])
            d = {'stored_sha256': sha(stored), 'reproduced_bit_identical': bool(np.array_equal(img.astype(np.float32), stored)),
                 'max_abs_diff': float(np.abs(img.astype(np.float32) - stored).max()), 'err_recomputed': R81.rel_l1(stored.astype(np.float64), hq, obj),
                 'hl_err_recomputed': R81.rel_l1(stored.astype(np.float64), hq, hl)}
            if a != 'REF':
                d['err_wl83'] = rows[image][a]['err']; d['hl_err_wl83'] = rows[image][a]['hl_err']
            else:
                d['err_wl83'] = rows[image]['REF_err']
            r['arms'][a] = d
        r['ref_hq_uncertainty_wl83'] = rows[image]['ref_hq_uncertainty']
        res['images'][image] = r
        print(image, json.dumps({a: (v['reproduced_bit_identical'], round(v['err_recomputed'] - v['err_wl83'], 12)) for a, v in r['arms'].items()}), flush=True)
    st = setup()[1]
    res['state_sha256'] = {f'{t}|{a}': hashlib.sha256(np.asarray(v[1], np.float32).tobytes()).hexdigest() for (t, a), v in st.items() if t in [image_spec(i)[0] for i in IMAGES]}
    res['REPRODUCED'] = all(v['reproduced_bit_identical'] and abs(v['err_recomputed'] - v['err_wl83']) <= 1e-6   # WL83 computed err from float64 images, stored float32
                            for r in res['images'].values() for v in r['arms'].values())
    json.dump(res, open(os.path.join(out, 'audit84.json'), 'w'), indent=1)
    print('REPRODUCED' if res['REPRODUCED'] else 'NOT_REPRODUCED', flush=True)


# ---------------------------------------------------------------- A2: GPU-pipeline validation (amendment 1)
def validate_g(out):
    """render_g must (a) reproduce the analytic Lambert plane under a point light (render83 validation (a) scene; pixel-centre rays are
    replaced by jittered rays, so the comparison is the plane-averaged ratio, tolerance 1e-3, and the per-pixel 4-se test) and (b) agree with the
    WL83 path (render_t) on the layered scene of every WL84 image at 1024 spp: object-summed ratio within 0.5 % (catches factor errors; MC
    differences of the independent streams are far smaller than the evaluation thresholds; a conservative z is reported)."""
    os.makedirs(out, exist_ok=True)
    T_ = R81.T
    I, rho = 1.0, 0.5
    sc = mi.load_dict({'type': 'scene', 'floor': {'type': 'rectangle', 'to_world': T_.scale([5, 5, 1]), 'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': rho}}},
                       'light': {'type': 'point', 'position': [0.3, -0.2, 1.0], 'intensity': {'type': 'rgb', 'value': I}}})
    img, acc, acc2, _ = render_g(sc, 256, 401, 256, max_depth=2)
    rays = R81.camera_rays(0, False)
    si = sc.ray_intersect(rays)
    p = np.stack([np.asarray(si.p[k]) for k in range(3)], 1).astype(np.float64)
    ok = np.asarray(si.is_valid())
    dvec = np.array([0.3, -0.2, 1.0])[None] - p
    dist2 = (dvec ** 2).sum(1)
    exact = rho / math.pi * I * np.maximum(dvec[:, 2], 0) / np.sqrt(dist2) / dist2
    g = img.reshape(-1, 3)[:, 0]
    var = (acc2[:, 0] / 256 - (acc[:, 0] / 256) ** 2) / 256
    z = (g[ok] - exact[ok]) / np.sqrt(var[ok] + 1e-30)
    res = {'a_lambert': {'mean_ratio': float(g[ok].sum() / exact[ok].sum()), 'frac_abs_z_gt_4': float(np.mean(np.abs(z) > 4)),
                         'note': 'jittered pixels vs pixel-centre exact values: the ratio is the decisive test (a factor error is caught), z is indicative'}}
    res['a_lambert']['PASS'] = bool(abs(res['a_lambert']['mean_ratio'] - 1) <= 1e-3)
    res['b_layered'] = {}
    for image in IMAGES:
        tag, light, seed, sref, _ = scenes(image)
        obj, _ = R81.masks(sref)
        gi, ga, ga2, _ = render_g(sref, 1024, 402 + IMAGES.index(image), 1024)
        ti, ta, ta2, _ = render_t(sref, 1024, 403 + IMAGES.index(image), 128)
        m = obj.ravel()
        sg = ga[m].sum(); st = ta[m].sum()
        # conservative variance of the channel sum (channels of one path are correlated: var(sum_c) <= (sum_c sd_c)^2)
        sdg = np.sqrt((ga2[m] / 1024 - (ga[m] / 1024) ** 2).clip(0)).sum(1); sdt = np.sqrt((ta2[m] / 1024 - (ta[m] / 1024) ** 2).clip(0)).sum(1)
        zz = (sg - st) / math.sqrt(1024 * ((sdg ** 2).sum() + (sdt ** 2).sum()) + 1e-30)
        res['b_layered'][image] = {'ratio_g_over_t': float(sg / st), 'z_conservative': float(zz), 'PASS': bool(abs(sg / st - 1) <= 0.005)}
    res['PASS'] = bool(res['a_lambert']['PASS'] and all(v['PASS'] for v in res['b_layered'].values()))
    json.dump(res, open(os.path.join(out, 'validate_g.json'), 'w'), indent=1)
    print('PASS' if res['PASS'] else 'FAIL', json.dumps(res), flush=True)


# ---------------------------------------------------------------- B: independent reference
def refgen(image, out):
    os.makedirs(out, exist_ok=True)
    tag, light, seed, sref, _ = scenes(image)
    R = P84['reference']
    base = R['seed_base'] + 1000 * IMAGES.index(image)
    halves = {}
    for h, off in [('A', 0), ('B', 1)]:
        acc = np.zeros((RR.RES * RR.RES, 3)); acc2 = np.zeros_like(acc); tt = []
        n_done = 0
        for c, n in enumerate(R['chunks_per_half']):
            _, a1, a2, T = render_g(sref, n, base + 100 * off + c, P84['reference']['launch_spp'])
            acc += a1; acc2 += a2; n_done += n; tt.append(T)
            np.save(os.path.join(out, f'{image}_ref84{h}_level{c}.npy'), (acc / n_done).reshape(RR.RES, RR.RES, 3).astype(np.float32))
            print(image, h, 'chunk', c, n, 'spp; cumulative', n_done, json.dumps({k: round(v, 2) for k, v in T.items()}), flush=True)
        halves[h] = {'mean': (acc / n_done).reshape(RR.RES, RR.RES, 3), 'var': (acc2 / n_done - (acc / n_done) ** 2).reshape(RR.RES, RR.RES, 3), 'spp': n_done, 'timing': tt}
    hq = 0.5 * (halves['A']['mean'] + halves['B']['mean'])
    np.save(os.path.join(out, f'{image}_ref84.npy'), hq.astype(np.float32))
    np.save(os.path.join(out, f'{image}_ref84A.npy'), halves['A']['mean'].astype(np.float32))
    np.save(os.path.join(out, f'{image}_ref84B.npy'), halves['B']['mean'].astype(np.float32))
    np.save(os.path.join(out, f'{image}_ref84var.npy'), (0.5 * (halves['A']['var'] + halves['B']['var'])).astype(np.float32))
    json.dump({'image': image, 'spp_total': 2 * halves['A']['spp'], 'seeds': {'A': [base + c for c in range(len(R['chunks_per_half']))], 'B': [base + 100 + c for c in range(len(R['chunks_per_half']))]},
               'timing': {h: halves[h]['timing'] for h in halves}}, open(os.path.join(out, f'{image}_ref84.json'), 'w'), indent=1)


# ---------------------------------------------------------------- C: candidate replicates
def replicates(image, out):
    os.makedirs(out, exist_ok=True)
    tag, light, seed, sref, sc = scenes(image)
    C = P84['replicates']
    for a in C['arms']:
        for q in range(C['n']):
            img, _, _, _ = render_g(sc[a], 64, C['seed_base'] + 1000 * IMAGES.index(image) + 10 * ARMS.index(a) + q, 64)
            np.save(os.path.join(out, f'{image}_{a}_rep{q}.npy'), img.astype(np.float32))
        print(image, a, 'replicates done', flush=True)


# ---------------------------------------------------------------- D: uncontended economics and time-to-quality
def econ(out):
    os.makedirs(out, exist_ok=True)
    E = P84['economics']
    me = str(os.getpid())
    foreign = lambda: ' '.join(l for l in os.popen('nvidia-smi --query-compute-apps=pid --format=csv,noheader').read().split() if l.strip() != me)
    apps = foreign()                                   # this process excluded
    res = {'foreign_gpu_processes_before': apps, 'gpu': os.popen('nvidia-smi --query-gpu=name,driver_version --format=csv,noheader').read().strip(), 'images': {}}
    for image in IMAGES:
        tag, light, seed, sref, sc = scenes(image)
        r = {}
        # cold / warm binding: closure construction (state -> Dr.Jit runtime) and first-launch (kernel compile) vs cached
        st = setup()[1][(tag, 'ER_L')]
        t0 = time.perf_counter(); cl = RR.CD.DrClosure83(st[0], st[1]); dr.eval(cl.lam[0][0]); dr.sync_thread()
        r['bind_ms'] = 1e3 * (time.perf_counter() - t0)
        for a in ['REF'] + E['arms']:
            s = sref if a == 'REF' else sc[a]
            reps = []
            for q in range(E['reps'] + 1):
                _, _, _, T = render_g(s, 64, E['seed_base'] + 17 * q, 64)
                reps.append(T)
            warm = reps[1:]
            keys = ['rays_s', 'trace_s', 'eval_s', 'readback_s', 'gpu_ms', 'wall_s']
            r[a] = {'first_call': reps[0], 'per_64spp_median': {k: float(np.median([t[k] for t in warm])) for k in keys}}
            # the WL83 pipeline (numpy rays, host accumulation) for the same work, for comparison only
            w = [render_t(s, 64, E['seed_base'] + 900 + q, 64)[3] for q in range(2)]
            r[a]['wl83_pipeline_per_64spp'] = {k: float(np.median([t[k] for t in w])) for k in keys}
        # time-to-quality against the qualified WL84 reference (spp ladders, warm, launch_spp batches)
        ref = np.load(os.path.join(P84['paths']['ref_dir'], f'{image}_ref84.npy')).astype(np.float64)
        obj, _ = R81.masks(sref)
        ladders = {'REF': E['ladder_ref'], **{a: E['ladder_closure'] for a in E['arms']}}
        curves = {}
        for a, lad in ladders.items():
            s = sref if a == 'REF' else sc[a]
            pts = []
            for spp in lad:
                img, _, _, T = render_g(s, spp, E['seed_base'] + 5000 + spp, P84['reference']['launch_spp'])
                pts.append({'spp': spp, 'err': R81.rel_l1(img, ref, obj), 'gpu_ms': T['gpu_ms'], 'wall_s': T['wall_s'], 'trace_s': T['trace_s']})
            curves[a] = pts
        r['curves'] = curves
        res['images'][image] = r
        print(image, json.dumps({a: r[a]['per_64spp_median'] for a in ['REF'] + E['arms']}), flush=True)
    res['foreign_gpu_processes_after'] = foreign()
    json.dump(res, open(os.path.join(out, 'econ84.json'), 'w'), indent=1)


# ---------------------------------------------------------------- C: evaluation (WL81 Stage F rule, unchanged)
def evaluate(out):
    rows = {}
    ec = json.load(open(os.path.join(out, 'econ', 'econ84.json')))
    rj = json.load(open(os.path.join(W83, 'render', 'render83.json')))
    w83 = {r['image']: r for r in rj['rows']}
    RD = P84['paths']['ref_dir']
    for image in IMAGES:
        tag, light, seed, sref, sc = scenes(image)
        ref = np.load(os.path.join(RD, f'{image}_ref84.npy')).astype(np.float64)
        A = np.load(os.path.join(RD, f'{image}_ref84A.npy')).astype(np.float64); B = np.load(os.path.join(RD, f'{image}_ref84B.npy')).astype(np.float64)
        old = np.load(os.path.join(W83, 'render', f'{image}_refhq.npy')).astype(np.float64)
        obj, ndv = R81.masks(sref)
        lr = R81.lum(ref); hl = obj & (lr >= 3 * np.median(lr[obj]))
        lo = R81.lum(old); hl_old = obj & (lo >= 3 * np.median(lo[obj]))
        unc = 0.5 * R81.rel_l1(A, B, obj)
        row = {'image': image, 'ref84_uncertainty': unc, 'ref83_uncertainty': w83[image]['ref_hq_uncertainty'],
               'old_vs_new_ref_rel_l1': R81.rel_l1(old, ref, obj), 'n_hl_new': int(hl.sum()), 'n_hl_old': int(hl_old.sum()),
               'hl_mask_jaccard': float((hl & hl_old).sum() / max((hl | hl_old).sum(), 1)), 'arms': {}}
        # time-matched reference spp from UNCONTENDED GPU time per 64 spp (econ), reference rendered with a new seed
        g = ec['images'][image]
        ref_gpu = g['REF']['per_64spp_median']['gpu_ms']
        for a in ARMS:
            stored = np.load(os.path.join(W83, 'render', f'{image}_{a}64.npy')).astype(np.float64)
            e = R81.rel_l1(stored, ref, obj); he = R81.rel_l1(stored, ref, hl)
            d = {'err': e, 'hl_err': he, 'hl_err_wl83_mask': R81.rel_l1(stored, ref, hl_old), 'gz_err': R81.rel_l1(stored, ref, obj & (ndv <= 0.25)),
                 'err_vs_old_ref': R81.rel_l1(stored, old, obj), 'ref_limited': unc > 0.25 * e}
            if a in P84['economics']['arms']:
                cg = g[a]['per_64spp_median']['gpu_ms']
                spp_m = max(1, int(math.floor(64 * cg / max(ref_gpu, 1e-9))))
                rm, _, _, _ = render_g(sref, spp_m, P84['time_matched_seed'] + IMAGES.index(image), P84['reference']['launch_spp'])
                d['time_matched_spp_uncontended'] = spp_m
                d['time_matched_ref_err'] = R81.rel_l1(rm, ref, obj)
                d['pass'] = bool(e <= d['time_matched_ref_err'] and he <= P83['render']['hl_err_max'] and not d['ref_limited'])
            reps = sorted(f for f in os.listdir(os.path.join(out, 'replicates')) if f.startswith(f'{image}_{a}_rep'))
            if reps:
                ee = [R81.rel_l1(np.load(os.path.join(out, 'replicates', f)).astype(np.float64), ref, obj) for f in reps]
                hh = [R81.rel_l1(np.load(os.path.join(out, 'replicates', f)).astype(np.float64), ref, hl) for f in reps]
                d['replicates'] = {'n': len(reps), 'err': ee, 'hl_err': hh, 'hl_err_mean': float(np.mean(hh)), 'hl_err_sd': float(np.std(hh, ddof=1)),
                                   'frac_hl_err_le_threshold': float(np.mean(np.array(hh) <= P83['render']['hl_err_max'])),
                                   'frac_ref_limited': float(np.mean([unc > 0.25 * x for x in ee]))}
            row['arms'][a] = d
        q = row['arms']['ER_L']
        row['ER_L_verdict'] = 'INDETERMINATE' if q['ref_limited'] else ('PASS' if q['pass'] else 'FAIL')
        row['E_L_verdict'] = 'INDETERMINATE' if row['arms']['E_L']['ref_limited'] else ('PASS' if row['arms']['E_L'].get('pass') else 'FAIL')
        rows[image] = row
        print(image, row['ER_L_verdict'], round(unc, 4), {a: (round(v['err'], 4), round(v['hl_err'], 4), v.get('pass')) for a, v in row['arms'].items()}, flush=True)
    json.dump({'rows': rows}, open(os.path.join(out, 'eval84.json'), 'w'), indent=1)


if __name__ == '__main__':
    w = sys.argv[1]
    if w == 'audit':
        audit(sys.argv[2])
    elif w == 'validate_g':
        validate_g(sys.argv[2])
    elif w == 'refgen':
        refgen(sys.argv[2], sys.argv[3])
    elif w == 'replicates':
        replicates(sys.argv[2], sys.argv[3])
    elif w == 'econ':
        econ(sys.argv[2])
    elif w == 'evaluate':
        evaluate(sys.argv[2])
    os._exit(0)
