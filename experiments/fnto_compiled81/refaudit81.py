"""WL81 Stage A reference audit (the reference is frozen only if every check below passes).

  python refaudit81.py iface  <out_dir>            # cuda: per-lane interface ports == Mitsuba 3.5.1 roughdielectric / roughconductor plugins
  python refaudit81.py walk   <out_dir>            # cuda: layered estimator internal consistency (sample vs eval albedo, reciprocity, depth truncation, noise)
  python refaudit81.py slab   <out_dir>            # llvm: independent Mitsuba volpath geometric slab vs layered estimator (estimator also run in llvm)
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
stage, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
import layered81 as L
mi = L.init('llvm_ad_rgb' if stage == 'slab' else 'cuda_ad_rgb')  # slab: llvm (LabServer63 cuda film defect)
import drjit as dr

SEED = 81100


def V3(a):
    a = np.ascontiguousarray(a, np.float32)
    return mi.Vector3f(mi.Float(a[:, 0]), mi.Float(a[:, 1]), mi.Float(a[:, 2]))


def np1(x):
    return np.asarray(x, np.float64)


def np3(v):
    return np.stack([np.asarray(v[k], np.float64) for k in range(3)], 1)


def sph(n, rng, lower=False):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def dirs(th_deg, ph_deg):
    t, p = np.radians(th_deg), np.radians(ph_deg)
    return np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1)


def mat_const(n, ai, eta, tau, alb, r, ab):
    f = lambda x: np.full(n, x, np.float32)
    c = lambda x: mi.Color3f(*[mi.Float(f(x[k])) for k in range(3)])
    return L.Mat(mi.Float(f(ai)), mi.Float(f(eta)), mi.Float(f(tau)), c(alb), c(r), mi.Float(f(ab)))


def raw_tex(rgb):
    b = mi.Bitmap(np.asarray(rgb, np.float32).reshape(1, 1, 3))
    return {'type': 'bitmap', 'bitmap': b, 'raw': True, 'filter_type': 'nearest'}


def rel(a, b, floor):
    return np.abs(a - b) / np.maximum(np.abs(b), floor)


# ---------------------------------------------------------------------------------------------------------------- iface
def iface():
    rng = np.random.default_rng(SEED)
    n = 1 << 16
    res = {}
    for (eta, a) in [(1.33, 0.05), (1.5, 0.2), (1.8, 0.5)]:
        bsdf = mi.load_dict({'type': 'roughdielectric', 'distribution': 'ggx', 'alpha': a, 'int_ior': eta, 'ext_ior': 1.0, 'sample_visible': True})
        m = mat_const(n, a, eta, 1.0, (0.5,) * 3, (0.5,) * 3, 0.1)
        wi, wo = sph(n, rng), sph(n, rng)
        si = dr.zeros(mi.SurfaceInteraction3f, n)
        si.wi = V3(wi)
        u1 = rng.random(n).astype(np.float32)
        u2 = rng.random((n, 2)).astype(np.float32)
        for mode_name, mode, radiance in [('radiance', mi.TransportMode.Radiance, True), ('importance', mi.TransportMode.Importance, False)]:
            for comp, mask, ro, to in [('both', mi.BSDFFlags.All, True, True), ('R', mi.BSDFFlags.GlossyReflection, True, False),
                                       ('T', mi.BSDFFlags.GlossyTransmission, False, True)]:
                ctx = mi.BSDFContext(mode)
                ctx.type_mask = mask
                ref_e = np3(bsdf.eval(ctx, si, V3(wo)))[:, 0]
                ref_p = np1(bsdf.pdf(ctx, si, V3(wo)))
                my_e = np1(L.diel_eval(m, V3(wi), V3(wo), radiance, ro, to))
                my_p = np1(L.diel_pdf(m, V3(wi), V3(wo), ro, to))
                bs, w = bsdf.sample(ctx, si, mi.Float(u1), mi.Point2f(mi.Float(u2[:, 0]), mi.Float(u2[:, 1])))
                ref_wo, ref_w, ref_sp = np3(bs.wo), np3(w)[:, 0], np1(bs.pdf)
                s_wo, s_w, s_p, _, s_ok = L.diel_sample(m, V3(wi), radiance, ro, to, mi.Float(u1), mi.Point2f(mi.Float(u2[:, 0]), mi.Float(u2[:, 1])))
                s_wo, s_w, s_p = np3(s_wo), np1(s_w), np1(s_p)
                ok = ref_sp > 0
                key = f'diel_eta{eta}_a{a}_{mode_name}_{comp}'
                res[key] = {
                    'eval_max_rel': float(rel(my_e, ref_e, 1e-3).max()), 'eval_nonzero_mismatch': int(((my_e > 0) != (ref_e > 0)).sum()),
                    'pdf_max_rel': float(rel(my_p, ref_p, 1e-3).max()), 'pdf_nonzero_mismatch': int(((my_p > 0) != (ref_p > 0)).sum()),
                    'sample_valid_mismatch': int(((s_p > 0) != ok).sum()),
                    'sample_wo_max_abs': float(np.abs(s_wo - ref_wo)[ok].max()) if ok.any() else 0.0,
                    'sample_weight_max_rel': float(rel(s_w, ref_w, 1e-3)[ok].max()) if ok.any() else 0.0,
                    'sample_pdf_max_rel': float(rel(s_p, ref_sp, 1e-3)[ok].max()) if ok.any() else 0.0}
    for (eta, k, a) in [((0.2, 0.9, 1.1), (3.9, 2.4, 2.2), 0.05), ((1.0 / 1.5,) * 3, (0.5, 3.0, 40.0), 0.3), ((1.0 / 1.8,) * 3, (8.0, 1.0, 0.2), 0.6)]:
        bsdf = mi.load_dict({'type': 'roughconductor', 'distribution': 'ggx', 'alpha': a, 'eta': raw_tex(eta), 'k': raw_tex(k), 'sample_visible': True})
        m = mat_const(n, 0.1, 1.5, 1.0, (0.5,) * 3, (0.5,) * 3, a)
        m.ce = mi.Color3f(*[mi.Float(np.full(n, eta[c], np.float32)) for c in range(3)])
        m.ck = mi.Color3f(*[mi.Float(np.full(n, k[c], np.float32)) for c in range(3)])
        wi, wo = sph(n, rng), sph(n, rng)
        si = dr.zeros(mi.SurfaceInteraction3f, n)
        si.wi = V3(wi)
        si.uv = mi.Point2f(0.5, 0.5)
        ctx = mi.BSDFContext()
        u2 = rng.random((n, 2)).astype(np.float32)
        ref_e, my_e = np3(bsdf.eval(ctx, si, V3(wo))), np3(L.cond_eval(m, V3(wi), V3(wo)))
        ref_p, my_p = np1(bsdf.pdf(ctx, si, V3(wo))), np1(L.cond_pdf(m, V3(wi), V3(wo)))
        bs, w = bsdf.sample(ctx, si, mi.Float(0.5), mi.Point2f(mi.Float(u2[:, 0]), mi.Float(u2[:, 1])))
        s_wo, s_w, s_p, s_ok = L.cond_sample(m, V3(wi), mi.Point2f(mi.Float(u2[:, 0]), mi.Float(u2[:, 1])))
        ok = (np1(bs.pdf) > 0) & (np3(w).max(1) > 0)   # Mitsuba leaves bs.pdf unmasked on rejected lanes; validity = nonzero weight
        key = f'cond_a{a}_eta{eta[0]:.3f}'
        res[key] = {
            'eval_max_rel': float(rel(my_e, ref_e, 1e-3).max()), 'eval_nonzero_mismatch': int(((my_e > 0) != (ref_e > 0)).sum()),
            'pdf_max_rel': float(rel(my_p, ref_p, 1e-3).max()), 'pdf_nonzero_mismatch': int(((my_p > 0) != (ref_p > 0)).sum()),
            'sample_valid_mismatch': int(((np1(s_p) > 0) != ok).sum()),
            'sample_wo_max_abs': float(np.abs(np3(s_wo) - np3(bs.wo))[ok].max()),
            'sample_weight_max_rel': float(rel(np3(s_w), np3(w), 1e-3)[ok].max()),
            'sample_pdf_max_rel': float(rel(np1(s_p), np1(bs.pdf), 1e-3)[ok].max())}
    worst = {k: max(v[k] for v in res.values()) for k in next(iter(res.values()))}
    res['_worst'] = worst
    json.dump(res, open(os.path.join(out, 'iface.json'), 'w'), indent=1)
    print(json.dumps(worst, indent=1))


# ---------------------------------------------------------------------------------------------------------------- walk
MATS = {  # audit materials (not in any WL81 population; hand-chosen to span the descriptor box corners)
    'thin_clear':   dict(ai=0.05, eta=1.5, tau=0.02, alb=(0.9, 0.9, 0.9), r=(0.95, 0.64, 0.54), ab=0.1),
    'mid':          dict(ai=0.2, eta=1.5, tau=0.5, alb=(0.9, 0.5, 0.1), r=(0.9, 0.6, 0.3), ab=0.25),
    'thick_white':  dict(ai=0.1, eta=1.33, tau=2.0, alb=(0.99, 0.99, 0.99), r=(0.98, 0.98, 0.98), ab=0.05),
    'rough_dark':   dict(ai=0.5, eta=1.8, tau=1.0, alb=(0.2, 0.0, 0.6), r=(0.05, 0.3, 0.7), ab=0.6),
}


def walk_eval(md, wi, wo, nwalk, seed, max_depth=64):
    """Mean and standard error of layered_eval over nwalk walks for each (wi, wo) pair."""
    npair = len(wi)
    n = npair * nwalk
    m = mat_const(n, **md)
    rng = mi.PCG32(size=n, initstate=seed, initseq=dr.arange(mi.UInt64, n))
    e = np3(L.layered_eval(m, V3(np.repeat(wi, nwalk, 0)), V3(np.repeat(wo, nwalk, 0)), rng, max_depth)).reshape(npair, nwalk, 3)
    return e.mean(1), e.std(1) / math.sqrt(nwalk)


def walk_albedo_sample(md, wi, nwalk, seed):
    n = len(wi) * nwalk
    m = mat_const(n, **md)
    rng = mi.PCG32(size=n, initstate=seed, initseq=dr.arange(mi.UInt64, n))
    _, w, _ = L.layered_sample(m, V3(np.repeat(wi, nwalk, 0)), rng)
    w = np3(w).reshape(len(wi), nwalk, 3)
    return w.mean(1), w.std(1) / math.sqrt(nwalk)


def walk():
    res = {}
    rng = np.random.default_rng(SEED + 1)
    chunk = 1 << 22
    for name, md in MATS.items():
        r = {}
        # (1) albedo: E[sample weight] vs MC integral of eval over uniform hemisphere wo (independent estimators)
        thetas = [0.0, 30.0, 60.0, 80.0, 89.0]
        wi = dirs(np.array(thetas), np.zeros(len(thetas)))
        a_s, se_s = walk_albedo_sample(md, wi, 1 << 20, SEED + 10)
        nw = 1 << 18   # uniform-hemisphere wo samples per wi (1 walk each)
        u = rng.random((len(thetas), nw, 2))
        z = u[..., 0]
        rr = np.sqrt(np.maximum(0, 1 - z * z))
        wo = np.stack([rr * np.cos(2 * np.pi * u[..., 1]), rr * np.sin(2 * np.pi * u[..., 1]), z], -1)
        vals = []
        for i in range(len(thetas)):
            e, _ = walk_eval(md, np.repeat(wi[i:i + 1], nw, 0), wo[i], 1, SEED + 20 + i)
            vals.append(e * 2 * math.pi)
        vals = np.stack(vals)
        a_e, se_e = vals.mean(1), vals.std(1) / math.sqrt(nw)
        z_alb = (a_s - a_e) / np.sqrt(se_s ** 2 + se_e ** 2 + 1e-30)
        r['albedo'] = {'theta': thetas, 'sample': a_s.tolist(), 'sample_se': se_s.tolist(), 'evalint': a_e.tolist(), 'evalint_se': se_e.tolist(),
                       'z': z_alb.tolist(), 'max_abs_z': float(np.abs(z_alb).max()), 'max_albedo': float(a_s.max())}
        # (2) reciprocity in expectation: f(wi, wo) = f(wo, wi)  (eval/cos_o vs eval/cos_i)
        th = np.array([[10, 50], [30, 70], [45, 85], [0, 60], [75, 80]], float)
        ph = np.array([[0, 180], [20, 100], [0, 0], [0, 90], [0, 200]], float)
        A, B = dirs(th[:, 0], ph[:, 0]), dirs(th[:, 1], ph[:, 1])
        e1, s1 = walk_eval(md, A, B, 1 << 20, SEED + 30)
        e2, s2 = walk_eval(md, B, A, 1 << 20, SEED + 31)
        f1, f2 = e1 / B[:, 2:3], e2 / A[:, 2:3]
        sf1, sf2 = s1 / B[:, 2:3], s2 / A[:, 2:3]
        zr = (f1 - f2) / np.sqrt(sf1 ** 2 + sf2 ** 2 + 1e-30)
        r['reciprocity'] = {'f_ab': f1.tolist(), 'f_ba': f2.tolist(), 'z': zr.tolist(), 'max_abs_z': float(np.abs(zr).max()),
                            'max_rel': float((np.abs(f1 - f2) / np.maximum(np.abs(f1) + np.abs(f2), 1e-12) * 2).max())}
        # (3) depth truncation: max_depth 64 vs 1024 (same pairs, independent walks)
        e64, s64 = walk_eval(md, A, B, 1 << 20, SEED + 40, 64)
        e1k, s1k = walk_eval(md, A, B, 1 << 20, SEED + 41, 1024)
        r['truncation'] = {'rel_diff': ((e1k - e64) / np.maximum(e1k, 1e-12)).tolist(),
                           'z': ((e1k - e64) / np.sqrt(s64 ** 2 + s1k ** 2 + 1e-30)).tolist()}
        # (4) per-walk noise: relative standard error at 1 walk -> walks needed for a target relative SE
        rse1 = (s1 * math.sqrt(1 << 20)) / np.maximum(e1, 1e-12)
        r['noise'] = {'rel_sd_one_walk': rse1.tolist()}
        res[name] = r
        print(name, json.dumps({'albedo_max_abs_z': r['albedo']['max_abs_z'], 'max_albedo': r['albedo']['max_albedo'],
                                'recip_max_abs_z': r['reciprocity']['max_abs_z'], 'trunc': r['truncation']['rel_diff']}), flush=True)
    # (5) cost: walks per second (layered_eval) on a mixed batch
    md = MATS['mid']
    n = 1 << 22
    wi = dirs(rng.uniform(0, 85, n), rng.uniform(0, 360, n))
    wo = dirs(rng.uniform(0, 85, n), rng.uniform(0, 360, n))
    for rep in range(3):
        m = mat_const(n, **md)
        rg = mi.PCG32(size=n, initstate=SEED + 50 + rep, initseq=dr.arange(mi.UInt64, n))
        dr.eval(m.ai, m.ce, m.ck, m.alb)
        dr.sync_thread()
        t0 = time.perf_counter()
        e = L.layered_eval(m, V3(wi), V3(wo), rg)
        dr.eval(e)
        dr.sync_thread()
        dt = time.perf_counter() - t0
    res['cost'] = {'walks': n, 'seconds_last_of_3': dt, 'walks_per_s': n / dt}
    json.dump(res, open(os.path.join(out, 'walk.json'), 'w'), indent=1)
    print(json.dumps(res['cost']))


# ---------------------------------------------------------------------------------------------------------------- slab
def slab():
    """Independent check: Mitsuba volpath on a geometric slab (rough dielectric top with an interior homogeneous isotropic medium of optical
    thickness tau, rough conductor floor, directional light of unit irradiance, radiance meter). L / E = eval(wi_cam, wo_light)."""
    T = mi.ScalarTransform4f
    cfg = [('mid', 0.0, 45.0, 0.0), ('mid', 60.0, 30.0, 180.0), ('mid', 75.0, 70.0, 90.0),
           ('thick_white', 30.0, 50.0, 180.0), ('rough_dark', 45.0, 20.0, 60.0), ('thin_clear', 20.0, 40.0, 180.0)]
    nrep, spp = 8, 1 << 18
    res = []
    for (name, tc, tl, dphi) in cfg:
        md = MATS[name]
        wc = dirs(np.array([tc]), np.array([0.0]))[0]
        wl = dirs(np.array([tl]), np.array([dphi]))[0]
        eta_c = [1.0 / md['eta']] * 3
        k_c = [2 * math.sqrt(x) / math.sqrt(1 - x) / md['eta'] for x in md['r']]
        Lext, d = 2000.0, 1.0
        vals = []
        t0 = time.time()
        for rep in range(nrep):
            scene = mi.load_dict({
                'type': 'scene',
                'integrator': {'type': 'volpath', 'max_depth': -1, 'rr_depth': 64},
                'top': {'type': 'rectangle', 'to_world': T.translate([0, 0, d]) @ T.scale([Lext, Lext, 1]),
                        'bsdf': {'type': 'roughdielectric', 'distribution': 'ggx', 'alpha': md['ai'], 'int_ior': md['eta'], 'ext_ior': 1.0},
                        'interior': {'type': 'homogeneous', 'sigma_t': md['tau'] / d, 'albedo': {'type': 'rgb', 'value': list(md['alb'])},
                                     'phase': {'type': 'isotropic'}}},
                'bot': {'type': 'rectangle', 'to_world': T.scale([Lext, Lext, 1]),
                        'bsdf': {'type': 'roughconductor', 'distribution': 'ggx', 'alpha': md['ab'], 'eta': raw_tex(eta_c), 'k': raw_tex(k_c)}},
                'light': {'type': 'directional', 'direction': (-wl).tolist(), 'irradiance': {'type': 'rgb', 'value': 1.0}},
                'sensor': {'type': 'radiancemeter', 'origin': (np.array([0, 0, d]) + wc * 0.5).tolist(), 'direction': (-wc).tolist(),
                           'film': {'type': 'hdrfilm', 'width': 1, 'height': 1, 'rfilter': {'type': 'box'}, 'pixel_format': 'rgb'},
                           'sampler': {'type': 'independent', 'sample_count': spp}}})
            img = np.asarray(mi.render(scene, seed=SEED + 100 * rep + len(res)), np.float64).reshape(-1)[:3]
            vals.append(img)
        vals = np.array(vals)
        t_mi = (time.time() - t0) / nrep
        mv, ms = vals.mean(0), vals.std(0, ddof=1) / math.sqrt(nrep)
        t0 = time.time()
        ev, es = walk_eval(md, wc[None], wl[None], 1 << 21, SEED + 200 + len(res), 1024)
        t_l = time.time() - t0
        z = (ev[0] - mv) / np.sqrt(es[0] ** 2 + ms ** 2 + 1e-30)
        rec = {'mat': name, 'theta_cam': tc, 'theta_light': tl, 'dphi': dphi, 'volpath': mv.tolist(), 'volpath_se': ms.tolist(),
               'layered': ev[0].tolist(), 'layered_se': es[0].tolist(), 'z': z.tolist(),
               'rel': ((ev[0] - mv) / np.maximum(mv, 1e-12)).tolist(), 'volpath_s_per_rep': t_mi, 'layered_s': t_l}
        res.append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(res, open(os.path.join(out, 'slab.json'), 'w'), indent=1)


def trunc():
    """max_depth 1024 vs 4096 on the near-white thick audit material (5 pairs, 2^21 walks each, independent streams)."""
    md = MATS['thick_white']
    th = np.array([[10, 50], [30, 70], [45, 85], [0, 60], [75, 80]], float)
    ph = np.array([[0, 180], [20, 100], [0, 0], [0, 90], [0, 200]], float)
    A, B = dirs(th[:, 0], ph[:, 0]), dirs(th[:, 1], ph[:, 1])
    e1, s1 = walk_eval(md, A, B, 1 << 21, SEED + 60, 1024)
    e4, s4 = walk_eval(md, A, B, 1 << 21, SEED + 61, 4096)
    rel = (e4 - e1) / np.maximum(e4, 1e-12)
    z = (e4 - e1) / np.sqrt(s1 ** 2 + s4 ** 2 + 1e-30)
    ok = bool(np.all((np.abs(rel) <= 1e-3) | (np.abs(z) <= 3)))
    r = {'rel': rel.tolist(), 'z': z.tolist(), 'PASS': ok}
    json.dump(r, open(os.path.join(out, 'trunc.json'), 'w'), indent=1)
    print(json.dumps(r))


{'iface': iface, 'walk': walk, 'slab': slab, 'trunc': trunc}[stage]()
