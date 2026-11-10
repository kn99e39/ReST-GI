"""WL85 focused validation of the characterisation renderer (protocol stage2_references.validation V1-V6, V8).

  python tests85.py <root> [--dev]
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt85 as PT  # noqa: E402
mi, dr, CD = PT.mi, PT.dr, PT.CD
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled83'))
import render83 as RR  # noqa: E402  (registers the WL83 cmp83 plugin)

DEV = '--dev' in sys.argv
FX = PT.FX
RES = 256
CAM = FX['camera']


def mi_path(root, cfg, spp, seed, b=64):
    mats = PT.Materials(os.path.join(root, 'results', 'census', 'fixture.npz'))
    bs = {}
    for name in PT.SHAPES:
        RR.REG[f'w85_{name}'] = ('ER', mats.states[PT.MAT_OF[name]])
        bs[name] = {'type': 'cmp83', 'key': f'w85_{name}'}
    scene = mi.load_dict(PT.scene_dict(cfg, None, bs))
    integ = mi.load_dict({'type': 'path', 'max_depth': PT.D + 1, 'rr_depth': 1000})
    sampler = mi.load_dict({'type': 'independent'})
    n = RES * RES
    acc = np.zeros((n, 3))
    done, k = 0, 0
    while done < spp:
        bb = min(b, spp - done)
        sampler.seed(seed * 7919 + k, n * bb)
        rng = mi.PCG32(size=n * bb, initstate=mi.UInt64(seed * 31 + k), initseq=mi.UInt64(0))
        rays, pix = PT.gen_rays(CAM, RES, bb, rng)
        Lr, _, _ = integ.sample(scene, sampler, rays, None, True)
        dr.eval(Lr)
        for c in range(3):
            buf = dr.zeros(mi.Float, n)
            dr.scatter_reduce(dr.ReduceOp.Add, buf, Lr[c], pix)
            acc[:, c] += np.asarray(buf, np.float64)
        done += bb; k += 1
    return (acc / spp).reshape(RES, RES, 3)


def run(root, cfg, ref, seed, spp, instrument, moved=None):
    mats, scene, ptr, other, mv = PT.setup(root, cfg)
    acc, C, T_ = PT.render(scene, ptr, mats, ref, CAM, RES, spp, seed, other, mv if moved is None else moved, instrument)
    return PT.pack(acc, RES, spp), scene, ptr


def main():
    root = sys.argv[1]
    out = os.path.join(root, 'results', 'tests'); os.makedirs(out, exist_ok=True)
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    spp = 64 if DEV else 1024
    res = {}
    t0 = time.time()
    # ---- V1: pt85 (ER, plain) vs Mitsuba path with the cmp83 plugin
    a, scene, ptr = run(root, 'T1', 'ER', 11, spp, False)
    m = mi_path(root, 'T1', spp, 12)
    mid, _ = PT.gbuffer(scene, ptr, CAM, RES)
    v1 = {}
    for name, k in [('floor', 0), ('panel', 1), ('sphere', 2), ('wall', 3), ('baffle', 4)]:
        msk = mid == k
        if msk.sum() > 50:
            v1[name] = float(a['tot'][msk].sum() / max(m[msk].sum(), 1e-30))
    v1['frame'] = float(a['tot'].sum() / m.sum())
    res['V1_vs_mitsuba_path_ratio'] = v1
    res['V1_PASS'] = bool(all(abs(v - 1) <= (0.10 if DEV else 0.02) for v in v1.values()))
    print('V1', json.dumps(v1), flush=True)
    # ---- V2 / V3: instrumented reconstruction and equality with the plain render (same seed)
    i1, _, _ = run(root, 'T1', 'ER', 11, spp, True)
    tm = i1['tot'].mean(2)
    rel = lambda x, y: float(abs(x - y) / max(abs(y), 1e-30))
    v2 = {'depth_sum': rel(i1['depth'].sum(), tm.sum()), 'class_sum': rel(i1['dir'].sum() + i1['mat'].sum(), tm.sum()),
          'dirty_depth_sum': rel(i1['ddepth'].sum(), i1['dirty'].mean(2).sum()), 'dirty_class_sum': rel(i1['ddir'].sum() + i1['dmat'].sum(), i1['dirty'].mean(2).sum()),
          'per_pixel_class_max': float(np.abs(i1['dir'].sum(2) + i1['mat'].sum(2) - tm).max() / tm.mean())}
    res['V2_reconstruction'] = v2
    res['V2_PASS'] = bool(max(v2['depth_sum'], v2['class_sum'], v2['dirty_depth_sum'], v2['dirty_class_sum']) <= 1e-4 and v2['per_pixel_class_max'] <= 1e-3)
    v3 = {'frame': rel(i1['tot'].sum(), a['tot'].sum()), 'per_pixel_max': float(np.abs(i1['tot'] - a['tot']).max() / a['tot'].mean())}
    res['V3_instrumented_vs_plain'] = v3
    res['V3_PASS'] = bool(v3['frame'] <= 1e-4 and v3['per_pixel_max'] <= 1e-3)
    print('V2', json.dumps(v2), 'V3', json.dumps(v3), flush=True)
    # ---- V4 / V5 / V8: null edit, move-and-return, clean-part identity under common random numbers
    i0, _, _ = run(root, 'T0', 'ER', 11, spp, True)
    nl, _, _ = run(root, 'NULL', 'ER', 11, spp, True)
    rt, _, _ = run(root, 'RET', 'ER', 11, spp, True)
    pp = lambda x, y: float(np.abs(x - y).max() / y.mean())
    res['V4_null'] = {'frame': rel(nl['tot'].sum(), i0['tot'].sum()), 'per_pixel_max': pp(nl['tot'], i0['tot']), 'dirty_sum': float(nl['dirty'].sum())}
    res['V4_PASS'] = bool(res['V4_null']['frame'] <= 1e-4 and res['V4_null']['per_pixel_max'] <= 1e-3 and res['V4_null']['dirty_sum'] == 0.0)
    res['V5_move_and_return'] = {'frame': rel(rt['tot'].sum(), i0['tot'].sum()), 'per_pixel_max': pp(rt['tot'], i0['tot'])}
    res['V5_PASS'] = bool(res['V5_move_and_return']['frame'] <= 1e-4 and res['V5_move_and_return']['per_pixel_max'] <= 1e-3)
    c0 = i0['tot'] - i0['dirty']; c1 = i1['tot'] - i1['dirty']
    res['V8_clean_identity'] = {'frame': rel(c1.sum(), c0.sum()), 'per_pixel_max': pp(c1, c0), 'dirty_share_T0': float(i0['dirty'].sum() / i0['tot'].sum()),
                                'dirty_share_T1': float(i1['dirty'].sum() / i1['tot'].sum())}
    res['V8_PASS'] = bool(res['V8_clean_identity']['frame'] <= 1e-4 and res['V8_clean_identity']['per_pixel_max'] <= 1e-3)
    print('V4', json.dumps(res['V4_null']), 'V5', json.dumps(res['V5_move_and_return']), 'V8', json.dumps(res['V8_clean_identity']), flush=True)
    # ---- V6: Dr.Jit closure vs the torch reference on the fixture ER closures
    import torch
    sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled82'))
    import closure82 as CL
    import closure83  # noqa: F401
    fj = json.load(open(os.path.join(root, 'results', 'census', 'fixture.json')))
    W83 = os.path.expanduser('~/rest_compiled83/results')
    rng = np.random.default_rng(85060)
    n = 1 << 18
    v6 = {}
    mats = PT.Materials(os.path.join(root, 'results', 'census', 'fixture.npz'))
    for name, k in [('B', 0), ('S1', 1), ('S2', 2)]:
        info = fj['materials'][name]
        z = torch.from_numpy(np.load(os.path.join(W83, 'fit', f'fit_{info["set"]}_L_ER_s8301.npz'))['z'][info['index']:info['index'] + 1]).cuda()
        with torch.no_grad():
            p = CL.decode('ER', 'cert', z)
        def dirs():
            mu = np.where(rng.random(n) < 0.5, rng.random(n), np.exp(rng.uniform(np.log(1e-4), 0, n))); ph = rng.uniform(0, 2 * np.pi, n); s = np.sqrt(1 - mu * mu)
            return np.stack([s * np.cos(ph), s * np.sin(ph), mu], 1).astype(np.float32)
        wi, wo = dirs(), dirs()
        V = lambda x: mi.Vector3f(mi.Float(x[:, 0]), mi.Float(x[:, 1]), mi.Float(x[:, 2]))
        cl = CD.DrClosure83('ER', mats.states[k])
        f, pd = cl.eval_pdf(V(wi), V(wo))
        f = np.stack([np.asarray(f[c], np.float64) for c in range(3)], 1); pd = np.asarray(pd, np.float64)
        with torch.no_grad():
            pl = CL.gather(p, torch.zeros(n, dtype=torch.long, device='cuda'))
            ft = CL.eval_closure('ER', pl, torch.from_numpy(wi).cuda(), torch.from_numpy(wo).cuda()).double().cpu().numpy()
            pt = CL.pdf_closure('ER', pl, torch.from_numpy(wi).cuda(), torch.from_numpy(wo).cuda()).double().cpu().numpy()
        re = (np.abs(f - ft) / np.maximum(np.abs(ft), 1e-8)).ravel(); rp = np.abs(pd - pt) / np.maximum(np.abs(pt), 1e-8)
        v6[name] = {'eval_median': float(np.median(re)), 'eval_p999': float(np.quantile(re, .999)), 'pdf_median': float(np.median(rp)), 'pdf_p999': float(np.quantile(rp, .999))}
    res['V6_closure_vs_torch'] = v6
    res['V6_PASS'] = bool(all(v['eval_median'] <= 1e-5 and v['eval_p999'] <= 1e-3 and v['pdf_median'] <= 1e-5 and v['pdf_p999'] <= 1e-3 for v in v6.values()))
    print('V6', json.dumps(v6), flush=True)
    res['PASS'] = bool(all(res[k] for k in res if k.endswith('_PASS')))
    res['seconds'] = time.time() - t0; res['spp'] = spp
    json.dump(res, open(os.path.join(out, 'tests85.json'), 'w'), indent=1)
    print('TESTS', 'PASS' if res['PASS'] else 'FAIL', flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
