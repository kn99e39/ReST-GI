"""WL86 focused validation of the transport module (protocol stage0.validation V1-V8).

  python tests86.py <root> [--dev]
V1 pt86 (ER) vs Mitsuba's path integrator with the WL83 cmp83 plugin at event depths 6 / 2 / 1 and for the two-light and spot sources
   (event-count <-> path-depth correspondence, emitter semantics)
V2 channel reconstruction e1 + e2 + h == plain total (same seed); class channels sum to e2 / h
V3 null edit (scene rebuilt) and move-and-return (in place) reproduce the image; in-place edit == rebuilt scene (E1 and E2, incl. yaw)
V4 probe field at the camera hit points == camera render (events 2..5), i.e. the lookup target has the declared path-family semantics
V5 patch / bin mapping round trips
V6 source encoding against the analytic irradiance on unoccluded floor patches; additivity over lights
V7 frozen ER states bit-identical to the WL85 fixture states
V8 LOW / ECRO estimator identity: e1, e2 of the ecro mode equal the camera mode (same seed)
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt86 as PT  # noqa: E402
mi, dr, S = PT.mi, PT.dr, PT.S
F, U = PT.F, PT.U
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled83'))
import render83 as RR  # noqa: E402  (registers the WL83 cmp83 plugin)

DEV = '--dev' in sys.argv
FX = PT.FX
RES = 256
CAM = FX['camera']
T = PT.T


def emitters(lights):
    out = []
    for l in lights:
        if 'aim' in l:
            out.append({'type': 'spot', 'to_world': T.look_at(origin=l['pos'], target=l['aim'], up=[0, 1, 0]), 'intensity': {'type': 'rgb', 'value': l['rgb']},
                        'cutoff_angle': l['cone_deg'], 'beam_width': l['cone_deg'] - 0.01})
        else:
            out.append({'type': 'point', 'position': l['pos'], 'intensity': {'type': 'rgb', 'value': l['rgb']}})
    return out


def mi_path(geom, lights, spp, seed, kmax, b=64):
    mats = PT.Mats()
    bs = []
    for i, m in enumerate(PT.SHAPE_MAT):
        RR.REG[f'w86_{i}'] = ('ER', mats.states[m])
        bs.append({'type': 'cmp83', 'key': f'w86_{i}'})
    ctx = PT.Ctx(geom, bsdfs=bs, emitters=emitters(lights))
    integ = mi.load_dict({'type': 'path', 'max_depth': kmax + 1, 'rr_depth': 1000})
    sampler = mi.load_dict({'type': 'independent'})
    n = RES * RES
    acc = np.zeros((n, 3))
    done, k = 0, 0
    while done < spp:
        bb = min(b, spp - done)
        sampler.seed(seed * 7919 + k, n * bb)
        rng = mi.PCG32(size=n * bb, initstate=mi.UInt64(seed * 31 + k), initseq=mi.UInt64(0))
        rays, pix = PT.gen_rays(CAM, RES, bb, rng)
        Lr, _, _ = integ.sample(ctx.scene, sampler, rays, None, True)
        dr.eval(Lr)
        for c in range(3):
            buf = dr.zeros(F, n)
            dr.scatter_reduce(dr.ReduceOp.Add, buf, Lr[c], pix)
            acc[:, c] += np.asarray(buf, np.float64)
        done += bb; k += 1
    return (acc / spp).reshape(RES, RES, 3)


def tot(o):
    return o['tot'] if 'tot' in o else o['e1'] + o['e2'] + o.get('h', 0)


def main():
    root = sys.argv[1]
    out = os.path.join(root, 'results', 'tests'); os.makedirs(out, exist_ok=True)
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    spp = 64 if DEV else 1024
    tol = 0.10 if DEV else 0.025
    res = {}
    E1 = FX['E1']; LS = FX['light_sets']
    ctx = PT.Ctx(E1['T1'])
    g = ctx.gbuffer(CAM, RES)
    objs = [('floor', [0]), ('ceiling', [1]), ('walls', [2, 3, 4]), ('panel', [5]), ('sphere', [6]), ('boxes', [7, 8])]
    # ---- V1
    v1 = {}
    for name, ls, kmax in [('L1_k6', 'L1', 6), ('L1_k2', 'L1', 2), ('L1_k1', 'L1', 1), ('LMIX_k6', 'LMIX', 6), ('LSPOT_k6', 'LSPOT', 6)]:
        a, _ = ctx.render(PT.Lights(LS[ls]), 'ER', CAM, RES, spp, 11, 'cam', plain=True, kmax=kmax, b=128)
        a = PT.pack(a, RES, spp)['tot']
        m = mi_path(E1['T1'], LS[ls], spp, 12, kmax)
        r = {}
        for on, ids in objs:
            mk = np.isin(g['sid'], ids)
            if mk.sum() > 50 and m[mk].sum() > 0:
                r[on] = float(a[mk].sum() / m[mk].sum())
        r['frame'] = float(a.sum() / m.sum())
        v1[name] = r
        print('V1', name, json.dumps(r), flush=True)
    res['V1_vs_mitsuba_path_ratio'] = v1
    res['V1_PASS'] = bool(all(abs(v - 1) <= tol for r in v1.values() for v in r.values()))
    # ---- V2
    lt = PT.Lights(LS['L1'])
    ai, _ = ctx.render(lt, 'ER', CAM, RES, spp, 21, 'cam'); ai = PT.pack(ai, RES, spp)
    ap, _ = ctx.render(lt, 'ER', CAM, RES, spp, 21, 'cam', plain=True); ap = PT.pack(ap, RES, spp)
    sc = float(ap['tot'].mean())
    res['V2'] = {'recon_max_rel_mean': float(np.abs(tot(ai) - ap['tot']).max() / sc),
                 'class_e2_max_rel_mean': float(np.abs(ai['c2'].sum(-1) - ai['e2'].mean(-1)).max() / sc),
                 'class_h_max_rel_mean': float(np.abs(ai['ch'].sum(-1) - ai['h'].mean(-1)).max() / sc)}
    res['V2_PASS'] = bool(max(res['V2'].values()) <= 2e-4)
    # ---- V8
    Bz = PT.fl(np.zeros((ctx.N, PT.NB, 3)))
    ae, _ = ctx.render(lt, 'ER', CAM, RES, spp, 21, 'ecro', B=Bz); ae = PT.pack(ae, RES, spp)
    res['V8'] = {'e1_max_rel_mean': float(np.abs(ae['e1'] - ai['e1']).max() / sc), 'e2_max_rel_mean': float(np.abs(ae['e2'] - ai['e2']).max() / sc),
                 'hl_zero_table_max': float(np.abs(ae['hl']).max())}
    res['V8_PASS'] = bool(res['V8']['e1_max_rel_mean'] <= 2e-4 and res['V8']['e2_max_rel_mean'] <= 2e-4 and res['V8']['hl_zero_table_max'] == 0.0)
    # ---- V3
    v3 = {}
    s2 = min(spp, 256)
    for e in ('E1', 'E2'):
        ex = FX[e]; l = PT.Lights(LS[ex['light_set']])
        fresh0 = PT.Ctx(ex['T0']); f0, _ = fresh0.render(l, 'ER', CAM, RES, s2, 31, 'cam', plain=True)
        null0 = PT.Ctx(ex['T0']); n0, _ = null0.render(l, 'ER', CAM, RES, s2, 31, 'cam', plain=True)
        fresh1 = PT.Ctx(ex['T1']); f1, _ = fresh1.render(l, 'ER', CAM, RES, s2, 31, 'cam', plain=True)
        fresh0.set_geom(ex['T1']); i1, _ = fresh0.render(l, 'ER', CAM, RES, s2, 31, 'cam', plain=True)
        st1 = fresh0.state_rays()
        fresh0.set_geom(ex['T0']); r0, _ = fresh0.render(l, 'ER', CAM, RES, s2, 31, 'cam', plain=True)
        P_ = lambda a: PT.pack(a, RES, s2)['tot']
        mm = float(P_(f0).mean())
        stf = fresh1.state_rays()
        v3[e] = {'null_max_rel_mean': float(np.abs(P_(n0) - P_(f0)).max() / mm), 'return_max_rel_mean': float(np.abs(P_(r0) - P_(f0)).max() / mm),
                 'inplace_vs_rebuilt_max_rel_mean': float(np.abs(P_(i1) - P_(f1)).max() / mm),
                 'inplace_state_equals_rebuilt': bool(np.array_equal(st1[0], stf[0]) and np.array_equal(st1[1], stf[1])),
                 'edit_changes_image_rel_l1': float(np.abs(P_(f1) - P_(f0)).sum() / P_(f0).sum())}
    res['V3'] = v3
    res['V3_PASS'] = bool(all(v['null_max_rel_mean'] <= 2e-4 and v['return_max_rel_mean'] <= 2e-4 and v['inplace_vs_rebuilt_max_rel_mean'] <= 2e-4
                              and v['inplace_state_equals_rebuilt'] and v['edit_changes_image_rel_l1'] > 0.005 for v in v3.values()))
    # ---- V4: probe field at camera hit points vs camera render with events 2..5
    n = RES * RES
    ok = g['ok'].ravel()
    P3 = lambda a: mi.Point3f(*[PT.fl(a[:, k]) for k in range(3)])
    y_all = P3(g['p'].reshape(-1, 3)); d_all = mi.Vector3f(*[PT.fl(g['d'].reshape(-1, 3)[:, k]) for k in range(3)])
    acc = np.zeros((n, 3)); cnt = np.zeros(n)
    bb = 32
    for launch in range(spp // bb):
        N = n * bb
        idx = dr.arange(U, N) % n
        y = dr.gather(mi.Point3f, y_all, idx); d = dr.gather(mi.Vector3f, d_all, idx)
        okl = dr.gather(mi.Bool, mi.Bool(np.ascontiguousarray(ok)), idx)
        sd = dr.opaque(mi.UInt64, 4100 + launch)
        rng = mi.PCG32(size=N, initstate=sd, initseq=mi.UInt64(0))
        o_, okp = ctx.trace(mi.Ray3f(y - d * 2e-3, d), N, rng, sd, 'ER', 5, lt, 'probe', probe_y=y, probe_ok=okl)
        o_['cnt'] = dr.select(okp, F(1.0), F(0.0))
        a = ctx.accum(o_, idx, n)
        for c in range(3):
            acc[:, c] += a[f'f{c}']
        cnt += a['cnt']
    pf = (acc / spp).reshape(RES, RES, 3)
    ac, _ = ctx.render(lt, 'ER', CAM, RES, spp, 41, 'cam', kmax=5); ac = PT.pack(ac, RES, spp)
    refi = ac['e2'] + ac['h']
    v4 = {'probe_valid_fraction': float(cnt.sum() / (ok.sum() * spp))}
    for on, ids in objs:
        mk = np.isin(g['sid'], ids)
        if mk.sum() > 50:
            v4[on] = float(pf[mk].sum() / refi[mk].sum())
    res['V4_probe_vs_camera_ratio'] = v4
    res['V4_PASS'] = bool(v4['probe_valid_fraction'] >= 0.995 and all(abs(v - 1) <= tol for k, v in v4.items() if k != 'probe_valid_fraction'))
    # ---- V5: mapping round trips
    Np = ctx.N * 64
    rng = mi.PCG32(size=Np, initstate=mi.UInt64(51), initseq=mi.UInt64(0))
    pid = dr.arange(U, Np) % ctx.N
    p, nn, (t, bv, n_) = ctx.patch_point(pid, rng.next_float32(), rng.next_float32())
    si = ctx.scene.ray_intersect(mi.Ray3f(p + nn * 0.02, -nn))
    sid = ctx.shape_id(si)
    ph, _ = ctx.patch_bin(si, sid, nn)
    good = si.is_valid() & (dr.norm(si.p - p) < 1e-3) & dr.eq(sid, U(np.ascontiguousarray(np.tile(ctx.G.PSID, 64).astype(np.uint32))))   # same shape (a box face lying on the floor is coplanar with it)
    match = float(dr.sum(dr.select(good & dr.eq(ph, pid), 1.0, 0.0))[0] / dr.sum(dr.select(good, 1.0, 0.0))[0])
    bn = dr.arange(U, Np) % PT.NB
    wl = ctx.dir_in_bin(bn, rng.next_float32(), rng.next_float32())
    bm = float(dr.sum(dr.select(dr.eq(ctx.bin_of(wl.x, wl.y, wl.z), bn), 1.0, 0.0))[0] / Np)
    res['V5'] = {'patch_round_trip': match, 'visible_fraction': float(dr.sum(dr.select(good, 1.0, 0.0))[0] / Np), 'bin_round_trip': bm}
    res['V5_PASS'] = bool(match >= 0.9999 and bm >= 0.9999)
    # ---- V6: source encoding
    s = ctx.source(lt)
    G = ctx.G
    Lp = np.array(LS['L1'][0]['pos']); I = LS['L1'][0]['rgb'][0]
    fl_ = np.arange(256)                                    # floor patches
    dl = Lp - G.PC[fl_]; d2 = (dl ** 2).sum(1)
    ana = I * (dl[:, 2] / np.sqrt(d2)) / d2
    enc = s[fl_].sum(1)[:, 0]
    lit = enc > 0.98 * ana                                  # unoccluded patches
    v6 = {'floor_unoccluded_patches': int(lit.sum()), 'median_ratio': float(np.median(enc[lit] / ana[lit])),
          'max_abs_dev': float(np.abs(enc[lit] / ana[lit] - 1).max())}
    sa = ctx.source(PT.Lights([LS['LMIX'][0]])); sb = ctx.source(PT.Lights([LS['LMIX'][1]])); sab = ctx.source(PT.Lights(LS['LMIX']))
    v6['additivity_rel'] = float(np.abs(sab - sa - sb).max() / sab.max())
    res['V6'] = v6
    res['V6_PASS'] = bool(abs(v6['median_ratio'] - 1) <= 0.01 and v6['max_abs_dev'] <= 0.05 and v6['additivity_rel'] <= 1e-5 and lit.sum() >= 50)
    # ---- V7
    mj = json.load(open(os.path.join(HERE, 'data', 'mats86.json')))
    import hashlib
    st = np.load(os.path.join(HERE, 'data', 'mats86.npz'))['states']
    res['V7'] = {'er_states_equal_wl85': mj['er_states_equal_wl85'], 'sha256_matches': hashlib.sha256(st[:3].tobytes()).hexdigest() == mj['er_states_sha256']}
    res['V7_PASS'] = bool(all(res['V7'].values()))
    res['PASS'] = bool(all(v for k, v in res.items() if k.endswith('_PASS')))
    res['spp'] = spp
    json.dump(res, open(os.path.join(out, 'tests86.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith('V1_vs')}), flush=True)
    print('TESTS', 'PASS' if res['PASS'] else 'FAIL', flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
