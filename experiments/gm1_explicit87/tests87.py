"""WL87 Stage 0: reference / contract validation and explicit-operator correctness tests (protocol stage0).

  python tests87.py <R87> <R86> [--dev] [--only=T_A,T_B]
Writes <R87>/results/tests/tests87.json; prints TESTS PASS / FAIL (gate tests only).
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fx87  # noqa: E402
import quad87 as Q  # noqa: E402
import xport87 as X  # noqa: E402
import torch  # noqa: E402
PT = fx87.PT
mi, dr, S, CD = PT.mi, PT.dr, PT.S, PT.CD
FX = PT.FX
P87 = fx87.P87
NB, K, KAPPA = S.NB, S.K, S.KAPPA
DEV = '--dev' in sys.argv
GATE = ['T_WL86', 'T_READOUT', 'T_SYM', 'T_FBAR', 'T_FRAMES', 'T_LAMB', 'T_Q', 'T_IMPL', 'T_ZERO', 'T_LIN', 'T_NONNEG', 'T_FURNACE', 'T_STATE', 'T_REAL',
        'T_RR', 'T_SEEDS']


def rel(a, b):
    return float(np.abs(np.asarray(a, np.float64) - b).sum() / max(np.abs(b).sum(), 1e-30))


def t_wl86(R86):
    sys.path.insert(0, fx87.E86)
    import analyze86 as A
    fin = json.load(open(os.path.join(R86, 'results', 'final.json')))
    s1 = json.load(open(os.path.join(R86, 'results', 'stage1.json')))
    md = 0.0
    for e in ('E1', 'E2'):
        M, _ = A.masks(R86, e)
        for t in ('T0', 'T1'):
            ref = A.load(R86, f'REF_ER_{e}_{t}')
            for sd in ('ecro_s8601', 'ecro_s8602'):
                T = A.eh_table(R86, ref, A.load(R86, f'EC_{sd}_ecro_{e}_{t}'), M)
                for k, v in T.items():
                    md = max(md, abs(v['E_H'] - fin['seeds'][sd]['fidelity'][e][t][k]['E_H']))
            T = A.eh_table(R86, ref, A.load(R86, f'ORA_{e}_{t}'), M)
            for k, v in T.items():
                md = max(md, abs(v['E_H'] - s1[e]['oracle'][t][k]['E_H']))
    return {'max_abs_diff': md, 'PASS': md <= 1e-9}


def t_readout(R86):
    import ecro86 as EC
    out = {}
    for tag, mt, var in (('ORA_E2_T1_r0', 'oracle:F_E2_T1', 'oracle'), ('EC_ecro_s8601_ecro_E2_T1_r0', 'ecro_s8601', 'ecro')):
        seed = 86400 if var == 'oracle' else 86600
        spp = 256 if DEV else 4096
        o, B, T_ = EC.render_cfg(R86, 'E2_T1', mt, var, seed, spp)
        st = np.load(os.path.join(R86, 'results', 'renders', f'{tag}.npz'))['hl'].astype(np.float64)
        out[tag] = rel(o['hl'], st)
    return {'hl_rel_l1': out, 'PASS': (not DEV) and max(out.values()) <= 1e-4}


def t_sym():
    st, lam, alb = Q.mats()
    n = 1 << (14 if DEV else 18)
    rng = np.random.default_rng(87901)
    def hemi(n):
        z = rng.random(n); ph = rng.random(n) * 2 * math.pi - math.pi; r = np.sqrt(1 - z * z)
        return np.stack([r * np.cos(ph), r * np.sin(ph), z], 1)
    a, b = hemi(n), hemi(n)
    V = lambda x: mi.Vector3f(*[PT.fl(x[:, k]) for k in range(3)])
    out = {}
    for m in Q.ER_MATS:
        cl = CD.DrClosure83('ER', st[m])
        fab = cl.eval(V(a), V(b)); fba = cl.eval(V(b), V(a))
        dr.eval(fab, fba)
        x = np.stack([np.asarray(fab[c], np.float64) for c in range(3)], 1) / b[:, 2:3]
        y = np.stack([np.asarray(fba[c], np.float64) for c in range(3)], 1) / a[:, 2:3]
        sc = np.maximum(np.maximum(x, y), 1e-6)
        out[m] = float(np.max(np.abs(x - y) / sc))
    return {'max_rel': out, 'PASS': max(out.values()) <= 1e-3}


def fbar_uniform(state, w, M, seed):
    """Independent per-bin uniform estimator of fbar for one direction w: [NB, 3] for two halves."""
    cl = CD.DrClosure83('ER', state)
    res = []
    for h in range(2):
        lanes = NB * M
        lane = dr.arange(PT.U, lanes)
        bn = lane // M
        rng = mi.PCG32(size=lanes, initstate=dr.opaque(mi.UInt64, seed + h), initseq=mi.UInt64(0))
        v = PT.Ctx.dir_in_bin(bn, rng.next_float32(), rng.next_float32())
        wv = mi.Vector3f(float(w[0]), float(w[1]), float(w[2]))
        f = cl.eval(wv, v, v.z > 1e-6) * dr.rcp(dr.maximum(v.z, 1e-12))
        dr.eval(f); dr.sync_thread()
        arr = np.stack([np.asarray(f[c], np.float64) for c in range(3)], 1).reshape(NB, M, 3).mean(1)
        res.append(arr)
    return res


def t_fbar():
    st, lam, alb = Q.mats()
    dirs = Q.ray_dirs(X.realization_us(86001))[::64]                       # 8 of the deployed ray directions
    out = {}
    ok = True
    Mu = 1 << (10 if DEV else 14)
    for m in Q.ER_MATS:
        fa = Q.fbar(st[m], dirs, 1 << (12 if DEV else 16), 87910 + m)
        fb = Q.fbar(st[m], dirs, 1 << (12 if DEV else 16), 87915 + m)
        rows = []
        for i, w in enumerate(dirs):
            ua, ub = fbar_uniform(st[m], w, Mu, 87920 + 10 * m + i)
            u = 0.5 * (ua + ub); fm = 0.5 * (fa[i] + fb[i])
            noise = math.sqrt((0.5 * rel(ua, ub)) ** 2 + (0.5 * rel(fa[i], fb[i])) ** 2)       # noise of the two half-means
            e = rel(fm, u)
            rows.append([e, noise])
            ok &= e <= 0.01 + 4 * noise
        out[m] = rows
    return {'rel_l1_and_uniform_noise': out, 'PASS': bool(ok)}


def t_frames():
    """Curved-patch estimator (integ_frames) with an identity frame reduces to fbar x DOMEGA (independent seeds, two halves each)."""
    st, lam, alb = Q.mats()
    dirs = Q.ray_dirs(X.realization_us(86001))[::64]
    M = 1 << (12 if DEV else 15)
    R = np.broadcast_to(np.eye(3), (len(dirs), 3, 3)).copy()
    a = Q.integ_frames(st[2], dirs, R, M, 87980) / S.DOMEGA; b = Q.integ_frames(st[2], dirs, R, M, 87981) / S.DOMEGA
    c = Q.fbar(st[2], dirs, M, 87982); d = Q.fbar(st[2], dirs, M, 87983)
    e = rel(0.5 * (a + b), 0.5 * (c + d)); noise = math.sqrt((0.5 * rel(a, b)) ** 2 + (0.5 * rel(c, d)) ** 2)
    # sphere band rows: total response bounded by the material albedo (no energy gain from the frame treatment)
    us = X.realization_us(86001)
    q = Q.Quad(us, 4096 if DEV else 16384, 87984)
    tot = [float(q.W[6 + iz].sum(0).mean(-1).max()) for iz in range(S.SPH_NZ)]
    return {'rel_l1': e, 'noise': noise, 'sphere_band_max_bin_albedo': tot, 'PASS': e <= 0.01 + 4 * noise and max(tot) <= 1.05}


def t_lamb(quad):
    cosr = quad.dirs[:, 2]
    out = {}
    for m in range(6):
        if quad.lam[m]:
            want = quad.alb[m] / math.pi * (cosr * S.DOMEGA / KAPPA).sum()
            got = quad.W[m].sum(0)
            out[m] = float(np.abs(got / want - 1).max())
    quad_pi = float((cosr * S.DOMEGA / KAPPA).sum() / math.pi)
    return {'max_rel': out, 'sum_cos_domega_over_pi': quad_pi, 'PASS': max(out.values()) <= 1e-6}


def t_alb(quad):
    """Reported: 512-ray quadrature directional albedo vs ER-sampling albedo per outgoing bin (view directions uniform in the bin)."""
    st = quad.states
    out = {}
    for m in Q.ER_MATS:
        got = quad.W[m].sum(0).mean(-1)                                          # [NB] mean over rgb
        cl = CD.DrClosure83('ER', st[m])
        M = 1 << (10 if DEV else 14)
        lanes = NB * M
        lane = dr.arange(PT.U, lanes)
        rng = mi.PCG32(size=lanes, initstate=dr.opaque(mi.UInt64, 87930 + m), initseq=mi.UInt64(0))
        v = PT.Ctx.dir_in_bin(lane // M, rng.next_float32(), rng.next_float32())
        wo, pdf, w, valid = cl.sample(v, rng.next_float32(), mi.Point2f(rng.next_float32(), rng.next_float32()), v.z > 1e-6)
        w = dr.select(valid, w, 0.0)
        dr.eval(w); dr.sync_thread()
        want = np.stack([np.asarray(w[c], np.float64) for c in range(3)], 1).reshape(NB, M, 3).mean(1).mean(-1)
        r = got / np.maximum(want, 1e-12) - 1
        out[m] = {'median_abs_rel': float(np.median(np.abs(r))), 'max_abs_rel': float(np.abs(r).max()), 'mean_albedo': float(want.mean())}
    return out


def t_q():
    ctx = PT.Ctx(FX['E1']['T1']); lt = PT.Lights(FX['light_sets']['L1'])
    q = Q.qtable(ctx, lt, 256, 64, 87940, occlusion=False)
    G = ctx.G
    st, lam, alb = Q.mats()
    L = FX['light_sets']['L1'][0]
    lp = np.array(L['pos']); I = np.array(L['rgb'])
    errs = []
    g = (np.arange(64) + 0.5) / 64
    for i in range(G.N):
        m = G.PMAT[i]
        if not lam[m] or G.PSID[i] == S.SID_SPHERE:
            continue
        pts = G.PO[i][None, None] + g[:, None, None] * G.PE1[i][None, None] + g[None, :, None] * G.PE2[i][None, None]
        dl = lp - pts
        d2 = (dl ** 2).sum(-1)
        cs = np.maximum((dl * G.PN[i]).sum(-1) / np.sqrt(d2), 0)
        E = (cs / d2).mean() * I
        want = alb[m] / math.pi * E
        got = q[i].mean(0)
        if want.max() > 0:
            errs.append(float(np.abs(got / want - 1).max()))
    return {'max_rel': max(errs), 'median_rel': float(np.median(errs)), 'patches': len(errs), 'PASS': max(errs) <= 0.01}


def numpy_apply(L, hit, dbin, W, wid, rows):
    out = np.zeros((len(rows), NB, 3))
    for a, i in enumerate(rows):
        for r in range(K):
            h = hit[i * K + r]
            if h < 0:
                continue
            out[a] += W[wid[i], r] * L[h, dbin[i * K + r]][None, :]
    return out


def main():
    R87, R86 = sys.argv[1], sys.argv[2]
    only = None
    for a in sys.argv[3:]:
        if a.startswith('--only='):
            only = a.split('=')[1].split(',')
    import ecro86 as EC
    E = P87['explicit']
    T = {}
    run = lambda name: only is None or name in only
    if run('T_WL86'):
        T['T_WL86'] = t_wl86(R86); print('T_WL86', T['T_WL86'], flush=True)
    if run('T_READOUT'):
        T['T_READOUT'] = t_readout(R86); print('T_READOUT', T['T_READOUT'], flush=True)
    if run('T_SYM'):
        T['T_SYM'] = t_sym(); print('T_SYM', T['T_SYM'], flush=True)
    if run('T_FRAMES'):
        T['T_FRAMES'] = t_frames(); print('T_FRAMES', T['T_FRAMES'], flush=True)
    if run('T_FBAR'):
        T['T_FBAR'] = t_fbar(); print('T_FBAR', T['T_FBAR']['PASS'], flush=True)
    ctx = PT.Ctx(FX['E1']['T1']); lt = PT.Lights(FX['light_sets']['L1'])
    Mr = 4096 if DEV else E['M_ray']
    quad = Q.Quad(ctx.u_state_np, Mr, E['seed_ray'])
    if run('T_LAMB'):
        T['T_LAMB'] = t_lamb(quad); print('T_LAMB', T['T_LAMB'], flush=True)
    if run('T_ALB'):
        T['T_ALB'] = t_alb(quad); print('T_ALB', T['T_ALB'], flush=True)
    if run('T_Q'):
        T['T_Q'] = t_q(); print('T_Q', T['T_Q'], flush=True)
    st = EC.build_state(ctx)
    G = ctx.G
    op = X.XOp(G, st.hit, st.dbin, quad.W)
    q = Q.qtable(ctx, lt, E['P'], 128 if DEV else E['Mq'], E['seed_q'])
    if run('T_IMPL'):
        rng = np.random.default_rng(87950)
        L = rng.random((G.N, NB, 3)).astype(np.float32)
        rows = np.concatenate([rng.choice(G.N - 32, 32, replace=False), rng.choice(np.arange(G.N - 32, G.N), 8, replace=False)])     # incl. sphere
        got = op.apply(torch.as_tensor(L, device=X.dev)).cpu().numpy()[rows]
        want = numpy_apply(L.astype(np.float64), st.hit.astype(np.int64), st.dbin.astype(np.int64), quad.W, Q.Quad.wid(G), rows)
        e = float(np.abs(got - want).max() / np.abs(want).max())
        T['T_IMPL'] = {'max_rel': e, 'PASS': e <= 1e-4}; print('T_IMPL', T['T_IMPL'], flush=True)
    if run('T_ZERO'):
        q0 = Q.qtable(ctx, PT.Lights([{'pos': FX['light_sets']['L1'][0]['pos'], 'rgb': [0.0, 0.0, 0.0]}]), E['P'], 64, 1)
        B0 = X.recurrence([op], q0)[0].cpu().numpy()
        opn = X.XOp(G, -np.ones_like(st.hit), st.dbin, quad.W)
        Bn, on = X.recurrence([opn], q)
        T['T_ZERO'] = {'q_zero_light_max': float(np.abs(q0).max()), 'B_zero_light_max': float(np.abs(B0).max()), 'no_visibility_max': float(Bn.abs().max()),
                       'PASS': float(np.abs(q0).max()) == 0 and float(np.abs(B0).max()) == 0 and float(Bn.abs().max()) == 0}
        print('T_ZERO', T['T_ZERO'], flush=True)
    B, ords = X.recurrence([op], q)
    B = B.cpu().numpy()
    if run('T_LIN'):
        q2 = Q.qtable(ctx, PT.Lights(FX['light_sets']['LMIX']), E['P'], 128 if DEV else E['Mq'], E['seed_q'] + 1)
        Bq2 = X.recurrence([op], q2)[0].cpu().numpy()
        Bs = X.recurrence([op], q + q2)[0].cpu().numpy()
        B3 = X.recurrence([op], 3 * q)[0].cpu().numpy()
        a, b = rel(Bs, B + Bq2), rel(B3, 3 * B)
        T['T_LIN'] = {'superposition_rel': a, 'scaling_rel': b, 'PASS': a <= 1e-5 and b <= 1e-5}; print('T_LIN', T['T_LIN'], flush=True)
    if run('T_NONNEG'):
        T['T_NONNEG'] = {'W_min': float(quad.W.min()), 'q_min': float(q.min()), 'B_min': float(B.min()),
                         'PASS': quad.W.min() >= 0 and q.min() >= 0 and B.min() >= 0}
        print('T_NONNEG', T['T_NONNEG'], flush=True)
    if run('T_FURNACE'):
        Lr = op.apply(torch.ones(G.N, NB, 3, device=X.dev)).cpu().numpy()
        ok = (st.hit.reshape(G.N, K) >= 0).astype(np.float64)
        cosr = quad.dirs[:, 2]
        errs = []
        for i in range(G.N):
            m = G.PMAT[i]
            if quad.lam[m]:
                want = quad.alb[m] / math.pi * (ok[i] * cosr * S.DOMEGA / KAPPA).sum()
                errs.append(float(np.abs(Lr[i] / max(want, 1e-30) - 1).max()) if want > 0 else float(np.abs(Lr[i]).max()))
        T['T_FURNACE'] = {'max_rel': max(errs), 'PASS': max(errs) <= 1e-5}; print('T_FURNACE', T['T_FURNACE'], flush=True)
    if run('T_STATE'):
        c2 = PT.Ctx(FX['E1']['T0'])
        s0 = EC.build_state(c2)
        c2.set_geom(FX['E1']['T1'])
        ssel, _ = EC.update_state(c2, s0, FX['E1']['moved'])
        sfull = EC.build_state(c2)
        Bsel = X.recurrence([X.XOp(c2.G, ssel.hit, ssel.dbin, quad.W)], q)[0].cpu().numpy()
        Bful = X.recurrence([X.XOp(c2.G, sfull.hit, sfull.dbin, quad.W)], q)[0].cpu().numpy()
        c2.set_geom(FX['E1']['T0'])
        sret, _ = EC.update_state(c2, ssel, FX['E1']['moved'])
        snull, nn = EC.update_state(c2, sret, [])
        B0a = X.recurrence([X.XOp(c2.G, s0.hit, s0.dbin, quad.W)], q)[0].cpu().numpy()
        B0b = X.recurrence([X.XOp(c2.G, sret.hit, sret.dbin, quad.W)], q)[0].cpu().numpy()
        T['T_STATE'] = {'selective_equals_full_B': bool(np.array_equal(Bsel, Bful)), 'fresh_equals_selective_B': bool(np.array_equal(Bsel, B)),
                        'move_and_return_B': bool(np.array_equal(B0a, B0b)), 'null_edit_dirty_rays': int(nn),
                        'PASS': bool(np.array_equal(Bsel, Bful) and np.array_equal(Bsel, B) and np.array_equal(B0a, B0b) and nn == 0)}
        print('T_STATE', T['T_STATE'], flush=True)
    if run('T_REAL'):
        same = bool(np.array_equal(X.realization_us(86001), ctx.u_state_np))
        us = X.realization_us(87201)
        qa = Q.Quad(us, 2048, E['seed_ray']); qb = Q.Quad(us, 2048, E['seed_ray'])
        wd = float(np.abs(qa.W - qb.W).max() / np.abs(qa.W).max())          # float atomics: bit identity is not expected
        T['T_REAL'] = {'deployed_offsets_identical': same, 'W_rebuilt_max_rel_diff': wd, 'PASS': same and wd <= 1e-5}
        print('T_REAL', T['T_REAL'], flush=True)
    if run('T_RR'):
        import pt87
        spp = 128 if DEV else 2048
        a, _ = ctx.render(lt, 'ER', FX['camera'], 512, spp, 87960, 'cam', plain=True, b=32)
        b_, _ = pt87.render_rr(ctx, lt, FX['camera'], 512, spp, 87961, 3, b=32)
        pa = PT.pack(a, 512, spp)['tot'].astype(np.float64); pb = PT.pack(b_, 512, spp)['tot'].astype(np.float64)
        sys.path.insert(0, fx87.E86)
        import analyze86 as A
        M, _ = A.masks(R86, 'E1')
        objs = {k: float(pb[M[k]].sum() / pa[M[k]].sum() - 1) for k in ('obj_floor', 'obj_walls', 'obj_panel', 'obj_sphere', 'obj_ceiling')}
        fr = float(pb.sum() / pa.sum() - 1)
        T['T_RR'] = {'frame_ratio_minus_1': fr, 'object_ratio_minus_1': objs,
                     'PASS': abs(fr) <= 0.005 and max(abs(v) for v in objs.values()) <= 0.015}
        print('T_RR', T['T_RR'], flush=True)
    if run('T_SEEDS'):
        ref = set(range(86100, 86108)) | set(range(86200, 86208)) | set(range(86300, 86304)) | set(range(87100, 87108)) | set(range(87120, 87128))
        mine = {E['seed_ray'], E['seed_dense'], E['seed_q'], E['seed_c3']} | set(P87['stage2']['seeds']) | {P87['stage3']['seed_base'], P87['stage3']['tf_seed_base']}
        T['T_SEEDS'] = {'overlap': sorted(ref & mine), 'PASS': not (ref & mine)}
        print('T_SEEDS', T['T_SEEDS'], flush=True)
    if run('T_ORDER_SUM'):
        import pt87
        fo = pt87.field_orders(ctx, lt, 64 if DEV else 512, 87970)
        zo = np.load(os.path.join(R86, 'results', 'fields', 'F_E1_T1.npz'))
        orc = 0.5 * (zo['fa'].astype(np.float64) + zo['fb'])
        w = (G.PAREA[:, None] * ((np.arange(NB) // S.NPHI + 0.5) / S.NZ)[None])[..., None]
        T['T_ORDER_SUM'] = {'flux_ratio_orders_over_oracle': ((fo.sum(0) * w).sum((0, 1)) / (orc * w).sum((0, 1))).tolist(),
                            'rel_l1': rel(fo.sum(0), orc), 'order_flux_shares': [float((o * w).sum() / (fo.sum(0) * w).sum()) for o in fo]}
        print('T_ORDER_SUM', T['T_ORDER_SUM'], flush=True)
    out_dir = os.path.join(R87, 'results', 'tests'); os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, 'tests87.json')
    prev = json.load(open(fn)) if (only and os.path.exists(fn)) else {}
    prev.update(T)
    prev['PASS'] = bool(all(prev.get(g, {}).get('PASS', False) for g in GATE))
    json.dump(prev, open(fn, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print('TESTS', 'PASS' if prev['PASS'] else 'FAIL', flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
