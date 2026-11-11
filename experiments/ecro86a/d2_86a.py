"""WL86A amendment 1 diagnostics: which part of the learned mapping fails (frozen WL86 operators; diagnostic controls are not ECRO candidates).

  python d2_86a.py split <root86> <out> <cfg> [--dev]     path-traced truth split by event order at the table patch and by last sender (panel / other)
  python d2_86a.py terms <root86> <out>                   frozen operators split into the single learned flight (all senders / panel senders) and the global term
  python d2_86a.py indist <root86> <out>                  patch-level level error of the frozen operators on seen (training) and unseen (validation) geometries
  python d2_86a.py fitdata <root86> <out> [--dev]         E2 T1 geometry under 64 + 8 random training-distribution lights (diagnostic oracle data)
  python d2_86a.py fit <root86> <out> <seed_tag> [--dev]  free-factor oracle fit of the operator family on that geometry (non-deployable control)
  python d2_86a.py analyze <root86> <out>                 attribution table for the amendment
"""
import glob, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'ecro86'))
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro86a.json')))
A1 = json.load(open(os.path.join(HERE, 'protocol', 'amendment1.json')))
DEV = '--dev' in sys.argv
SEEDS = P['seeds']
PARTS = ('e2_p', 'e2_o', 'e3_p', 'e3_o')


def split(root, out, cfg):
    import pt86a as A
    PT = A.PT
    geom, ls = PT.config(cfg)
    ctx = PT.Ctx(geom); lt = PT.Lights(PT.FX['light_sets'][ls])
    spp = 64 if DEV else A1['split']['spp_per_half']
    d = {}
    for h, sd in (('a', 87400), ('b', 87405)):
        t = A.field_src(ctx, A.Src(lt, 'true', ()), spp, sd, split=True)
        for k, v in t.items():
            d[f'{k}_{h}'] = v
    np.savez_compressed(os.path.join(out, 'results', 'fields', f'D_SPLIT_{cfg}.npz'), **d)
    print('split', cfg, {k: float(v.sum()) for k, v in d.items()}, flush=True)


def terms(root, out):
    import torch
    import pt86a as A
    import ecro86 as EC
    import model86 as M
    PT = A.PT
    NB, K, KAPPA = PT.NB, PT.K, PT.KAPPA
    for sd in SEEDS:
        mdl = EC.load_model(root, sd)
        for cfg in P['configs']:
            geom, ls = PT.config(cfg)
            ctx = PT.Ctx(geom)
            st = EC.build_state(ctx)
            s = np.load(os.path.join(out, 'results', 'fields', f'S_{cfg}.npz'))['s16']
            B, f = EC.respond(mdl, st, s)
            Bs, _ = EC.respond(mdl, st, s, use_global=False, factors=f)
            Bg, _ = EC.respond(mdl, st, s, use_single=False, factors=f)
            with torch.no_grad():
                gs = M.GeoState(st.G, st.hit, st.dbin, EC.dev)
                N = gs.N
                st_ = torch.as_tensor(s, device=EC.dev).view(1, N, NB, 3)
                u = torch.einsum('nmcr,lnmc->lncr', f['c'], st_)
                Bq = torch.einsum('ndcr,lncr->lndc', f['a'], u).reshape(1, N * NB, 3)
                okp = ((gs.hit >= 0) & (gs.mat[gs.hit.clamp(min=0)] == 1)).view(1, N * K, 1)
                idx = (gs.hit.clamp(min=0) * NB + gs.dbin).view(-1)
                E1 = ((Bq[:, idx] * okp).view(1, N, NB, KAPPA, 3) * mdl.rw.view(1, 1, NB, KAPPA, 1)).sum(3)
                Bsp = torch.einsum('ndcr,lncr->lndc', f['a'], torch.einsum('nmcr,lnmc->lncr', f['c'], E1))[0].cpu().numpy()
            np.savez_compressed(os.path.join(out, 'results', 'fields', f'M2_{sd}_{cfg}.npz'), B=B, single=Bs, glob=Bg, single_panel=Bsp)
            print('terms', sd, cfg, float(np.abs(B - Bs - Bg).max() / B.max()), flush=True)


def indist(root, out):
    import torch
    import ecro86 as EC
    import train86 as TR
    import model86 as M
    import scene86 as S
    rw = M.ray_weights() / math.pi
    pc = np.array(S.FX['panel']['center'])
    res = {}
    for sd in SEEDS:
        mdl = EC.load_model(root, sd)
        for split_ in ('trainhq', 'val'):
            items = TR.load(root, split_)
            rows = []
            with torch.no_grad():
                for it in items:
                    p = mdl(it.gs(), torch.as_tensor(it.s, device=EC.dev)).cpu().numpy().astype(np.float64)
                    fa = it.f.astype(np.float64); fb = it.fb.astype(np.float64)
                    pmat = it.tab['pmat']
                    hp = (it.hit >= 0) & (pmat[np.clip(it.hit, 0, None)] == 1)
                    fpan = (hp.reshape(len(pmat), -1) * rw[None]).sum(1)
                    for l in range(p.shape[0]):
                        tp = 0.5 * (fa[l] + fb[l]).sum((1, 2)); pp = p[l].sum((1, 2)); na = fa[l].sum((1, 2)); nb = fb[l].sum((1, 2))
                        rows.append(np.stack([pp, tp, na, nb, fpan, pmat.astype(np.float64)], 1))
            X = np.concatenate(rows)                       # [lightings x patches, 6]
            broad = np.isin(X[:, 5], [0, 3, 4, 5]) & (X[:, 1] > 0)
            o = {}
            for name, lo, hi in (('f<0.01', -1, 0.01), ('0.01-0.03', 0.01, 0.03), ('0.03-0.10', 0.03, 0.10), ('>=0.10', 0.10, 9)):
                m = broad & (X[:, 4] >= lo) & (X[:, 4] < hi)
                if m.sum() < 50:
                    continue
                r = X[m, 0] / X[m, 1] - 1.0
                cvn = (X[m, 0] - X[m, 2]) * (X[m, 0] - X[m, 3]) / (X[m, 2] * X[m, 3])      # noise-debiased squared relative error
                o[name] = {'n': int(m.sum()), 'mean_abs_level_error': float(np.abs(r).mean()), 'mean_level_error': float(r.mean()),
                           'rms_level_error_debiased': float(math.sqrt(max(np.mean(cvn), 0.0))),
                           'target_noise_patch_level': float((0.5 * np.abs(X[m, 2] - X[m, 3]) / X[m, 1]).mean())}
            res[f'{sd}:{split_}'] = o
            print(sd, split_, json.dumps(o), flush=True)
    json.dump(res, open(os.path.join(out, 'results', 'indist.json'), 'w'), indent=1)


def fitdata(root, out):
    import pt86a as A
    import scene86 as S
    PT = A.PT
    geom, _ = PT.config('E2_T1')
    ctx = PT.Ctx(geom)
    rng = np.random.default_rng(A1['fit']['light_seed'])
    nf, nh = (8, 2) if DEV else (A1['fit']['fit_lights'], A1['fit']['heldout_lights'])
    lights = [S.sample_light(rng, geom) for _ in range(nf + nh)]
    hit, bh, seg = ctx.state_rays()
    s, f, fb = [], [], []
    for i, l in enumerate(lights):
        lt = PT.Lights(l)
        s.append(ctx.source(lt))
        if i < nf:
            f.append(ctx.field(lt, 64 if DEV else A1['fit']['fit_spp'], 87500 + i, b=128)[0])
        else:
            hs = 64 if DEV else A1['fit']['heldout_spp_per_half']
            f.append(ctx.field(lt, hs, 87500 + i, b=128)[0]); fb.append(ctx.field(lt, hs, 87700 + i, b=128)[0])
    np.savez_compressed(os.path.join(out, 'results', 'fitdata_E2_T1.npz'), lights=json.dumps(lights), hit=hit, dbin=bh, s=np.stack(s),
                        f_fit=np.stack(f[:nf]), f_ho_a=np.stack(f[nf:]), f_ho_b=np.stack(fb))
    print('fitdata', len(lights), flush=True)


def fit(root, out, sd):
    import torch
    import torch.nn.functional as Fn
    import pt86a as A
    import ecro86 as EC
    import model86 as M
    import scene86 as S
    NB = S.NB
    dev = EC.dev
    torch.manual_seed(0)
    mdl = EC.load_model(root, sd)
    z = np.load(os.path.join(out, 'results', 'fitdata_E2_T1.npz'))
    G = S.Geom(S.FX['E2']['T1'])
    gs = M.GeoState(G, z['hit'], z['dbin'], dev)
    with torch.no_grad():
        f0 = mdl.factors(gs)
    T = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=dev)
    s_fit, t_fit = T(z['s'][:z['f_fit'].shape[0]]), T(z['f_fit'])
    s_ho = T(z['s'][z['f_fit'].shape[0]:]); ta, tb = T(z['f_ho_a']), T(z['f_ho_b'])
    sz = np.load(os.path.join(out, 'results', 'fields', 'S_E2_T1.npz'))
    s_test = T(sz['s16'][None])
    zt = np.load(os.path.join(root, 'results', 'fields', 'F_E2_T1.npz'))
    t_test = 0.5 * (zt['fa'].astype(np.float64) + zt['fb'])
    W = np.load(os.path.join(out, 'stageB', 'importance_E2_T1.npz'))
    res = {}

    def ev(f):
        with torch.no_grad():
            ph = mdl.apply(f, gs, s_ho)
            e = {'heldout_rel_l1': float((ph - 0.5 * (ta + tb)).abs().sum() / (0.5 * (ta + tb)).sum()),
                 'heldout_rel_rms_debiased': float(torch.sqrt(((ph - ta) * (ph - tb)).sum().clamp(min=0) / (ta * tb).sum())),
                 'fit_rel_l1': float((mdl.apply(f, gs, s_fit[:16]) - t_fit[:16]).abs().sum() / t_fit[:16].sum())}
            Bt = mdl.apply(f, gs, s_test)[0].cpu().numpy().astype(np.float64)
            Bh = ph.cpu().numpy().astype(np.float64); th = (0.5 * (ta + tb)).cpu().numpy().astype(np.float64)
            for rk in ('C', 'obj_panel', 'D'):
                Wr = W[rk].astype(np.float64).reshape(G.N, NB, 3)
                e[f'P_test_L2_{rk}_over_h'] = float((Wr * (Bt - t_test)).sum() / (Wr * t_test).sum())
                e[f'P_heldout_{rk}_over_h'] = [float((Wr * (Bh[i] - th[i])).sum() / (Wr * th[i]).sum()) for i in range(Bh.shape[0])]
            return e, Bt
    res['frozen'], B0 = ev(f0)
    isp = lambda y: torch.log(torch.expm1(y.clamp(min=1e-6)))
    steps = 100 if DEV else A1['fit']['steps']
    for variant, free in (('PQ', ('P', 'Q')), ('ALL', ('c', 'a', 'P', 'Q'))):
        raw = {k: isp(v.detach().clone()).requires_grad_(k in free) for k, v in f0.items()}
        opt = torch.optim.Adam([raw[k] for k in free], lr=A1['fit']['lr'])
        rng = np.random.default_rng(1)
        t0 = time.time()
        for i in range(steps):
            b = torch.as_tensor(rng.choice(s_fit.shape[0], min(8, s_fit.shape[0]), replace=False), device=dev)
            f = {k: Fn.softplus(v) for k, v in raw.items()}
            loss = M.loss_fn(mdl.apply(f, gs, s_fit[b]), t_fit[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        f = {k: Fn.softplus(v).detach() for k, v in raw.items()}
        res[variant], Bt = ev(f)
        res[variant]['seconds'] = time.time() - t0; res[variant]['final_loss'] = float(loss)
        np.save(os.path.join(out, 'results', f'fit_{sd}_{variant}_E2_T1.npy'), Bt.astype(np.float32))
        print(sd, variant, json.dumps({k: v for k, v in res[variant].items() if not isinstance(v, list)}), flush=True)
    lights = json.loads(str(z['lights']))
    pc = np.array(S.FX['panel']['center'])
    res['heldout_light_panel_distance'] = [float(np.linalg.norm(np.array(l[0]['pos']) - pc)) for l in lights[z['f_fit'].shape[0]:]]
    res['heldout_light_pos'] = [l[0]['pos'] for l in lights[z['f_fit'].shape[0]:]]
    json.dump(res, open(os.path.join(out, 'results', f'fit_{sd}.json'), 'w'), indent=1)
    print(sd, 'frozen', json.dumps({k: v for k, v in res['frozen'].items() if not isinstance(v, list)}), flush=True)


def analyze(root, out):
    import scene86 as S
    NB = S.NB
    R = os.path.join(out, 'results')
    A0 = json.load(open(os.path.join(R, 'attribution.json')))
    res = {'terms': {}, 'fit': {}, 'indist': json.load(open(os.path.join(R, 'indist.json')))}
    prim = [tuple(c) for c in P['cases']['primary']]
    for cfg, rk in prim + [tuple(c) for c in P['cases']['controls']]:
        W = np.load(os.path.join(out, 'stageB', f'importance_{cfg}.npz'))[rk].astype(np.float64).reshape(-1, NB, 3)
        z = np.load(os.path.join(R, 'fields', f'D_SPLIT_{cfg}.npz'))
        T = {k: 0.5 * (z[f'{k}_a'].astype(np.float64) + z[f'{k}_b']) for k in PARTS}
        Th = {h: sum(z[f'{k}_{h}'].astype(np.float64) for k in PARTS) for h in 'ab'}
        zo = np.load(os.path.join(root, 'results', 'fields', f'F_{cfg}.npz')); Bo = 0.5 * (zo['fa'].astype(np.float64) + zo['fb'])
        base = (W * Bo).sum()
        ts = base / A0['cases'][f'{cfg}:{rk}']['H_share']                      # = sum tot over the region
        row = {'truth_share_read': {k: float((W * T[k]).sum() / (W * sum(T.values())).sum()) for k in PARTS},
               'split_sum_vs_stored_over_h': float((W * (sum(T.values()) - Bo)).sum() / base),
               'split_halves_over_h': [float((W * Th[h]).sum() / base - 1) for h in 'ab'], 'seeds': {}}
        for sd in SEEDS:
            m = np.load(os.path.join(R, 'fields', f'M2_{sd}_{cfg}.npz'))
            g = lambda X, Y: float((W * (X.astype(np.float64) - Y)).sum() / ts)
            row['seeds'][sd] = {'P_total': g(m['B'], Bo), 'P_single_vs_e2': g(m['single'], T['e2_p'] + T['e2_o']), 'P_global_vs_e3': g(m['glob'], T['e3_p'] + T['e3_o']),
                                'P_single_panel_vs_e2p': g(m['single_panel'], T['e2_p']), 'P_single_other_vs_e2o': g(m['single'] - m['single_panel'], T['e2_o']),
                                'model_share_read': {'single': float((W * m['single']).sum() / (W * m['B']).sum()), 'glob': float((W * m['glob']).sum() / (W * m['B']).sum()),
                                                     'single_panel': float((W * m['single_panel']).sum() / (W * m['B']).sum())}}
        res['terms'][f'{cfg}:{rk}'] = row
    for sd in SEEDS:
        res['fit'][sd] = json.load(open(os.path.join(R, f'fit_{sd}.json')))
    # ---- classification (amendment 1 rules)
    cl = {}
    keys = [f'{c}:{r}' for c, r in prim]
    gl = []; sg = []
    for k in keys:
        for sd in SEEDS:
            t = res['terms'][k]['seeds'][sd]
            gl.append(t['P_global_vs_e3'] / t['P_total']); sg.append(t['P_single_vs_e2'] / t['P_total'])
    gm = [float(np.mean(gl[2 * i:2 * i + 2])) for i in range(3)]; sm = [float(np.mean(sg[2 * i:2 * i + 2])) for i in range(3)]
    cl['term_localisation'] = {'global_share_of_error': gm, 'single_share_of_error': sm,
                               'label': 'GLOBAL TERM' if sum(v >= 2 / 3 for v in gm) >= 2 else ('SINGLE FLIGHT' if sum(v >= 2 / 3 for v in sm) >= 2 else 'MIXED')}
    ind = res['indist']
    rat = []
    for sd in SEEDS:
        for b in ('0.03-0.10', '>=0.10'):
            a, v = ind[f'{sd}:trainhq'].get(b), ind[f'{sd}:val'].get(b)
            if a and v:
                rat.append(a['rms_level_error_debiased'] / max(v['rms_level_error_debiased'], 1e-30))
    cl['coverage_generalisation'] = {'seen_over_unseen_rms_level_error': rat,
                                     'label': 'WEAKENED' if rat and min(rat) >= 2 / 3 else ('SUPPORTED' if rat and max(rat) < 1 / 3 else 'INDETERMINATE')}
    cap = {}
    for v in ('PQ', 'ALL'):
        r_ = [abs(res['fit'][sd][v]['P_test_L2_C_over_h']) / max(abs(res['fit'][sd]['frozen']['P_test_L2_C_over_h']), 1e-30) for sd in SEEDS]
        gen = [res['fit'][sd][v]['heldout_rel_l1'] / max(res['fit'][sd][v]['fit_rel_l1'], 1e-30) for sd in SEEDS]
        cap[v] = {'residual_over_frozen': r_, 'heldout_over_fit_field_error': gen,
                  'label': 'SUFFICIENT' if max(r_) <= 1 / 3 else ('INSUFFICIENT' if min(r_) >= 2 / 3 else 'INDETERMINATE')}
    cl['operator_family_capacity'] = cap
    res['classification'] = cl
    json.dump(res, open(os.path.join(R, 'attribution2.json'), 'w'), indent=1)
    r4 = lambda v: round(v, 4)
    for k, row in res['terms'].items():
        print(k, 'truth read', {a: r4(b) for a, b in row['truth_share_read'].items()}, 'split-vs-stored', r4(row['split_sum_vs_stored_over_h']))
        for sd in SEEDS:
            t = row['seeds'][sd]
            print('   ', sd, {a: (r4(b) if not isinstance(b, dict) else {x: r4(y) for x, y in b.items()}) for a, b in t.items()})
    for sd in SEEDS:
        f = res['fit'][sd]
        print('fit', sd, {v: {a: r4(b) for a, b in f[v].items() if not isinstance(b, list)} for v in ('frozen', 'PQ', 'ALL')})
        print('   heldout P_C frozen', [r4(x) for x in f['frozen']['P_heldout_C_over_h']], 'ALL', [r4(x) for x in f['ALL']['P_heldout_C_over_h']], 'dist', [round(x, 2) for x in f['heldout_light_panel_distance']])
    print(json.dumps(cl))


if __name__ == '__main__':
    cmd, root, out = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(os.path.join(out, 'results', 'fields'), exist_ok=True)
    if cmd == 'split':
        split(root, out, sys.argv[4])
    elif cmd == 'fit':
        fit(root, out, sys.argv[4])
    else:
        {'terms': terms, 'indist': indist, 'fitdata': fitdata, 'analyze': analyze}[cmd](root, out)
    os._exit(0)
