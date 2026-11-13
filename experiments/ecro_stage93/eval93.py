"""WL93 evaluation: identifiability gate, synthetic tests, per-configuration tables and local diagnostics, local consistency, timing.

  python eval93.py ident  <R> <R86> <R92A>            source-stage identifiability gate (STOP_TARGET_NOT_IDENTIFIABLE)
  python eval93.py tests  <R> <R86> <R92A> <pre|post> synthetic tests
  python eval93.py tables <R> <R86> <R92A> <tag>      tables + diagnostics of every arm (and WL92A sum / own context) for a configuration
  python eval93.py local  <R> <R86> <R92A>            panel matrices across E2 T0 / T1 and seeds
  python eval93.py ofield <R> <R86> <tag>             oracle field of an E6 configuration
  python eval93.py timing <R> <R86> <R92A>            parameters, FLOPs and warm inference time per arm (E6 T1)
"""
import glob, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('ecro86', 'ecro_attr92', 'ecro_own92a', 'ecro_cf92b'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
sys.path.insert(0, HERE)
DEV = '--dev' in sys.argv
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_stage93.json')))
ARMS = list(P['arms'])
SEEDS = P['training']['seeds']


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1)


def load_arm(R, R92A, arm, seed, trained=True):
    import torch
    import model93 as MM
    import train93 as TR
    dev = torch.device('cuda')
    own = TR.load_own(R92A, seed, dev)
    net = MM.Local(own, P['arms'][arm]['mode'], cm_seed=seed + 9300).to(dev)
    if trained:
        net.load_state_dict(torch.load(os.path.join(R, 'results', 'models', f'{arm}_s{seed}.pt'), map_location=dev))
    return net.eval(), own


def single_panel(net, f, gs, s):
    import torch
    import scene86 as S
    NB, K, KAPPA = S.NB, S.K, S.KAPPA
    Ls, N = s.shape[0], gs.N
    Bq = torch.einsum('ndcr,lncr->lndc', f['a_s'], torch.einsum('nmcr,lnmc->lncr', f['c_s'], s))
    okp = ((gs.hit >= 0) & (gs.mat[gs.hit.clamp(min=0)] == 1)).view(1, N * K, 1)
    idx = (gs.hit.clamp(min=0) * NB + gs.dbin).view(-1)
    E1 = ((Bq.reshape(Ls, N * NB, 3)[:, idx] * okp).view(Ls, N, NB, KAPPA, 3) * net.g.rw.view(1, 1, NB, KAPPA, 1)).sum(3)
    return torch.einsum('ndcr,lncr->lndc', f['a_r'], torch.einsum('nmcr,lnmc->lncr', f['c_r'], E1))


def pmetrics(Mf, Mq, PM, zb):
    """panel (material 1) matrix metrics of factor dict side ('s' or 'r') against frozen ER, flux norm."""
    import diag92 as DG
    sel = np.nonzero(PM == 1)[0]
    rows = [DG.mat_metrics(Mf[i, :, :, c], Mq[1, :, :, c], zb) for i in sel for c in range(3)]
    return {k: float(np.mean([r[k] for r in rows])) for k in ('fro', 'peak', 'mass', 'l1')}


# ------------------------------------------------------------------------------------------------ identifiability gate
def ident(R, R86, R92A):
    import torch
    import fx92
    import diag92 as DG
    import xport87 as X
    import train93 as TR
    _, Mq = DG.local_matrices()
    dev = torch.device('cuda')
    val = TR.population(R, R86, R92A, 'val', 8 if DEV else None)
    res = {'val': {}, 'E2': {}}
    for seed in SEEDS:
        own = TR.load_own(R92A, seed, dev)
        a = {'enc_n': 0.0, 'own_n': 0.0, 'd': 0.0}
        with torch.no_grad():
            for it in val:
                gs = it.it.gs(); s, t, t2, q = it.tens(dev)
                f = own.factors(gs)
                Bq = torch.einsum('ndcr,lncr->lndc', f['a'], torch.einsum('nmcr,lnmc->lncr', f['c'], s)).double().cpu().numpy()
                G = type('G', (), {'PMAT': it.it.tab['pmat'], 'N': len(it.it.tab['pmat'])})()
                pm = it.it.tab['pmat'] == 1
                qn = q.double().cpu().numpy(); sn = s.double().cpu().numpy()
                for l in range(sn.shape[0]):
                    eq = X.enc_q(G, sn[l], Mq)
                    a['enc_n'] += float(np.abs(eq[pm] - qn[l][pm]).sum()); a['own_n'] += float(np.abs(Bq[l][pm] - qn[l][pm]).sum()); a['d'] += float(qn[l][pm].sum())
        res['val'][seed] = {'irreducible_panel_rel_l1': a['enc_n'] / a['d'], 'own_learned_panel_rel_l1': a['own_n'] / a['d']}
    for tag in ('E2_T1', 'E2_T0'):
        m = json.load(open(os.path.join(R, 'results', 'cf92b', f'CF_{tag}.json')))['field']
        res['E2'][tag] = {'irreducible_panel_rel_l1': m['own_s8601']['encq_vs_q_panel_rel_l1'], 'own_learned_panel_rel_l1': {n: m[n]['Bq_vs_q_panel_rel_l1'] for n in ('own_s8601', 'own_s8602')}}
    ok = all(v['irreducible_panel_rel_l1'] <= 0.5 * v['own_learned_panel_rel_l1'] for v in res['val'].values()) and \
        all(v['irreducible_panel_rel_l1'] <= 0.5 * min(v['own_learned_panel_rel_l1'].values()) for v in res['E2'].values())
    res['PASS'] = bool(ok)
    jd(res, os.path.join(R, 'results', 'ident.json'))
    print('IDENT', 'PASS' if ok else 'FAIL', json.dumps(res['val']), flush=True)


# ------------------------------------------------------------------------------------------------ synthetic tests
def tests(R, R86, R92A, phase):
    import torch
    import fx92
    import ecro86 as EC
    import model86 as M
    import model93 as MM
    import train93 as TR
    PT, FX, S = fx92.PT, fx92.FX, fx92.S
    dev = torch.device('cuda')
    geom, ls, e, cfg = fx92.cfg_ls('E2_T1')
    ctx = PT.Ctx(geom); st = EC.build_state(ctx)
    s = torch.as_tensor(ctx.source(PT.Lights(FX['light_sets'][ls])), device=dev).view(1, ctx.N, S.NB, 3)
    gs = M.GeoState(ctx.G, st.hit, st.dbin, dev)
    out = {}
    with torch.no_grad():
        for seed in SEEDS:
            own = TR.load_own(R92A, seed, dev)
            ref = own(gs, s)
            G0 = MM.global_term(own, gs, s)
            for arm in ARMS:
                trained = phase == 'post'
                net, _ = load_arm(R, R92A, arm, seed, trained=trained)
                f = net.factors(gs)
                S_, Bq, _ = net.single(f, gs, s)
                G = MM.global_term(own, gs, s)
                key = f'{arm}_s{seed}'
                o = {}
                if phase == 'pre':
                    o['init_equiv'] = float((S_ + G - ref).abs().max() / ref.abs().max())
                o['global_identity'] = float((G - G0).abs().max())
                tied = P['arms'][arm]['mode'] != 'untied'
                o['tied_ok'] = bool((f['c_s'] is f['c_r'] and f['a_s'] is f['a_r']) if tied else (f['c_s'] is not f['c_r'] and f['a_s'] is not f['a_r']))
                S2, _, _ = net.single(f, gs, 2 * s); S3, _, _ = net.single(f, gs, s + 0.5 * s)
                o['scaling'] = float((S2 - 2 * S_).abs().max() / S_.abs().max()); o['superposition'] = float((S3 - 1.5 * S_).abs().max() / S_.abs().max())
                o['zero'] = float(net.single(f, gs, torch.zeros_like(s))[0].abs().max()); o['min'] = float(min(S_.min(), Bq.min()))
                o['det'] = float((net.single(net.factors(gs), gs, s)[0] - S_).abs().max())
                o['n_trainable'] = net.n_trainable()
                o['global_heads_frozen'] = not any(p.requires_grad for p in list(net.g.head_Q.parameters()) + list(net.g.head_P.parameters()))
                o['base_frozen'] = not any(p.requires_grad for p in own.parameters())
                out[key] = o
    # gradient ownership (untied): a source-only loss must not move the receiver heads
    seed = SEEDS[0]
    net, own = load_arm(R, R92A, 'U0', seed, trained=False)
    net.train()
    f = net.factors(gs)
    Bq = torch.einsum('ndcr,lncr->lndc', f['a_s'], torch.einsum('nmcr,lnmc->lncr', f['c_s'], s))
    Bq.sum().backward()
    gr = sum(float(p.grad.abs().sum()) for p in list(net.head_c2.parameters()) + list(net.head_a2.parameters()) if p.grad is not None)
    gsrc = sum(float(p.grad.abs().sum()) for p in list(net.g.head_c.parameters()) + list(net.g.head_a.parameters()) if p.grad is not None)
    out['grad_ownership_untied'] = {'receiver_head_grad': gr, 'source_head_grad': gsrc}
    # q target construction: WL87 qtable with the WL87 seed reproduces the stored D q of E2 T1
    if phase == 'pre':
        import quad87 as Q
        E87 = fx92.E87
        qq = Q.qtable(ctx, PT.Lights(FX['light_sets'][ls]), E87['P'], E87['Mq'], E87['seed_q'])
        Dq = np.load(os.path.join(R, 'results', 'fields', 'D_E2_T1.npz'))['q'].astype(np.float64)
        out['qtarget'] = {'max_rel': float(np.abs(qq - Dq).max() / Dq.max())}
    v = [x for k, x in out.items() if k.endswith(tuple(f's{sd}' for sd in SEEDS))]
    ok = all(x['global_identity'] == 0.0 and x['tied_ok'] and x['scaling'] <= 1e-4 and x['superposition'] <= 1e-4 and x['zero'] == 0.0 and x['min'] >= 0 and x['det'] == 0.0 for x in v)
    ok &= out['grad_ownership_untied']['receiver_head_grad'] == 0.0 and out['grad_ownership_untied']['source_head_grad'] > 0
    ok &= all(x['n_trainable'] > 0 and x['global_heads_frozen'] and x['base_frozen'] for x in v)
    if phase == 'post':                                   # non-degeneracy of the controls: trained tiedcm / untied arms differ from the tied arm
        nd = {}
        with torch.no_grad():
            for seed in SEEDS:
                ref_S = None
                for arm in ('T0', 'C0', 'U0'):
                    net, _ = load_arm(R, R92A, arm, seed)
                    S_ = net.single(net.factors(gs), gs, s)[0]
                    if ref_S is None:
                        ref_S = S_
                    else:
                        nd[f'{arm}_vs_T0_s{seed}'] = float((S_ - ref_S).abs().max() / ref_S.abs().max())
        out['nondegenerate'] = nd
        ok &= all(x > 1e-4 for x in nd.values())
    nt = {a: out[f'{a}_s{SEEDS[0]}']['n_trainable'] for a in ARMS}
    out['capacity'] = {'n_trainable': nt, 'untied_eq_tiedcm': nt['U0'] == nt['C0'] == nt['UQ'] == nt['CQ'], 'tied_eq': nt['T0'] == nt['TQ']}
    ok &= out['capacity']['untied_eq_tiedcm'] and out['capacity']['tied_eq'] and nt['U0'] > nt['T0']
    if phase == 'pre':
        ok &= all(x['init_equiv'] <= 1e-5 for x in v) and out['qtarget']['max_rel'] <= 1e-4
    out['PASS'] = bool(ok)
    jd(out, os.path.join(R, 'results', f'tests93_{phase}.json'))
    print('TESTS', phase, 'PASS' if ok else 'FAIL', json.dumps({k: x for k, x in list(out.items())[:2]})[:400], flush=True)


# ------------------------------------------------------------------------------------------------ tables + diagnostics
def tables(R, R86, R92A, tag):
    import torch
    import prep93 as PR
    fx92 = PR.inject(R)
    import ecro86 as EC
    import model86 as M
    import model93 as MM
    import diag92 as DG
    import cf92b as CB
    PT, FX, S, E87 = fx92.PT, fx92.FX, fx92.S, fx92.E87
    dev = torch.device('cuda')
    geom, ls, e, cfg = fx92.cfg_ls(tag)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls]); G_ = ctx.G
    st = EC.build_state(ctx)
    s = torch.as_tensor(ctx.source(lt), device=dev).view(1, ctx.N, S.NB, 3)
    gs = M.GeoState(G_, st.hit, st.dbin, dev)
    _, Mq = DG.local_matrices(); zb = DG.flux_w()
    W2 = (zb[:, None] * zb[None, :])[None, :, :, None]
    Mn = torch.as_tensor(Mq[G_.PMAT].astype(np.float32), device=dev)
    fD = os.path.join(R, 'results', 'fields', f'D_{tag}.npz'); fW = os.path.join(R, 'results', 'fields', f'W_{tag}.npz')
    D = np.load(fD) if os.path.exists(fD) else None; Wz = np.load(fW) if os.path.exists(fW) else None
    if D is not None:
        assert np.array_equal(st.hit, D['hit']), 'state mismatch'
        L2T, L3T, E2P = CB.hm(D, ('e2_p', 'e2_o')), CB.hm(D, ('e3_p', 'e3_o')), CB.hm(D, ('e2_p',)); qn = D['q'].astype(np.float64)
    out = {}; diag = {}
    pm = G_.PMAT == 1
    with torch.no_grad():
        for seed in SEEDS:
            for arm in ARMS:
                if not os.path.exists(os.path.join(R, 'results', 'models', f'{arm}_s{seed}.pt')):
                    continue
                net, own = load_arm(R, R92A, arm, seed)
                f = net.factors(gs)
                S_, Bq, E1 = net.single(f, gs, s)
                G = MM.global_term(own, gs, s)
                k = f'{arm}_s{seed}'
                out[f'B_{k}'] = (S_ + G)[0].cpu().numpy(); out[f'S_{k}'] = S_[0].cpu().numpy(); out[f'G_{k}'] = G[0].cpu().numpy()
                out[f'SP_{k}'] = single_panel(net, f, gs, s)[0].cpu().numpy(); out[f'Bq_{k}'] = Bq[0].cpu().numpy()
                Ms = (torch.einsum('ndcr,nmcr->nmdc', f['a_s'], f['c_s']).double().cpu().numpy())
                Mr = (torch.einsum('ndcr,nmcr->nmdc', f['a_r'], f['c_r']).double().cpu().numpy())
                d = {'source_matrix_panel': pmetrics(Ms, Mq, G_.PMAT, zb), 'receiver_matrix_panel': pmetrics(Mr, Mq, G_.PMAT, zb),
                     'source_minus_receiver_panel_rel': float(np.mean([np.linalg.norm((Ms[i] - Mr[i]) * W2[0]) / max(np.linalg.norm(Mr[i] * W2[0]), 1e-30) for i in np.nonzero(pm)[0]]))}
                Rx = MM_exact(Mn, E1)[0].cpu().numpy(); Sn = out[f'S_{k}'].astype(np.float64)
                d['receiver_stage_vs_M64_rel_l1'] = float(np.abs(Sn - Rx).sum() / max(Rx.sum(), 1e-30))
                if D is not None:
                    Bqn = out[f'Bq_{k}'].astype(np.float64)
                    d.update({'Bq_vs_q_rel_l1': float(np.abs(Bqn - qn).sum() / qn.sum()), 'Bq_vs_q_panel_rel_l1': float(np.abs(Bqn[pm] - qn[pm]).sum() / qn[pm].sum()),
                              'Bq_vs_q_panel_flux': float(Bqn[pm].sum() / qn[pm].sum()), 'single_vs_L2_rel_l1': float(np.abs(Sn - L2T).sum() / L2T.sum())})
                    if Wz is not None and 'C' in Wz.files:
                        Wr = Wz['C'].astype(np.float64).reshape(G_.N, -1, 3)
                        rd = CB.reading(Wz['C'], Sn, L2T, G_)
                        d['C_reading'] = {'single_level': rd['level'], 'single_directional': rd['directional'], 'total': CB.reading(Wz['C'], out[f'B_{k}'].astype(np.float64), L2T + L3T, G_)['signed'],
                                          'panel_family_delivered': float((Wr * out[f'SP_{k}']).sum()) / max(float((Wr * E2P).sum()), 1e-30)}
                diag[k] = d
            for base in ('own', 'sum'):                       # WL92A context (and T_REPRO: own on E2 re-rendered vs WL92A's own renders)
                m = M.ECRO().to(dev); m.load_state_dict(torch.load(os.path.join(R92A, 'results', 'models', f'{base}_s{seed}.pt'), map_location=dev)); m.eval()
                out[f'B_{base}92a_s{seed}'] = m(gs, s)[0].cpu().numpy()
    if tag.startswith('E6') and tag.endswith(('_T0', '_T1')):
        import quad87 as Q
        import xport87 as X
        F = np.load(os.path.join(R, 'results', 'fields', f'F_{tag}.npz'))
        out['ORACLE'] = (0.5 * (F['fa'].astype(np.float64) + F['fb'])).astype(np.float32)
        quad = Q.Quad(X.realization_us(86001), E87['M_ray'], E87['seed_ray'])
        Bx, _, _, _, _ = X.build(ctx, lt, quad, E87['P'], E87['Mq'], E87['seed_q'], st=st)
        out['X_EXP'] = Bx.astype(np.float32)
    np.savez_compressed(os.path.join(R, 'results', 'tables', f'I_{tag}.npz'), **{k: np.asarray(v, np.float32) for k, v in out.items()})
    jd(diag, os.path.join(R, 'results', 'tables', f'I_{tag}.json'))
    print('tables', tag, json.dumps({k: round(v.get('Bq_vs_q_panel_rel_l1', -1), 3) for k, v in diag.items()}), flush=True)


def MM_exact(Mn, E):
    import torch
    return torch.einsum('lnmc,nmdc->lndc', E, Mn)


def local(R, R86, R92A):
    import torch
    import fx92
    import ecro86 as EC
    import model86 as M
    import diag92 as DG
    PT, FX, S = fx92.PT, fx92.FX, fx92.S
    zb = DG.flux_w(); W2 = (zb[:, None] * zb[None, :])[None, :, :, None]
    Ms = {}
    for tag in ('E2_T0', 'E2_T1'):
        geom, ls, e, cfg = fx92.cfg_ls(tag)
        ctx = PT.Ctx(geom); st = EC.build_state(ctx)
        gs = M.GeoState(ctx.G, st.hit, st.dbin, EC.dev)
        PM = ctx.G.PMAT
        for seed in SEEDS:
            for arm in ARMS:
                net, _ = load_arm(R, R92A, arm, seed)
                with torch.no_grad():
                    f = net.factors(gs)
                    for side in ('s', 'r'):
                        Ms[(tag, arm, seed, side)] = torch.einsum('ndcr,nmcr->nmdc', f[f'a_{side}'], f[f'c_{side}']).double().cpu().numpy() * W2
    sel = np.nonzero(PM == 1)[0]
    rel = lambda a, b: float(np.mean([np.linalg.norm(a[i] - b[i]) / max(np.linalg.norm(b[i]), 1e-30) for i in sel]))
    res = {'geometry_variation_panel': {f'{arm}_s{seed}': {side: rel(Ms[('E2_T1', arm, seed, side)], Ms[('E2_T0', arm, seed, side)]) for side in ('s', 'r')} for arm in ARMS for seed in SEEDS},
           'seed_variation_panel_E2_T1': {arm: {side: rel(Ms[('E2_T1', arm, SEEDS[1], side)], Ms[('E2_T1', arm, SEEDS[0], side)]) for side in ('s', 'r')} for arm in ARMS}}
    jd(res, os.path.join(R, 'results', 'local93.json'))
    print('LOCAL', json.dumps(res['geometry_variation_panel'])[:400], flush=True)


def ofield(R, R86, tag):
    import prep93 as PR
    fx92 = PR.inject(R)
    PT, FX = fx92.PT, fx92.FX
    geom, ls, e, cfg = fx92.cfg_ls(tag)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    o = P['e6']['references']['oracle_field']
    spp = 64 if DEV else o['spp']
    fa, _ = ctx.field(lt, spp // 2, o['seed']); fb, _ = ctx.field(lt, spp // 2, o['seed'] + 500)
    np.savez_compressed(os.path.join(R, 'results', 'fields', f'F_{tag}.npz'), fa=fa, fb=fb)
    print('ofield', tag, flush=True)


def timing(R, R86, R92A):
    import torch
    import prep93 as PR
    fx92 = PR.inject(R)
    import ecro86 as EC
    import model86 as M
    import model93 as MM
    PT, FX, S = fx92.PT, fx92.FX, fx92.S
    ch = json.load(open(os.path.join(R, 'results', 'e6_choice.json')))['chosen']
    tag = f'E6k{ch}_T1' if ch is not None else 'E2_T1'
    geom, ls, e, cfg = fx92.cfg_ls(tag)
    ctx = PT.Ctx(geom); st = EC.build_state(ctx)
    s = torch.as_tensor(ctx.source(PT.Lights(FX['light_sets'][ls])), device=EC.dev).view(1, ctx.N, S.NB, 3)
    gs = M.GeoState(ctx.G, st.hit, st.dbin, EC.dev)
    sync = EC.sync

    def tm(fn, n=10):
        fn(); fn(); ts = []
        for _ in range(n):
            sync(); t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
        return float(np.median(ts))
    res = {'tag': tag, 'gpu': os.popen('nvidia-smi --query-gpu=name,driver_version --format=csv,noheader').read().strip(), 'arms': {}}
    N, R_ = ctx.N, 16
    for arm in ARMS:
        net, own = load_arm(R, R92A, arm, SEEDS[0])
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            f = net.factors(gs)
            heads = 2 if P['arms'][arm]['mode'] != 'tied' else 1
            hd = net.g.head_c
            hflops = sum(2 * l.in_features * l.out_features for l in hd if hasattr(l, 'in_features')) * N * S.NB * 2 * heads
            res['arms'][arm] = {'trainable_params': net.n_trainable(), 'head_flops_per_eval': hflops,
                                'factors_s': tm(lambda: net.factors(gs)), 'single_s': tm(lambda: net.single(f, gs, s)), 'global_s': tm(lambda: MM.global_term(own, gs, s)),
                                'torch_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20,
                                'checkpoint_bytes': os.path.getsize(os.path.join(R, 'results', 'models', f'{arm}_s{SEEDS[0]}.pt'))}
    jd(res, os.path.join(R, 'results', 'timing93.json'))
    print('TIMING', json.dumps({a: {k: (round(v * 1000, 2) if k.endswith('_s') else v) for k, v in x.items()} for a, x in res['arms'].items()}), flush=True)


if __name__ == '__main__':
    import fx92
    fx92.dr.set_flag(fx92.dr.JitFlag.KernelHistory, True)
    cmd, R, R86 = sys.argv[1], sys.argv[2], sys.argv[3]
    a = [x for x in sys.argv[4:] if not x.startswith('--')]
    for d in ('fields', 'tables', 'renders', 'masks', 'lists'):
        os.makedirs(os.path.join(R, 'results', d), exist_ok=True)
    {'ident': ident, 'tests': tests, 'tables': tables, 'local': local, 'ofield': ofield, 'timing': timing}[cmd](R, R86, *a)
    os._exit(0)
