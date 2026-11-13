"""WL93A evaluation: source-stage tables, tests, Task B stage metrics, render tables.

  python eval93a.py stage  <R> <R86> <R92A> <name>     source-stage tables Bq of model <name> on every configuration (enc | own_s<seed> | T0_s<seed> | TQ_s<seed>)
  python eval93a.py tests  <R> <R86> <R92A> <pre|post> focused tests
  python eval93a.py evalB  <R> <R86> <R92A>            Task B stage metrics vs the B5 reference -> results/evalB.json
  python eval93a.py tables <R> <R86> <R92A> <tag>      render tables (B = S + G) of T0 / TQ / own92a for a configuration
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('ecro86', 'ecro_attr92', 'ecro_own92a', 'ecro_cf92b', 'ecro_stage93'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
sys.path.insert(0, HERE)
DEV = '--dev' in sys.argv
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_stage93a.json')))
SEEDS = P['task_b']['training']['seeds']
ARMS = list(P['task_b']['arms'])
_PM = {}


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1)


def jl(fn):
    return json.load(open(fn))


def panel_mask(R86, c):
    """Panel (material 1) patches; the patch layout / materials are fixed for the WL86 family (quad87 _q_static)."""
    if 'pm' not in _PM:
        f = sorted(glob.glob(os.path.join(R86, 'data', 'val', 'g*.npz')))[0]
        _PM['pm'] = np.load(f)['pmat'] == 1
    return _PM['pm']


def load_stage(R, i):
    out = {}
    for fn in glob.glob(os.path.join(R, 'results', 'stage', '*.npz')):
        z = np.load(fn)
        if f'c{i:03d}' in z.files:
            out[os.path.basename(fn)[:-4]] = z[f'c{i:03d}']
    return out


def load_local(R, R92A, arm, seed, trained=True):
    import torch
    import model93 as MM
    import train93 as TR
    dev = torch.device('cuda')
    own = TR.load_own(R92A, seed, dev)
    net = MM.Local(own, P['task_b']['arms'][arm]['mode']).to(dev)
    if trained:
        net.load_state_dict(torch.load(os.path.join(R, 'results', 'models', f'{arm}_s{seed}.pt'), map_location=dev))
    return net.eval(), own


def cfg_inputs(R86, c, dev):
    """-> (gs, s [L, N, NB, 3] torch, G-like for enc_q)."""
    import torch
    import fx92
    import ecro86 as EC
    import model86 as M
    import train86 as T86
    if c['kind'] == 'sharp':
        PT, FX, S = fx92.PT, fx92.FX, fx92.S
        geom, ls, e, cfg = fx92.cfg_ls(c['tag'])
        ctx = PT.Ctx(geom); st = EC.build_state(ctx)
        s = torch.as_tensor(ctx.source(PT.Lights(FX['light_sets'][ls])), device=dev).view(1, ctx.N, S.NB, 3)
        return M.GeoState(ctx.G, st.hit, st.dbin, dev), s, ctx.G
    sp = 'train' if c['kind'] == 'train_rep' else 'val'
    it = T86.Item(os.path.join(R86, 'data', sp, c['file']))
    G = type('G', (), {'PMAT': it.tab['pmat'], 'N': len(it.tab['pmat'])})()
    return it.gs(), torch.as_tensor(it.s, device=dev), G


def stage(R, R86, R92A, name):
    import torch
    import diag92 as DG
    import xport87 as X
    import train93 as TR
    dev = torch.device('cuda')
    C = jl(os.path.join(R, 'results', 'configs.json'))['configs']
    out = {}
    if name == 'enc':
        _, Mq = DG.local_matrices()
    elif name.startswith('own_s'):
        mdl = TR.load_own(R92A, int(name[5:]), dev)
    else:
        arm, sd = name.split('_s'); mdl, _ = load_local(R, R92A, arm, int(sd))
    with torch.no_grad():
        for c in C:
            gs, s, G = cfg_inputs(R86, c, dev)
            if name == 'enc':
                sn = s.double().cpu().numpy()
                Bq = np.stack([X.enc_q(G, sn[l], Mq) for l in range(sn.shape[0])])
            else:
                f = mdl.factors(gs)
                c_, a_ = (f['c'], f['a']) if 'c' in f else (f['c_s'], f['a_s'])
                Bq = torch.einsum('ndcr,lncr->lndc', a_, torch.einsum('nmcr,lnmc->lncr', c_, s)).cpu().numpy()
            out[f"c{c['cfg']:03d}"] = Bq.astype(np.float32)
    os.makedirs(os.path.join(R, 'results', 'stage'), exist_ok=True)
    np.savez(os.path.join(R, 'results', 'stage', f'{name}.npz'), **out)
    print('STAGE', name, len(out), flush=True)


# ------------------------------------------------------------------------------------------------ tests
def tests(R, R86, R92A, phase):
    import torch
    import fx92
    import ecro86 as EC
    import model86 as M
    import model93 as MM
    import train93 as TR
    import quad87 as Q
    import qref93a as QR
    PT, FX, S, E87 = fx92.PT, fx92.FX, fx92.S, fx92.E87
    dev = torch.device('cuda')
    geom, ls, e, cfg = fx92.cfg_ls('E2_T1')
    ctx = PT.Ctx(geom); st = EC.build_state(ctx); lt = PT.Lights(FX['light_sets'][ls])
    s = torch.as_tensor(ctx.source(lt), device=dev).view(1, ctx.N, S.NB, 3)
    gs = M.GeoState(ctx.G, st.hit, st.dbin, dev)
    out = {}; ok = True
    with torch.no_grad():
        for seed in SEEDS:
            own = TR.load_own(R92A, seed, dev)
            ref = own(gs, s); G0 = MM.global_term(own, gs, s)
            nets = {a: load_local(R, R92A, a, seed, trained=(phase == 'post'))[0] for a in ARMS}
            o = {}
            if phase == 'pre':
                sd = {a: n.state_dict() for a, n in nets.items()}
                o['init_identical'] = all(torch.equal(sd['T0'][k], sd['TQ'][k]) for k in sd['T0'])
                o['init_equiv'] = {a: float((n.single(n.factors(gs), gs, s)[0] + G0 - ref).abs().max() / ref.abs().max()) for a, n in nets.items()}
                ok &= o['init_identical'] and all(v <= 1e-5 for v in o['init_equiv'].values())
            o['params'] = {a: n.n_trainable() for a, n in nets.items()}
            o['global_identity'] = float((MM.global_term(own, gs, s) - G0).abs().max())
            o['tied'] = all(n.factors(gs)['c_s'] is n.factors(gs)['c_r'] or torch.equal(n.factors(gs)['c_s'], n.factors(gs)['c_r']) for n in nets.values())
            o['base_frozen'] = not any(p.requires_grad for p in own.parameters())
            o['det'] = max(float((n.single(n.factors(gs), gs, s)[0] - n.single(n.factors(gs), gs, s)[0]).abs().max()) for n in nets.values())
            ok &= len(set(o['params'].values())) == 1 and o['params']['T0'] > 0 and o['global_identity'] == 0.0 and o['tied'] and o['base_frozen'] and o['det'] == 0.0
            out[f's{seed}'] = o
    if phase == 'pre':
        # target generation: B0 with the WL87 seed reproduces the stored E2_T1 D q (WL92B field), and every budget runs with its P / Mq
        Dq = np.load(os.path.join(R, 'results', 'fields', 'D_E2_T1.npz'))['q'].astype(np.float64)
        qq = Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q'])
        out['qgen_wl87_repro_max_rel'] = float(np.abs(qq - Dq).max() / Dq.max())
        out['qgen_budgets'] = {}
        for b in QR.BNAMES:
            Pp, Mq = QR.budget(b)
            x = Q.qtable(ctx, lt, Pp, Mq, 12345)
            out['qgen_budgets'][b] = {'P': Pp, 'Mq': Mq, 'finite': bool(np.isfinite(x).all()), 'nonneg': bool(x.min() >= 0), 'flux_ratio_vs_wl87': float(x.sum() / qq.sum())}
        ok &= out['qgen_wl87_repro_max_rel'] <= 1e-6 and all(v['finite'] and v['nonneg'] for v in out['qgen_budgets'].values())
        # seed disjointness
        cf = jl(os.path.join(R, 'results', 'configs.json'))
        ref_s = {QR.ref_seed(c['cfg'], b, r, li) for c in cf['configs'] for b in QR.BNAMES for r in range(QR.NREAL) for li in range(4)}
        tr_s = {QR.SEED_TR + 4 * int(n[1:6]) + li for n in cf['train_subset'] for li in range(4)}
        tv_s = {QR.SEED_TV + 4 * int(n[1:6]) + li for n in cf['validation'] for li in range(4)}
        w93 = {93000 + 4 * i + li + o for i in range(1024) for li in range(4) for o in (0, 500000)}
        sets = {'ref': ref_s, 'train': tr_s, 'trainval': tv_s, 'wl93': w93, 'wl87': {E87['seed_q']}}
        names = list(sets)
        out['seed_overlaps'] = {f'{a}&{b}': len(sets[a] & sets[b]) for i, a in enumerate(names) for b in names[i + 1:]}
        ok &= all(v == 0 for v in out['seed_overlaps'].values())
        # split separation
        trs, val, rep = set(cf['train_subset']), set(cf['validation']), set(cf['train_representative'])
        tq = {os.path.basename(f) for f in glob.glob(os.path.join(R, 'data', 'train_q', 'g*.npz'))}
        vq = {os.path.basename(f) for f in glob.glob(os.path.join(R, 'data', 'val_q', 'g*.npz'))}
        import hashlib
        gh = lambda sp, n: hashlib.sha256(str(np.load(os.path.join(R86, 'data', sp, n))['geom']).encode()).hexdigest()
        htr = {gh('train', n) for n in trs}; hva = {gh('val', n) for n in val}       # train / val directories reuse file names: compare geometry content
        out['split'] = {'train_val_overlap': len(htr & hva), 'rep_subset_of_train': rep <= trs, 'train_q_files_eq_subset': tq == trs,
                        'val_q_files_eq_validation': vq == val,
                        'train_files_in_train_dir': all(os.path.exists(os.path.join(R86, 'data', 'train', n)) for n in trs)}
        ok &= out['split']['val_q_files_eq_validation'] and out['split']['train_files_in_train_dir']
        ok &= out['split']['train_val_overlap'] == 0 and out['split']['rep_subset_of_train'] and out['split']['train_q_files_eq_subset']
    out['PASS'] = bool(ok)
    jd(out, os.path.join(R, 'results', f'tests93a_{phase}.json'))
    print('TESTS', phase, 'PASS' if ok else 'FAIL', flush=True)


# ------------------------------------------------------------------------------------------------ Task B stage evaluation
def evalB(R, R86, R92A):
    import qref93a as QR
    import eval93 as E93
    cf = jl(os.path.join(R, 'results', 'configs.json')); C = cf['configs']
    names = ['enc'] + [f'own_s{s}' for s in SEEDS] + [f'{a}_s{s}' for a in ARMS for s in SEEDS]
    res = {'groups': {}, 'noise_fit': {}, 'training': {}}
    for gname in ('val', 'train_rep', 'sharp'):
        ids = QR.group(C, gname)
        st = {n: [] for n in names}
        fit = {n: [0.0, 0.0] for n in names}
        for i in ids:
            Qr = QR.load_ref(R, i, 'B5'); X = load_stage(R, i); pm = panel_mask(R86, C[i])
            for n in names:
                st[n].append(QR.stat(X[n], Qr, pm))
            if gname == 'train_rep':                                            # fit to the model's own (single-realization) training target
                qt = np.load(os.path.join(R, 'data', 'train_q', C[i]['file']))['q'].astype(np.float64)[:, pm]
                for n in names:
                    x = X[n][:, pm].astype(np.float64)
                    fit[n][0] += float(((x - qt) ** 2).sum()); fit[n][1] += float((qt ** 2).sum())
        g = {n: {**QR.pooled(st[n]), 'boot': QR.boot(st[n])} for n in names}
        for s in SEEDS:
            a, b = st[f'TQ_s{s}'], st[f'T0_s{s}']
            dM = [QR.M_of(a, range(len(a)), p) - QR.M_of(b, range(len(b)), p) for p in range(3)]
            g[f'dM_s{s}'] = {'dM': dM[0], 'pairings': dM, 'spread': max(dM) - min(dM), 'boot': QR.boot(a, other=b)}
            g[f'dM_init_s{s}'] = {'T0_minus_own': QR.M_of(b, range(len(b))) - QR.M_of(st[f'own_s{s}'], range(len(b))),
                                  'TQ_minus_own': QR.M_of(a, range(len(a))) - QR.M_of(st[f'own_s{s}'], range(len(a)))}
        res['groups'][gname] = g
        if gname == 'train_rep':
            res['noise_fit'] = {n: {'rel_l2_vs_own_training_target': math.sqrt(fit[n][0] / fit[n][1]), 'M_vs_independent_B5': g[n]['M']} for n in names}
    # local matrices: panel peak retention of a c^T vs frozen-ER M64 on E2_T1 (sharp-direction retention of the operator)
    import torch
    import fx92
    import ecro86 as EC
    import model86 as M
    import diag92 as DG
    PT = fx92.PT
    geom, ls, e, cfg = fx92.cfg_ls('E2_T1')
    ctx = PT.Ctx(geom); st_ = EC.build_state(ctx); gs = M.GeoState(ctx.G, st_.hit, st_.dbin, EC.dev)
    _, Mq = DG.local_matrices(); zb = DG.flux_w()
    mats = {}
    import train93 as TR
    with torch.no_grad():
        for s in SEEDS:
            own = TR.load_own(R92A, s, EC.dev); f = own.factors(gs)
            mats[f'own_s{s}'] = E93.pmetrics(torch.einsum('ndcr,nmcr->nmdc', f['a'], f['c']).double().cpu().numpy(), Mq, ctx.G.PMAT, zb)
            for a in ARMS:
                net, _ = load_local(R, R92A, a, s); f = net.factors(gs)
                mats[f'{a}_s{s}'] = E93.pmetrics(torch.einsum('ndcr,nmcr->nmdc', f['a_s'], f['c_s']).double().cpu().numpy(), Mq, ctx.G.PMAT, zb)
    res['matrix_panel_E2_T1'] = mats
    for a in ARMS:
        for s in SEEDS:
            fn = os.path.join(R, 'results', 'models', f'{a}_s{s}.json')
            if os.path.exists(fn):
                d = jl(fn); res['training'][f'{a}_s{s}'] = {k: d[k] for k in ('val', 'nonfinite_steps', 'train_s', 'trainable_params', 'init_sha256', 'final_sha256', 'items')}
    jd(res, os.path.join(R, 'results', 'evalB.json'))
    V = res['groups']['val']
    print('EVALB val M', json.dumps({n: round(V[n]['M'], 4) for n in names}), 'dM', json.dumps({s: round(V[f'dM_s{s}']['dM'], 4) for s in SEEDS}), flush=True)


# ------------------------------------------------------------------------------------------------ render tables
def tables(R, R86, R92A, tag):
    import torch
    import prep93 as PR
    fx92 = PR.inject(R)
    import ecro86 as EC
    import model86 as M
    import model93 as MM
    import train93 as TR
    PT, FX, S = fx92.PT, fx92.FX, fx92.S
    dev = torch.device('cuda')
    geom, ls, e, cfg = fx92.cfg_ls(tag)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls]); st = EC.build_state(ctx)
    s = torch.as_tensor(ctx.source(lt), device=dev).view(1, ctx.N, S.NB, 3)
    gs = M.GeoState(ctx.G, st.hit, st.dbin, dev)
    out = {}
    with torch.no_grad():
        for seed in SEEDS:
            own = TR.load_own(R92A, seed, dev)
            out[f'B_own92a_s{seed}'] = own(gs, s)[0].cpu().numpy()
            G = MM.global_term(own, gs, s)
            for a in ARMS:
                net, _ = load_local(R, R92A, a, seed)
                S_ = net.single(net.factors(gs), gs, s)[0]
                out[f'B_{a}_s{seed}'] = (S_ + G)[0].cpu().numpy()
    os.makedirs(os.path.join(R, 'results', 'tables'), exist_ok=True)
    np.savez_compressed(os.path.join(R, 'results', 'tables', f'I_{tag}.npz'), **{k: v.astype(np.float32) for k, v in out.items()})
    print('tables', tag, flush=True)


if __name__ == '__main__':
    import fx92
    fx92.dr.set_flag(fx92.dr.JitFlag.KernelHistory, True)
    cmd, R, R86, R92A = sys.argv[1:5]
    a = [x for x in sys.argv[5:] if not x.startswith('--')]
    {'stage': stage, 'tests': tests, 'evalB': evalB, 'tables': tables}[cmd](R, R86, R92A, *a)
    os._exit(0)
