"""WL93C evaluation: oracle composition (Task A) + gate, tests, Task B accounting, render tables.

  python eval93c.py reuse   <R> <R86> <R92A> <R93A>         copy + verify WL93A T0 / TQ (T_REUSE; eval93b.reuse)
  python eval93c.py oracle  <R> <R86> <R92A>                Task A accounting of ORA / variants / T0 / TQ / own -> results/oracle.json; gate -> oracle_gate.json
  python eval93c.py tests   <R> <R86> <R92A> <R92B> <pre|post>   focused tests
  python eval93c.py account <R> <R86> <R92A>                Task B accounting (T0 / TQ / TR / TQR / own) -> results/account.json
  python eval93c.py tables  <R> <R86> <R92A> <tag> <oracle|arms>   render tables
  python eval93c.py call    <R> <R86> <module.func> <args>  WL92 function with E5 injected
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('ecro86', 'ecro_attr92', 'ecro_own92a', 'ecro_cf92b', 'ecro_stage93', 'ecro_stage93a', 'ecro_stage93b'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
sys.path.insert(0, HERE)
DEV = '--dev' in sys.argv
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_stage93c.json')))
ARMS = list(P['task_b_training']['arms'])
SEEDS = [8601, 8602]


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1)


def jl(fn):
    return json.load(open(fn))


def load_tied(R, R92A, name, seed, trained=True):
    import torch
    import model93 as MM
    import train93 as TR
    dev = torch.device('cuda')
    own = TR.load_own(R92A, seed, dev)
    net = MM.Local(own, 'tied').to(dev)
    if trained:
        net.load_state_dict(torch.load(os.path.join(R, 'results', 'models', f'{name}_s{seed}.pt'), map_location=dev))
    return net.eval(), own


def apply_f(f, E):
    import torch
    return torch.einsum('ndcr,lncr->lndc', f['a_r'], torch.einsum('nmcr,lnmc->lncr', f['c_r'], E))


def src_f(f, s):
    import torch
    return torch.einsum('ndcr,lncr->lndc', f['a_s'], torch.einsum('nmcr,lnmc->lncr', f['c_s'], s))


def exact(Mn, E):
    import torch
    return torch.einsum('lnmc,nmdc->lndc', E, Mn)


def compose(kind, net, own, gs, s, Mn, qm):
    """-> dict S, G, Bq, Rq (torch) for a model / oracle kind on one configuration (no grad)."""
    import model93 as MM
    import eval93b as EB
    rw = own.rw
    G = MM.global_term(own, gs, s)
    Eq = EB.flight(rw, gs, qm)
    if kind == 'arm':
        f = net.factors(gs); S, Bq, E1 = net.single(f, gs, s)
        return {'S': S, 'G': G, 'Bq': Bq, 'Rq': apply_f(f, Eq), 'Rown_exact': exact(Mn, E1)}
    if kind == 'ORA':
        Bq = exact(Mn, s); E1 = EB.flight(rw, gs, Bq)
        return {'S': exact(Mn, E1), 'G': G, 'Bq': Bq, 'Rq': exact(Mn, Eq), 'Rown_exact': exact(Mn, E1)}
    if kind == 'ORA_SRC':
        f = net.factors(gs); Bq = exact(Mn, s); E1 = EB.flight(rw, gs, Bq)
        return {'S': apply_f(f, E1), 'G': G, 'Bq': Bq, 'Rq': apply_f(f, Eq), 'Rown_exact': exact(Mn, E1)}
    if kind == 'ORA_RCV':
        f = net.factors(gs); Bq = src_f(f, s); E1 = EB.flight(rw, gs, Bq)
        return {'S': exact(Mn, E1), 'G': G, 'Bq': Bq, 'Rq': exact(Mn, Eq), 'Rown_exact': exact(Mn, E1)}
    if kind == 'ORA_PHYS':
        return {'S': exact(Mn, Eq), 'G': G, 'Bq': qm, 'Rq': exact(Mn, Eq), 'Rown_exact': exact(Mn, Eq)}
    raise ValueError(kind)


def sl(X, T):
    return [float((X - T).sum()), float(T.sum())]


def accounting(R, R86, R92A, entries, out_name):
    """entries: list of (name, kind, armname or None, seed)."""
    import torch
    import diag92 as DG
    import qref93a as QR
    import eval93a as EA
    import eval93b as EB
    dev = torch.device('cuda')
    C = jl(os.path.join(R, 'results', 'configs.json'))['configs']
    if DEV:                                                   # plumbing only: a few configurations, values never printed
        C = [c for c in C if c['kind'] == 'train_rep'][:1] + [c for c in C if c['kind'] == 'val'][:3] + [c for c in C if c['kind'] == 'sharp'][:1]
    _, Mq = DG.local_matrices()
    nets = {}
    for name, kind, arm, sd in entries:
        if arm == 'own':
            nets[name] = load_tied(R, R92A, 'T0', sd, trained=False)
        elif arm is not None:
            nets[name] = load_tied(R, R92A, arm, sd)
        else:
            import train93 as TR
            nets[name] = (None, TR.load_own(R92A, sd, dev))
    per = {n: [] for n, _, _, _ in entries}
    with torch.no_grad():
        for c in C:
            gs, s, G_ = EA.cfg_inputs(R86, c, dev)
            t, ta, tb, t2, t2a, t2b = EB.targets(R86, R92A, R, c)
            Q4 = QR.load_ref(R, c['cfg'], 'B5'); qm = torch.as_tensor(Q4.mean(0).astype(np.float32), device=dev)
            pm = EA.panel_mask(R86, c)
            Mn = torch.as_tensor(Mq[G_.PMAT].astype(np.float32), device=dev)
            floor = None
            for name, kind, arm, sd in entries:
                net, own = nets[name]
                o = compose(kind, net, own, gs, s, Mn, qm)
                Sn, Gn, Bqn, Rqn = (o[k].double().cpu().numpy() for k in ('S', 'G', 'Bq', 'Rq'))
                Ro = o['Rown_exact'].double().cpu().numpy()
                if floor is None:
                    floor = exact(Mn, EB.flight(own.rw, gs, qm)).double().cpu().numpy()
                W = Sn + Gn
                d = {'cfg': c['cfg'], 'src': QR.stat(Bqn, Q4, pm), 'single_l1': EB.l1s(Sn, t2), 'whole_l1': EB.l1s(W, t), 'global_l1': EB.l1s(Gn, t - t2),
                     'recv_q_l1': EB.l1s(Rqn, t2), 'recv_q_floor_l1': EB.l1s(floor, t2), 'recv_own_vs_M64_l1': EB.l1s(Sn, Ro),
                     'single_flux': [float(np.abs(Sn.sum((2, 3)) - t2.sum((2, 3))).sum()), float(t2.sum())],
                     'single_signed': sl(Sn, t2), 'global_signed': sl(Gn, t - t2), 'whole_signed': sl(W, t),
                     'recv_vs_exact_on_q': EB.l1s(Rqn, floor)}
                if t2b is not None:
                    d['single_xl2'] = EB.xl2(Sn, t2a, t2b)
                if tb is not None:
                    d['whole_xl2'] = EB.xl2(W, ta, tb)
                per[name].append(d)
    groups = {g: [i for i, c in enumerate(C) if c['kind'] == g] for g in ('val', 'train_rep', 'sharp')}
    pc = jl(os.path.join(R, 'results', 'taskA_wl93a.json'))['per_cfg']
    groups['val_high_confidence'] = [i for i in groups['val'] if pc[str(C[i]['cfg'])]['N_B5'] <= 0.10]
    res = {'models': [e[0] for e in entries], 'groups': {}, 'per_cfg': per}
    for g, idx in groups.items():
        res['groups'][g] = {}
        for n in per:
            sm = EB.summary([per[n][i] for i in idx])
            rows = [per[n][i] for i in idx]
            for k in ('single_signed', 'global_signed', 'whole_signed', 'recv_vs_exact_on_q'):
                sm[k.upper()] = EB.ratio(rows, k)
            res['groups'][g][n] = sm
    jd(res, os.path.join(R, 'results', out_name))
    V = res['groups']['val']
    if DEV:
        print('ACCOUNT', out_name, 'dev (values not printed)', len(C), flush=True)
    else:
        print('ACCOUNT', out_name, json.dumps({n: {k: round(V[n][k], 4) for k in ('SRC', 'RECV_Q', 'SINGLE', 'WHOLE', 'SINGLE_SIGNED', 'GLOBAL_SIGNED')} for n in per}), flush=True)
    return res


def pooled(rows, key):
    if key == 'SRC':
        return sum(r['src']['l1'] for r in rows) / sum(r['src']['sq'] for r in rows)
    return sum(r[key][0] for r in rows) / sum(r[key][1] for r in rows)


KEYS = {'SRC': 'SRC', 'RECV': 'recv_q_l1', 'SINGLE': 'single_l1', 'WHOLE': 'whole_l1'}


def contrasts(R, res, pairs, seed=93003):
    kinds = {c['cfg']: c['kind'] for c in jl(os.path.join(R, 'results', 'configs.json'))['configs']}
    first = next(iter(res['per_cfg']))
    vidx = [j for j, r in enumerate(res['per_cfg'][first]) if kinds[r['cfg']] == 'val']
    rows = {n: [res['per_cfg'][n][i] for i in vidx] for n in res['per_cfg']}
    rng = np.random.default_rng(seed); B = [rng.integers(0, len(vidx), len(vidx)) for _ in range(2000)]
    out = {}
    for x, y in pairs:
        for k, key in KEYS.items():
            d0 = pooled(rows[x], key) - pooled(rows[y], key)
            bs = [pooled([rows[x][j] for j in b], key) - pooled([rows[y][j] for j in b], key) for b in B]
            se = float(np.std(bs, ddof=1))
            out[f'{x}-{y}:{k}'] = {'diff': d0, 'se': se, 'margin': 2 * se, 'ci95': [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
    return out


def oracle(R, R86, R92A):
    E = []
    for s in SEEDS:
        E += [(f'ORA_s{s}', 'ORA', None, s), (f'ORA_SRC_s{s}', 'ORA_SRC', 'T0', s), (f'ORA_RCV_s{s}', 'ORA_RCV', 'T0', s), (f'ORA_PHYS_s{s}', 'ORA_PHYS', None, s),
              (f'own_s{s}', 'arm', 'own', s), (f'T0_s{s}', 'arm', 'T0', s), (f'TQ_s{s}', 'arm', 'TQ', s)]
    res = accounting(R, R86, R92A, E, 'oracle.json')
    pairs = [(f'ORA_s{s}', f'{y}_s{s}') for s in SEEDS for y in ('TQ', 'T0')] + [(f'TQ_s{s}', f'T0_s{s}') for s in SEEDS]
    D = contrasts(R, res, pairs)
    V = res['groups']['val']
    gate = {'per_seed': {}, 'contrasts': D}
    ok_all = True
    lim = {}
    for s in SEEDS:
        gap = V[f'TQ_s{s}']['WHOLE'] - V[f'T0_s{s}']['WHOLE']
        w = D[f'ORA_s{s}-TQ_s{s}:WHOLE']; sg = D[f'ORA_s{s}-TQ_s{s}:SINGLE']
        ok = w['diff'] <= -max(w['margin'], 0.5 * gap) and sg['diff'] <= -sg['margin']
        s0 = D[f'ORA_s{s}-T0_s{s}:SINGLE']; w0 = D[f'ORA_s{s}-T0_s{s}:WHOLE']
        if s0['diff'] > -s0['margin']:
            lim[s] = 'LOCAL_OPERATOR'
        elif w0['diff'] > -w0['margin']:
            lim[s] = 'GLOBAL_COMPENSATION'
        else:
            lim[s] = 'NOT_LIMITING'
        gate['per_seed'][s] = {'gap': gap, 'whole_vs_TQ': w, 'single_vs_TQ': sg, 'single_vs_T0': s0, 'whole_vs_T0': w0, 'PASS': bool(ok), 'limiting_factor': lim[s]}
        ok_all &= ok
    gate['limiting_factor'] = lim[SEEDS[0]] if len(set(lim.values())) == 1 else 'INTERACTION'
    gate['ORACLE_FEASIBLE'] = bool(ok_all)
    jd(gate, os.path.join(R, 'results', 'oracle_gate.json'))
    print('ORACLE', 'FEASIBLE' if ok_all else 'LIMITED', 'limiting', gate['limiting_factor'], '' if DEV else json.dumps({s: {k: round(v['diff'], 5) for k, v in g.items() if isinstance(v, dict)} for s, g in gate['per_seed'].items()}), flush=True)


def account(R, R86, R92A):
    E = []
    for s in SEEDS:
        E += [(f'own_s{s}', 'arm', 'own', s)] + [(f'{a}_s{s}', 'arm', a, s) for a in ARMS if os.path.exists(os.path.join(R, 'results', 'models', f'{a}_s{s}.pt'))]
        E += [(f'ORA_s{s}', 'ORA', None, s)]
    res = accounting(R, R86, R92A, E, 'account.json')
    pairs = [(f'{x}_s{s}', f'{y}_s{s}') for s in SEEDS for x, y in (('TQR', 'TQ'), ('TQR', 'T0'), ('TQR', 'TR'), ('TR', 'T0'), ('TR', 'TQ'), ('TQ', 'T0'))]
    jd(contrasts(R, res, pairs), os.path.join(R, 'results', 'contrasts.json'))
    print('CONTRASTS done', flush=True)


# ------------------------------------------------------------------------------------------------ tests
def tests(R, R86, R92A, R92B, phase):
    import torch
    import fx92
    import ecro86 as EC
    import model86 as M
    import model93 as MM
    import train93 as TR
    import diag92 as DG
    import eval93b as EB
    import qref93a as QR
    PT, FX, S = fx92.PT, fx92.FX, fx92.S
    dev = torch.device('cuda')
    geom, ls, e, cfg = fx92.cfg_ls('E2_T1')
    ctx = PT.Ctx(geom); st = EC.build_state(ctx); lt = PT.Lights(FX['light_sets'][ls])
    s = torch.as_tensor(ctx.source(lt), device=dev).view(1, ctx.N, S.NB, 3)
    gs = M.GeoState(ctx.G, st.hit, st.dbin, dev)
    _, Mq = DG.local_matrices(); Mn = torch.as_tensor(Mq[ctx.G.PMAT].astype(np.float32), device=dev)
    D = np.load(os.path.join(R, 'results', 'fields', 'D_E2_T1.npz'))
    q = torch.as_tensor(D['q'].astype(np.float32), device=dev).view(1, ctx.N, S.NB, 3)
    out = {}; ok = True
    arms = ['T0', 'TQ'] + (['TR', 'TQR'] if phase == 'post' else [])
    with torch.no_grad():
        for seed in SEEDS:
            own = TR.load_own(R92A, seed, dev); ref = own(gs, s); G0 = MM.global_term(own, gs, s)
            o = {}
            # oracle composition = WL92B X_M12 (own_s<seed>)
            cf = np.load(os.path.join(R92B, 'results', 'tables', 'CF_E2_T1.npz'))
            ora = compose('ORA', None, own, gs, s, Mn, q)
            X = (ora['S'] + ora['G'])[0].cpu().numpy(); Y = cf[f'X_M12_own_s{seed}'].astype(np.float64)
            o['oracle_vs_wl92b_X_M12'] = float(np.abs(X - Y).max() / np.abs(Y).max())
            ok &= o['oracle_vs_wl92b_X_M12'] <= 1e-5
            # receiver target R*: linear, non-negative, zero -> zero, model-independent
            Eq = EB.flight(own.rw, gs, q); Rs = exact(Mn, Eq)
            o['Rstar_scaling'] = float((exact(Mn, EB.flight(own.rw, gs, 2 * q)) - 2 * Rs).abs().max() / Rs.abs().max())
            o['Rstar_superposition'] = float((exact(Mn, EB.flight(own.rw, gs, q + 0.5 * q)) - 1.5 * Rs).abs().max() / Rs.abs().max())
            o['Rstar_min'] = float(Rs.min()); o['Rstar_zero'] = float(exact(Mn, EB.flight(own.rw, gs, torch.zeros_like(q))).abs().max())
            hm = lambda keys: sum(0.5 * (D[f'{k}_a'].astype(np.float64) + D[f'{k}_b']) for k in keys)
            t2 = hm(('e2_p', 'e2_o'))
            o['Rstar_correspondence_vs_t2_rel_l1'] = float(np.abs(Rs[0].double().cpu().numpy() - t2).sum() / t2.sum())
            ok &= o['Rstar_scaling'] <= 1e-5 and o['Rstar_superposition'] <= 1e-5 and o['Rstar_min'] >= 0 and o['Rstar_zero'] == 0.0
            for a in arms:
                net, _ = load_tied(R, R92A, a, seed, trained=True)
                f = net.factors(gs); S_, Bq, E1 = net.single(f, gs, s)
                o[f'{a}_flight_equal'] = float((EB.flight(own.rw, gs, Bq) - E1).abs().max())
                o[f'{a}_tied'] = bool(f['c_s'] is f['c_r'] and f['a_s'] is f['a_r'])
                o[f'{a}_global'] = float((MM.global_term(own, gs, s) - G0).abs().max())
                o[f'{a}_det'] = float((net.single(net.factors(gs), gs, s)[0] - S_).abs().max())
                o[f'{a}_Rstar_model_independent'] = True                         # R* above uses only (q, rw, M64); asserted by construction
                ok &= o[f'{a}_flight_equal'] == 0.0 and o[f'{a}_tied'] and o[f'{a}_global'] == 0.0 and o[f'{a}_det'] == 0.0
            if phase == 'pre':
                init = {a: load_tied(R, R92A, a, seed, trained=False)[0] for a in ('T0', 'TQ', 'TR', 'TQR')}
                sd = {a: n.state_dict() for a, n in init.items()}
                o['init_identical'] = all(torch.equal(sd['T0'][k], sd[a][k]) for a in sd for k in sd['T0'])
                o['init_equiv'] = float((init['TR'].single(init['TR'].factors(gs), gs, s)[0] + G0 - ref).abs().max() / ref.abs().max())
                ok &= o['init_identical'] and o['init_equiv'] <= 1e-5
            out[f's{seed}'] = o
    if phase == 'pre':
        cf_ = jl(os.path.join(R, 'results', 'configs.json'))
        ref_s = {QR.ref_seed(c['cfg'], b, r, li) for c in cf_['configs'] for b in QR.BNAMES for r in range(QR.NREAL) for li in range(4)}
        tr_s = {QR.SEED_TR + 4 * int(n[1:6]) + li for n in cf_['train_subset'] for li in range(4)}
        out['seed_overlap_ref_train'] = len(ref_s & tr_s)
        import hashlib
        gh = lambda sp, n: hashlib.sha256(str(np.load(os.path.join(R86, 'data', sp, n))['geom']).encode()).hexdigest()
        out['train_val_geometry_overlap'] = len({gh('train', n) for n in cf_['train_subset']} & {gh('val', n) for n in cf_['validation']})
        ok &= out['seed_overlap_ref_train'] == 0 and out['train_val_geometry_overlap'] == 0
    out['PASS'] = bool(ok)
    jd(out, os.path.join(R, 'results', f'tests93c_{phase}.json'))
    print('TESTS', phase, 'PASS' if ok else 'FAIL', flush=True)


# ------------------------------------------------------------------------------------------------ render tables
def tables(R, R86, R92A, tag, which):
    import torch
    import prep93 as PR
    fx92 = PR.inject(R)
    import ecro86 as EC
    import model86 as M
    import model93 as MM
    import train93 as TR
    import diag92 as DG
    import eval93b as EB
    PT, FX, S = fx92.PT, fx92.FX, fx92.S
    dev = torch.device('cuda')
    geom, ls, e, cfg = fx92.cfg_ls(tag)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls]); st = EC.build_state(ctx)
    s = torch.as_tensor(ctx.source(lt), device=dev).view(1, ctx.N, S.NB, 3)
    gs = M.GeoState(ctx.G, st.hit, st.dbin, dev)
    _, Mq = DG.local_matrices(); Mn = torch.as_tensor(Mq[ctx.G.PMAT].astype(np.float32), device=dev)
    fn = os.path.join(R, 'results', 'tables', f'I_{tag}.npz')
    out = dict(np.load(fn)) if os.path.exists(fn) else {}
    with torch.no_grad():
        for seed in SEEDS:
            own = TR.load_own(R92A, seed, dev); G = MM.global_term(own, gs, s)
            if which == 'oracle':
                Bq = exact(Mn, s); S_ = exact(Mn, EB.flight(own.rw, gs, Bq))
                out[f'B_ORA_s{seed}'] = (S_ + G)[0].cpu().numpy()
            else:
                for a in ARMS:
                    if os.path.exists(os.path.join(R, 'results', 'models', f'{a}_s{seed}.pt')):
                        net, _ = load_tied(R, R92A, a, seed)
                        out[f'B_{a}_s{seed}'] = (net.single(net.factors(gs), gs, s)[0] + G)[0].cpu().numpy()
    np.savez_compressed(fn, **{k: np.asarray(v, np.float32) for k, v in out.items()})
    print('tables', tag, which, sorted(out), flush=True)


def call(R, R86, fn, *args):
    import prep93 as PR
    PR.inject(R)
    mod, f = fn.split('.')
    getattr(__import__(mod), f)(R, R86, *args)


def reuse(R, R86, R92A, R93A):
    import eval93b as EB
    EB.reuse(R, R86, R92A, R93A)


if __name__ == '__main__':
    import fx92
    fx92.dr.set_flag(fx92.dr.JitFlag.KernelHistory, True)
    cmd, R, R86 = sys.argv[1], sys.argv[2], sys.argv[3]
    a = [x for x in sys.argv[4:] if not x.startswith('--')]
    for d in ('tables', 'masks', 'renders', 'models'):
        os.makedirs(os.path.join(R, 'results', d), exist_ok=True)
    {'reuse': reuse, 'oracle': oracle, 'tests': tests, 'account': account, 'tables': tables, 'call': call}[cmd](R, R86, *a)
    os._exit(0)
