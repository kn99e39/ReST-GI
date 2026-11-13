"""WL93B evaluation: reuse verification, tests, stage / integrated accounting, operators, pattern diagnostic, render tables.

  python eval93b.py reuse   <R> <R86> <R92A> <R93A>         copy + verify the WL93A T0 / TQ checkpoints (T_REUSE)
  python eval93b.py tests   <R> <R86> <R92A> <pre|post>     focused tests
  python eval93b.py account <R> <R86> <R92A>                stage / integrated accounting of every model -> results/account.json
  python eval93b.py pattern <R> <R86>                       LPAT lights, representability gate, masks -> results/lpat.json
  python eval93b.py tables  <R> <R86> <R92A> <tag>          render tables (B = S + G) of every arm and WL92A own
  python eval93b.py call    <R> <R86> <module.func> <args>  run a WL92 function with E5 and LPAT injected
"""
import glob, hashlib, json, math, os, shutil, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('ecro86', 'ecro_attr92', 'ecro_own92a', 'ecro_cf92b', 'ecro_stage93', 'ecro_stage93a'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
sys.path.insert(0, HERE)
DEV = '--dev' in sys.argv
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_stage93b.json')))
ARMS = list(P['arms'])
SEEDS = [8601, 8602]
REPRO = ['R_T0', 'R_TQ']


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1)


def jl(fn):
    return json.load(open(fn))


def mode_of(name):
    return P['arms'][name[2:] if name.startswith('R_') else name]['mode']


def load_local(R, R92A, name, seed, trained=True):
    import torch
    import model93 as MM
    import train93 as TR
    dev = torch.device('cuda')
    own = TR.load_own(R92A, seed, dev)
    net = MM.Local(own, mode_of(name), cm_seed=seed + 9300).to(dev)
    if trained:
        net.load_state_dict(torch.load(os.path.join(R, 'results', 'models', f'{name}_s{seed}.pt'), map_location=dev))
    return net.eval(), own


def inject(R):
    import prep93 as PR
    fx92 = PR.inject(R)
    fn = os.path.join(R, 'results', 'lpat.json')
    if os.path.exists(fn):
        fx92.FX['light_sets']['LPAT'] = jl(fn)['lights']
    return fx92


# ------------------------------------------------------------------------------------------------ reuse of WL93A T0 / TQ
def reuse(R, R86, R92A, R93A):
    import torch
    import train93 as TR
    out = {}; ok = True
    os.makedirs(os.path.join(R, 'results', 'models'), exist_ok=True)
    for a in ('T0', 'TQ'):
        for s in SEEDS:
            for ext in ('pt', 'json'):
                shutil.copy2(os.path.join(R93A, 'results', 'models', f'{a}_s{s}.{ext}'), os.path.join(R, 'results', 'models', f'{a}_s{s}.{ext}'))
            rec = jl(os.path.join(R, 'results', 'models', f'{a}_s{s}.json'))
            net, _ = load_local(R, R92A, a, s)
            h = TR.sd_hash(net)
            init, _ = load_local(R, R92A, a, s, trained=False)
            out[f'{a}_s{s}'] = {'final_sha256_recorded': rec['final_sha256'], 'final_sha256_loaded': h, 'init_sha256_recorded': rec['init_sha256'],
                                'init_sha256_rebuilt': TR.sd_hash(init), 'order_sha256': rec['order_sha256'], 'steps': rec['steps'], 'items': rec['items']}
            ok &= h == rec['final_sha256'] and TR.sd_hash(init) == rec['init_sha256'] and rec['steps'] == 2000 and rec['items'] == 128
    out['PASS'] = bool(ok)
    jd(out, os.path.join(R, 'results', 'reuse.json'))
    print('REUSE', 'PASS' if ok else 'FAIL', flush=True)


# ------------------------------------------------------------------------------------------------ tests
def tests(R, R86, R92A, phase):
    import torch
    import fx92
    import ecro86 as EC
    import model86 as M
    import model93 as MM
    import train93 as TR
    import qref93a as QR
    PT, FX, S = fx92.PT, fx92.FX, fx92.S
    dev = torch.device('cuda')
    geom, ls, e, cfg = fx92.cfg_ls('E2_T1')
    ctx = PT.Ctx(geom); st = EC.build_state(ctx); lt = PT.Lights(FX['light_sets'][ls])
    s = torch.as_tensor(ctx.source(lt), device=dev).view(1, ctx.N, S.NB, 3)
    gs = M.GeoState(ctx.G, st.hit, st.dbin, dev)
    out = {}; ok = True
    for seed in SEEDS:
        own = TR.load_own(R92A, seed, dev)
        with torch.no_grad():
            ref = own(gs, s); G0 = MM.global_term(own, gs, s)
        o = {'params': {}, 'init_equiv': {}, 'global_identity': {}, 'tied_same_tensor': {}}
        outs = {}
        for a in ARMS:
            net, _ = load_local(R, R92A, a, seed, trained=(phase == 'post'))
            with torch.no_grad():
                f = net.factors(gs); S_ = net.single(f, gs, s)[0]
                o['global_identity'][a] = float((MM.global_term(own, gs, s) - G0).abs().max())
                if phase == 'pre':
                    o['init_equiv'][a] = float((S_ + G0 - ref).abs().max() / ref.abs().max())
                o['tied_same_tensor'][a] = bool(f['c_s'] is f['c_r'] and f['a_s'] is f['a_r'])
                o.setdefault('det', {})[a] = float((net.single(net.factors(gs), gs, s)[0] - S_).abs().max())
            o['params'][a] = net.n_trainable(); outs[a] = S_
        ok &= all(v == 0.0 for v in o['global_identity'].values()) and all(v == 0.0 for v in o['det'].values())
        ok &= all(o['tied_same_tensor'][a] == (mode_of(a) != 'untied') for a in ARMS)
        ok &= o['params']['U0'] == o['params']['UQ'] == o['params']['C0'] == o['params']['CQ'] > o['params']['T0'] == o['params']['TQ']
        ok &= not any(p.requires_grad for p in own.parameters())
        if phase == 'pre':
            ok &= all(v <= 1e-5 for v in o['init_equiv'].values())
            # ownership: a source-only loss leaves the untied receiver heads at zero gradient
            net, _ = load_local(R, R92A, 'UQ', seed, trained=False); net.train()
            f = net.factors(gs)
            Bq = torch.einsum('ndcr,lncr->lndc', f['a_s'], torch.einsum('nmcr,lnmc->lncr', f['c_s'], s))
            Bq.sum().backward()
            gr = sum(float(p.grad.abs().sum()) for p in list(net.head_c2.parameters()) + list(net.head_a2.parameters()) if p.grad is not None)
            gsrc = sum(float(p.grad.abs().sum()) for p in list(net.g.head_c.parameters()) + list(net.g.head_a.parameters()) if p.grad is not None)
            o['ownership'] = {'receiver_head_grad': gr, 'source_head_grad': gsrc}
            ok &= gr == 0.0 and gsrc > 0
        else:
            o['nondegenerate'] = {f'{a}_vs_{b}': float((outs[a] - outs[b]).abs().max() / outs[b].abs().max()) for a, b in (('C0', 'T0'), ('CQ', 'TQ'), ('U0', 'T0'), ('UQ', 'TQ'))}
            ok &= all(v > 1e-4 for v in o['nondegenerate'].values())
        out[f's{seed}'] = o
    if phase == 'pre':
        cf = jl(os.path.join(R, 'results', 'configs.json'))
        ref_s = {QR.ref_seed(c['cfg'], b, r, li) for c in cf['configs'] for b in QR.BNAMES for r in range(QR.NREAL) for li in range(4)}
        tr_s = {QR.SEED_TR + 4 * int(n[1:6]) + li for n in cf['train_subset'] for li in range(4)}
        tv_s = {QR.SEED_TV + 4 * int(n[1:6]) + li for n in cf['validation'] for li in range(4)}
        out['seed_overlaps'] = {'ref&train': len(ref_s & tr_s), 'ref&trainval': len(ref_s & tv_s), 'train&trainval': len(tr_s & tv_s)}
        ok &= all(v == 0 for v in out['seed_overlaps'].values())
        tq = {os.path.basename(f) for f in glob.glob(os.path.join(R, 'data', 'train_q', 'g*.npz'))}
        gh = lambda sp, n: hashlib.sha256(str(np.load(os.path.join(R86, 'data', sp, n))['geom']).encode()).hexdigest()
        out['split'] = {'train_val_geometry_overlap': len({gh('train', n) for n in cf['train_subset']} & {gh('val', n) for n in cf['validation']}),
                        'train_q_eq_subset': tq == set(cf['train_subset']),
                        'train_target_budget': str(np.load(os.path.join(R, 'data', 'train_q', cf['train_subset'][0]))['budget'])}
        ok &= out['split']['train_val_geometry_overlap'] == 0 and out['split']['train_q_eq_subset']
    out['PASS'] = bool(ok)
    jd(out, os.path.join(R, 'results', f'tests93b_{phase}.json'))
    print('TESTS', phase, 'PASS' if ok else 'FAIL', flush=True)


# ------------------------------------------------------------------------------------------------ accounting
def flight(rw, gs, L):
    import fx92
    S = fx92.S
    NB, K, KAPPA = S.NB, S.K, S.KAPPA
    Ls, N = L.shape[0], gs.N
    ok = (gs.hit >= 0).view(1, N * K, 1)
    idx = (gs.hit.clamp(min=0) * NB + gs.dbin).view(-1)
    return ((L.reshape(Ls, N * NB, 3)[:, idx] * ok).view(Ls, N, NB, KAPPA, 3) * rw.view(1, 1, NB, KAPPA, 1)).sum(3)


def l1s(X, T):
    return [float(np.abs(X - T).sum()), float(T.sum())]


def xl2(X, A, B):
    """cross-product sums: sum (X - A)(X - B), sum A B (A, B independent halves)."""
    return [float(((X - A) * (X - B)).sum()), float((A * B).sum())]


def targets(R86, R92A, R, c):
    """-> t, t_a, t_b, t2, t2_a, t2_b as float64 numpy [L, N, NB, 3] (halves None if not available)."""
    if c['kind'] == 'sharp':
        D = np.load(os.path.join(R, 'results', 'fields', f"D_{c['tag']}.npz"))
        h = lambda keys, x: sum(D[f'{k}_{x}'].astype(np.float64) for k in keys)[None]
        ta, tb = h(('e2_p', 'e2_o', 'e3_p', 'e3_o'), 'a'), h(('e2_p', 'e2_o', 'e3_p', 'e3_o'), 'b')
        t2a, t2b = h(('e2_p', 'e2_o'), 'a'), h(('e2_p', 'e2_o'), 'b')
        return 0.5 * (ta + tb), ta, tb, 0.5 * (t2a + t2b), t2a, t2b
    sp = 'train' if c['kind'] == 'train_rep' else 'val'
    z = np.load(os.path.join(R86, 'data', sp, c['file'])); z2 = np.load(os.path.join(R92A, 'data', sp + '_f2', c['file']))
    fa = z['f'].astype(np.float64); fb = z['fb'].astype(np.float64) if 'fb' in z.files else None
    f2a = z2['f2'].astype(np.float64); f2b = z2['f2b'].astype(np.float64) if 'f2b' in z2.files else None
    t = fa if fb is None else 0.5 * (fa + fb); t2 = f2a if f2b is None else 0.5 * (f2a + f2b)
    return t, fa, fb, t2, f2a, f2b


def model_names():
    return [f'own_s{s}' for s in SEEDS] + [f'{a}_s{s}' for a in ARMS for s in SEEDS] + [f'{a}_s{s}' for a in REPRO for s in SEEDS]


def account(R, R86, R92A):
    import torch
    import diag92 as DG
    import qref93a as QR
    import eval93a as EA
    import model93 as MM
    dev = torch.device('cuda')
    C = jl(os.path.join(R, 'results', 'configs.json'))['configs']
    _, Mq = DG.local_matrices()
    nets = {}
    for n in model_names():
        a, sd = n.rsplit('_s', 1); sd = int(sd)
        if a == 'own':
            nets[n] = load_local(R, R92A, 'T0', sd, trained=False)
        elif os.path.exists(os.path.join(R, 'results', 'models', f'{n}.pt')):
            nets[n] = load_local(R, R92A, a, sd)
    per = {n: [] for n in nets}
    with torch.no_grad():
        for c in C:
            gs, s, G = EA.cfg_inputs(R86, c, dev)
            t, ta, tb, t2, t2a, t2b = targets(R86, R92A, R, c)
            Q4 = QR.load_ref(R, c['cfg'], 'B5'); qm = torch.as_tensor(Q4.mean(0).astype(np.float32), device=dev)
            pm = EA.panel_mask(R86, c)
            Mn = torch.as_tensor(Mq[G.PMAT].astype(np.float32), device=dev)
            Eq_cache = {}
            for n, (net, own) in nets.items():
                f = net.factors(gs)
                S_, Bq, E1 = net.single(f, gs, s)
                Gt = MM.global_term(own, gs, s)
                rw = net.g.rw
                if 'Eq' not in Eq_cache:
                    Eq_cache['Eq'] = flight(rw, gs, qm)
                    Eq_cache['exact'] = torch.einsum('lnmc,nmdc->lndc', Eq_cache['Eq'], Mn).double().cpu().numpy()
                Rq = torch.einsum('ndcr,lncr->lndc', f['a_r'], torch.einsum('nmcr,lnmc->lncr', f['c_r'], Eq_cache['Eq'])).double().cpu().numpy()
                Rown_exact = torch.einsum('lnmc,nmdc->lndc', E1, Mn).double().cpu().numpy()
                Sn, Gn, Bqn = S_.double().cpu().numpy(), Gt.double().cpu().numpy(), Bq.double().cpu().numpy()
                W = Sn + Gn
                d = {'cfg': c['cfg'], 'src': QR.stat(Bqn, Q4, pm), 'single_l1': l1s(Sn, t2), 'whole_l1': l1s(W, t), 'global_l1': l1s(Gn, t - t2),
                     'recv_q_l1': l1s(Rq, t2), 'recv_q_floor_l1': l1s(Eq_cache['exact'], t2), 'recv_own_vs_M64_l1': l1s(Sn, Rown_exact),
                     'single_flux': [float(np.abs(Sn.sum((2, 3)) - t2.sum((2, 3))).sum()), float(t2.sum())]}
                if t2b is not None:
                    d['single_xl2'] = xl2(Sn, t2a, t2b)
                if tb is not None:
                    d['whole_xl2'] = xl2(W, ta, tb)
                per[n].append(d)
    groups = {g: [i for i, c in enumerate(C) if c['kind'] == g] for g in ('val', 'train_rep', 'sharp')}
    hc = None
    TA = os.path.join(R, 'results', 'taskA_wl93a.json')
    if os.path.exists(TA):
        pc = jl(TA)['per_cfg']; hc = [i for i in groups['val'] if pc[str(C[i]['cfg'])]['N_B5'] <= 0.10]
    res = {'models': list(nets), 'groups': {}, 'per_cfg': {n: per[n] for n in nets}}
    for g, idx in list(groups.items()) + ([('val_high_confidence', hc)] if hc else []):
        res['groups'][g] = {n: summary([per[n][i] for i in idx]) for n in nets}
    jd(res, os.path.join(R, 'results', 'account.json'))
    V = res['groups']['val']
    print('ACCOUNT val', json.dumps({n: {k: round(V[n][k], 4) for k in ('SRC', 'WHOLE', 'SINGLE', 'RECV_Q')} for n in nets}), flush=True)


def ratio(rows, key, p=None):
    num = sum(r[key][0] for r in rows if key in r); den = sum(r[key][1] for r in rows if key in r)
    return num / den if den else None


def summary(rows):
    import qref93a as QR
    src = QR.pooled([r['src'] for r in rows])
    out = {'n': len(rows), 'SRC': src['rel_l1'], 'SRC_M': src['M'], 'SRC_U': src['U'], 'SRC_flux': src['flux_err'], 'SRC_dir': src['dir_err'], 'SRC_peak': src['peak_retention'],
           'WHOLE': ratio(rows, 'whole_l1'), 'SINGLE': ratio(rows, 'single_l1'), 'GLOBAL': ratio(rows, 'global_l1'), 'RECV_Q': ratio(rows, 'recv_q_l1'),
           'RECV_Q_floor': ratio(rows, 'recv_q_floor_l1'), 'RECV_OWN_vs_M64': ratio(rows, 'recv_own_vs_M64_l1'), 'SINGLE_flux': ratio(rows, 'single_flux')}
    for k in ('single_xl2', 'whole_xl2'):
        if all(k in r for r in rows):
            v = ratio(rows, k); out[k.upper()] = math.sqrt(max(v, 0.0)) if v is not None else None
    return out


# ------------------------------------------------------------------------------------------------ pattern diagnostic
def pattern(R, R86):
    fx92 = inject(R)
    import masks86
    PT, FX, S = fx92.PT, fx92.FX, fx92.S
    PD = P['pattern_diagnostic']
    geom = FX['E2']['T1']
    ctx = PT.Ctx(geom); G = ctx.G
    pan = np.nonzero(G.PMAT == 1)[0]
    cells = [((-0.1, 0.3), (0.9, 1.3)), ((0.3, 0.7), (0.5, 0.9))]
    dist = math.hypot(1.2, 0.8)
    half = math.degrees(math.atan(0.13 / dist))          # oblique from -y (clear of the moved box B at T1), spot inside its 40 cm cell
    unit = [{'pos': [-0.3, 0.5 * (y[0] + y[1]) - 0.8, 0.5 * (z[0] + z[1])], 'rgb': [1.0, 1.0, 1.0], 'aim': [-1.5, 0.5 * (y[0] + y[1]), 0.5 * (z[0] + z[1])], 'cone_deg': half} for y, z in cells]
    lvl = lambda lights: ctx.source(PT.Lights(lights)).reshape(G.N, S.NB, 3).sum((1, 2))
    l1 = lvl(FX['light_sets'][FX['E2']['light_set']]); lu = lvl(unit)
    a = G.PAREA
    scale = float((a[pan] * l1[pan]).sum() / (a[pan] * lu[pan]).sum())
    lights = [{**l, 'rgb': [scale] * 3} for l in unit]
    lp = lvl(lights)
    pc = G.PC[pan]
    inc = np.zeros(len(pan), bool)
    for y, z in cells:
        inc |= (pc[:, 1] > y[0]) & (pc[:, 1] < y[1]) & (pc[:, 2] > z[0]) & (pc[:, 2] < z[1])
    mx = float(lp[pan][inc].max()) if inc.any() else 0.0
    lit_ok = bool(inc.sum() == 8 and (lp[pan][inc] >= 0.3 * mx).all())
    dark_ok = bool((lp[pan][~inc] <= 0.05 * mx).all())
    from scene86 import light_ok
    pos_ok = all(light_ok(l['pos'], geom) and light_ok(l['pos'], FX['E2']['T0']) for l in lights)
    ok = lit_ok and dark_ok and pos_ok
    out = {'lights': lights, 'cells': cells, 'cone_half_deg': half, 'flux_scale': scale, 'lit_patches': int(inc.sum()),
           'lit_levels_rel_max': [float(v / mx) for v in lp[pan][inc]], 'max_dark_rel_max': float(lp[pan][~inc].max() / mx) if mx else None,
           'panel_flux_ratio_vs_L1': float((a[pan] * lp[pan]).sum() / (a[pan] * l1[pan]).sum()), 'light_positions_ok': pos_ok,
           'off_panel_level_rel_panel': float((a * lp).sum() / max((a[pan] * lp[pan]).sum(), 1e-30)) - 1.0, 'PASS': ok}
    jd(out, os.path.join(R, 'results', 'lpat.json'))
    if ok:
        FX['light_sets']['LPAT'] = lights
        mk, aux = masks86.compute('E2', ls='LPAT')
        np.savez_compressed(os.path.join(R, 'results', 'masks', 'E2_LPAT.npz'), **mk, **aux)
        out['mask_pixels'] = {k: int(v.sum()) for k, v in mk.items()}
        jd(out, os.path.join(R, 'results', 'lpat.json'))
    print('PATTERN', 'PASS' if ok else 'FAIL', json.dumps({k: out[k] for k in ('lit_patches', 'max_dark_rel_max', 'panel_flux_ratio_vs_L1', 'light_positions_ok')}), out.get('mask_pixels'), flush=True)


# ------------------------------------------------------------------------------------------------ render tables / call
def tables(R, R86, R92A, tag):
    import torch
    fx92 = inject(R)
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
                out[f'B_{a}_s{seed}'] = (net.single(net.factors(gs), gs, s)[0] + G)[0].cpu().numpy()
    os.makedirs(os.path.join(R, 'results', 'tables'), exist_ok=True)
    np.savez_compressed(os.path.join(R, 'results', 'tables', f'I_{tag}.npz'), **{k: v.astype(np.float32) for k, v in out.items()})
    print('tables', tag, flush=True)


def operators(R, R86, R92A):
    import torch
    import fx92
    import ecro86 as EC
    import model86 as M
    import diag92 as DG
    import eval93 as E93
    PT = fx92.PT
    geom, ls, e, cfg = fx92.cfg_ls('E2_T1')
    ctx = PT.Ctx(geom); st = EC.build_state(ctx); gs = M.GeoState(ctx.G, st.hit, st.dbin, EC.dev)
    _, Mq = DG.local_matrices(); zb = DG.flux_w(); W2 = (zb[:, None] * zb[None, :])[:, :, None]
    pan = np.nonzero(ctx.G.PMAT == 1)[0]
    res = {}
    with torch.no_grad():
        for n in model_names():
            a, sd = n.rsplit('_s', 1); sd = int(sd)
            if a == 'own':
                net, _ = load_local(R, R92A, 'T0', sd, trained=False)
            elif os.path.exists(os.path.join(R, 'results', 'models', f'{n}.pt')):
                net, _ = load_local(R, R92A, a, sd)
            else:
                continue
            f = net.factors(gs)
            Ms = torch.einsum('ndcr,nmcr->nmdc', f['a_s'], f['c_s']).double().cpu().numpy()
            Mr = torch.einsum('ndcr,nmcr->nmdc', f['a_r'], f['c_r']).double().cpu().numpy()
            res[n] = {'source_panel': E93.pmetrics(Ms, Mq, ctx.G.PMAT, zb), 'receiver_panel': E93.pmetrics(Mr, Mq, ctx.G.PMAT, zb),
                      'source_minus_receiver_rel': float(np.mean([np.linalg.norm((Ms[i] - Mr[i]) * W2) / max(np.linalg.norm(Mr[i] * W2), 1e-30) for i in pan]))}
    jd(res, os.path.join(R, 'results', 'operators.json'))
    print('OPERATORS', json.dumps({n: (round(v['source_panel']['peak'], 3), round(v['receiver_panel']['peak'], 3), round(v['source_minus_receiver_rel'], 3)) for n, v in res.items()}), flush=True)


def call(R, R86, fn, *args):
    inject(R)
    mod, f = fn.split('.')
    getattr(__import__(mod), f)(R, R86, *args)


if __name__ == '__main__':
    import fx92
    fx92.dr.set_flag(fx92.dr.JitFlag.KernelHistory, True)
    cmd, R, R86 = sys.argv[1], sys.argv[2], sys.argv[3]
    a = [x for x in sys.argv[4:] if not x.startswith('--')]
    for d in ('tables', 'masks', 'renders', 'models'):
        os.makedirs(os.path.join(R, 'results', d), exist_ok=True)
    {'reuse': reuse, 'tests': tests, 'account': account, 'pattern': pattern, 'tables': tables, 'operators': operators, 'call': call}[cmd](R, R86, *a)
    os._exit(0)
