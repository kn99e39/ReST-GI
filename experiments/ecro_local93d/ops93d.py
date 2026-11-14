"""WL93D frozen local-operator ownership attribution (no training; frozen checkpoints only).

  python ops93d.py cfgs       <R> <R86>                       configurations (WL93A configs + E2 / E5k995 / E6 sharp pairs), partners
  python ops93d.py ckpt       <R> <R92A> <R93A>               checkpoint state hashes (T_CKPT)
  python ops93d.py fields     <R> <R86> <R92A>                incident fields (source s16, receiver flight(q)) at the selected patches
  python ops93d.py ops        <R> <R86> <R92A> <R93A> <fam> <seed>   effective local operators of the selected patches per configuration
  python ops93d.py functional <R> <R86>                       Task A / C metrics -> results/functional93d.json
  python ops93d.py account    <R> <R86> <R92A> <R93A>         Task B table-level interventions -> results/account93d.json, contrasts93d.json
  python ops93d.py tables     <R> <R86> <R92A> <R93A> <tag>   render tables (arms x families x seeds)
  python ops93d.py tests      <R> <R86> <R92A> <R93A> <R92B>  focused tests
"""
import glob, hashlib, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for d in ('ecro86', 'ecro_attr92', 'ecro_own92a', 'ecro_cf92b', 'ecro_stage93', 'ecro_stage93a', 'ecro_stage93b', 'ecro_stage93c'):
    sys.path.insert(0, os.path.join(HERE, '..', d))
sys.path.insert(0, HERE)
DEV = '--dev' in sys.argv
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_local93d.json')))
FAMS = ['own', 'T0']
SEEDS = [8601, 8602]
ARMS = ['C0', 'C1_src', 'C1_rcv', 'C1_both', 'C2_src', 'C2_rcv', 'C2_both']
NB = 64


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1)


def jl(fn):
    return json.load(open(fn))


# ------------------------------------------------------------------------------------------------ configurations
def cfgs(R, R86):
    import prep93 as PR
    PR.inject(R)
    base = jl(os.path.join(R, 'results', 'configs.json'))['configs']
    C = [dict(c) for c in base]
    extra = ['E5k995_T0', 'E5k995_T1']
    ch = os.path.join(R, 'results', 'e6_choice.json')
    e6 = jl(ch)['chosen'] if os.path.exists(ch) else None
    if e6 is not None:
        extra += [f'E6k{e6}_T0', f'E6k{e6}_T1']
    for t in extra:
        C.append({'cfg': len(C), 'kind': 'sharp', 'tag': t})
    if DEV:
        C = [c for c in C if c['kind'] == 'train_rep'][:2] + [c for c in C if c['kind'] == 'val'][:4] + [c for c in C if c['kind'] == 'sharp']
    part = {}
    for kind in ('train_rep', 'val'):
        ids = [c['cfg'] for c in C if c['kind'] == kind]
        for k, i in enumerate(ids):
            part[str(i)] = ids[(k + 1) % len(ids)]
    tags = {c['tag']: c['cfg'] for c in C if c['kind'] == 'sharp'}
    for t, i in tags.items():
        o = t[:-1] + ('1' if t.endswith('0') else '0')
        if o in tags:
            part[str(i)] = tags[o]
    out = {'configs': C, 'partner': part, 'e6': e6, 'dev': DEV}
    jd(out, os.path.join(R, 'results', 'configs93d.json'))
    print('CFGS', len(C), 'e6', e6, flush=True)


def load_cfgs(R):
    return jl(os.path.join(R, 'results', 'configs93d.json'))


def selected(R86):
    """Selected patches per material group (stationary): S1 panel, S2 sphere, B floor (every 4th), LAM material-3 rank % 13."""
    f = sorted(glob.glob(os.path.join(R86, 'data', 'val', 'g*.npz')))[0]
    pm = np.load(f)['pmat']
    g = {'S1': np.nonzero(pm == 1)[0], 'S2': np.nonzero(pm == 2)[0], 'B': np.nonzero(pm == 0)[0][::4], 'LAM': np.nonzero(pm == 3)[0][::13]}
    idx = np.concatenate([g[k] for k in ('S1', 'S2', 'B', 'LAM')])
    grp = {k: np.arange(sum(len(g[j]) for j in ('S1', 'S2', 'B', 'LAM')[:i]), sum(len(g[j]) for j in ('S1', 'S2', 'B', 'LAM')[:i + 1])) for i, k in enumerate(('S1', 'S2', 'B', 'LAM'))}
    return idx, grp, pm


# ------------------------------------------------------------------------------------------------ models
def load_family(R92A, R93A, fam, seed):
    import torch
    import train93 as TR
    import model93 as MM
    dev = torch.device('cuda')
    own = TR.load_own(R92A, seed, dev)
    if fam == 'own':
        return {'fam': fam, 'own': own, 'net': None}
    net = MM.Local(own, 'tied').to(dev)
    net.load_state_dict(torch.load(os.path.join(R93A, 'results', 'models', f'T0_s{seed}.pt'), map_location=dev))
    return {'fam': fam, 'own': own, 'net': net.eval()}


def local_factors(M, gs):
    """Tied local factors (c, a) of the family at geometry gs."""
    if M['fam'] == 'own':
        f = M['own'].factors(gs)
        return {'c': f['c'], 'a': f['a']}
    f = M['net'].factors(gs)
    return {'c': f['c_s'], 'a': f['a_s']}


def global_term(M, gs, s):
    import model93 as MM
    return MM.global_term(M['own'], gs, s)


def eff_op(f):
    import torch
    return torch.einsum('ndcr,nmcr->nmdc', f['a'], f['c'])          # [N, bi, bo, 3]


def ckpt(R, R92A, R93A, R92B):
    import torch
    import train93 as TR
    out = {}; ok = True
    rep = jl(os.path.join(R92B, 'results', 'repro.json'))['checkpoints']          # WL92B recorded file sha256 of the WL92A own checkpoints
    for fam in FAMS:
        for s in SEEDS:
            M = load_family(R92A, R93A, fam, s)
            if fam == 'own':
                fn = os.path.join(R92A, 'results', 'models', f'own_s{s}.pt')
                h = hashlib.sha256(open(fn, 'rb').read()).hexdigest(); ref = rep[f'own_s{s}']; kind = 'file sha256 vs WL92B repro.json'
            else:
                h = TR.sd_hash(M['net']); ref = jl(os.path.join(R93A, 'results', 'models', f'T0_s{s}.json'))['final_sha256']; kind = 'state sha256 vs WL93A final_sha256'
            out[f'{fam}_s{s}'] = {'loaded': h, 'recorded': ref, 'match': h == ref, 'kind': kind}
            ok &= h == ref
    out['PASS'] = bool(ok)
    jd(out, os.path.join(R, 'results', 'ckpt93d.json'))
    print('CKPT', 'PASS' if ok else 'FAIL', json.dumps({k: v['match'] for k, v in out.items() if isinstance(v, dict)}), flush=True)


# ------------------------------------------------------------------------------------------------ incident fields
def cfg_ref_q(R, c):
    """Reference once-scattered table for the receiver first flight: B5 four-realization mean (WL93A configs) else D q."""
    import qref93a as QR
    fn = os.path.join(R, 'data', 'qref', f"c{c['cfg']:03d}_B5.npz")
    if c['kind'] != 'sharp' or (c.get('tag') in ('E2_T0', 'E2_T1') and os.path.exists(fn)):
        return QR.load_ref(R, c['cfg'], 'B5').mean(0), QR.load_ref(R, c['cfg'], 'B5')
    D = np.load(os.path.join(R, 'results', 'fields', f"D_{c['tag']}.npz"))
    q = D['q'].astype(np.float64)[None]
    return q, None


def fields(R, R86, R92A):
    import torch
    import prep93 as PR
    PR.inject(R)
    import eval93a as EA
    import eval93b as EB
    import train93 as TR
    dev = torch.device('cuda')
    C = load_cfgs(R)['configs']
    idx, grp, pm = selected(R86)
    own = TR.load_own(R92A, SEEDS[0], dev)
    out = {}
    with torch.no_grad():
        for c in C:
            gs, s, G_ = EA.cfg_inputs(R86, c, dev)
            qm, _ = cfg_ref_q(R, c)
            Eq = EB.flight(own.rw, gs, torch.as_tensor(qm.astype(np.float32), device=dev))
            out[f"src_c{c['cfg']:03d}"] = s[:, idx].cpu().numpy().astype(np.float32)
            out[f"rcv_c{c['cfg']:03d}"] = Eq[:, idx].cpu().numpy().astype(np.float32)
    np.savez(os.path.join(R, 'results', 'fields93d.npz'), idx=idx, **out)
    print('FIELDS', len(C), len(idx), flush=True)


def ops(R, R86, R92A, R93A, fam, seed):
    import torch
    import prep93 as PR
    PR.inject(R)
    import eval93a as EA
    dev = torch.device('cuda')
    C = load_cfgs(R)['configs']
    idx, grp, pm = selected(R86)
    M = load_family(R92A, R93A, fam, int(seed))
    out = {}
    with torch.no_grad():
        for c in C:
            gs, s, G_ = EA.cfg_inputs(R86, c, dev)
            A = eff_op(local_factors(M, gs))[idx]
            out[f"c{c['cfg']:03d}"] = A.cpu().numpy().astype(np.float32)
    os.makedirs(os.path.join(R, 'results', 'ops'), exist_ok=True)
    np.savez(os.path.join(R, 'results', 'ops', f'{fam}_s{seed}.npz'), idx=idx, **out)
    print('OPS', fam, seed, len(C), flush=True)


# ------------------------------------------------------------------------------------------------ Task A / C analysis (numpy)
def functional(R, R86):
    import diag92 as DG
    CF = load_cfgs(R); C = CF['configs']; part = CF['partner']
    idx, grp, pm = selected(R86)
    _, Mq = DG.local_matrices(); zb = DG.flux_w()
    Msel = Mq[pm[idx]].astype(np.float64)                                    # [n, bi, bo, 3]
    Fz = np.load(os.path.join(R, 'results', 'fields93d.npz'))
    kinds = {c['cfg']: c['kind'] for c in C}
    probe_ids = [c['cfg'] for c in C if c['kind'] == 'train_rep']
    eval_ids = [c['cfg'] for c in C if c['kind'] in ('val', 'sharp')]
    pairs = sorted({tuple(sorted((int(i), int(j)))) for i, j in part.items() if int(i) in eval_ids and int(j) in eval_ids})
    W2 = (zb[:, None] * zb[None, :])[None, :, :, None]

    def basis(gname, use):
        X = []
        for i in probe_ids:
            x = Fz[f"{use}_c{i:03d}"][:, grp[gname]]                         # [L, n, 64, 3]
            X.append(np.transpose(x, (0, 1, 3, 2)).reshape(-1, NB))
        X = np.concatenate(X).astype(np.float64)
        U, Sv, Vt = np.linalg.svd(X, full_matrices=False)
        e = np.cumsum(Sv ** 2) / np.sum(Sv ** 2)
        d = int(min(np.searchsorted(e, 0.99) + 1, 48))
        return Vt[:d].T, {'dim': d, 'energy_at_dim': float(e[d - 1]), 'cond': float(Sv[0] / max(Sv[d - 1], 1e-30)), 'samples': int(X.shape[0])}
    bases = {(g, u): basis(g, u) for g in grp for u in ('src', 'rcv')}

    def wnorm2(A, Pm=None):
        """flux-weighted squared norm of operator(s) A [..., bi, bo, 3]; canonical (Pm None) uses X = A z_i z_o; workload uses out = P^T A, z_o-weighted."""
        if Pm is None:
            return float(((A * W2) ** 2).sum())
        Y = np.einsum('mk,...mdc->...kdc', Pm, A) * zb[None, :, None]
        return float((Y ** 2).sum())
    res = {'bases': {f'{g}:{u}': v[1] for (g, u), v in bases.items()}, 'pairs': pairs, 'families': {}}
    OPS = {}
    for fam in FAMS:
        for s in SEEDS:
            OPS[(fam, s)] = np.load(os.path.join(R, 'results', 'ops', f'{fam}_s{s}.npz'))
    for fam in FAMS:
        res['families'][fam] = {}
        for gname, gi in grp.items():
            Mg = Msel[gi]
            rec = {}
            for probe in ('canonical', 'src', 'rcv'):
                Pm = None if probe == 'canonical' else bases[(gname, probe)][0]
                nM = wnorm2(Mg, Pm) or 1e-30
                per_seed = {}
                for s in SEEDS:
                    O = OPS[(fam, s)]
                    A = {i: O[f'c{i:03d}'][gi].astype(np.float64) for i in eval_ids}
                    Abar = np.mean([A[i] for i in eval_ids], 0)
                    e_phys = math.sqrt(np.mean([wnorm2(A[i] - Mg, Pm) for i in eval_ids]) / nM)
                    d_geo = float(np.mean([math.sqrt(wnorm2(A[i] - A[j], Pm) / nM) for i, j in pairs]))
                    s_geo = sum(wnorm2(A[i] - Abar, Pm) for i in eval_ids) / max(sum(wnorm2(A[i] - Mg, Pm) for i in eval_ids), 1e-30)
                    e_common = math.sqrt(wnorm2(Abar - Mg, Pm) / nM)
                    ring = None
                    if probe == 'canonical':
                        ring = []
                        for r in range(4):
                            rows = slice(16 * r, 16 * (r + 1))
                            ring.append(math.sqrt(np.mean([float(((((A[i] - Mg) * W2)[:, rows]) ** 2).sum()) for i in eval_ids]) / max(float((((Mg * W2)[:, rows]) ** 2).sum()), 1e-30)))
                    sharp_pairs = [(i, j) for i, j in pairs if kinds[i] == 'sharp']
                    d_geo_sharp = float(np.mean([math.sqrt(wnorm2(A[i] - A[j], Pm) / nM) for i, j in sharp_pairs])) if sharp_pairs else None
                    per_seed[s] = {'E_phys': e_phys, 'D_geo': d_geo, 'D_geo_sharp_pairs': d_geo_sharp, 'S_geo': s_geo, 'E_common': e_common, 'E_phys_by_ring': ring}
                A1 = OPS[(fam, SEEDS[0])]; A2 = OPS[(fam, SEEDS[1])]
                d_seed = math.sqrt(np.mean([wnorm2(A1[f'c{i:03d}'][gi].astype(np.float64) - A2[f'c{i:03d}'][gi].astype(np.float64), Pm) for i in eval_ids]) / nM)
                rec[probe] = {'per_seed': per_seed, 'D_seed': d_seed}
            res['families'][fam][gname] = rec
        # Task C: in / out of the material-level span on evaluation incident fields (S1, S2), per use
        sub = {}
        for gname in ('S1', 'S2'):
            gi = grp[gname]; Mg = Msel[gi]
            for use in ('src', 'rcv'):
                V = bases[(gname, use)][0]; Pv = V @ V.T
                for s in SEEDS:
                    O = OPS[(fam, s)]
                    a = {'in': 0.0, 'out': 0.0, 'm': 0.0, 'xout': 0.0, 'x': 0.0, 'gd_in': 0.0, 'gd_out': 0.0, 'gd_m': 0.0}
                    for i in eval_ids:
                        x = Fz[f'{use}_c{i:03d}'][:, gi].astype(np.float64)        # [L, n, 64, 3]
                        xin = np.einsum('mk,lnkc->lnmc', Pv, x); xo = x - xin
                        E = O[f'c{i:03d}'][gi].astype(np.float64) - Mg
                        app = lambda Aop, xx: np.einsum('lnmc,nmdc->lndc', xx, Aop) * zb[None, None, :, None]
                        a['in'] += float((app(E, xin) ** 2).sum()); a['out'] += float((app(E, xo) ** 2).sum()); a['m'] += float((app(Mg, x) ** 2).sum())
                        a['xout'] += float((xo ** 2).sum()); a['x'] += float((x ** 2).sum())
                    for i, j in pairs:
                        D_ = O[f'c{i:03d}'][gi].astype(np.float64) - O[f'c{j:03d}'][gi].astype(np.float64)
                        x = Fz[f'{use}_c{i:03d}'][:, gi].astype(np.float64)
                        xin = np.einsum('mk,lnkc->lnmc', Pv, x); xo = x - xin
                        app = lambda Aop, xx: np.einsum('lnmc,nmdc->lndc', xx, Aop) * zb[None, None, :, None]
                        a['gd_in'] += float((app(D_, xin) ** 2).sum()); a['gd_out'] += float((app(D_, xo) ** 2).sum()); a['gd_m'] += float((app(Mg, x) ** 2).sum())
                    sub[f'{gname}:{use}:s{s}'] = {'e_in': math.sqrt(a['in'] / a['m']), 'e_out': math.sqrt(a['out'] / a['m']), 'x_out_energy_share': a['xout'] / a['x'],
                                                  'drift_in': math.sqrt(a['gd_in'] / a['gd_m']), 'drift_out': math.sqrt(a['gd_out'] / a['gd_m'])}
        res['families'][fam]['subspace'] = sub
    jd(res, os.path.join(R, 'results', 'functional93d.json'))
    for fam in FAMS:
        f = res['families'][fam]
        print('FUNCTIONAL', fam, json.dumps({g: {p: {s: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f[g][p]['per_seed'][s].items() if k in ('E_phys', 'D_geo', 'S_geo')} for s in SEEDS} for p in ('canonical', 'src')} for g in ('S1', 'LAM')}), 'D_seed S1', round(f['S1']['src']['D_seed'], 4), flush=True)


# ------------------------------------------------------------------------------------------------ Task B interventions (tables)
def arm_eval(M, f_dep, f_par, Mn, gs, s, Eq, arm, rw):
    """Single flight with the arm's source / receiver local uses; global term fixed (deployed factors)."""
    import torch
    import eval93b as EB
    ap = lambda f, x: torch.einsum('ndcr,lncr->lndc', f['a'], torch.einsum('nmcr,lnmc->lncr', f['c'], x))
    ex = lambda x: torch.einsum('lnmc,nmdc->lndc', x, Mn)
    src = {'C0': 'dep', 'C1_src': 'par', 'C1_rcv': 'dep', 'C1_both': 'par', 'C2_src': 'm64', 'C2_rcv': 'dep', 'C2_both': 'm64'}[arm]
    rcv = {'C0': 'dep', 'C1_src': 'dep', 'C1_rcv': 'par', 'C1_both': 'par', 'C2_src': 'dep', 'C2_rcv': 'm64', 'C2_both': 'm64'}[arm]
    use = lambda k, x: ex(x) if k == 'm64' else ap(f_dep if k == 'dep' else f_par, x)
    Bq = use(src, s)
    E1 = EB.flight(rw, gs, Bq)
    S = use(rcv, E1)
    Rq = use(rcv, Eq)
    return S, Bq, Rq, {'src': src, 'rcv': rcv}


def account(R, R86, R92A, R93A):
    import torch
    import prep93 as PR
    PR.inject(R)
    import diag92 as DG
    import qref93a as QR
    import eval93a as EA
    import eval93b as EB
    dev = torch.device('cuda')
    CF = load_cfgs(R); C = CF['configs']; part = CF['partner']
    _, Mq = DG.local_matrices()
    models = {(fam, s): load_family(R92A, R93A, fam, s) for fam in FAMS for s in SEEDS}
    GS = {}
    with torch.no_grad():
        for c in C:
            GS[c['cfg']] = EA.cfg_inputs(R86, c, dev)
        per = {}
        for c in C:
            i = c['cfg']; gs, s, G_ = GS[i]; gsp = GS[part[str(i)]][0]
            t, ta, tb, t2, t2a, t2b = EB.targets(R86, R92A, R, c)
            qm, Q4 = cfg_ref_q(R, c)
            pmask = EA.panel_mask(R86, c)
            Mn = torch.as_tensor(Mq[G_.PMAT].astype(np.float32), device=dev)
            for (fam, sd), M in models.items():
                f_dep = local_factors(M, gs); f_par = local_factors(M, gsp)
                G = global_term(M, gs, s)
                rw = M['own'].rw
                Eq = EB.flight(rw, gs, torch.as_tensor(qm.astype(np.float32), device=dev))
                Gn = G.double().cpu().numpy()
                for arm in ARMS:
                    S, Bq, Rq, _ = arm_eval(M, f_dep, f_par, Mn, gs, s, Eq, arm, rw)
                    Sn, Bqn, Rqn = S.double().cpu().numpy(), Bq.double().cpu().numpy(), Rq.double().cpu().numpy()
                    W = Sn + Gn
                    d = {'cfg': i, 'single_l1': EB.l1s(Sn, t2), 'whole_l1': EB.l1s(W, t), 'global_l1': EB.l1s(Gn, t - t2), 'recv_q_l1': EB.l1s(Rqn, t2),
                         'single_signed': [float((Sn - t2).sum()), float(t2.sum())], 'whole_signed': [float((W - t).sum()), float(t.sum())]}
                    if Q4 is not None:
                        d['src'] = QR.stat(Bqn, Q4, pmask)
                    else:
                        d['src_l1_vs_Dq'] = [float(np.abs(Bqn[:, pmask] - qm[:, pmask]).sum()), float(qm[:, pmask].sum())]
                    if t2b is not None:
                        d['single_xl2'] = EB.xl2(Sn, t2a, t2b)
                    per.setdefault(f'{fam}_s{sd}:{arm}', []).append(d)
    groups = {g: [k for k, c in enumerate(C) if c['kind'] == g] for g in ('val', 'train_rep', 'sharp')}
    summ = {}
    for key, rows in per.items():
        summ[key] = {}
        for g, ids in groups.items():
            rr = [rows[k] for k in ids]
            if not rr:
                continue
            o = {k.upper(): EB.ratio(rr, k) for k in ('single_l1', 'whole_l1', 'global_l1', 'recv_q_l1', 'single_signed', 'whole_signed')}
            if all('src' in r for r in rr):
                o['SRC'] = sum(r['src']['l1'] for r in rr) / sum(r['src']['sq'] for r in rr)
            summ[key][g] = o
    jd({'summary': summ, 'per_cfg': per, 'configs': C}, os.path.join(R, 'results', 'account93d.json'))
    # paired bootstrap contrasts (validation), arm - C0
    vidx = groups['val']
    rng = np.random.default_rng(93004); B = [rng.integers(0, len(vidx), len(vidx)) for _ in range(2000)]
    pooled = lambda rows, k: sum(r[k][0] for r in rows) / sum(r[k][1] for r in rows)
    con = {}
    for fam in FAMS:
        for sd in SEEDS:
            base = [per[f'{fam}_s{sd}:C0'][k] for k in vidx]
            for arm in ARMS[1:]:
                x = [per[f'{fam}_s{sd}:{arm}'][k] for k in vidx]
                for key in ('single_l1', 'whole_l1'):
                    d0 = pooled(x, key) - pooled(base, key)
                    bs = [pooled([x[j] for j in b], key) - pooled([base[j] for j in b], key) for b in B]
                    con[f'{fam}_s{sd}:{arm}-C0:{key}'] = {'diff': d0, 'se': float(np.std(bs, ddof=1)), 'ci95': [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
    jd(con, os.path.join(R, 'results', 'contrasts93d.json'))
    print('ACCOUNT', json.dumps({k: round(v['val']['SINGLE_L1'], 4) for k, v in summ.items() if 'val' in v}) if not DEV else 'dev', flush=True)


# ------------------------------------------------------------------------------------------------ render tables
def tables(R, R86, R92A, R93A, tag):
    import torch
    import prep93 as PR
    fx92 = PR.inject(R)
    import ecro86 as EC
    import model86 as M86
    import diag92 as DG
    import eval93b as EB
    PT, FX, S = fx92.PT, fx92.FX, fx92.S
    dev = torch.device('cuda')
    _, Mq = DG.local_matrices()
    geom, ls, e, cfg = fx92.cfg_ls(tag)
    o = tag[:-1] + ('0' if tag.endswith('1') else '1') if tag[-2:] in ('T0', 'T1') else None
    other = fx92.cfg_ls(o)[0] if o else None

    def gstate(g):
        ctx = PT.Ctx(g); st = EC.build_state(ctx)
        return ctx, M86.GeoState(ctx.G, st.hit, st.dbin, dev)
    ctx, gs = gstate(geom); _, gsp = gstate(other)
    lt = PT.Lights(FX['light_sets'][ls])
    s = torch.as_tensor(ctx.source(lt), device=dev).view(1, ctx.N, S.NB, 3)
    Mn = torch.as_tensor(Mq[ctx.G.PMAT].astype(np.float32), device=dev)
    D = np.load(os.path.join(R, 'results', 'fields', f'D_{tag}.npz'))
    Eq_q = torch.as_tensor(D['q'].astype(np.float32), device=dev).view(1, ctx.N, S.NB, 3)
    arms = ARMS if tag.endswith('T1') else ['C0', 'C2_src', 'C2_rcv', 'C2_both']
    out = {}
    with torch.no_grad():
        for fam in FAMS:
            for sd in SEEDS:
                M = load_family(R92A, R93A, fam, sd)
                f_dep = local_factors(M, gs); f_par = local_factors(M, gsp)
                G = global_term(M, gs, s); rw = M['own'].rw
                Eq = EB.flight(rw, gs, Eq_q)
                for arm in arms:
                    S_, _, _, _ = arm_eval(M, f_dep, f_par, Mn, gs, s, Eq, arm, rw)
                    out[f'B_{fam}_s{sd}_{arm}'] = (S_ + G)[0].cpu().numpy().astype(np.float32)
    os.makedirs(os.path.join(R, 'results', 'tables'), exist_ok=True)
    np.savez_compressed(os.path.join(R, 'results', 'tables', f'I_{tag}.npz'), **out)
    print('tables', tag, len(out), flush=True)


# ------------------------------------------------------------------------------------------------ tests
def tests(R, R86, R92A, R93A, R92B):
    import torch
    import prep93 as PR
    fx92 = PR.inject(R)
    import ecro86 as EC
    import model86 as M86
    import diag92 as DG
    import eval93b as EB
    PT, FX, S = fx92.PT, fx92.FX, fx92.S
    dev = torch.device('cuda')
    _, Mq = DG.local_matrices()
    out = {}; ok = True
    # T_M64
    I = np.eye(NB)
    out['T_M64'] = float(max(np.abs(np.einsum('km,mdc->kdc', I, Mq[m]) - Mq[m]).max() for m in range(Mq.shape[0])))
    ok &= out['T_M64'] == 0.0
    geom1, ls, e, cfg = fx92.cfg_ls('E2_T1'); geom0 = fx92.cfg_ls('E2_T0')[0]
    ctx = PT.Ctx(geom1); st = EC.build_state(ctx); gs = M86.GeoState(ctx.G, st.hit, st.dbin, dev)
    ctx0 = PT.Ctx(geom0); st0 = EC.build_state(ctx0); gsp = M86.GeoState(ctx0.G, st0.hit, st0.dbin, dev)
    s = torch.as_tensor(ctx.source(PT.Lights(FX['light_sets'][ls])), device=dev).view(1, ctx.N, S.NB, 3)
    Mn = torch.as_tensor(Mq[ctx.G.PMAT].astype(np.float32), device=dev)
    q = torch.as_tensor(np.load(os.path.join(R, 'results', 'fields', 'D_E2_T1.npz'))['q'].astype(np.float32), device=dev).view(1, ctx.N, S.NB, 3)
    # T_FRAME
    G1, G0 = ctx.G, ctx0.G
    stat = ~np.isin(G1.PMAT, [4, 5])
    out['T_FRAME'] = {'stationary_max_diff': float(max(np.abs(G1.PC[stat] - G0.PC[stat]).max(), np.abs(G1.PN[stat] - G0.PN[stat]).max(), np.abs(G1.PT[stat] - G0.PT[stat]).max())),
                      'box_patches': int((~stat).sum())}
    ok &= out['T_FRAME']['stationary_max_diff'] == 0.0
    dig = lambda f: hashlib.sha256(np.concatenate([f['c'].detach().cpu().numpy().ravel(), f['a'].detach().cpu().numpy().ravel()]).tobytes()).hexdigest()[:16]
    with torch.no_grad():
        for fam in FAMS:
            for sd in SEEDS:
                M = load_family(R92A, R93A, fam, sd)
                f_dep = local_factors(M, gs); f_par = local_factors(M, gsp); rw = M['own'].rw
                G = global_term(M, gs, s); Eq = EB.flight(rw, gs, q)
                o = {}
                # T_IDENTITY: C0 through the substitution path = deployed model output
                S0, _, _, _ = arm_eval(M, f_dep, f_par, Mn, gs, s, Eq, 'C0', rw)
                if fam == 'own':
                    dep = M['own'](gs, s)
                else:
                    dep = M['net'].single(M['net'].factors(gs), gs, s)[0] + G
                o['identity_rel'] = float(((S0 + G) - dep).abs().max() / dep.abs().max())
                # T_GLOBAL: deployed global equals the model's own global
                if fam == 'own':
                    o['global_vs_deployed'] = float((G - M['own'].apply(M['own'].factors(gs), gs, s, use_single=False)).abs().max())
                else:
                    import model93 as MM
                    o['global_vs_deployed'] = float((G - MM.global_term(M['own'], gs, s)).abs().max())
                # T_SWAP / T_ISOLATION: digests of the factors actually used per use
                o['digest_dep'], o['digest_par'] = dig(f_dep), dig(f_par)
                o['partner_differs'] = o['digest_dep'] != o['digest_par']
                # T_USED: doubling the partner factors changes the C1_both output; doubling deployed changes C0; M64 used in C2
                f2 = {'c': 2 * f_par['c'], 'a': f_par['a']}
                S1, _, _, _ = arm_eval(M, f_dep, f_par, Mn, gs, s, Eq, 'C1_both', rw)
                S1b, _, _, _ = arm_eval(M, f_dep, f2, Mn, gs, s, Eq, 'C1_both', rw)
                S1s, _, _, _ = arm_eval(M, f_dep, f2, Mn, gs, s, Eq, 'C1_rcv', rw)     # source deployed, receiver partner (doubled c) -> 2x
                S1r, _, _, _ = arm_eval(M, f_dep, f_par, Mn, gs, s, Eq, 'C1_rcv', rw)
                o['used_partner'] = float((S1b - S1).abs().max() / S1.abs().max())
                o['isolation_rcv_scales'] = float(((S1s - 2 * S1r).abs().max()) / S1r.abs().max())
                S2, _, _, _ = arm_eval(M, f_dep, f_par, 2 * Mn, gs, s, Eq, 'C2_both', rw)
                S2r, _, _, _ = arm_eval(M, f_dep, f_par, Mn, gs, s, Eq, 'C2_both', rw)
                o['used_m64_both_scales4'] = float(((S2 - 4 * S2r).abs().max()) / S2r.abs().max())
                o['det'] = float((arm_eval(M, f_dep, f_par, Mn, gs, s, Eq, 'C1_src', rw)[0] - arm_eval(M, f_dep, f_par, Mn, gs, s, Eq, 'C1_src', rw)[0]).abs().max())
                if fam == 'own':
                    cf = np.load(os.path.join(R92B, 'results', 'tables', 'CF_E2_T1.npz'))
                    Y = cf[f'X_M12_own_s{sd}'].astype(np.float64)
                    o['C2_both_vs_wl92b_X_M12'] = float(np.abs((S2r + G)[0].cpu().numpy() - Y).max() / np.abs(Y).max())
                ok &= o['identity_rel'] <= 1e-6 and o['global_vs_deployed'] == 0.0 and o['partner_differs'] and o['used_partner'] > 1e-3 and o['isolation_rcv_scales'] <= 1e-5 and o['used_m64_both_scales4'] <= 1e-5 and o['det'] == 0.0
                if fam == 'own':
                    ok &= o['C2_both_vs_wl92b_X_M12'] <= 1e-5
                out[f'{fam}_s{sd}'] = o
    out['PASS'] = bool(ok)
    jd(out, os.path.join(R, 'results', 'tests93d.json'))
    print('TESTS', 'PASS' if ok else 'FAIL', flush=True)


if __name__ == '__main__':
    import fx92
    fx92.dr.set_flag(fx92.dr.JitFlag.KernelHistory, True)
    cmd = sys.argv[1]
    a = [x for x in sys.argv[2:] if not x.startswith('--')]
    for d in ('tables', 'masks', 'renders', 'ops', 'fields'):
        os.makedirs(os.path.join(a[0], 'results', d), exist_ok=True)
    {'cfgs': cfgs, 'ckpt': ckpt, 'fields': fields, 'ops': ops, 'functional': functional, 'account': account, 'tables': tables, 'tests': tests}[cmd](*a)
    os._exit(0)
