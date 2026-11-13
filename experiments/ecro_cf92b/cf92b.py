"""WL92B: causal attribution of ECRO OWN's remaining sharp / source error and an independently qualified held-out edit E5
(protocol ecro_cf92b). Nothing is trained; every model parameter is frozen.

  python cf92b.py repro    <R> <R86> <R92A>          WL92A reproduction (tables, render E_H, validation component errors)
  python cf92b.py cftables <R> <R86> <R92A> <tag>    per model: original terms and the declared counterfactual tables + reading metrics
  python cf92b.py local    <R> <R86> <R92A>          learned local matrices a c^T across configurations / seeds vs frozen ER
  python cf92b.py e5gen    <R> <R86>                 ordered geometry-only E5 candidates, disjointness, masks
  python cf92b.py e5admit  <R> <R86> <k>             admission of candidate k (WL86 stage-0 + sharp family)
  python cf92b.py e5tables <R> <R86> <R92A> <tag>    tables of every model on an E5 configuration (+ STALE, ORACLE, X_EXP)
  python cf92b.py ofield   <R> <R86> <tag>           oracle field of an E5 configuration
  python cf92b.py call     <R> <R86> <module.func> <args...>   run a WL92 function with the E5 candidates injected
"""
import glob, hashlib, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_attr92'))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_own92a'))
sys.path.insert(0, HERE)
DEV = '--dev' in sys.argv
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_cf92b.json')))
MODELS = ['ecro_s8601', 'ecro_s8602', 'sum_s8601', 'sum_s8602', 'own_s8601', 'own_s8602']
CFS = ['ORIG', 'X_L2', 'X_L3', 'X_M1', 'X_M2', 'X_M12', 'X_Q', 'X_SHQ']


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1)


def jl(fn):
    return json.load(open(fn))


def sha(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def inject(R):
    """E4 (WL92) and every recorded E5 candidate as FX['E5k<k>'] (light L1 / L2 by parity)."""
    import fx92
    fx92.inject(R)
    fn = os.path.join(R, 'results', 'e5_candidates.json')
    if os.path.exists(fn):
        for c in jl(fn)['geometry_admissible']:
            fx92.FX[f"E5k{c['k']}"] = {'T0': c['T0'], 'T1': c['T1'], 'moved': [c['moved']], 'light_set': c['light_set'], 'role': 'WL92B E5 candidate'}
    return fx92


def model_path(R86, R92A, name):
    return os.path.join(R86 if name.startswith('ecro') else R92A, 'results', 'models', f'{name}.pt')


def load_model(R86, R92A, name):
    import torch
    import ecro86 as EC
    import model86 as M
    m = M.ECRO().to(EC.dev)
    m.load_state_dict(torch.load(model_path(R86, R92A, name), map_location=EC.dev))
    return m.eval()


def hm(D, keys):
    return sum(0.5 * (D[f'{k}_a'].astype(np.float64) + D[f'{k}_b'].astype(np.float64)) for k in keys)


# ------------------------------------------------------------------------------------------------ reproduction
def repro(R, R86, R92A):
    import torch
    fx92 = inject(R)
    import ecro86 as EC
    import model86 as M
    import diag92 as DG
    import analyze92 as AN
    PT, FX, S = fx92.PT, fx92.FX, fx92.S
    out = {'checkpoints': {n: sha(model_path(R86, R92A, n)) for n in MODELS}, 'tables': {}, 'renders': {}, 'field': {}}
    with torch.no_grad():
        for tag in ('E2_T1', 'E2_T0', 'E1_T1_LSPOT', 'E4_T1'):
            geom, ls, e, cfg = fx92.cfg_ls(tag)
            ctx = PT.Ctx(geom); st = EC.build_state(ctx)
            s = torch.as_tensor(ctx.source(PT.Lights(FX['light_sets'][ls])), device=EC.dev).view(1, ctx.N, S.NB, 3)
            gs = M.GeoState(ctx.G, st.hit, st.dbin, EC.dev)
            I = np.load(os.path.join(R92A, 'results', 'tables', f'I_{tag}.npz'))
            for n in MODELS:
                mdl = load_model(R86, R92A, n)
                f = mdl.factors(gs)
                Bs = mdl.apply(f, gs, s, use_global=False)[0].cpu().numpy(); Bg = mdl.apply(f, gs, s, use_single=False)[0].cpu().numpy()
                sp = DG.single_panel(mdl, f, gs, s)[0].cpu().numpy()
                d = max(float(np.abs(Bs + Bg - I[f'B_{n}']).max() / I[f'B_{n}'].max()), float(np.abs(Bs - I[f'S_{n}']).max() / I[f'S_{n}'].max()),
                        float(np.abs(Bg - I[f'G_{n}']).max() / I[f'G_{n}'].max()), float(np.abs(sp - I[f'SP_{n}']).max() / max(I[f'SP_{n}'].max(), 1e-30)))
                out['tables'][f'{n}:{tag}'] = d
    F = jl(os.path.join(R92A, 'results', 'final92a.json'))
    for n in MODELS:
        for e in ('E1', 'E2', 'E4'):
            for t in ('T0', 'T1'):
                x = AN.hl(R92A, f'I92A_{n}_{e}_{t}')
                rf = A_ref(R, R86, R92A, f'{e}_{t}')
                T = AN.eh(rf, x, AN.mask_of(R, R86, f'{e}_T1'))
                ref = F['models'][n]['fidelity'][e][t]
                out['renders'][f'{n}:{e}_{t}'] = max(abs(T[k]['E_H'] - ref[k]['E_H']) for k in ('C', 'stationary_all', 'D'))
        x = AN.hl(R92A, f'I92A_{n}_E1_T1_LSPOT')
        T = AN.eh(A_ref(R, R86, R92A, 'E1_T1_LSPOT'), x, AN.mask_of(R, R86, 'E1_T1_LSPOT'))
        out['renders'][f'{n}:E1_T1_LSPOT'] = abs(T['stationary_all']['E_H'] - F['models'][n]['sources']['E1_T1_LSPOT']['stationary_all']['E_H'])
    # validation component errors (55 WL92A validation geometries)
    import own92a as OW
    its = OW.load_items(R92A, R86, 'val', 8 if DEV else None)
    fe = jl(os.path.join(R92A, 'results', 'fieldeval.json'))
    with torch.no_grad():
        for n in MODELS:
            mdl = load_model(R86, R92A, n)
            sn = sd_ = gn = gd = 0.0
            for it in its:
                gs = it.gs(); s = torch.as_tensor(it.it.s, device=EC.dev)
                f = mdl.factors(gs)
                Bs = mdl.apply(f, gs, s, use_global=False); Bg = mdl.apply(f, gs, s, use_single=False)
                t = it.it.target(); t2 = it.t2(EC.dev)
                sn += float((Bs - t2).abs().sum()); sd_ += float(t2.sum()); gn += float((Bg - (t - t2)).abs().sum()); gd += float((t - t2).sum())
            mine = {'single': sn / sd_, 'global': gn / gd}
            ref = fe['models'][n]['val']
            out['field'][n] = {'mine': mine, 'wl92a': {'single': ref['single_vs_L2_rel_l1'], 'global': ref['global_vs_L3_rel_l1']},
                               'max_rel_diff': max(abs(mine['single'] / ref['single_vs_L2_rel_l1'] - 1), abs(mine['global'] / ref['global_vs_L3_rel_l1'] - 1))}
    ok = (max(out['tables'].values()) <= 1e-4 and max(out['renders'].values()) <= 1e-9 and (DEV or max(v['max_rel_diff'] for v in out['field'].values()) <= 1e-4))
    out['PASS'] = bool(ok)
    jd(out, os.path.join(R, 'results', 'repro.json'))
    print('REPRO', 'PASS' if ok else 'FAIL', max(out['tables'].values()), max(out['renders'].values()), max(v['max_rel_diff'] for v in out['field'].values()), flush=True)


def A_ref(R, R86, R92A, tag):
    import analyze86 as A
    for root in (R, R92A, R86):
        if glob.glob(os.path.join(root, 'results', 'renders', f'REF_ER_{tag}_r*.npz')):
            return A.load(root, f'REF_ER_{tag}')
    raise FileNotFoundError(tag)


# ------------------------------------------------------------------------------------------------ counterfactual tables
def flight(mdl, gs, L):
    """First-flight irradiance by incident bin from an outgoing-radiance table L [N, NB, 3] through the deployed state rays."""
    import fx92
    S = fx92.S
    N, NB, K, KAPPA = gs.N, S.NB, S.K, S.KAPPA
    ok = (gs.hit >= 0).view(N * K, 1)
    idx = (gs.hit.clamp(min=0) * NB + gs.dbin).view(-1)
    Lk = L.reshape(N * NB, 3)[idx] * ok
    return (Lk.view(N, NB, KAPPA, 3) * mdl.rw.view(1, NB, KAPPA, 1)).sum(2)


def learned_local(f, E):
    import torch
    return torch.einsum('ndcr,ncr->ndc', f['a'], torch.einsum('nmcr,nmc->ncr', f['c'], E))


def exact_local(Mn, E):
    import torch
    return torch.einsum('nmc,nmdc->ndc', E, Mn)


def reading(W, X, T, G):
    """Signed reader error of table X against truth T (fractions of h) with the WL86A level / directional split."""
    Wr = W.astype(np.float64).reshape(G.N, -1, 3)
    h = float((Wr * T).sum())
    tot = float((Wr * (X - T)).sum())
    rp = X.sum((1, 2)) / np.maximum(T.sum((1, 2)), 1e-30)
    lvl = float(((rp - 1)[:, None, None] * Wr * T).sum())
    return {'signed': tot / h, 'level': lvl / h, 'directional': (tot - lvl) / h, 'h': h}


def cftables(R, R86, R92A, tag):
    import torch
    fx92 = inject(R)
    import ecro86 as EC
    import model86 as M
    import diag92 as DG
    import pt86a as PA
    PT, FX, S = fx92.PT, fx92.FX, fx92.S
    dev = EC.dev
    geom, ls, e, cfg = fx92.cfg_ls(tag)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls]); G = ctx.G
    st = EC.build_state(ctx)
    D = np.load(os.path.join(R, 'results', 'fields', f'D_{tag}.npz'))
    Wz = np.load(os.path.join(R, 'results', 'fields', f'W_{tag}.npz'))
    assert np.array_equal(st.hit, D['hit']) and np.array_equal(st.dbin, D['dbin']), 'state mismatch'
    L2T, L3T, E2P = hm(D, ('e2_p', 'e2_o')), hm(D, ('e3_p', 'e3_o')), hm(D, ('e2_p',))
    T = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=dev)
    s = T(D['s16']).view(1, G.N, S.NB, 3)
    s_hq = T(PA.source_hq(ctx, lt, 64 if DEV else 1024)).view(1, G.N, S.NB, 3)
    _, Mq = DG.local_matrices()
    import xport87 as X
    Mn = T(Mq[G.PMAT].astype(np.float32))                                     # [N, bi, bo, 3]
    encq = T(X.enc_q(G, D['s16'], Mq))                                         # exact-local once-scattered table from the deployed encoding
    q = T(D['q'])
    gs = M.GeoState(G, st.hit, st.dbin, dev)
    out = {'ORA': (L2T + L3T).astype(np.float32)}
    meta = {'tag': tag, 'cf_identity': {}, 'reading': {}, 'linear': {}}
    regs = [k for k in ('C', 'obj_panel', 'stationary_all', 'D') if k in Wz.files]
    with torch.no_grad():
        for n in MODELS:
            mdl = load_model(R86, R92A, n)
            f = mdl.factors(gs)
            S_ = mdl.apply(f, gs, s, use_global=False)[0]; G_ = mdl.apply(f, gs, s, use_single=False)[0]
            Bq = learned_local(f, s[0])
            E1 = flight(mdl, gs, Bq)
            S_id = learned_local(f, E1)
            meta['cf_identity'][n] = float((S_id - S_).abs().max() / S_.abs().max())
            tabs = {'ORIG': S_ + G_, 'X_L2': T(L2T) + G_, 'X_L3': S_ + T(L3T),
                    'X_M1': learned_local(f, flight(mdl, gs, encq)) + G_,
                    'X_M2': exact_local(Mn, E1) + G_,
                    'X_M12': exact_local(Mn, flight(mdl, gs, encq)) + G_,
                    'X_Q': learned_local(f, flight(mdl, gs, q)) + G_,
                    'X_SHQ': mdl(gs, s_hq)[0]}
            # linearity / non-negativity of the diagnostic compositions (scaled source)
            meta['linear'][n] = {'X_M12_scale': float((exact_local(Mn, flight(mdl, gs, 2 * encq)) - 2 * exact_local(Mn, flight(mdl, gs, encq))).abs().max() / exact_local(Mn, flight(mdl, gs, encq)).abs().max()),
                                 'min': min(float(v.min()) for v in tabs.values())}
            parts = {'S': S_, 'G': G_, 'SP': DG.single_panel(mdl, f, gs, s)[0], 'Bq': Bq}
            for k, v in tabs.items():
                out[f'{k}_{n}'] = v.cpu().numpy().astype(np.float32)
            for k, v in parts.items():
                out[f'{k}_{n}'] = v.cpu().numpy().astype(np.float32)
            rd = {}
            for rk in regs:
                W = Wz[rk]
                rd[rk] = {k: reading(W, out[f'{k}_{n}'].astype(np.float64), L2T + L3T, G) for k in CFS}
                rd[rk]['single_vs_L2'] = reading(W, out[f'S_{n}'].astype(np.float64), L2T, G)
                rd[rk]['global_vs_L3'] = reading(W, out[f'G_{n}'].astype(np.float64), L3T, G)
                h = reading(W, L2T + L3T, L2T + L3T, G)['h']
                Wr = W.astype(np.float64).reshape(G.N, -1, 3)
                rd[rk]['panel_family_share'] = float((Wr * E2P).sum()) / h
                rd[rk]['panel_family_delivered'] = float((Wr * out[f'SP_{n}']).sum()) / max(float((Wr * E2P).sum()), 1e-30)
            meta['reading'][n] = rd
            # field-level errors of the stages (all patches; panel patches)
            pm = G.PMAT == 1
            qn = D['q'].astype(np.float64); Bqn = out[f'Bq_{n}'].astype(np.float64)
            meta.setdefault('field', {})[n] = {'Bq_vs_q_rel_l1': float(np.abs(Bqn - qn).sum() / qn.sum()), 'Bq_vs_q_panel_rel_l1': float(np.abs(Bqn[pm] - qn[pm]).sum() / qn[pm].sum()),
                                               'Bq_vs_q_panel_flux': float(Bqn[pm].sum() / qn[pm].sum()), 'encq_vs_q_rel_l1': float(np.abs(encq.cpu().numpy() - qn).sum() / qn.sum()),
                                               'encq_vs_q_panel_rel_l1': float(np.abs(encq.cpu().numpy()[pm] - qn[pm]).sum() / qn[pm].sum()),
                                               'shq_vs_s16_rel_l1': float(np.abs(s_hq.cpu().numpy() - s.cpu().numpy()).sum() / s.cpu().numpy().sum()),
                                               'single_vs_L2_rel_l1': float(np.abs(out[f'S_{n}'] - L2T).sum() / L2T.sum()), 'global_vs_L3_rel_l1': float(np.abs(out[f'G_{n}'] - L3T).sum() / L3T.sum())}
    np.savez_compressed(os.path.join(R, 'results', 'tables', f'CF_{tag}.npz'), **out)
    jd(meta, os.path.join(R, 'results', 'tables', f'CF_{tag}.json'))
    print('cftables', tag, 'identity', max(meta['cf_identity'].values()), json.dumps({n: round(meta['reading'][n].get('C', meta['reading'][n].get('stationary_all'))['ORIG']['signed'], 4) for n in MODELS}), flush=True)


# ------------------------------------------------------------------------------------------------ learned local matrices across configurations
def local(R, R86, R92A):
    import torch
    fx92 = inject(R)
    import ecro86 as EC
    import model86 as M
    import diag92 as DG
    PT, FX, S = fx92.PT, fx92.FX, fx92.S
    _, Mq = DG.local_matrices()
    zb = DG.flux_w()
    W2 = (zb[:, None] * zb[None, :])[None, :, :, None]
    cfgs = ['E2_T0', 'E2_T1', 'E1_T1', 'E4_T0', 'E4_T1']
    Ms = {}
    for tag in cfgs:
        geom, ls, e, cfg = fx92.cfg_ls(tag)
        ctx = PT.Ctx(geom); st = EC.build_state(ctx)
        gs = M.GeoState(ctx.G, st.hit, st.dbin, EC.dev)
        for n in MODELS:
            mdl = load_model(R86, R92A, n)
            with torch.no_grad():
                f = mdl.factors(gs)
                Ms[(tag, n)] = (torch.einsum('ndcr,nmcr->nmdc', f['a'], f['c']).double().cpu().numpy() * W2)    # flux matrices [N, bi, bo, 3]
        PM = ctx.G.PMAT
    Xq = Mq * W2[0]
    res = {'metrics': {}, 'geometry_variation': {}, 'seed_variation': {}, 'config_variation': {}}
    mats = {'floor': 0, 'panel': 1, 'sphere': 2, 'walls': 3}
    rel = lambda a, b, sel: float(np.mean([np.linalg.norm(a[i] - b[i]) / max(np.linalg.norm(b[i]), 1e-30) for i in sel]))
    for tag in cfgs:
        for n in MODELS:
            X = Ms[(tag, n)]
            o = {}
            for nm, m in mats.items():
                sel = np.nonzero(PM == m)[0]
                rows = [DG.mat_metrics(X[i, :, :, c] / W2[0, :, :, 0], Mq[m, :, :, c], zb) for i in sel for c in range(3)]
                o[nm] = {k: float(np.mean([r[k] for r in rows])) for k in ('fro', 'peak', 'mass', 'l1')}
            res['metrics'][f'{n}:{tag}'] = o
    for n in MODELS:
        res['geometry_variation'][n] = {f'{a}->{b}': {nm: rel(Ms[(b, n)], Ms[(a, n)], np.nonzero(PM == m)[0]) for nm, m in mats.items()}
                                       for a, b in (('E2_T0', 'E2_T1'), ('E4_T0', 'E4_T1'), ('E1_T1', 'E2_T1'))}
    for arm in ('ecro', 'sum', 'own'):
        res['seed_variation'][arm] = {tag: {nm: rel(Ms[(tag, f'{arm}_s8602')], Ms[(tag, f'{arm}_s8601')], np.nonzero(PM == m)[0]) for nm, m in mats.items()} for tag in ('E2_T1', 'E2_T0')}
    jd(res, os.path.join(R, 'results', 'local92b.json'))
    for n in MODELS:
        print('local', n, json.dumps({k: {nm: round(v, 3) for nm, v in x.items()} for k, x in res['geometry_variation'][n].items()}), 'panel peak E2_T1', round(res['metrics'][f'{n}:E2_T1']['panel']['peak'], 3), flush=True)


# ------------------------------------------------------------------------------------------------ E5 candidates
def e5gen(R, R86):
    import fx92
    fx92.inject(R)
    import masks86
    S, FX = fx92.S, fx92.FX
    E = P['e5']
    tr = {'A': [], 'B': []}
    for sp in ('train', 'val', 'trainhq'):
        for fn in sorted(glob.glob(os.path.join(R86, 'data', sp, 'g*.npz'))):
            g = json.loads(str(np.load(fn)['geom']))
            for b in 'AB':
                tr[b].append(g[b]['c'])
    tr = {b: np.array(v) for b, v in tr.items()}
    disjoint, admissible = [], []
    nmax = 400 if DEV else 20000
    out_m = os.path.join(R, 'results', 'masks'); os.makedirs(out_m, exist_ok=True)
    t0 = time.time()
    for k in range(nmax):
        rng = np.random.default_rng([92550, k])
        g0 = S.sample_geom(rng)
        m = 'A' if rng.random() < 0.5 else 'B'
        g1 = json.loads(json.dumps(g0)); found = False
        for _ in range(200):
            h = g0[m]['h']
            c = [rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), rng.uniform(h[2] + 0.02, 2.9 - h[2])] if m == 'A' else [rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), h[2]]
            if np.linalg.norm(np.array(c) - np.array(g0[m]['c'])) < 0.6:
                continue
            g1[m] = {'c': [float(x) for x in c], 'h': h, 'yaw': float(rng.uniform(0, math.pi / 2))}
            if S.geom_ok(g1):
                found = True
                break
        if not found:
            continue
        ls = 'L1' if k % 2 == 0 else 'L2'
        lpos = FX['light_sets'][ls][0]['pos']
        dmin = min(float(np.linalg.norm(tr[b] - np.array(g[b]['c']), axis=1).min()) for g in (g0, g1) for b in 'AB')
        if dmin < (0.0 if DEV else 0.15) or not (S.light_ok(lpos, g0) and S.light_ok(lpos, g1)):
            continue
        cand = {'k': k, 'T0': g0, 'T1': g1, 'moved': m, 'light_set': ls, 'min_box_distance_to_training_m': dmin}
        disjoint.append(cand)
        FX['E5c'] = {'T0': g0, 'T1': g1, 'moved': [m], 'light_set': ls}
        mk, aux = masks86.compute('E5c')
        cnt = {kk: int(v.sum()) for kk, v in mk.items()}
        cand['counts'] = cnt
        rule = all(cnt[r] >= 1000 for r in ('A', 'C', 'D', 'E')) and cnt['C_footprint'] >= 1000
        cand['geometry_admissible'] = bool(rule)
        print('E5 candidate', k, m, ls, round(dmin, 3), json.dumps({r: cnt[r] for r in ('A', 'C', 'D', 'E', 'C_footprint', 'C_mirror')}), rule, flush=True)
        if rule:
            admissible.append(cand)
            np.savez_compressed(os.path.join(out_m, f'E5k{k}.npz'), **mk, **aux)
            for src in ('LMIX', 'LSPOT'):
                ms, auxs = masks86.compute('E5c', ls=src)
                np.savez_compressed(os.path.join(out_m, f'E5k{k}_{src}.npz'), **ms, **auxs)
            if len(admissible) >= 3:
                break
    jd({'rule': {k: E[k] for k in ('population', 'light', 'disjointness', 'geometry_rule')}, 'generated_up_to_k': k, 'disjoint': disjoint, 'geometry_admissible': admissible,
        'seconds': time.time() - t0, 'note': 'geometry / visibility only; no radiance of any candidate exists at this point'}, os.path.join(R, 'results', 'e5_candidates.json'))
    print('E5GEN', 'disjoint', len(disjoint), 'admissible', [c['k'] for c in admissible], flush=True)


def e5admit(R, R86, k):
    import analyze86 as A
    fx92 = inject(R)
    import analyze92 as AN
    G0 = A.G0
    e = f'E5k{k}'
    M = AN.mask_of(R, R86, f'{e}_T1')
    p0, p1 = A.load(R, f'REF_P_{e}_T0'), A.load(R, f'REF_P_{e}_T1')
    r1 = A.load(R, f'REF_ER_{e}_T1'); r0 = A.load(R, f'REF_ER_{e}_T0')
    Y = A.Y
    d, da, db = A.halves(Y(p1['tot']) - Y(p0['tot']))
    dh_, dha, dhb = A.halves(Y(p1['h']) - Y(p0['h']))
    t0 = Y(p0['tot']).mean(0); t1, t1a, t1b = A.halves(p1['tot']); e1r, e1a, e1b = A.halves(r1['tot'])
    Rg = {}
    for kk in A.REG:
        if kk not in M or M[kk].sum() < 30:
            Rg[kk] = {'pixels': int(M[kk].sum()) if kk in M else 0}
            continue
        m = M[kk]; b = max(float(t0[m].mean()), 1e-30)
        Rg[kk] = {'pixels': int(m.sum()), 'abs_change_rel': float(np.abs(d[m]).mean() / b), 'abs_change_floor_rel': float(0.5 * np.abs(da[m] - db[m]).mean() / b),
                  'absH_change_rel': float(np.abs(dh_[m]).mean() / b), 'absH_change_floor_rel': float(0.5 * np.abs(dha[m] - dhb[m]).mean() / b),
                  'H_share_T1': float(p1['h'].mean(0)[m].sum() / t1[m].sum()), 'uncertainty_P_T1': 0.5 * A.rl(t1a, t1b, m), 'uncertainty_ER_T1': 0.5 * A.rl(e1a, e1b, m)}
    mC = M['C']; sm_recv = M['obj_panel'] | M['obj_sphere']
    dc = np.abs((r1['c2'] - r0['c2']).mean(0)) + np.abs((r1['ch'] - r0['ch']).mean(0))
    allc = dc.sum(-1)
    sm = float((allc[mC & sm_recv].sum() + dc[..., :2].sum(-1)[mC & ~sm_recv].sum()) / max(allc[mC].sum(), 1e-30)) if mC.sum() else 0.0
    g = lambda kk, f: Rg.get(kk, {}).get(f, 0.0)
    sig = lambda kk, f='abs_change_rel', fl_='abs_change_floor_rel': g(kk, f) >= G0['snr'] * g(kk, fl_)
    q = {'Q1_A_direct_change': {'value': g('A', 'abs_change_rel'), 'PASS': bool(g('A', 'abs_change_rel') >= G0['A_min'] and sig('A'))},
         'Q2_B_indirect_change': {'value': g('Bset', 'abs_change_rel'), 'PASS': bool(g('Bset', 'abs_change_rel') >= G0['B_min'] and sig('Bset'))},
         'Q3_C_sharp_moderate': {'value': g('C', 'abs_change_rel'), 'sm_share': sm, 'pixels': Rg['C']['pixels'],
                                 'PASS': bool(Rg['C']['pixels'] >= G0['min_pixels'] and g('C', 'abs_change_rel') >= G0['C_min'] and sig('C') and sm >= G0['C_sm_share_min'])}}
    ratio = g('D', 'absH_change_rel') / max(g('D', 'abs_change_rel'), 1e-30)
    q['Q4_D_broad_multibounce'] = {'value': g('D', 'abs_change_rel'), 'absH_over_abs': ratio, 'absH': g('D', 'absH_change_rel'), 'absH_floor': g('D', 'absH_change_floor_rel'),
                                   'PASS': bool(g('D', 'abs_change_rel') >= G0['D_min'] and sig('D') and ratio >= G0['D_H_ratio_min'] and sig('D', 'absH_change_rel', 'absH_change_floor_rel'))}
    q['Q6_H_share'] = {'value': g('stationary_all', 'H_share_T1'), 'PASS': bool(g('stationary_all', 'H_share_T1') >= G0['H_share_min'])}
    un = max(g('stationary_all', 'uncertainty_P_T1'), g('stationary_all', 'uncertainty_ER_T1'))
    q['Q7_uncertainty'] = {'value': un, 'PASS': bool(un <= G0['uncertainty_max'])}
    # sharp family: light -> panel -> receiver 2-event share of the region-C higher-order reading at T1 (two halves)
    D = np.load(os.path.join(R, 'results', 'fields', f'D_{e}_T1.npz')); Wz = np.load(os.path.join(R, 'results', 'fields', f'W_{e}_T1.npz'))
    W = Wz['C'].astype(np.float64).reshape(-1, fx92.S.NB, 3) if 'C' in Wz.files else None
    sh = {}
    if W is not None:
        for hh in 'ab':
            tot = sum(D[f'{p}_{hh}'].astype(np.float64) for p in ('e2_p', 'e2_o', 'e3_p', 'e3_o'))
            sh[hh] = float((W * D[f'e2_p_{hh}']).sum() / max((W * tot).sum(), 1e-30))
        share = 0.5 * (sh['a'] + sh['b']); noise = 0.5 * abs(sh['a'] - sh['b'])
    else:
        share, noise = 0.0, 1.0
    q['SHARP_FAMILY'] = {'share': share, 'half_split_noise': noise, 'PASS': bool(share >= 0.08 and share >= P['thresholds']['snr'] * noise)}
    adm = bool(all(v['PASS'] for v in q.values()))
    jd({'candidate': k, 'regions': Rg, 'qualification': q, 'ADMITTED': adm}, os.path.join(R, 'results', f'e5_admission_k{k}.json'))
    print('E5ADMIT', k, 'ADMITTED' if adm else 'NOT_ADMITTED', json.dumps({kk: v['PASS'] for kk, v in q.items()}), round(share, 4), round(noise, 4), flush=True)


def e5tables(R, R86, R92A, tag):
    import torch
    fx92 = inject(R)
    import ecro86 as EC
    import model86 as M
    import diag92 as DG
    PT, FX, S, E87 = fx92.PT, fx92.FX, fx92.S, fx92.E87
    geom, ls, e, cfg = fx92.cfg_ls(tag)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    st = EC.build_state(ctx)
    s = torch.as_tensor(ctx.source(lt), device=EC.dev).view(1, ctx.N, S.NB, 3)
    gs = M.GeoState(ctx.G, st.hit, st.dbin, EC.dev)
    stale = tag.endswith('_T1')
    if stale:
        st0 = EC.build_state(PT.Ctx(FX[e]['T0'])); gs0 = M.GeoState(st0.G, st0.hit, st0.dbin, EC.dev)
    out = {}
    with torch.no_grad():
        for n in MODELS:
            mdl = load_model(R86, R92A, n)
            f = mdl.factors(gs)
            Bs = mdl.apply(f, gs, s, use_global=False)[0]; Bg = mdl.apply(f, gs, s, use_single=False)[0]
            out[f'B_{n}'] = (Bs + Bg).cpu().numpy(); out[f'S_{n}'] = Bs.cpu().numpy(); out[f'G_{n}'] = Bg.cpu().numpy()
            out[f'SP_{n}'] = DG.single_panel(mdl, f, gs, s)[0].cpu().numpy()
            if stale:
                out[f'STALE_{n}'] = mdl(gs0, s)[0].cpu().numpy()
    if tag.endswith(('_T0', '_T1')):
        import quad87 as Q
        import xport87 as X
        F = np.load(os.path.join(R, 'results', 'fields', f'F_{tag}.npz'))
        out['ORACLE'] = (0.5 * (F['fa'].astype(np.float64) + F['fb'])).astype(np.float32)
        quad = Q.Quad(X.realization_us(86001), E87['M_ray'], E87['seed_ray'])
        Bx, _, _, _, _ = X.build(ctx, lt, quad, E87['P'], E87['Mq'], E87['seed_q'], st=st)
        out['X_EXP'] = Bx.astype(np.float32)
    np.savez_compressed(os.path.join(R, 'results', 'tables', f'I_{tag}.npz'), **{k: v.astype(np.float32) for k, v in out.items()})
    print('e5tables', tag, flush=True)


def ofield(R, R86, tag):
    fx92 = inject(R)
    PT, FX = fx92.PT, fx92.FX
    geom, ls, e, cfg = fx92.cfg_ls(tag)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    spp = 64 if DEV else 4096
    fa, ca = ctx.field(lt, spp // 2, 92460); fb, cb = ctx.field(lt, spp // 2, 92460 + 500)
    np.savez_compressed(os.path.join(R, 'results', 'fields', f'F_{tag}.npz'), fa=fa, fb=fb)
    print('ofield', tag, flush=True)


def call(R, R86, fn, *args):
    inject(R)
    mod, f = fn.split('.')
    m = __import__(mod)
    getattr(m, f)(R, R86, *args)


if __name__ == '__main__':
    import fx92
    fx92.dr.set_flag(fx92.dr.JitFlag.KernelHistory, True)
    cmd, R, R86 = sys.argv[1], sys.argv[2], sys.argv[3]
    a = [x for x in sys.argv[4:] if not x.startswith('--')]
    for d in ('fields', 'tables', 'renders', 'masks', 'lists'):
        os.makedirs(os.path.join(R, 'results', d), exist_ok=True)
    {'repro': repro, 'cftables': cftables, 'local': local, 'e5gen': e5gen, 'e5admit': e5admit, 'e5tables': e5tables, 'ofield': ofield, 'call': call}[cmd](R, R86, *a)
    os._exit(0)
