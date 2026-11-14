"""WL94C (protocol ecro_q94c): panel-mediated query-level supervision smoke. WL94 / WL94A / WL94B modules are imported UNCHANGED.
Candidates D (panel-dominated box queries emphasised) and E (matched non-panel box queries emphasised) re-weight the per-(lighting, patch)
query terms of the unchanged OWN objective inside the WL94A C item stream (same items, order, init, steps); item mean weight = 1.

  python q94.py manifest <R> <R86> <R92A>              query manifests, weights, accounting, feasibility gates (results/manifest/)
  python q94.py train    <R> <R86> <R92A> <seed> <D|E> c94.train with the weighted objective
  python q94.py an       <R> <seed>                    metrics own / C / D / E (results/an/q_s<seed>.json)
  python q94.py final    <R> <out_dir>                 outcome, answers, figures, manifest
"""
import json, os, sys, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_exp94b'))
import b94 as B  # noqa: E402  (imports mat94 / c94 / an94 / r94 unchanged)

MT, C94, AN, R94 = B.MT, B.C94, B.AN, B.R94
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_q94c.json')))
DEV = MT.DEV
SEEDS = [8601, 8602]
TAU, TAU_E, KAPPA = 0.5, 0.10, 8.0
VAR, TEX = AN.VARIANTS, AN.TEX
CUR = {}


def jd(o, fn):
    AN.jd(o, fn)


def box_ids(R, R86):
    import scene86 as S
    nm = C94.sel_files(R, 'train')[0]['file']
    G = S.Geom(json.loads(str(np.load(os.path.join(R86, 'data', 'train', nm))['geom'])))
    return np.concatenate([MT.rect_patches(G, r) for r in B.BOXA + B.BOXB]), np.concatenate([MT.rect_patches(G, r) for r in B.BOXA])


# ------------------------------------------------------------------------------------------------ weighted objective
def wobjective(Bs, Bg, t, t2):
    """c94.objective with a per-(lighting, patch) weight w [L, N, 1, 1] (item mean 1) on all three terms; returns the weighted total for the
    gradient and the UNWEIGHTED components for the training log (comparable with C)."""
    den = t.mean(2, keepdim=True) + 0.05 * t.mean(dim=(1, 2, 3), keepdim=True)
    e1 = ((Bs + Bg - t) / den) ** 2; e2 = ((Bs - t2) / den) ** 2; e3 = ((Bg - (t - t2)) / den) ** 2
    w = CUR['w']
    return (w * e1).mean() + (w * e2).mean() + (w * e3).mean(), e1.mean(), e2.mean(), e3.mean()


def query_err(model, it, dev):
    """per-(lighting, patch) objective error of a model on one item: mean over bins / channels of the three normalised squared terms."""
    import torch
    with torch.no_grad():
        gs = it.gs(); s = torch.as_tensor(it.it.s, device=dev)
        F = model.factors(gs)
        Bs = model.apply(F, gs, s, use_global=False); Bg = model.apply(F, gs, s, use_single=False)
        t = it.tgt(dev, 'f'); t2 = it.tgt(dev, 'f2')
        den = t.mean(2, keepdim=True) + 0.05 * t.mean(dim=(1, 2, 3), keepdim=True)
        e = (((Bs + Bg - t) / den) ** 2 + ((Bs - t2) / den) ** 2 + ((Bg - (t - t2)) / den) ** 2).mean((2, 3))
        return e.cpu().numpy().astype(np.float64), (Bs, Bg, t, t2)


# ------------------------------------------------------------------------------------------------ manifests
def weights_of(sel, Q):
    """item weight map: KAPPA on the k selected queries, the rest (Q - KAPPA k) / (Q - k): item mean exactly 1."""
    k = int(sel.sum())
    w = np.ones(sel.shape, np.float64)
    if k:
        w[sel] = KAPPA; w[~sel] = (Q - KAPPA * k) / (Q - k)
    return w


def manifest(R, R86, R92A):
    import torch
    dev = torch.device('cuda')
    files = C94.sel_files(R, 'train'); names = [r['file'] for r in files]
    asg = R94.assignments(len(files), 'C')
    bAB, bA = box_ids(R, R86)
    items = R94.mix_items(R, R86, 'train', 'C')
    assert [it.name for it in items] == names
    T, G = {}, {}
    agg = {}
    pid = 0.0
    for nm in names:
        a = np.load(os.path.join(R, 'results', 'audit', 'tgt', nm)); b = np.load(os.path.join(R, 'results', 'audit', 'ind', nm))
        pid = max(pid, float(np.abs(a['nm'].sum(0) - a['nt']).max() / a['nt'].max()))
        agg[nm] = {'pm': (a['nm'][0] + a['nm'][2]) / np.maximum(a['nt'], 1e-30), 'pmi': (b['nm'][0] + b['nm'][2]) / np.maximum(b['nt'], 1e-30),
                   'nt': a['nt'], 'pmm': a['nm'][0] + a['nm'][2], 'ex': a['ex'].sum(0)}
    T['T_PATH_ID'] = {'pass': pid <= 1e-6, 'max_rel_parts_vs_total': pid,
                      'wl94b': 'WL94B T_PATH_ID: parts sum to the stored f (4.7e-8) and e2_p + e2_o to f2 (5.8e-8)'}
    L, N = agg[names[0]]['nt'].shape
    Q = L * N
    isbox = np.zeros((L, N), bool); isbox[:, bAB] = True
    D = {nm: isbox & (agg[nm]['pm'] >= TAU) for nm in names}
    Dind = {nm: isbox & (agg[nm]['pmi'] >= TAU) for nm in names}
    nD = int(sum(v.sum() for v in D.values()))
    agree = float(sum((D[n] & Dind[n]).sum() for n in names) / max(nD, 1))
    G['G_POP'] = {'pass': nD >= 500 and sum(int(D[n].any()) for n in names) >= 40, 'queries': nD, 'items': sum(int(D[n].any()) for n in names),
                  'box_queries': int(isbox.sum() * len(names))}
    G['G_REL'] = {'pass': agree >= 0.80, 'selection_agreement_independent_realization': agree}
    # T_OBJ_ID: the weighted objective with w = 1 equals c94.objective
    out = {'tau': TAU, 'tau_E': TAU_E, 'kappa': KAPPA, 'Q': Q, 'D_counts': {n: int(D[n].sum()) for n in names}, 'seeds': {}}
    objd = 0.0
    for sd in SEEDS:
        model = MT.load_frozen(R92A, sd)
        err = {}
        for k_, it in enumerate(items):
            e, (Bs, Bg, t, t2) = query_err(model, it, dev)
            err[it.name] = e
            if k_ < 4:
                CUR['w'] = torch.ones((L, N, 1, 1), device=dev)
                a1 = C94.objective(Bs, Bg, t, t2); a2 = wobjective(Bs, Bg, t, t2)
                objd = max(objd, max(abs(float(x) - float(y)) / max(abs(float(x)), 1e-30) for x, y in zip(a1, a2)))
        # matched negative control E (pre-freeze feasibility refinement): pool = box queries of the SAME material half (any item) with a
        # positive target and pm < TAU_E, not in D; optimal one-to-one assignment (Hungarian) on standardised log err0 and log exitance
        # (SD over positive-target box queries of the seed) + 1.0 if the box (A / B) differs. Same total count as D.
        from scipy.optimize import linear_sum_assignment
        pos = {nm: isbox & (agg[nm]['ex'] > 1e-12) for nm in names}
        lE = {nm: np.log(np.maximum(err[nm], 1e-30)) for nm in names}; lX = {nm: np.log(np.maximum(agg[nm]['ex'], 1e-30)) for nm in names}
        sdE = float(np.std(np.concatenate([lE[nm][pos[nm]] for nm in names]))); sdX = float(np.std(np.concatenate([lX[nm][pos[nm]] for nm in names])))
        inA = np.zeros(N, bool); inA[bA] = True
        E = {nm: np.zeros((L, N), bool) for nm in names}; pairs = []; short = 0
        for half in (0, 1):
            hn = names[half::2]
            dq = [(nm, int(l), int(n)) for nm in hn for l, n in np.argwhere(D[nm])]
            cq = [(nm, int(l), int(n)) for nm in hn for l, n in np.argwhere(pos[nm] & (agg[nm]['pm'] < TAU_E) & ~D[nm])]
            short += max(len(dq) - len(cq), 0)
            de = np.array([lE[x][l, n] for x, l, n in dq]); dx = np.array([lX[x][l, n] for x, l, n in dq]); dA = np.array([inA[n] for x, l, n in dq])
            ce = np.array([lE[x][l, n] for x, l, n in cq]); cx = np.array([lX[x][l, n] for x, l, n in cq]); cA = np.array([inA[n] for x, l, n in cq])
            cost = np.abs(de[:, None] - ce[None]) / sdE + np.abs(dx[:, None] - cx[None]) / sdX + 1.0 * (dA[:, None] != cA[None])
            ri, ci = linear_sum_assignment(cost)
            for i_, c_ in zip(ri, ci):
                (x, l, n), (y, cl, cn) = dq[i_], cq[c_]
                E[y][cl, cn] = True
                pairs.append((de[i_], ce[c_], dx[i_], cx[c_], agg[x]['pm'][l, n], agg[y]['pm'][cl, cn], int(dA[i_]), int(cA[c_]), int(x == y)))
        pr = np.array(pairs)

        def smd(x, y):
            return float((x.mean() - y.mean()) / max(np.sqrt(0.5 * (x.var() + y.var())), 1e-12))
        bal = {'smd_log_err0': smd(pr[:, 0], pr[:, 1]), 'smd_log_exitance': smd(pr[:, 2], pr[:, 3]),
               'mean_pm_D': float(pr[:, 4].mean()), 'mean_pm_E': float(pr[:, 5].mean()),
               'frac_boxA_D': float(pr[:, 6].mean()), 'frac_boxA_E': float(pr[:, 7].mean()), 'frac_same_item': float(pr[:, 8].mean()),
               'median_abs_dlog_err0': float(np.median(np.abs(pr[:, 0] - pr[:, 1]))), 'median_abs_dlog_exitance': float(np.median(np.abs(pr[:, 2] - pr[:, 3]))),
               'short': short}
        cc = B.c_counts(sd, len(names))
        W = {c: {nm: weights_of(S_[nm], Q) for nm in names} for c, S_ in (('D', D), ('E', E))}
        W['C'] = {nm: np.ones((L, N)) for nm in names}
        acc = {}
        for c in ('C', 'D', 'E'):
            tot = sum(cnt * (W[c][nm] * agg[nm]['nt']).sum() for cnt, nm in zip(cc, names))
            acc[c] = {'box_panel_mediated_mass': sum(cnt * (W[c][nm] * agg[nm]['pmm'] * isbox).sum() for cnt, nm in zip(cc, names)) / tot,
                      'panel_mediated_mass_all': sum(cnt * (W[c][nm] * agg[nm]['pmm']).sum() for cnt, nm in zip(cc, names)) / tot,
                      'D_query_weight_share': sum(cnt * (W[c][nm] * agg[nm]['nt'] * D[nm]).sum() for cnt, nm in zip(cc, names)) / tot,
                      'E_query_weight_share': sum(cnt * (W[c][nm] * agg[nm]['nt'] * E[nm]).sum() for cnt, nm in zip(cc, names)) / tot,
                      'box_mass': sum(cnt * (W[c][nm] * agg[nm]['nt'] * isbox).sum() for cnt, nm in zip(cc, names)) / tot,
                      'initial_err_weighted': sum(cnt * (W[c][nm] * err[nm]).sum() for cnt, nm in zip(cc, names)) / sum(cnt * Q for cnt in cc)}
        ratio = {c: {k: acc[c][k] / max(acc['C'][k], 1e-30) for k in acc['C']} for c in ('D', 'E')}
        budget = max(abs(W[c][nm].mean() - 1) for c in ('D', 'E') for nm in names)
        same_ms = int(sum(D[nm].sum() for nm in names)) == int(sum(E[nm].sum() for nm in names))
        vD = float(sum(c_ * D[nm].sum() for c_, nm in zip(cc, names))); vE = float(sum(c_ * E[nm].sum() for c_, nm in zip(cc, names)))
        cnt_eq = abs(vD - vE) / max(vD, 1) <= 0.05
        out['seeds'][str(sd)] = {'balance': bal, 'accounting': acc, 'ratio_vs_C': ratio, 'budget_max_abs_mean_dev': budget,
                                 'equal_total_counts': bool(same_ms), 'emphasised_query_visits': {'D': vD, 'E': vE}, 'visits_within_5pct': bool(cnt_eq),
                                 'w_min_E': float(min(W['E'][nm].min() for nm in names)),
                                 'w_min': float(min(W['D'][nm].min() for nm in names)), 'C_counts': [int(x) for x in cc]}
        np.savez_compressed(os.path.join(R, 'results', 'manifest', f'W_s{sd}.npz'),
                            **{f'D|{nm}': W['D'][nm].astype(np.float32) for nm in names}, **{f'E|{nm}': W['E'][nm].astype(np.float32) for nm in names},
                            **{f'Dsel|{nm}': D[nm] for nm in names}, **{f'Esel|{nm}': E[nm] for nm in names})
        del model
        torch.cuda.empty_cache()
    T['T_OBJ_ID'] = {'pass': objd <= 1e-6, 'max_rel_diff_w1': objd}
    sv = list(out['seeds'].values())
    T['T_BUDGET'] = {'pass': all(s['budget_max_abs_mean_dev'] <= 1e-9 and s['equal_total_counts'] and s['visits_within_5pct'] for s in sv)}
    T['T_MIX'] = {'pass': asg.count('A') == asg.count('B') == len(names) // 2}
    G['G_EXPO_D'] = {'pass': all(s['ratio_vs_C']['D']['box_panel_mediated_mass'] >= 2.0 and s['ratio_vs_C']['D']['D_query_weight_share'] >= 4.0 for s in sv),
                     'values': [{k: s['ratio_vs_C']['D'][k] for k in ('box_panel_mediated_mass', 'D_query_weight_share', 'panel_mediated_mass_all')} for s in sv]}
    G['G_EXPO_E'] = {'pass': all(s['ratio_vs_C']['E']['box_panel_mediated_mass'] <= 1.10 for s in sv),
                     'values': [{k: s['ratio_vs_C']['E'][k] for k in ('box_panel_mediated_mass', 'E_query_weight_share', 'panel_mediated_mass_all')} for s in sv]}
    G['G_MATCH'] = {'pass': all(abs(s['balance']['smd_log_err0']) <= 0.10 and abs(s['balance']['smd_log_exitance']) <= 0.10 and s['balance']['mean_pm_E'] <= 0.05
                                and s['balance']['short'] == 0 for s in sv), 'balance': [s['balance'] for s in sv]}
    if not all(t['pass'] for t in T.values()):
        st = 'STOP_TEST'
    elif not all(g['pass'] for g in G.values()):
        st = 'STOP_INVALID_INTERVENTION'
    else:
        st = 'PASS'
    out.update({'status': st, 'tests': T, 'gates': G, 'protocol_sha256': MT.sha_file(os.path.join(HERE, 'protocol', 'ecro_q94c.json'))})
    jd(out, os.path.join(R, 'results', 'manifest', 'manifest.json'))
    print('MANIFEST', st, json.dumps({k: v['pass'] for k, v in {**T, **G}.items()}), flush=True)


# ------------------------------------------------------------------------------------------------ training
def train(R, R86, R92A, seed, cand):
    import torch
    W = np.load(os.path.join(R, 'results', 'manifest', f'W_s{int(seed)}.npz'))
    dev = torch.device('cuda')
    otgt = C94.It.tgt

    def load(R_, R86_, split, assign, limit=None):
        if split == 'train' and assign == cand:
            items = R94.mix_items(R_, R86_, 'train', 'C')
            for it in items:
                it.qw = torch.as_tensor(W[f'{cand}|{it.name}'], device=dev).view(*W[f'{cand}|{it.name}'].shape, 1, 1)

                def tg(self, dev_, key):
                    if key == 'f':
                        CUR['w'] = self.qw
                    return otgt(self, dev_, key)
                it.tgt = types.MethodType(tg, it)
            return items
        return R94._orig_load(R_, R86_, split, assign, limit)
    C94.load_items = load
    C94.objective = wobjective
    C94.train(R, R86, R92A, seed, cand)


# ------------------------------------------------------------------------------------------------ analysis
def an(R, seed):
    seed = int(seed)
    F = AN.fref(R); M = AN.masks(R)
    out = {'seed': seed, 'models': {}, 'ref_split': B.ref_split(R, M)}
    for c in ('own', 'C', 'D', 'E'):
        tg = f'{c}_s{seed}'
        tab = dict(np.load(os.path.join(R, 'results', 'tables', f'FZ_own_s{seed}.npz' if c == 'own' else f'CD_{tg}.npz')))
        m = {'table': AN.strip(AN.table_metrics(tab, F, M, ['C0', 'C2_srcS1'])), 'image': AN.image_metrics(R, M, tg, ['C0', 'C2_srcS1']),
             'pop': AN.pop_metrics(R, f'P_{tg}'), 'face': B.face_ext(tab, F, M), 'face_bx38': B.face_ext(tab, F, M, 'bx38')}
        if c != 'own':
            j = MT.jl(os.path.join(R, 'results', 'models', f'{tg}.json'))
            m['fit'] = {k: j[k] for k in ('val_orig_rel_l1_init', 'val_orig_rel_l1_final', 'L_sum_first200', 'L_sum_last200', 'nonfinite_steps',
                                          'init_sha256', 'final_sha256', 'train_s', 'gpu_mem_peak_mib', 'params')}
        out['models'][c] = m
    ms = out['models']
    eo = ms['own']['image']['C0']['none|ROI_T|E_H']
    crit = {}
    for c in ('C', 'D', 'E'):
        m = ms[c]
        f2 = m['fit']['val_orig_rel_l1_final'] <= 1.10 * m['fit']['val_orig_rel_l1_init'] and m['fit']['nonfinite_steps'] == 0
        mr_t = AN.mr_ok({'C0': m['table']['C0']})
        pl = m['pop']['B|lambert']
        mr_p = 0.85 <= pl['amplitude'] <= 1.15 and (pl['corr'] or 0) >= 0.90
        e = m['image']['C0']['none|ROI_T|E_H']
        tol = max(2 * max(e['noise_bound'], eo['noise_bound']), 0.005)
        crit[c] = {'F2': bool(f2), 'MR': bool(mr_t and mr_p), 'MR_table': bool(mr_t), 'MR_pop': bool(mr_p),
                   'SHARP_OK_vs_own': bool(e['E_H'] <= eo['E_H'] + tol), 'E_H': e['E_H'], 'signed': e['signed'], 'noise_bound': e['noise_bound'],
                   'E_H_own': eo['E_H'], 'tol_own': tol, 'SSR': AN.ssr(m['image']),
                   'textured_E_H': {v: m['image']['C0'][f'{v}|ROI_T|E_H']['E_H'] for v in TEX}}

    def cmp(x, y, key='none|ROI_T|E_H'):
        a, b = ms[x]['image']['C0'][key], ms[y]['image']['C0'][key]
        tol = max(2 * max(a['noise_bound'], b['noise_bound']), 0.005)
        return {'diff_y_minus_x': b['E_H'] - a['E_H'], 'tol': tol, 'x_better': bool(b['E_H'] - a['E_H'] > tol), 'x_worse': bool(a['E_H'] - b['E_H'] > tol)}
    pair = {'D_vs_C': cmp('D', 'C'), 'E_vs_C': cmp('E', 'C'), 'D_vs_E': cmp('D', 'E')}
    comp = {}
    for X, key in (('sender', 'sender_C0|panel_face'), ('global', 'G|panel_face')):
        v = {c: ms[c]['face'][key] for c in ms}
        nz = v['D']['noise']

        def imp(x, y):
            return bool(v[x]['rel_l1'] <= 0.8 * v[y]['rel_l1'] and v[y]['rel_l1'] - v[x]['rel_l1'] > 2 * nz)
        comp[X] = {'rel_l1': {c: v[c]['rel_l1'] for c in v}, 'signed': {c: v[c]['signed'] for c in v}, 'noise': nz,
                   'D_improved_vs_C': imp('D', 'C'), 'E_improved_vs_C': imp('E', 'C'), 'D_better_than_E': imp('D', 'E')}
    out['criteria'] = crit; out['pairs'] = pair; out['components'] = comp
    out['init_ok'] = ms['D']['fit']['init_sha256'] == ms['E']['fit']['init_sha256'] == ms['C']['fit']['init_sha256']
    jd(out, os.path.join(R, 'results', 'an', f'q_s{seed}.json'))
    print('QAN', seed, flush=True)


def outcome(rs):
    if len(rs) < 2 or not all(r['init_ok'] for r in rs) or any(r['models'][c]['fit']['nonfinite_steps'] for r in rs for c in 'DE'):
        return 'INCONCLUSIVE'
    dC = [r['pairs']['D_vs_C']['x_better'] for r in rs]
    eC = [r['pairs']['E_vs_C']['x_better'] for r in rs]
    dE = [r['pairs']['D_vs_E']['x_better'] for r in rs]
    eqDE = [not r['pairs']['D_vs_E']['x_better'] and not r['pairs']['D_vs_E']['x_worse'] for r in rs]
    keep = [r['criteria']['D']['F2'] and r['criteria']['D']['MR'] for r in rs]
    if all(dC) and all(dE) and all(keep):
        return 'PANEL_SPECIFIC_IMPROVEMENT'
    if all(dC) and all(eC) and (all(eqDE) or all(r['pairs']['D_vs_E']['x_worse'] for r in rs)):
        return 'GENERIC_SUPERVISION_EFFECT'
    if not any(dC):
        return 'NO_RELIABLE_IMPROVEMENT'
    return 'INCONCLUSIVE'


def yn(v):
    return 'YES' if all(v) else ('NO' if not any(v) else 'MIXED')


def final(R, outd):
    rs = [MT.jl(os.path.join(R, 'results', 'an', f'q_s{s}.json')) for s in SEEDS if os.path.exists(os.path.join(R, 'results', 'an', f'q_s{s}.json'))]
    mf = MT.jl(os.path.join(R, 'results', 'manifest', 'manifest.json'))
    oc = outcome(rs)
    ans = {'exposure_increased': mf['gates']['G_EXPO_D'],
           'panel_specific_vs_E': yn([r['pairs']['D_vs_E']['x_better'] for r in rs]),
           'sender_improved_vs_C': yn([r['components']['sender']['D_improved_vs_C'] for r in rs]),
           'global_improved_vs_C': yn([r['components']['global']['D_improved_vs_C'] for r in rs]),
           'sharp_improved_vs_C': yn([r['pairs']['D_vs_C']['x_better'] for r in rs]),
           'sharp_at_own_level': yn([r['criteria']['D']['SHARP_OK_vs_own'] for r in rs]),
           'F2_D': {r['seed']: r['criteria']['D']['F2'] for r in rs}, 'MR_D': {r['seed']: r['criteria']['D']['MR'] for r in rs},
           'F2_E': {r['seed']: r['criteria']['E']['F2'] for r in rs}, 'MR_E': {r['seed']: r['criteria']['E']['MR'] for r in rs}}
    res = {'outcome': oc, 'answers': ans, 'seeds': [r['seed'] for r in rs]}
    jd(res, os.path.join(R, 'results', 'an', 'final_q.json'))
    files = figures(R, outd, rs, mf)
    jd({'worklog': '94C', 'files': [{'file': os.path.basename(f), 'sha256': MT.sha_file(f)} for f in files], 'source_root': R},
       os.path.join(outd, 'manifest.json'))
    print('FINAL', oc, flush=True)


def stop_final(R, outd, status):
    jd({'outcome': 'STOP_INVALID_INTERVENTION' if status == 'STOP_INVALID_INTERVENTION' else 'INCONCLUSIVE', 'stop': status},
       os.path.join(R, 'results', 'an', 'final_q.json'))
    print('FINAL STOP', status, flush=True)


# ------------------------------------------------------------------------------------------------ figures
def figures(R, outd, rs, mf):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import viz94 as VZ
    os.makedirs(outd, exist_ok=True)
    files = []
    nm = ['own', 'C', 'D', 'E']
    fig, ax = plt.subplots(2, 3, figsize=(21, 10))
    for si, r in enumerate(rs):
        ms = r['models']; s = r['seed']; off = si * 0.4
        e = [ms[c]['image']['C0']['none|ROI_T|E_H'] for c in nm]
        ax[0, 0].bar(np.arange(4) + off, [x['E_H'] for x in e], 0.38, yerr=[2 * x['noise_bound'] for x in e], label=f's{s}')
        ax[0, 1].bar(np.arange(4) + off, [ms[c]['image']['C0']['none|ROI_T|E_H']['signed'] for c in nm], 0.38, label=f's{s}')
        ax[0, 2].bar(np.arange(4) + off, [ms[c]['image']['C2_srcS1']['none|ROI_T|E_H']['E_H'] for c in nm], 0.38, label=f's{s}')
        for k, (key, ttl) in enumerate((('sender_C0|panel_face', 'panel-sender single flight vs e2_p, panel face: rel L1 (bars 2 x noise)'),
                                        ('G|panel_face', 'global term vs e3 reference, panel face: rel L1'))):
            ax[1, k].bar(np.arange(4) + off, [ms[c]['face'][key]['rel_l1'] for c in nm], 0.38, yerr=[2 * ms[c]['face'][key]['noise'] for c in nm], label=f's{s}')
            ax[1, k].set_title(ttl, fontsize=8)
        ax[1, 2].bar(np.arange(4) + off, [ms[c]['face']['G|panel_face']['signed'] for c in nm], 0.38, label=f's{s}')
    for a_, t in ((ax[0, 0], 'untextured ROI_T E_H (C0), bars 2 x noise bound'), (ax[0, 1], 'untextured ROI_T signed error (C0)'),
                  (ax[0, 2], 'untextured ROI_T E_H with the exact panel sender (C2_srcS1)'), (ax[1, 2], 'global term signed bias, panel face')):
        a_.set_title(t, fontsize=8)
    for a_ in ax.ravel():
        a_.set_xticks(np.arange(4) + 0.2); a_.set_xticklabels(nm); a_.legend(fontsize=7); a_.axhline(0, color='k', lw=0.6)
    fig.suptitle('WL94C: own (frozen) / C (50:50 uniform weights) / D (panel-dominated box queries x8) / E (matched non-panel box queries x8); ' +
                 '; '.join(f"s{r['seed']} V_orig " + ', '.join(f"{c} {r['models'][c]['fit']['val_orig_rel_l1_final']:.4f}" for c in 'CDE') +
                           f" (F2 limit {1.1 * r['models']['D']['fit']['val_orig_rel_l1_init']:.4f}), V_ck Lambert amp " +
                           ', '.join(f"{c} {r['models'][c]['pop']['B|lambert']['amplitude']:.2f}" for c in nm) for r in rs), fontsize=9)
    fn = os.path.join(outd, 'figQ1_summary.png'); fig.savefig(fn, dpi=80, bbox_inches='tight'); plt.close(fig); files.append(fn)
    # textured / checkerboard response
    fig, ax = plt.subplots(1, 3, figsize=(21, 4.8))
    for si, r in enumerate(rs):
        ms = r['models']
        for c, mk in zip(nm, ('k*-', 'o-', 'D-', 's-')):
            ax[0].plot(VAR, [ms[c]['image']['C0'][f'{v}|ROI_T|E_H']['E_H'] for v in VAR], mk, ls='-' if si == 0 else '--', label=f"{c} s{r['seed']}")
            ax[1].plot(TEX, [ms[c]['table']['C0'][f'{v}|B|T']['amplitude'] for v in TEX], mk, ls='-' if si == 0 else '--', label=f"{c} s{r['seed']}")
            ax[2].plot(TEX, [ms[c]['image']['C0'][f'{v}|ROI_T|response']['slope'] for v in TEX], mk, ls='-' if si == 0 else '--', label=f"{c} s{r['seed']}")
    ax[0].set_title('ROI_T E_H per variant (C0)', fontsize=8); ax[1].set_title('CK2 box-A B amplitude (C0); band = MR', fontsize=8)
    ax[1].axhspan(0.85, 1.15, color='g', alpha=0.1); ax[2].set_title('ROI_T texture-response slope (image; 1 = reference)', fontsize=8); ax[2].axhline(1, color='k', lw=0.6)
    for a_ in ax:
        a_.legend(fontsize=6, ncol=2)
    fn = os.path.join(outd, 'figQ2_textured.png'); fig.savefig(fn, dpi=80, bbox_inches='tight'); plt.close(fig); files.append(fn)
    # training curves
    fig, ax = plt.subplots(1, 3, figsize=(20, 4.5))
    for r in rs:
        s = r['seed']
        for c, ls in (('C', '-'), ('D', '--'), ('E', ':')):
            j = MT.jl(os.path.join(R, 'results', 'models', f'{c}_s{s}.json'))
            st = [x['step'] for x in j['log']]
            for k, q in enumerate(('L_sum', 'L_single', 'L_global')):
                ax[k].plot(st, [x[q] for x in j['log']], ls, label=f'{c} s{s}'); ax[k].set_yscale('log')
                ax[k].set_title(f'{q} (UNWEIGHTED, training, per 100 steps)', fontsize=8)
    for a_ in ax:
        a_.legend(fontsize=7)
    fn = os.path.join(outd, 'figQ3_training.png'); fig.savefig(fn, dpi=80, bbox_inches='tight'); plt.close(fig); files.append(fn)
    # images
    M = AN.masks(R)
    refs = {v: AN.load_ref(R, v) for v in VAR}
    href = {v: refs[v]['h'].mean(0) for v in VAR}
    l0 = AN.lum(refs['none']['tot'].mean(0))
    key = 0.5 / float(np.median(l0[l0 > 0]))
    cy, cx = VZ.crop_of(M)
    cols = []
    for r in rs:
        s = r['seed']
        cols += [(f'own_s{s}', 'C0'), (f'C_s{s}', 'C0'), (f'D_s{s}', 'C0'), (f'E_s{s}', 'C0'), (f'D_s{s}', 'C2_srcS1')]
    hl = {}
    for mt, a in cols:
        for v in VAR:
            x = AN.load_hl(R, v, mt, a)
            if x is not None:
                hl[(mt, a, v)] = x.mean(0)
    E_MAX = float(np.quantile(np.concatenate([np.abs(AN.lum(x - href[v]))[cy, cx].ravel() for (mt, a, v), x in hl.items()]), 0.999))
    D_MAX = float(np.quantile(np.concatenate([np.abs(AN.lum(href[v] - href['none']))[cy, cx].ravel() for v in TEX]), 0.999))
    for v in VAR:
        cc_ = [(mt, a) for mt, a in cols if (mt, a, v) in hl]
        fig, ax = plt.subplots(3, len(cc_) + 1, figsize=(3.0 * (len(cc_) + 1), 9.5))
        ax[0, 0].imshow(VZ.tm(href[v][cy, cx], key)); ax[0, 0].set_title(f'REF h [{v}]', fontsize=8); ax[1, 0].axis('off')
        if v != 'none':
            VZ.sq(ax[2, 0], AN.lum(href[v] - href['none'])[cy, cx], D_MAX, plt, 'REF texture response')
        else:
            ax[2, 0].axis('off')
        for j, (mt, a) in enumerate(cc_, 1):
            x = hl[(mt, a, v)]
            ax[0, j].imshow(VZ.tm(x[cy, cx], key)); ax[0, j].set_title(f'{mt} {a}' + (' (exact panel sender)' if a != 'C0' else ''), fontsize=7)
            VZ.sq(ax[1, j], AN.lum(x - href[v])[cy, cx], E_MAX, plt, 'signed error hl - h')
            if v != 'none' and (mt, a, 'none') in hl:
                VZ.sq(ax[2, j], AN.lum(x - hl[(mt, a, 'none')])[cy, cx], D_MAX, plt, 'texture response')
            else:
                ax[2, j].axis('off')
        for a_ in ax[0]:
            a_.set_xticks([]); a_.set_yticks([])
        fig.suptitle(f'WL94C CK2 {v} ({VZ.VLAB[v]}): higher-order-only sharp-panel close-up; REF / own / C / D / E (C0) / D with exact panel sender, both seeds - '
                     f'shared absolute scales (error max {E_MAX:.3g}, response max {D_MAX:.3g}); one exposure', fontsize=9)
        fn = os.path.join(outd, f'figQ4_images_{v}.png'); fig.savefig(fn, dpi=70, bbox_inches='tight'); plt.close(fig); files.append(fn)
    return files


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    cmd = a[0]
    if cmd == 'manifest':
        manifest(a[1], a[2], a[3])
    elif cmd == 'train':
        train(*a[1:6])
    elif cmd == 'an':
        an(a[1], a[2])
    elif cmd == 'final':
        final(a[1], a[2])
    elif cmd == 'stop':
        stop_final(a[1], a[2], a[3])
    os._exit(0)
