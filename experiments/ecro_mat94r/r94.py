"""WL94A (brief label WL94-R; protocol ecro_mat94r): 50/50 original + diversified fine-tuning of the existing ECRO.
WL94 modules (experiments/ecro_mat94: mat94, c94, an94) are imported UNCHANGED; candidate C only changes the training item list.

  python r94.py repro <R> <R86> <R92A>                      T_REPRO / T_MIX / T_LEAK (results/pre/repro.json)
  python r94.py train <R> <R86> <R92A> <seed> <C|Ar|A>  c94.train with the mixed (C), repeated-A (Ar) or original (A) item list
  python r94.py an    <R> <seed>                        criteria and secondary measurements (results/an/r_s<seed>.json)
  python r94.py final <R> <out_dir>                     outcome, answers, figures, manifest
"""
import glob, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_mat94'))
import mat94 as MT  # noqa: E402
import c94 as C94  # noqa: E402
import an94 as AN  # noqa: E402

P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_mat94r.json')))
_orig_load = C94.load_items
TEX, CHECK, VAR = AN.TEX, AN.CHECK, AN.VARIANTS


def assignments(n, cand):
    return ['A'] * n if cand in ('A', 'Ar') else ['A' if k % 2 == 0 else 'B' for k in range(n)]


def mix_items(R, R86, split, assign, limit=None):
    if split == 'train' and assign in ('C', 'Ar'):
        from concurrent.futures import ThreadPoolExecutor
        files = C94.sel_files(R, 'train')[:limit]
        asg = assignments(len(files), assign)
        with ThreadPoolExecutor(16) as ex:
            return list(ex.map(lambda fa: C94.It(R, R86, 'train', fa[1], fa[0]), zip(files, asg)))
    return _orig_load(R, R86, split, assign, limit)


def step_split(seed, n=96, steps=2000):
    rng = np.random.default_rng(int(seed) + 94); order = []; c = [0, 0]
    for _ in range(steps):
        if not order:
            order = rng.permutation(n).tolist()
        c[order.pop() % 2] += 1
    return {'original_steps': c[0], 'diversified_steps': c[1]}


def train(R, R86, R92A, seed, cand):
    C94.load_items = mix_items
    C94.train(R, R86, R92A, seed, cand)


# ------------------------------------------------------------------------------------------------ Stage A
def repro(R, R86, R92A):
    T = {}
    js = {c: MT.jl(os.path.join(R, 'results', 'models', f'{c}_s8601.json')) for c in 'AB'}
    sh = {}
    for c in 'AB':
        m = MT.load_cand(R, f'{c}_s8601')
        sh[c] = {'state_sha256': C94.sd_hash(m), 'wl94_final_sha256': js[c]['final_sha256'],
                 'file_sha256': MT.sha_file(os.path.join(R, 'results', 'models', f'{c}_s8601.pt'))}
    rt = {}
    for c in 'AB':
        MT.ck_tables(R, R92A, f'cand:{c}_s8601', f'RP_{c}_s8601')
        a = np.load(os.path.join(R, 'results', 'tables', f'RP_{c}_s8601.npz')); b = np.load(os.path.join(R, 'results', 'tables', f'CD_{c}_s8601.npz'))
        rt[c] = max(float(np.abs(a[k] - b[k]).max() / max(np.abs(b[k]).max(), 1e-30)) for k in b.files)
    F = AN.fref(R); M = AN.masks(R)
    S0 = MT.jl(os.path.join(R, 'results', 'an', 'wl94_stageC_s8601.json'))
    dm = {}
    for c in 'AB':
        tm = AN.table_metrics(dict(np.load(os.path.join(R, 'results', 'tables', f'RP_{c}_s8601.npz'))), F, M, ['C0', 'C2_rcv', 'C2_srcS1'])
        dm[c] = max(abs(tm[a][f'{v}|{k}|T']['amplitude'] - S0['models'][c]['table'][a][f'{v}|{k}|T']['amplitude'])
                    for a in ('C0', 'C2_rcv', 'C2_srcS1') for v in TEX for k in 'BSG')
    T['T_REPRO'] = {'pass': all(sh[c]['state_sha256'] == sh[c]['wl94_final_sha256'] for c in 'AB') and max(rt.values()) <= 1e-5 and max(dm.values()) <= 1e-6,
                    'checkpoints': sh, 'tables_rel_max': rt, 'metric_abs_max': dm}
    files = C94.sel_files(R, 'train')
    asg = assignments(len(files), 'C')
    import scene86 as S
    ok_dig = True; leak67 = 0; nonuni = 0; cnt = {'A': 0, 'B': 0}
    for rec, a in zip(files, asg):
        z = np.load(os.path.join(R, 'data', f'train_{a}', rec['file']))
        pm = z['pmat'].astype(np.int64)
        ok_dig &= str(z['digest']) == rec[f'digest_{a}'] == MT.pm_digest(pm)
        leak67 += int(np.isin(pm, [6, 7]).sum()); cnt[a] += 1
        geom = json.loads(str(np.load(os.path.join(R86, 'data', 'train', rec['file']))['geom']))
        G = S.Geom(geom)
        nonuni += sum(int(len(np.unique(pm[MT.rect_patches(G, r)])) > 1) for r in range(len(G.RMAT)))
    sp = {str(s): step_split(s) for s in (8601, 8602)}
    T['T_MIX'] = {'pass': cnt == {'A': 48, 'B': 48} and ok_dig and len({r['file'] for r in files}) == 96
                  and all(sp[k] == P['candidate_C']['step_split_precomputed'][k] for k in sp), 'examples': cnt, 'digests_ok': ok_dig, 'step_split': sp}
    T['T_LEAK'] = {'pass': leak67 == 0 and nonuni == 0, 'ids_6_7': leak67, 'nonuniform_rects': nonuni}
    status = 'PASS' if all(v['pass'] for v in T.values()) else 'STOP'
    MT.jd({'status': status, 'tests': T, 'protocol_sha256': MT.sha_file(os.path.join(HERE, 'protocol', 'ecro_mat94r.json'))},
          os.path.join(R, 'results', 'pre', 'repro.json'))
    print('REPRO', status, json.dumps({k: v['pass'] for k, v in T.items()}), flush=True)


# ------------------------------------------------------------------------------------------------ analysis
def face_metrics(tab, F, M, arm, v='none'):
    T_, Tp = M['T'], M['T_panel_face']
    o = {}
    for comp, km, kr in (('B', f'B_{arm}', 'B'), ('S', f'S_{arm}', 'S'), ('G', 'G', 'G')):
        X = tab[f'{v}|{km}'].astype(np.float64).sum(2); Fr = F[v][kr].sum(2)
        for fn, fm in (('panel_face', Tp), ('other_faces', T_ & ~Tp)):
            x, f = X[fm], Fr[fm]
            o[f'{comp}|{fn}'] = {'rel_l1': float(np.abs(x - f).sum() / f.sum()), 'signed': float((x - f).sum() / f.sum())}
    X = tab[f'{v}|S_C0_fromS1'].astype(np.float64).sum(2)[Tp]; Fr = F[v]['Sp'].sum(2)[Tp]
    o['panel_sender_S|panel_face'] = {'rel_l1': float(np.abs(X - Fr).sum() / Fr.sum()), 'signed': float((X - Fr).sum() / Fr.sum())}
    return o


def an(R, seed):
    seed = int(seed)
    F = AN.fref(R); M = AN.masks(R)
    out = {'seed': seed, 'models': {}}
    cands = [c for c in ('A', 'B', 'C', 'Ar') if os.path.exists(os.path.join(R, 'results', 'tables', f'CD_{c}_s{seed}.npz'))]
    for c in cands:
        j = MT.jl(os.path.join(R, 'results', 'models', f'{c}_s{seed}.json'))
        tab = dict(np.load(os.path.join(R, 'results', 'tables', f'CD_{c}_s{seed}.npz')))
        tm = AN.table_metrics(tab, F, M, ['C0', 'C2_rcv', 'C2_srcS1'])
        im = AN.image_metrics(R, M, f'{c}_s{seed}', ['C0', 'C2_srcS1'])
        pm = AN.pop_metrics(R, f'P_{c}_s{seed}')
        out['models'][c] = {'fit': {k: j[k] for k in ('val_orig_rel_l1_init', 'val_orig_rel_l1_final', 'L_sum_first200', 'L_sum_last200', 'nonfinite_steps',
                                                       'init_sha256', 'final_sha256', 'train_s', 'gpu_mem_peak_mib', 'params')},
                            'table': AN.strip(tm), 'image': im, 'pop': pm, 'face': face_metrics(tab, F, M, 'C0'),
                            'face_textured_bx38': face_metrics(tab, F, M, 'C0', 'bx38')}
    out['pop_init'] = AN.pop_metrics(R, f'P_own_s{seed}')
    crit = {}
    A_ = out['models']['A']
    for c in cands:
        m = out['models'][c]
        f2 = m['fit']['val_orig_rel_l1_final'] <= 1.10 * m['fit']['val_orig_rel_l1_init'] and m['fit']['nonfinite_steps'] == 0
        mr_t = AN.mr_ok({'C0': m['table']['C0']})
        pl = m['pop']['B|lambert']
        mr_p = 0.85 <= pl['amplitude'] <= 1.15 and (pl['corr'] or 0) >= 0.90
        eC, eA = m['image']['C0']['none|ROI_T|E_H'], A_['image']['C0']['none|ROI_T|E_H']
        tol = max(2 * max(eC['noise_bound'], eA['noise_bound']), 0.005)
        ret = eC['E_H'] <= eA['E_H'] + tol
        crit[c] = {'F2': bool(f2), 'MR_table': bool(mr_t), 'MR_pop': bool(mr_p), 'MR': bool(mr_t and mr_p), 'RET': bool(ret),
                   'RET_detail': {'E_H': eC['E_H'], 'E_H_A': eA['E_H'], 'tol': tol, 'signed': eC['signed']},
                   'SSR': AN.ssr(m['image']) if 'C2_srcS1' in m['image'] else None,
                   'init_matches_A': m['fit']['init_sha256'] == A_['fit']['init_sha256']}
    out['criteria'] = crit
    c = crit.get('C', {})
    out['run_next_seed'] = bool(seed == 8601 and c.get('F2') and c.get('MR'))
    MT.jd(out, os.path.join(R, 'results', 'an', f'r_s{seed}.json'))
    print('RAN', seed, 'NEXT', out['run_next_seed'], flush=True)


def outcome(rs):
    cs = [r['criteria']['C'] for r in rs]
    if not all(c['init_matches_A'] for c in cs) or any(r['models']['C']['fit']['nonfinite_steps'] for r in rs):
        return 'INCONCLUSIVE'
    if all(c['F2'] and c['MR'] and c['RET'] for c in cs):
        return 'COEXISTENCE_OBSERVED'
    if all(c['MR'] and not (c['F2'] and c['RET']) for c in cs):
        return 'MATERIAL_GAIN_WITH_BASELINE_LOSS'
    if all(c['F2'] and c['RET'] and not c['MR'] for c in cs):
        return 'BASELINE_RETAINED_MATERIAL_GAIN_LOST'
    return 'INCONCLUSIVE'


def final(R, outd):
    rs = [MT.jl(f) for f in sorted(glob.glob(os.path.join(R, 'results', 'an', 'r_s*.json')))]
    oc = outcome(rs)
    r1 = rs[0]; c1 = r1['criteria']['C']
    gA = r1['models']['A']['face']['G|panel_face']['signed']; gC = r1['models']['C']['face']['G|panel_face']['signed']
    gB = r1['models']['B']['face']['G|panel_face']['signed'] if 'B' in r1['models'] else None
    if c1['F2'] and c1['RET'] and abs(gC - gA) <= 0.05:
        fg = 'SUPPORTED'
    elif (not c1['F2'] or not c1['RET']) and gB is not None and abs(gC) >= 0.5 * abs(gB):
        fg = 'CONTRADICTED'
    else:
        fg = 'INCONCLUSIVE'
    res = {'outcome': oc, 'seeds': [r['seed'] for r in rs], 'criteria': {r['seed']: r['criteria'] for r in rs},
           'forgetting_main_reason': fg, 'panel_face_G_signed': {'A': gA, 'B': gB, 'C': gC}}
    MT.jd(res, os.path.join(R, 'results', 'an', 'final_r.json'))
    files = figures(R, outd, rs)
    MT.jd({'worklog': '94A', 'files': [{'file': os.path.basename(f), 'sha256': MT.sha_file(f)} for f in files], 'source_root': R},
          os.path.join(outd, 'manifest.json'))
    print('FINAL', oc, 'forgetting', fg, flush=True)


# ------------------------------------------------------------------------------------------------ figures
def figures(R, outd, rs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import viz94 as VZ
    os.makedirs(outd, exist_ok=True)
    files = []
    M = AN.masks(R)
    refs = {v: AN.load_ref(R, v) for v in VAR}
    href = {v: refs[v]['h'].mean(0) for v in VAR}
    key = 0.5 / float(np.median(AN.lum(refs['none']['tot'].mean(0))[AN.lum(refs['none']['tot'].mean(0)) > 0]))
    cy, cx = VZ.crop_of(M)
    cols = []
    for r in rs:
        s = r['seed']
        cols += [(f'own_s{s}', 'C0')] + [(f'{c}_s{s}', 'C0') for c in ('A', 'B', 'C') if c in r['models']]
    hl = {}
    for mt, a in cols:
        for v in VAR:
            x = AN.load_hl(R, v, mt, a)
            if x is not None:
                hl[(mt, v)] = x.mean(0)
    E_MAX = float(np.quantile(np.concatenate([np.abs(AN.lum(x - href[v]))[cy, cx].ravel() for (mt, v), x in hl.items()]), 0.999))
    D_MAX = float(np.quantile(np.concatenate([np.abs(AN.lum(href[v] - href['none']))[cy, cx].ravel() for v in TEX]), 0.999))
    for v in VAR:
        cc = [mt for mt, _ in cols if (mt, v) in hl]
        fig, ax = plt.subplots(3, len(cc) + 1, figsize=(3.3 * (len(cc) + 1), 10))
        ax[0, 0].imshow(VZ.tm(href[v][cy, cx], key)); ax[0, 0].set_title(f'REF h [{v}]', fontsize=8); ax[1, 0].axis('off')
        if v != 'none':
            VZ.sq(ax[2, 0], AN.lum(href[v] - href['none'])[cy, cx], D_MAX, plt, 'REF texture response')
        else:
            ax[2, 0].axis('off')
        for j, mt in enumerate(cc, 1):
            x = hl[(mt, v)]
            ax[0, j].imshow(VZ.tm(x[cy, cx], key)); ax[0, j].set_title(f'{mt} C0: hl', fontsize=8)
            VZ.sq(ax[1, j], AN.lum(x - href[v])[cy, cx], E_MAX, plt, 'signed error hl - h')
            if v != 'none' and (mt, 'none') in hl:
                VZ.sq(ax[2, j], AN.lum(x - hl[(mt, 'none')])[cy, cx], D_MAX, plt, 'texture response')
            else:
                ax[2, j].axis('off')
        for a_ in ax[0]:
            a_.set_xticks([]); a_.set_yticks([])
        fig.suptitle(f'WL94A CK2 {v} ({VZ.VLAB[v]}): higher-order close-up; REF / frozen own / A fixed / B diverse / C 50:50 - shared absolute scales '
                     f'(error max {E_MAX:.3g}, response max {D_MAX:.3g}); one exposure', fontsize=10)
        fn = os.path.join(outd, f'figR3_images_{v}.png'); fig.savefig(fn, dpi=75, bbox_inches='tight'); plt.close(fig); files.append(fn)
    for r in rs:
        s = r['seed']; ms = r['models']
        fig, ax = plt.subplots(2, 3, figsize=(20, 10))
        for k, comp in enumerate('BSG'):
            for c, mk in (('A', 'o-'), ('B', 's-'), ('C', 'D-'), ('Ar', 'x:')):
                if c in ms:
                    ax[0, k].plot(TEX, [ms[c]['table']['C0'][f'{v}|{comp}|T']['amplitude'] for v in TEX], mk, label=c)
            ax[0, k].axhspan(0.85, 1.15, color='g', alpha=0.1); ax[0, k].axhline(1, color='k', lw=0.8)
            ax[0, k].set_title(f's{s}: CK2 box-A {comp} amplitude (C0); band = MR [0.85, 1.15]', fontsize=8); ax[0, k].legend(fontsize=7)
        nm = [c for c in ('A', 'B', 'C', 'Ar') if c in ms]
        e0 = [ms[c]['image']['C0']['none|ROI_T|E_H'] for c in nm]
        ax[1, 0].bar(nm, [e['E_H'] for e in e0], yerr=[2 * e['noise_bound'] for e in e0])
        ax[1, 0].axhline(r['criteria']['C']['RET_detail']['E_H_A'] + r['criteria']['C']['RET_detail']['tol'], color='r', ls='--', label='RET limit')
        ax[1, 0].set_title('untextured ROI_T E_H (C0), error bar 2 x noise bound', fontsize=8); ax[1, 0].legend(fontsize=7)
        w = 0.8 / len(nm)
        for i, c in enumerate(nm):
            ax[1, 1].bar(np.arange(len(VAR)) + i * w, [ms[c]['image']['C0'][f'{v}|ROI_T|E_H']['E_H'] for v in VAR], w, label=c)
        ax[1, 1].set_xticks(np.arange(len(VAR)) + 0.4 - w / 2); ax[1, 1].set_xticklabels(VAR); ax[1, 1].legend(fontsize=7)
        ax[1, 1].set_title('ROI_T E_H per variant (C0)', fontsize=8)
        for i, comp in enumerate('BSG'):
            ax[1, 2].bar(np.arange(len(nm)) + i * 0.27, [ms[c]['face'][f'{comp}|panel_face']['signed'] for c in nm], 0.27, label=f'{comp} panel face')
        ax[1, 2].set_xticks(np.arange(len(nm)) + 0.27); ax[1, 2].set_xticklabels(nm); ax[1, 2].axhline(0, color='k', lw=0.8); ax[1, 2].legend(fontsize=7)
        ax[1, 2].set_title('untextured signed error on the panel-facing box face (C0)', fontsize=8)
        fig.suptitle(f'WL94A s{s}: V_orig rel L1 ' + ', '.join(f"{c} {ms[c]['fit']['val_orig_rel_l1_final']:.4f}" for c in nm) +
                     f" (init {ms['A']['fit']['val_orig_rel_l1_init']:.4f}, F2 limit {1.1 * ms['A']['fit']['val_orig_rel_l1_init']:.4f}); V_ck Lambert amplitude " +
                     ', '.join(f"{c} {ms[c]['pop']['B|lambert']['amplitude']:.2f}" for c in nm), fontsize=9)
        fn = os.path.join(outd, f'figR1_summary_s{s}.png'); fig.savefig(fn, dpi=80, bbox_inches='tight'); plt.close(fig); files.append(fn)
        fig, ax = plt.subplots(1, 3, figsize=(20, 4.5))
        for c in nm:
            j = MT.jl(os.path.join(R, 'results', 'models', f'{c}_s{s}.json'))
            st = [x['step'] for x in j['log']]
            for k, q in enumerate(('L_sum', 'L_single', 'L_global')):
                ax[k].plot(st, [x[q] for x in j['log']], label=c); ax[k].set_yscale('log'); ax[k].set_title(f's{s} {q} (training, per 100 / 500 steps)', fontsize=8)
        for a_ in ax:
            a_.legend(fontsize=7)
        fn = os.path.join(outd, f'figR2_training_s{s}.png'); fig.savefig(fn, dpi=80, bbox_inches='tight'); plt.close(fig); files.append(fn)
    return files


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    if a[0] == 'repro':
        repro(a[1], a[2], a[3])
    elif a[0] == 'train':
        train(*a[1:6])
    elif a[0] == 'an':
        an(a[1], a[2])
    elif a[0] == 'final':
        final(a[1], a[2])
    os._exit(0)
