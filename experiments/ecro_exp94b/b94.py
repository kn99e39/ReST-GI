"""WL94B (protocol ecro_exp94b): sharp-panel transport exposure smoke. WL94 / WL94A modules are imported UNCHANGED; candidate D uses the
WL94A C item list with a stratified per-item step allocation (panel-mediated top quartile per material half -> 50 % of the steps).

  python b94.py agen   <R> <R86> <tgt|ind> <start> <count>     split re-render of training items (path labels) -> results/audit/<which>/
  python b94.py audit  <R> <R86>                                coverage audit, gates, stratum, allocation, sampling manifests
  python b94.py repro  <R> <R92A>                                    T_REPRO (recomputed RP_* tables vs stored; metric replays)
  python b94.py train  <R> <R86> <R92A> <seed>                  c94.train with the D step stream
  python b94.py an     <R> <seed>                               metrics own / A / C / D (results/an/b_s<seed>.json)
  python b94.py final  <R> <out_dir>                            outcome, answers, figures, manifest
"""
import glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_mat94'))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_mat94r'))
import mat94 as MT  # noqa: E402
import c94 as C94  # noqa: E402
import an94 as AN  # noqa: E402
import r94 as R94  # noqa: E402

P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_exp94b.json')))
DEV = MT.DEV
SPP = {'tgt': 256, 'ind': 16 if DEV else 64}
SEED0 = {'tgt': 940000000, 'ind': 943000000}
N_AUD = 8 if DEV else 96
PARTS = ('e2_p', 'e2_o', 'e3_p', 'e3_o')
SEEDS = [8601, 8602]
SIGMA = 0.5
PANEL_RECT = 5
BOXA, BOXB = list(range(6, 12)), list(range(12, 18))
VAR, TEX = AN.VARIANTS, AN.TEX
OWN_SHA = {8601: '739e7eb5', 8602: 'cee9a09a'}


def jd(o, fn):
    AN.jd(o, fn)


# ------------------------------------------------------------------------------------------------ Stage A: split re-render
def agen(R, R86, which, start, count):
    import fx92
    fx92.inject(R)
    PT, S, A = MT.install()
    files = C94.sel_files(R, 'train'); asg = R94.assignments(len(files), 'C')
    out = os.path.join(R, 'results', 'audit', which); os.makedirs(out, exist_ok=True)
    W = AN.W
    ctx = None; t0 = time.time(); rows = []
    for k in range(int(start), min(int(start) + int(count), N_AUD)):
        rec, a = files[k], asg[k]
        fn = os.path.join(out, rec['file'])
        if os.path.exists(fn):
            continue
        z = np.load(os.path.join(R86, 'data', 'train', rec['file']))
        geom = json.loads(str(z['geom'])); lights = json.loads(str(z['lights']))
        if ctx is None:
            ctx = PT.Ctx(geom)
        else:
            ctx.set_geom(geom)
        pm = C94.pmat_of(ctx.G, a, rec['idx'])
        assert MT.pm_digest(pm) == rec['digest_' + a], 'T_MIX digest'
        MT.set_pmat(PT, ctx, pm)
        X = {p: [] for p in PARTS}
        for li, l in enumerate(lights):
            lt = PT.Lights(l)
            t = A.field_src(ctx, A.Src(lt, 'true', ()), SPP[which], SEED0[which] + 16 * rec['idx'] + li, b=128, split=True)
            for p in PARTS:
                X[p].append(t[p])
        X = {p: np.stack(v).astype(np.float32) for p, v in X.items()}            # [L, N, NB, 3]
        f = ((X['e2_p'] + X['e2_o']) + X['e3_p']) + X['e3_o']                       # = c94.gen sum(t.values()) order
        f2 = X['e2_p'] + X['e2_o']
        idt = {}
        if which == 'tgt':
            st = np.load(os.path.join(R, 'data', f'train_{a}', rec['file']))
            fs, f2s = st['f'], st['f2']
            idt = {'id_f': float(np.abs(f.astype(np.float64) - fs).sum() / fs.astype(np.float64).sum()),
                   'id_f2': float(np.abs(f2.astype(np.float64) - f2s).sum() / f2s.astype(np.float64).sum()),
                   'bitwise_f': float((f == fs).mean())}
            fden = fs.astype(np.float64)
        else:
            fden = f.astype(np.float64)
        den = fden.mean(2, keepdims=True) + 0.05 * fden.mean(axis=(1, 2, 3), keepdims=True)
        nm = np.stack([(X[p].astype(np.float64) / den).sum((2, 3)) for p in PARTS])            # [4, L, N]
        nt = (fden / den).sum((2, 3))                                                          # [L, N]
        exi = np.stack([(X[p].astype(np.float64) * W[None, None, :, None]).sum((2, 3)) for p in PARTS])
        np.savez_compressed(fn + '.tmp.npz', nm=nm, nt=nt, ex=exi, pmat=pm.astype(np.int16), L=len(lights), N=ctx.G.N, assign=a, **idt)
        os.replace(fn + '.tmp.npz', fn)
        rows.append({'file': rec['file'], 'assign': a, **idt})
    jd({'which': which, 'start': int(start), 'count': int(count), 'rows': rows, 's': time.time() - t0, 'patch_sha256': MT._INST['patch_sha']},
       os.path.join(out, f'check_{int(start):05d}.json'))
    print('AGEN_DONE', which, start, len(rows), round(time.time() - t0, 1), 's', flush=True)


# ------------------------------------------------------------------------------------------------ Stage A: audit
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def c_counts(seed, n=96, steps=None):
    steps = steps or C94.STEPS
    rng = np.random.default_rng(int(seed) + 94); order = []; c = np.zeros(n, np.int64)
    for _ in range(steps):
        if not order:
            order = rng.permutation(n).tolist()
        c[order.pop()] += 1
    return c


def allocate(pm_half, steps_half):
    """pm_half: PM of the 48 items of one half (select order). Returns counts (H = top quartile by PM)."""
    n = len(pm_half); nh = n // 4
    rank = np.argsort(-pm_half, kind='stable')
    H = np.zeros(n, bool); H[rank[:nh]] = True
    cnt = np.zeros(n, np.int64)
    for mask, budget in ((H, int(round(SIGMA * steps_half))), (~H, steps_half - int(round(SIGMA * steps_half)))):
        ids = np.where(mask)[0]
        ids = ids[np.argsort(-pm_half[ids], kind='stable')]
        q, r = divmod(budget, len(ids))
        cnt[ids] = q; cnt[ids[:r]] += 1
    return H, cnt


def d_items_expanded(names, counts):
    """the per-step copies in select order (what train() passes to c94.train)."""
    out = []
    for nm in names:
        out += [nm] * int(counts[nm])
    return out


def replay(seed, expanded):
    rng = np.random.default_rng(int(seed) + 94)
    order = rng.permutation(len(expanded)).tolist()
    return [expanded[order.pop()] for _ in range(C94.STEPS)]


def audit(R, R86):
    import scene86 as S
    files = C94.sel_files(R, 'train')[:N_AUD]; names = [r['file'] for r in files]; asg = R94.assignments(len(files), 'C')
    z0 = np.load(os.path.join(R86, 'data', 'train', names[0]))
    G0 = S.Geom(json.loads(str(z0['geom'])))
    bA = np.concatenate([MT.rect_patches(G0, r) for r in BOXA]); bAB = np.concatenate([MT.rect_patches(G0, r) for r in BOXA + BOXB])
    panel = MT.rect_patches(G0, PANEL_RECT)
    pc = np.array(S.FX['panel']['center'], np.float64)
    T = {}; rows = []; idmax = {'id_f': 0.0, 'id_f2': 0.0}; pid_ok = True; Ls, Ns = set(), set()
    sc = {k: {} for k in ('tgt', 'ind')}
    for k_, nm in enumerate(names):
        z = np.load(os.path.join(R86, 'data', 'train', nm))
        G = S.Geom(json.loads(str(z['geom'])))
        facing = np.zeros(G.N, bool)
        for r in BOXA + BOXB:
            n = np.cross(G.RA1[r], G.RA2[r]); d = pc - G.RC[r]
            if n @ d > 0 and G.RC[r][0] > pc[0]:
                facing[MT.rect_patches(G, r)] = True
        for which in ('tgt', 'ind'):
            a = np.load(os.path.join(R, 'results', 'audit', which, nm))
            nmass, nt = a['nm'].sum(1), a['nt'].sum(0)               # [4, N], [N] (lightings summed)
            tot = nt.sum(); pmed = nmass[0] + nmass[2]
            s = {'PM': float(pmed[bAB].sum() / tot), 'PMb': float(pmed[bAB].sum() / nt[bAB].sum()), 'PMb_A': float(pmed[bA].sum() / nt[bA].sum()),
                 'PM_A': float(pmed[bA].sum() / tot),
                 'share_parts_item': [float(nmass[i].sum() / tot) for i in range(4)],
                 'panel_share_e2_box': float(nmass[0][bAB].sum() / max(nmass[0][bAB].sum() + nmass[1][bAB].sum(), 1e-30)),
                 'panel_share_e3_box': float(nmass[2][bAB].sum() / max(nmass[2][bAB].sum() + nmass[3][bAB].sum(), 1e-30)),
                 'facing_share_of_box_pm': float(pmed[facing].sum() / max(pmed[bAB].sum(), 1e-30)),
                 'box_share_item': float(nt[bAB].sum() / tot), 'panel_share_item': float(nt[panel].sum() / tot),
                 'has_facing_face': bool(facing.any())}
            sc[which][nm] = s
            if which == 'tgt':
                for q in idmax:
                    idmax[q] = max(idmax[q], float(a[q]))
                pmv = a['pmat'].astype(np.int64)
                pid_ok &= bool((pmv[panel] == 1).all() and (np.where(pmv == 1)[0].size == panel.size))
                Ls.add(int(a['L'])); Ns.add(int(a['N']))
        rows.append({'file': nm, 'assign': asg[k_], 'tgt': sc['tgt'][nm], 'ind': sc['ind'][nm]})
    T['T_PATH_ID'] = {'pass': idmax['id_f'] <= 1e-6 and idmax['id_f2'] <= 1e-6, **idmax}
    T['T_PANEL_ID'] = {'pass': pid_ok}
    T['T_QUERY'] = {'pass': Ls == {4} and len(Ns) == 1, 'L': sorted(Ls), 'N': sorted(Ns),
                    'queries_per_step': 4 * min(Ns) * AN.NB * 3 if Ns else None}
    mix = {'A': asg.count('A'), 'B': asg.count('B')}
    T['T_MIX'] = {'pass': mix == {'A': len(names) // 2, 'B': len(names) // 2}, 'examples': mix}
    PMt = np.array([sc['tgt'][n]['PM'] for n in names]); PMi = np.array([sc['ind'][n]['PM'] for n in names])
    halves = {'orig': np.arange(0, len(names), 2), 'div': np.arange(1, len(names), 2)}
    counts = np.zeros(len(names), np.int64); Hm = np.zeros(len(names), bool); agree = {}
    for hn, idx in halves.items():
        H, cnt = allocate(PMt[idx], C94.STEPS // 2)
        Hi, _ = allocate(PMi[idx], C94.STEPS // 2)
        counts[idx] = cnt; Hm[idx] = H
        agree[hn] = float((H & Hi).sum() / max(H.sum(), 1))
    cmap = {n: int(c) for n, c in zip(names, counts)}
    exp_ = d_items_expanded(names, cmap)
    alloc = {'counts': cmap, 'stratum_H': [n for n, h in zip(names, Hm) if h], 'sigma': SIGMA}
    sched = {}; expo = {}
    for sd in SEEDS:
        seq = replay(sd, exp_)
        rc = {n: seq.count(n) for n in names}
        cc = c_counts(sd, len(names))
        eD = float(sum(cmap[n] * sc['tgt'][n]['PM'] for n in names) / C94.STEPS)
        eC = float((cc * PMt).sum() / cc.sum())
        expo[str(sd)] = {'D': eD, 'C': eC, 'ratio': eD / max(eC, 1e-30),
                         'D_PMb_A': float(sum(cmap[n] * sc['tgt'][n]['PMb_A'] for n in names) / C94.STEPS),
                         'C_PMb_A': float(sum(c * sc['tgt'][n]['PMb_A'] for c, n in zip(cc, names)) / cc.sum()),
                         'C_counts_orig_div': [int(cc[0::2].sum()), int(cc[1::2].sum())]}
        sched[str(sd)] = {'sequence': seq, 'replayed_counts_equal': rc == cmap}
        jd({'seed': sd, 'sequence': seq, 'counts': cmap, 'C_counts': {n: int(c) for n, c in zip(names, cc)}},
           os.path.join(R, 'results', 'audit', f'schedule_s{sd}.json'))
    ok_alloc = (int(counts.sum()) == C94.STEPS and int(counts[0::2].sum()) == C94.STEPS // 2 and int(counts[1::2].sum()) == C94.STEPS // 2
                and all(abs(counts[idx][Hm[idx]].sum() / counts[idx].sum() - SIGMA) < 1e-9 for idx in halves.values())
                and (DEV or int(counts.min()) >= 1) and all(s['replayed_counts_equal'] for s in sched.values()))
    T['T_ALLOC'] = {'pass': bool(ok_alloc), 'total': int(counts.sum()), 'orig': int(counts[0::2].sum()), 'div': int(counts[1::2].sum()),
                    'H_counts': sorted({int(c) for c in counts[Hm]}), 'generic_counts': sorted({int(c) for c in counts[~Hm]}),
                    'min_visits': int(counts.min())}
    # CK2 comparison (untextured field reference, box A mask)
    M = AN.masks(R)
    zf = np.load(os.path.join(R, 'results', 'fields', 'CKF_none.npz'))
    Xr = {p: 0.5 * (zf[f'{p}_a'].astype(np.float64) + zf[f'{p}_b'].astype(np.float64)) for p in PARTS}
    fr = sum(Xr.values())
    den = fr.mean(1, keepdims=True) + 0.05 * fr.mean()
    nmr = {p: (Xr[p] / den).sum((1, 2)) for p in PARTS}; ntr = (fr / den).sum((1, 2))
    Tm, Tp = M['T'], M['T_panel_face']
    pr = nmr['e2_p'] + nmr['e3_p']
    ck = {'PMb_A': float(pr[Tm].sum() / ntr[Tm].sum()), 'PM_A': float(pr[Tm].sum() / ntr.sum()),
          'panel_share_e2_boxA': float(nmr['e2_p'][Tm].sum() / (nmr['e2_p'][Tm].sum() + nmr['e2_o'][Tm].sum())),
          'panel_share_e3_boxA': float(nmr['e3_p'][Tm].sum() / (nmr['e3_p'][Tm].sum() + nmr['e3_o'][Tm].sum())),
          'panel_face_share_of_boxA_pm': float(pr[Tp].sum() / pr[Tm].sum()),
          'panel_face_e3p_share_of_e3': float(nmr['e3_p'][Tp].sum() / (nmr['e3_p'][Tp].sum() + nmr['e3_o'][Tp].sum())),
          'panel_face_e2p_share_of_e2': float(nmr['e2_p'][Tp].sum() / (nmr['e2_p'][Tp].sum() + nmr['e2_o'][Tp].sum()))}
    pbA = np.array([sc['tgt'][n]['PMb_A'] for n in names])
    reach = float((pbA >= ck['PMb_A']).mean())
    cmean = float(np.mean([e['C_PMb_A'] for e in expo.values()]))
    if reach < 0.10 and cmean < 0.5 * ck['PMb_A']:
        und = 'UNDERREPRESENTED'
    elif reach >= 0.25:
        und = 'REPRESENTED'
    else:
        und = 'PARTIAL'
    g = {}
    g['G_REL'] = {'pass': spearman(PMt, PMi) >= 0.90 and min(agree.values()) >= 0.80, 'spearman': spearman(PMt, PMi), 'membership_agreement': agree}
    cH, cG = float(PMt[Hm].mean()), float(PMt[~Hm].mean())
    g['G_CONTRAST'] = {'pass': cH >= 2 * cG, 'mean_PM_H': cH, 'mean_PM_generic': cG, 'ratio': cH / max(cG, 1e-30)}
    g['G_EXPOSURE'] = {'pass': all(e['ratio'] >= 1.25 for e in expo.values()), 'per_seed': expo}
    if not all(t['pass'] for t in T.values()):
        status = 'STOP_TEST'
    elif not all(x['pass'] for x in g.values()):
        status = 'STOP_INVALID_INTERVENTION'
    else:
        status = 'PASS'
    dist = {}
    for key in ('PM', 'PMb', 'PMb_A', 'panel_share_e2_box', 'panel_share_e3_box', 'facing_share_of_box_pm', 'box_share_item'):
        v = np.array([sc['tgt'][n][key] for n in names])
        dist[key] = {'mean': float(v.mean()), 'median': float(np.median(v)), 'p10': float(np.quantile(v, 0.1)), 'p90': float(np.quantile(v, 0.9)),
                     'max': float(v.max()), 'H_mean': float(v[Hm].mean()), 'generic_mean': float(v[~Hm].mean())}
    sp = np.array([sc['tgt'][n]['share_parts_item'] for n in names])
    out = {'status': status, 'tests': T, 'gates': g, 'underrepresented': und,
           'underrepresented_detail': {'items_reaching_ck2_PMb_A': reach, 'C_stream_mean_PMb_A': cmean, 'ck2': ck},
           'distribution': dist, 'item_mass_share_parts_mean': dict(zip(PARTS, sp.mean(0).tolist())),
           'items_with_panel_facing_box_face': int(sum(sc['tgt'][n]['has_facing_face'] for n in names)),
           'allocation': alloc, 'rows': rows,
           'semantics': P['stage_A_audit']['semantics_note'],
           'protocol_sha256': MT.sha_file(os.path.join(HERE, 'protocol', 'ecro_exp94b.json'))}
    jd(out, os.path.join(R, 'results', 'audit', 'audit.json'))
    jd(alloc, os.path.join(R, 'results', 'audit', 'alloc.json'))
    print('AUDIT', status, json.dumps({k: v['pass'] for k, v in {**T, **g}.items()}), flush=True)


# ------------------------------------------------------------------------------------------------ T_REPRO
def repro(R, R92A):
    out = {}
    tabs = {}
    for t, s in (('own', 8601), ('own', 8602), ('C', 8601), ('C', 8602)):
        a = np.load(os.path.join(R, 'results', 'tables', f'RP_{t}_s{s}.npz'))
        b = np.load(os.path.join(R, 'results', 'tables', (f'FZ_own_s{s}.npz' if t == 'own' else f'CD_C_s{s}.npz')))
        ks = [k for k in b.files if k in a.files]
        tabs[f'{t}_s{s}'] = {'rel_max': max(float(np.abs(a[k] - b[k]).max() / max(np.abs(b[k]).max(), 1e-30)) for k in ks), 'keys': len(ks),
                             'missing': len(b.files) - len(ks)}
    out['tables'] = tabs
    ck = {}
    for s in SEEDS:
        own_sha = MT.sha_file(os.path.join(R92A, 'results', 'models', f'own_s{s}.pt'))
        j = MT.jl(os.path.join(R, 'results', 'models', f'C_s{s}.json'))
        m = MT.load_cand(R, f'C_s{s}')
        ck[s] = {'C_state_sha256_ok': C94.sd_hash(m) == j['final_sha256'], 'own_file_sha256': own_sha, 'own_ok': own_sha.startswith(OWN_SHA[s])}
    out['checkpoints'] = ck
    F = AN.fref(R); M = AN.masks(R)
    SB = MT.jl(os.path.join(R, 'results', 'an', 'wl94_stageB.json'))
    dm = 0.0
    for s in SEEDS:
        tab = dict(np.load(os.path.join(R, 'results', 'tables', f'RP_own_s{s}.npz')))
        tm = AN.table_metrics(tab, F, M, ['C0'])
        for v in TEX:
            dm = max(dm, abs(tm['C0'][f'{v}|B|T']['amplitude'] - SB['table'][f's{s}']['C0'][f'{v}|B|T']['amplitude']))
        im = AN.image_metrics(R, M, f'own_s{s}', ['C0'])
        dm = max(dm, abs(im['C0']['none|ROI_T|E_H']['E_H'] - SB['image'][f's{s}']['C0']['none|ROI_T|E_H']['E_H']))
        rs = MT.jl(os.path.join(R, 'results', 'an', f'wl94a_r_s{s}.json'))
        tabc = dict(np.load(os.path.join(R, 'results', 'tables', f'RP_C_s{s}.npz')))
        fm = R94.face_metrics(tabc, F, M, 'C0')
        dm = max(dm, abs(fm['G|panel_face']['signed'] - rs['models']['C']['face']['G|panel_face']['signed']))
        imc = AN.image_metrics(R, M, f'C_s{s}', ['C0'])
        dm = max(dm, abs(imc['C0']['none|ROI_T|E_H']['E_H'] - rs['criteria']['C']['RET_detail']['E_H']))
    out['metric_abs_max'] = dm
    ok = (max(v['rel_max'] for v in tabs.values()) <= 1e-5 and all(v['missing'] == 0 for v in tabs.values())
          and all(c['C_state_sha256_ok'] and c['own_ok'] for c in ck.values()) and dm <= 1e-6)
    out['status'] = 'PASS' if ok else 'STOP_REPRO'
    jd(out, os.path.join(R, 'results', 'pre', 'repro.json'))
    print('REPRO', out['status'], flush=True)


# ------------------------------------------------------------------------------------------------ training
def train(R, R86, R92A, seed):
    alloc = MT.jl(os.path.join(R, 'results', 'audit', 'alloc.json'))
    counts = alloc['counts']

    def load(R_, R86_, split, assign, limit=None):
        if split == 'train' and assign == 'D':
            items = R94.mix_items(R_, R86_, 'train', 'C')
            out = []
            for it in items:
                out += [it] * int(counts.get(it.name, 0))
            assert len(out) == C94.STEPS, 'T_ALLOC length'
            return out
        return R94._orig_load(R_, R86_, split, assign, limit)
    C94.load_items = load
    C94.train(R, R86, R92A, seed, 'D')


# ------------------------------------------------------------------------------------------------ analysis
def half_noise(F, v, key, sel):
    a, b = F[v][f'{key}_a'].sum(2)[sel], F[v][f'{key}_b'].sum(2)[sel]
    return float(0.5 * np.abs(a - b).sum() / max(F[v][key].sum(2)[sel].sum(), 1e-30))


def face_ext(tab, F, M, v='none'):
    """panel sender (C0 learned, C2_srcS1 exact) and deployed global per box-A face group, with reference half noise."""
    Tm, Tp = M['T'], M['T_panel_face']
    o = {}
    for fn, fm in (('panel_face', Tp), ('other_faces', Tm & ~Tp)):
        for nm, km, kr in (('sender_C0', 'S_C0_fromS1', 'Sp'), ('G', 'G', 'G'), ('S_C0', 'S_C0', 'S'), ('B_C0', 'B_C0', 'B'),
                           ('B_C2_srcS1', 'B_C2_srcS1', 'B'), ('S_C2_srcS1', 'S_C2_srcS1', 'S')):
            if f'{v}|{km}' not in tab:
                continue
            X = tab[f'{v}|{km}'].astype(np.float64).sum(2)[fm]; Fr = F[v][kr].sum(2)[fm]
            o[f'{nm}|{fn}'] = {'rel_l1': float(np.abs(X - Fr).sum() / Fr.sum()), 'signed': float((X - Fr).sum() / Fr.sum()),
                               'noise': half_noise(F, v, kr, fm)}
    return o


def ref_split(R, M):
    zf = np.load(os.path.join(R, 'results', 'fields', 'CKF_none.npz'))
    r = {p: 0.5 * (zf[f'{p}_a'].astype(np.float64) + zf[f'{p}_b'].astype(np.float64)).sum(2) for p in PARTS}
    o = {}
    for fn, fm in (('panel_face', M['T_panel_face']), ('other_faces', M['T'] & ~M['T_panel_face'])):
        tot = sum(r[p][fm].sum() for p in PARTS)
        o[fn] = {p: float(r[p][fm].sum() / tot) for p in PARTS}
    return o


def model_list(seed):
    return ['own', 'A', 'C', 'D']


def tag_of(c, seed):
    return f'own_s{seed}' if c == 'own' else f'{c}_s{seed}'


def an(R, seed):
    seed = int(seed)
    F = AN.fref(R); M = AN.masks(R)
    out = {'seed': seed, 'models': {}, 'ref_split': ref_split(R, M)}
    for c in model_list(seed):
        tg = tag_of(c, seed)
        tab = dict(np.load(os.path.join(R, 'results', 'tables', f'FZ_own_s{seed}.npz' if c == 'own' else f'CD_{tg}.npz')))
        m = {'table': AN.strip(AN.table_metrics(tab, F, M, ['C0', 'C2_srcS1'])), 'image': AN.image_metrics(R, M, tg, ['C0', 'C2_srcS1']),
             'pop': AN.pop_metrics(R, f'P_{tg}'), 'face': face_ext(tab, F, M), 'face_bx38': face_ext(tab, F, M, 'bx38')}
        if c != 'own':
            j = MT.jl(os.path.join(R, 'results', 'models', f'{tg}.json'))
            m['fit'] = {k: j[k] for k in ('val_orig_rel_l1_init', 'val_orig_rel_l1_final', 'L_sum_first200', 'L_sum_last200', 'nonfinite_steps',
                                          'init_sha256', 'final_sha256', 'train_s', 'gpu_mem_peak_mib', 'params', 'items')}
        out['models'][c] = m
    ms = out['models']
    crit = {}
    eo = ms['own']['image']['C0']['none|ROI_T|E_H']
    for c in ('A', 'C', 'D'):
        m = ms[c]
        f2 = m['fit']['val_orig_rel_l1_final'] <= 1.10 * m['fit']['val_orig_rel_l1_init'] and m['fit']['nonfinite_steps'] == 0
        mr_t = AN.mr_ok({'C0': m['table']['C0']})
        pl = m['pop']['B|lambert']
        mr_p = 0.85 <= pl['amplitude'] <= 1.15 and (pl['corr'] or 0) >= 0.90
        e = m['image']['C0']['none|ROI_T|E_H']
        tol = max(2 * max(e['noise_bound'], eo['noise_bound']), 0.005)
        crit[c] = {'F2': bool(f2), 'MR_table': bool(mr_t), 'MR_pop': bool(mr_p), 'MR': bool(mr_t and mr_p),
                   'SHARP_OK': bool(e['E_H'] <= eo['E_H'] + tol), 'SHARP_IMPROVED_vs_own': bool(e['E_H'] <= eo['E_H'] - tol),
                   'SHARP_detail': {'E_H': e['E_H'], 'signed': e['signed'], 'noise_bound': e['noise_bound'], 'E_H_own': eo['E_H'],
                                    'signed_own': eo['signed'], 'noise_bound_own': eo['noise_bound'], 'tol': tol},
                   'SSR': AN.ssr(m['image']) if 'C2_srcS1' in m['image'] else None}
    eD, eC = ms['D']['image']['C0']['none|ROI_T|E_H'], ms['C']['image']['C0']['none|ROI_T|E_H']
    tolc = max(2 * max(eD['noise_bound'], eC['noise_bound']), 0.005)
    crit['D']['SHARP_vs_C'] = 'IMPROVED' if eD['E_H'] <= eC['E_H'] - tolc else ('WORSE' if eD['E_H'] >= eC['E_H'] + tolc else 'SAME')
    comp = {}
    for X, key in (('sender', 'sender_C0|panel_face'), ('global', 'G|panel_face')):
        d, cc, ow = ms['D']['face'][key], ms['C']['face'][key], ms['own']['face'][key]
        nz = d['noise']
        if d['rel_l1'] <= 0.8 * cc['rel_l1'] and cc['rel_l1'] - d['rel_l1'] > 2 * nz:
            st = 'IMPROVED'
        elif d['rel_l1'] <= ow['rel_l1'] + 2 * nz:
            st = 'ANCHOR'
        else:
            st = 'LIMITED'
        comp[X] = {'status': st, 'D': d['rel_l1'], 'C': cc['rel_l1'], 'own': ow['rel_l1'], 'A': ms['A']['face'][key]['rel_l1'], 'noise': nz,
                   'signed': {c: ms[c]['face'][key]['signed'] for c in ms}}
    crit['D']['components'] = comp
    crit['D']['init_matches_C'] = ms['D']['fit']['init_sha256'] == ms['C']['fit']['init_sha256']
    out['criteria'] = crit
    jd(out, os.path.join(R, 'results', 'an', f'b_s{seed}.json'))
    print('BAN', seed, flush=True)


def outcome(rs):
    cs = [r['criteria']['D'] for r in rs]
    if len(rs) < 2 or not all(c['init_matches_C'] for c in cs) or any(r['models']['D']['fit']['nonfinite_steps'] for r in rs):
        return 'INCONCLUSIVE'
    st = lambda X: [c['components'][X]['status'] for c in cs]
    if all(c['F2'] and c['MR'] and c['SHARP_OK'] for c in cs):
        return 'SHARP_FIDELITY_PRESERVED'
    if all(s == 'IMPROVED' for s in st('sender')) and all(s == 'LIMITED' for s in st('global')):
        return 'SENDER_IMPROVED_GLOBAL_LIMITED'
    if all(s == 'IMPROVED' for s in st('global')) and all(s == 'LIMITED' for s in st('sender')):
        return 'GLOBAL_IMPROVED_SENDER_LIMITED'
    if all((c['F2'] and c['MR']) and not c['SHARP_OK'] for c in cs) or all(c['SHARP_OK'] and not (c['F2'] and c['MR']) for c in cs):
        return 'MATERIAL_SHARP_TRADEOFF_REMAINS'
    return 'INCONCLUSIVE'


def yn(v):
    return 'YES' if all(v) else ('NO' if not any(v) else 'MIXED')


def final(R, outd):
    rs = [MT.jl(os.path.join(R, 'results', 'an', f'b_s{s}.json')) for s in SEEDS if os.path.exists(os.path.join(R, 'results', 'an', f'b_s{s}.json'))]
    au = MT.jl(os.path.join(R, 'results', 'audit', 'audit.json'))
    oc = outcome(rs)
    cs = [r['criteria']['D'] for r in rs]
    sh = [c['SHARP_OK'] for c in cs]
    ans = {'underrepresented': au['underrepresented'],
           'sender_improved': yn([c['components']['sender']['status'] == 'IMPROVED' for c in cs]),
           'global_improved': yn([c['components']['global']['status'] == 'IMPROVED' for c in cs]),
           'sharp': ('IMPROVED' if all(c['SHARP_IMPROVED_vs_own'] for c in cs) else 'PRESERVED') if all(sh) else ('LOST' if not any(sh) else 'MIXED'),
           'F2': {r['seed']: r['criteria']['D']['F2'] for r in rs}, 'MR': {r['seed']: r['criteria']['D']['MR'] for r in rs},
           'exposure_sufficient': 'SUPPORTED_AS_SUFFICIENT' if oc == 'SHARP_FIDELITY_PRESERVED' else 'NOT_SHOWN'}
    res = {'outcome': oc, 'answers': ans, 'seeds': [r['seed'] for r in rs], 'criteria': {r['seed']: r['criteria'] for r in rs}}
    jd(res, os.path.join(R, 'results', 'an', 'final_b.json'))
    files = figures(R, outd, rs, au)
    jd({'worklog': '94B', 'files': [{'file': os.path.basename(f), 'sha256': MT.sha_file(f)} for f in files], 'source_root': R},
       os.path.join(outd, 'manifest.json'))
    print('FINAL', oc, flush=True)


def stop_final(R, outd, status):
    res = {'outcome': 'INCONCLUSIVE', 'stop': status}
    jd(res, os.path.join(R, 'results', 'an', 'final_b.json'))
    ap = os.path.join(R, 'results', 'audit', 'audit.json')
    if os.path.exists(ap):
        files = figures(R, outd, [], MT.jl(ap))
        jd({'worklog': '94B', 'files': [{'file': os.path.basename(f), 'sha256': MT.sha_file(f)} for f in files], 'source_root': R},
           os.path.join(outd, 'manifest.json'))
    print('FINAL STOP', status, flush=True)


# ------------------------------------------------------------------------------------------------ figures
def figures(R, outd, rs, au):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import viz94 as VZ
    os.makedirs(outd, exist_ok=True)
    files = []
    # B0 audit
    rows = au['rows']; H = set(au['allocation']['stratum_H'])
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    for hn, mk in (('A', 'o'), ('B', 's')):
        rr = [r for r in rows if r['assign'] == hn]
        x = np.arange(len(rr))
        v = np.array([r['tgt']['PMb_A'] for r in rr]); o = np.argsort(-v)
        ax[0].scatter(x, v[o], marker=mk, c=['r' if rr[i]['file'] in H else 'gray' for i in o], s=14, label=f'{"original" if hn == "A" else "diversified"} half')
    ax[0].axhline(au['underrepresented_detail']['ck2']['PMb_A'], color='b', ls='--', label='CK2 box A (untextured reference)')
    ax[0].set_title('panel-mediated share of the loss-normalised box-A target per training item (red = stratum H)', fontsize=8); ax[0].legend(fontsize=7)
    ax[0].set_xlabel('items sorted per half'); ax[0].set_yscale('symlog', linthresh=1e-3)
    pm = np.array([r['tgt']['PM'] for r in rows]); pi = np.array([r['ind']['PM'] for r in rows])
    ax[1].loglog(np.maximum(pm, 1e-6), np.maximum(pi, 1e-6), '.', ms=5); ax[1].plot([1e-6, 1], [1e-6, 1], 'k:', lw=0.8)
    ax[1].set_title(f"stratum score PM: target realization vs independent 64 spp (Spearman {au['gates']['G_REL']['spearman']:.3f})", fontsize=8)
    cnt = au['allocation']['counts']; cc = MT.jl(os.path.join(R, 'results', 'audit', 'schedule_s8601.json'))['C_counts']
    names = [r['file'] for r in rows]
    o = np.argsort(-pm)
    ax[2].bar(np.arange(len(names)), [cc[names[i]] for i in o], color='gray', label='C (s8601, uniform)', alpha=0.6)
    ax[2].bar(np.arange(len(names)), [cnt[names[i]] for i in o], color='r', alpha=0.4, label='D (stratified)')
    ax[2].set_title('training visits per item, items sorted by PM (2,000 steps each)', fontsize=8); ax[2].legend(fontsize=7)
    fig.suptitle(f"WL94B Stage A coverage audit: underrepresented = {au['underrepresented']}; D / C panel-mediated exposure "
                 + ', '.join(f"s{k} {v['ratio']:.2f}x" for k, v in au['gates']['G_EXPOSURE']['per_seed'].items()), fontsize=10)
    fn = os.path.join(outd, 'figB0_audit.png'); fig.savefig(fn, dpi=80, bbox_inches='tight'); plt.close(fig); files.append(fn)
    if not rs:
        return files
    # B1 summary per seed
    nm = ['own', 'A', 'C', 'D']
    fig, ax = plt.subplots(2, 3, figsize=(21, 10))
    for si, r in enumerate(rs):
        ms = r['models']; s = r['seed']; off = si * 0.4
        e = [ms[c]['image']['C0']['none|ROI_T|E_H'] for c in nm]
        ax[0, 0].bar(np.arange(4) + off, [x['E_H'] for x in e], 0.38, yerr=[2 * x['noise_bound'] for x in e], label=f's{s}')
        ax[0, 0].hlines(r['criteria']['D']['SHARP_detail']['E_H_own'] + r['criteria']['D']['SHARP_detail']['tol'], -0.2 + off, 3.2 + off, colors=f'C{si}', linestyles='--')
        e2 = [ms[c]['image']['C2_srcS1']['none|ROI_T|E_H']['E_H'] for c in nm]
        ax[0, 1].bar(np.arange(4) + off, e2, 0.38, label=f's{s}')
        ax[0, 2].bar(np.arange(4) + off, [ms[c]['image']['C0']['none|ROI_T|E_H']['signed'] for c in nm], 0.38, label=f's{s}')
        for k, (key, ttl) in enumerate((('sender_C0|panel_face', 'panel-sender single flight vs e2_p, panel face: rel L1'),
                                        ('G|panel_face', 'global term vs e3 reference, panel face: rel L1'),
                                        ('sender_C0|other_faces', 'panel-sender vs e2_p, other box-A faces: rel L1'))):
            ax[1, k].bar(np.arange(4) + off, [ms[c]['face'][key]['rel_l1'] for c in nm], 0.38, yerr=[2 * ms[c]['face'][key]['noise'] for c in nm], label=f's{s}')
            ax[1, k].set_title(ttl, fontsize=8)
    for a_, t in ((ax[0, 0], 'untextured ROI_T E_H (C0), bars 2 x noise; dashed = frozen-own anchor + tol'),
                  (ax[0, 1], 'untextured ROI_T E_H with the exact panel sender (C2_srcS1 control)'), (ax[0, 2], 'untextured ROI_T signed error (C0)')):
        a_.set_title(t, fontsize=8)
    for a_ in ax.ravel():
        a_.set_xticks(np.arange(4) + 0.2); a_.set_xticklabels(nm); a_.legend(fontsize=7); a_.axhline(0, color='k', lw=0.6)
    fig.suptitle('WL94B: own (frozen) / A (fixed) / C (50:50 uniform) / D (50:50 + panel-mediated exposure); ' +
                 '; '.join(f"s{r['seed']} V_orig " + ', '.join(f"{c} {r['models'][c]['fit']['val_orig_rel_l1_final']:.4f}" for c in ('A', 'C', 'D')) +
                           f" (F2 limit {1.1 * r['models']['D']['fit']['val_orig_rel_l1_init']:.4f}), V_ck Lambert amp " +
                           ', '.join(f"{c} {r['models'][c]['pop']['B|lambert']['amplitude']:.2f}" for c in nm) for r in rs), fontsize=9)
    fn = os.path.join(outd, 'figB1_summary.png'); fig.savefig(fn, dpi=80, bbox_inches='tight'); plt.close(fig); files.append(fn)
    # B2 training
    fig, ax = plt.subplots(1, 3, figsize=(20, 4.5))
    for r in rs:
        s = r['seed']
        for c, ls in (('C', '-'), ('D', '--'), ('A', ':')):
            j = MT.jl(os.path.join(R, 'results', 'models', f'{c}_s{s}.json'))
            st = [x['step'] for x in j['log']]
            for k, q in enumerate(('L_sum', 'L_single', 'L_global')):
                ax[k].plot(st, [x[q] for x in j['log']], ls, label=f'{c} s{s}'); ax[k].set_yscale('log'); ax[k].set_title(f'{q} (training, per 100 steps)', fontsize=8)
    for a_ in ax:
        a_.legend(fontsize=7)
    fn = os.path.join(outd, 'figB2_training.png'); fig.savefig(fn, dpi=80, bbox_inches='tight'); plt.close(fig); files.append(fn)
    # B3 images
    M = AN.masks(R)
    refs = {v: AN.load_ref(R, v) for v in VAR}
    href = {v: refs[v]['h'].mean(0) for v in VAR}
    l0 = AN.lum(refs['none']['tot'].mean(0))
    key = 0.5 / float(np.median(l0[l0 > 0]))
    cy, cx = VZ.crop_of(M)
    cols = []
    for r in rs:
        s = r['seed']
        cols += [(f'own_s{s}', 'C0'), (f'A_s{s}', 'C0'), (f'C_s{s}', 'C0'), (f'D_s{s}', 'C0'), (f'D_s{s}', 'C2_srcS1')]
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
        fig.suptitle(f'WL94B CK2 {v} ({VZ.VLAB[v]}): higher-order-only close-up; REF / own / A / C / D (C0) / D with exact panel sender, both seeds - '
                     f'shared absolute scales (error max {E_MAX:.3g}, response max {D_MAX:.3g}); one exposure', fontsize=9)
        fn = os.path.join(outd, f'figB3_images_{v}.png'); fig.savefig(fn, dpi=70, bbox_inches='tight'); plt.close(fig); files.append(fn)
    return files


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    cmd = a[0]
    if cmd == 'agen':
        agen(*a[1:6])
    elif cmd == 'audit':
        audit(a[1], a[2])
    elif cmd == 'repro':
        repro(a[1], a[2])
    elif cmd == 'train':
        train(*a[1:5])
    elif cmd == 'an':
        an(a[1], a[2])
    elif cmd == 'final':
        final(a[1], a[2])
    elif cmd == 'stop':
        stop_final(a[1], a[2], a[3])
    os._exit(0)
