"""WL96A (protocol ecro_util96a): learned higher-order transport utility ceiling on CK2 'none' (learning-free diagnostics).
Production code (pt86 / pt86a / ecro86 / model86 / quad87 / xport87 / mat94 / an94) is imported UNCHANGED; nothing is trained or tuned.

  python w96a.py stage0 <R> <R95A>              contract tests (results/s0.json)
  python w96a.py fext   <R> <i>                 extension field half i (seed 96101 + i; only on the S1_UNRESOLVED rule)
  python w96a.py tails  <R> [--ext]             explicit tails (deployed + pooled realizations), field metrics, render tables and list
  python w96a.py an     <R> [--ext]             image metrics and the Stage 1 decision (results/s1.json)
  python w96a.py s2ref  <R> <i0> <i1>           Stage 2 fit-light references (only on S1_HEADROOM)
  python w96a.py s2fit  <R>                     Stage 2 free fits, held-out / in-sample field metrics, render tables
  python w96a.py cost   <R> <R92A> [s]          Stage 3 lifecycle on CK2 (patch refinement s; run ALONE on the GPU)
  python w96a.py final  <R> <out>               verdict, figures (results/final.json)
"""
import copy, glob, hashlib, inspect, json, math, os, re, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_mat94'))
import mat94 as MT  # noqa: E402
import an94 as AN  # noqa: E402
import fx92  # noqa: E402,F401  (puts gm1_explicit87 / ecro86 / ecro86a on sys.path)

P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_util96a.json')))
DEV = MT.DEV
EXT = '--ext' in sys.argv
QUIET = DEV                                            # dev / probe runs print no metric values
S1 = np.arange(1088, 1130)
BOXA = np.arange(1130, 1184)
PFACE = np.arange(1130, 1139)
SEEDS = [8601, 8602]
NB = 64
DOM = 2 * math.pi / NB
ZB = ((np.arange(NB) // 16) + 0.5) / 4
WT = ZB * DOM                                          # corrected cos dOmega (4 rings x 16 sectors), sums to pi
R_POOL = 2 if DEV else 16
REPS = 2 if DEV else 4
REPS_EXT = 4 if DEV else 12
RSPP = 32 if DEV else 4096
FSPP = 32 if DEV else 4096
EXT_SEEDS = [96101 + i for i in range(6)]
M_IMG = 0.005
ARMS = ['F', 'Fa', 'Fb', 'Z', 'XD', 'XP', 'XOD', 'XOP', 'XSD', 'XSP', 'G8601', 'G8602']
FAMS = ('e2_p', 'e2_o', 'e3_p', 'e3_o')


def _js(o):
    return o.item() if hasattr(o, 'item') else float(o)


def jd(o, fn):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    json.dump(o, open(fn, 'w'), indent=1, default=_js)


def jl(fn):
    return json.load(open(fn))


def say(*a):
    if not QUIET:
        print(*a, flush=True)


def setup(R):
    PT, S, A, FX, ctx, lt = MT.ctx_ck2(R)
    MT.set_pmat(PT, ctx, MT.variant_pmat(ctx.G, 'none'))
    return PT, S, A, FX, ctx, lt


def regions(G):
    pm = np.asarray(G.PMAT)
    return {'boxA': BOXA, 'panel_face': PFACE, 'S1': S1, 'nonS1': np.nonzero(pm != 1)[0], 'all': np.arange(G.N)}


# ------------------------------------------------------------------------------------------------ references
def ref_fields(R, ext=False):
    """Family tables: mean, half a, half b (with the extension: means of the 4 + 4 halves of the predeclared grouping)."""
    z = np.load(os.path.join(R, 'results', 'fields', 'CKF_none.npz'))
    g = lambda k: z[k].astype(np.float64)
    out = {}
    if not ext:
        for k in FAMS:
            a, b = g(f'{k}_a'), g(f'{k}_b')
            out[k], out[k + '_a'], out[k + '_b'] = 0.5 * (a + b), a, b
        return out
    E = [np.load(os.path.join(R, 'results', 'fields', f'CKF96_ext_{i}.npz')) for i in range(6)]
    for k in FAMS:
        ga = [g(f'{k}_a')] + [E[i][k].astype(np.float64) for i in (0, 2, 4)]
        gb = [g(f'{k}_b')] + [E[i][k].astype(np.float64) for i in (1, 3, 5)]
        a, b = np.mean(ga, 0), np.mean(gb, 0)
        out[k], out[k + '_a'], out[k + '_b'] = 0.5 * (a + b), a, b
    return out


def e2(rf, s=''):
    return rf['e2_p' + s] + rf['e2_o' + s]


def e3(rf, s=''):
    return rf['e3_p' + s] + rf['e3_o' + s]


# ------------------------------------------------------------------------------------------------ explicit transport
def explicit_op(ctx, us, seed_ray):
    """XOp of one state realization + the s1 = panel mask of every state ray [N, K, 1] (torch)."""
    import torch
    import ecro86 as EC
    import fx92
    import quad87 as Q
    import xport87 as X
    X.set_realization(ctx, us)
    st = EC.build_state(ctx)
    quad = Q.Quad(us, fx92.E87['M_ray'], seed_ray)
    op = X.XOp(ctx.G, st.hit, st.dbin, quad.W)
    h = np.asarray(st.hit).reshape(ctx.N, -1)
    pm = (h >= 0) & (np.asarray(ctx.G.PMAT)[np.clip(h, 0, None)] == 1)
    return op, torch.as_tensor(pm[..., None].astype(np.float32), device=X.dev), st


def tail(op, mp, L2):
    """e3_p_exp = T_p(L2 + L3 + L4), e3_o_exp = T_o(L2 + L3 + L4) (last gather split by the material of s1); orders for the tests."""
    import torch
    import xport87 as X
    L2 = torch.as_tensor(np.asarray(L2, np.float32), device=X.dev)
    L3 = op.apply(L2); L4 = op.apply(L3); L5 = op.apply(L4)
    g = op.gather(L2 + L3 + L4)
    tp = op.scatter(g * mp); to = op.scatter(g * (1.0 - mp))
    return tp.double().cpu().numpy(), to.double().cpu().numpy(), (L3, L4, L5)


def realizations(ctx):
    import fx92
    import xport87 as X
    E87 = fx92.E87
    yield 'deployed', X.realization_us(86001), E87['seed_ray']
    for d in range(R_POOL):
        yield f'pool{d}', X.realization_us(960000 + d), E87['seed_ray'] + 1000 * (d + 1)


# ------------------------------------------------------------------------------------------------ metrics
def fmet(X, ref, ra, rb, idx):
    W = WT[None, :, None]
    ex = lambda T: (np.asarray(T)[idx] * W).sum((1, 2))
    eX, eR, ea, eb = ex(X), ex(ref), ex(ra), ex(rb)
    tw = lambda T: (np.abs(np.asarray(T)[idx]) * W).sum()
    den_t = max(tw(ref), 1e-30); den_e = max(eR.sum(), 1e-30)
    o = {'flux_ratio': float(eX.sum() / den_e), 'signed': float((eX - eR).sum() / den_e),
         'exit_rel_l1': float(np.abs(eX - eR).sum() / den_e), 'exit_n0': float(0.5 * np.abs(ea - eb).sum() / den_e),
         'table_rel_l1': float(tw(np.asarray(X) - ref) / den_t), 'table_n0': float(0.5 * tw(np.asarray(ra) - rb) / den_t),
         'ref_flux': float(eR.sum())}
    o['exit_excess'] = o['exit_rel_l1'] - o['exit_n0']; o['table_excess'] = o['table_rel_l1'] - o['table_n0']
    if len(idx) > 2 and eR.std() > 0 and eX.std() > 0:
        o['corr'] = float(np.corrcoef(eX, eR)[0, 1])
    return o


def jk_se(draws, fn):
    X = np.asarray(draws)
    D = len(X)
    if D < 3:
        return float('nan')
    tot = X.sum(0)
    loo = np.array([fn((tot - X[i]) / (D - 1)) for i in range(D)])
    return float(np.sqrt((D - 1) / D * ((loo - loo.mean()) ** 2).sum()))


# ------------------------------------------------------------------------------------------------ Stage 0
def stage0(R, R95A):
    import torch
    import ecro86 as EC
    import xport87 as X
    import pt86 as PTm
    PT, S, A, FX, ctx, lt = setup(R)
    T = {}
    rf = ref_fields(R)
    fj = jl(os.path.join(R, 'results', 'fields', 'CKF_none.json'))
    T['T_PMAT'] = {'digest': MT.pm_digest(ctx.G.PMAT), 'expected': fj['pmat_digest'], 'pass': MT.pm_digest(ctx.G.PMAT) == fj['pmat_digest']}
    T['T_KMAX'] = {'D_minus_1': PTm.D - 1, 'J': X.J, 'pass': PTm.D - 1 == 5 and X.J == 4}
    # field split (installed field_src, 64 spp)
    spp = 16 if DEV else 64
    tot = A.field_src(ctx, A.Src(lt, 'true', ()), spp, 96001, b=64, split=False).astype(np.float64)
    prt = A.field_src(ctx, A.Src(lt, 'true', ()), spp, 96001, b=64, split=True)
    sm = sum(prt[k].astype(np.float64) for k in FAMS)
    rel = float(np.abs(sm - tot).sum() / tot.sum())
    T['T_SPLIT_FIELD'] = {'rel_l1': rel, 'min_part': float(min(prt[k].min() for k in FAMS)), 'pass': rel <= 1e-6 and min(prt[k].min() for k in FAMS) >= 0}
    s2 =MT._patch_src(MT._INST['orig_field'], 'ctx', 'pt86a.field_src', probe_k1=True)
    c1 = len(re.findall(r'if k == 2:\s*\n\s*v2p = dr\.eq\(mid, U\(1\)\)', s2)) == 1
    c2 = len(re.findall(r'sel = \(v2p if q % 2 == 0 else ~v2p\) if \(\(q < 2\) == \(k == 2\)\) else mi\.Bool\(False\)', s2)) == 1
    c3 = 'for k in range(1, kmax + 1):' in s2 and "('e2_p', 'e2_o', 'e3_p', 'e3_o')" in s2
    T['T_SPLIT_TAG'] = {'v2p_at_k2': c1, 'parts_by_k': c2, 'loop_and_names': c3, 'patched_sha': hashlib.sha256(s2.encode()).hexdigest(),
                        'installed_sha': MT._INST['patch_sha']['field_src'], 'pass': bool(c1 and c2 and c3 and hashlib.sha256(s2.encode()).hexdigest() == MT._INST['patch_sha']['field_src'])}
    # reference reconstruction vs the WL95A floor
    fl = np.load(os.path.join(R95A, 'results', 'tables', 'T95A.npz'))['floor'].astype(np.float64)
    mine = e2(rf) + e3(rf)
    rr = float(np.abs(mine - fl).sum() / fl.sum())
    T['T_REF_RECON'] = {'rel_l1': rr, 'pass': rr <= 1e-6}
    # explicit: deployed realization = default state
    ctx2 = setup(R)[4]
    st0 = EC.build_state(ctx2)
    op, mp, st = explicit_op(ctx, X.realization_us(86001), __import__('fx92').E87['seed_ray'])
    same = bool(np.array_equal(np.asarray(st.hit), np.asarray(st0.hit)) and np.array_equal(np.asarray(st.dbin), np.asarray(st0.dbin)))
    T['T_EXP_DEPLOYED'] = {'pass': same}
    L2 = e2(rf)
    tp, to, (L3, L4, L5) = tail(op, mp, L2)
    full = (L3 + L4 + L5).double().cpu().numpy()
    r1 = float(np.abs(tp + to - full).sum() / full.sum())
    a_ = torch.as_tensor(rf['e2_p'].astype(np.float32), device=X.dev); b_ = torch.as_tensor(rf['e2_o'].astype(np.float32), device=X.dev)
    lin = float((op.apply(a_ + b_) - op.apply(a_) - op.apply(b_)).abs().sum() / op.apply(a_ + b_).abs().sum())
    T['T_EXP_SPLIT'] = {'split_rel': r1, 'linearity_rel': lin, 'min_tp': float(tp.min()), 'min_to': float(to.min()),
                        'pass': r1 <= 1e-5 and lin <= 1e-5 and tp.min() >= 0 and to.min() >= 0}
    # readout: masks and the WL95A floor render
    m_me = hashlib.sha256(open(os.path.join(R, 'results', 'pre', 'masks94.npz'), 'rb').read()).hexdigest()
    m_95 = hashlib.sha256(open(os.path.join(R95A, 'results', 'pre', 'masks94.npz'), 'rb').read()).hexdigest()
    cam = FX['camera']
    sp = 32 if DEV else 4096
    acc, _ = ctx2.render(lt, 'ER', cam, cam['res'], sp, 86600, 'ecro', B=PT.fl(fl.astype(np.float32)), b=32)
    hl = PT.pack(acc, cam['res'], sp)['hl'].astype(np.float64)
    old = np.load(os.path.join(R95A, 'results', 'renders', 'CK_none_floor_X_CK2_T1_r0.npz'))['hl'].astype(np.float64)
    dmax = float(np.abs(hl - old).max() / max(old.mean(), 1e-30))
    T['T_READOUT'] = {'masks_sha_equal': m_me == m_95, 'render_max_rel': dmax, 'pass': bool(m_me == m_95 and (DEV or dmax <= 1e-4))}
    ok = all(v['pass'] for v in T.values())
    T['ALL'] = bool(ok)
    jd(T, os.path.join(R, 'results', 's0.json'))
    print('STAGE0', 'PASS' if ok else 'FAIL', json.dumps({k: bool(v['pass']) for k, v in T.items() if isinstance(v, dict)}), flush=True)


# ------------------------------------------------------------------------------------------------ extension field halves
def fext(R, i):
    PT, S, A, FX, ctx, lt = setup(R)
    t0 = time.time()
    t = A.field_src(ctx, A.Src(lt, 'true', ()), FSPP, EXT_SEEDS[i], b=128, split=True)
    fn = os.path.join(R, 'results', 'fields', f'CKF96_ext_{i}.npz')
    np.savez_compressed(fn, spp=FSPP, seed=EXT_SEEDS[i], **t)
    jd({'spp': FSPP, 'seed': EXT_SEEDS[i], 'seconds': time.time() - t0, 'pmat_digest': MT.pm_digest(ctx.G.PMAT)}, fn.replace('.npz', '.json'))
    print('FEXT', i, round(time.time() - t0, 1), flush=True)


# ------------------------------------------------------------------------------------------------ Stage 1 tables
def tails(R):
    import torch
    PT, S, A, FX, ctx, lt = setup(R)
    rf = ref_fields(R, EXT)
    G = ctx.G; RG = regions(G)
    t0 = time.time(); TM = {}
    out = {'ext': EXT, 'R_pool': R_POOL}
    dep, pool = None, []
    cross = {}
    for name, us, sr in realizations(ctx):
        t1 = time.time()
        op, mp, st = explicit_op(ctx, us, sr)
        tp, to, _ = tail(op, mp, e2(rf))
        if name == 'deployed':
            dep = (tp, to)
            for h in ('a', 'b'):
                cp, co, _ = tail(op, mp, e2(rf, '_' + h))
                cross[h] = (cp, co)
            out['deployed_rays_to_panel'] = int(mp.sum().item())
            out['state_rays'] = int(np.asarray(st.hit).size)
        else:
            pool.append((tp, to))
        del op
        torch.cuda.empty_cache()
        TM[name] = time.time() - t1
    MT.set_pmat(PT, ctx, MT.variant_pmat(ctx.G, 'none'))
    ctx.u_state_np = __import__('xport87').realization_us(86001); ctx.set_tables(ctx.G)
    PP = np.asarray([p[0] for p in pool]); PO = np.asarray([p[1] for p in pool])
    pool_p, pool_o = PP.mean(0), PO.mean(0)
    # OWN learned global term (WL92A frozen, CK2 none) from the WL94 tables
    Gs = {sd: np.load(os.path.join(R, 'results', 'tables', f'FZ_own_s{sd}.npz'))['none|G'].astype(np.float64) for sd in SEEDS}
    # field metrics
    FM = {}
    fam_ref = {'e3_o': ('e3_o',), 'e3_p': ('e3_p',), 'e3': ('e3_p', 'e3_o')}
    rsum = lambda ks, s='': sum(rf[k + s] for k in ks)
    arms_f = {'XD': {'e3_p': dep[0], 'e3_o': dep[1]}, 'XP': {'e3_p': pool_p, 'e3_o': pool_o}}
    for sd in SEEDS:
        arms_f[f'G{sd}'] = {'e3': Gs[sd]}
    for arm, fams in arms_f.items():
        for fam, ks in fam_ref.items():
            if fam == 'e3':
                X_ = fams['e3'] if 'e3' in fams else fams['e3_p'] + fams['e3_o']
            elif fam in fams:
                X_ = fams[fam]
            else:
                continue
            for rn, idx in RG.items():
                m = fmet(X_, rsum(ks), rsum(ks, '_a'), rsum(ks, '_b'), idx)
                if arm == 'XP':
                    sel = {'e3_p': (PP,), 'e3_o': (PO,), 'e3': (PP, PO)}[fam]
                    draws = sum(sel)
                    m['se_table_rel_l1'] = jk_se(draws, lambda Mn: fmet(Mn, rsum(ks), rsum(ks, '_a'), rsum(ks, '_b'), idx)['table_rel_l1'])
                    m['se_exit_rel_l1'] = jk_se(draws, lambda Mn: fmet(Mn, rsum(ks), rsum(ks, '_a'), rsum(ks, '_b'), idx)['exit_rel_l1'])
                    per = [fmet(d, rsum(ks), rsum(ks, '_a'), rsum(ks, '_b'), idx)['table_rel_l1'] for d in draws]
                    m['per_realization_table_rel_l1'] = {'median': float(np.median(per)), 'p90': float(np.percentile(per, 90)), 'worst': float(np.max(per))}
                FM[f'{arm}|{fam}|{rn}'] = m
    # cross-half: explicit from e2 half a vs tail half b (and b vs a)
    for h, o in (('a', 'b'), ('b', 'a')):
        cp, co = cross[h]
        for fam, X_ in (('e3_o', co), ('e3_p', cp), ('e3', cp + co)):
            ks = fam_ref[fam]
            for rn in ('boxA', 'all'):
                idx = RG[rn]
                W = WT[None, :, None]
                ref_o = rsum(ks, '_' + o); ref_h = rsum(ks, '_' + h)
                tw = lambda T: (np.abs(np.asarray(T)[idx]) * W).sum()
                FM[f'XD_cross_{h}to{o}|{fam}|{rn}'] = {'table_rel_l1': float(tw(X_ - ref_o) / tw(ref_o)), 'half_vs_half_rel_l1': float(tw(ref_h - ref_o) / tw(ref_o))}
    # tail magnitude relative to the prefix (context)
    mag = {}
    for rn, idx in RG.items():
        ex = lambda T: float((np.asarray(T)[idx] * WT[None, :, None]).sum())
        mag[rn] = {'e2': ex(e2(rf)), 'e3_p': ex(rf['e3_p']), 'e3_o': ex(rf['e3_o'])}
    # explicit e3_o by s1 material class (descriptive): recomputed for the deployed realization
    out['field'] = FM; out['magnitude'] = mag; out['timing_s'] = TM
    # render tables
    E2 = e2(rf)
    tab = {'F': E2 + e3(rf), 'Fa': E2 + e3(rf, '_a'), 'Fb': E2 + e3(rf, '_b'), 'Z': E2,
           'XD': E2 + dep[0] + dep[1], 'XP': E2 + pool_p + pool_o,
           'XOD': E2 + rf['e3_p'] + dep[1], 'XOP': E2 + rf['e3_p'] + pool_o,
           'XSD': E2 + dep[0] + rf['e3_o'], 'XSP': E2 + pool_p + rf['e3_o']}
    for sd in SEEDS:
        tab[f'G{sd}'] = E2 + Gs[sd]
    assert sorted(tab) == sorted(ARMS)
    sfx = '_ext' if EXT else ''
    np.savez_compressed(os.path.join(R, 'results', 'tables', f'T96A{sfx}.npz'), **{k: v.astype(np.float32) for k, v in tab.items()})
    np.savez_compressed(os.path.join(R, 'results', f'tails{sfx}.npz'), dep_p=dep[0].astype(np.float32), dep_o=dep[1].astype(np.float32),
                        pool_p=PP.astype(np.float32), pool_o=PO.astype(np.float32))
    out['total_s'] = time.time() - t0
    out['gpu_mem_peak_mib_torch'] = torch.cuda.max_memory_allocated() / 2 ** 20
    jd(out, os.path.join(R, 'results', f'tails{sfx}.json'))
    lst = os.path.join(R, 'results', 'lists', f'r96a{sfx}.txt'); os.makedirs(os.path.dirname(lst), exist_ok=True)
    nrep = REPS_EXT if EXT else REPS
    with open(lst, 'w') as fh:
        for k in ARMS:
            for r in range(nrep):
                fh.write(f'none results/tables/T96A{sfx}.npz:{k} {86600 + r} {RSPP} CK_none_{k}{sfx}_X_CK2_T1_r{r}\n')
    print('TAILS DONE', round(out['total_s'], 1), 's', flush=True)


# ------------------------------------------------------------------------------------------------ Stage 1 analysis
def load_reps(R, key, n):
    fns = [os.path.join(R, 'results', 'renders', f'CK_none_{key}_X_CK2_T1_r{r}.npz') for r in range(n)]
    if not all(os.path.exists(f) for f in fns):
        return None
    return np.stack([np.load(f)['hl'].astype(np.float64) for f in fns])


def paired(xX, xF, tot, m):
    """paired change of arm X vs F (same camera random numbers per rep)."""
    d = xX - xF
    k = d.shape[0]
    den = max(tot[m].sum(), 1e-30)
    dm = d.mean(0)
    return {'dI': float(np.abs(dm[m]).sum() / den), 'signed': float(dm[m].sum() / den),
            'nb': float(0.5 * np.abs(d[:k // 2].mean(0)[m] - d[k // 2:].mean(0)[m]).sum() / den)}


def images(R, ext):
    sfx = '_ext' if ext else ''
    n = REPS_EXT if ext else REPS
    M = AN.masks(R)
    ref = AN.load_ref(R, 'none'); tot = ref['tot'].mean(0)
    X = {k: load_reps(R, k + sfx, n) for k in ARMS}
    miss = [k for k, v in X.items() if v is None]
    if miss:
        raise SystemExit(f'missing renders {miss}')
    img = {}
    for rn in ('ROI_T', 'DIFF'):
        m = M[rn]
        for k in ARMS:
            e = AN.eh(ref['h'], X[k], tot, m)
            e.update(paired(X[k], X['F'], tot, m))
            img[f'{k}|{rn}'] = e
        img[f'n_img|{rn}'] = 0.5 * paired(X['Fa'], X['Fb'], tot, m)['dI']
    return img, X, M, ref


def s1_decide(img):
    rn = 'ROI_T'
    x, f = img[f'XOD|{rn}'], img[f'F|{rn}']
    n_img = img[f'n_img|{rn}']
    H = x['dI'] - n_img
    U = H + 2 * x['nb']; Lo = H - 2 * x['nb']
    Rz = x['E_H'] - f['E_H']
    tol = max(2 * max(x['noise_bound'], f['noise_bound']), 0.005)
    if U < M_IMG:
        o = 'S1_NO_HEADROOM'
    elif Lo >= M_IMG and Rz > tol:
        o = 'S1_HEADROOM'
    else:
        o = 'S1_UNRESOLVED'
    return {'H': H, 'U': U, 'Lo': Lo, 'n_img': n_img, 'nb': x['nb'], 'dI_XOD': x['dI'], 'R_realized': Rz, 'tol_E': tol, 'M_img': M_IMG, 'outcome': o}


def an(R):
    img, X, M, ref = images(R, EXT)
    dec = s1_decide(img)
    if EXT and dec['outcome'] == 'S1_UNRESOLVED':
        dec['outcome'] = 'S1_INCONCLUSIVE'
    sfx = '_ext' if EXT else ''
    jd({'images': img, 'decision': dec, 'ext': EXT}, os.path.join(R, 'results', f's1{sfx}.json'))
    print('S1', 'DEV' if DEV else dec['outcome'], flush=True)                # dev: no outcome exposure; extension path always exercised
    if (dec['outcome'] == 'S1_UNRESOLVED' or DEV) and not EXT:
        print('AN EXTEND', flush=True)
    say(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in dec.items()}, default=_js))


# ------------------------------------------------------------------------------------------------ Stage 2 (conditional)
def fit_lights(geom, n=160, seed=96200):
    import scene86 as S
    rng = np.random.default_rng(seed)
    ck = np.array(MT.SC['lights'][0]['pos'])
    out = []
    while len(out) < n:
        pos = [rng.uniform(-1.6, 1.6), rng.uniform(-1.6, 1.0), rng.uniform(1.2, 2.8)]
        if S.light_ok(pos, geom) and np.linalg.norm(np.array(pos) - ck) >= 0.3:
            out.append([{'pos': [float(v) for v in pos], 'rgb': [30.0, 30.0, 30.0]}])
    return out


def s2ref(R, i0, i1):
    PT, S, A, FX, ctx, lt = setup(R)
    L = fit_lights(MT.SC['geom'], 8 if DEV else 160)
    spp = 32 if DEV else 512
    for l in range(i0, min(i1, len(L))):
        fn = os.path.join(R, 'results', 'fit', f'S2F_{l:03d}.npz')
        if os.path.exists(fn):
            continue
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        lt_ = PT.Lights(L[l]); o = {}
        for h, sd in (('a', 96300 + 2 * l), ('b', 96301 + 2 * l)):
            t = A.field_src(ctx, A.Src(lt_, 'true', ()), spp, sd, b=128, split=True)
            o[f'e2_{h}'] = (t['e2_p'] + t['e2_o']).astype(np.float32); o[f'e3_o_{h}'] = t['e3_o']; o[f'e3_p_{h}'] = t['e3_p']
        np.savez_compressed(fn, light=np.array(L[l][0]['pos']), **o)
    print('S2REF', i0, i1, flush=True)


def s2fit(R):
    import torch
    dev = torch.device('cuda')
    PT, S, A, FX, ctx, lt = setup(R)
    fns = sorted(glob.glob(os.path.join(R, 'results', 'fit', 'S2F_*.npz')))
    Z = [np.load(f) for f in fns]
    D = ctx.N * NB
    w = torch.as_tensor(np.tile(WT, ctx.N), dtype=torch.float32, device=dev)[:, None]          # [D, 1]
    get = lambda k: torch.as_tensor(np.stack([z[k].reshape(D, 3) for z in Z]), dtype=torch.float32, device=dev)   # [L, D, 3]
    Xa, Ya, Xb, Yb = get('e2_a'), get('e3_o_a'), get('e2_b'), get('e3_o_b')
    rf = ref_fields(R, os.path.exists(os.path.join(R, 'results', 'fields', 'CKF96_ext_5.npz')))
    xh = torch.as_tensor(e2(rf).reshape(D, 3), dtype=torch.float32, device=dev)
    res = {'n_lights': len(Z)}
    tabs = {}
    RG = regions(ctx.G)
    for rho in (24, 64):
        g = torch.Generator(device='cpu').manual_seed(96400 + rho)
        Pm = (torch.rand(3, D, rho, generator=g) * 1e-3).to(dev).requires_grad_(True)
        Qm = (torch.rand(3, D, rho, generator=g) * 1e-3).to(dev).requires_grad_(True)
        opt = torch.optim.Adam([Pm, Qm], lr=1e-2)
        den = (w[None] * Ya ** 2).sum((1, 2)).clamp(min=1e-30)

        def pred(Xs):
            c = torch.einsum('cdr,ldc->lcr', Qm, Xs)
            return torch.einsum('cdr,lcr->ldc', Pm, c)
        steps = 200 if DEV else 5000
        hist = []
        for it in range(steps):
            opt.zero_grad()
            loss = ((w[None] * (pred(Xa) - Ya) ** 2).sum((1, 2)) / den).mean()
            loss.backward(); opt.step()
            with torch.no_grad():
                Pm.clamp_(min=0); Qm.clamp_(min=0)
            if it % 500 == 0 or it == steps - 1:
                hist.append([it, float(loss)])
        with torch.no_grad():
            rl = lambda Yp, Yt: float(((w[None] * (Yp - Yt).abs()).sum()) / (w[None] * Yt.abs()).sum())
            pa, pb = pred(Xa), pred(Xb)
            ho = pred(xh[None])[0].double().cpu().numpy().reshape(ctx.N, NB, 3)
        r = {'loss_hist': hist, 'insample_a_rel_l1': rl(pa, Ya), 'insample_b_rel_l1': rl(pb, Yb), 'half_noise_rel_l1': rl(Ya, Yb) * 0.5,
             'heldout': {rn: fmet(ho, rf['e3_o'], rf['e3_o_a'], rf['e3_o_b'], idx) for rn, idx in RG.items()}}
        res[f'rho{rho}'] = r
        tabs[f'S2r{rho}'] = e2(rf) + rf['e3_p'] + ho
    jd(res, os.path.join(R, 'results', 's2fit.json'))
    np.savez_compressed(os.path.join(R, 'results', 'tables', 'T96A_S2.npz'), **{k: v.astype(np.float32) for k, v in tabs.items()})
    lst = os.path.join(R, 'results', 'lists', 'r96a_s2.txt')
    with open(lst, 'w') as fh:
        for k in tabs:
            for r in range(REPS):
                fh.write(f'none results/tables/T96A_S2.npz:{k} {86600 + r} {RSPP} CK_none_{k}_X_CK2_T1_r{r}\n')
    print('S2FIT DONE', flush=True)


# ------------------------------------------------------------------------------------------------ Stage 3 lifecycle
_KH = {'exec_ms': 0.0, 'compile_ms': 0.0, 'misses': 0}


def _install_kh():
    import pt86 as PTm
    dr = PTm.dr
    orig = dr.kernel_history

    def kh(*a, **k):
        h = orig(*a, **k)
        for e in h:
            _KH['exec_ms'] += float(e.get('execution_time', 0.0))
            bt = float(e.get('backend_time', 0.0))
            if bt > 0.0:
                _KH['compile_ms'] += bt; _KH['misses'] += 1
        return h
    dr.kernel_history = kh
    dr.set_flag(dr.JitFlag.KernelHistory, True)
    return dr


def refine(S, s):
    """econ88.refine without the WL88 material-family switch: patch refinement by factor s (stationary grids, box faces, sphere bands)."""
    if s == 1.0:
        return
    o_r, o_f, o_s = S.stationary_rects, S.box_faces, (S.SPH_NPHI, S.SPH_NZ)
    S.stationary_rects = lambda: [r[:5] + (max(1, int(round(r[5] * s))), max(1, int(round(r[6] * s))), r[7]) for r in o_r()]
    n = max(1, int(round(3 * s)))
    S.box_faces = lambda box, mat: [f[:5] + (n, n, f[7]) for f in o_f(box, mat)]
    S.SPH_NPHI = max(2, int(round(o_s[0] * s))); S.SPH_NZ = max(1, int(round(o_s[1] * s)))


def cost(R, R92A, s=1.0):
    import torch
    import ecro86 as EC
    import fx92
    import model86 as M86
    import quad87 as Q
    import xport87 as X
    import scene86 as S
    dr = _install_kh()
    refine(S, s)
    PT, S_, A, FX, ctx, lt = setup(R)
    E87 = fx92.E87
    def sync():
        dr.sync_thread(); torch.cuda.synchronize()

    def drain():
        dr.kernel_history()
        r = dict(_KH); _KH.update(exec_ms=0.0, compile_ms=0.0, misses=0)
        return r

    class Clock:
        def __init__(self):
            sync(); drain()
            self.t0 = time.perf_counter(); self.c0 = time.process_time(); self.parts = {}; self.tp = self.t0
            self.comp = 0.0; self.exec_ms = 0.0; self.miss = 0

        def _take(self):
            k = drain(); self.exec_ms += k['exec_ms']; self.miss += k['misses']; self.comp += k['compile_ms'] / 1e3
            return k['compile_ms'] / 1e3

        def part(self, name):
            sync(); t = time.perf_counter(); c = self._take()
            self.parts[name] = self.parts.get(name, 0.0) + (t - self.tp) - c; self.tp = t

        def done(self):
            sync(); t = time.perf_counter(); self._take()
            raw = t - self.t0
            return {'wall_s': raw - self.comp, 'wall_raw_s': raw, 'compile_s': self.comp, 'kernel_cache_misses': self.miss,
                    'cpu_s': time.process_time() - self.c0, 'gpu_kernel_ms': self.exec_ms, 'parts': dict(self.parts), 'parts_sum_s': sum(self.parts.values())}
    cam = FX['camera']; res = cam['res']
    spp = 8 if DEV else 64
    g0 = copy.deepcopy(MT.SC['geom']); g1 = copy.deepcopy(g0); g1['A']['c'][1] += 0.15
    quad = Q.Quad(ctx.u_state_np, E87['M_ray'], E87['seed_ray'])
    model = MT.load_frozen(R92A, 8601)
    D = ctx.N * NB
    gen = torch.Generator(device='cpu').manual_seed(96600)
    LAM = {r: (torch.rand(3, D, r, generator=gen).cuda(), torch.rand(3, D, r, generator=gen).cuda()) for r in (24, 64)}

    def lam_apply(rho, L2):
        Pm, Qm = LAM[rho]
        x = L2.reshape(D, 3)
        c = torch.einsum('cdr,dc->cr', Qm, x)
        return torch.einsum('cdr,cr->dc', Pm, c).reshape(ctx.N, NB, 3)

    class Pipe:
        def __init__(self, arm):
            self.arm = arm
            self.st = EC.build_state(ctx)
            self.op = X.XOp(ctx.G, self.st.hit, self.st.dbin, quad.W)
            self.B = None; self.nd = 0

        def _table(self, lt_, ck):
            q = Q.qtable(ctx, lt_, E87['P'], E87['Mq'], E87['seed_q']); ck.part('source_q_table')
            L = torch.as_tensor(np.asarray(q, np.float32), device=X.dev)
            L2 = self.op.apply(L); ck.part('prefix_order')
            if self.arm == 'R0':
                L3 = self.op.apply(L2); L4 = self.op.apply(L3); L5 = self.op.apply(L4)
                B = L2 + L3 + L4 + L5; ck.part('tail_orders')
            elif self.arm.startswith('LAM'):
                B = L2 + lam_apply(int(self.arm[3:]), L2); ck.part('lambda_apply')
            else:
                B = L2
            self.B = F_(B); dr.eval(self.B); ck.part('table_handover')

        def _frame(self, lt_, seed, ck):
            ctx.render(lt_, 'ER', cam, res, spp, seed, 'ecro', B=self.B, plain=True, b=min(64, spp)); ck.part('render')

        def edit(self, geom, lt_, seed):
            ck = Clock()
            ctx.set_geom(geom); ck.part('geometry_update')
            self.st, self.nd = EC.update_state(ctx, self.st, ['A']); ck.part('relation_update')
            self.op = X.XOp(ctx.G, self.st.hit, self.st.dbin, quad.W); ck.part('relation_coefficients')
            if self.arm.startswith('LAM'):
                with torch.no_grad():
                    gs = M86.GeoState(ctx.G, self.st.hit, self.st.dbin, EC.dev)
                    f = model.factors(gs)
                    del f
                ck.part('generator')
            self._table(lt_, ck); self._frame(lt_, seed, ck)
            r = ck.done(); r['relation_rays_retraced'] = int(self.nd)
            return r

        def source(self, lt_, seed):
            ck = Clock(); self._table(lt_, ck); self._frame(lt_, seed, ck)
            return ck.done()
    F_ = lambda B: PT.F(B.reshape(-1).contiguous())
    import pt86 as PTm
    srcs = []
    rng = np.random.default_rng(96500)
    while len(srcs) < (3 if DEV else 16):
        pos = [rng.uniform(-1.6, 1.6), rng.uniform(-1.6, 1.0), rng.uniform(1.2, 2.8)]
        if S.light_ok(pos, g0):
            srcs.append([{'pos': [float(v) for v in pos], 'rgb': [30.0, 30.0, 30.0]}])
    nE = 2 if DEV else 6
    gpu = os.popen('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader').read().strip()
    foreign = os.popen('nvidia-smi --query-compute-apps=pid --format=csv,noheader').read().split()
    out = {'s': s, 'N': int(ctx.N), 'gpu': gpu, 'foreign_before': [p for p in foreign if p != str(os.getpid())], 'frame_spp': spp, 'res': res, 'arms': {}}
    med = lambda rows, k: float(np.median([r[k] for r in rows]))
    for arm in ('R0', 'LAM24', 'LAM64', 'ZERO'):
        ctx.set_geom(g0)
        torch.cuda.reset_peak_memory_stats()
        p = Pipe(arm)
        for g in (g1, g0):
            p.edit(g, lt, 1)
        for ls in srcs[:2]:
            p.source(PTm.Lights(ls), 2)
        ed = [p.edit(g1 if i % 2 == 0 else g0, lt, 96810 + i) for i in range(nE)]
        so = [p.source(PTm.Lights(ls), 96830 + i) for i, ls in enumerate(srcs)]

        def summ(rows):
            keys = sorted({k for r in rows for k in r['parts']})
            return {'wall_s': med(rows, 'wall_s'), 'wall_raw_s': med(rows, 'wall_raw_s'), 'compile_s_total': float(sum(r['compile_s'] for r in rows)),
                    'events_with_compilation': int(sum(r['kernel_cache_misses'] > 0 for r in rows)), 'cpu_s': med(rows, 'cpu_s'),
                    'gpu_kernel_ms': med(rows, 'gpu_kernel_ms'), 'parts': {k: float(np.median([r['parts'].get(k, 0.0) for r in rows])) for k in keys},
                    'parts_sum_over_wall': float(np.median([r['parts_sum_s'] / r['wall_s'] for r in rows])), 'n': len(rows)}
        out['arms'][arm] = {'edit': summ(ed), 'source': summ(so), 'dirty_rays_median': float(np.median([r['relation_rays_retraced'] for r in ed])),
                            'torch_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20}
        say(arm, round(out['arms'][arm]['edit']['wall_s'] * 1e3, 2), round(out['arms'][arm]['source']['wall_s'] * 1e3, 2))
        if arm == 'LAM64':
            # component medians (20 warm repetitions)
            def tm(fn, n=20):
                fn(); ts = []
                for _ in range(n):
                    sync(); drain(); t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0 - drain()['compile_ms'] / 1e3)
                return float(np.median(ts))
            q = Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q'])
            L2 = p.op.apply(torch.as_tensor(np.asarray(q, np.float32), device=X.dev))
            comp = {'q_table_s': tm(lambda: Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q'])),
                    'one_order_s': tm(lambda: p.op.apply(L2)),
                    'tail_3_orders_s': tm(lambda: p.op.apply(p.op.apply(p.op.apply(L2)))),
                    'lambda24_apply_s': tm(lambda: lam_apply(24, L2)), 'lambda64_apply_s': tm(lambda: lam_apply(64, L2))}
            with torch.no_grad():
                gs = M86.GeoState(ctx.G, p.st.hit, p.st.dbin, EC.dev)
                comp['generator_factors_s'] = tm(lambda: model.factors(gs), 5)
                f = model.factors(gs)
                s16 = torch.zeros(1, ctx.N, NB, 3, device=EC.dev)
                comp['ecro_apply_s'] = tm(lambda: model.apply(f, gs, s16), 10)
            comp['frame_s'] = tm(lambda: ctx.render(lt, 'ER', cam, res, spp, 7, 'ecro', B=p.B, plain=True, b=min(64, spp)), 5)
            out['components'] = comp
        del p
        torch.cuda.empty_cache()
    # ray counts and memory
    out['rays'] = {'relation_rays': int(ctx.N * S.K), 'q_shadow_rays_per_light': int(ctx.N * E87['P']),
                   'render_rays_nominal_per_frame': int(res * res * spp * 2 * 2), 'dirty_rays_per_edit_median': out['arms']['R0']['dirty_rays_median']}
    me = str(os.getpid())
    out['process_gpu_mem'] = [l for l in os.popen('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader').read().splitlines() if l.split(',')[0].strip() == me]
    U = {}
    for arm in ('LAM24', 'LAM64', 'ZERO'):
        U[arm] = {k: out['arms']['R0'][k]['wall_s'] / out['arms'][arm][k]['wall_s'] for k in ('edit', 'source')}
    out['U'] = U
    out['boundary_ok'] = bool(all(0.95 <= out['arms'][a][k]['parts_sum_over_wall'] <= 1.05 for a in out['arms'] for k in ('edit', 'source')))
    jd(out, os.path.join(R, 'results', f'cost_s{s}.json'))
    print('COST DONE', s, 'boundary_ok', out['boundary_ok'], flush=True)
    if s == 1.0 and max(U['ZERO'].values()) >= 1.15 and not DEV:
        print('COST EXTEND', flush=True)


# ------------------------------------------------------------------------------------------------ verdict
def final(R, outdir):
    s0 = jl(os.path.join(R, 'results', 's0.json'))
    ext = os.path.exists(os.path.join(R, 'results', 's1_ext.json'))
    s1 = jl(os.path.join(R, 'results', 's1_ext.json' if ext else 's1.json'))
    o1 = s1['decision']['outcome']
    s2 = None
    acc_conf = False
    if o1 == 'S1_HEADROOM' and os.path.exists(os.path.join(R, 'results', 's2fit.json')):
        s2 = jl(os.path.join(R, 'results', 's2fit.json'))
        tl = jl(os.path.join(R, 'results', 'tails_ext.json' if ext else 'tails.json'))
        n = REPS_EXT if ext else REPS
        M = AN.masks(R); ref = AN.load_ref(R, 'none'); tot = ref['tot'].mean(0)
        xF = load_reps(R, 'F' + ('_ext' if ext else ''), n)
        s2['image'] = {}
        for rho in (24, 64):
            x = load_reps(R, f'S2r{rho}', REPS)
            if x is None:
                continue
            pr = paired(x, xF[:REPS], tot, M['ROI_T'])
            s2['image'][rho] = pr
            xo = tl['field']['XD|e3_o|boxA']
            ok = (pr['dI'] <= s1['images']['XOD|ROI_T']['dI'] - M_IMG and
                  s2[f'rho{rho}']['heldout']['boxA']['table_rel_l1'] < xo['table_rel_l1'] - max(xo['table_n0'] / 2, 0.02))
            acc_conf |= bool(ok)
    costs = {}
    for fn in sorted(glob.glob(os.path.join(R, 'results', 'cost_s*.json'))):
        c = jl(fn); costs[c['s']] = c
    cost_met = any(c['U'][a][k] >= 1.25 for c in costs.values() for a in ('LAM24', 'LAM64') for k in ('edit', 'source'))
    acc = o1 == 'S1_HEADROOM' and acc_conf
    if not s0['ALL']:
        v = 'INCONCLUSIVE'
    elif acc and cost_met:
        v = 'BOTH'
    elif acc:
        v = 'ACCURACY_HEADROOM'
    elif cost_met:
        v = 'COST_HEADROOM'
    elif o1 == 'S1_INCONCLUSIVE':
        v = 'INCONCLUSIVE'
    else:
        v = 'NO_MEANINGFUL_HEADROOM'
    res = {'stage0': s0['ALL'], 'S1': o1, 'S1_ext': ext, 'S2': s2, 'accuracy_confirmed': acc_conf, 'cost_met': cost_met,
           'U': {s: c['U'] for s, c in costs.items()}, 'LEARNED_RESOLVENT_UTILITY': v}
    jd(res, os.path.join(R, 'results', 'final.json'))
    print('FINAL', 'DEV' if DEV else v, flush=True)
    try:
        figures(R, outdir, ext)
    except Exception as e:                                                       # figures never change the verdict
        print('FIG_ERROR', repr(e), flush=True)


def figures(R, outdir, ext):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    sfx = '_ext' if ext else ''
    n = REPS_EXT if ext else REPS
    M = AN.masks(R); ref = AN.load_ref(R, 'none')
    h = AN.lum(ref['h'].mean(0)); tot = AN.lum(ref['tot'].mean(0))
    show = ['F', 'Z', 'XD', 'XP', 'XOD', 'XOP', 'XSD', 'G8601', 'G8602']
    X = {k: AN.lum(load_reps(R, k + sfx, n).mean(0)) for k in show}
    res = int(round(math.sqrt(h.size)))
    ys, xs = np.nonzero(M['ROI_T'].reshape(res, res))
    y0, y1, x0, x1 = max(ys.min() - 8, 0), min(ys.max() + 9, res), max(xs.min() - 8, 0), min(xs.max() + 9, res)
    crop = lambda a: a.reshape(res, res)[y0:y1, x0:x1]
    diffs = {k: X[k] - X['F'] for k in show if k != 'F'}
    emax = max(float(np.abs(v).max()) for v in diffs.values())
    vmax = float(np.percentile(np.concatenate([crop(X[k]).ravel() for k in show] + [crop(h).ravel()]), 99.5))
    fig, ax = plt.subplots(2, len(show), figsize=(2.3 * len(show), 5.0))
    for j, k in enumerate(show):
        ax[0, j].imshow(np.sqrt(np.clip(crop(X[k]), 0, None) / vmax), cmap='gray', vmin=0, vmax=1); ax[0, j].set_title(k, fontsize=8)
        if k != 'F':
            im = ax[1, j].imshow(np.sqrt(np.abs(crop(diffs[k])) / emax), cmap='magma', vmin=0, vmax=1)
        else:
            ax[1, j].imshow(np.sqrt(np.clip(crop(h), 0, None) / vmax), cmap='gray', vmin=0, vmax=1); ax[1, j].set_title('REF h', fontsize=8)
        for a in ax[:, j]:
            a.set_xticks([]); a.set_yticks([])
    cb = fig.colorbar(im, ax=ax[1, :].tolist(), fraction=0.02)
    cb.set_ticks([0, 0.25, 0.5, 1.0]); cb.set_ticklabels([f'{emax * t * t:.3g}' for t in (0, 0.25, 0.5, 1.0)])
    fig.suptitle('WL96A CK2 none higher-order (hl) close-ups: table = reference e2 + tail; row 2 = |arm - F| (sqrt map, shared absolute scale)', fontsize=9)
    fig.savefig(os.path.join(outdir, 'figW1_tail_arms.png'), dpi=130); plt.close(fig)
    # field: boxA exitance per patch, e3_o and e3_p, reference vs explicit
    rf = ref_fields(R, ext)
    T = np.load(os.path.join(R, 'results', f'tails{sfx}.npz'))
    ex = lambda Tb: (np.asarray(Tb, np.float64)[BOXA] * WT[None, :, None]).sum((1, 2))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for a, fam, dk, pk in ((ax[0], 'e3_o', 'dep_o', 'pool_o'), (ax[1], 'e3_p', 'dep_p', 'pool_p')):
        a.plot(ex(rf[fam]), 'k-', lw=1.5, label='reference (mean)')
        a.fill_between(np.arange(len(BOXA)), ex(rf[fam + '_a']), ex(rf[fam + '_b']), color='0.8', label='halves')
        a.plot(ex(T[dk]), 'C3-', lw=1, label='explicit deployed')
        a.plot(ex(np.asarray(T[pk]).mean(0)), 'C0--', lw=1, label='explicit pooled')
        a.set_title(f'box A exitance, {fam}', fontsize=9); a.set_xlabel('box A patch'); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'figW2_boxA_tail_field.png'), dpi=130); plt.close(fig)
    costs = sorted(glob.glob(os.path.join(R, 'results', 'cost_s*.json')))
    if costs:
        c = jl(costs[0])
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        for a, kind in zip(ax, ('edit', 'source')):
            arms = list(c['arms'])
            keys = sorted({k for ar in arms for k in c['arms'][ar][kind]['parts']})
            bottom = np.zeros(len(arms))
            for i, k in enumerate(keys):
                v = np.array([c['arms'][ar][kind]['parts'].get(k, 0.0) * 1e3 for ar in arms])
                a.bar(arms, v, bottom=bottom, label=k, color=plt.cm.tab20(i)); bottom += v
            a.set_ylabel('ms (median, warm)'); a.set_title(f'CK2 {kind} -> frame (N {c["N"]}, {c["frame_spp"]} spp)', fontsize=9)
        ax[1].legend(fontsize=7, loc='upper right')
        fig.tight_layout(); fig.savefig(os.path.join(outdir, 'figW3_lifecycle.png'), dpi=130); plt.close(fig)
    print('FIGS', outdir, flush=True)


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    cmd = a[0]
    if cmd == 'stage0':
        stage0(a[1], a[2])
    elif cmd == 'fext':
        fext(a[1], int(a[2]))
    elif cmd == 'tails':
        tails(a[1])
    elif cmd == 'an':
        an(a[1])
    elif cmd == 's2ref':
        s2ref(a[1], int(a[2]), int(a[3]))
    elif cmd == 's2fit':
        s2fit(a[1])
    elif cmd == 'cost':
        cost(a[1], a[2], float(a[3]) if len(a) > 3 else 1.0)
    elif cmd == 'final':
        final(a[1], a[2])
    os._exit(0)
