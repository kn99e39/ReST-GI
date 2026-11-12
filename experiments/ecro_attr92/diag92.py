"""WL92 diagnostics on the frozen ECRO v0 family (stages S0-S5 of protocol ecro_attr92). Nothing here is a deployable candidate.

  python diag92.py prov     <R> <R86>                 provenance hashes (checkpoints, data, references, environment)
  python diag92.py split    <R> <R86> <tag>           path-traced truth by family (e2_p, e2_o, e3_p, e3_o), q table, deployed encoding / state, importance
  python diag92.py local    <R> <R86>                 local operator: M64, singular spectrum, rank-16 NMF, learned a c^T of the frozen generators
  python diag92.py tables   <R> <R86> <tag>           ladder of 2-event variants (+ path-traced L3+), pooled realisations, frozen-operator terms
  python diag92.py fitdata  <R> <R86> <cfg>           fixed-geometry fit / validation lights with split targets
  python diag92.py fit      <R> <R86> <cfg> <variant> free-factor fit of the unchanged operator family (non-deployable oracle)
  python diag92.py render   <R> <R86> <tag> <list>    LOW + lookup renders of tables (list lines: <npz>:<key> <seed> <spp> <outtag>)
"""
import glob, hashlib, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEV = '--dev' in sys.argv
PARTS = ('e2_p', 'e2_o', 'e3_p', 'e3_o')


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1)


def sha(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def listing_sha(fns):
    return hashlib.sha256('\n'.join(f'{os.path.basename(f)}:{os.path.getsize(f)}' for f in sorted(fns)).encode()).hexdigest()


def halves_mean(z, keys):
    return sum(0.5 * (z[f'{k}_a'].astype(np.float64) + z[f'{k}_b'].astype(np.float64)) for k in keys)


# ------------------------------------------------------------------------------------------------ provenance
def prov(R, R86):
    import torch
    import fx92
    E86 = os.path.join(HERE, '..', 'ecro86')
    o = {'checkpoints': {sd: sha(os.path.join(R86, 'results', 'models', f'{sd}.pt')) for sd in fx92.SEEDS},
         'er_states': sha(os.path.join(E86, 'data', 'mats86.npz')), 'protocol86': sha(os.path.join(E86, 'protocol', 'ecro86.json')),
         'model86': sha(os.path.join(E86, 'model86.py')), 'train86': sha(os.path.join(E86, 'train86.py')), 'data': {}, 'references': {}, 'fields': {}, 'masks': {}}
    for sp in ('train', 'val', 'trainhq'):
        fns = sorted(glob.glob(os.path.join(R86, 'data', sp, 'g*.npz')))
        o['data'][sp] = {'files': len(fns), 'listing_sha256': listing_sha(fns), 'first4_sha256': [sha(f) for f in fns[:4]]}
    for pat in ('REF_ER_*', 'REF_P_*', 'ORA_*', 'EC_*'):
        fns = glob.glob(os.path.join(R86, 'results', 'renders', pat + '.npz'))
        o['references'][pat] = {'files': len(fns), 'listing_sha256': listing_sha(fns)}
    for fn in sorted(glob.glob(os.path.join(R86, 'results', 'fields', 'F_*.npz'))):
        o['fields'][os.path.basename(fn)] = sha(fn)
    for fn in sorted(glob.glob(os.path.join(R86, 'results', 'masks', '*.npz'))):
        o['masks'][os.path.basename(fn)] = sha(fn)
    o['wl86_final_sha256'] = sha(os.path.join(R86, 'results', 'final.json'))
    o['env'] = {'torch': torch.__version__, 'mitsuba': fx92.mi.__version__, 'drjit': fx92.dr.__version__,
                'gpu': os.popen('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader').read().strip()}
    cf = os.path.join(R, 'code', 'COMMIT')
    o['code_commit'] = open(cf).read().strip() if os.path.exists(cf) else None
    jd(o, os.path.join(R, 'results', 'prov.json'))
    print('PROV', json.dumps(o['checkpoints']), o['data']['train']['files'], flush=True)


# ------------------------------------------------------------------------------------------------ truth split
def split(R, R86, tag):
    import fx92
    fx92.inject(R)
    import pt86a as A
    import ecro86 as EC
    import quad87 as Q
    import diag86a as DG
    PT, FX, E87 = fx92.PT, fx92.FX, fx92.E87
    geom, ls, e, cfg = fx92.cfg_ls(tag)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    spp = 64 if DEV else fx92.P92['truth']['spp_per_half']
    d = {}
    for h, sd in zip('ab', fx92.P92['truth']['seeds']):
        t = A.field_src(ctx, A.Src(lt, 'true', ()), spp, sd, split=True)
        for k, v in t.items():
            d[f'{k}_{h}'] = v
    st = EC.build_state(ctx)
    d.update(s16=ctx.source(lt), q=Q.qtable(ctx, lt, E87['P'], E87['Mq'], E87['seed_q']).astype(np.float32), hit=st.hit, dbin=st.dbin, seg=st.seg)
    mz = np.load(fx92.mask_file(R, R86, tag))
    mks = {k: mz[k] for k in ('C', 'obj_panel', 'D', 'stationary_all', 'A') if k in mz.files and mz[k].sum() >= 30}
    cam = FX['camera']
    W = DG.importance(PT, ctx, cam, cam['res'], 64 if DEV else 1024, 86600, mks)
    out = os.path.join(R, 'results', 'fields')
    np.savez_compressed(os.path.join(out, f'D_{tag}.npz'), **d)
    np.savez_compressed(os.path.join(out, f'W_{tag}.npz'), **{k: v.astype(np.float32) for k, v in W.items()})
    print('split', tag, {k: round(float(v.sum()), 2) for k, v in d.items() if k[:2] in ('e2', 'e3')}, st.digest(), flush=True)


# ------------------------------------------------------------------------------------------------ local operator
def nmf(V, r, iters, restarts, seed):
    """Frobenius NMF V ~ W H (multiplicative updates), best of `restarts`. V [n, m] >= 0."""
    best = None
    eps = 1e-12
    sc = math.sqrt(max(V.mean(), 1e-30) / r)
    for t in range(restarts):
        rng = np.random.default_rng([seed, t])
        W = rng.random((V.shape[0], r)) * sc + eps; H = rng.random((r, V.shape[1])) * sc + eps
        for _ in range(iters):
            H *= (W.T @ V) / (W.T @ W @ H + eps)
            W *= (V @ H.T) / (W @ H @ H.T + eps)
        err = float(np.linalg.norm(V - W @ H))
        if best is None or err < best[0]:
            best = (err, W.copy(), H.copy())
    return best


def flux_w():
    """Mean cosine of every bin's ring: z_b. M[bi, bo] z_bi z_bo is the flux-transfer matrix (incident-cosine-weighted irradiance in a bin
    -> outgoing flux in a bin); its rows sum to the directional albedo, so all incident bins are commensurable."""
    import fx92
    S = fx92.S
    return ((np.arange(S.NB) // S.NPHI) + 0.5) / S.NZ


def mat_metrics(Mh, Mt, zb):
    """Mh, Mt [bi, bo] compared as flux-transfer matrices: relative Frobenius residual, relative L1 misplacement, retention of the true
    peak (mirror) bin of every incident bin, total mass ratio."""
    Xh, Xt = Mh * zb[:, None] * zb[None, :], Mt * zb[:, None] * zb[None, :]
    pk = Xt.argmax(1)
    ii = np.arange(Xt.shape[0])
    return {'fro': float(np.linalg.norm(Xh - Xt) / max(np.linalg.norm(Xt), 1e-30)), 'l1': float(np.abs(Xh - Xt).sum() / max(Xt.sum(), 1e-30)),
            'peak': float(Xh[ii, pk].sum() / max(Xt[ii, pk].sum(), 1e-30)), 'mass': float(Xh.sum() / max(Xt.sum(), 1e-30)),
            'peak_share_of_truth': float(Xt[ii, pk].sum() / max(Xt.sum(), 1e-30))}


def local_matrices():
    import fx92
    import quad87 as Q
    import xport87 as X
    E87 = fx92.E87
    quad = Q.Quad(X.realization_us(86001), E87['M_ray'], E87['seed_ray'], E87['M_dense'], E87['seed_dense'])
    return quad, quad.enc_matrix()                       # Mq [6, bi, bo, 3]


def local(R, R86):
    import torch
    import fx92
    import ecro86 as EC
    import model86 as M
    S, FX = fx92.S, fx92.FX
    NB = S.NB
    L = fx92.P92['local_operator']
    quad, Mq = local_matrices()
    r = L['nmf']['rank']
    res = {'materials': {}, 'learned': {}}
    Cn = np.zeros((3, 3, NB, r)); An = np.zeros((3, 3, NB, r))
    names = {0: 'B_floor', 1: 'S1_panel', 2: 'S2_sphere'}
    zb = flux_w()
    for m in range(3):
        o = {}
        for c in range(3):
            V = Mq[m, :, :, c]
            X = V * zb[:, None] * zb[None, :]                                   # flux-transfer matrix
            U_, sv, Vt = np.linalg.svd(X)
            e2 = sv ** 2 / (sv ** 2).sum()
            X16 = (U_[:, :r] * sv[:r]) @ Vt[:r]
            it, rs = (200, 2) if DEV else (L['nmf']['iterations'], L['nmf']['restarts'])
            err, W, H = nmf(X, r, it, rs, L['nmf']['seed'] + 10 * m + c)
            Cn[m, c] = W / zb[:, None]; An[m, c] = H.T / zb[:, None]             # V ~ Cn An^T, non-negative, rank 16
            svu = np.linalg.svd(V, compute_uv=False); eu = svu ** 2 / (svu ** 2).sum()
            o[c] = {'singular_values': sv.tolist(), 'eckart_young_rank16_fro': float(math.sqrt(e2[r:].sum())),
                    'rank_for_energy': {str(q): int(np.searchsorted(np.cumsum(e2), q) + 1) for q in (0.9, 0.99, 0.999)},
                    'participation_rank': float((sv.sum() ** 2) / (sv ** 2).sum()),
                    'svd16': mat_metrics(X16 / zb[:, None] / zb[None, :], V, zb), 'nmf16': mat_metrics(Cn[m, c] @ An[m, c].T, V, zb),
                    'directional_albedo_mean': float((X.sum(1) * (2 * math.pi / NB) / zb).mean()),
                    'unweighted': {'eckart_young_rank16_fro': float(math.sqrt(eu[r:].sum())), 'note': 'plain Frobenius norm of M64 (dominated by the grazing ring); not used'}}
        res['materials'][names[m]] = o
    np.savez_compressed(os.path.join(R, 'results', 'nmf.npz'), Cn=Cn, An=An, Mq=Mq.astype(np.float32))
    # learned a c^T of the frozen generators on the E2_T1 geometry
    z = np.load(os.path.join(R, 'results', 'fields', 'D_E2_T1.npz'))
    G = S.Geom(FX['E2']['T1'])
    gs = M.GeoState(G, z['hit'], z['dbin'], EC.dev)
    for sd in fx92.SEEDS:
        mdl = EC.load_model(R86, sd)
        with torch.no_grad():
            f = mdl.factors(gs)
            Ml = torch.einsum('ndcr,nmcr->nmdc', f['a'], f['c']).double().cpu().numpy()       # [N, bi, bo, 3]
        o = {}
        for m in range(6):
            sel = np.nonzero(G.PMAT == m)[0]
            if not len(sel):
                continue
            rows = [mat_metrics(Ml[n, :, :, c], Mq[m, :, :, c], zb) for n in sel for c in range(3)]
            o[str(m)] = {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}
            o[str(m)]['patches'] = int(len(sel))
            sv = np.linalg.svd(Ml[sel[0], :, :, 1], compute_uv=False)
            o[str(m)]['participation_rank_first_patch'] = float((sv.sum() ** 2) / (sv ** 2).sum())
        res['learned'][sd] = o
    jd(res, os.path.join(R, 'results', 'local.json'))
    for k, o in res['materials'].items():
        print('local', k, {c: (round(o[c]['eckart_young_rank16_fro'], 3), round(o[c]['nmf16']['fro'], 3), round(o[c]['nmf16']['peak'], 3), o[c]['rank_for_energy']['0.99']) for c in range(3)}, flush=True)
    for sd in fx92.SEEDS:
        print('learned', sd, {m: {k: round(v, 3) for k, v in o.items() if k in ('fro', 'l1', 'peak', 'mass')} for m, o in res['learned'][sd].items()}, flush=True)


# ------------------------------------------------------------------------------------------------ factor constructions (ECRO form)
def f_exact(G, Mq, dev):
    """R = 64 (non-deployable capacity control): c = identity over incident bins, a = M64 of the patch material."""
    import torch
    N, NB = G.N, Mq.shape[1]
    c = torch.eye(NB, device=dev).view(1, NB, 1, NB).expand(N, NB, 3, NB).contiguous()
    a = torch.as_tensor(Mq[G.PMAT].astype(np.float32), device=dev).permute(0, 2, 3, 1).contiguous()       # [N, bo, col, r = bi]
    return {'c': c, 'a': a}


def f_nmf(G, Cn, An, dev):
    """R = 16, in-family: rank-16 non-negative factors of M64 by patch material (Lambert: exact rank 1)."""
    import torch
    import quad87 as Q
    _, lam, alb = Q.mats()
    N, NB, r = G.N, Cn.shape[2], Cn.shape[3]
    c = np.zeros((N, NB, 3, r), np.float32); a = np.zeros((N, NB, 3, r), np.float32)
    for m in range(6):
        sel = G.PMAT == m
        if not sel.any():
            continue
        if lam[m]:
            c[sel, :, :, 0] = 1.0; a[sel, :, :, 0] = alb[m] / math.pi
        else:
            c[sel] = Cn[m].transpose(1, 0, 2)[None]            # [bi, col, r]
            a[sel] = An[m].transpose(1, 0, 2)[None]            # [bo, col, r]
    return {'c': torch.as_tensor(c, device=dev), 'a': torch.as_tensor(a, device=dev)}


def rw_of(us):
    import fx92
    S = fx92.S
    k = np.arange(S.K)
    z = ((k // S.KAPPA) // S.NPHI + us[2, 0]) / S.NZ
    return (z * S.DOMEGA / S.KAPPA).astype(np.float32)


def single_panel(mdl, f, gs, s):
    """Single-flight term restricted to state rays that hit a panel patch (material 1): [L, N, NB, 3]."""
    import torch
    import fx92
    S = fx92.S
    NB, K, KAPPA = S.NB, S.K, S.KAPPA
    Ls, N = s.shape[0], gs.N
    u = torch.einsum('nmcr,lnmc->lncr', f['c'], s)
    Bq = torch.einsum('ndcr,lncr->lndc', f['a'], u).reshape(Ls, N * NB, 3)
    okp = ((gs.hit >= 0) & (gs.mat[gs.hit.clamp(min=0)] == 1)).view(1, N * K, 1)
    idx = (gs.hit.clamp(min=0) * NB + gs.dbin).view(-1)
    E1 = ((Bq[:, idx] * okp).view(Ls, N, NB, KAPPA, 3) * mdl.rw.view(1, 1, NB, KAPPA, 1)).sum(3)
    return torch.einsum('ndcr,lncr->lndc', f['a'], torch.einsum('nmcr,lnmc->lncr', f['c'], E1))


# ------------------------------------------------------------------------------------------------ ladder tables
def tables(R, R86, tag):
    import torch
    import fx92
    fx92.inject(R)
    import ecro86 as EC
    import model86 as M
    import quad87 as Q
    import xport87 as X
    PT, FX, S, E87, P92 = fx92.PT, fx92.FX, fx92.S, fx92.E87, fx92.P92
    NB = S.NB
    dev = EC.dev
    geom, ls, e, cfg = fx92.cfg_ls(tag)
    z = np.load(os.path.join(R, 'results', 'fields', f'D_{tag}.npz'))
    Wz = np.load(os.path.join(R, 'results', 'fields', f'W_{tag}.npz'))
    L2T, L3T, E2P = halves_mean(z, ('e2_p', 'e2_o')), halves_mean(z, ('e3_p', 'e3_o')), halves_mean(z, ('e2_p',))
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls]); G = ctx.G
    st = EC.build_state(ctx)
    assert np.array_equal(st.hit, z['hit']) and np.array_equal(st.dbin, z['dbin']), 'state mismatch'
    us0 = X.realization_us(86001)
    quad0 = Q.Quad(us0, E87['M_ray'], E87['seed_ray'], E87['M_dense'], E87['seed_dense'])
    Mq = quad0.enc_matrix()
    nm = np.load(os.path.join(R, 'results', 'nmf.npz'))
    T = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=dev)
    s = T(z['s16']).view(1, G.N, NB, 3)
    q = T(z['q'])
    mdl0 = EC.load_model(R86, fx92.SEEDS[0])
    rw0 = mdl0.rw.clone()
    fE, fK = f_exact(G, Mq, dev), f_nmf(G, nm['Cn'], nm['An'], dev)
    gs = M.GeoState(G, st.hit, st.dbin, dev)
    with torch.no_grad():
        sing = lambda f, g=gs: mdl0.apply(f, g, s, use_global=False)[0].double().cpu().numpy()
        op = X.XOp(G, st.hit, st.dbin, quad0.W)
        encq = T(X.enc_q(G, z['s16'], Mq))
        L2 = {'RS': X.raysend_first(ctx, st, lt, quad0.W).astype(np.float64), 'EXP': op.apply(q).double().cpu().numpy(),
              'ENCQ': op.apply(encq).double().cpu().numpy(), 'ENCB': sing(fE), 'R16': sing(fK)}
        form = X.XOp(G, st.hit, st.dbin, quad0.binq_W()).apply(encq).double().cpu().numpy()
        Bx, _ = X.recurrence([op], z['q'])
        Bx = Bx.double().cpu().numpy()
        # pooled realisations (non-deployable adequate-sampling control)
        seeds = P92['ladder']['pooled']['seeds'][:(2 if DEV else None)]
        pool = {v: np.zeros_like(L2T) for v in ('EXP', 'ENCB', 'R16')}
        per = {v: [] for v in pool}
        Wc = Wz['C'].astype(np.float64).reshape(G.N, NB, 3) if 'C' in Wz.files else None
        for i, rs in enumerate(seeds):
            us = X.realization_us(rs)
            X.set_realization(ctx, us)
            sti = EC.build_state(ctx)
            qi = quad0 if i == 0 else Q.Quad(us, E87['M_ray'], E87['seed_ray'] + 1000 * i)
            gi = M.GeoState(G, sti.hit, sti.dbin, dev)
            mdl0.rw.copy_(T(rw_of(us)))
            cur = {'EXP': X.XOp(G, sti.hit, sti.dbin, qi.W).apply(q).double().cpu().numpy(), 'ENCB': sing(fE, gi), 'R16': sing(fK, gi)}
            for v in pool:
                pool[v] += cur[v] / len(seeds)
                if Wc is not None:
                    per[v].append(float((Wc * (cur[v] - L2T)).sum() / (Wc * (L2T + L3T)).sum()))
            if i == 0:
                assert np.abs(cur['EXP'] - L2['EXP']).max() <= 1e-5 * max(L2['EXP'].max(), 1e-30), 'deployed realisation mismatch'
        X.set_realization(ctx, us0); mdl0.rw.copy_(rw0)
        out = {'L2T': L2T, 'L3T': L3T, 'E2P': E2P, 'ORA2': L2T + L3T, 'X_EXP': Bx, 'FORM_BINQ': form}
        for v, x in L2.items():
            out[f'L2_{v}'] = x; out[f'L_{v}'] = x + L3T
        for v, x in pool.items():
            out[f'P2_{v}'] = x; out[f'P_{v}'] = x + L3T
        meta = {'form_rel_l1_ENCB_vs_binq': float(np.abs(L2['ENCB'] - form).sum() / max(form.sum(), 1e-30)), 'pooled_seeds': seeds,
                'per_realisation_signed_C_over_h': per, 'state_digest': st.digest(), 'own_sum': {}}
        for sd in fx92.SEEDS:
            mdl = EC.load_model(R86, sd)
            f = mdl.factors(gs)
            B = mdl.apply(f, gs, s)[0].double().cpu().numpy()
            Bs = mdl.apply(f, gs, s, use_global=False)[0].double().cpu().numpy()
            Bg = mdl.apply(f, gs, s, use_single=False)[0].double().cpu().numpy()
            Bsp = single_panel(mdl, f, gs, s)[0].double().cpu().numpy()
            u = torch.einsum('nmcr,lnmc->lncr', f['c'], s)
            Bq = torch.einsum('ndcr,lncr->lndc', f['a'], u)[0].double().cpu().numpy()
            meta['own_sum'][sd] = float(np.abs(B - Bs - Bg).max() / max(B.max(), 1e-30))
            out.update({f'B_{sd}': B, f'S_{sd}': Bs, f'G_{sd}': Bg, f'SP_{sd}': Bsp, f'Bq_{sd}': Bq, f'SL_{sd}': Bs + L3T, f'LG_{sd}': L2T + Bg})
    np.savez_compressed(os.path.join(R, 'results', 'tables', f'T_{tag}.npz'), **{k: v.astype(np.float32) for k, v in out.items()})
    jd(meta, os.path.join(R, 'results', 'tables', f'T_{tag}.json'))
    print('tables', tag, json.dumps({'form': round(meta['form_rel_l1_ENCB_vs_binq'], 4), 'own_sum': meta['own_sum']}), flush=True)


# ------------------------------------------------------------------------------------------------ fixed-geometry fits
def fitdata(R, R86, cfg):
    import fx92
    fx92.inject(R)
    import pt86a as A
    PT, FX, S, P92 = fx92.PT, fx92.FX, fx92.S, fx92.P92
    Lc = P92['fits']['lights']
    geom, ls, e, _ = fx92.cfg_ls(cfg)
    ctx = PT.Ctx(geom)
    rng = np.random.default_rng([Lc['seed'], P92['fits']['geometries'].index(cfg)])
    nf, nv = (6, 2) if DEV else (Lc['fit'], Lc['val'])
    lights = [S.sample_light(rng, geom) for _ in range(nf + nv)]
    sf, sv = (32, 32) if DEV else (Lc['fit_spp'], Lc['val_spp_per_half'])
    s, f2, f3, f2p, va, vb = [], [], [], [], {k: [] for k in ('f2', 'f3')}, {k: [] for k in ('f2', 'f3')}
    t0 = time.time()
    for i, l in enumerate(lights):
        lt = PT.Lights(l)
        s.append(ctx.source(lt))
        if i < nf:
            t = A.field_src(ctx, A.Src(lt, 'true', ()), sf, Lc['seed_fit'] + i, split=True)
            f2.append((t['e2_p'] + t['e2_o']).astype(np.float16)); f3.append((t['e3_p'] + t['e3_o']).astype(np.float16)); f2p.append(t['e2_p'].astype(np.float16))
        else:
            for dst, sd0 in ((va, Lc['seed_val'][0]), (vb, Lc['seed_val'][1])):
                t = A.field_src(ctx, A.Src(lt, 'true', ()), sv, sd0 + i, split=True)
                dst['f2'].append((t['e2_p'] + t['e2_o']).astype(np.float16)); dst['f3'].append((t['e3_p'] + t['e3_o']).astype(np.float16))
        if i % 16 == 0:
            print('fitdata', cfg, i, round(time.time() - t0, 1), 's', flush=True)
    np.savez_compressed(os.path.join(R, 'results', 'fits', f'fitdata_{cfg}.npz'), lights=json.dumps(lights), nf=nf, s=np.stack(s),
                        f2=np.stack(f2), f3=np.stack(f3), f2p=np.stack(f2p), v2a=np.stack(va['f2']), v3a=np.stack(va['f3']), v2b=np.stack(vb['f2']), v3b=np.stack(vb['f3']))
    tl = S.test_light_positions()
    dmin = float(min(np.linalg.norm(tl - np.array(l[0]['pos']), axis=1).min() for l in lights))
    jd({'cfg': cfg, 'fit_lights': nf, 'val_lights': nv, 'min_distance_to_test_light_m': dmin, 'seconds': time.time() - t0, 'gpu': PT.gpu_mem()},
       os.path.join(R, 'results', 'fits', f'fitdata_{cfg}.json'))
    print('FITDATA_DONE', cfg, nf, nv, round(time.time() - t0, 1), 's', flush=True)


def fit(R, R86, cfg, variant):
    import torch
    import torch.nn.functional as Fn
    import fx92
    fx92.inject(R)
    import ecro86 as EC
    import model86 as M
    FX, S, P92 = fx92.FX, fx92.S, fx92.P92
    NB = S.NB
    dev = EC.dev
    O = P92['fits']['optimiser']
    obj, ini = variant.split('-')
    geom, ls, e, _ = fx92.cfg_ls(cfg)
    G = S.Geom(geom)
    D = np.load(os.path.join(R, 'results', 'fields', f'D_{cfg}.npz'))
    Wz = np.load(os.path.join(R, 'results', 'fields', f'W_{cfg}.npz'))
    z = np.load(os.path.join(R, 'results', 'fits', f'fitdata_{cfg}.npz'))
    nm = np.load(os.path.join(R, 'results', 'nmf.npz'))
    gs = M.GeoState(G, D['hit'], D['dbin'], dev)
    mdl = EC.load_model(R86, fx92.SEEDS[0])                 # used for apply() only; its generator weights are not used by a fit
    T = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=dev)
    nf = int(z['nf'])
    s_fit, t2, t3 = T(z['s'][:nf]), T(z['f2']), T(z['f3'])
    s_val = T(z['s'][nf:]); v2a, v3a, v2b, v3b = T(z['v2a']), T(z['v3a']), T(z['v2b']), T(z['v3b'])
    s_w = T(D['s16']).view(1, G.N, NB, 3)
    L2T, L3T, E2P = halves_mean(D, ('e2_p', 'e2_o')), halves_mean(D, ('e3_p', 'e3_o')), halves_mean(D, ('e2_p',))
    with torch.no_grad():
        if ini[0] == 'f':
            f0 = EC.load_model(R86, 'ecro_s' + ini[1:]).factors(gs)
        elif ini[0] == 'r':
            torch.manual_seed(92000 + int(ini[1:]))
            f0 = M.ECRO().to(dev).factors(gs)
        else:
            f0 = dict(f_nmf(G, nm['Cn'], nm['An'], dev))
            fz = mdl.factors(gs)
            f0['Q'], f0['P'] = fz['Q'], fz['P']
    f0 = {k: v.detach().clone() for k, v in f0.items()}

    def terms(f, s):
        return mdl.apply(f, gs, s, use_global=False), mdl.apply(f, gs, s, use_single=False)

    def losses(Bs, Bg, a2, a3):
        t = a2 + a3
        den = t.mean(2, keepdim=True) + 0.05 * t.mean(dim=(1, 2, 3), keepdim=True)
        return (((Bs + Bg - t) / den) ** 2).mean(), (((Bs - a2) / den) ** 2).mean() + (((Bg - a3) / den) ** 2).mean()

    @torch.no_grad()
    def ev(f):
        o = {}
        acc = {k: 0.0 for k in ('sum_n', 'sum_d', 's_n', 's_d', 'g_n', 'g_d', 'sq', 'sqd', 's_sq', 's_sqd', 'g_sq', 'g_sqd', 'Ls', 'Lc')}
        nb = 0
        for i in range(0, s_val.shape[0], 8):
            Bs, Bg = terms(f, s_val[i:i + 8])
            a2, a3, b2, b3 = v2a[i:i + 8], v3a[i:i + 8], v2b[i:i + 8], v3b[i:i + 8]
            m2, m3 = 0.5 * (a2 + b2), 0.5 * (a3 + b3)
            B = Bs + Bg
            acc['sum_n'] += float((B - m2 - m3).abs().sum()); acc['sum_d'] += float((m2 + m3).sum())
            acc['s_n'] += float((Bs - m2).abs().sum()); acc['s_d'] += float(m2.sum())
            acc['g_n'] += float((Bg - m3).abs().sum()); acc['g_d'] += float(m3.sum())
            acc['sq'] += float(((B - a2 - a3) * (B - b2 - b3)).sum()); acc['sqd'] += float(((a2 + a3) * (b2 + b3)).sum())
            acc['s_sq'] += float(((Bs - a2) * (Bs - b2)).sum()); acc['s_sqd'] += float((a2 * b2).sum())
            acc['g_sq'] += float(((Bg - a3) * (Bg - b3)).sum()); acc['g_sqd'] += float((a3 * b3).sum())
            ls_, lc_ = losses(Bs, Bg, m2, m3)
            acc['Ls'] += float(ls_); acc['Lc'] += float(lc_); nb += 1
        rr = lambda n, d: math.sqrt(max(acc[n], 0.0) / max(acc[d], 1e-30))
        o['val'] = {'sum_rel_l1': acc['sum_n'] / acc['sum_d'], 'single_vs_L2_rel_l1': acc['s_n'] / acc['s_d'], 'global_vs_L3_rel_l1': acc['g_n'] / acc['g_d'],
                    'sum_rel_rms_debiased': rr('sq', 'sqd'), 'single_vs_L2_rel_rms_debiased': rr('s_sq', 's_sqd'), 'global_vs_L3_rel_rms_debiased': rr('g_sq', 'g_sqd'),
                    'single_over_L2_flux': None, 'L_sum': acc['Ls'] / nb, 'L_comp': acc['Lc'] / nb}
        n32 = min(32, nf)
        fs_n = fs_d = fl = fc = 0.0; k_ = 0
        s_flux = t_flux = 0.0
        for i in range(0, n32, 8):
            Bs, Bg = terms(f, s_fit[i:i + 8])
            a2, a3 = t2[i:i + 8], t3[i:i + 8]
            fs_n += float((Bs + Bg - a2 - a3).abs().sum()); fs_d += float((a2 + a3).sum())
            s_flux += float(Bs.sum()); t_flux += float(a2.sum())
            ls_, lc_ = losses(Bs, Bg, a2, a3)
            fl += float(ls_); fc += float(lc_); k_ += 1
        o['fit32'] = {'sum_rel_l1': fs_n / fs_d, 'L_sum': fl / k_, 'L_comp': fc / k_, 'single_over_L2_flux': s_flux / max(t_flux, 1e-30)}
        Bs, Bg = terms(f, s_w)
        Bsp = single_panel(mdl, f, gs, s_w)
        tb = {'B': (Bs + Bg)[0].double().cpu().numpy(), 'S': Bs[0].double().cpu().numpy(), 'G': Bg[0].double().cpu().numpy(), 'SP': Bsp[0].double().cpu().numpy()}
        o['workload'] = {'field': {'sum_rel_l1': float(np.abs(tb['B'] - L2T - L3T).sum() / (L2T + L3T).sum()), 'single_vs_L2_rel_l1': float(np.abs(tb['S'] - L2T).sum() / L2T.sum()),
                                   'global_vs_L3_rel_l1': float(np.abs(tb['G'] - L3T).sum() / L3T.sum())}, 'read': {}}
        for rk in Wz.files:
            Wr = Wz[rk].astype(np.float64).reshape(G.N, NB, 3)
            rd = lambda X_: float((Wr * X_).sum())
            o['workload']['read'][rk] = {'h': rd(L2T + L3T), 'L2': rd(L2T), 'L2_panel': rd(E2P), 'L3': rd(L3T), 'B': rd(tb['B']), 'S': rd(tb['S']), 'G': rd(tb['G']), 'SP': rd(tb['SP'])}
        return o, tb

    res = {'cfg': cfg, 'variant': variant, 'objective': obj, 'init': ini}
    res['init_eval'], tb0 = ev(f0)
    isp = lambda y: torch.log(torch.expm1(y.clamp(min=1e-5)))
    raw = {k: isp(v).requires_grad_(True) for k, v in f0.items()}
    opt = torch.optim.Adam(list(raw.values()), lr=O['lr'])
    steps = 1000 if DEV else O['steps']
    torch.manual_seed(O['torch_seed'])
    rng = np.random.default_rng(O['batch_seed'])
    bl = min(O['batch_lights'], nf)
    curve = []; run = []
    t0 = time.time()
    for i in range(steps):
        for g in opt.param_groups:
            g['lr'] = O['lr_final'] + 0.5 * (O['lr'] - O['lr_final']) * (1 + math.cos(math.pi * i / max(steps - 1, 1)))
        b = torch.as_tensor(rng.choice(nf, bl, replace=False), device=dev)
        f = {k: Fn.softplus(v) for k, v in raw.items()}
        Bs, Bg = terms(f, s_fit[b])
        ls_, lc_ = losses(Bs, Bg, t2[b], t3[b])
        loss = ls_ if obj == 'SUM' else ls_ + lc_
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        run.append((float(loss), float(ls_), float(lc_)))
        if (i + 1) % 250 == 0 or i + 1 == steps:
            m = np.mean(run, 0)
            curve.append({'step': i + 1, 'objective': float(m[0]), 'L_sum': float(m[1]), 'L_comp': float(m[2])}); run = []
    f = {k: Fn.softplus(v).detach() for k, v in raw.items()}
    res['final_eval'], tb = ev(f)
    res['curve'] = curve
    n = 2 if len(curve) >= 4 else 1                                                        # 2 records = 500 steps
    last = np.mean([c['objective'] for c in curve[-n:]])
    prev = np.mean([c['objective'] for c in curve[-2 * n:-n]]) if len(curve) >= 2 * n else float('inf')
    res['convergence'] = {'last': float(last), 'previous': float(prev), 'relative_change': float(abs(last - prev) / max(prev, 1e-30)),
                          'converged': bool(abs(last - prev) <= 0.02 * prev)}
    res['seconds'] = time.time() - t0; res['gpu_mem_peak_mib'] = torch.cuda.max_memory_allocated() / 2 ** 20
    # superposition / scaling / zero of the fitted factor set
    with torch.no_grad():
        a_, b_ = s_fit[:1], s_fit[1:2]
        lin = float(((mdl.apply(f, gs, a_ + b_) - mdl.apply(f, gs, a_) - mdl.apply(f, gs, b_)).abs().max() / mdl.apply(f, gs, a_ + b_).abs().max()))
        res['contracts'] = {'superposition': lin, 'zero': float(mdl.apply(f, gs, torch.zeros_like(a_)).abs().max()), 'min': float(mdl.apply(f, gs, a_).min())}
    out = os.path.join(R, 'results', 'fits')
    np.savez_compressed(os.path.join(out, f'fit_{cfg}_{variant}.npz'), **{k: v.astype(np.float32) for k, v in tb.items()}, SL=(tb['S'] + L3T).astype(np.float32),
                        S0=tb0['S'].astype(np.float32), SL0=(tb0['S'] + L3T).astype(np.float32), B0=tb0['B'].astype(np.float32))
    jd(res, os.path.join(out, f'fit_{cfg}_{variant}.json'))
    r4 = lambda v: round(v, 4)
    print('fit', cfg, variant, json.dumps({'L_sum_val': [r4(res['init_eval']['val']['L_sum']), r4(res['final_eval']['val']['L_sum'])],
                                           'single_L2_val': [r4(res['init_eval']['val']['single_vs_L2_rel_l1']), r4(res['final_eval']['val']['single_vs_L2_rel_l1'])],
                                           'sum_val': [r4(res['init_eval']['val']['sum_rel_l1']), r4(res['final_eval']['val']['sum_rel_l1'])],
                                           'conv': res['convergence']['converged'], 's': round(res['seconds'], 1)}), flush=True)


# ------------------------------------------------------------------------------------------------ renders
def render(R, R86, tag, listfn):
    import fx92
    fx92.inject(R)
    PT, FX = fx92.PT, fx92.FX
    geom, ls, e, cfg = fx92.cfg_ls(tag)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    cam = FX['camera']
    out = os.path.join(R, 'results', 'renders'); os.makedirs(out, exist_ok=True)
    cache = {}
    for line in open(listfn):
        if not line.strip():
            continue
        src, seed, spp, otag = line.split()
        if os.path.exists(os.path.join(out, f'{otag}.npz')):
            continue
        fn, key = src.split(':')
        if fn not in cache:
            cache = {fn: np.load(os.path.join(R, fn) if not fn.startswith('/') else fn)}
        B = cache[fn][key].astype(np.float32)
        acc, T_ = ctx.render(lt, 'ER', cam, cam['res'], int(spp), int(seed), 'ecro', B=PT.fl(B), b=32)
        o = PT.pack(acc, cam['res'], int(spp))
        np.savez_compressed(os.path.join(out, f'{otag}.npz'), hl=o['hl'])
        print(otag, round(T_['wall_s'], 2), flush=True)
    print('RENDER_DONE', tag, flush=True)


if __name__ == '__main__':
    cmd, R, R86 = sys.argv[1], sys.argv[2], sys.argv[3]
    a = [x for x in sys.argv[4:] if not x.startswith('--')]
    for d in ('fields', 'tables', 'fits', 'renders', 'lists'):
        os.makedirs(os.path.join(R, 'results', d), exist_ok=True)
    {'prov': prov, 'split': split, 'local': local, 'tables': tables, 'fitdata': fitdata, 'fit': fit, 'render': render}[cmd](R, R86, *a)
    os._exit(0)
