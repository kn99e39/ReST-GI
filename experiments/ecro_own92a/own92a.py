"""WL92A: ECRO physical component supervision - controlled generator learning (protocol ecro_own92a).

  python own92a.py excl     <R> <R86>                       training exclusion list for E4 (geometry / light positions only)
  python own92a.py gen      <R> <R86> <split> <start> <count>  component targets of the stored WL86 items (same seeds and samples) + identity check
  python own92a.py train    <R> <R86> <seed> <sum|own>       WL86 training contract with the SUM or OWN objective
  python own92a.py fieldeval <R> <R86>                      field-level whole / component errors of every model, contracts, T_CTL
  python own92a.py mtables  <R> <R86> <tag>                 tables and terms of every model (+ STALE; + ORACLE / X_EXP on E4)
  python own92a.py local    <R> <R86>                       learned local matrices a c^T and Bq of every model against frozen ER
  python own92a.py timing   <R> <R86>                       runtime costs per model on E4 (warm)
  python own92a.py ladder   <R> <R86>                       conditional quality ladder on E4 T1 (outcome A only)
Uses the WL86 / WL86A / WL87 / WL92 modules unchanged.
"""
import glob, hashlib, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_attr92'))
sys.path.insert(0, HERE)
DEV = '--dev' in sys.argv
P = json.load(open(os.path.join(HERE, 'protocol', 'ecro_own92a.json')))
FROZEN = ['ecro_s8601', 'ecro_s8602']


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1)


def jl(fn):
    return json.load(open(fn))


def models(R):
    out = [('frz', n) for n in FROZEN]
    for arm in ('sum', 'own'):
        for sd in P['training']['seeds']:
            if os.path.exists(os.path.join(R, 'results', 'models', f'{arm}_s{sd}.pt')):
                out.append(('new', f'{arm}_s{sd}'))
    return out


def load_model(R, R86, kind, name):
    import ecro86 as EC
    return EC.load_model(R86 if kind == 'frz' else R, name)


# ------------------------------------------------------------------------------------------------ exclusion
def excl(R, R86):
    import fx92
    fx92.inject(R)
    FX = fx92.FX
    mb, ml = fx92.S.P['training']['exclusion_margin_m']['box'], fx92.S.P['training']['exclusion_margin_m']['light']
    boxes = [FX['E4']['T0'], FX['E4']['T1']]
    lpos = np.array([l['pos'] for k in ('L4', 'LMIX4', 'LSPOT4') for l in FX['light_sets'][k]])
    out = {'rule': P['training']['exclusion'], 'E4_light_positions': lpos.tolist(), 'excluded': {}, 'kept': {}, 'min_distance_kept': {}}
    for sp in ('train', 'val', 'trainhq'):
        ex, kept, dbx, dlt = [], [], [], []
        for fn in sorted(glob.glob(os.path.join(R86, 'data', sp, 'g*.npz'))):
            z = np.load(fn)
            g = json.loads(str(z['geom'])); ls = json.loads(str(z['lights']))
            db = min(np.linalg.norm(np.array(g[b]['c']) - np.array(t[b]['c'])) for t in boxes for b in 'AB')
            dl = min(float(np.linalg.norm(lpos - np.array(l[0]['pos']), axis=1).min()) for l in ls)
            if db < mb or dl < ml:
                ex.append([os.path.basename(fn), float(db), float(dl)])
            else:
                kept.append(os.path.basename(fn)); dbx.append(db); dlt.append(dl)
        out['excluded'][sp] = ex; out['kept'][sp] = len(kept)
        out['min_distance_kept'][sp] = {'box_m': float(min(dbx)), 'light_m': float(min(dlt))}
    jd(out, os.path.join(R, 'results', 'exclusion.json'))
    print('EXCL', {k: len(v) for k, v in out['excluded'].items()}, out['kept'], flush=True)


def kept_files(R, R86, sp, limit=None):
    ex = {n for n, _, _ in jl(os.path.join(R, 'results', 'exclusion.json'))['excluded'].get(sp, [])}
    fns = [f for f in sorted(glob.glob(os.path.join(R86, 'data', sp, 'g*.npz'))) if os.path.basename(f) not in ex]
    return fns[:limit]


# ------------------------------------------------------------------------------------------------ component targets
def gen(R, R86, split, start, count):
    import fx92
    import pt86a as A
    PT, S = fx92.PT, fx92.S
    TR = S.P['training']
    start, count = int(start), int(count)
    seed = TR['splits'][split]['seed']
    spp = TR['splits'][split].get('target_spp', TR['target_spp']); off = TR['splits'][split].get('field_seed_offset', 0)
    if DEV:
        spp = 32 if split == 'train' else 64
    keep = {os.path.basename(f) for f in kept_files(R, R86, split)}
    out = os.path.join(R, 'data', split + '_f2'); os.makedirs(out, exist_ok=True)
    ctx = None; chk = []
    t0 = time.time()
    for idx in range(start, start + count):
        name = f'g{idx:05d}.npz'
        src = os.path.join(R86, 'data', split, name); fn = os.path.join(out, name)
        if not os.path.exists(src) or name not in keep:
            continue
        z = np.load(src)
        geom = json.loads(str(z['geom'])); lights = json.loads(str(z['lights']))
        if ctx is None:
            ctx = PT.Ctx(geom)
        else:
            ctx.set_geom(geom)
        d = {'f2': [], 'f2p': [], 'f2b': [], 'f2pb': []}
        row = {'file': name, 'rel_l1': [], 'flux': []}
        for li, l in enumerate(lights):
            lt = PT.Lights(l)
            sd = (seed % 1000) * 100000 + idx * 16 + li + off
            halves = [(spp, sd, z['f'])] if split == 'train' else [(spp // 2, sd, z['f']), (spp // 2, sd + 8, z['fb'])]
            for h, (sp_, s_, ref) in enumerate(halves):
                t = A.field_src(ctx, A.Src(lt, 'true', ()), sp_, s_, b=128, split=True)
                tot = sum(t.values()); r = ref[li].astype(np.float32)
                row['rel_l1'].append(float(np.abs(tot - r).sum() / max(r.sum(), 1e-30))); row['flux'].append(float(tot.sum() / max(r.sum(), 1e-30)))
                sfx = '' if h == 0 else 'b'
                d['f2' + sfx].append((t['e2_p'] + t['e2_o']).astype(np.float16)); d['f2p' + sfx].append(t['e2_p'].astype(np.float16))
        np.savez_compressed(fn + '.tmp.npz', **{k: np.stack(v) for k, v in d.items() if v})
        os.replace(fn + '.tmp.npz', fn)
        chk.append({'file': name, 'max_rel_l1': max(row['rel_l1']), 'max_flux_dev': max(abs(x - 1) for x in row['flux'])})
        if (idx - start) % 16 == 0:
            print(split, idx, round(time.time() - t0, 1), 's', json.dumps(chk[-1]), flush=True)
    jd({'split': split, 'start': start, 'count': count, 'spp': spp, 'rows': chk, 'seconds': time.time() - t0}, os.path.join(out, f'check_{start:05d}.json'))
    print('GEN_DONE', split, start, len(chk), round(time.time() - t0, 1), 's', max((c['max_rel_l1'] for c in chk), default=None), flush=True)


class It:
    """WL86 train86.Item + component targets."""

    def __init__(self, R, sp, fn):
        import train86 as TR
        self.it = TR.Item(fn)
        z2 = np.load(os.path.join(R, 'data', sp + '_f2', os.path.basename(fn)))
        self.f2 = z2['f2']; self.f2b = z2['f2b'] if 'f2b' in z2.files else None
        self.f2p = z2['f2p']; self.f2pb = z2['f2pb'] if 'f2pb' in z2.files else None

    def gs(self):
        return self.it.gs()

    def t2(self, dev):
        import torch
        t = torch.as_tensor(self.f2.astype(np.float32), device=dev)
        if self.f2b is not None:
            t = 0.5 * (t + torch.as_tensor(self.f2b.astype(np.float32), device=dev))
        return t

    def t2p(self, dev):
        import torch
        t = torch.as_tensor(self.f2p.astype(np.float32), device=dev)
        if self.f2pb is not None:
            t = 0.5 * (t + torch.as_tensor(self.f2pb.astype(np.float32), device=dev))
        return t


def load_items(R, R86, sp, limit=None):
    from concurrent.futures import ThreadPoolExecutor
    fns = [f for f in kept_files(R, R86, sp) if os.path.exists(os.path.join(R, 'data', sp + '_f2', os.path.basename(f)))][:limit]
    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(lambda f: It(R, sp, f), fns))


def sd_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def terms(model, gs, s):
    f = model.factors(gs)
    return model.apply(f, gs, s, use_global=False), model.apply(f, gs, s, use_single=False)


def objective(arm, Bs, Bg, t, t2):
    den = t.mean(2, keepdim=True) + 0.05 * t.mean(dim=(1, 2, 3), keepdim=True)
    ls = (((Bs + Bg - t) / den) ** 2).mean()
    lc2 = (((Bs - t2) / den) ** 2).mean(); lc3 = (((Bg - (t - t2)) / den) ** 2).mean()
    return (ls if arm == 'sum' else ls + lc2 + lc3), ls, lc2, lc3


def train(R, R86, seed, arm):
    import torch
    import fx92
    import train86 as TR
    import model86 as M
    CFG = fx92.S.P['training']
    dev = TR.dev
    seed = int(seed)
    tag = f'{arm}_s{seed}'
    out = os.path.join(R, 'results', 'models'); os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    items = load_items(R, R86, 'train', 48 if DEV else None); val = load_items(R, R86, 'val', 8 if DEV else None)
    steps = 300 if DEV else CFG['steps']
    print('loaded', len(items), len(val), round(time.time() - t0, 1), 's', flush=True)
    model = M.ECRO().to(dev)
    init_hash = sd_hash(model)
    opt_ = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    wu = CFG['warmup_steps']

    def lr(i):
        if i < wu:
            return CFG['lr'] * (i + 1) / wu
        return CFG['lr_final'] + 0.5 * (CFG['lr'] - CFG['lr_final']) * (1 + math.cos(math.pi * (i - wu) / max(steps - wu, 1)))
    rng = np.random.default_rng(seed)
    order = []; log = []; run = []; gn = []; nonfinite = 0
    t0 = time.time()
    for i in range(steps):
        if not order:
            order = rng.permutation(len(items)).tolist()
        it = items[order.pop()]
        for g in opt_.param_groups:
            g['lr'] = lr(i)
        gs = it.gs(); s = torch.as_tensor(it.it.s, device=dev); t = it.it.target(); t2 = it.t2(dev)
        Bs, Bg = terms(model, gs, s)
        loss, ls, lc2, lc3 = objective(arm, Bs, Bg, t, t2)
        opt_.zero_grad(set_to_none=True); loss.backward()
        g_ = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        if not (math.isfinite(float(loss)) and math.isfinite(g_)):
            nonfinite += 1
        opt_.step()
        run.append((float(loss), float(ls), float(lc2), float(lc3))); gn.append(g_)
        if (i + 1) % 500 == 0 or i + 1 == steps:
            m = np.mean(run, 0)
            rec = {'step': i + 1, 'objective': float(m[0]), 'L_sum': float(m[1]), 'L_single': float(m[2]), 'L_global': float(m[3]),
                   'grad_norm_mean': float(np.mean(gn)), 'grad_norm_max': float(np.max(gn)), 'clipped_fraction': float(np.mean(np.array(gn) > 10.0)), 's': round(time.time() - t0, 1)}
            if (i + 1) % 4000 == 0 or i + 1 == steps:
                rec['val_rel_l1'] = TR.evaluate(model, [v.it for v in val[:24]])['rel_l1']
            log.append(rec); run = []; gn = []
            print(json.dumps(rec), flush=True)
    res = {'seed': seed, 'arm': arm, 'steps': steps, 'params': sum(p.numel() for p in model.parameters()), 'init_sha256': init_hash,
           'train_s': time.time() - t0, 'items': len(items), 'val_items': len(val), 'nonfinite_steps': nonfinite,
           'gpu_mem_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20, 'log': log}
    torch.save(model.state_dict(), os.path.join(out, f'{tag}.pt'))
    jd(res, os.path.join(out, f'{tag}.json'))
    print('FINAL', tag, json.dumps({k: res[k] for k in ('train_s', 'nonfinite_steps', 'init_sha256')}), log[-1], flush=True)


# ------------------------------------------------------------------------------------------------ field-level evaluation
def fieldeval(R, R86):
    import torch
    import fx92
    import ecro86 as EC
    dev = EC.dev
    sets = {'val': load_items(R, R86, 'val', 8 if DEV else None), 'trainhq': load_items(R, R86, 'trainhq', 4 if DEV else None)}
    res = {'models': {}}
    for kind, name in models(R):
        mdl = load_model(R, R86, kind, name)
        o = {}
        with torch.no_grad():
            for sp, its in sets.items():
                a = {k: 0.0 for k in ('w_n', 'w_d', 's_n', 's_d', 'g_n', 'g_d', 'sq', 'sqd', 'p_n', 'p_d', 'sp_n', 'sp_d', 'L_sum', 'L_single', 'L_global')}
                for it in its:
                    gs = it.gs(); s = torch.as_tensor(it.it.s, device=dev)
                    f = mdl.factors(gs)
                    Bs = mdl.apply(f, gs, s, use_global=False); Bg = mdl.apply(f, gs, s, use_single=False); B = Bs + Bg
                    t = it.it.target(); t2 = it.t2(dev); t3 = t - t2
                    a['w_n'] += float((B - t).abs().sum()); a['w_d'] += float(t.sum())
                    a['s_n'] += float((Bs - t2).abs().sum()); a['s_d'] += float(t2.sum())
                    a['g_n'] += float((Bg - t3).abs().sum()); a['g_d'] += float(t3.sum())
                    fa = torch.as_tensor(it.it.f.astype(np.float32), device=dev); fb = torch.as_tensor(it.it.fb.astype(np.float32), device=dev)
                    a['sq'] += float(((B - fa) * (B - fb)).sum()); a['sqd'] += float((fa * fb).sum())
                    pm = torch.as_tensor(it.it.tab['pmat'] == 1, device=dev).view(1, -1, 1, 1)
                    a['p_n'] += float(((B - t).abs() * pm).sum()); a['p_d'] += float((t * pm).sum())
                    a['sp_n'] += float(((Bs - t2).abs() * pm).sum()); a['sp_d'] += float((t2 * pm).sum())
                    _, ls, l2, l3 = objective('own', Bs, Bg, t, t2)
                    a['L_sum'] += float(ls); a['L_single'] += float(l2); a['L_global'] += float(l3)
                n = max(len(its), 1)
                o[sp] = {'whole_rel_l1': a['w_n'] / a['w_d'], 'whole_rel_rms_debiased': math.sqrt(max(a['sq'], 0) / max(a['sqd'], 1e-30)),
                         'single_vs_L2_rel_l1': a['s_n'] / a['s_d'], 'global_vs_L3_rel_l1': a['g_n'] / a['g_d'],
                         'panel_patches_whole_rel_l1': a['p_n'] / max(a['p_d'], 1e-30), 'panel_patches_single_vs_L2_rel_l1': a['sp_n'] / max(a['sp_d'], 1e-30),
                         'L_sum': a['L_sum'] / n, 'L_single': a['L_single'] / n, 'L_global': a['L_global'] / n, 'n': len(its)}
            # contracts on one validation geometry
            it = sets['val'][0]; gs = it.gs(); s = torch.as_tensor(it.it.s, device=dev)
            f = mdl.factors(gs)
            a_, b_ = s[:1], s[1:2]
            ab = mdl.apply(f, gs, a_ + b_); sc = float(ab.abs().max())
            o['contracts'] = {'superposition': float((ab - mdl.apply(f, gs, a_) - mdl.apply(f, gs, b_)).abs().max() / sc),
                              'scaling': float((mdl.apply(f, gs, 3 * a_) - 3 * mdl.apply(f, gs, a_)).abs().max() / sc),
                              'zero': float(mdl.apply(f, gs, torch.zeros_like(a_)).abs().max()), 'min': float(mdl.apply(f, gs, s).min()),
                              'single_plus_global_minus_B': float((mdl.apply(f, gs, s, use_global=False) + mdl.apply(f, gs, s, use_single=False) - mdl.apply(f, gs, s)).abs().max() / float(mdl.apply(f, gs, s).abs().max())),
                              'replay': float((mdl(gs, s) - mdl(gs, s)).abs().max())}
        res['models'][name] = o
        print('fieldeval', name, json.dumps({sp: {k: round(v, 4) for k, v in o[sp].items() if k != 'n'} for sp in sets}), flush=True)
    ctl = {}
    for sd in P['training']['seeds']:
        if f'sum_s{sd}' in res['models']:
            ctl[sd] = res['models'][f'sum_s{sd}']['val']['whole_rel_l1'] / res['models'][f'ecro_s{sd}']['val']['whole_rel_l1']
    res['T_CTL'] = {'sum_over_frozen_val_rel_l1': ctl, 'PASS': bool(ctl and max(ctl.values()) <= 1.15)}
    inits = {}
    for sd in P['training']['seeds']:
        h = [jl(os.path.join(R, 'results', 'models', f'{a}_s{sd}.json'))['init_sha256'] for a in ('sum', 'own') if os.path.exists(os.path.join(R, 'results', 'models', f'{a}_s{sd}.json'))]
        inits[sd] = h
    res['T_INIT'] = {'hashes': inits, 'PASS': bool(inits and all(len(v) == 2 and v[0] == v[1] for v in inits.values()))}
    c = [m['contracts'] for m in res['models'].values()]
    res['T_LINEAR'] = {'PASS': bool(all(x['superposition'] <= 1e-4 and x['scaling'] <= 1e-4 and x['zero'] == 0.0 and x['min'] >= 0 for x in c))}
    res['T_OWNERSHIP_CODE'] = {'max': max(x['single_plus_global_minus_B'] for x in c), 'PASS': bool(max(x['single_plus_global_minus_B'] for x in c) <= 1e-5)}
    res['T_REPLAY'] = {'max': max(x['replay'] for x in c), 'PASS': bool(max(x['replay'] for x in c) == 0.0)}
    jd(res, os.path.join(R, 'results', 'fieldeval.json'))
    print('FIELDEVAL', json.dumps({k: res[k]['PASS'] for k in ('T_CTL', 'T_INIT', 'T_LINEAR', 'T_OWNERSHIP_CODE', 'T_REPLAY')}), json.dumps(ctl), flush=True)


# ------------------------------------------------------------------------------------------------ tables per configuration
def mtables(R, R86, tag):
    import torch
    import fx92
    fx92.inject(R)
    import ecro86 as EC
    import model86 as M
    import diag92 as DG
    PT, FX, S, E87 = fx92.PT, fx92.FX, fx92.S, fx92.E87
    geom, ls, e, cfg = fx92.cfg_ls(tag)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    st = EC.build_state(ctx)
    s_np = ctx.source(lt)
    s = torch.as_tensor(s_np, device=EC.dev).view(1, ctx.N, S.NB, 3)
    gs = M.GeoState(ctx.G, st.hit, st.dbin, EC.dev)
    stale = tag in ('E2_T1', 'E4_T1')
    if stale:
        st0 = EC.build_state(PT.Ctx(FX[e]['T0']))
        gs0 = M.GeoState(st0.G, st0.hit, st0.dbin, EC.dev)
    out = {}
    with torch.no_grad():
        for kind, name in models(R):
            mdl = load_model(R, R86, kind, name)
            f = mdl.factors(gs)
            Bs = mdl.apply(f, gs, s, use_global=False)[0]; Bg = mdl.apply(f, gs, s, use_single=False)[0]
            u = torch.einsum('nmcr,lnmc->lncr', f['c'], s)
            out[f'B_{name}'] = (Bs + Bg).cpu().numpy(); out[f'S_{name}'] = Bs.cpu().numpy(); out[f'G_{name}'] = Bg.cpu().numpy()
            out[f'SP_{name}'] = DG.single_panel(mdl, f, gs, s)[0].cpu().numpy()
            out[f'Bq_{name}'] = torch.einsum('ndcr,lncr->lndc', f['a'], u)[0].cpu().numpy()
            if stale:
                out[f'STALE_{name}'] = mdl(gs0, s)[0].cpu().numpy()
    if tag in ('E4_T0', 'E4_T1'):
        import quad87 as Q
        import xport87 as X
        F = np.load(os.path.join(R, 'results', 'fields', f'F_{tag}.npz'))
        out['ORACLE'] = (0.5 * (F['fa'].astype(np.float64) + F['fb'])).astype(np.float32)
        quad = Q.Quad(X.realization_us(86001), E87['M_ray'], E87['seed_ray'])
        Bx, _, _, _, _ = X.build(ctx, lt, quad, E87['P'], E87['Mq'], E87['seed_q'], st=st)
        out['X_EXP'] = Bx.astype(np.float32)
    np.savez_compressed(os.path.join(R, 'results', 'tables', f'I_{tag}.npz'), **{k: v.astype(np.float32) for k, v in out.items()})
    print('mtables', tag, [n for _, n in models(R)], 'stale' if stale else '', flush=True)


# ------------------------------------------------------------------------------------------------ learned local operator
def local(R, R86):
    import torch
    import fx92
    fx92.inject(R)
    import ecro86 as EC
    import model86 as M
    import diag92 as DG
    PT, FX, S = fx92.PT, fx92.FX, fx92.S
    _, Mq = DG.local_matrices()
    zb = DG.flux_w()
    res = {}
    for tag in ('E4_T1', 'E2_T1'):
        geom, ls, e, cfg = fx92.cfg_ls(tag)
        G = S.Geom(geom)
        D = np.load(os.path.join(R, 'results', 'fields', f'D_{tag}.npz'))
        gs = M.GeoState(G, D['hit'], D['dbin'], EC.dev)
        I = np.load(os.path.join(R, 'results', 'tables', f'I_{tag}.npz'))
        q = D['q'].astype(np.float64)
        res[tag] = {}
        for kind, name in models(R):
            mdl = load_model(R, R86, kind, name)
            with torch.no_grad():
                f = mdl.factors(gs)
                Ml = torch.einsum('ndcr,nmcr->nmdc', f['a'], f['c']).double().cpu().numpy()
            o = {}
            for m in (0, 1, 2):
                sel = np.nonzero(G.PMAT == m)[0]
                rows = [DG.mat_metrics(Ml[n, :, :, c], Mq[m, :, :, c], zb) for n in sel for c in range(3)]
                o[str(m)] = {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}
                Bq = I[f'Bq_{name}'].astype(np.float64)[sel]; qq = q[sel]
                o[str(m)]['Bq_vs_q_rel_l1'] = float(np.abs(Bq - qq).sum() / max(qq.sum(), 1e-30)); o[str(m)]['Bq_vs_q_flux'] = float(Bq.sum() / max(qq.sum(), 1e-30))
            res[tag][name] = o
            print('local', tag, name, {m: {k: round(v, 3) for k, v in x.items() if k in ('fro', 'peak', 'mass', 'Bq_vs_q_rel_l1', 'Bq_vs_q_flux')} for m, x in o.items()}, flush=True)
    jd(res, os.path.join(R, 'results', 'local92a.json'))


# ------------------------------------------------------------------------------------------------ runtime
def timing(R, R86):
    import torch
    import fx92
    fx92.inject(R)
    import ecro86 as EC
    import model86 as M
    PT, FX, S = fx92.PT, fx92.FX, fx92.S
    ex = FX['E4']
    lt = PT.Lights(FX['light_sets']['L4'])
    ctx = PT.Ctx(ex['T0'])
    sync = EC.sync

    def tm(fn, n=10):
        fn(); fn()
        ts = []
        for _ in range(n):
            sync(); t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
        return float(np.median(ts))
    st0 = EC.build_state(ctx)
    res = {'gpu': os.popen('nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader').read().strip(), 'foreign': [x for x in os.popen('nvidia-smi --query-compute-apps=pid --format=csv,noheader').read().split() if x != str(os.getpid())]}
    res['state_full_s'] = tm(lambda: EC.build_state(ctx))
    ctx.set_geom(ex['T1'])
    res['state_selective_s'] = tm(lambda: EC.update_state(ctx, st0, ex['moved']))
    st = EC.build_state(ctx)
    res['source_encoding_s'] = tm(lambda: ctx.source(lt))
    s = torch.as_tensor(ctx.source(lt), device=EC.dev).view(1, ctx.N, S.NB, 3)
    res['models'] = {}
    for kind, name in models(R):
        mdl = load_model(R, R86, kind, name)
        with torch.no_grad():
            gs = M.GeoState(ctx.G, st.hit, st.dbin, EC.dev)
            f = mdl.factors(gs)
            res['models'][name] = {'factors_s': tm(lambda: mdl.factors(M.GeoState(ctx.G, st.hit, st.dbin, EC.dev))), 'apply_s': tm(lambda: mdl.apply(f, gs, s)),
                                   'transfer_s': tm(lambda: PT.fl(mdl.apply(f, gs, s)[0].cpu().numpy())),
                                   'checkpoint_bytes': os.path.getsize(os.path.join(R86 if kind == 'frz' else R, 'results', 'models', f'{name}.pt'))}
    res['torch_peak_mib'] = torch.cuda.max_memory_allocated() / 2 ** 20
    jd(res, os.path.join(R, 'results', 'timing92a.json'))
    print('TIMING', json.dumps({k: (round(v * 1000, 2) if k.endswith('_s') else v) for k, v in res.items() if k != 'models'}),
          json.dumps({n: {k: round(v * 1000, 2) for k, v in x.items() if k.endswith('_s')} for n, x in res['models'].items()}), flush=True)


# ------------------------------------------------------------------------------------------------ conditional ladder
def ladder(R, R86):
    import fx92
    fx92.inject(R)
    import analyze92 as AN
    PT, FX = fx92.PT, fx92.FX
    tag = 'E4_T1'
    geom, ls, e, cfg = fx92.cfg_ls(tag)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    cam = FX['camera']; res_ = cam['res']
    I = np.load(os.path.join(R, 'results', 'tables', f'I_{tag}.npz'))
    ref = AN.ref_of(R, R86, tag); M = AN.mask_of(R, R86, tag)
    h = ref['h'].mean(0); tot = ref['tot'].mean(0)
    arms = {'own_s8601': PT.fl(I['B_own_s8601']), 'sum_s8601': PT.fl(I['B_sum_s8601'])}
    spp_t = [16, 64] if DEV else [16, 64, 256, 1024, 4096]
    spp_p = [64, 256] if DEV else [64, 256, 1024, 4096, 16384]
    rows = []
    ctx.render(lt, 'ER', cam, res_, 4, 1, 'cam', plain=False, b=4)
    for B in arms.values():
        ctx.render(lt, 'ER', cam, res_, 4, 1, 'ecro', B=B, b=4)
    for arm in list(arms) + ['PT']:
        for spp in (spp_p if arm == 'PT' else spp_t):
            t0 = time.perf_counter()
            if arm == 'PT':
                acc, T_ = ctx.render(lt, 'ER', cam, res_, spp, 92900 + spp, 'cam', b=min(32, spp))
            else:
                acc, T_ = ctx.render(lt, 'ER', cam, res_, spp, 92900 + spp, 'ecro', B=arms[arm], b=min(32, spp))
            w = time.perf_counter() - t0
            o = PT.pack(acc, res_, spp)
            x = o['h'] if arm == 'PT' else o['hl']
            r = {'arm': arm, 'spp': spp, 'wall_s': w, 'gpu_ms': T_['gpu_ms']}
            for k in ('stationary_all', 'C'):
                m = M[k]
                r['E_H_' + k] = float(np.abs(x[m] - h[m]).sum() / tot[m].sum())
            rows.append(r)
            print('ladder', json.dumps(r), flush=True)
    jd({'rows': rows, 'note': 'E_H of the higher-order channel against REF-ER (PT: its own h channel at that spp); costs: GPU kernel time excludes compilation'},
       os.path.join(R, 'results', 'ladder92a.json'))


if __name__ == '__main__':
    import fx92
    fx92.dr.set_flag(fx92.dr.JitFlag.KernelHistory, True)                     # GPU kernel time in pt86 render timings (ladder)
    cmd, R, R86 = sys.argv[1], sys.argv[2], sys.argv[3]
    a = [x for x in sys.argv[4:] if not x.startswith('--')]
    for d in ('fields', 'tables', 'renders', 'masks', 'models', 'lists'):
        os.makedirs(os.path.join(R, 'results', d), exist_ok=True)
    {'excl': excl, 'gen': gen, 'train': train, 'fieldeval': fieldeval, 'mtables': mtables, 'local': local, 'timing': timing, 'ladder': ladder}[cmd](R, R86, *a)
    os._exit(0)
