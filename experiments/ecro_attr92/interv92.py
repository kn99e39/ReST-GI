"""WL92 held-out condition E4 and the conditional learning-contract intervention (protocol sections e4 / intervention).

  python interv92.py e4select <R> <R86>                    geometry / visibility-only choice of E4, masks, training-exclusion list (no radiance)
  python interv92.py ref      <R> <R86> <tag> <ER|P> <seed> <spp> <outtag>
  python interv92.py ofield   <R> <R86> <tag>              oracle field of a configuration (two halves)
  python interv92.py gen      <R> <R86> <split> <start> <count>   component target f2 (exactly 2 events) for the stored WL86 data, same seeds / samples
  python interv92.py train    <R> <R86> <seed> <ctl|own>   WL86 training loop; ctl = WL86 objective, own = + component ownership
  python interv92.py mtables  <R> <R86> <tag>              tables and terms of every model (frozen, ctl, own) for a configuration
The intervention commands run only if the gate of analyze92.py admits the intervention.
"""
import glob, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEV = '--dev' in sys.argv


def jd(o, fn):
    json.dump(o, open(fn, 'w'), indent=1)


def e4select(R, R86):
    import fx92
    import masks86
    S, P92 = fx92.S, fx92.P92
    rule = P92['e4']['rule']
    rows = {}; chosen = None
    out = os.path.join(R, 'results', 'masks'); os.makedirs(out, exist_ok=True)
    for name in (['E3dev'] if DEV else P92['e4']['order']):
        fx92.inject(name=name)
        c = fx92.FX['E4']; lpos = fx92.FX['light_sets']['L4'][0]['pos']
        gok = bool(S.geom_ok(c['T0']) and S.geom_ok(c['T1']) and S.light_ok(lpos, c['T0']) and S.light_ok(lpos, c['T1']))
        m, aux = masks86.compute('E4')
        cnt = {k: int(v.sum()) for k, v in m.items()}
        ok = (gok and all(cnt[k] >= rule['min_pixels'] for k in rule['regions']) and cnt.get('C_mirror', 0) >= rule['min_pixels']) or DEV
        rows[name] = {'counts': cnt, 'geometry_ok': gok, 'admissible': bool(ok)}
        print('E4 candidate', name, json.dumps(cnt), ok, flush=True)
        if ok and chosen is None:
            chosen = name
            np.savez_compressed(os.path.join(out, 'E4.npz'), **m, **aux)
            for ls in ('LMIX4', 'LSPOT4'):
                ms, auxs = masks86.compute('E4', ls=ls)
                np.savez_compressed(os.path.join(out, f'E4_{ls}.npz'), **ms, **auxs)
            break
    excl = {}
    if chosen is not None:
        mg = fx92.S.P['training']['exclusion_margin_m']['box']
        tg = [fx92.FX['E4']['T0'], fx92.FX['E4']['T1']]
        for sp in ('train', 'val', 'trainhq'):
            near = []
            for fn in sorted(glob.glob(os.path.join(R86, 'data', sp, 'g*.npz'))):
                g = json.loads(str(np.load(fn)['geom']))
                d = min(np.linalg.norm(np.array(g[b]['c']) - np.array(t[b]['c'])) for t in tg for b in 'AB')
                if d < mg:
                    near.append([os.path.basename(fn), float(d)])
            excl[sp] = near
    jd({'chosen': chosen, 'candidates': rows, 'rule': rule, 'training_exclusion': excl, 'note': 'geometry / visibility only; no radiance of any E4 candidate exists'},
       os.path.join(R, 'results', 'e4_choice.json'))
    print('E4_CHOSEN', chosen, {k: len(v) for k, v in excl.items()}, flush=True)


def ref(R, R86, tag, kind, seed, spp, otag):
    import fx92
    fx92.inject(R)
    PT, FX = fx92.PT, fx92.FX
    geom, ls, e, cfg = fx92.cfg_ls(tag)
    ctx = PT.Ctx(geom)
    acc, T_ = ctx.render(PT.Lights(FX['light_sets'][ls]), kind, FX['camera'], FX['camera']['res'], int(spp), int(seed), 'cam', b=32)
    out = os.path.join(R, 'results', 'renders'); os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f'{otag}.npz'), **PT.pack(acc, FX['camera']['res'], int(spp)))
    jd({'tag': tag, 'ref': kind, 'seed': int(seed), 'spp': int(spp), 'timing': T_}, os.path.join(out, f'{otag}.json'))
    print(otag, json.dumps(T_), flush=True)


def ofield(R, R86, tag):
    import fx92
    fx92.inject(R)
    PT, FX = fx92.PT, fx92.FX
    geom, ls, e, cfg = fx92.cfg_ls(tag)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    o = fx92.P92['e4']['references']['oracle_field']
    spp = 64 if DEV else o['spp']
    fa, ca = ctx.field(lt, spp // 2, o['seed']); fb, cb = ctx.field(lt, spp // 2, o['seed'] + 500)
    np.savez_compressed(os.path.join(R, 'results', 'fields', f'F_{tag}.npz'), fa=fa, fb=fb, B=(0.5 * (fa + fb)).astype(np.float32))
    print('ofield', tag, flush=True)


def gen(R, R86, split, start, count):
    """f2 of the stored WL86 items (geometry and lights read from the stored files; the WL86 generator's seeds, sample counts and in-place geometry path)."""
    import fx92
    import pt86a as A
    PT, S = fx92.PT, fx92.S
    TR = S.P['training']
    start, count = int(start), int(count)
    seed = TR['splits'][split]['seed']
    spp = TR['splits'][split].get('target_spp', TR['target_spp']); off = TR['splits'][split].get('field_seed_offset', 0)
    if DEV:
        spp = 32
    out = os.path.join(R, 'data', split + '_f2'); os.makedirs(out, exist_ok=True)
    ctx = None
    chk = []
    t0 = time.time()
    for idx in range(start, start + count):
        src = os.path.join(R86, 'data', split, f'g{idx:05d}.npz')
        fn = os.path.join(out, f'g{idx:05d}.npz')
        if not os.path.exists(src) or os.path.exists(fn):
            continue
        z = np.load(src)
        geom = json.loads(str(z['geom'])); lights = json.loads(str(z['lights']))
        if ctx is None:
            ctx = PT.Ctx(geom)
        else:
            ctx.set_geom(geom)
        f2, f2b = [], []
        for li, l in enumerate(lights):
            lt = PT.Lights(l)
            sd = (seed % 1000) * 100000 + idx * 16 + li + off
            if split == 'train':
                t = A.field_src(ctx, A.Src(lt, 'true', ()), spp, sd, b=128, split=True)
                f2.append(t['e2_p'] + t['e2_o'])
                tot = sum(t.values())
                ref_ = z['f'][li].astype(np.float32)
            else:
                t = A.field_src(ctx, A.Src(lt, 'true', ()), spp // 2, sd, b=128, split=True)
                tb = A.field_src(ctx, A.Src(lt, 'true', ()), spp // 2, sd + 8, b=128, split=True)
                f2.append(t['e2_p'] + t['e2_o']); f2b.append(tb['e2_p'] + tb['e2_o'])
                tot = sum(t.values())
                ref_ = z['f'][li].astype(np.float32)
            if li == 0 and (idx - start) < 4:
                chk.append({'file': os.path.basename(src), 'flux_ratio': float(tot.sum() / max(ref_.sum(), 1e-30)), 'rel_l1': float(np.abs(tot - ref_).sum() / max(ref_.sum(), 1e-30))})
        d = {'f2': np.stack(f2).astype(np.float16)}
        if f2b:
            d['f2b'] = np.stack(f2b).astype(np.float16)
        np.savez_compressed(fn + '.tmp.npz', **d)
        os.replace(fn + '.tmp.npz', fn)
        if (idx - start) % 16 == 0:
            print(split, idx, round(time.time() - t0, 1), 's', flush=True)
    jd(chk, os.path.join(out, f'check_{start:05d}.json'))
    print('GEN_DONE', split, start, count, round(time.time() - t0, 1), 's', json.dumps(chk[:2]), flush=True)


def train(R, R86, seed, mode):
    """The WL86 loop (train86.main) with the objective switch. ctl: model86.loss_fn on the sum. own: + component ownership (protocol)."""
    import torch
    import fx92
    import train86 as TR
    import model86 as M
    S = fx92.S
    CFG = S.P['training']
    dev = TR.dev
    seed = int(seed)
    tag = f'{mode}_s{seed}'
    out = os.path.join(R, 'results', 'models'); os.makedirs(out, exist_ok=True)
    excl = json.load(open(os.path.join(R, 'results', 'e4_choice.json')))['training_exclusion']
    drop = lambda sp: {n for n, _ in excl.get(sp, [])}
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()

    def load(sp, limit=None):
        fns = [f for f in sorted(glob.glob(os.path.join(R86, 'data', sp, 'g*.npz'))) if os.path.basename(f) not in drop(sp)][:limit]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(16) as ex:
            items = list(ex.map(TR.Item, fns))
        for it, fn in zip(items, fns):
            f2 = os.path.join(R, 'data', sp + '_f2', os.path.basename(fn))
            it.f2 = it.f2b = None
            if os.path.exists(f2):
                z2 = np.load(f2)
                it.f2 = z2['f2']; it.f2b = z2['f2b'] if 'f2b' in z2.files else None
        return items
    items = load('train', 48 if DEV else None); val = load('val', 8 if DEV else None)
    if mode == 'own':
        assert all(it.f2 is not None for it in items), 'component targets missing'
    steps = 300 if DEV else CFG['steps']
    print('loaded', len(items), len(val), round(time.time() - t0, 1), 's', flush=True)
    model = M.ECRO().to(dev)
    opt_ = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
    wu = CFG['warmup_steps']

    def lr(i):
        if i < wu:
            return CFG['lr'] * (i + 1) / wu
        return CFG['lr_final'] + 0.5 * (CFG['lr'] - CFG['lr_final']) * (1 + math.cos(math.pi * (i - wu) / max(steps - wu, 1)))
    rng = np.random.default_rng(seed)
    order = []; log = []; run = []
    t0 = time.time()
    for i in range(steps):
        if not order:
            order = rng.permutation(len(items)).tolist()
        it = items[order.pop()]
        for g in opt_.param_groups:
            g['lr'] = lr(i)
        gs = it.gs(); s = torch.as_tensor(it.s, device=dev); t = it.target()
        if mode == 'ctl':
            loss = M.loss_fn(model(gs, s), t)
        else:
            f = model.factors(gs)
            Bs = model.apply(f, gs, s, use_global=False); Bg = model.apply(f, gs, s, use_single=False)
            t2 = torch.as_tensor(it.f2.astype(np.float32), device=dev); t3 = t - t2
            den = t.mean(2, keepdim=True) + 0.05 * t.mean(dim=(1, 2, 3), keepdim=True)
            loss = (((Bs + Bg - t) / den) ** 2).mean() + (((Bs - t2) / den) ** 2).mean() + (((Bg - t3) / den) ** 2).mean()
        opt_.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt_.step()
        run.append(float(loss))
        if (i + 1) % 500 == 0 or i + 1 == steps:
            rec = {'step': i + 1, 'loss': float(np.mean(run)), 's': round(time.time() - t0, 1)}
            if (i + 1) % 4000 == 0 or i + 1 == steps:
                rec['val'] = TR.evaluate(model, val[:24])['rel_l1']
            log.append(rec); run = []
            print(json.dumps(rec), flush=True)

    @torch.no_grad()
    def comp_eval(its):
        n2 = d2 = n3 = d3 = 0.0
        for it in its:
            if it.f2 is None:
                continue
            gs = it.gs(); s = torch.as_tensor(it.s, device=dev)
            f = model.factors(gs)
            Bs = model.apply(f, gs, s, use_global=False); Bg = model.apply(f, gs, s, use_single=False)
            t = it.target(); t2 = torch.as_tensor(it.f2.astype(np.float32), device=dev)
            if it.f2b is not None:
                t2 = 0.5 * (t2 + torch.as_tensor(it.f2b.astype(np.float32), device=dev))
            n2 += float((Bs - t2).abs().sum()); d2 += float(t2.sum()); n3 += float((Bg - (t - t2)).abs().sum()); d3 += float((t - t2).sum())
        return {'single_vs_L2_rel_l1': n2 / d2, 'global_vs_L3_rel_l1': n3 / d3} if d2 else None
    res = {'seed': seed, 'mode': mode, 'steps': steps, 'params': sum(p.numel() for p in model.parameters()), 'train_s': time.time() - t0, 'log': log,
           'items': len(items), 'dropped_train': len(drop('train')), 'gpu_mem_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20,
           'val': TR.evaluate(model, val), 'trainhq': TR.evaluate(model, load('trainhq', 4 if DEV else None)),
           'val_no_global': TR.evaluate(model, val, use_global=False), 'val_no_single': TR.evaluate(model, val, use_single=False), 'val_components': comp_eval(val)}
    torch.save(model.state_dict(), os.path.join(out, f'{tag}.pt'))
    jd(res, os.path.join(out, f'{tag}.json'))
    print('FINAL', tag, json.dumps({k: res[k] for k in ('val', 'trainhq', 'val_components', 'train_s')}), flush=True)


def models(R):
    import fx92
    ms = [('frz', sd) for sd in fx92.SEEDS]
    for fn in sorted(glob.glob(os.path.join(R, 'results', 'models', '*.pt'))):
        ms.append(('new', os.path.basename(fn)[:-3]))
    return ms


def mtables(R, R86, tag):
    import torch
    import fx92
    fx92.inject(R)
    import ecro86 as EC
    import model86 as M
    import diag92 as DG
    PT, FX, S = fx92.PT, fx92.FX, fx92.S
    geom, ls, e, cfg = fx92.cfg_ls(tag)
    ctx = PT.Ctx(geom); lt = PT.Lights(FX['light_sets'][ls])
    st = EC.build_state(ctx)
    s = torch.as_tensor(ctx.source(lt), device=EC.dev).view(1, ctx.N, S.NB, 3)
    gs = M.GeoState(ctx.G, st.hit, st.dbin, EC.dev)
    out = {}
    with torch.no_grad():
        for kind, name in models(R):
            mdl = EC.load_model(R86 if kind == 'frz' else R, name)
            f = mdl.factors(gs)
            Bs = mdl.apply(f, gs, s, use_global=False)[0]; Bg = mdl.apply(f, gs, s, use_single=False)[0]
            out[f'B_{name}'] = (Bs + Bg).cpu().numpy(); out[f'S_{name}'] = Bs.cpu().numpy(); out[f'G_{name}'] = Bg.cpu().numpy()
            out[f'SP_{name}'] = DG.single_panel(mdl, f, gs, s)[0].cpu().numpy()
    np.savez_compressed(os.path.join(R, 'results', 'tables', f'I_{tag}.npz'), **out)
    print('mtables', tag, [n for _, n in models(R)], flush=True)


if __name__ == '__main__':
    cmd, R, R86 = sys.argv[1], sys.argv[2], sys.argv[3]
    a = [x for x in sys.argv[4:] if not x.startswith('--')]
    for d in ('fields', 'tables', 'renders', 'masks', 'models', 'lists'):
        os.makedirs(os.path.join(R, 'results', d), exist_ok=True)
    {'e4select': e4select, 'ref': ref, 'ofield': ofield, 'gen': gen, 'train': train, 'mtables': mtables}[cmd](R, R86, *a)
    os._exit(0)
