"""WL94 Stage C (conditional paired smoke): candidate A (original fixed material-object assignment) vs B (controlled material-diverse
assignment), same ECRO architecture, same init (frozen own_s<seed>), same geometries / lights / target sampling / schedule / order.

  python c94.py select  <R> <R86> <R92A>                                training / validation file sets + material maps + T_LEAK
  python c94.py gen     <R> <R86> <split> <assign> <start> <count>      targets (train: A | B, 256 spp; val: orig | ck, 2 x 512 spp)
  python c94.py train   <R> <R86> <R92A> <seed> <A|B>                    2,000-step OWN fine-tune from own_s<seed>
  python c94.py popeval <R> <R86> <R92A> <spec> <otag>                  V_orig / V_ck evaluation of one model (spec own:<seed> | cand:<tag>)
"""
import glob, hashlib, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mat94 as MT  # noqa: E402

DEV = MT.DEV
CP = MT.P['stage_C']
N_TRAIN, N_VAL = (8, 4) if DEV else (96, 24)
STEPS = 60 if DEV else 2000
STEPS = int(os.environ.get('C94_PROBE_STEPS', STEPS))         # timing probe only (never set in the evidence chain)
TRAIN_SPP, VAL_SPP = (16, 16) if DEV else (256, 512)


def name_idx(name):
    return int(os.path.basename(name)[1:6])


# ------------------------------------------------------------------------------------------------ selection
def ck2_ok(geom, lights):
    for b in 'AB':
        if np.linalg.norm(np.array(geom[b]['c']) - np.array(MT.SC['geom'][b]['c'])) < 0.15:
            return False
    lp = np.array([l['pos'] for l in MT.SC['lights']])
    for l in lights:
        if np.linalg.norm(lp - np.array(l[0]['pos']), axis=1).min() < 0.3:
            return False
    return True


def select(R, R86, R92A):
    import scene86 as S
    ex = MT.jl(os.path.join(R92A, 'results', 'exclusion.json'))['excluded']
    out = {'rule': CP['training_population'], 'splits': {}}
    leak = {'train_ids_6_7': 0, 'train_nonuniform_rects': 0, 'ck2_rule_violations': 0}
    for sp, n in (('train', N_TRAIN), ('val', N_VAL)):
        exn = {e[0] for e in ex.get(sp, [])}
        sel, rej = [], 0
        for fn in sorted(glob.glob(os.path.join(R86, 'data', sp, 'g*.npz'))):
            nm = os.path.basename(fn)
            if nm in exn:
                continue
            z = np.load(fn)
            g = json.loads(str(z['geom'])); ls = json.loads(str(z['lights']))
            if not ck2_ok(g, ls):
                rej += 1
                continue
            G = S.Geom(g)
            rec = {'file': nm, 'idx': name_idx(nm)}
            if sp == 'train':
                pmB = MT.train_pmat_B(G, rec['idx'])
                rec['digest_A'] = MT.pm_digest(MT.default_pmat(G)); rec['digest_B'] = MT.pm_digest(pmB)
                leak['train_ids_6_7'] += int(np.isin(pmB, [6, 7]).sum())
                leak['train_nonuniform_rects'] += sum(int(len(np.unique(pmB[MT.rect_patches(G, r)])) > 1) for r in range(len(G.RMAT)))
                rec['B_albedos'] = {str(r): MT.ALB.get(int(pmB[MT.rect_patches(G, r)][0])) for r in MT.LAMBERT_RECTS}
            else:
                rec['digest_orig'] = MT.pm_digest(MT.default_pmat(G)); rec['digest_ck'] = MT.pm_digest(MT.val_pmat_ck(G, rec['idx']))
            sel.append(rec)
            if len(sel) == n:
                break
        out['splits'][sp] = {'files': sel, 'rejected_by_ck2_rule_before_quota': rej}
    tr = {r['file'] for r in out['splits']['train']['files']}; va = {r['file'] for r in out['splits']['val']['files']}
    leak['train_val_overlap_names'] = len(tr & va)
    out['T_LEAK'] = {'pass': leak['train_ids_6_7'] == 0 and leak['train_nonuniform_rects'] == 0 and len(out['splits']['train']['files']) == N_TRAIN
                     and len(out['splits']['val']['files']) == N_VAL, **leak,
                     'note': 'train and val are different WL86 splits (seeds 86100 / 86200); every selected file passed the WL92A exclusion and the CK2 rule'}
    hist = {}
    for r in out['splits']['train']['files']:
        for a in r['B_albedos'].values():
            hist[str(a)] = hist.get(str(a), 0) + 1
    out['B_albedo_histogram_rects'] = hist
    MT.jd(out, os.path.join(R, 'results', 'select94.json'))
    print('SELECT', 'PASS' if out['T_LEAK']['pass'] else 'FAIL', len(tr), len(va), flush=True)


def sel_files(R, sp):
    return MT.jl(os.path.join(R, 'results', 'select94.json'))['splits'][sp]['files']


def pmat_of(G, assign, idx):
    if assign in ('A', 'orig'):
        return MT.default_pmat(G)
    return MT.train_pmat_B(G, idx) if assign == 'B' else MT.val_pmat_ck(G, idx)


# ------------------------------------------------------------------------------------------------ targets
def gen(R, R86, split, assign, start, count):
    import fx92
    fx92.inject(R)
    PT, S, A = MT.install()
    files = sel_files(R, split)[int(start):int(start) + int(count)]
    out = os.path.join(R, 'data', f'{split}_{assign}'); os.makedirs(out, exist_ok=True)
    ctx = None; t0 = time.time(); rows = []
    for rec in files:
        fn = os.path.join(out, rec['file'])
        if os.path.exists(fn):
            continue
        z = np.load(os.path.join(R86, 'data', split, rec['file']))
        geom = json.loads(str(z['geom'])); lights = json.loads(str(z['lights']))
        if ctx is None:
            ctx = PT.Ctx(geom)
        else:
            ctx.set_geom(geom)
        pm = pmat_of(ctx.G, assign, rec['idx'])
        MT.set_pmat(PT, ctx, pm)
        d = {'f': [], 'f2': [], 'fb': [], 'f2b': []}
        for li, l in enumerate(lights):
            lt = PT.Lights(l)
            base = (940000000 if split == 'train' else 941000000) + 16 * rec['idx'] + li
            halves = [('', TRAIN_SPP, base)] if split == 'train' else [('', VAL_SPP, base), ('b', VAL_SPP, base + 8)]
            for sfx, spp, sd in halves:
                t = A.field_src(ctx, A.Src(lt, 'true', ()), spp, sd, b=128, split=True)
                d['f' + sfx].append(sum(t.values()).astype(np.float32)); d['f2' + sfx].append((t['e2_p'] + t['e2_o']).astype(np.float32))
        np.savez_compressed(fn + '.tmp.npz', pmat=pm.astype(np.int16), digest=MT.pm_digest(pm), **{k: np.stack(v) for k, v in d.items() if v})
        os.replace(fn + '.tmp.npz', fn)
        rows.append({'file': rec['file'], 'digest': MT.pm_digest(pm)})
    MT.jd({'split': split, 'assign': assign, 'start': int(start), 'count': int(count), 'rows': rows, 'seconds': time.time() - t0,
           'patch_sha256': MT._INST['patch_sha']}, os.path.join(out, f'check_{int(start):05d}.json'))
    print('GEN_DONE', split, assign, start, len(rows), round(time.time() - t0, 1), 's', flush=True)


class It:
    def __init__(self, R, R86, split, assign, rec):
        import train86 as TR
        self.it = TR.Item(os.path.join(R86, 'data', split, rec['file']))
        z = np.load(os.path.join(R, 'data', f'{split}_{assign}', rec['file']))
        self.f = z['f']; self.f2 = z['f2']
        self.fb = z['fb'] if 'fb' in z.files else None; self.f2b = z['f2b'] if 'f2b' in z.files else None
        self.pm = z['pmat'].astype(np.int64)
        exp = rec.get('digest_' + assign)
        assert str(z['digest']) == MT.pm_digest(self.pm) and (exp is None or exp == str(z['digest'])), 'T_ITEM_PMAT'
        self.it.tab['pmat'] = self.pm
        self.name = rec['file']

    def gs(self):
        return self.it.gs()

    def tgt(self, dev, key):
        import torch
        a = getattr(self, key); b = getattr(self, key + 'b')
        t = torch.as_tensor(a, device=dev)
        return t if b is None else 0.5 * (t + torch.as_tensor(b, device=dev))


def load_items(R, R86, split, assign, limit=None):
    from concurrent.futures import ThreadPoolExecutor
    files = sel_files(R, split)[:limit]
    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(lambda r: It(R, R86, split, assign, r), files))


def sd_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def objective(Bs, Bg, t, t2):
    den = t.mean(2, keepdim=True) + 0.05 * t.mean(dim=(1, 2, 3), keepdim=True)
    ls = (((Bs + Bg - t) / den) ** 2).mean()
    lc2 = (((Bs - t2) / den) ** 2).mean(); lc3 = (((Bg - (t - t2)) / den) ** 2).mean()
    return ls + lc2 + lc3, ls, lc2, lc3


def val_whole(model, items, dev):
    import torch
    num = den = 0.0
    with torch.no_grad():
        for it in items:
            gs = it.gs(); s = torch.as_tensor(it.it.s, device=dev)
            p = model(gs, s); t = it.tgt(dev, 'f')
            num += float((p - t).abs().sum()); den += float(t.sum())
    return num / den


def train(R, R86, R92A, seed, cand):
    import torch
    import model86 as M86
    dev = torch.device('cuda')
    seed = int(seed); tag = f'{cand}_s{seed}'
    out = os.path.join(R, 'results', 'models'); os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    items = load_items(R, R86, 'train', cand); val = load_items(R, R86, 'val', 'orig')
    for it in items[:4] + val[:2]:
        assert np.array_equal(it.gs().mat.cpu().numpy(), it.pm), 'T_ITEM_PMAT (model input map)'
    print('loaded', len(items), len(val), round(time.time() - t0, 1), 's', flush=True)
    model = M86.ECRO().to(dev)
    model.load_state_dict(torch.load(os.path.join(R92A, 'results', 'models', f'own_s{seed}.pt'), map_location=dev))
    MT.extend_buffer(model)
    init_hash = sd_hash(model)
    opt_ = torch.optim.Adam(model.parameters(), lr=3e-4)
    wu, lr0, lr1 = 200, 3e-4, 1e-5

    def lr(i):
        if i < wu:
            return lr0 * (i + 1) / wu
        return lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * (i - wu) / max(STEPS - wu, 1)))
    rng = np.random.default_rng(seed + 94)
    order = []; log = []; run = []; gn = []; nonfinite = 0; first = []; last = []
    v0 = val_whole(model, val, dev)
    t0 = time.time()
    for i in range(STEPS):
        if not order:
            order = rng.permutation(len(items)).tolist()
        it = items[order.pop()]
        for g in opt_.param_groups:
            g['lr'] = lr(i)
        gs = it.gs(); s = torch.as_tensor(it.it.s, device=dev); t = it.tgt(dev, 'f'); t2 = it.tgt(dev, 'f2')
        F = model.factors(gs)
        Bs = model.apply(F, gs, s, use_global=False); Bg = model.apply(F, gs, s, use_single=False)
        loss, ls, lc2, lc3 = objective(Bs, Bg, t, t2)
        opt_.zero_grad(set_to_none=True); loss.backward()
        g_ = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        if not (math.isfinite(float(loss)) and math.isfinite(g_)):
            nonfinite += 1
        opt_.step()
        rec = (float(loss), float(ls), float(lc2), float(lc3))
        run.append(rec); gn.append(g_)
        if i < 200:
            first.append(rec[1])
        if i >= STEPS - 200:
            last.append(rec[1])
        if (i + 1) % 100 == 0 or i + 1 == STEPS:
            m = np.mean(run, 0)
            r = {'step': i + 1, 'objective': float(m[0]), 'L_sum': float(m[1]), 'L_single': float(m[2]), 'L_global': float(m[3]),
                 'grad_norm_mean': float(np.mean(gn)), 'grad_norm_max': float(np.max(gn)), 's': round(time.time() - t0, 1)}
            if (i + 1) in (STEPS // 2, STEPS):
                r['val_orig_rel_l1'] = val_whole(model, val, dev)
            log.append(r); run = []; gn = []
            if not DEV:
                print(json.dumps({k: r[k] for k in ('step', 's')}), flush=True)
    res = {'tag': tag, 'seed': seed, 'candidate': cand, 'steps': STEPS, 'params': sum(p.numel() for p in model.parameters()),
           'init_sha256': init_hash, 'final_sha256': sd_hash(model), 'train_s': time.time() - t0, 'items': len(items), 'val_items': len(val),
           'nonfinite_steps': nonfinite, 'L_sum_first200': float(np.mean(first)), 'L_sum_last200': float(np.mean(last)),
           'val_orig_rel_l1_init': v0, 'val_orig_rel_l1_final': log[-1]['val_orig_rel_l1'],
           'gpu_mem_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20, 'log': log}
    torch.save(model.state_dict(), os.path.join(out, f'{tag}.pt'))
    MT.jd(res, os.path.join(out, f'{tag}.json'))
    print('FINAL', tag, round(res['train_s'], 1), 's', 'nonfinite', nonfinite, flush=True)


# ------------------------------------------------------------------------------------------------ population evaluation
def exitance_w(S):
    z = (np.arange(S.NB) // S.NPHI + 0.5) / S.NZ
    return (z * S.DOMEGA).astype(np.float32)


def popeval(R, R86, R92A, spec, otag):
    import torch
    import scene86 as S
    dev = torch.device('cuda')
    model = MT.load_spec(R, R92A, spec)
    w = torch.as_tensor(exitance_w(S), device=dev).view(1, 1, S.NB, 1)
    out = {}; summ = {}
    for assign in ('orig', 'ck'):
        items = load_items(R, R86, 'val', assign)
        num = {'B': 0.0, 'S': 0.0, 'G': 0.0}; den = {'B': 0.0, 'S': 0.0, 'G': 0.0}
        ex = {k: [] for k in ('mB', 'mS', 'mG', 'rB', 'rS', 'rG', 'rBa', 'rBb', 'rSa', 'rSb', 'rGa', 'rGb')}
        cls = []
        with torch.no_grad():
            for it in items:
                gs = it.gs(); s = torch.as_tensor(it.it.s, device=dev)
                F = model.factors(gs)
                Bs = model.apply(F, gs, s, use_global=False); Bg = model.apply(F, gs, s, use_single=False)
                fa, fb = torch.as_tensor(it.f, device=dev), torch.as_tensor(it.fb, device=dev)
                f2a, f2b = torch.as_tensor(it.f2, device=dev), torch.as_tensor(it.f2b, device=dev)
                t, t2 = 0.5 * (fa + fb), 0.5 * (f2a + f2b)
                for k, p, q in (('B', Bs + Bg, t), ('S', Bs, t2), ('G', Bg, t - t2)):
                    num[k] += float((p - q).abs().sum()); den[k] += float(q.abs().sum())
                E = lambda x: (x * w).sum(dim=(0, 2, 3)).cpu().numpy()            # per patch exitance, summed over the 4 lightings
                for k, x in (('mB', Bs + Bg), ('mS', Bs), ('mG', Bg), ('rB', t), ('rS', t2), ('rG', t - t2), ('rBa', fa), ('rBb', fb),
                             ('rSa', f2a), ('rSb', f2b), ('rGa', fa - f2a), ('rGb', fb - f2b)):
                    ex[k].append(E(x))
                c = np.full(it.pm.shape[0], 2)                                   # 0 Lambert, 1 S1 panel, 2 other (floor B, sphere S2)
                c[np.isin(it.pm, [3, 4, 5] + MT.NEW_IDS)] = 0; c[it.pm == 1] = 1
                cls.append(c)
        summ[assign] = {k: num[k] / den[k] for k in num}
        for k, v in ex.items():
            out[f'{assign}|{k}'] = np.stack(v).astype(np.float64)
        out[f'{assign}|cls'] = np.stack(cls)
        out[f'{assign}|pmat'] = np.stack([it.pm for it in items])
    fn = os.path.join(R, 'results', 'pop', f'{otag}.npz'); os.makedirs(os.path.dirname(fn), exist_ok=True)
    np.savez_compressed(fn, **out)
    MT.jd({'spec': spec, 'rel_l1': summ}, fn.replace('.npz', '.json'))
    print('POPEVAL', otag, flush=True)


if __name__ == '__main__':
    a = [x for x in sys.argv[1:] if not x.startswith('--')]
    cmd = a[0]
    if cmd == 'select':
        select(*a[1:4])
    elif cmd == 'gen':
        gen(*a[1:7])
    elif cmd == 'train':
        train(*a[1:6])
    elif cmd == 'popeval':
        popeval(*a[1:6])
    os._exit(0)
