"""WL86 ECRO v0 training (response supervision; no test configuration, no image, no source identity).

  python train86.py <root> <seed> [--split=train] [--steps=N] [--tag=name]
Writes <root>/results/models/<tag>.pt (state dict) and <tag>.json (log, train-subset / validation field errors).
"""
import glob, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scene86 as S  # noqa: E402
import model86 as M  # noqa: E402
TR = S.P['training']
dev = torch.device('cuda')


class Item:
    def __init__(self, fn):
        z = np.load(fn)
        self.geom = json.loads(str(z['geom']))
        self.hit = z['hit']; self.dbin = z['dbin']; self.s = z['s']; self.f = z['f']
        self.fb = z['fb'] if 'fb' in z.files else None
        self.tab = {k: z[k] for k in ('pc', 'pn', 'pt', 'parea', 'pmat')}

    def gs(self):
        g = type('G', (), {})()
        g.N = self.tab['pc'].shape[0]
        g.PC, g.PN, g.PT, g.PAREA, g.PMAT = (self.tab[k] for k in ('pc', 'pn', 'pt', 'parea', 'pmat'))
        return M.GeoState(g, self.hit, self.dbin, dev)

    def target(self):
        f = torch.as_tensor(self.f.astype(np.float32), device=dev)
        if self.fb is not None:
            f = 0.5 * (f + torch.as_tensor(self.fb.astype(np.float32), device=dev))
        return f


def load(root, split, limit=None):
    from concurrent.futures import ThreadPoolExecutor
    fns = sorted(glob.glob(os.path.join(root, 'data', split, 'g*.npz')))[:limit]
    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(Item, fns))


@torch.no_grad()
def evaluate(model, items, **kw):
    num = den = 0.0; per = []; nf_num = 0.0; sq = sqd = 0.0
    for it in items:
        p = model(it.gs(), torch.as_tensor(it.s, device=dev), **kw); t = it.target()
        e = (p - t).abs().sum(dim=(1, 2, 3)); d = t.sum(dim=(1, 2, 3))
        num += float(e.sum()); den += float(d.sum()); per += (e / d).tolist()
        if it.fb is not None:
            fa = torch.as_tensor(it.f.astype(np.float32), device=dev); fb = torch.as_tensor(it.fb.astype(np.float32), device=dev)
            nf_num += 0.5 * float((fa - fb).abs().sum())
            sq += float(((p - fa) * (p - fb)).sum()); sqd += float((fa * fb).sum())        # target noise cancels in expectation
    return {'rel_l1': num / den, 'per_lighting_median': float(np.median(per)), 'per_lighting_q90': float(np.quantile(per, 0.9)),
            'target_noise_rel_l1': nf_num / den if nf_num else None, 'rel_rms_debiased': math.sqrt(max(sq, 0.0) / sqd) if sqd else None, 'n': len(per)}


def main():
    root, seed = sys.argv[1], int(sys.argv[2])
    opt = {a.split('=')[0]: a.split('=')[1] for a in sys.argv[3:] if '=' in a}
    split = opt.get('--split', 'train'); steps = int(opt.get('--steps', TR['steps'])); tag = opt.get('--tag', f'ecro_s{seed}')
    out = os.path.join(root, 'results', 'models'); os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    items = load(root, split)
    val = load(root, 'val') if split == 'train' else load(root, 'dev')[-8:]
    if split != 'train':
        items = items[:-8]
    print('loaded', len(items), len(val), round(time.time() - t0, 1), 's', flush=True)
    model = M.ECRO().to(dev)
    opt_ = torch.optim.Adam(model.parameters(), lr=TR['lr'])
    wu = TR['warmup_steps']

    def lr(i):
        if i < wu:
            return TR['lr'] * (i + 1) / wu
        return TR['lr_final'] + 0.5 * (TR['lr'] - TR['lr_final']) * (1 + math.cos(math.pi * (i - wu) / max(steps - wu, 1)))
    rng = np.random.default_rng(seed)
    order = []
    log = []; run = []
    t0 = time.time()
    for i in range(steps):
        if not order:
            order = rng.permutation(len(items)).tolist()
        it = items[order.pop()]
        for g in opt_.param_groups:
            g['lr'] = lr(i)
        pred = model(it.gs(), torch.as_tensor(it.s, device=dev))
        loss = M.loss_fn(pred, it.target())
        opt_.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt_.step()
        run.append(float(loss))
        if (i + 1) % 500 == 0 or i + 1 == steps:
            rec = {'step': i + 1, 'loss': float(np.mean(run)), 's': round(time.time() - t0, 1)}
            if (i + 1) % 4000 == 0 or i + 1 == steps:
                rec['val'] = evaluate(model, val[:24])['rel_l1']
            log.append(rec); run = []
            print(json.dumps(rec), flush=True)
    res = {'seed': seed, 'steps': steps, 'split': split, 'params': sum(p.numel() for p in model.parameters()), 'train_s': time.time() - t0,
           'log': log, 'gpu_mem_peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20,
           'val': evaluate(model, val), 'train_subset': evaluate(model, items[:len(val)]), 'trainhq': evaluate(model, load(root, 'trainhq')) if split == 'train' else None,
           'val_no_global': evaluate(model, val, use_global=False), 'val_no_single': evaluate(model, val, use_single=False)}
    torch.save(model.state_dict(), os.path.join(out, f'{tag}.pt'))
    json.dump(res, open(os.path.join(out, f'{tag}.json'), 'w'), indent=1)
    print('FINAL', json.dumps({k: res[k] for k in ('val', 'trainhq', 'train_subset', 'val_no_global', 'val_no_single', 'params', 'train_s')}), flush=True)


if __name__ == '__main__':
    main()
