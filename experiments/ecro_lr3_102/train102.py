"""WL102 ECRO training on the LR3 dataset: unchanged ECRO v0 network (model86.py, 735,664 parameters) and the WL100 objective and
schedule verbatim (WL92A OWN loss: total + single + global, target-patch-mean denominator; Adam 1e-3 -> 1e-5 cosine, warm-up 500,
clip 10, 24,000 steps, final step, one item = 4 lights per step). Only the material BUFFER is replaced (one encoder row per patch,
WL100 precedent) and the geometry is fixed (one GeoState for every item).

  python train102.py <data_root> <out_dir> [steps]
"""
import hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ecro86'))
import model86 as M  # noqa: E402

torch.set_num_threads(4)
DEV = torch.device('cuda')
TR = {'seed': 10201, 'steps': 24000, 'lr': 1e-3, 'lr_final': 1e-5, 'warmup': 500, 'clip': 10.0}


def geostate(root):
    z = np.load(root / 'geometry.npz')
    g = type('G', (), {})()
    g.N = len(z['pc']); g.PC, g.PN, g.PT, g.PAREA, g.PMAT = z['pc'], z['pn'], z['pt'], z['parea'], z['pmat']
    return M.GeoState(g, z['hit'], z['dbin'], DEV), torch.as_tensor(z['rows'], device=DEV)


class Item:
    def __init__(self, path):
        z = np.load(path)
        self.s, self.f2, self.fh = z['s'], z['f2'], z['fh']
        if 'f2b' in z.files:
            self.f2 = .5 * (self.f2 + z['f2b']); self.fh = .5 * (self.fh + z['fhb'])

    def tensors(self):
        return [torch.as_tensor(x, device=DEV) for x in (self.s, self.f2, self.fh)]


@torch.no_grad()
def evaluate(model, gs, items):
    errors = np.zeros(3); truth = np.zeros(3)
    f = model.factors(gs)
    for it in items:
        s, t2, th = it.tensors()
        a = model.apply(f, gs, s, use_global=False); b = model.apply(f, gs, s, use_single=False)
        for i, (p, t) in enumerate([(a + b, t2 + th), (a, t2), (b, th)]):
            errors[i] += float((p - t).abs().sum()); truth[i] += float(t.sum())
    return dict(zip(['whole_rel_l1', 'single_rel_l1', 'global_rel_l1'], (errors / np.maximum(truth, 1e-20)).tolist()))


def main(root, out, steps=None):
    root, out = Path(root), Path(out); out.mkdir(parents=True, exist_ok=True)
    steps = int(steps or TR['steps'])
    torch.manual_seed(TR['seed']); np.random.seed(TR['seed'])
    gs, rows = geostate(root)
    train = [Item(f) for f in sorted((root / 'train').glob('g*.npz'))]
    val = [Item(f) for f in sorted((root / 'val').glob('g*.npz'))]
    model = M.ECRO().to(DEV); model.mat_states = rows
    count = sum(x.numel() for x in model.parameters()); assert count == 735664
    opt = torch.optim.Adam(model.parameters(), lr=TR['lr']); rng = np.random.default_rng(TR['seed']); order = []; log = []; running = []
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter(); print('TRAIN_START', count, len(train), len(val), gs.N, flush=True)
    for step in range(steps):
        if not order:
            order = rng.permutation(len(train)).tolist()
        item = train[order.pop()]
        lr = TR['lr'] * (step + 1) / TR['warmup'] if step < TR['warmup'] else \
            TR['lr_final'] + .5 * (TR['lr'] - TR['lr_final']) * (1 + math.cos(math.pi * (step - TR['warmup']) / max(steps - TR['warmup'], 1)))
        for gr in opt.param_groups:
            gr['lr'] = lr
        s, t2, th = item.tensors(); f = model.factors(gs)
        a = model.apply(f, gs, s, use_global=False); b = model.apply(f, gs, s, use_single=False)
        t = t2 + th; den = t.mean(2, keepdim=True) + .05 * t.mean(dim=(1, 2, 3), keepdim=True)
        losses = [(((a + b - t) / den) ** 2).mean(), (((a - t2) / den) ** 2).mean(), (((b - th) / den) ** 2).mean()]
        loss = sum(losses); assert torch.isfinite(loss), 'STOP_NONFINITE_LOSS'
        opt.zero_grad(set_to_none=True); loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), TR['clip'])
        assert torch.isfinite(norm), 'STOP_NONFINITE_GRADIENT'
        opt.step(); running.append([float(x.detach()) for x in losses])
        if (step + 1) % 500 == 0 or step + 1 == steps:
            row = {'step': step + 1, 'losses': np.mean(running, axis=0).tolist(), 'seconds': time.perf_counter() - t0}; running = []
            if (step + 1) % 4000 == 0 or step + 1 == steps:
                row['val'] = evaluate(model, gs, val)
            log.append(row); print(json.dumps(row), flush=True)
    torch.cuda.synchronize(); train_s = time.perf_counter() - t0
    ck = out / 'model.pt'; torch.save(model.state_dict(), ck)
    res = {'seed': TR['seed'], 'steps': steps, 'params': count, 'train_s': train_s, 'train_items': len(train), 'val_items': len(val),
           'N': gs.N, 'log': log, 'validation': evaluate(model, gs, val), 'recipe': TR, 'objective': 'WL92A OWN (1,1,1)',
           'architecture_changed': False, 'model86_sha256': hashlib.sha256((HERE.parent / 'ecro86' / 'model86.py').read_bytes()).hexdigest(),
           'checkpoint_sha256': hashlib.sha256(ck.read_bytes()).hexdigest(), 'torch': torch.__version__,
           'gpu': torch.cuda.get_device_name(0), 'peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20}
    json.dump(res, open(out / 'training.json', 'w'), indent=1)
    print('TRAIN_COMPLETE', train_s, res['validation'], flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
