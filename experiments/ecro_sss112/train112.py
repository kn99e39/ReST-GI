"""WL112 Stage C training: matched arms on identical D112 data, seed, schedule and objective.

  python train112.py ROOT ARM [STEPS]      ARM in O_U (coarse U), O_G (coarse G), S2 (ECRO-S2, 2x2 measure from the Stage B decision)
Masked WL92A OWN objective (whole + single + global, 1/1/1); final step is the delivered checkpoint (no selection).
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'ecro86'))
import common112 as C  # noqa: E402
import model86 as M  # noqa: E402
import model112 as M2  # noqa: E402
sys.path.insert(0, str(HERE.parent / 'ecro_material100'))
import common100 as C100  # noqa: E402

torch.set_num_threads(4)
DEV = torch.device('cuda')


def target_keys(arm, dec):
    if arm == 'O_U':
        return 'Uc', False
    if arm == 'O_G':
        return 'Gc', False
    return ('Us' if dec['outcome'] == 'SPATIAL_U' else 'Gs'), True


class Item:
    def __init__(self, path, key, half=False):
        z = np.load(path)
        self.geo = {k: z[k] for k in ('pc', 'pn', 'pt', 'parea', 'pmat', 'hit', 'dbin', 's')}
        f2, fh, m = z[f'{key}_f2'], z[f'{key}_fh'], z[f'{key}_m']
        if half and f'{key}_f2b' in z.files:
            f2 = 0.5 * (f2.astype(np.float32) + z[f'{key}_f2b'].astype(np.float32))
            fh = 0.5 * (fh.astype(np.float32) + z[f'{key}_fhb'].astype(np.float32))
            m = m & z[f'{key}_mb']
        self.f2, self.fh, self.m = f2, fh, m

    def tensors(self):
        z = self.geo
        g = type('G', (), {})(); g.N = len(z['pc'])
        g.PC, g.PN, g.PT, g.PAREA, g.PMAT = z['pc'], z['pn'], z['pt'], z['parea'], z['pmat']
        gs = M.GeoState(g, z['hit'], z['dbin'], DEV)
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), device=DEV).to(dt)
        return gs, t(z['s']), t(self.f2), t(self.fh), t(self.m, torch.float32)


def own_losses(a, b, t2, th, m):
    """Masked OWN: a single, b global predictions; t2 / th targets; m mask (no channel axis)."""
    t = t2 + th
    mm = m.unsqueeze(-1)
    red = tuple(range(2, t.dim() - 1))                                  # (sub, bins) or (bins)
    pm = (t * mm).sum(red, keepdim=True) / mm.sum(red, keepdim=True).clamp(min=1)
    gm = (t * mm).sum(tuple(range(1, t.dim() - 1)), keepdim=True) / mm.sum(tuple(range(1, t.dim() - 1)), keepdim=True).clamp(min=1)
    den = pm + 0.05 * gm
    nrm = mm.expand_as(t).sum().clamp(min=1)
    f = lambda p, q: ((((p - q) / den) ** 2) * mm).sum() / nrm
    return [f(a + b, t), f(a, t2), f(b, th)]


@torch.no_grad()
def evaluate(model, items):
    err = np.zeros(3); tru = np.zeros(3)
    for it in items:
        gs, s, t2, th, m = it.tensors(); f = model.factors(gs)
        a = model.apply(f, gs, s, use_global=False); b = model.apply(f, gs, s, use_single=False)
        mm = m.unsqueeze(-1)
        for i, (p, q) in enumerate([(a + b, t2 + th), (a, t2), (b, th)]):
            err[i] += float(((p - q).abs() * mm).sum()); tru[i] += float((q * mm).sum())
    return dict(zip(['whole_rel_l1', 'single_rel_l1', 'global_rel_l1'], (err / np.maximum(tru, 1e-20)).tolist()))


def main(root, arm, steps=None):
    root = Path(root); p = C.P['stage_C']['training']; steps = int(steps or C.dv(p['steps'], 200))
    dec = json.loads((root / 'stageB' / 'decision.json').read_text())
    key, sub = target_keys(arm, dec)
    seed = p['seed']; torch.manual_seed(seed); np.random.seed(seed)
    dd = root / 'stageC' / 'data'
    train = [Item(f, key) for f in sorted((dd / 'train').glob('g*.npz')) if '.tmp' not in f.name]
    val = [Item(f, key, half=True) for f in sorted((dd / 'val').glob('g*.npz')) if '.tmp' not in f.name]
    assert len(train) == C.dv(C.P['stage_C']['splits']['train']['geometries'], 6) and len(val) == C.dv(C.P['stage_C']['splits']['val']['geometries'], 2)
    model = (M2.ECROS2() if sub else M.ECRO()).to(DEV)
    model.mat_states = torch.as_tensor(C100.materials('sss'), device=DEV)
    count = sum(x.numel() for x in model.parameters())
    out = root / 'stageC' / arm; out.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=p['lr']); rng = np.random.default_rng(seed); order = []; log = []; running = []
    t0 = time.perf_counter(); print('TRAIN_START', arm, key, count, len(train), flush=True)
    for step in range(steps):
        if not order:
            order = rng.permutation(len(train)).tolist()
        it = train[order.pop()]
        lr = p['lr'] * (step + 1) / p['warmup'] if step < p['warmup'] else \
            p['lr_final'] + .5 * (p['lr'] - p['lr_final']) * (1 + math.cos(math.pi * (step - p['warmup']) / max(steps - p['warmup'], 1)))
        for g in opt.param_groups:
            g['lr'] = lr
        gs, s, t2, th, m = it.tensors(); f = model.factors(gs)
        a = model.apply(f, gs, s, use_global=False); b = model.apply(f, gs, s, use_single=False)
        losses = own_losses(a, b, t2, th, m)
        loss = sum(losses); assert torch.isfinite(loss), 'STOP_NONFINITE_LOSS'
        opt.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), p['gradient_clip']); assert torch.isfinite(norm), 'STOP_NONFINITE_GRADIENT'
        opt.step(); running.append([float(x.detach()) for x in losses])
        if (step + 1) % 500 == 0 or step + 1 == steps:
            row = {'step': step + 1, 'losses': np.mean(running, axis=0).tolist(), 'seconds': time.perf_counter() - t0}; running = []
            if (step + 1) % 4000 == 0 or step + 1 == steps:
                row['val'] = evaluate(model, val)
            log.append(row); print(json.dumps(row), flush=True)
    torch.cuda.synchronize(); train_s = time.perf_counter() - t0
    ck = out / 'model.pt'; torch.save(model.state_dict(), ck)
    C.jd({'arm': arm, 'target': key, 'sub': sub, 'seed': seed, 'steps': steps, 'params': count, 'train_s': train_s,
          'train_items': len(train), 'val_items': len(val), 'log': log, 'validation': evaluate(model, val),
          'checkpoint_sha256': C.sha(ck), 'model86_sha256': C.sha(HERE.parent / 'ecro86/model86.py'), 'model112_sha256': C.sha(HERE / 'model112.py'),
          'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20},
         out / 'training.json')
    print('TRAIN_COMPLETE', arm, train_s, flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
    sys.stdout.flush(); os._exit(0)
