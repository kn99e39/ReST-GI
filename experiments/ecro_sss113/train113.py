"""WL113 Stage C training: matched O_U (model86, coarse U) and CAND (ECRO-L4 / -S4: coarse OWN + refined OWN).

  python train113.py ROOT ARM [STEPS]
Identical data files, seed, schedule; shared parameters initialised identically (constructed first under the same seed).
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'ecro86')); sys.path.insert(0, str(HERE.parent / 'ecro_material100'))
import common113 as C  # noqa: E402
import model86 as M  # noqa: E402
import model113 as M3  # noqa: E402
import common100 as C100  # noqa: E402

torch.set_num_threads(4)
DEV = torch.device('cuda')
P = {'seed': 11330001, 'steps': 24000, 'lr': 1e-3, 'lr_final': 1e-5, 'warmup': 500, 'clip': 10.0}


class Item:
    def __init__(self, path, refined, val=False):
        z = np.load(path)
        self.g = {k: z[k] for k in ('pc', 'pn', 'pt', 'parea', 'pmat', 'hit', 'dbin', 's', 'desc', 'support')}
        def avg(k):
            a = z[k].astype(np.float32)
            return 0.5 * (a + z[k + 'b'].astype(np.float32)) if val else a
        self.c = (avg('c_f2'), avg('c_fh'), z['c_m'] & (z['c_mb'] if val else True))
        self.s = (avg('s_f2'), avg('s_fh'), z['s_m'] & (z['s_mb'] if val else True)) if refined else None

    def tensors(self):
        z = self.g
        g = type('G', (), {})(); g.N = len(z['pc'])
        g.PC, g.PN, g.PT, g.PAREA, g.PMAT = z['pc'], z['pn'], z['pt'], z['parea'], z['pmat']
        gs = M.GeoState(g, z['hit'], z['dbin'], DEV)
        gs.sup = torch.as_tensor(np.flatnonzero(z['support']), device=DEV); gs.desc = torch.as_tensor(z['desc'], device=DEV)
        t = lambda a: torch.as_tensor(np.asarray(a), device=DEV).float()
        c = tuple(t(x) for x in self.c)
        s = tuple(t(x) for x in self.s) if self.s is not None else None
        return gs, t(z['s']), c, s


def own(a, b, t2, th, m):
    t = t2 + th; mm = m.unsqueeze(-1)
    red = tuple(range(2, t.dim() - 1))
    pm = (t * mm).sum(red, keepdim=True) / mm.sum(red, keepdim=True).clamp(min=1)
    gm = (t * mm).sum(tuple(range(1, t.dim() - 1)), keepdim=True) / mm.sum(tuple(range(1, t.dim() - 1)), keepdim=True).clamp(min=1)
    den = pm + 0.05 * gm; nrm = mm.expand_as(t).sum().clamp(min=1)
    f = lambda p, q: ((((p - q) / den) ** 2) * mm).sum() / nrm
    return [f(a + b, t), f(a, t2), f(b, th)]


def forward(model, arm, gs, s):
    f = model.factors(gs)
    if arm == 'O_U':
        return model.apply(f, gs, s, use_global=False), model.apply(f, gs, s, use_single=False), None, None
    Ba, Bsa = model.outputs(f, gs, s, use_global=False); Bb, Bsb = model.outputs(f, gs, s, use_single=False)
    return Ba, Bb, Bsa, Bsb


@torch.no_grad()
def evaluate(model, arm, items):
    err = np.zeros(4); tru = np.zeros(4)
    for it in items:
        gs, s, c, sb = it.tensors()
        a, b, sa, sbb = forward(model, arm, gs, s)
        mm = c[2].unsqueeze(-1)
        err[0] += float(((a + b - c[0] - c[1]).abs() * mm).sum()); tru[0] += float(((c[0] + c[1]) * mm).sum())
        if sa is not None:
            ms = sb[2].unsqueeze(-1)
            err[1] += float(((sa + sbb - sb[0] - sb[1]).abs() * ms).sum()); tru[1] += float(((sb[0] + sb[1]) * ms).sum())
            # coarse output of the supported patches vs refined targets: the information the refinement adds
            cs = (a + b)[:, gs.sup].unsqueeze(2)
            err[2] += float(((cs - sb[0] - sb[1]).abs() * ms).sum()); tru[2] += float(((sb[0] + sb[1]) * ms).sum())
    r = err / np.maximum(tru, 1e-20)
    return {'coarse_whole_rel_l1': float(r[0]), 'refined_whole_rel_l1': float(r[1]) if tru[1] else None,
            'coarse_output_vs_refined_targets_rel_l1': float(r[2]) if tru[2] else None}


def noise_floor(items):
    """Target half-noise (val halves) for the coarse and refined targets."""
    e = np.zeros(2); t = np.zeros(2)
    for it in items:
        z = np.load(it.path)
        a = z['c_f2'] + z['c_fh']; b = z['c_f2b'] + z['c_fhb']; m = (z['c_m'] & z['c_mb'])[..., None]
        e[0] += float((np.abs(a - b) * m).sum()) / 2; t[0] += float((0.5 * (a + b) * m).sum())
        if 's_f2' in z.files:
            a = z['s_f2'].astype(np.float32) + z['s_fh']; b = z['s_f2b'].astype(np.float32) + z['s_fhb']; m = (z['s_m'] & z['s_mb'])[..., None]
            e[1] += float((np.abs(a - b) * m).sum()) / 2; t[1] += float((0.5 * (a + b) * m).sum())
    return {'coarse_half_noise_rel_l1': e[0] / max(t[0], 1e-20), 'refined_half_noise_rel_l1': e[1] / max(t[1], 1e-20)}


def main(root, arm, steps=None):
    root = Path(root); steps = int(steps or C.dv(P['steps'], 200))
    torch.manual_seed(P['seed']); np.random.seed(P['seed'])
    dd = root / 'stageC' / 'data'
    refined = arm == 'CAND'
    trf = [f for f in sorted((dd / 'train').glob('g*.npz')) if '.tmp' not in f.name]
    vaf = [f for f in sorted((dd / 'val').glob('g*.npz')) if '.tmp' not in f.name]
    trf, vaf = trf[:C.dv(192, 6)], vaf[:C.dv(16, 2)]
    assert len(trf) == C.dv(192, 6) and len(vaf) == C.dv(16, 2)
    train = [Item(f, refined) for f in trf]
    val = [Item(f, refined, val=True) for f in vaf]
    for it, f in zip(val, vaf):
        it.path = f
    model = (M3.ECROL4() if refined else M.ECRO()).to(DEV)
    model.mat_states = torch.as_tensor(C100.materials('sss'), device=DEV)
    count = sum(x.numel() for x in model.parameters())
    out = root / 'stageC' / arm; out.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=P['lr']); rng = np.random.default_rng(P['seed']); order = []; log = []; run = []
    torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    print('TRAIN_START', arm, count, len(train), flush=True)
    for step in range(steps):
        if not order:
            order = rng.permutation(len(train)).tolist()
        it = train[order.pop()]
        lr = P['lr'] * (step + 1) / P['warmup'] if step < P['warmup'] else \
            P['lr_final'] + .5 * (P['lr'] - P['lr_final']) * (1 + math.cos(math.pi * (step - P['warmup']) / max(steps - P['warmup'], 1)))
        for g in opt.param_groups:
            g['lr'] = lr
        gs, s, c, sb = it.tensors()
        a, b, sa, sbb = forward(model, arm, gs, s)
        losses = own(a, b, *c)
        if refined and sa is not None:
            losses += own(sa, sbb, *sb)
        loss = sum(losses); assert torch.isfinite(loss), 'STOP_NONFINITE_LOSS'
        opt.zero_grad(set_to_none=True); loss.backward()
        nrm = torch.nn.utils.clip_grad_norm_(model.parameters(), P['clip']); assert torch.isfinite(nrm), 'STOP_NONFINITE_GRADIENT'
        opt.step(); run.append([float(x.detach()) for x in losses])
        if (step + 1) % 500 == 0 or step + 1 == steps:
            row = {'step': step + 1, 'losses': np.mean(run, 0).tolist(), 'seconds': time.perf_counter() - t0}; run = []
            if (step + 1) % 4000 == 0 or step + 1 == steps:
                row['val'] = evaluate(model, arm, val)
            log.append(row); print(json.dumps(row), flush=True)
    torch.cuda.synchronize(); train_s = time.perf_counter() - t0
    ck = out / 'model.pt'; torch.save(model.state_dict(), ck)
    C.jd({'arm': arm, 'seed': P['seed'], 'steps': steps, 'params': count, 'train_s': train_s, 'peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20,
          'log': log, 'validation': evaluate(model, arm, val), 'target_noise': noise_floor(val), 'checkpoint_sha256': C.sha(ck),
          'model86_sha256': C.sha(HERE.parent / 'ecro86/model86.py'), 'model113_sha256': C.sha(HERE / 'model113.py'),
          'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}, out / 'training.json')
    print('TRAIN_COMPLETE', arm, train_s, flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
    sys.stdout.flush(); os._exit(0)
