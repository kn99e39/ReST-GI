"""WL114 Stage C training: the unchanged WL113 recipe (train113: OWN loss, Adam 1e-3 -> 1e-5 cosine, warm-up 500, clip 10, 24,000
steps, final step; model86 ECRO v0 for coarse arms, model113 ECRO-S4 dense for dense arms) on D114 with D6 or full-depth targets.

  python train114.py ROOT ARM [STEPS]      ARM in A1r, B1, A2, B2, OB1 (protocol fd114.json stage_C.arms)
The ONLY difference between a D6 and a full arm of the same seed is the global target key (c_fh / s_fh vs c_fhF / s_fhF).
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'ecro_sss113')); sys.path.insert(0, str(HERE.parent / 'ecro86'))
sys.path.insert(0, str(HERE.parent / 'ecro_material100'))
import common114 as C  # noqa: E402
import train113 as TR  # noqa: E402
import model86 as M  # noqa: E402
import model113 as M3  # noqa: E402
import common100 as C100  # noqa: E402

DEV = TR.DEV
ARMS = {'A1r': ('dense', 'D6', 11330001), 'B1': ('dense', 'F', 11330001), 'A2': ('dense', 'D6', 11330002),
        'B2': ('dense', 'F', 11330002), 'OB1': ('coarse', 'F', 11330001)}


class Item(TR.Item):
    def __init__(self, path, refined, depth, val=False):
        z = np.load(path)
        self.path = path
        self.g = {k: z[k] for k in ('pc', 'pn', 'pt', 'parea', 'pmat', 'hit', 'dbin', 's', 'desc', 'support')}
        gk = 'fh' if depth == 'D6' else 'fhF'
        def avg(k):
            a = z[k].astype(np.float32)
            return 0.5 * (a + z[k + 'b'].astype(np.float32)) if val else a
        self.c = (avg('c_f2'), avg(f'c_{gk}'), z['c_m'] & (z['c_mb'] if val else True))
        self.s = (avg('s_f2'), avg(f's_{gk}'), z['s_m'] & (z['s_mb'] if val else True)) if refined else None


def noise_floor(items, depth):
    gk = 'fh' if depth == 'D6' else 'fhF'
    e = np.zeros(2); t = np.zeros(2)
    for it in items:
        z = np.load(it.path)
        a = z['c_f2'] + z[f'c_{gk}']; b = z['c_f2b'] + z[f'c_{gk}b']; m = (z['c_m'] & z['c_mb'])[..., None]
        e[0] += float((np.abs(a - b) * m).sum()) / 2; t[0] += float((0.5 * (a + b) * m).sum())
        a = z['s_f2'].astype(np.float32) + z[f's_{gk}']; b = z['s_f2b'].astype(np.float32) + z[f's_{gk}b']; m = (z['s_m'] & z['s_mb'])[..., None]
        e[1] += float((np.abs(a - b) * m).sum()) / 2; t[1] += float((0.5 * (a + b) * m).sum())
    return {'coarse_half_noise_rel_l1': e[0] / max(t[0], 1e-20), 'refined_half_noise_rel_l1': e[1] / max(t[1], 1e-20)}


def main(root, arm, steps=None):
    root = Path(root); kind, depth, seed = ARMS[arm]
    P = dict(TR.P, seed=seed)
    steps = int(steps or C.dv(P['steps'], 200))
    torch.manual_seed(P['seed']); np.random.seed(P['seed'])
    dd = root / 'stageC' / 'data'
    refined = kind == 'dense'
    trf = [f for f in sorted((dd / 'train').glob('g*.npz')) if '.tmp' not in f.name]
    vaf = [f for f in sorted((dd / 'val').glob('g*.npz')) if '.tmp' not in f.name]
    trf, vaf = trf[:C.dv(192, 6)], vaf[:C.dv(16, 2)]
    assert len(trf) == C.dv(192, 6) and len(vaf) == C.dv(16, 2)
    train = [Item(f, refined, depth) for f in trf]
    val = [Item(f, refined, depth, val=True) for f in vaf]
    model = (M3.ECROL4() if refined else M.ECRO()).to(DEV)
    model.mat_states = torch.as_tensor(C100.materials('sss'), device=DEV)
    count = sum(x.numel() for x in model.parameters())
    assert count == (778464 if refined else 735664), count
    tag = 'CAND' if refined else 'O_U'
    out = root / 'stageC' / arm; out.mkdir(parents=True, exist_ok=True)
    init_sha = __import__('hashlib').sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
    opt = torch.optim.Adam(model.parameters(), lr=P['lr']); rng = np.random.default_rng(P['seed']); order = []; log = []; run = []
    torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    print('TRAIN_START', arm, kind, depth, seed, count, len(train), flush=True)
    for step in range(steps):
        if not order:
            order = rng.permutation(len(train)).tolist()
        it = train[order.pop()]
        lr = P['lr'] * (step + 1) / P['warmup'] if step < P['warmup'] else \
            P['lr_final'] + .5 * (P['lr'] - P['lr_final']) * (1 + math.cos(math.pi * (step - P['warmup']) / max(steps - P['warmup'], 1)))
        for g in opt.param_groups:
            g['lr'] = lr
        gs, s, c, sb = it.tensors()
        a, b, sa, sbb = TR.forward(model, tag, gs, s)
        losses = TR.own(a, b, *c)
        if refined and sa is not None:
            losses += TR.own(sa, sbb, *sb)
        loss = sum(losses); assert torch.isfinite(loss), 'STOP_NONFINITE_LOSS'
        opt.zero_grad(set_to_none=True); loss.backward()
        nrm = torch.nn.utils.clip_grad_norm_(model.parameters(), P['clip']); assert torch.isfinite(nrm), 'STOP_NONFINITE_GRADIENT'
        opt.step(); run.append([float(x.detach()) for x in losses])
        if (step + 1) % 500 == 0 or step + 1 == steps:
            row = {'step': step + 1, 'losses': np.mean(run, 0).tolist(), 'seconds': time.perf_counter() - t0}; run = []
            if (step + 1) % 4000 == 0 or step + 1 == steps:
                row['val'] = TR.evaluate(model, tag, val)
            log.append(row); print(json.dumps(row), flush=True)
    torch.cuda.synchronize(); train_s = time.perf_counter() - t0
    ck = out / 'model.pt'; torch.save(model.state_dict(), ck)
    C.jd({'arm': arm, 'kind': kind, 'depth': depth, 'seed': seed, 'steps': steps, 'params': count, 'init_param_sha256': init_sha,
          'train_s': train_s, 'peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20, 'log': log, 'validation': TR.evaluate(model, tag, val),
          'target_noise': noise_floor(val, depth), 'checkpoint_sha256': C.sha(ck), 'recipe': P,
          'model86_sha256': C.sha(HERE.parent / 'ecro86/model86.py'), 'model113_sha256': C.sha(HERE.parent / 'ecro_sss113/model113.py'),
          'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}, out / 'training.json')
    print('TRAIN_COMPLETE', arm, train_s, flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
    sys.stdout.flush(); os._exit(0)
