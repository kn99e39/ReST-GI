"""WL102 post-hoc DIAGNOSTIC (not evidence, nothing adopted): why the LR3 training collapsed to a zero output.

  python diag102.py <data_root> <out_dir> <variant> [steps=1500]
variants (each changes exactly one input, everything else = train102 / WL100 recipe):
  D0  original (reproduces the collapse)
  D1  log-area input floored at the ECRO-room scale: area -> max(area, A_MIN) for the generator input only
      (A_MIN = 1/16 of a 25 cm cell; cell-boundary slivers reach 6.8e-9 m^2, log(area)+3 = -15.8, vs 0.23 for every ECRO-room patch)
  D2  lights whose source maximum exceeds the ECRO-room scale are dropped from the items (s.max() > S_MAX; S_MAX = the light
      intensity / 0.4 m^2, the closest light-to-wall distance of the ECRO-room light distribution)
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import train102 as T

M = T.M
data, out, var = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
out.mkdir(parents=True, exist_ok=True)
R = T.TR
torch.manual_seed(R['seed']); np.random.seed(R['seed'])
gs, rows = T.geostate(data)
if var == 'D1':
    gs.area = torch.clamp(gs.area, min=0.0625 / 16)
train = [T.Item(f) for f in sorted((data / 'train').glob('g*.npz'))]
val = [T.Item(f) for f in sorted((data / 'val').glob('g*.npz'))]
dropped = 0
if var == 'D2':
    smax = M.S.FX['light_intensity'] / 0.4 ** 2
    for it in train:
        keep = it.s.reshape(len(it.s), -1).max(1) <= smax
        dropped += int((~keep).sum())
        it.s, it.f2, it.fh = it.s[keep], it.f2[keep], it.fh[keep]
    train = [it for it in train if len(it.s)]
model = M.ECRO().to(T.DEV); model.mat_states = rows
opt = torch.optim.Adam(model.parameters(), lr=R['lr']); rng = np.random.default_rng(R['seed']); order = []; log = []; run = []
t0 = time.perf_counter()
for step in range(steps):
    if not order:
        order = rng.permutation(len(train)).tolist()
    it = train[order.pop()]
    lr = R['lr'] * (step + 1) / R['warmup'] if step < R['warmup'] else \
        R['lr_final'] + .5 * (R['lr'] - R['lr_final']) * (1 + math.cos(math.pi * (step - R['warmup']) / max(R['steps'] - R['warmup'], 1)))
    for g in opt.param_groups:
        g['lr'] = lr
    s, t2, th = it.tensors(); f = model.factors(gs)
    a = model.apply(f, gs, s, use_global=False); b = model.apply(f, gs, s, use_single=False)
    t = t2 + th; den = t.mean(2, keepdim=True) + .05 * t.mean(dim=(1, 2, 3), keepdim=True)
    losses = [(((a + b - t) / den) ** 2).mean(), (((a - t2) / den) ** 2).mean(), (((b - th) / den) ** 2).mean()]
    loss = sum(losses)
    opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), R['clip']))
    opt.step(); run.append([float(x.detach()) for x in losses] + [gn, float((a + b).detach().mean() / t.mean())])
    if (step + 1) % 100 == 0:
        m = np.mean(run, 0).tolist(); run = []
        row = {'step': step + 1, 'losses': m[:3], 'grad_norm_preclip': m[3], 'pred_over_target_mean': m[4], 'lr': lr}
        log.append(row); print(json.dumps(row), flush=True)
res = {'variant': var, 'steps': steps, 'dropped_lights': dropped, 'log': log, 'val': T.evaluate(model, gs, val),
       'seconds': time.perf_counter() - t0, 'note': 'post-hoc diagnostic, not evidence; same schedule as the 24k-step run (lr warm-up 500)'}
json.dump(res, open(out / f'diag_{var}.json', 'w'), indent=1)
print('DIAG', var, res['val'], flush=True)
