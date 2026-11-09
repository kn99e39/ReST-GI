"""WL82 per-material direct fits (protocol fitting section).

  python fit82.py <data82> <set> <arm> <mode> <objective> <steps> <init> <out> [--dev]
    arm: C0 | E | Einc | ER       mode: cert | unb       objective: metric | shape       init: generic | reinit
Fit data: fit82_<set>.npz (64 wi x 64 wo per material, importance weights use / q of the WL81 wo mixture).
Objectives (per incident-direction group g of a material, averaged over its groups, summed over materials):
  metric: sum_k w |f_cl - f_ref|_1 / sum_k w |f_ref|_1  +  sum_c |a_cl,c - a_ref,c| / sum_c a_ref,c          (energy-weighted relative L1 + albedo)
  shape:  sum_c a_ref,c TV_c / sum_c a_ref,c,  TV_c = sum_k w |f_cl,c / a_cl,c - f_ref,c / a_ref,c|          (normalised angular shape only)
  with a_c = sum_k w f_c (same samples). Adam, lr 0.03 -> 3e-4 cosine, 50 warm-up steps (WL81 direct-fit optimiser).
"""
import json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import closure82 as CL  # noqa: E402
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
import train80c as TR  # noqa: E402  (lr_at)

P82 = json.load(open(os.path.join(HERE, 'protocol', 'compiled82.json')))
SD = P82['seeds']
OPT = P82['fitting']['optimiser']
dev_t = torch.device('cuda')


def load_fit(path):
    z = np.load(path)
    T = {k: torch.from_numpy(z[k]).to(dev_t) for k in ['m', 'wi', 'wo', 'f']}
    T['w'] = torch.from_numpy(z['use'].astype(np.float32) / z['q']).to(dev_t)
    T['n_wi'], T['n_wo'], T['n_mat'] = int(z['n_wi']), int(z['n_wo']), int(z['n_mat'])
    return T


def group_terms(pred, f, w, n_wo):
    """Per-group metric and shape terms (pred, f [L,3], w [L]) -> dict of [G] tensors."""
    G = pred.shape[0] // n_wo
    P = pred.view(G, n_wo, 3); Fr = f.view(G, n_wo, 3); W = w.view(G, n_wo, 1)
    a_cl = (P * W).sum(1); a_ref = (Fr * W).sum(1)
    den = a_ref.sum(1).clamp_min(1e-12)
    e1 = ((P - Fr).abs() * W).sum((1, 2)) / den
    e2 = (a_cl - a_ref).abs().sum(1) / den
    pc = torch.where(a_cl[:, None] > 1e-12, P / a_cl.clamp_min(1e-12)[:, None], torch.zeros_like(P))
    tv = (W * (pc - Fr / a_ref.clamp_min(1e-12)[:, None]).abs()).sum(1)
    es = (a_ref * tv).sum(1) / den
    return {'metric': e1 + e2, 'shape': es, 'e1': e1, 'e2': e2}


def fit(arm, mode, objective, z0, T, steps, log_every=500):
    z = torch.nn.Parameter(z0.clone())
    opt = torch.optim.Adam([z], lr=OPT['lr'])
    nm, nwi, nwo = T['n_mat'], T['n_wi'], T['n_wo']
    mid = T['m'].long()
    log = []
    t0 = time.time()
    for step in range(steps + 1):
        if step < steps:
            for pg in opt.param_groups:
                pg['lr'] = TR.lr_at(step, steps, OPT['lr'], OPT['lr_end'], OPT['warmup'])
        p = CL.decode(arm, mode, z)
        pred = CL.eval_closure(arm, CL.gather(p, mid), T['wi'], T['wo'])
        g = group_terms(pred, T['f'], T['w'], nwo)
        per = g[objective].view(nm, nwi).mean(1)
        if not torch.isfinite(per).all():
            raise RuntimeError(f'non-finite loss at step {step}')
        if step % log_every == 0 or step == steps:
            log.append({'step': step, 'median_loss': float(per.median()), 'mean_loss': float(per.mean()),
                        'median_e1': float(g['e1'].view(nm, nwi).mean(1).median()), 'median_shape': float(g['shape'].view(nm, nwi).mean(1).median()),
                        't': time.time() - t0})
        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        per.sum().backward()
        opt.step()
    return z.detach(), per.detach(), log


def tag(arm, mode, objective, steps, init):
    return f'{arm}_{mode}_{objective}_{steps}_{init}'


def main():
    data, setname, arm, mode, objective, steps, init, out = sys.argv[1:9]
    steps = int(steps)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(SD['fit_init'])
    T = load_fit(os.path.join(data, f'fit82_{setname}.npz'))
    z0 = CL.init_raw(arm, mode, T['n_mat'], dev_t)
    if init == 'reinit':
        g = torch.Generator(device='cpu').manual_seed(SD['reinit'])
        z0 = z0 + P82['fitting']['reinit_sigma'] * torch.randn(z0.shape, generator=g).to(dev_t)
    t0 = time.time()
    z, per, log = fit(arm, mode, objective, z0, T, steps)
    tg = tag(arm, mode, objective, steps, init)
    np.savez(os.path.join(out, f'fit_{setname}_{tg}.npz'), z=z.cpu().numpy(), loss=per.cpu().numpy())
    r = {'set': setname, 'arm': arm, 'mode': mode, 'objective': objective, 'steps': steps, 'init': init, 'seconds': time.time() - t0,
         'median_final_loss': float(per.median()), 'log': log, 'peak_mem_gb': torch.cuda.max_memory_allocated() / 1e9,
         'n_params_per_material': CL.n_params(arm)}
    json.dump(r, open(os.path.join(out, f'fit_{setname}_{tg}.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != 'log'}), flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
