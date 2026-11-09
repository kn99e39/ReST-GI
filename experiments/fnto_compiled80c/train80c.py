"""WL80C Stage 2 fitting: shared compiler training (per seed) and per-material
oracle fits (representation control).

  python train80c.py compiler <data_dir> <out_dir> <seed> [--dev]
  python train80c.py oracle   <data_dir> <out_dir> <compiler_ckpt> [--dev]
"""
import json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
import closure80c as C
from data80c import normalise, P

dev_t = torch.device('cuda')


def loss_fn(pred, f):
    return ((torch.log1p(pred) - torch.log1p(f)) ** 2).mean()


def load_pairs(path):
    z = np.load(path)
    return {k: torch.from_numpy(z[k]).to(dev_t) for k in ['m', 'wi', 'wo', 'f']}


def per_material_index(m, n_mat):
    """Lane indices grouped per material (equal counts per material by construction)."""
    order = torch.argsort(m, stable=True)
    k = len(m) // n_mat
    return order.view(n_mat, k)


def lr_at(step, T, lr0, lr_end, warm):
    if step < warm:
        return lr0 * (step + 1) / warm
    t = (step - warm) / max(1, T - warm)
    return lr_end + 0.5 * (lr0 - lr_end) * (1 + math.cos(math.pi * t))


def train_compiler(data, out, seed, dev):
    T = P['training']
    steps = T['dev_steps'] if dev else T['steps']
    torch.manual_seed(seed); np.random.seed(seed)
    mats = np.load(os.path.join(data, 'materials.npz'))
    xtr = torch.from_numpy(normalise(mats['train'])).to(dev_t)
    xva = torch.from_numpy(normalise(mats['val'])).to(dev_t)
    tr = load_pairs(os.path.join(data, 'pairs_train.npz'))
    va = load_pairs(os.path.join(data, 'pairs_val.npz'))
    idx = per_material_index(tr['m'], len(xtr))
    net = C.Compiler(T['width'], T['depth']).to(dev_t)
    opt = torch.optim.Adam(net.parameters(), lr=T['lr'])
    g = torch.Generator(device=dev_t); g.manual_seed(seed)
    log = []
    t0 = time.time()
    B, K = T['batch_materials'], T['pairs_per_material']
    for step in range(steps):
        for pg in opt.param_groups:
            pg['lr'] = lr_at(step, steps, T['lr'], T['lr_end'], T['warmup'])
        mb = torch.randint(0, len(xtr), (B,), device=dev_t, generator=g)
        cols = torch.randint(0, idx.shape[1], (B, K), device=dev_t, generator=g)
        lanes = idx[mb[:, None], cols].reshape(-1)
        p = C.decode(net(xtr[mb]))
        pl = C.gather(p, torch.arange(B, device=dev_t).repeat_interleave(K))
        pred = C.eval_closure(pl, tr['wi'][lanes], tr['wo'][lanes])
        loss = loss_fn(pred, tr['f'][lanes])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), T['grad_clip'])
        opt.step()
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite loss at step {step}')
        if step % T['log_every'] == 0 or step == steps - 1:
            with torch.no_grad():
                pv = C.decode(net(xva))
                vl = 0.0
                for s in range(0, len(va['m']), 1 << 20):
                    e = slice(s, s + (1 << 20))
                    pr = C.eval_closure(C.gather(pv, va['m'][e].long()), va['wi'][e], va['wo'][e])
                    vl += float(((torch.log1p(pr) - torch.log1p(va['f'][e])) ** 2).sum())
                vl /= va['f'].numel()
            log.append({'step': step, 'train_loss': float(loss), 'val_loss': vl, 't': time.time() - t0})
            print(json.dumps(log[-1]), flush=True)
    os.makedirs(out, exist_ok=True)
    torch.save({'state_dict': net.state_dict(), 'seed': seed, 'steps': steps,
                'n_params': sum(p.numel() for p in net.parameters())}, os.path.join(out, f'compiler_s{seed}.pt'))
    json.dump({'log': log, 'seconds': time.time() - t0,
               'peak_mem_gb': torch.cuda.max_memory_allocated() / 1e9}, open(os.path.join(out, f'train_s{seed}.json'), 'w'))


def fit_free(z0, fit, n_mat, steps, lr0, lr_end, lam_mode):
    """Per-material free optimisation of raw closure parameters (all materials in parallel)."""
    z = torch.nn.Parameter(z0.clone())
    opt = torch.optim.Adam([z], lr=lr0)
    idx = per_material_index(fit['m'], n_mat)
    lanes = idx.reshape(-1)
    mid = torch.arange(n_mat, device=dev_t).repeat_interleave(idx.shape[1])
    wi, wo, f = fit['wi'][lanes], fit['wo'][lanes], fit['f'][lanes]
    for step in range(steps):
        for pg in opt.param_groups:
            pg['lr'] = lr_at(step, steps, lr0, lr_end, 50)
        p = C.decode(z, lam_mode)
        pred = C.eval_closure(C.gather(p, mid), wi, wo)
        per = ((torch.log1p(pred) - torch.log1p(f)) ** 2).mean(1).view(n_mat, -1).mean(1)
        opt.zero_grad(set_to_none=True)
        per.sum().backward()
        opt.step()
    with torch.no_grad():
        p = C.decode(z, lam_mode)
        pred = C.eval_closure(C.gather(p, mid), wi, wo)
        per = ((torch.log1p(pred) - torch.log1p(f)) ** 2).mean(1).view(n_mat, -1).mean(1)
    return z.detach(), per


def oracle(data, out, ckpt, dev):
    O = P['oracle']
    steps = O['dev_steps'] if dev else O['steps']
    torch.manual_seed(P['seeds']['oracle'])
    mats = np.load(os.path.join(data, 'materials.npz'))
    T = P['training']
    net = C.Compiler(T['width'], T['depth']).to(dev_t)
    net.load_state_dict(torch.load(ckpt, map_location=dev_t)['state_dict'])
    res = {}
    t0 = time.time()
    for name in ['test_iid', 'test_block']:
        x = torch.from_numpy(normalise(mats[name])).to(dev_t)
        fit = load_pairs(os.path.join(data, f'fit_{name}.npz'))
        n = len(x)
        with torch.no_grad():
            z_comp = net(x)
        z_gen = C.init_bias().to(dev_t)[None].repeat(n, 1)
        za, la = fit_free(z_gen, fit, n, steps, O['lr'], O['lr_end'], 'passive')
        zb, lb = fit_free(z_comp, fit, n, steps, O['lr'], O['lr_end'], 'passive')
        pick_b = lb < la
        zP = torch.where(pick_b[:, None], zb, za)
        lP = torch.minimum(la, lb)
        # diagnostic oracle-U: same basis, lambda >= 0 without the sum bound (init from oracle-P)
        pP = C.decode(zP, 'passive')['lam']
        zU0 = zP.clone()
        lg = zU0[:, 3 * C.N_GGX:].view(n, 3, C.R_LOBES + 1)
        lg[..., :C.R_LOBES] = torch.log(torch.expm1(pP.clamp_min(1e-6)))
        zU, lU = fit_free(zU0, fit, n, steps, O['lr'], O['lr_end'], 'nonneg')
        np.savez(os.path.join(out, f'oracle_{name}.npz'), zP=zP.cpu().numpy(), zU=zU.cpu().numpy(),
                 loss_generic=la.cpu().numpy(), loss_compinit=lb.cpu().numpy(), lossU=lU.cpu().numpy(),
                 pick_compinit=pick_b.cpu().numpy())
        res[name] = {'frac_compiler_init_chosen': float(pick_b.float().mean()),
                     'median_fit_loss_P': float(lP.median()), 'median_fit_loss_U': float(lU.median())}
        print(name, json.dumps(res[name]), flush=True)
    res['seconds'] = time.time() - t0
    res['peak_mem_gb'] = torch.cuda.max_memory_allocated() / 1e9
    json.dump(res, open(os.path.join(out, 'oracle.json'), 'w'), indent=1)


if __name__ == '__main__':
    mode = sys.argv[1]
    dev = '--dev' in sys.argv
    if mode == 'compiler':
        train_compiler(sys.argv[2], sys.argv[3], int(sys.argv[4]), dev)
    elif mode == 'oracle':
        oracle(sys.argv[2], sys.argv[3], sys.argv[4], dev)
