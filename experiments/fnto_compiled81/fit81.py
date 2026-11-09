"""WL81 fitting: per-material direct fits (Stage C gate set, Stage D oracle on test sets) and shared compiler training (Stage D).

  python fit81.py direct  <data> <out> <arm> <set>                 # Stage C: generic init only
  python fit81.py compiler <data> <out> <arm> <seed> [n_train]     # Stage D
  python fit81.py oracle  <data> <out> <arm> <set> <compiler.pt>   # Stage D: best of (generic, compiler) inits; + oracle-U for control
arms: control (WL80C softmax sum bound), candidate (amendment 1), unbounded (oracle-U diagnostic: softplus, no energy bound)
Reused from WL80C (imported): loss_fn, per_material_index, lr_at (train80c), decode / init_bias / compiled_state (closure80c).
"""
import json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import closure81 as K
C = K.install()
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
import train80c as TR  # noqa: E402  (loss_fn, per_material_index, lr_at)
import energy81 as E  # noqa: E402
from data81 import normalise, P  # noqa: E402

dev_t = torch.device('cuda')
FIT = P['fitting']
SD = P['seeds']
DEV = '--dev' in sys.argv


def load_pairs(path):
    z = np.load(path)
    return {k: torch.from_numpy(z[k]).to(dev_t) for k in ['m', 'wi', 'wo', 'f']}


class Compiler(torch.nn.Module):
    """Shared material compiler: 10-d normalised descriptor -> 48 raw closure parameters (WL80C architecture, input 10)."""

    def __init__(self, arm, width, depth):
        super().__init__()
        layers, d = [], 10
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.SiLU()]
            d = width
        self.body = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(d, C.N_RAW)
        with torch.no_grad():
            self.head.weight.mul_(0.1)
            self.head.bias.copy_(E.init_raw('candidate' if arm == 'candidate' else 'control', 1, 'cpu')[0])

    def forward(self, x):
        return self.head(self.body(x))


def arm_params(arm, z, wf=None, uf=None):
    """Training-time closure parameters (candidate: differentiable coarse-grid M)."""
    if arm == 'control':
        return C.decode(z, 'passive')
    p = C.decode(z, 'nonneg')
    if arm == 'unbounded':
        return p
    return E.scaled(p, E.fit_M(p, wf, uf))


def fit_grids():
    return E.fit_grid(dev_t), K.strat_u(16, 32, SD['fit_M'], dev_t)


def direct_fit(arm, z0, fit, n_mat, steps):
    O = {'lr': 0.03, 'lr_end': 3e-4, 'warm': 50}
    wf, uf = fit_grids()
    z = torch.nn.Parameter(z0.clone())
    opt = torch.optim.Adam([z], lr=O['lr'])
    idx = TR.per_material_index(fit['m'], n_mat)
    lanes = idx.reshape(-1)
    mid = torch.arange(n_mat, device=dev_t).repeat_interleave(idx.shape[1])
    wi, wo, f = fit['wi'][lanes], fit['wo'][lanes], fit['f'][lanes]
    for step in range(steps):
        for pg in opt.param_groups:
            pg['lr'] = TR.lr_at(step, steps, O['lr'], O['lr_end'], O['warm'])
        p = arm_params(arm, z, wf, uf)
        pred = K.eval_closure(C.gather(p, mid), wi, wo)
        per = ((torch.log1p(pred) - torch.log1p(f)) ** 2).mean(1).view(n_mat, -1).mean(1)
        opt.zero_grad(set_to_none=True)
        per.sum().backward()
        opt.step()
    with torch.no_grad():
        p = arm_params(arm, z, wf, uf)
        pred = K.eval_closure(C.gather(p, mid), wi, wo)
        per = ((torch.log1p(pred) - torch.log1p(f)) ** 2).mean(1).view(n_mat, -1).mean(1)
    return z.detach(), per


def cmd_direct(data, out, arm, setname):
    steps = 50 if DEV else 3000
    torch.manual_seed(SD['oracle'])
    mats = np.load(os.path.join(data, 'materials.npz'))
    n = len(mats[setname])
    fit = load_pairs(os.path.join(data, f'fit_{setname}.npz'))
    t0 = time.time()
    z, loss = direct_fit(arm, E.init_raw(arm, n, dev_t), fit, n, steps)
    secs = time.time() - t0
    np.savez(os.path.join(out, f'direct_{arm}_{setname}.npz'), z=z.cpu().numpy(), loss=loss.cpu().numpy())
    r = {'arm': arm, 'set': setname, 'steps': steps, 'seconds': secs, 'median_fit_loss': float(loss.median()),
         'peak_mem_gb': torch.cuda.max_memory_allocated() / 1e9}
    json.dump(r, open(os.path.join(out, f'direct_{arm}_{setname}.json'), 'w'), indent=1)
    print(json.dumps(r), flush=True)


def cmd_compiler(data, out, arm, seed, n_train=None):
    T = FIT['compiler']
    steps = 200 if DEV else T['steps']
    torch.manual_seed(seed); np.random.seed(seed)
    mats = np.load(os.path.join(data, 'materials.npz'))
    dtr = mats['train'] if n_train is None else mats['train'][:n_train]
    xtr = torch.from_numpy(normalise(dtr)).to(dev_t)
    xva = torch.from_numpy(normalise(mats['val'])).to(dev_t)
    tr = load_pairs(os.path.join(data, 'pairs_train.npz'))
    va = load_pairs(os.path.join(data, 'pairs_val.npz'))
    idx = TR.per_material_index(tr['m'], len(mats['train']))[:len(dtr)]
    net = Compiler(arm, T['width'], T['depth']).to(dev_t)
    opt = torch.optim.Adam(net.parameters(), lr=T['lr'])
    g = torch.Generator(device=dev_t); g.manual_seed(seed)
    wf, uf = fit_grids()
    log = []
    t0 = time.time()
    B, Kp = T['batch_materials'], T['pairs_per_material']
    for step in range(steps):
        for pg in opt.param_groups:
            pg['lr'] = TR.lr_at(step, steps, T['lr'], T['lr_end'], T['warmup'])
        mb = torch.randint(0, len(xtr), (B,), device=dev_t, generator=g)
        cols = torch.randint(0, idx.shape[1], (B, Kp), device=dev_t, generator=g)
        lanes = idx[mb[:, None], cols].reshape(-1)
        p = arm_params(arm, net(xtr[mb]), wf, uf)
        pl = C.gather(p, torch.arange(B, device=dev_t).repeat_interleave(Kp))
        pred = K.eval_closure(pl, tr['wi'][lanes], tr['wo'][lanes])
        loss = TR.loss_fn(pred, tr['f'][lanes])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), T['grad_clip'])
        opt.step()
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite loss at step {step}')
        if step % 1000 == 0 or step == steps - 1:
            with torch.no_grad():
                pv = arm_params(arm, net(xva), wf, uf)
                vl = 0.0
                for s in range(0, len(va['m']), 1 << 20):
                    e = slice(s, s + (1 << 20))
                    pr = K.eval_closure(C.gather(pv, va['m'][e].long()), va['wi'][e], va['wo'][e])
                    vl += float(((torch.log1p(pr) - torch.log1p(va['f'][e])) ** 2).sum())
                vl /= va['f'].numel()
            log.append({'step': step, 'train_loss': float(loss), 'val_loss': vl, 't': time.time() - t0})
            print(json.dumps(log[-1]), flush=True)
    tag = f'{arm}_s{seed}' + ('' if n_train is None else f'_n{n_train}')
    torch.save({'state_dict': net.state_dict(), 'arm': arm, 'seed': seed, 'steps': steps, 'n_train': len(dtr),
                'n_params': sum(p.numel() for p in net.parameters())}, os.path.join(out, f'compiler_{tag}.pt'))
    json.dump({'log': log, 'seconds': time.time() - t0, 'peak_mem_gb': torch.cuda.max_memory_allocated() / 1e9},
              open(os.path.join(out, f'train_{tag}.json'), 'w'), indent=1)


def load_compiler(path):
    ck = torch.load(path, map_location=dev_t)
    T = FIT['compiler']
    net = Compiler(ck['arm'], T['width'], T['depth']).to(dev_t)
    net.load_state_dict(ck['state_dict'])
    net.eval()
    return net, ck


def cmd_oracle(data, out, arm, setname, ckpt):
    steps = 50 if DEV else 3000
    torch.manual_seed(SD['oracle'])
    mats = np.load(os.path.join(data, 'materials.npz'))
    x = torch.from_numpy(normalise(mats[setname])).to(dev_t)
    n = len(x)
    fit = load_pairs(os.path.join(data, f'fit_{setname}.npz'))
    net, _ = load_compiler(ckpt)
    t0 = time.time()
    with torch.no_grad():
        z_comp = net(x)
    za, la = direct_fit(arm, E.init_raw(arm, n, dev_t), fit, n, steps)
    zb, lb = direct_fit(arm, z_comp, fit, n, steps)
    pick = lb < la
    zP = torch.where(pick[:, None], zb, za)
    lP = torch.minimum(la, lb)
    res = {'arm': arm, 'set': setname, 'frac_compiler_init_chosen': float(pick.float().mean()), 'median_fit_loss_P': float(lP.median()),
           'median_fit_loss_generic': float(la.median()), 'median_fit_loss_compinit': float(lb.median())}
    sv = {'zP': zP.cpu().numpy(), 'loss_generic': la.cpu().numpy(), 'loss_compinit': lb.cpu().numpy(), 'pick_compinit': pick.cpu().numpy()}
    if arm == 'control':
        # oracle-U diagnostic (WL80C): same basis, lambda = softplus without any energy bound, init from the control oracle-P
        pP = C.decode(zP, 'passive')['lam']
        zU0 = zP.clone()
        lg = zU0[:, 3 * C.N_GGX:].view(n, 3, C.R_LOBES + 1)
        lg[..., :C.R_LOBES] = torch.log(torch.expm1(pP.clamp_min(1e-6)))
        zU, lU = direct_fit('unbounded', zU0, fit, n, steps)
        sv['zU'] = zU.cpu().numpy(); sv['lossU'] = lU.cpu().numpy()
        res['median_fit_loss_U'] = float(lU.median())
    res['seconds'] = time.time() - t0
    np.savez(os.path.join(out, f'oracle_{arm}_{setname}.npz'), **sv)
    json.dump(res, open(os.path.join(out, f'oracle_{arm}_{setname}.json'), 'w'), indent=1)
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    os.makedirs(sys.argv[3], exist_ok=True)
    if mode == 'direct':
        cmd_direct(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif mode == 'compiler':
        nt = int(sys.argv[6]) if len(sys.argv) > 6 and not sys.argv[6].startswith('--') else None
        cmd_compiler(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), nt)
    elif mode == 'oracle':
        cmd_oracle(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    os._exit(0)
