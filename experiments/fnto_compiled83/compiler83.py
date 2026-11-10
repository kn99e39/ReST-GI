"""WL83 learned material compiler: descriptor (10, WL81 normalisation) -> raw closure parameters (64) -> closure82.decode(arm, 'cert').

Learns the scattering RESPONSE (not regression onto non-unique direct-fit parameters): the loss is the WL82 per-incidence metric objective
(energy-weighted relative L1 + directional-albedo error) of the decoded closure against the layered reference on the training response set.
The certificate of the cert mode holds for every network output (softmax weights, s in [0, 1], eta_e in [1, 2.5]).

  python compiler83.py train <data83> <arm> <seed> <out> [--dev]
  python compiler83.py predict <data83> <w81> <out_ckpt_dir> <arm> <seed> <set> <fitdir>
"""
import glob, hashlib, json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled82'))
import closure82 as CL  # noqa: E402
import closure83  # noqa: E402,F401  (robust refraction patch, see closure83.py)
import fit82 as FT  # noqa: E402
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
import train80c as TR  # noqa: E402
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled81'))
from data81 import normalise  # noqa: E402

P83 = json.load(open(os.path.join(HERE, 'protocol', 'compiled83.json')))
CP = P83['compiler']
dev_t = torch.device('cuda')
DEV = '--dev' in sys.argv


class Net(torch.nn.Module):
    def __init__(self, arm):
        super().__init__()
        layers, d = [], 10
        for _ in range(CP['depth']):
            layers += [torch.nn.Linear(d, CP['width']), torch.nn.SiLU()]
            d = CP['width']
        self.body = torch.nn.Sequential(*layers)
        self.head = torch.nn.Linear(d, CL.N_RAW)
        with torch.no_grad():
            self.head.weight.mul_(CP['head_weight_scale'])
            self.head.bias.copy_(CL.init_raw(arm, 'cert', 1, 'cpu')[0])

    def forward(self, x):
        return self.head(self.body(x))


def load_response(data, name):
    """All shards of fit82_<name>[_i].npz concatenated (material-major lanes; global material index)."""
    fs = sorted(glob.glob(os.path.join(data, f'fit82_{name}_*.npz'))) or [os.path.join(data, f'fit82_{name}.npz')]
    parts = [np.load(f) for f in fs]
    parts.sort(key=lambda z: int(z['m_offset']))
    T = {}
    T['m'] = torch.from_numpy(np.concatenate([z['m'].astype(np.int64) + int(z['m_offset']) for z in parts])).to(dev_t)
    for k in ['wi', 'wo', 'f']:
        T[k] = torch.from_numpy(np.concatenate([z[k] for z in parts])).to(dev_t)
    T['w'] = torch.from_numpy(np.concatenate([z['use'].astype(np.float32) / z['q'] for z in parts])).to(dev_t)
    T['n_wi'], T['n_wo'] = int(parts[0]['n_wi']), int(parts[0]['n_wo'])
    T['n_mat'] = sum(int(z['n_mat']) for z in parts)
    return T


def descriptors(data, w81, name):
    import data83 as D83
    return D83.descs(data, w81, name)


def val_loss(arm, net, xv, V):
    """Mean over validation materials of the per-material metric objective (all groups)."""
    nm, nwi, nwo = V['n_mat'], V['n_wi'], V['n_wo']
    tot = torch.zeros(nm, device=dev_t)
    with torch.no_grad():
        p = CL.decode(arm, 'cert', net(xv))
        per = 1 << 21
        for s in range(0, len(V['m']), per):
            e = slice(s, min(len(V['m']), s + per))
            pred = CL.eval_closure(arm, CL.gather(p, V['m'][e]), V['wi'][e], V['wo'][e])
            g = FT.group_terms(pred, V['f'][e], V['w'][e], nwo)['metric']
            gi = torch.arange(s // nwo, e.stop // nwo, device=dev_t)
            tot.index_add_(0, gi // nwi, g)
    return float((tot / nwi).mean()), float((tot / nwi).median())


def train(data, arm, seed, out):
    w81 = os.environ.get('W81', os.path.expanduser('~/rest_compiled81/results/data'))
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    T = load_response(data, 'train83'); V = load_response(data, 'val83')
    xt = torch.from_numpy(normalise(descriptors(data, w81, 'train83'))).to(dev_t)
    xv = torch.from_numpy(normalise(descriptors(data, w81, 'val83'))).to(dev_t)
    assert T['n_mat'] == len(xt) and V['n_mat'] == len(xv)
    net = Net(arm).to(dev_t)
    opt = torch.optim.Adam(net.parameters(), lr=CP['lr'])
    g = torch.Generator(device=dev_t); g.manual_seed(seed)
    steps = 400 if DEV else CP['steps']
    every = 100 if DEV else CP['val_every']
    B, Gw, nwi, nwo = CP['batch_materials'], CP['groups_per_material'], T['n_wi'], T['n_wo']
    per_mat = nwi * nwo
    jj = torch.arange(nwo, device=dev_t)
    log, best = [], (float('inf'), -1, None)
    t0 = time.time()
    for step in range(steps):
        for pg in opt.param_groups:
            pg['lr'] = TR.lr_at(step, steps, CP['lr'], CP['lr_end'], CP['warmup'])
        mb = torch.randint(0, T['n_mat'], (B,), device=dev_t, generator=g)
        gb = torch.randint(0, nwi, (B, Gw), device=dev_t, generator=g)
        lanes = ((mb[:, None] * per_mat + gb * nwo)[..., None] + jj).reshape(-1)
        p = CL.decode(arm, 'cert', net(xt[mb]))
        loc = torch.arange(B, device=dev_t).repeat_interleave(Gw * nwo)
        pred = CL.eval_closure(arm, CL.gather(p, loc), T['wi'][lanes], T['wo'][lanes])
        loss = FT.group_terms(pred, T['f'][lanes], T['w'][lanes], nwo)['metric'].mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f'non-finite loss at step {step}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CP['grad_clip'])
        opt.step()
        if (step + 1) % every == 0 or step == steps - 1:
            vm, vmed = val_loss(arm, net, xv, V)
            log.append({'step': step + 1, 'train_loss_batch': float(loss), 'val_loss_mean': vm, 'val_loss_median': vmed, 't': time.time() - t0})
            print(json.dumps(log[-1]), flush=True)
            if vm < best[0]:
                best = (vm, step + 1, {k: v.detach().clone() for k, v in net.state_dict().items()})
    tag = f'{arm}_s{seed}'
    ck = {'state_dict': best[2], 'arm': arm, 'seed': seed, 'selected_step': best[1], 'selected_val_loss': best[0], 'steps': steps,
          'n_params': sum(p.numel() for p in net.parameters()), 'width': CP['width'], 'depth': CP['depth']}
    torch.save(ck, os.path.join(out, f'compiler_{tag}.pt'))
    torch.save({**ck, 'state_dict': net.state_dict(), 'selected_step': steps}, os.path.join(out, f'compiler_{tag}_final.pt'))
    sha = hashlib.sha256(open(os.path.join(out, f'compiler_{tag}.pt'), 'rb').read()).hexdigest()
    json.dump({'log': log, 'seconds': time.time() - t0, 'selected_step': best[1], 'selected_val_loss': best[0], 'n_params': ck['n_params'],
               'checkpoint_sha256': sha, 'peak_mem_gb': torch.cuda.max_memory_allocated() / 1e9, 'gpu': torch.cuda.get_device_name(0)},
              open(os.path.join(out, f'train_{tag}.json'), 'w'), indent=1)


def load_net(ckdir, arm, seed, final=False):
    ck = torch.load(os.path.join(ckdir, f'compiler_{arm}_s{seed}{"_final" if final else ""}.pt'), map_location=dev_t)
    net = Net(arm).to(dev_t)
    net.load_state_dict(ck['state_dict'])
    net.eval()
    return net, ck


def predict(data, w81, ckdir, arm, seed, setname, fitdir):
    """Raw compiler outputs for a material set, stored like a direct fit: fit_<set>_L_<arm>_s<seed>.npz (z [M, 64])."""
    import data83 as D83
    if setname == 'train_eval':
        d = np.load(os.path.join(w81, 'materials.npz'))['train'][:300]
    elif setname == 'wl82_test_iid':
        d = np.load(os.path.join(w81, 'materials.npz'))['test_iid']
    else:
        d = D83.descs(data, w81, setname)
    net, ck = load_net(ckdir, arm, seed)
    with torch.no_grad():
        z = net(torch.from_numpy(normalise(d)).to(dev_t)).cpu().numpy()
    os.makedirs(fitdir, exist_ok=True)
    np.savez(os.path.join(fitdir, f'fit_{setname}_L_{arm}_s{seed}.npz'), z=z)


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
    else:
        predict(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), sys.argv[7], sys.argv[8])
    os._exit(0)
