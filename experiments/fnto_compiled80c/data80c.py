"""WL80C Stage 0/2 data: material splits (reference-passive rejection) and
reference direction sets, all from the Mitsuba principled reference.

Usage: python data80c.py <out_dir> [--dev]
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import ref80c as R

P = json.load(open(os.path.join(os.path.dirname(__file__), 'protocol', 'compiled80c.json')))
BOX = P['materials']['box']
BLOCK = P['materials']['heldout_block']
KEYS = ['metallic', 'roughness', 'anisotropic', 'clearcoat', 'clearcoat_gloss', 'sheen']


def sample_box(rng, n, box):
    d = np.empty((n, 9), np.float32)
    d[:, 0:3] = rng.uniform(*box['base'], (n, 3))
    for j, k in zip(range(3, 9), KEYS):
        d[:, j] = rng.uniform(*box[k], n)
    return d


def in_block(d):
    m = np.ones(len(d), bool)
    for k, (lo, hi) in BLOCK.items():
        j = 3 + KEYS.index(k)
        m &= (d[:, j] >= lo) & (d[:, j] <= hi)
    return m


def normalise(d):
    """Descriptor -> [-1, 1] per dimension over the frozen box."""
    lo = np.array([BOX['base'][0]] * 3 + [BOX[k][0] for k in KEYS], np.float32)
    hi = np.array([BOX['base'][1]] * 3 + [BOX[k][1] for k in KEYS], np.float32)
    return (2 * (d - lo) / (hi - lo) - 1).astype(np.float32)


def passive_mask(d, rng):
    """Reference passivity filter: point-estimate albedo <= 1 for all channels on
    the frozen wi grid (P['materials']['passivity_filter'])."""
    pf = P['materials']['passivity_filter']
    ct = np.linspace(np.cos(np.radians(pf['theta_max_deg'])), 1.0, pf['n_wi'])
    ns = pf['samples']
    bsdf, n = R.textured_principled(d)
    amax = np.zeros(len(d))
    step = max(1, (1 << 23) // (pf['n_wi'] * ns))
    for k0 in range(0, len(d), step):
        ks = np.arange(k0, min(k0 + step, len(d)))
        mm = np.repeat(ks, pf['n_wi'] * ns)
        th = np.tile(np.repeat(np.arccos(ct), ns), len(ks))
        wi = R.sph(th, rng.uniform(0, 2 * np.pi, len(mm)))
        wo, q, f, use = R.mixture_dirs(bsdf, n, mm, wi, rng)
        g = np.where(use[:, None], f.astype(np.float64) / q[:, None], 0.0).reshape(len(ks), pf['n_wi'], ns, 3)
        amax[ks] = g.mean(2).max((1, 2))
    return amax <= 1.0, amax


def draw_passive(rng, n_needed, box, want_block):
    out, amax_all, drawn = [], [], 0
    while sum(len(o) for o in out) < n_needed:
        d = sample_box(rng, 2000, box)
        b = in_block(d)
        d = d[b] if want_block else d[~b]
        ok, amax = passive_mask(d, rng)
        drawn += len(d)
        out.append(d[ok]); amax_all.append(amax)
    d = np.concatenate(out)[:n_needed]
    acc = sum(len(o) for o in out) / drawn
    return d, acc


def train_style(bsdf, n, mats, n_wi, n_wo, rng, band):
    """Training-style pairs: wi cos-uniform in [cos 88deg, 1] excluding the
    held-out incident band; n_wo mixture wo per wi."""
    cmin = np.cos(np.radians(88.0))
    tot = len(mats) * n_wi
    th = np.empty(0)
    while len(th) < tot:
        c = rng.uniform(cmin, 1.0, 2 * tot)
        t = np.degrees(np.arccos(c))
        th = np.concatenate([th, np.radians(t[(t < band[0]) | (t > band[1])])])
    th = th[:tot]
    wi1 = R.sph(th, rng.uniform(0, 2 * np.pi, tot))
    mm = np.repeat(mats, n_wi * n_wo)
    wi = np.repeat(wi1, n_wo, 0)
    wo, q, f, use = R.mixture_dirs(bsdf, n, mm, wi, rng)
    return {'m': mm.astype(np.int32), 'wi': wi.astype(np.float32), 'wo': wo, 'f': f}


def eval_style(bsdf, n, mats, rng, thetas, n_phi, n_wo):
    nm = len(mats)
    phis = rng.uniform(0, 2 * np.pi, (nm, n_phi))
    th = np.radians(np.array(thetas, np.float64))
    # lanes ordered (material, theta, phi, sample)
    T = np.broadcast_to(th[None, :, None], (nm, len(th), n_phi))
    PH = np.broadcast_to(phis[:, None, :], (nm, len(th), n_phi))
    wi1 = R.sph(T.ravel(), PH.ravel())
    mm = np.repeat(np.repeat(mats, len(th) * n_phi), n_wo)
    wi = np.repeat(wi1, n_wo, 0)
    wo, q, f, use = R.mixture_dirs(bsdf, n, mm, wi, rng)
    return {'m': mm.astype(np.int32), 'wi': wi.astype(np.float32), 'wo': wo, 'f': f,
            'q': q.astype(np.float32), 'use': use, 'phis': phis.astype(np.float32),
            'shape': np.array([nm, len(th), n_phi, n_wo])}


def main():
    out = sys.argv[1]
    dev = '--dev' in sys.argv
    os.makedirs(out, exist_ok=True)
    S = P['dev_sizes'] if dev else P['sizes']
    seeds = P['seeds']['dev_data'] if dev else P['seeds']['data']
    rng = np.random.default_rng(seeds)
    t0 = time.time()
    n_nb = S['train'] + S['val'] + S['test_iid']
    d_nb, acc_nb = draw_passive(rng, n_nb, BOX, False)
    blk_box = dict(BOX); blk_box.update({k: tuple(v) for k, v in BLOCK.items()})
    d_b, acc_b = draw_passive(rng, S['test_block'], blk_box, True)
    splits = {'train': d_nb[:S['train']], 'val': d_nb[S['train']:S['train'] + S['val']],
              'test_iid': d_nb[S['train'] + S['val']:], 'test_block': d_b}
    np.savez(os.path.join(out, 'materials.npz'), **splits)
    meta = {'accept_nonblock': acc_nb, 'accept_block': acc_b, 'sizes': S, 'seed': seeds,
            't_materials_s': time.time() - t0}
    band = P['angular']['heldout_incident_band_deg']
    A = P['angular']
    # training / validation sets
    for name, nwi, nwo in [('train', A['train_wi'], A['train_wo']), ('val', A['val_wi'], A['val_wo'])]:
        d = splits[name]
        b, n = R.textured_principled(d)
        ds = train_style(b, n, np.arange(len(d)), nwi, nwo, rng, band)
        np.savez(os.path.join(out, f'pairs_{name}.npz'), **ds)
    # oracle fit sets (fresh directions, test materials)
    for name in ['test_iid', 'test_block']:
        d = splits[name]
        b, n = R.textured_principled(d)
        ds = train_style(b, n, np.arange(len(d)), A['train_wi'], A['train_wo'], rng, band)
        np.savez(os.path.join(out, f'fit_{name}.npz'), **ds)
    # evaluation sets (fresh directions)
    ev = {'test_iid': splits['test_iid'], 'test_block': splits['test_block'],
          'train_eval': splits['train'][:S['train_eval']]}
    for name, d in ev.items():
        b, n = R.textured_principled(d)
        ds = eval_style(b, n, np.arange(len(d)), rng, A['eval_theta_deg'], A['eval_n_phi'], A['eval_wo'])
        np.savez(os.path.join(out, f'eval_{name}.npz'), **ds)
    meta['t_total_s'] = time.time() - t0
    json.dump(meta, open(os.path.join(out, 'data_meta.json'), 'w'), indent=1)
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
    os._exit(0)
