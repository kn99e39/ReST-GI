"""WL83 data (reference walks = WL81 layered81 via data81.ref_eval; WL81 directional conventions).

  python data83.py <out> <w81data> materials                    # materials83.npz: test83_iid / test83_block / test83_hard (new, seed 83010)
  python data83.py <out> <w81data> fitset <set> [--shard=i/K]   # fit82_<set>[_i].npz: 64 wi x 64 wo x 1024 walks (train83 = WL81 train,
                                                                 #   val83 = WL81 val, test83_* = new population)
  python data83.py <out> <w81data> evalset <set> --shard=i/K    # eval shard (WL81 eval layout, 4096 walks)
  python data83.py <out> <w81data> evalmerge <set> K            # eval83_<set>.npz + WL81 noise rule (top-up rounds if needed)
  python data83.py <out> <w81data> hlset <set> --shard=i/K      # highlight grids (WL81 definition) for a material range
  python data83.py <out> <w81data> hlmerge <set> K
"""
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled82'))
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled81'))
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
import data81 as DA  # noqa: E402
import data82 as D82  # noqa: E402

P83 = json.load(open(os.path.join(HERE, 'protocol', 'compiled83.json')))
SD = P83['seeds']
POP = P83['population']
SETS = ['train83', 'val83', 'test83_iid', 'test83_block', 'test83_hard']
DEV = '--dev' in sys.argv
DEVN = {'train83': 64, 'val83': 16, 'test83_iid': 16, 'test83_block': 16, 'test83_hard': 16}


def descs(out, w81, name):
    if name == 'real83':
        return D82.real_desc()
    if name in ('train83', 'val83'):
        d = np.load(os.path.join(w81, 'materials.npz'))[{'train83': 'train', 'val83': 'val'}[name]]
    else:
        d = np.load(os.path.join(out, 'materials83.npz'))[name]
    return d[:DEVN[name]] if DEV else d


def hard_mask(d):
    H = POP['hard_box']
    return ((d[:, 2] >= H['tau'][0]) & (d[:, 2] <= H['tau'][1]) & (d[:, 9] >= H['alpha_b'][0]) & (d[:, 9] <= H['alpha_b'][1])
            & (d[:, 6:9].mean(1) >= H['r_mean_min']))


def materials(out, w81):
    rng = np.random.default_rng(SD['materials'])
    iid, blk, hard = [], [], []
    while sum(map(len, iid)) < POP['test83_iid'] or sum(map(len, blk)) < POP['test83_block']:
        d = DA.sample_box(rng, 20000)
        b = DA.in_block(d)
        iid.append(d[~b]); blk.append(d[b])
    H = POP['hard_box']
    rh = np.random.default_rng(SD['materials'] + 1)
    while sum(map(len, hard)) < POP['test83_hard']:
        d = DA.sample_box(rh, 20000)
        u = rh.random((len(d), 2))
        d[:, 2] = np.exp(np.log(H['tau'][0]) + u[:, 0] * (np.log(H['tau'][1]) - np.log(H['tau'][0])))
        d[:, 9] = np.exp(np.log(H['alpha_b'][0]) + u[:, 1] * (np.log(H['alpha_b'][1]) - np.log(H['alpha_b'][0])))
        hard.append(d[hard_mask(d) & ~DA.in_block(d)])
    sp = {'test83_iid': np.concatenate(iid)[:POP['test83_iid']], 'test83_block': np.concatenate(blk)[:POP['test83_block']],
          'test83_hard': np.concatenate(hard)[:POP['test83_hard']]}
    np.savez(os.path.join(out, 'materials83.npz'), **sp)
    m81 = np.load(os.path.join(w81, 'materials.npz'))
    prev = np.concatenate([m81[k] for k in ['train', 'val', 'gate', 'test_iid', 'test_block']] + [D82.real_desc()])
    P = {tuple(np.round(x, 12)) for x in prev}
    over = {k: int(sum(tuple(np.round(x, 12)) in P for x in v)) for k, v in sp.items()}
    json.dump({'sizes': {k: len(v) for k, v in sp.items()}, 'overlap_with_wl81_wl82_materials': over,
               'hard_fraction_of_iid': float(hard_mask(sp['test83_iid']).mean())}, open(os.path.join(out, 'materials83_meta.json'), 'w'), indent=1)
    print(json.dumps(over), flush=True)


def fitset(out, w81, name, shard):
    F = P83['fitset']
    k = SETS.index(name)
    d = descs(out, w81, name)
    i, K = shard if shard else (0, 1)
    m0, m1 = len(d) * i // K, len(d) * (i + 1) // K
    rng = np.random.default_rng([SD['fit_dirs'], k, i])
    t0 = time.time()
    mm, wi, wo, q, use = DA.train_style(d[m0:m1], F['n_wi'], F['n_wo'], rng)
    fA, fB = DA.ref_eval(d[m0:m1], mm, wi, wo, F['walks'], SD['fit_walks'] + 100 * k + 10007 * i)
    tag = name if K == 1 else f'{name}_{i}'
    np.savez(os.path.join(out, f'fit82_{tag}.npz'), m=mm.astype(np.int32), wi=wi.astype(np.float32), wo=wo.astype(np.float32),
             q=q.astype(np.float32), use=use, f=(0.5 * (fA + fB)).astype(np.float32), fA=fA.astype(np.float32), fB=fB.astype(np.float32),
             n_wi=F['n_wi'], n_wo=F['n_wo'], n_mat=m1 - m0, m_offset=m0)
    print(name, i, K, 'fitset', len(wi), 'pairs', round(time.time() - t0, 1), 's', flush=True)


def _eval_dirs(out, w81, name):
    k = SETS.index(name)
    d = descs(out, w81, name)
    rng = np.random.default_rng([SD['eval_dirs'], k])
    return d, k, DA.eval_style(d, rng)


def evalset(out, w81, name, shard):
    d, k, (mm, wi, wo, q, use, phis, shape) = _eval_dirs(out, w81, name)
    i, K = shard
    b0, b1 = len(wi) * i // K, len(wi) * (i + 1) // K
    t0 = time.time()
    gA, gB = DA.ref_eval(d, mm[b0:b1], wi[b0:b1], wo[b0:b1], DA.WALK['eval_initial'], SD['eval_walks'] + 1000 * k, 0, pair0=b0)
    np.savez(os.path.join(out, f'evshard_{name}_{i}of{K}.npz'), fA=gA, fB=gB, b0=b0, b1=b1, seconds=time.time() - t0)
    print(name, 'evshard', i, K, round(time.time() - t0, 1), 's', flush=True)


def evalmerge(out, w81, name, K):
    d, k, (mm, wi, wo, q, use, phis, shape) = _eval_dirs(out, w81, name)
    fA = np.empty((len(wi), 3)); fB = np.empty((len(wi), 3))
    for i in range(K):
        z = np.load(os.path.join(out, f'evshard_{name}_{i}of{K}.npz'))
        fA[int(z['b0']):int(z['b1'])] = z['fA']; fB[int(z['b0']):int(z['b1'])] = z['fB']
    ev = {'m': mm.astype(np.int32), 'wi': wi.astype(np.float32), 'wo': wo.astype(np.float32), 'q': q.astype(np.float32), 'use': use,
          'phis': phis.astype(np.float32), 'shape': shape}
    nw, rnd, hist = DA.WALK['eval_initial'], 0, []
    while True:
        el1, ealb = DA.noise_floor(ev, fA, fB)
        s = DA.nf_summary(el1, ealb); s['walks'] = nw; hist.append(s)
        print(name, json.dumps(s), flush=True)
        if s['PASS'] or nw >= 65536:
            break
        rnd += 1
        gA, gB = DA.ref_eval(d, mm, wi, wo, nw, SD['eval_walks'] + 1000 * k, rnd)
        fA, fB = 0.5 * (fA + gA), 0.5 * (fB + gB); nw *= 2
    ev.update({'f': (0.5 * (fA + fB)).astype(np.float32), 'fA': fA.astype(np.float32), 'fB': fB.astype(np.float32)})
    np.savez(os.path.join(out, f'eval83_{name}.npz'), **ev)
    json.dump({'history': hist, 'PASS': hist[-1]['PASS'], 'walks': nw}, open(os.path.join(out, f'noise83_{name}.json'), 'w'), indent=1)


def hlset(out, w81, name, shard):
    d = descs(out, w81, name)
    ev = np.load(os.path.join(out, f'eval83_{name}.npz'))
    phis = ev['phis'][:, 0].astype(np.float64)
    i, K = shard
    m0, m1 = len(d) * i // K, len(d) * (i + 1) // K
    t0 = time.time()
    o = D82.highlight_ref(d[m0:m1], phis[m0:m1], SD['hl_walks'] + 1000 * SETS.index(name) + 101 * i)
    o['mat'] = o['mat'] + m0
    np.savez(os.path.join(out, f'hlshard_{name}_{i}of{K}.npz'), **o)
    print(name, 'hlshard', i, K, round(time.time() - t0, 1), 's', flush=True)


def hlmerge(out, name, K):
    parts = [dict(np.load(os.path.join(out, f'hlshard_{name}_{i}of{K}.npz'))) for i in range(K)]
    o = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(os.path.join(out, f'hl83_{name}.npz'), **o)
    pr = o['pron']
    dots = np.clip((o['A_peak_wo'] * o['B_peak_wo']).sum(1), -1, 1)
    pk = np.degrees(np.arccos(dots))
    res = ~(o['A_touch'] | o['B_touch'])
    lw = 0.5 * np.log(o['A_area'] / np.maximum(o['B_area'], 1))
    s = {'n_cases': int(len(pr)), 'n_pronounced': int(pr.sum()), 'peak_err_deg_median_halves': float(np.median(pk[pr])) if pr.any() else None,
         'abs_logw_median_halves': float(np.median(np.abs(lw[pr & res]))) if (pr & res).any() else None}
    json.dump(s, open(os.path.join(out, f'hl83_noise_{name}.json'), 'w'), indent=1)
    print(name, json.dumps(s), flush=True)


SHARD = None
for _a in sys.argv:
    if _a.startswith('--shard='):
        _i, _k = _a.split('=')[1].split('/'); SHARD = (int(_i), int(_k))

if __name__ == '__main__':
    out, w81, what = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    DA.mi_init()
    if what == 'materials':
        materials(out, w81)
    elif what == 'fitset':
        fitset(out, w81, sys.argv[4], SHARD)
    elif what == 'evalset':
        evalset(out, w81, sys.argv[4], SHARD)
    elif what == 'evalmerge':
        evalmerge(out, w81, sys.argv[4], int(sys.argv[5]))
    elif what == 'hlset':
        hlset(out, w81, sys.argv[4], SHARD)
    elif what == 'hlmerge':
        hlmerge(out, sys.argv[4], int(sys.argv[5]))
    os._exit(0)
