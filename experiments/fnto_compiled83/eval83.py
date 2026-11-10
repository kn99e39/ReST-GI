"""WL83 evaluation of direct-fit (DF) and learned (L) closures against the layered reference (WL81 / WL82 metrics).

  python eval83.py <data83> <w81data> <fitdir> <w82fitdir> <set> <out> [--dev]
sets: test83_iid, test83_block, test83_hard, val83 (new / WL81-val), train_eval (WL81 train subset, fresh directions; L only),
      wl82_test_iid, real83 (previously inspected WL82 sets: test_iid and the 8 parameterised finishes; DF = the stored WL82 fits)
Per (material, wi) group: E_L1, E_alb, E_shape (eval82.group_metrics), and for L the compiler TRANSFER error against the DF of the same arm:
  E_tr = sum_k w |f_L - f_DF|_1 / sum_c a_ref,c   (same samples; representation error = REF vs DF, total learned error = REF vs L).
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled82'))
import closure82 as CL  # noqa: E402
import closure83  # noqa: E402,F401  (robust refraction patch, see closure83.py)
import eval82 as E82  # noqa: E402
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled81'))
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
import eval80c as EV  # noqa: E402
import eval81 as EV81  # noqa: E402
import data81 as DA  # noqa: E402
import data83 as D83  # noqa: E402

P83 = json.load(open(os.path.join(HERE, 'protocol', 'compiled83.json')))
DFS = int(os.environ.get('W83_DF_STEPS', '12000'))   # dev smoke only overrides
dev_t = torch.device('cuda')
DEV = '--dev' in sys.argv
ARMS = ['C0', 'E', 'ER']
SEEDS = P83['compiler']['seeds']
W82FIT = [None]


def set_files(data, w81, s):
    if s == 'train_eval':
        return np.load(os.path.join(w81, 'materials.npz'))['train'][:300], os.path.join(w81, 'eval_train_eval.npz'), None
    if s == 'wl82_test_iid':
        return np.load(os.path.join(w81, 'materials.npz'))['test_iid'], os.path.join(w81, 'eval_test_iid.npz'), os.path.join(w81, 'hl_test_iid.npz')
    if s == 'real83':
        w82d = os.path.join(os.path.dirname(W82FIT[0]), 'data')
        return D83.descs(data, w81, s), os.path.join(w82d, 'eval82_real.npz'), os.path.join(w82d, 'hl82_real.npz')
    hl = os.path.join(data, f'hl83_{s}.npz')
    return D83.descs(data, w81, s), os.path.join(data, f'eval83_{s}.npz'), hl if os.path.exists(hl) else None


def closures(fitdir, w82fit, s):
    out = {}
    for arm in ARMS:
        w82name = {'wl82_test_iid': 'test_iid', 'real83': 'real'}.get(s)
        df = os.path.join(w82fit if w82name else fitdir, f'fit_{w82name or s}_{arm}_cert_metric_{12000 if w82name else DFS}_generic.npz')
        if os.path.exists(df):
            out[f'DF_{arm}'] = (arm, df)
        for sd in SEEDS:
            fl = os.path.join(fitdir, f'fit_{s}_L_{arm}_s{sd}.npz')
            if os.path.exists(fl):
                out[f'L_{arm}_s{sd}'] = (arm, fl)
    return out


def main():
    data, w81, fitdir, w82fit, s, out = sys.argv[1:7]
    W82FIT[0] = w82fit
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    d, evp, hlp = set_files(data, w81, s)
    ev = dict(np.load(evp))
    nm, nth, nph, nwo = [int(v) for v in ev['shape']]
    nm = min(nm, len(d))
    G = nm * nth * nph; L = G * nwo
    hl = dict(np.load(hlp)) if hlp else None
    if hl is not None and nm < int(hl['mat'].max()) + 1:
        k = hl['mat'] < nm
        hl = {kk: (v[k] if (hasattr(v, 'shape') and v.shape[:1] == k.shape) else v) for kk, v in hl.items()}
    f = torch.from_numpy(ev['f'][:L]).to(dev_t).double().view(G, nwo, 3)
    w = (torch.from_numpy(ev['use'][:L].astype(np.float64)).to(dev_t) / torch.from_numpy(ev['q'][:L]).to(dev_t).double()).view(G, nwo, 1)
    fA = torch.from_numpy(ev['fA'][:L]).to(dev_t).double().view(G, nwo, 3); fB = torch.from_numpy(ev['fB'][:L]).to(dev_t).double().view(G, nwo, 3)
    ones = torch.ones(G, 3, dtype=torch.float64, device=dev_t)
    nz, _, _ = E82.group_metrics(fA, fB, w, ones)
    noise = {'E_L1_median': EV.q(nz['el1'] / 2, .5), 'E_L1_p90': EV.q(nz['el1'] / 2, .9), 'E_alb_median': EV.q(nz['ealb'] / 2, .5),
             'E_shape_median': EV.q(nz['eshape'] / 2, .5), 'per_material_E_L1_median': [float(np.median(x)) for x in (nz['el1'] / 2).reshape(nm, -1)]}
    del fA, fB
    den = (f * w).sum(1).sum(1)
    hard = D83.hard_mask(d[:nm])
    res = {'set': s, 'n_materials': nm, 'noise_floor': noise, 'n_hard_box_members': int(hard.sum()), 'closures': {}}
    lanes = {}
    preds = {}
    cl = closures(fitdir, w82fit, s)
    for name, (arm, fn) in cl.items():
        ta = time.time()
        z = torch.from_numpy(np.load(fn)['z'][:nm]).to(dev_t)
        with torch.no_grad():
            p = CL.decode(arm, 'cert', z)
        pred = E82.closure_lanes(arm, p, ev['m'][:L], ev['wi'][:L], ev['wo'][:L]).double().view(G, nwo, 3)
        gm, _, _ = E82.group_metrics(pred, f, w, ones)
        sh = lambda v: v.reshape(nm, nth, nph)
        S = EV.summarise(sh(gm['el1']), sh(gm['ealb']))
        r = {'arm': arm, 'summary': S, 'shape': E82.shape_summary(sh(gm['eshape'])),
             'strata_E_L1': EV81.family_strata(sh(gm['el1']), d[:nm]),
             'failure_cluster_set_terciles': None}
        cm = E82.cluster_mask(d[:nm])
        mm = sh(gm['el1']).mean((1, 2))
        r['failure_cluster_set_terciles'] = {'n': int(cm.sum()), 'E_L1_median': EV.q(mm[cm], .5),
                                             'E_shape_median': EV.q(sh(gm['eshape']).mean((1, 2))[cm], .5)}
        r['hard_box_members'] = {'n': int(hard.sum()), 'E_L1_median': EV.q(mm[hard], .5)}
        r['per_material_E_L1_mean'] = mm.tolist()
        if name.startswith('L_') and f'DF_{arm}' in preds:
            tr = ((pred - preds[f'DF_{arm}']).abs() * w).sum((1, 2)) / den
            tr = tr.cpu().numpy()
            r['transfer'] = {'median': EV.q(tr, .5), 'p90': EV.q(tr, .9), 'max': float(tr.max())}
            lanes[name + '__etr'] = sh(tr).astype(np.float32)
        if name.startswith('DF_'):
            preds[name] = pred
        if hl is not None:
            H = E82.highlight_arm(arm, p, hl, np.ones((len(hl['mat']), 3)))['base']
            r['highlight'] = H
            r['gate'] = EV81.representation_gate(S, H)
        else:
            g = P83['gates']['representation']
            r['gate'] = {'R1': S['E_L1']['median'] <= g['E_L1_median'] and S['E_L1']['p90'] <= g['E_L1_p90'],
                         'R2': S['E_alb']['median'] <= g['E_alb_median'] and S['E_alb']['p95'] <= g['E_alb_p95'], 'R3': None}
        r['seconds'] = time.time() - ta
        res['closures'][name] = r
        for k in ['el1', 'ealb', 'eshape']:
            lanes[f'{name}__{k}'] = sh(gm[k]).astype(np.float32)
        print(s, name, 'E_L1', round(S['E_L1']['median'], 4), round(S['E_L1']['p90'], 4), 'E_alb', round(S['E_alb']['median'], 4),
              'E_shape', round(r['shape']['median'], 4), 'transfer', r.get('transfer', {}).get('median'), flush=True)
    np.savez_compressed(os.path.join(out, f'lanes83_{s}.npz'), **lanes)
    res['seconds'] = time.time() - t0
    json.dump(res, open(os.path.join(out, f'eval83_{s}.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
    os._exit(0)
