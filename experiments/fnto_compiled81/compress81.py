"""WL81 compression comparator: isotropic tabulations of the reference f(theta_i, theta_o, phi_d) at 8^3 / 16^3 / 32^3 nodes
(node values from the layered reference), trilinear lookup, E_L1 / E_alb on the test_iid evaluation set (protocol compression).

  python compress81.py <data> <evalD_dir> <out> <primary_arm>
"""
import json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
import data81 as DA  # noqa: E402
P = DA.P
dev_t = torch.device('cuda')
WALKS = 1024


def nodes(n):
    ti = (np.arange(n) + 0.5) / n * (np.pi / 2)
    pd = (np.arange(n) + 0.5) / n * np.pi
    return ti, ti.copy(), pd


def build_table(d, n, seed):
    ti, to, pd = nodes(n)
    TI, TO, PD = np.meshgrid(ti, to, pd, indexing='ij')
    wi1 = np.stack([np.sin(TI.ravel()), np.zeros(TI.size), np.cos(TI.ravel())], 1)
    wo1 = np.stack([np.sin(TO.ravel()) * np.cos(PD.ravel()), np.sin(TO.ravel()) * np.sin(PD.ravel()), np.cos(TO.ravel())], 1)
    nm = len(d)
    mm = np.repeat(np.arange(nm), n ** 3)
    wi = np.tile(wi1, (nm, 1)); wo = np.tile(wo1, (nm, 1))
    fA, fB = DA.ref_eval(d, mm, wi, wo, WALKS, seed)
    ev = 0.5 * (fA + fB)
    f = ev / np.maximum(wo[:, 2:3], 1e-6)            # f = eval / cos_o
    return f.reshape(nm, n, n, n, 3)


def lookup(tab, m, wi, wo):
    """Trilinear lookup of f in (theta_i, theta_o, phi_d) with clamped edges; returns eval = f cos_o [N,3]."""
    nm, n = tab.shape[0], tab.shape[1]
    ti = np.arccos(np.clip(wi[:, 2], -1, 1)); to = np.arccos(np.clip(wo[:, 2], -1, 1))
    pd = np.abs(np.mod(np.arctan2(wo[:, 1], wo[:, 0]) - np.arctan2(wi[:, 1], wi[:, 0]) + np.pi, 2 * np.pi) - np.pi)
    x = np.stack([ti / (np.pi / 2) * n - 0.5, to / (np.pi / 2) * n - 0.5, pd / np.pi * n - 0.5], 1)
    x = np.clip(x, 0, n - 1)
    i0 = np.floor(x).astype(int).clip(0, n - 2) if n > 1 else np.zeros_like(x, int)
    fr = x - i0
    out = np.zeros((len(wi), 3))
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                w = (fr[:, 0] if a else 1 - fr[:, 0]) * (fr[:, 1] if b else 1 - fr[:, 1]) * (fr[:, 2] if c else 1 - fr[:, 2])
                out += w[:, None] * tab[m, i0[:, 0] + a, i0[:, 1] + b, i0[:, 2] + c]
    return out * np.maximum(wo[:, 2:3], 0)


def metrics(pred, ev):
    nm, nth, nph, nwo = [int(v) for v in ev['shape']]
    G = nm * nth * nph
    f = ev['f'].astype(np.float64)
    w = (ev['use'].astype(np.float64) / ev['q'].astype(np.float64))[:, None]
    el1 = (np.abs(pred - f) * w).sum(1).reshape(G, nwo).sum(1) / np.maximum((f * w).sum(1).reshape(G, nwo).sum(1), 1e-12)
    a_c = (pred * w).reshape(G, nwo, 3).sum(1); a_r = (f * w).reshape(G, nwo, 3).sum(1)
    ealb = np.abs(a_c - a_r).sum(1) / np.maximum(a_r.sum(1), 1e-12)
    return {'E_L1_median': float(np.median(el1)), 'E_L1_p90': float(np.quantile(el1, .9)), 'E_alb_median': float(np.median(ealb))}


def main():
    data, evalD, out, primary = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    d = np.load(os.path.join(data, 'materials.npz'))['test_iid']
    ev = dict(np.load(os.path.join(data, 'eval_test_iid.npz')))
    s0 = P['fitting']['compiler']['seeds'][0]
    sd = json.load(open(os.path.join(evalD, 'stageD_eval.json')))
    e_cl = sd['summary'][f'{primary}_s{s0}']['test_iid']['E_L1']['median']
    res = {'closure_E_L1_median': e_cl, 'closure_state_floats': 53, 'tables': {}}
    for n in [8, 16, 32]:
        t0 = time.time()
        tab = build_table(d, n, P['seeds']['tables'] + n)
        pred = lookup(tab, ev['m'].astype(int), ev['wi'].astype(np.float64), ev['wo'].astype(np.float64))
        r = metrics(pred, ev)
        r.update({'floats': 3 * n ** 3, 'bytes_fp32': 12 * n ** 3, 'seconds': time.time() - t0, 'walks_per_node': WALKS})
        res['tables'][n] = r
        print(n, json.dumps(r), flush=True)
    ok = [n for n in [8, 16, 32] if res['tables'][n]['E_L1_median'] <= e_cl]
    if ok:
        comp = min(ok); lower_bound = False
    else:
        comp = 32; lower_bound = True
    ratio = res['tables'][comp]['floats'] / 53
    res['comparator'] = {'table_n': comp, 'ratio_table_over_closure': ratio, 'ratio_is_lower_bound': lower_bound}
    res['PASS_size'] = bool(ratio >= 10)
    json.dump(res, open(os.path.join(out, 'compression.json'), 'w'), indent=1)
    print(json.dumps(res['comparator']), res['PASS_size'])


if __name__ == '__main__':
    main()
    os._exit(0)
