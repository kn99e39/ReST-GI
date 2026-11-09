"""WL81 physical contract: WL80C contract80c (C1-C7, C5b; its code and thresholds, imported) run on the corrected runtime
(closure81_dr, by module substitution), plus the WL81 grazing-limit passivity check G81 and the arm-specific C3 rule.

  python contract81.py stageC <data> <evalC_dir> <out>              # direct fits on the gate set + adversarial, per arm
  python contract81.py stageD <data> <evalD_dir> <out> <arm> ...    # learned closures (both seeds x test_iid / test_block) + adversarial
"""
import json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import closure81 as K
C = K.install()
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled80c'))
import contract80c as CO  # noqa: E402  (sets the Mitsuba variant through ref80c.mi_init)
import closure81_dr as CD  # noqa: E402
import energy81 as E  # noqa: E402
import layered81 as L  # noqa: E402
from data81 import P  # noqa: E402

CO.CD = CD
mi = CO.mi
import drjit as dr  # noqa: E402
SD = P['seeds']
dev_t = torch.device('cuda')
V3, np3, unit = CO.V3, CO.np3, CO.unit


def adversarial(arm):
    """50 adversarial raw closures (seed adversarial, sigma 3, a quarter saturated), compiled by the arm's own rule."""
    g = torch.Generator().manual_seed(SD['adversarial'])
    na = 50
    z = torch.randn(na, C.N_RAW, generator=g) * 3.0
    z[: na // 4, :3 * C.N_GGX] = torch.sign(z[: na // 4, :3 * C.N_GGX]) * 12.0
    z = z.to(dev_t)
    if arm == 'control':
        p = C.decode(z, 'passive')
    else:
        p, _ = E.compiled_params('candidate', z, E.compile_grids(dev_t), K.strat_u(128, 128, SD['compile_M'], dev_t))
    return C.compiled_state(p).cpu().numpy()


_SEQ = {'k': 0}


def own_sampler_albedo(tab, cs, wi_rows, ns, rng):
    """Albedo by the closure's own sampler for closures cs (each with its own wi rows [len(cs), W, 3]); random numbers from GPU PCG32
    streams (seeded from the contract seed and a call counter), per-row mean / standard error by dr.block_sum. Returns a, se [len(cs)*W, 3]."""
    W = wi_rows.shape[1]
    nrow = len(cs) * W
    n = nrow * ns
    _SEQ['k'] += 1
    lane = dr.arange(mi.UInt32, n)
    row = lane // ns
    idx = dr.gather(mi.UInt32, mi.UInt32(np.ascontiguousarray(np.repeat(cs, W), np.uint32)), row)
    wr = np.ascontiguousarray(wi_rows.reshape(-1, 3), np.float32)
    wi = mi.Vector3f(*[dr.gather(mi.Float, mi.Float(wr[:, k]), row) for k in range(3)])
    rg = mi.PCG32(size=n, initstate=SD['grazing'] * 1000003 + _SEQ['k'], initseq=dr.arange(mi.UInt64, n))
    u1 = rg.next_float32()
    u2 = mi.Point2f(rg.next_float32(), rg.next_float32())
    cl = CD.DrClosure(tab, idx)
    _, _, w, _ = cl.sample(wi, u1, u2)
    s1 = np.stack([np.asarray(dr.block_sum(w[k], ns), np.float64) for k in range(3)], 1)
    s2 = np.stack([np.asarray(dr.block_sum(w[k] * w[k], ns), np.float64) for k in range(3)], 1)
    a = s1 / ns
    var = np.maximum(s2 / ns - a * a, 0)
    return a, np.sqrt(var / ns)


def g81(tab, rng):
    M = len(tab)
    viol, amax, cases = 0, 0.0, 0
    worst = None
    # (a) fixed near-grazing angles and mu = 2e-7, 4 azimuths, 16384 samples
    mus = np.concatenate([np.cos(np.radians([89.9, 89.99, 89.999, 89.9999])), [2e-7]])
    ns = 16384
    per = max(1, (1 << 24) // (len(mus) * 4 * ns))
    for c0 in range(0, M, per):
        cs = np.arange(c0, min(M, c0 + per))
        ph = rng.uniform(0, 2 * np.pi, (len(cs), len(mus) * 4))
        mu = np.tile(np.repeat(mus, 4), (len(cs), 1))
        st = np.sqrt(1 - mu * mu)
        rows = np.stack([st * np.cos(ph), st * np.sin(ph), mu], -1)
        a, se = own_sampler_albedo(tab, cs, rows, ns, rng)
        v = (a - 1 > np.maximum(4 * se, 1e-3)).any(1)
        viol += int(v.sum()); cases += len(a)
        if a.max() > amax:
            amax = float(a.max()); worst = {'closure': int(cs[np.argmax(a.max(1)) // rows.shape[1]]), 'albedo': amax}
    ra = {'cases': cases, 'violations': viol, 'max_albedo': amax, 'worst': worst}
    # (b) 1024 random wi per closure (half log-uniform mu in [2e-7, 1e-2], half uniform in [0, 1]), 65536 samples
    nw, ns = 1024, 65536
    viol2, amax2, cases2 = 0, 0.0, 0
    for c in range(M):
        mu = np.concatenate([np.exp(rng.uniform(np.log(2e-7), np.log(1e-2), nw // 2)), rng.uniform(2e-7, 1, nw // 2)])
        ph = rng.uniform(0, 2 * np.pi, nw)
        st = np.sqrt(1 - mu * mu)
        rows = np.stack([st * np.cos(ph), st * np.sin(ph), mu], -1)
        for r0 in range(0, nw, 256):
            a, se = own_sampler_albedo(tab, np.array([c]), rows[None, r0:r0 + 256], ns, rng)
            viol2 += int((a - 1 > np.maximum(4 * se, 1e-3)).any(1).sum()); cases2 += len(a)
            amax2 = max(amax2, float(a.max()))
    rb = {'cases': cases2, 'violations': viol2, 'max_albedo': amax2}
    return {'fixed_grazing': ra, 'random_wi': rb, 'PASS': ra['violations'] == 0 and rb['violations'] == 0}


def utility_factory(tab, desc):
    """Report-only sampling utility: relative variance per sample of int f cos L, closure sampler vs the reference's Sample_f weight."""
    def sampling_utility(tabs, mats, rng):
        U = CO.CT['utility']
        ths = np.radians(np.array(U['theta_deg']))
        ns = U['samples']
        Mm = min(len(tab), 64)
        ratios = {}
        for kind in CO.CT['lightings']:
            Gn = Mm * len(ths)
            idx = np.repeat(np.arange(Mm), len(ths) * ns)
            th = np.tile(np.repeat(ths, ns), Mm)
            wi = CO.R.sph(th, np.repeat(rng.uniform(0, 2 * np.pi, Gn), ns))
            dvec = np.repeat(unit(rng, Gn, True), ns, 0)
            cl = CO.closure_lanes(tab, idx)
            u1 = rng.random(len(idx)); u2 = rng.random((len(idx), 2))
            wo, _, w, _ = cl.sample(V3(wi), mi.Float(u1.astype(np.float32)), mi.Point2f(mi.Float(u2[:, 0].astype(np.float32)), mi.Float(u2[:, 1].astype(np.float32))))
            Ic = (np3(w).mean(1) * CO.lighting(kind, np3(wo), dvec)).reshape(Gn, ns)
            tabd = [mi.Float(np.ascontiguousarray(desc[idx, j], np.float32)) for j in range(10)]
            mat = L.Mat(tabd[0], tabd[1], tabd[2], mi.Color3f(tabd[3], tabd[4], tabd[5]), mi.Color3f(tabd[6], tabd[7], tabd[8]), tabd[9])
            rg = mi.PCG32(size=len(idx), initstate=SD['contract'] + 5, initseq=dr.arange(mi.UInt64, len(idx)))
            wo_r, w_r, ok = L.layered_sample(mat, V3(wi), rg)
            Ir = (np3(w_r).mean(1) * CO.lighting(kind, np3(wo_r), dvec)).reshape(Gn, ns)
            rv_c = Ic.var(1) / np.maximum(Ic.mean(1) ** 2, 1e-30)
            rv_r = Ir.var(1) / np.maximum(Ir.mean(1) ** 2, 1e-30)
            ratios[kind] = (rv_c / np.maximum(rv_r, 1e-30)).tolist()
        allr = np.concatenate([np.array(v) for v in ratios.values()])
        out = {'median_ratio': float(np.median(allr)), 'p90_ratio': float(np.quantile(allr, .9)),
               'by_lighting_median': {k: float(np.median(v)) for k, v in ratios.items()}, 'note': 'report only in WL81 (no gate)'}
        out['PASS'] = out['median_ratio'] <= U['median_ratio_max'] and out['p90_ratio'] <= U['p90_ratio_max']
        return out
    return sampling_utility


def fixed_states():
    """Synthetic closures with every sampled lobe alpha >= 0.05 (C5b supplement; tests the shared sampler implementation)."""
    cases = [({1: 1.0}, {1: (0.3, 0.3)}), ({1: 1.0}, {1: (0.3, 0.1)}), ({0: 0.5, 1: 0.5}, {1: (0.3, 0.3)}),
             ({2: 0.5, 5: 0.5}, {2: (0.05, 0.05), 5: (0.3, 0.3)}), ({3: 1.0}, {3: (0.4, 0.1)}),
             ({0: 0.3, 1: 0.3, 4: 0.2, 6: 0.2}, {1: (0.08, 0.08), 4: (0.2, 0.12), 6: (0.5, 0.5)})]
    out = []
    for lam, al in cases:
        p = {'lam': torch.zeros(1, 3, 8), 'au': torch.full((1, 7), 0.3), 'av': torch.full((1, 7), 0.3), 'f0': torch.full((1, 7), 0.5)}
        for r, l in lam.items():
            p['lam'][0, :, r] = l
        for r, (a, b) in al.items():
            p['au'][0, r - 1] = a; p['av'][0, r - 1] = b
        out.append(C.compiled_state(p).numpy()[0])
    return np.stack(out).astype(np.float32)


def run_arm(arm, tabs, util_tab, util_desc, data, out):
    os.makedirs(out, exist_ok=True)
    CO.states_for = lambda fit_dir, mats: tabs
    CO.sampling_utility = utility_factory(util_tab, util_desc)
    CO.run(None, data, out)
    r = json.load(open(os.path.join(out, 'stage3.json')))
    rng = np.random.default_rng(SD['grazing'])
    r['G81'] = {}
    for name, tab in tabs.items():
        r['G81'][name] = g81(tab, rng)
        print(arm, name, 'G81', json.dumps(r['G81'][name]), flush=True)
    checks = dict(r['verdict']['checks'])
    # C5b eligibility (WL80C: closures whose sampled lobes all have alpha >= 0.05) can be empty for sharp materials; the chi-square
    # test then cannot be evaluated. Pre-freeze rule: if fewer than 20 eligible cases exist, C5b is also run on fixed synthetic closures
    # (fixed_states) with the same runtime; C5b passes iff the supplement passes and any evaluable real cases pass.
    if r['C5b']['cases'] < 20:
        sup = CO.chi2_tests({'fixed': fixed_states()}, np.random.default_rng(SD['contract'] + 7))
        r['C5b_supplement'] = sup
        checks['C5b'] = bool(sup['PASS'] and (r['C5b']['cases'] == 0 or r['C5b']['PASS']))
    if arm == 'candidate':
        # protocol C3_arm_rule: the candidate's C3 is the sampler albedo test only (its construction is not the sum bound)
        checks['C3'] = all(r[n]['C3']['violations'] == 0 for n in tabs)
        checks['C3_rule'] = 'candidate: sampler albedo test only (sum bound not applicable)'
    checks['G81'] = all(v['PASS'] for v in r['G81'].values())
    r['verdict_wl81'] = {'checks': checks, 'PHYSICAL_CONTRACT_PASS': bool(all(v for k, v in checks.items() if k != 'C3_rule')),
                         'max_sum_lambda': {n: r[n]['max_sum_lambda'] for n in tabs}}
    json.dump(r, open(os.path.join(out, 'contract.json'), 'w'), indent=1)
    print(arm, json.dumps(r['verdict_wl81']), flush=True)
    return r['verdict_wl81']


if __name__ == '__main__':
    stage, data, src, out = sys.argv[1:5]
    mats = np.load(os.path.join(data, 'materials.npz'))
    res = {}
    t0 = time.time()
    if stage == 'stageC':
        for arm in ['control', 'candidate']:
            tabs = {'gate_direct': np.load(os.path.join(src, f'states_{arm}_gate.npy')), 'adversarial': adversarial(arm)}
            res[arm] = run_arm(arm, tabs, tabs['gate_direct'], mats['gate'], data, os.path.join(out, arm))
    else:
        seeds = P['fitting']['compiler']['seeds']
        for arm in [a for a in sys.argv[5:] if not a.startswith('--')]:
            tabs = {f's{s}_{n}': np.load(os.path.join(src, f'states_{arm}_s{s}_{n}.npy')) for s in seeds for n in ['test_iid', 'test_block']}
            tabs['adversarial'] = adversarial(arm)
            res[arm] = run_arm(arm, tabs, tabs[f's{seeds[0]}_test_iid'], mats['test_iid'], data, os.path.join(out, arm))
    res['seconds'] = time.time() - t0
    json.dump(res, open(os.path.join(out, f'{stage}_contract.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))
    os._exit(0)
