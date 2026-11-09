"""WL80C Stage 3: Evaluate / Sample / PDF contract tests on the Dr.Jit runtime
closure, plus the sampling-utility comparison against the reference sampler.

  python contract80c.py <data_dir> <fit_dir> <out_dir> [--dev]
"""
import json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
import ref80c as R
mi = R.mi_init()
import drjit as dr
import closure80c as C
import closure80c_dr as CD
from data80c import normalise, P

dev_t = torch.device('cuda')
CT = P['contract']


def states_for(fit_dir, mats):
    T = P['training']
    out = {}
    for s in P['seeds']['train']:
        net = C.Compiler(T['width'], T['depth']).to(dev_t)
        net.load_state_dict(torch.load(os.path.join(fit_dir, f'compiler_s{s}.pt'), map_location=dev_t)['state_dict'])
        for name in ['test_iid', 'test_block']:
            with torch.no_grad():
                p = C.decode(net(torch.from_numpy(normalise(mats[name])).to(dev_t)))
                out[f's{s}_{name}'] = C.compiled_state(p).cpu().numpy()
    g = torch.Generator().manual_seed(P['seeds']['adversarial'])
    na = CT['n_adversarial']
    z = torch.randn(na, C.N_RAW, generator=g) * CT['adversarial_sigma']
    z[: na // 4, :3 * C.N_GGX] = torch.sign(z[: na // 4, :3 * C.N_GGX]) * 12.0   # saturated alpha / F0
    out['adversarial'] = C.compiled_state(C.decode(z)).numpy()
    return out


def V3(a):
    a = np.ascontiguousarray(a, np.float32)
    return mi.Vector3f(mi.Float(a[:, 0]), mi.Float(a[:, 1]), mi.Float(a[:, 2]))


def np3(v):
    return np.stack([np.asarray(v[k]) for k in range(3)], 1).astype(np.float64)


def closure_lanes(tab, idx):
    return CD.DrClosure(tab, mi.UInt32(np.ascontiguousarray(idx, np.uint32)))


def unit(rng, n, upper=None):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    if upper is True:
        v[:, 2] = np.abs(v[:, 2])
    return v


def lighting(kind, wo, d):
    if kind == 'env_lobe':
        return 1.0 + 4.0 * np.maximum(0, (wo * d).sum(1)) ** 8
    if kind == 'sky':
        return 0.2 + wo[:, 2] ** 2
    if kind == 'sun64':
        return 0.05 + 20.0 * np.maximum(0, (wo * d).sum(1)) ** 64
    raise ValueError(kind)


def run(fit_dir, data, out):
    t0 = time.time()
    mats = np.load(os.path.join(data, 'materials.npz'))
    rng = np.random.default_rng(P['seeds']['contract'])
    tabs = states_for(fit_dir, mats)
    res = {}
    for name, tab in tabs.items():
        M = len(tab)
        r = {'n_closures': M}
        # analytic passivity bound
        lam = tab[:, :24].reshape(M, 3, 8)
        r['max_sum_lambda'] = float(lam.sum(2).max())
        # C1: finite / nonneg / invalid inputs
        n = CT['c1_lanes']
        idx = rng.integers(0, M, n)
        wi = unit(rng, n); wo = unit(rng, n)
        k = n // 16
        wo[:k] = -wi[:k]                       # degenerate half vector
        wi[k:2 * k, 2] = 0.0; wi[k:2 * k] /= np.linalg.norm(wi[k:2 * k], axis=1, keepdims=True)
        wo[2 * k:3 * k] = wi[2 * k:3 * k]      # wi = wo
        wi[3 * k:4 * k] = [0, 0, 1]; wo[3 * k:4 * k] = [0, 0, 1]
        wi[4 * k:5 * k, 2] = 1e-8
        cl = closure_lanes(tab, idx)
        f, pdf = cl.eval_pdf(V3(wi), V3(wo))
        f = np3(f); pdf = np.asarray(pdf, np.float64)
        below = (wi[:, 2] <= 0) | (wo[:, 2] <= 0)
        u1 = rng.random(n); u2 = rng.random((n, 2))
        wos, ps, ws, vs = cl.sample(V3(wi), mi.Float(u1.astype(np.float32)),
                                     mi.Point2f(mi.Float(u2[:, 0].astype(np.float32)), mi.Float(u2[:, 1].astype(np.float32))))
        wos = np3(wos); ps = np.asarray(ps, np.float64); ws = np3(ws); vs = np.asarray(vs)
        r['C1'] = {'nonfinite_eval': int((~np.isfinite(f)).sum()), 'negative_eval': int((f < 0).sum()),
                   'nonfinite_pdf': int((~np.isfinite(pdf)).sum()), 'negative_pdf': int((pdf < 0).sum()),
                   'below_horizon_nonzero': int(((f.max(1) != 0) | (pdf != 0))[below].sum()),
                   'sample_nonfinite': int((~np.isfinite(ws)).sum() + (~np.isfinite(ps)).sum()),
                   'sample_negative_weight': int((ws < 0).sum()),
                   'sample_valid_below_horizon': int((vs & (wos[:, 2] <= 0)).sum()),
                   'sample_valid_with_wi_below': int((vs & (wi[:, 2] <= 0)).sum())}
        r['C1']['PASS'] = all(v == 0 for v in r['C1'].values())
        # C5a identity: returned pdf vs pdf(wo), weight vs eval/pdf
        sel = vs
        f2, p2 = cl.eval_pdf(V3(wi), V3(wos))
        f2 = np3(f2)[sel]; p2 = np.asarray(p2, np.float64)[sel]
        rp = np.abs(p2 - ps[sel]) / np.maximum(p2, 1e-30)
        rw = np.abs(ws[sel] - f2 / p2[:, None]) / np.maximum(np.abs(f2 / p2[:, None]), 1e-12)
        r['C5a'] = {'n': int(sel.sum()), 'max_rel_pdf': float(rp.max()) if sel.any() else 0.0,
                    'max_rel_weight': float(rw.max()) if sel.any() else 0.0}
        r['C5a']['PASS'] = r['C5a']['max_rel_pdf'] <= CT['identity_rel'] and r['C5a']['max_rel_weight'] <= CT['identity_rel']
        # C2 reciprocity and C7 support on upper-hemisphere pairs
        wi = unit(rng, n, True); wo = unit(rng, n, True)
        f1, p1 = cl.eval_pdf(V3(wi), V3(wo)); g1 = np3(f1) / wo[:, 2:3]
        f2, _ = cl.eval_pdf(V3(wo), V3(wi)); g2 = np3(f2) / wi[:, 2:3]
        mx = np.maximum(np.abs(g1), np.abs(g2))
        m = (mx > CT['recip_floor']) & (wi[:, 2:3] > 1e-4) & (wo[:, 2:3] > 1e-4)
        rel = (np.abs(g1 - g2) / np.maximum(mx, 1e-30))[m]
        r['C2'] = {'n': int(m.sum()), 'max_rel': float(rel.max()), 'p99_rel': float(np.quantile(rel, .99))}
        r['C2']['PASS'] = r['C2']['max_rel'] <= CT['recip_max_rel']
        p1 = np.asarray(p1, np.float64)
        r['C7'] = {'support_violations': int(((np3(f1).max(1) > 1e-6) & (p1 <= 0)).sum())}
        r['C7']['PASS'] = r['C7']['support_violations'] == 0
        # C3 passivity via the closure's own sampler
        ths = np.radians(np.array(CT['albedo_theta_deg']))
        ns = CT['albedo_samples']
        alb_viol, alb_max, n_cases = 0, 0.0, 0
        for c0 in range(0, M, max(1, (1 << 23) // (len(ths) * ns))):
            cs = np.arange(c0, min(M, c0 + max(1, (1 << 23) // (len(ths) * ns))))
            G = len(cs) * len(ths)
            idx = np.repeat(cs, len(ths) * ns)
            th = np.tile(np.repeat(ths, ns), len(cs))
            wi = R.sph(th, rng.uniform(0, 2 * np.pi, len(idx)))
            u1 = rng.random(len(idx)); u2 = rng.random((len(idx), 2))
            cl = closure_lanes(tab, idx)
            _, _, w, _ = cl.sample(V3(wi), mi.Float(u1.astype(np.float32)),
                                   mi.Point2f(mi.Float(u2[:, 0].astype(np.float32)), mi.Float(u2[:, 1].astype(np.float32))))
            w = np3(w).reshape(G, ns, 3)
            a = w.mean(1); se = w.std(1) / math.sqrt(ns)
            v = (a - 1 > np.maximum(4 * se, 1e-3))
            alb_viol += int(v.any(1).sum()); alb_max = max(alb_max, float(a.max())); n_cases += G
        r['C3'] = {'cases': n_cases, 'violations': alb_viol, 'max_albedo_estimate': alb_max}
        r['C3']['PASS'] = alb_viol == 0 and r['max_sum_lambda'] <= 1 + 1e-6
        # C4 per-lobe pdf normalisation: VNDF failure fraction vs independent NDF integral
        ths = np.radians(np.array(CT['norm_theta_deg']))
        ns = CT['norm_samples']
        zs = []
        csub = np.arange(M)[: CT['norm_max_closures']]
        for kk in range(7):
            for c0 in range(0, len(csub), max(1, (1 << 22) // (len(ths) * ns))):
                cs = csub[c0:c0 + max(1, (1 << 22) // (len(ths) * ns))]
                G = len(cs) * len(ths)
                idx = np.repeat(cs, len(ths) * ns)
                th = np.tile(np.repeat(ths, ns), len(cs))
                wi = R.sph(th, rng.uniform(0, 2 * np.pi, len(idx)))
                u2 = rng.random((len(idx), 2)); uh = rng.random((len(idx), 2))
                cl = closure_lanes(tab, idx)
                fail, g = cl.lobe_vndf_fail_and_norm(kk, V3(wi), mi.Point2f(mi.Float(u2[:, 0].astype(np.float32)), mi.Float(u2[:, 1].astype(np.float32))),
                                                     mi.Point2f(mi.Float(uh[:, 0].astype(np.float32)), mi.Float(uh[:, 1].astype(np.float32))))
                fail = np.asarray(fail).reshape(G, ns).astype(np.float64); g = np.asarray(g, np.float64).reshape(G, ns)
                a = 1 - fail.mean(1); b = g.mean(1)
                se = np.sqrt(fail.var(1) / ns + g.var(1) / ns)
                zs.append(np.abs(a - b) / (se + 1e-12) * (np.abs(a - b) > 1e-3))
        zs = np.concatenate(zs)
        r['C4'] = {'cases': int(len(zs)), 'frac_z_gt4': float(np.mean(zs > 4)), 'max_z': float(zs.max())}
        r['C4']['PASS'] = r['C4']['frac_z_gt4'] <= CT['mc_frac_z4_max']
        # C6 MC estimator consistency: VNDF mixture sampler vs independent NDF mixture sampler
        ths = np.radians(np.array(CT['mc_theta_deg']))
        ns = CT['mc_samples']
        zz, rels = [], []
        csub = np.arange(M)[: CT['mc_max_closures']]
        for kind in CT['lightings']:
            for c0 in range(0, len(csub), max(1, (1 << 22) // (len(ths) * ns))):
                cs = csub[c0:c0 + max(1, (1 << 22) // (len(ths) * ns))]
                G = len(cs) * len(ths)
                idx = np.repeat(cs, len(ths) * ns)
                th = np.tile(np.repeat(ths, ns), len(cs))
                wi = R.sph(th, rng.uniform(0, 2 * np.pi, len(idx)))
                dvec = np.repeat(unit(rng, G, True), ns, 0)
                cl = closure_lanes(tab, idx)
                u1 = rng.random(len(idx)); u2 = rng.random((len(idx), 2))
                wo1, _, w1, v1 = cl.sample(V3(wi), mi.Float(u1.astype(np.float32)),
                                           mi.Point2f(mi.Float(u2[:, 0].astype(np.float32)), mi.Float(u2[:, 1].astype(np.float32))))
                wo1 = np3(wo1); I1 = (np3(w1).mean(1) * lighting(kind, wo1, dvec)).reshape(G, ns)
                u1 = rng.random(len(idx)); u2 = rng.random((len(idx), 2))
                wo2, pa, v2 = cl.alt_sample(V3(wi), mi.Float(u1.astype(np.float32)),
                                            mi.Point2f(mi.Float(u2[:, 0].astype(np.float32)), mi.Float(u2[:, 1].astype(np.float32))))
                f2 = np3(cl.eval(V3(wi), wo2)); wo2 = np3(wo2); pa = np.asarray(pa, np.float64); v2 = np.asarray(v2)
                I2 = np.where(v2, f2.mean(1) / np.where(v2, pa, 1.0), 0.0) * lighting(kind, wo2, dvec)
                I2 = I2.reshape(G, ns)
                m1, m2 = I1.mean(1), I2.mean(1)
                se = np.sqrt(I1.var(1) / ns + I2.var(1) / ns)
                zz.append(np.abs(m1 - m2) / (se + 1e-12)); rels.append((m1 - m2) / np.maximum(m2, 1e-12))
        zz = np.concatenate(zz); rels = np.concatenate(rels)
        tstat = float(rels.mean() / (rels.std() / math.sqrt(len(rels)) + 1e-30))
        r['C6'] = {'cases': int(len(zz)), 'frac_z_gt4': float(np.mean(zz > 4)), 'max_z': float(zz.max()),
                   'mean_rel_diff': float(rels.mean()), 't_mean_rel_diff': tstat}
        r['C6']['PASS'] = r['C6']['frac_z_gt4'] <= CT['mc_frac_z4_max'] and abs(tstat) <= 4
        res[name] = r
        print(name, json.dumps({k: (v['PASS'] if isinstance(v, dict) and 'PASS' in v else v) for k, v in r.items()}), flush=True)
    # C5b chi-square on moderate closures (all lobes with p_r > 0.01 have min alpha >= alpha_min)
    res['C5b'] = chi2_tests(tabs, rng)
    # sampling utility vs the reference sampler
    res['sampling_utility'] = sampling_utility(tabs, mats, rng)
    passes = {k: all(res[n][k]['PASS'] for n in tabs) for k in ['C1', 'C2', 'C3', 'C4', 'C5a', 'C6', 'C7']}
    passes['C5b'] = res['C5b']['PASS']
    res['verdict'] = {'checks': passes, 'PHYSICAL_CONTRACT_PASS': all(passes.values()),
                      'SAMPLING_UTILITY_PASS': res['sampling_utility']['PASS']}
    res['seconds'] = time.time() - t0
    json.dump(res, open(os.path.join(out, 'stage3.json'), 'w'), indent=1)
    print(json.dumps(res['verdict'], indent=1))


def chi2_tests(tabs, rng):
    from scipy.stats import chi2 as chi2d
    X = CT['chi2']
    nth, nph, sub = X['bins_theta'], X['bins_phi'], X['sub']
    pv = []
    cases = 0
    for name, tab in tabs.items():
        M = len(tab)
        au, av, pr = tab[:, 24:31], tab[:, 31:38], tab[:, 45:53]
        amin = np.where(pr[:, 1:] > 0.01, np.minimum(au, av), 9).min(1)
        ok = np.nonzero(amin >= X['alpha_min'])[0][: X['max_closures_per_set']]
        for c in ok:
            for tdeg in X['theta_deg']:
                wi1 = R.sph(np.radians(tdeg), rng.uniform(0, 2 * np.pi))
                N = X['samples']
                cl = CD.DrClosure(tab[c])
                u1 = rng.random(N); u2 = rng.random((N, 2))
                wiN = np.repeat(wi1[None], N, 0)
                wo, _, _, v = cl.sample(V3(wiN), mi.Float(u1.astype(np.float32)),
                                        mi.Point2f(mi.Float(u2[:, 0].astype(np.float32)), mi.Float(u2[:, 1].astype(np.float32))))
                wo = np3(wo); v = np.asarray(v)
                th = np.arccos(np.clip(wo[v, 2], -1, 1)); ph = np.mod(np.arctan2(wo[v, 1], wo[v, 0]), 2 * np.pi)
                obs, _, _ = np.histogram2d(th, ph, bins=[nth, nph], range=[[0, np.pi / 2], [0, 2 * np.pi]])
                # expected: midpoint sub-grid integration of pdf
                tg = (np.arange(nth * sub) + 0.5) * (np.pi / 2) / (nth * sub)
                pg = (np.arange(nph * sub) + 0.5) * (2 * np.pi) / (nph * sub)
                T, PH = np.meshgrid(tg, pg, indexing='ij')
                wq = R.sph(T.ravel(), PH.ravel())
                pq = np.asarray(cl.pdf(V3(np.repeat(wi1[None], len(wq), 0)), V3(wq)), np.float64)
                dA = np.sin(T.ravel()) * (np.pi / 2 / (nth * sub)) * (2 * np.pi / (nph * sub))
                ex = (pq * dA).reshape(nth, sub, nph, sub).sum((1, 3)) * N
                obs = np.append(obs.ravel(), N - v.sum()); ex = np.append(ex.ravel(), max(N - ex.sum(), 0.0))
                small = ex < 5
                o2 = np.append(obs[~small], obs[small].sum()); e2 = np.append(ex[~small], ex[small].sum())
                keep = e2 > 0
                stat = float(((o2[keep] - e2[keep]) ** 2 / e2[keep]).sum())
                pv.append(float(chi2d.sf(stat, keep.sum() - 1)))
                cases += 1
    pv = np.array(pv)
    out = {'cases': cases, 'frac_p_lt_1e-3': float(np.mean(pv < 1e-3)) if cases else None,
           'min_p': float(pv.min()) if cases else None}
    out['PASS'] = cases > 0 and out['frac_p_lt_1e-3'] <= X['max_frac_fail']
    return out


def sampling_utility(tabs, mats, rng):
    """Relative variance per sample of int f cos L: closure sampler + closure eval vs
    reference principled sampler + reference eval (same materials, wi, L)."""
    U = CT['utility']
    s0 = P['seeds']['train'][0]
    tab = tabs[f's{s0}_test_iid']
    d = mats['test_iid']
    bsdf, nb = R.textured_principled(d)
    ths = np.radians(np.array(U['theta_deg']))
    ns = U['samples']
    M = len(tab)
    ratios = {k: [] for k in CT['lightings']}
    for kind in CT['lightings']:
        G = M * len(ths)
        idx = np.repeat(np.arange(M), len(ths) * ns)
        th = np.tile(np.repeat(ths, ns), M)
        phi = np.repeat(rng.uniform(0, 2 * np.pi, G), ns)
        wi = R.sph(th, phi)
        dvec = np.repeat(unit(rng, G, True), ns, 0)
        cl = closure_lanes(tab, idx)
        u1 = rng.random(len(idx)); u2 = rng.random((len(idx), 2))
        wo, _, w, v = cl.sample(V3(wi), mi.Float(u1.astype(np.float32)),
                                mi.Point2f(mi.Float(u2[:, 0].astype(np.float32)), mi.Float(u2[:, 1].astype(np.float32))))
        Ic = (np3(w).mean(1) * lighting(kind, np3(wo), dvec)).reshape(G, ns)
        u1 = rng.random(len(idx)); u2 = rng.random((len(idx), 2))
        wo_r, _, w_r, ok = R.ref_sample(bsdf, nb, idx, wi, u1, u2)
        Ir = (np.where(ok[:, None], w_r, 0).mean(1) * lighting(kind, wo_r.astype(np.float64), dvec)).reshape(G, ns)
        rv_c = Ic.var(1) / np.maximum(Ic.mean(1) ** 2, 1e-30)
        rv_r = Ir.var(1) / np.maximum(Ir.mean(1) ** 2, 1e-30)
        ratios[kind] = (rv_c / np.maximum(rv_r, 1e-30)).tolist()
    allr = np.concatenate([np.array(v) for v in ratios.values()])
    out = {'median_ratio': float(np.median(allr)), 'p90_ratio': float(np.quantile(allr, .9)),
           'by_lighting_median': {k: float(np.median(v)) for k, v in ratios.items()}}
    out['PASS'] = out['median_ratio'] <= U['median_ratio_max'] and out['p90_ratio'] <= U['p90_ratio_max']
    return out


if __name__ == '__main__':
    data, fit_dir, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    run(fit_dir, data, out)
    os._exit(0)
