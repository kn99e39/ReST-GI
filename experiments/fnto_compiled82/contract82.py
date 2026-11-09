"""WL82 focused tests (pre-evidence, synthetic / development data) and physical contract (fitted closures).

  python contract82.py tests <data82> <w81data> <out> [--dev]
  python contract82.py contract <data82> <fitdir> <out> <steps_json> [--dev]

tests:    T0 nesting / equivalence (C0 == WL81 closure81; E(s=1) == C0; ER(eta=1) == E; Einc(s=1) == C0), T5 sampler / pdf consistency and
          chi-square (synthetic closures), T6 refracted-lobe solid-angle identity, T8 fixed-seed fit reproducibility, T9 split separation,
          T10 oracle decomposition identity (synthetic).
contract: on the primary direct fits of every arm (gate, test_iid, test_block, real) + 50 adversarial closures per arm:
          T1 reciprocity, T2 non-negativity / finiteness incl. the grazing limit, T3 certificate conditions (cert arms), T4 own-sampler
          albedo audit at near-grazing and random incidence (cert arms: zero violations required; unbounded arms: report only).
"""
import json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import closure82 as CL  # noqa: E402
K = CL.K
P82 = json.load(open(os.path.join(HERE, 'protocol', 'compiled82.json')))
SD = P82['seeds']
T = P82['tests']
dev_t = torch.device('cuda')
DEV = '--dev' in sys.argv


def rand_dirs(n, g, mu_lo=0.0, dtype=torch.float64):
    u = torch.rand(n, 2, generator=g, device=dev_t, dtype=dtype)
    mu = mu_lo + (1 - mu_lo) * u[:, 0]
    st = torch.sqrt((1 - mu * mu).clamp_min(0)); ph = 2 * math.pi * u[:, 1]
    return torch.stack([st * torch.cos(ph), st * torch.sin(ph), mu], 1)


def rand_params(arm, mode, n, g, dtype=torch.float64):
    """Random closures with every key (decoded as ER; the C0 / E / Einc evaluations ignore the keys they do not use)."""
    z = torch.randn(n, CL.N_RAW, generator=g, device=dev_t, dtype=dtype) * 1.5
    return CL.decode('ER', mode, z)


def to(p, dtype):
    return {k: v.to(dtype) for k, v in p.items()}


# ---------------------------------------------------------------- T0 nesting / equivalence
def t0(g):
    n = 1 << 18
    r = {}
    for mode in ['cert', 'unb']:
        pe = rand_params('ER', mode, n, g)
        wi, wo = rand_dirs(n, g, 1e-3), rand_dirs(n, g, 1e-3)
        c0 = CL.eval_closure('C0', pe, wi, wo)
        k81 = K.eval_closure({k: pe[k] for k in ['lam', 'au', 'av', 'f0']}, wi, wo)
        p1 = dict(pe); p1['s'] = torch.ones_like(pe['s'])
        e1 = CL.eval_closure('E', p1, wi, wo); i1 = CL.eval_closure('Einc', p1, wi, wo)
        pr = dict(pe); pr['eta'] = torch.ones_like(pe['eta'])
        er1 = CL.eval_closure('ER', pr, wi, wo); e = CL.eval_closure('E', pe, wi, wo)
        rel = lambda a, b: float(((a - b).abs() / b.abs().clamp_min(1e-12)).max())
        r[mode] = {'C0_vs_closure81': rel(c0, k81), 'E_s1_vs_C0': rel(e1, c0), 'Einc_s1_vs_C0': rel(i1, c0), 'ER_eta1_vs_E': rel(er1, e)}
    r['PASS'] = all(v <= T['nesting_rel'] for m in ['cert', 'unb'] for v in r[m].values())
    return r


# ---------------------------------------------------------------- sampler helpers
def own_albedo(arm, p, cidx, wi, ns, g, chunk=1 << 22):
    """Own-sampler albedo (mean of eval / pdf over ns samples) per row (closure cidx[r], incidence wi[r]) -> a, se [R,3] (float64)."""
    R = len(cidx)
    a = torch.zeros(R, 3, dtype=torch.float64, device=dev_t); se = torch.zeros_like(a)
    rows = max(1, chunk // ns)
    for r0 in range(0, R, rows):
        r1 = min(R, r0 + rows)
        idx = cidx[r0:r1].repeat_interleave(ns)
        WI = wi[r0:r1].repeat_interleave(ns, 0)
        pl = CL.gather(p, idx)
        u = torch.rand(len(idx), 3, generator=g, device=dev_t, dtype=WI.dtype)
        wo, ok, _ = CL.sample_closure(arm, pl, WI, u[:, 0], u[:, 1], u[:, 2])
        pdf = CL.pdf_closure(arm, pl, WI, wo)
        ev = CL.eval_closure(arm, pl, WI, wo)
        wt = torch.where((ok & (pdf > 0))[:, None], ev / pdf.clamp_min(1e-30)[:, None], torch.zeros_like(ev)).double().view(r1 - r0, ns, 3)
        a[r0:r1] = wt.mean(1); se[r0:r1] = wt.std(1) / math.sqrt(ns)
    return a, se


def quad_albedo(arm, p, cidx, wi, nq, g, chunk=1 << 22):
    """Uniform-hemisphere quadrature: albedo, pdf integral (with standard errors) per row."""
    R = len(cidx)
    a = torch.zeros(R, 3, dtype=torch.float64, device=dev_t); sa = torch.zeros_like(a)
    pi_ = torch.zeros(R, dtype=torch.float64, device=dev_t); spi = torch.zeros_like(pi_)
    rows = max(1, chunk // nq)
    for r0 in range(0, R, rows):
        r1 = min(R, r0 + rows)
        idx = cidx[r0:r1].repeat_interleave(nq)
        WI = wi[r0:r1].repeat_interleave(nq, 0)
        pl = CL.gather(p, idx)
        wo = rand_dirs(len(idx), g, 0.0, WI.dtype)
        ev = (2 * math.pi * CL.eval_closure(arm, pl, WI, wo)).double().view(r1 - r0, nq, 3)
        pd = (2 * math.pi * CL.pdf_closure(arm, pl, WI, wo)).double().view(r1 - r0, nq)
        a[r0:r1] = ev.mean(1); sa[r0:r1] = ev.std(1) / math.sqrt(nq)
        pi_[r0:r1] = pd.mean(1); spi[r0:r1] = pd.std(1) / math.sqrt(nq)
    return a, sa, pi_, spi


def synthetic(arm, dtype=torch.float32):
    """Fixed synthetic closures (sampled lobes alpha >= 0.1) for the chi-square and consistency tests."""
    S = []
    for j, (lam, al, s, eta) in enumerate(T['synthetic']):
        p = {'lam': torch.zeros(1, 3, 8), 'au': torch.full((1, 7), 0.3), 'av': torch.full((1, 7), 0.3), 'f0': torch.full((1, 7), 0.6),
             's': torch.tensor(s, dtype=torch.float32).view(1, 1, CL.NK).repeat(1, 3, 1), 'eta': torch.tensor([eta])}
        for r, l in lam.items():
            p['lam'][0, :, int(r)] = l
        for r, (a, b) in al.items():
            p['au'][0, int(r) - 1] = a; p['av'][0, int(r) - 1] = b
        S.append(p)
    return {k: torch.cat([q[k] for q in S]).to(dev_t, dtype) for k in S[0]}


def t5(g):
    """Sampler vs quadrature (valid fraction = pdf integral; sampler albedo = quadrature albedo) on synthetic + random closures; chi-square."""
    from scipy.stats import chi2 as CHI2
    out = {}
    ths = np.radians(T['sampler_theta_deg'])
    for arm in ['C0', 'E', 'Einc', 'ER']:
        ps = synthetic(arm)
        pr = to(rand_params(arm, 'cert', 32, g), torch.float32)
        # sampled lobes alpha >= 0.1: the uniform-quadrature standard error is unreliable for near-delta lobes (heavy-tailed integrand;
        # dev smoke 2026-11-10: C0 z 3.6 vs Einc z 12 on identical sampler code), which would test the estimator, not the sampler
        pr['au'] = pr['au'].clamp_min(0.1); pr['av'] = pr['av'].clamp_min(0.1)
        p = {k: torch.cat([ps[k], pr[k]]) for k in ps}
        M = p['lam'].shape[0]
        cidx = torch.arange(M, device=dev_t).repeat_interleave(len(ths))
        th = torch.tensor(np.tile(ths, M), device=dev_t, dtype=torch.float32)
        ph = torch.rand(len(th), generator=g, device=dev_t) * 2 * math.pi
        wi = torch.stack([torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), torch.cos(th)], 1)
        ns = T['sampler_samples']
        # valid fraction from the sampler
        vf = torch.zeros(len(cidx), dtype=torch.float64, device=dev_t)
        for r in range(len(cidx)):
            idx = cidx[r].repeat(ns)
            u = torch.rand(ns, 3, generator=g, device=dev_t)
            _, ok, _ = CL.sample_closure(arm, CL.gather(p, idx), wi[r].expand(ns, 3), u[:, 0], u[:, 1], u[:, 2])
            vf[r] = ok.double().mean()
        a_s, se_s = own_albedo(arm, p, cidx, wi, ns, g)
        a_q, se_q, pint, spint = quad_albedo(arm, p, cidx, wi, T['quad_points'], g)
        za = ((a_s - a_q).abs() - T['abs_slack']).clamp_min(0) / torch.sqrt(se_s ** 2 + se_q ** 2).clamp_min(1e-12)
        sev = torch.sqrt(vf * (1 - vf) / ns + spint ** 2)
        zv = ((vf - pint).abs() - T['abs_slack']).clamp_min(0) / sev.clamp_min(1e-12)
        r = {'cases': len(cidx), 'max_z_albedo': float(za.max()), 'max_z_valid_fraction': float(zv.max()),
             'max_sampler_albedo': float(a_s.max())}
        # chi-square on the synthetic closures only
        pv = []
        nb_t, nb_p = T['chi2_bins']
        for c in range(ps['lam'].shape[0]):
            for t in np.radians(T['chi2_theta_deg']):
                w1 = torch.tensor([[math.sin(t), 0.0, math.cos(t)]], device=dev_t)
                n = T['chi2_samples']
                idx = torch.full((n,), c, device=dev_t, dtype=torch.long)
                u = torch.rand(n, 3, generator=g, device=dev_t)
                wo, ok, _ = CL.sample_closure(arm, CL.gather(ps, idx), w1.expand(n, 3), u[:, 0], u[:, 1], u[:, 2])
                wo = wo[ok]
                bt = (wo[:, 2].clamp(0, 1 - 1e-7) * nb_t).long(); bp = ((torch.atan2(wo[:, 1], wo[:, 0]) % (2 * math.pi)) / (2 * math.pi) * nb_p).long().clamp(0, nb_p - 1)
                obs = torch.bincount(bt * nb_p + bp, minlength=nb_t * nb_p).double()
                nq = T['chi2_quad']
                idq = torch.full((nq,), c, device=dev_t, dtype=torch.long)
                wq = rand_dirs(nq, g, 0.0, torch.float32)
                pdq = CL.pdf_closure(arm, CL.gather(ps, idq), w1.expand(nq, 3), wq).double()
                bq = (wq[:, 2].clamp(0, 1 - 1e-7) * nb_t).long() * nb_p + ((torch.atan2(wq[:, 1], wq[:, 0]) % (2 * math.pi)) / (2 * math.pi) * nb_p).long().clamp(0, nb_p - 1)
                exp = torch.bincount(bq, weights=pdq, minlength=nb_t * nb_p) * (2 * math.pi / nq) * n
                o, e = obs.cpu().numpy(), exp.cpu().numpy()
                big = e >= 5
                stat = float((((o[big] - e[big]) ** 2) / e[big]).sum() + ((o[~big].sum() - e[~big].sum()) ** 2) / max(e[~big].sum(), 1e-9) * (e[~big].sum() >= 5))
                dof = int(big.sum() + (e[~big].sum() >= 5))
                pv.append(float(CHI2.sf(stat, dof)))
        r['chi2_min_p'] = float(min(pv)); r['chi2_cases'] = len(pv)
        r['PASS'] = r['max_z_albedo'] <= T['z_max'] and r['max_z_valid_fraction'] <= T['z_max'] and r['chi2_min_p'] >= T['chi2_min_p']
        out[arm] = r
    out['PASS'] = all(out[a]['PASS'] for a in ['C0', 'E', 'Einc', 'ER'])
    return out


def t6(g):
    """Refracted lobe (weight 1, s = 1): outside albedo (uniform quadrature of eval) == inside-cone VNDF integral E[F G2/G1 1(exit)]."""
    res = []
    for al in T['jacobian_alpha']:
        for eta in T['jacobian_eta']:
            for thd in T['jacobian_theta_deg']:
                p = {'lam': torch.zeros(1, 3, 8, device=dev_t, dtype=torch.float64), 'au': torch.full((1, 7), al, device=dev_t, dtype=torch.float64),
                     'av': torch.full((1, 7), al, device=dev_t, dtype=torch.float64), 'f0': torch.full((1, 7), 0.9, device=dev_t, dtype=torch.float64),
                     's': torch.ones(1, 3, CL.NK, device=dev_t, dtype=torch.float64), 'eta': torch.tensor([eta], device=dev_t, dtype=torch.float64)}
                p['lam'][0, :, 5] = 1.0                       # inner GGX lobe k = 4
                t = math.radians(thd)
                wi = torch.tensor([[math.sin(t), 0.0, math.cos(t)]], device=dev_t, dtype=torch.float64)
                n = T['jacobian_samples']
                a_q, se_q, _, _ = quad_albedo('ER', p, torch.zeros(1, dtype=torch.long, device=dev_t), wi, n, g)
                # inside: h ~ VNDF(t(wi)), wo' = reflect; integrand F * G2/G1(wi') * 1(wo' exits)
                w_in = CL.refract_up(wi, p['eta'])
                u = torch.rand(n, 2, generator=g, device=dev_t, dtype=torch.float64)
                h = CL.vndf_h(torch.full((n,), al, device=dev_t, dtype=torch.float64), torch.full((n,), al, device=dev_t, dtype=torch.float64),
                              w_in.expand(n, 3), u[:, 0], u[:, 1])
                wr = 2 * (w_in * h).sum(1, keepdim=True) * h - w_in
                _, ex = CL.exit_up(wr, p['eta'].expand(n))
                ci, co = w_in[:, 2], wr[:, 2].clamp_min(0)
                si = torch.sqrt(ci ** 2 + al ** 2 * (w_in[:, 0] ** 2 + w_in[:, 1] ** 2))
                so = torch.sqrt(co ** 2 + al ** 2 * (wr[:, 0] ** 2 + wr[:, 1] ** 2))
                ratio = (ci + si) * co / (co * si + ci * so).clamp_min(1e-30)
                F = 0.9 + 0.1 * (1 - (w_in * h).sum(1).clamp(0, 1)) ** 5
                v = torch.where(ex & (wr[:, 2] > 1e-7), F * ratio, torch.zeros_like(ratio))
                a_in, se_in = float(v.mean()), float(v.std() / math.sqrt(n))
                z = max(0.0, abs(float(a_q[0, 0]) - a_in) - T['abs_slack']) / math.sqrt(float(se_q[0, 0]) ** 2 + se_in ** 2)
                res.append({'alpha': al, 'eta': eta, 'theta': thd, 'outside': float(a_q[0, 0]), 'inside': a_in, 'z': z})
    return {'cases': res, 'max_z': max(r['z'] for r in res), 'PASS': max(r['z'] for r in res) <= T['z_max']}


def t8(data):
    """Fixed-seed fit reproducibility (16 gate materials, 300 steps, E_cert and ER_cert metric): bit-identical raw parameters."""
    import fit82 as FT
    Tf = FT.load_fit(os.path.join(data, 'fit82_gate.npz'))
    nm = 16
    L = nm * Tf['n_wi'] * Tf['n_wo']
    Ts = dict(Tf); Ts.update({k: Tf[k][:L] for k in ['m', 'wi', 'wo', 'f', 'w']}); Ts['n_mat'] = nm
    r = {}
    for arm in ['E', 'ER']:
        zs = []
        for _ in range(2):
            torch.manual_seed(SD['fit_init'])
            z, _, _ = FT.fit(arm, 'cert', 'metric', CL.init_raw(arm, 'cert', nm, dev_t), Ts, 300, log_every=1000)
            zs.append(z)
        r[arm] = {'bit_identical': bool(torch.equal(zs[0], zs[1])), 'max_abs_diff': float((zs[0] - zs[1]).abs().max())}
    r['PASS'] = all(v['bit_identical'] for v in r.values())
    return r


def t9(w81):
    m = np.load(os.path.join(w81, 'materials.npz'))
    sets = {k: m[k] for k in ['train', 'val', 'gate', 'test_iid', 'test_block']}
    sets['real'] = np.array([x['descriptor'] for x in P82['real_materials']['list']])
    over = {}
    keys = list(sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            A = {tuple(np.round(x, 12)) for x in sets[a]}
            over[f'{a}|{b}'] = int(sum(tuple(np.round(x, 12)) in A for x in sets[b]))
    seeds82 = {k: v for k, v in SD.items()}
    seeds81 = json.load(open(os.path.join(HERE, '..', 'fnto_compiled81', 'protocol', 'compiled81.json')))['seeds']
    flat81 = {v for v in seeds81.values() if isinstance(v, int)}
    clash = [k for k, v in seeds82.items() if isinstance(v, int) and v in flat81]
    r = {'material_overlaps': over, 'seed_clash_with_wl81': clash}
    r['PASS'] = all(v == 0 for v in over.values()) and not clash
    return r


def t10(g):
    import eval82 as E82
    G, n = 64, 512
    pred = torch.rand(G, n, 3, generator=g, device=dev_t, dtype=torch.float64) * (torch.rand(G, 1, 3, generator=g, device=dev_t, dtype=torch.float64) + 0.01)
    f = torch.rand(G, n, 3, generator=g, device=dev_t, dtype=torch.float64)
    pred[0, :, 2] = 0                               # a zero-albedo closure channel
    w = torch.rand(G, n, 1, generator=g, device=dev_t, dtype=torch.float64) + 0.1
    gm, _, _ = E82.group_metrics(pred, f, w, torch.ones(G, 3, dtype=torch.float64, device=dev_t))
    d = float(np.abs(gm['el1_same'] - gm['eshape']).max()); z = float(gm['ealb_same'][1:].max())   # E_alb = 0 where every closure channel has albedo > 0
    return {'max_abs_el1_same_minus_eshape': d, 'max_ealb_same': z, 'zero_channels_counted': float(gm['zero_channels'][0]),
            'PASS': bool(d <= 1e-12 and z <= 1e-12 and gm['zero_channels'][0] == 1)}


def tests(data, w81, out):
    os.makedirs(out, exist_ok=True)
    g = torch.Generator(device=dev_t); g.manual_seed(SD['tests'])
    r = {}
    t0_ = time.time()
    for name, fn in [('T0_nesting', lambda: t0(g)), ('T10_oracle_identity', lambda: t10(g)), ('T9_split', lambda: t9(w81)),
                     ('T6_refracted_jacobian', lambda: t6(g)), ('T5_sampler_pdf', lambda: t5(g)), ('T8_reproducibility', lambda: t8(data))]:
        r[name] = fn()
        print(name, bool(r[name]['PASS']), round(time.time() - t0_, 1), 's', flush=True)
    r['PASS'] = all(v['PASS'] for v in r.values())
    r['seconds'] = time.time() - t0_
    json.dump(r, open(os.path.join(out, 'tests.json'), 'w'), indent=1, default=float)
    print('TESTS', 'PASS' if r['PASS'] else 'FAIL', flush=True)


# ---------------------------------------------------------------- physical contract on fitted closures
def adversarial(arm, mode):
    g = torch.Generator().manual_seed(SD['adversarial'])
    na = 50
    z = torch.randn(na, CL.N_RAW, generator=g) * 3.0
    z[: na // 4, :21] = torch.sign(z[: na // 4, :21]) * 12.0
    return CL.decode(arm, mode, z.to(dev_t))


def contract(data, fitdir, out, steps):
    os.makedirs(out, exist_ok=True)
    g = torch.Generator(device=dev_t); g.manual_seed(SD['contract'])
    C = P82['contract']
    res = {}
    sets = ['gate', 'test_iid', 'test_block', 'real']
    for tagname in P82['arms_contract']:
        arm, mode, obj = tagname.split('_')
        st = steps.get(f'{arm}_{mode}_{obj}', P82['fitting']['steps_primary'])
        ps = []
        names = []
        for s in sets:
            fn = os.path.join(fitdir, f'fit_{s}_{arm}_{mode}_{obj}_{st}_generic.npz')
            if os.path.exists(fn):
                ps.append(CL.decode(arm, mode, torch.from_numpy(np.load(fn)['z']).to(dev_t))); names.append(s)
        ps.append(adversarial(arm, mode)); names.append('adversarial')
        p = {k: torch.cat([q[k] for q in ps]) for k in ps[0]}
        M = p['lam'].shape[0]
        r = {'closures': {n: int(q['lam'].shape[0]) for n, q in zip(names, ps)}}
        # T1 reciprocity, T2 non-negativity / finiteness (float64), incl. grazing incidences
        pd = to(p, torch.float64)
        n = C['pairs']
        idx = torch.randint(0, M, (n,), generator=g, device=dev_t)
        wi, wo = rand_dirs(n, g, 1e-3), rand_dirs(n, g, 1e-3)
        a = CL.eval_closure(arm, CL.gather(pd, idx), wi, wo) / wo[:, 2:3]
        b = CL.eval_closure(arm, CL.gather(pd, idx), wo, wi) / wi[:, 2:3]
        sc = torch.maximum(a.abs(), b.abs())
        m = sc > C['recip_floor']
        r['T1_reciprocity'] = {'max_rel': float(((a - b).abs()[m] / sc[m]).max()) if m.any() else 0.0}
        r['T1_reciprocity']['PASS'] = r['T1_reciprocity']['max_rel'] <= C['recip_max_rel'] if arm != 'Einc' else None
        mus = torch.tensor([0.0, 1e-12, 2e-7, 1e-4], device=dev_t, dtype=torch.float64)
        wg = rand_dirs(n, g, 0.0)
        mu = mus[torch.randint(0, 4, (n,), generator=g, device=dev_t)]
        st_ = torch.sqrt(1 - mu * mu)
        phi = torch.atan2(wg[:, 1], wg[:, 0])
        wgz = torch.stack([st_ * torch.cos(phi), st_ * torch.sin(phi), mu], 1)
        e1 = CL.eval_closure(arm, CL.gather(pd, idx), wgz, wo); e2 = CL.eval_closure(arm, CL.gather(pd, idx), wo, wgz)
        r['T2_nonneg_finite'] = {'min_eval': float(min(a.min(), e1.min(), e2.min())), 'all_finite': bool(torch.isfinite(torch.cat([a, b, e1, e2])).all()),
                                 'max_eval_grazing_wi': float(e1.max()), 'max_f_grazing_wo': float((e2 / wo[:, 2:3].clamp_min(1e-300)).max())}
        r['T2_nonneg_finite']['PASS'] = r['T2_nonneg_finite']['min_eval'] >= 0 and r['T2_nonneg_finite']['all_finite']
        # T3 certificate conditions (cert arms)
        if mode == 'cert':
            cc = {'max_sum_lambda': float(p['lam'].sum(2).max())}
            if 's' in p:
                cc['s_min'] = float(p['s'].min()); cc['s_max'] = float(p['s'].max())
            if 'eta' in p:
                cc['eta_min'] = float(p['eta'].min()); cc['eta_max'] = float(p['eta'].max())
            cc['PASS'] = cc['max_sum_lambda'] <= 1 + 1e-6 and cc.get('s_min', 0) >= 0 and cc.get('s_max', 1) <= 1 and cc.get('eta_min', 1) >= 1
            r['T3_certificate'] = cc
        # T4 own-sampler albedo audit: fixed near-grazing mu x 4 azimuths + random incidences
        mu_f = torch.tensor(C['audit_mu'], device=dev_t)
        nf = len(C['audit_mu']) * 4
        nr = C['audit_random_wi']
        rows_c = torch.arange(M, device=dev_t).repeat_interleave(nf + nr)
        muv = torch.cat([mu_f.repeat_interleave(4)[None].expand(M, nf),
                         torch.cat([torch.exp(torch.empty(M, nr // 2, device=dev_t).uniform_(math.log(2e-7), math.log(1e-2), generator=g)),
                                    torch.empty(M, nr - nr // 2, device=dev_t).uniform_(2e-7, 1.0, generator=g)], 1)], 1).reshape(-1)
        phv = torch.rand(len(muv), generator=g, device=dev_t) * 2 * math.pi
        stv = torch.sqrt((1 - muv * muv).clamp_min(0))
        wa = torch.stack([stv * torch.cos(phv), stv * torch.sin(phv), muv], 1)
        ns = C['audit_samples'] if mode == 'cert' else C['audit_samples_unb']
        if DEV:
            ns = min(ns, 4096)
        aa, se = own_albedo(arm, p, rows_c, wa, ns, g)
        viol = (aa - 1 > torch.clamp(4 * se, min=1e-3)).any(1)
        per_closure_max = aa.max(1).values.view(M, nf + nr).max(1).values
        r['T4_albedo_audit'] = {'rows': int(len(rows_c)), 'samples': ns, 'violations': int(viol.sum()), 'max_albedo': float(aa.max()),
                                'frac_closures_max_albedo_gt_1': float((per_closure_max > 1).float().mean()),
                                'albedo_at_mu_2e-7_max': float(aa.view(M, nf + nr, 3)[:, : nf][:, [i for i in range(nf) if C['audit_mu'][i // 4] <= 2e-7]].max())}
        r['T4_albedo_audit']['PASS'] = r['T4_albedo_audit']['violations'] == 0 if mode == 'cert' else None
        checks = [r['T2_nonneg_finite']['PASS']]
        if arm != 'Einc':
            checks.append(r['T1_reciprocity']['PASS'])
        if mode == 'cert':
            checks += [r['T3_certificate']['PASS'], r['T4_albedo_audit']['PASS']]
        r['PHYSICAL_CONTRACT_PASS'] = bool(all(checks)) if mode == 'cert' else None
        r['report_only_reason'] = None if mode == 'cert' else 'unbounded / non-reciprocal diagnostic arm: physical validity is not claimed'
        res[tagname] = r
        print(tagname, json.dumps({k: (v.get('PASS') if isinstance(v, dict) else v) for k, v in r.items()}), flush=True)
    json.dump(res, open(os.path.join(out, 'contract.json'), 'w'), indent=1, default=float)


if __name__ == '__main__':
    if sys.argv[1] == 'tests':
        tests(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        contract(sys.argv[2], sys.argv[3], sys.argv[4], json.loads(sys.argv[5]))
    os._exit(0)
