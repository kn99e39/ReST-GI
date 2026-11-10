"""WL83 runtime correctness (Dr.Jit closure83_dr vs the WL82 torch reference closure82) and the Dr.Jit passivity audit.

  python tests83.py runtime <out> [--dev]                       # R0-R4 on random / synthetic / sharp closures (pre-evidence)
  python tests83.py audit <data83> <fitdir> <out> [--dev]       # R5: own-sampler albedo audit of every DF / L closure + adversarial
  python tests83.py independence <data83> <w81> <out>           # A4: population / seed independence

R0 binding: uniform (opaque scalars) and per-lane table bindings give bit-identical eval / pdf.
R1 eval / pdf vs torch on 2^20 random pairs per arm (mu down to 1e-6): median rel <= 1e-5, p99.9 <= 1e-3 (WL81 B2 tolerances).
R2 sampler: returned pdf == eval_pdf pdf; sampler valid fraction == pdf integral and sampler albedo == quadrature albedo (z <= 5 beyond
   2e-3 slack; synthetic + random closures, sampled lobes alpha >= 0.1); chi-square on synthetic closures, min p >= 1e-4.
R3 reciprocity of the Dr.Jit eval (float32): max rel <= 1e-4 (values > 1e-6, cos >= 1e-3).
R4 finite and non-negative eval / pdf / sample weights at mu in {0, 1e-12, 2e-7, 1e-4}.
R5 (audit) zero violations a - 1 > max(4 se, 1e-3) of the Dr.Jit own-sampler albedo (65536 samples) at mu in WL82 audit_mu x 4 azimuths
   + 16 random incidences, every closure (certificate check of the deployed runtime).
"""
import json, math, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled82'))
sys.path.insert(0, os.path.join(HERE, '..', 'fnto_compiled81'))
import closure82 as CL  # noqa: E402
import closure83  # noqa: E402,F401  (robust refraction patch, see closure83.py)
import layered81 as L81  # noqa: E402
mi = L81.init('cuda_ad_rgb')
import drjit as dr  # noqa: E402
import closure83_dr as CD  # noqa: E402

P83 = json.load(open(os.path.join(HERE, 'protocol', 'compiled83.json')))
P82 = json.load(open(os.path.join(HERE, '..', 'fnto_compiled82', 'protocol', 'compiled82.json')))
SD = P83['seeds']
TT = P83['runtime_tests']
dev_t = torch.device('cuda')
DEV = '--dev' in sys.argv
ARMS = ['C0', 'E', 'ER']


def V3(a):
    a = np.ascontiguousarray(a, np.float32)
    return mi.Vector3f(mi.Float(a[:, 0]), mi.Float(a[:, 1]), mi.Float(a[:, 2]))


def np3(v):
    return np.stack([np.asarray(v[k]) for k in range(3)], 1).astype(np.float64)


def dirs(rng, n, mu=None):
    if mu is None:
        mu = np.where(rng.random(n) < 0.5, rng.random(n), np.exp(rng.uniform(np.log(1e-6), 0, n)))
    ph = rng.uniform(0, 2 * np.pi, n)
    st = np.sqrt(np.maximum(0, 1 - mu * mu))
    return np.stack([st * np.cos(ph), st * np.sin(ph), mu], 1)


def rand_closures(arm, n, rng, amin=None):
    z = torch.from_numpy(rng.normal(0, 1.5, (n, CL.N_RAW)).astype(np.float32)).to(dev_t)
    p = CL.decode(arm, 'cert', z)
    if amin is not None:
        p['au'] = p['au'].clamp_min(amin); p['av'] = p['av'].clamp_min(amin)
    return p


def synthetic(arm):
    S = []
    for lam, al, s, eta in P82['tests']['synthetic']:
        p = {'lam': torch.zeros(1, 3, 8), 'au': torch.full((1, 7), 0.3), 'av': torch.full((1, 7), 0.3), 'f0': torch.full((1, 7), 0.6),
             's': torch.tensor(s, dtype=torch.float32).view(1, 1, 5).repeat(1, 3, 1), 'eta': torch.tensor([eta])}
        for r, l in lam.items():
            p['lam'][0, :, int(r)] = l
        for r, (a, b) in al.items():
            p['au'][0, int(r) - 1] = a; p['av'][0, int(r) - 1] = b
        S.append(p)
    p = {k: torch.cat([q[k] for q in S]).to(dev_t) for k in S[0]}
    if arm == 'C0':
        p = {k: v for k, v in p.items() if k not in ('s', 'eta')}
    elif arm == 'E':
        p = {k: v for k, v in p.items() if k != 'eta'}
    return p


def table_closure(arm, st, idx):
    return CD.DrClosure83(arm, st, mi.UInt32(np.ascontiguousarray(idx, np.uint32)))


def rel(a, b):
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-8)


def r0_r1_r3_r4(rng):
    out = {}
    n = 1 << (16 if DEV else 20)
    for arm in ARMS:
        p = rand_closures(arm, 64, rng)
        st = CD.compiled_state(arm, p)
        idx = rng.integers(0, 64, n)
        wi, wo = dirs(rng, n), dirs(rng, n)
        cl = table_closure(arm, st, idx)
        f, pd = cl.eval_pdf(V3(wi), V3(wo))
        f, pd = np3(f), np.asarray(pd, np.float64)
        with torch.no_grad():
            pl = CL.gather(p, torch.from_numpy(idx).to(dev_t))
            tw = lambda a: torch.from_numpy(a.astype(np.float32)).to(dev_t)
            ft = CL.eval_closure(arm, pl, tw(wi), tw(wo)).double().cpu().numpy()
            pt = CL.pdf_closure(arm, pl, tw(wi), tw(wo)).double().cpu().numpy()
        re, rp = rel(f, ft).ravel(), rel(pd, pt)
        r = {'eval_rel_median': float(np.median(re)), 'eval_rel_p999': float(np.quantile(re, .999)),
             'pdf_rel_median': float(np.median(rp)), 'pdf_rel_p999': float(np.quantile(rp, .999))}
        r['R1_PASS'] = r['eval_rel_median'] <= TT['rel_median'] and r['eval_rel_p999'] <= TT['rel_p999'] and \
            r['pdf_rel_median'] <= TT['rel_median'] and r['pdf_rel_p999'] <= TT['rel_p999']
        # R0: uniform binding == table binding (bit-identical) on the first 8 closures
        ok = True
        for k in range(8):
            m = idx == k
            cu = CD.DrClosure83(arm, st[k])
            fu, pu = cu.eval_pdf(V3(wi[m]), V3(wo[m]))
            ok &= bool(np.array_equal(np3(fu), f[m]) and np.array_equal(np.asarray(pu, np.float64), pd[m]))
        r['R0_binding_identical'] = ok
        # R3 reciprocity
        fr = np3(cl.eval(V3(wo), V3(wi)))
        a, b = f / np.maximum(wo[:, 2:3], 1e-30), fr / np.maximum(wi[:, 2:3], 1e-30)
        msk = (np.maximum(np.abs(a), np.abs(b)) > 1e-6) & (wi[:, 2:3] >= 1e-3) & (wo[:, 2:3] >= 1e-3)
        r['R3_recip_max_rel'] = float((np.abs(a - b) / np.maximum(np.abs(a), np.abs(b)))[msk].max())
        r['R3_PASS'] = r['R3_recip_max_rel'] <= TT['recip_max_rel']
        # R4 finiteness at grazing incidences
        mus = np.array([0.0, 1e-12, 2e-7, 1e-4])
        wg = dirs(rng, n, mus[rng.integers(0, 4, n)])
        f1, p1 = cl.eval_pdf(V3(wg), V3(wo)); f2, p2 = cl.eval_pdf(V3(wo), V3(wg))
        u = rng.random((n, 3)).astype(np.float32)
        wos, ps, ws, _ = cl.sample(V3(wg), mi.Float(u[:, 0]), mi.Point2f(mi.Float(u[:, 1]), mi.Float(u[:, 2])))
        arrs = [np3(f1), np.asarray(p1), np3(f2), np.asarray(p2), np3(ws), np.asarray(ps)]
        r['R4_finite_nonneg'] = bool(all(np.isfinite(x).all() and (x >= 0).all() for x in arrs))
        out[arm] = r
    return out


def own_albedo(cl_fn, rows_wi, ns, seed, chunk=1 << 22):
    """Dr.Jit own-sampler albedo per row; cl_fn(idx_rows) -> closure bound to the row's material index per lane. Returns a, se [R,3]."""
    R = len(rows_wi)
    a = np.zeros((R, 3)); se = np.zeros((R, 3))
    rows = max(1, chunk // ns)
    for r0 in range(0, R, rows):
        r1 = min(R, r0 + rows)
        n = (r1 - r0) * ns
        lane = dr.arange(mi.UInt32, n)
        row = lane // ns
        cl = cl_fn(r0, r1, row)
        wr = np.ascontiguousarray(rows_wi[r0:r1], np.float32)
        wi = mi.Vector3f(*[dr.gather(mi.Float, mi.Float(wr[:, k]), row) for k in range(3)])
        rg = mi.PCG32(size=n, initstate=seed, initseq=mi.UInt64(r0 * ns))
        u1 = rg.next_float32(); u2 = mi.Point2f(rg.next_float32(), rg.next_float32())
        _, _, w, _ = cl.sample(wi, u1, u2)
        s1 = np.stack([np.asarray(dr.block_sum(mi.Float64(w[k]), ns), np.float64) for k in range(3)], 1)
        s2 = np.stack([np.asarray(dr.block_sum(mi.Float64(w[k]) * mi.Float64(w[k]), ns), np.float64) for k in range(3)], 1)
        a[r0:r1] = s1 / ns
        se[r0:r1] = np.sqrt(np.maximum(s2 / ns - (s1 / ns) ** 2, 0) / ns)
    return a, se


def r2(rng):
    from scipy.stats import chi2 as CHI2
    out = {}
    ths = np.radians(TT['sampler_theta_deg'])
    ns = TT['sampler_samples'] if not DEV else 1 << 16
    for arm in ARMS:
        ps = synthetic(arm)
        pr = rand_closures(arm, 16, rng, amin=0.1)
        p = {k: torch.cat([ps[k], pr[k]]) for k in ps}
        st = CD.compiled_state(arm, p)
        M = len(st)
        rows_c = np.repeat(np.arange(M), len(ths))
        th = np.tile(ths, M); ph = rng.uniform(0, 2 * np.pi, len(th))
        wi = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
        mk = lambda r0, r1, row: CD.DrClosure83(arm, st, dr.gather(mi.UInt32, mi.UInt32(rows_c[r0:r1].astype(np.uint32)), row))
        a_s, se_s = own_albedo(mk, wi, ns, SD['runtime_tests'])
        # quadrature (uniform hemisphere) of eval and pdf; sampler valid fraction; pdf consistency of returned samples
        nq = ns
        a_q = np.zeros((len(wi), 3)); se_q = np.zeros_like(a_q); pint = np.zeros(len(wi)); spint = np.zeros(len(wi)); vf = np.zeros(len(wi))
        pdf_dev = []
        for r in range(len(wi)):
            cl = CD.DrClosure83(arm, st[rows_c[r]])
            u = rng.random((nq, 2))
            mu = u[:, 0]; ph2 = 2 * np.pi * u[:, 1]; s_ = np.sqrt(1 - mu * mu)
            wq = np.stack([s_ * np.cos(ph2), s_ * np.sin(ph2), mu], 1)
            W = np.repeat(wi[r:r + 1], nq, 0)
            fq, pq = cl.eval_pdf(V3(W), V3(wq))
            fq = np3(fq) * 2 * np.pi; pq = np.asarray(pq, np.float64) * 2 * np.pi
            a_q[r] = fq.mean(0); se_q[r] = fq.std(0) / math.sqrt(nq); pint[r] = pq.mean(); spint[r] = pq.std() / math.sqrt(nq)
            uu = rng.random((nq, 3)).astype(np.float32)
            wo_s, p_s, _, valid = cl.sample(V3(W), mi.Float(uu[:, 0]), mi.Point2f(mi.Float(uu[:, 1]), mi.Float(uu[:, 2])))
            vf[r] = np.asarray(valid).mean()
            _, p_e = cl.eval_pdf(V3(W), wo_s)
            v = np.asarray(valid)
            if v.any():
                pdf_dev.append(rel(np.asarray(p_s, np.float64)[v], np.asarray(p_e, np.float64)[v]))
        za = np.maximum(np.abs(a_s - a_q) - TT['abs_slack'], 0) / np.maximum(np.sqrt(se_s ** 2 + se_q ** 2), 1e-12)
        zv = np.maximum(np.abs(vf - pint) - TT['abs_slack'], 0) / np.maximum(np.sqrt(vf * (1 - vf) / nq + spint ** 2), 1e-12)
        pdf_dev = np.concatenate(pdf_dev)
        # amendment 1 (pre-evidence): sample-returned pdf vs eval_pdf recomputed in a separate kernel, R1 tolerances (median / p99.9);
        # the maximum is reported (float32 conditioning of the refracted-lobe pdf at near-horizon wo, see worklog 83)
        r_ = {'cases': len(wi), 'max_z_albedo': float(za.max()), 'max_z_valid_fraction': float(zv.max()),
              'sample_pdf_consistency_median_rel': float(np.median(pdf_dev)), 'sample_pdf_consistency_p999_rel': float(np.quantile(pdf_dev, .999)),
              'sample_pdf_consistency_max_rel': float(pdf_dev.max()), 'sample_pdf_samples': int(len(pdf_dev))}
        # chi-square on the synthetic closures
        pv = []
        nb_t, nb_p = TT['chi2_bins']
        n = TT['chi2_samples'] if not DEV else 1 << 16
        for c in range(len(ps['lam'])):
            cl = CD.DrClosure83(arm, st[c])
            for t in np.radians(TT['chi2_theta_deg']):
                W = np.repeat([[math.sin(t), 0.0, math.cos(t)]], n, 0)
                uu = rng.random((n, 3)).astype(np.float32)
                wo_s, _, _, valid = cl.sample(V3(W), mi.Float(uu[:, 0]), mi.Point2f(mi.Float(uu[:, 1]), mi.Float(uu[:, 2])))
                wo_s = np3(wo_s)[np.asarray(valid)]
                bins = lambda w_: np.clip((w_[:, 2] * nb_t).astype(int), 0, nb_t - 1) * nb_p + \
                    np.clip(((np.arctan2(w_[:, 1], w_[:, 0]) % (2 * np.pi)) / (2 * np.pi) * nb_p).astype(int), 0, nb_p - 1)
                obs = np.bincount(bins(wo_s), minlength=nb_t * nb_p).astype(np.float64)
                nq2 = TT['chi2_quad'] if not DEV else 1 << 20
                u = rng.random((nq2, 2)); mu = u[:, 0]; s_ = np.sqrt(1 - mu * mu); ph2 = 2 * np.pi * u[:, 1]
                wq = np.stack([s_ * np.cos(ph2), s_ * np.sin(ph2), mu], 1)
                pq = np.asarray(cl.pdf(V3(np.repeat(W[:1], nq2, 0)), V3(wq)), np.float64)
                exp = np.bincount(bins(wq), weights=pq, minlength=nb_t * nb_p) * (2 * np.pi / nq2) * n
                big = exp >= 5
                rest = exp[~big].sum()
                stat = (((obs[big] - exp[big]) ** 2) / exp[big]).sum() + ((((obs[~big].sum() - rest) ** 2) / rest) if rest >= 5 else 0)
                pv.append(float(CHI2.sf(stat, int(big.sum() + (rest >= 5)))))
        r_['chi2_min_p'] = float(min(pv)); r_['chi2_cases'] = len(pv)
        r_['PASS'] = r_['max_z_albedo'] <= TT['z_max'] and r_['max_z_valid_fraction'] <= TT['z_max'] and r_['chi2_min_p'] >= TT['chi2_min_p'] \
            and r_['sample_pdf_consistency_median_rel'] <= TT['rel_median'] and r_['sample_pdf_consistency_p999_rel'] <= TT['rel_p999']
        out[arm] = r_
    return out


def runtime(out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SD['runtime_tests'])
    t0 = time.time()
    a = r0_r1_r3_r4(rng)
    b = r2(rng)
    res = {'R0_R1_R3_R4': a, 'R2_sampler': b}
    res['PASS'] = bool(all(a[k]['R1_PASS'] and a[k]['R0_binding_identical'] and a[k]['R3_PASS'] and a[k]['R4_finite_nonneg'] for k in ARMS)
                       and all(b[k]['PASS'] for k in ARMS))
    res['seconds'] = time.time() - t0
    json.dump(res, open(os.path.join(out, 'runtime_tests.json'), 'w'), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


def audit(data, fitdir, out):
    """R5 on every DF and learned closure of every evaluated set + 50 adversarial closures per arm."""
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SD['audit'])
    mus = np.array(P82['contract']['audit_mu'])
    ns = P82['contract']['audit_samples'] if not DEV else 4096
    res = {}
    for arm in ARMS:
        zs, names = [], []
        for fn in sorted(os.listdir(fitdir)):
            if fn.endswith('.npz') and (f'_L_{arm}_s' in fn or f'_{arm}_cert_metric_' in fn):
                z = np.load(os.path.join(fitdir, fn))['z']
                zs.append(z); names.append((fn, len(z)))
        g = torch.Generator().manual_seed(SD['adversarial'])
        za = torch.randn(50, CL.N_RAW, generator=g) * 3.0
        za[:12, :21] = torch.sign(za[:12, :21]) * 12.0
        zs.append(za.numpy()); names.append(('adversarial', 50))
        z = torch.from_numpy(np.concatenate(zs)).to(dev_t)
        with torch.no_grad():
            st = CD.compiled_state(arm, CL.decode(arm, 'cert', z))
        M = len(st)
        nf = len(mus) * 4
        mu = np.concatenate([np.tile(np.repeat(mus, 4), (M, 1)),
                             np.concatenate([np.exp(rng.uniform(np.log(2e-7), np.log(1e-2), (M, 8))), rng.uniform(2e-7, 1, (M, 8))], 1)], 1).ravel()
        ph = rng.uniform(0, 2 * np.pi, len(mu)); sn = np.sqrt(np.maximum(0, 1 - mu * mu))
        wi = np.stack([sn * np.cos(ph), sn * np.sin(ph), mu], 1)
        rows_c = np.repeat(np.arange(M), nf + 16)
        mk = lambda r0, r1, row: CD.DrClosure83(arm, st, dr.gather(mi.UInt32, mi.UInt32(rows_c[r0:r1].astype(np.uint32)), row))
        a, se = own_albedo(mk, wi, ns, SD['audit'] + ARMS.index(arm))
        viol = (a - 1 > np.maximum(4 * se, 1e-3)).any(1)
        res[arm] = {'closures': names, 'n_closures': M, 'rows': int(len(wi)), 'samples': ns, 'violations': int(viol.sum()), 'max_albedo': float(a.max()),
                    'max_albedo_row_se': float(se[np.unravel_index(a.argmax(), a.shape)]), 'PASS': int(viol.sum()) == 0}
        print(arm, json.dumps({k: v for k, v in res[arm].items() if k != 'closures'}), flush=True)
    res['PASS'] = all(res[a]['PASS'] for a in ARMS)
    json.dump(res, open(os.path.join(out, 'audit.json'), 'w'), indent=1, default=float)


def independence(data, w81, out):
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(os.path.join(data, 'materials83_meta.json')))
    m83 = np.load(os.path.join(data, 'materials83.npz'))
    inner = {}
    ks = list(m83.keys())
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            A = {tuple(np.round(x, 12)) for x in m83[a]}
            inner[f'{a}|{b}'] = int(sum(tuple(np.round(x, 12)) in A for x in m83[b]))
    s81 = json.load(open(os.path.join(HERE, '..', 'fnto_compiled81', 'protocol', 'compiled81.json')))['seeds']
    flat = lambda S: {v for v in S.values() if isinstance(v, int)} | {x for v in S.values() if isinstance(v, list) for x in v}
    prev = flat(s81) | flat(P82['seeds'])
    clash = [k for k, v in SD.items() if (isinstance(v, int) and v in prev) or (isinstance(v, list) and any(x in prev for x in v))]
    r = {'overlap_with_wl81_wl82': meta['overlap_with_wl81_wl82_materials'], 'overlap_within_wl83': inner, 'seed_clash': clash,
         'note': 'train83 = WL81 train, val83 = WL81 val (never fitted, trained on or evaluated by any WL81 / WL82 candidate); test83_* are new draws'}
    r['PASS'] = all(v == 0 for v in meta['overlap_with_wl81_wl82_materials'].values()) and all(v == 0 for v in inner.values()) and not clash
    json.dump(r, open(os.path.join(out, 'independence.json'), 'w'), indent=1)
    print(json.dumps(r), flush=True)


if __name__ == '__main__':
    w = sys.argv[1]
    if w == 'runtime':
        runtime(sys.argv[2])
    elif w == 'audit':
        audit(sys.argv[2], sys.argv[3], sys.argv[4])
    elif w == 'independence':
        independence(sys.argv[2], sys.argv[3], sys.argv[4])
    os._exit(0)
