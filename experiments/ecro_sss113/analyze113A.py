"""WL113 frozen analysis of Stage A (focused qualification Q1-Q8, oracle arms) and the Stage B representation selection.

  python analyze113A.py ROOT   -> ROOT/analysis/{stageA_qualification.json, stageA_metrics.json, maps_A_*.npz}, ROOT/stageB/decision.json
"""
import json, math, os, sys
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'ecro_sss112'))
import common113 as C  # noqa: E402
import metrics113 as MX  # noqa: E402
import analyzeAB112 as A112  # noqa: E402

N, NS, NB, K = 1270, 16, 64, 4
LUMW = MX.LUMW
H_MIN = 2.124e-7
BN, BC = 0.0649, 0.0579
FIXI = {'F0': 0, 'F1': 1, 'F2': 2, 'F3': 3}
ARMS = ['U', 'S2', 'S4', 'L4', 'R0']
NREP = C.dv(4, 2)


def load_acc(p):
    with np.load(p) as z:
        return {k: z[k].astype(np.float64) for k in z.files if k != 'meta'}


def sub_rect(g, pid, sub):
    sa, sb = sub % K, sub // K
    p0 = g.PO[pid] + sa / K * g.PE1[pid] + sb / K * g.PE2[pid]; p1 = p0 + (g.PE1[pid] + g.PE2[pid]) / K
    return np.minimum(p0, p1)[:2], np.maximum(p0, p1)[:2]


def footprint4(g, geom, pid, sub):
    lo, hi = sub_rect(g, pid, sub)
    cov = 0.
    for bname in 'AB':
        bx = geom[bname]
        if abs(bx['c'][2] - bx['h'][2]) > 1e-9:
            continue
        ax = C.C112.box_axes(bx['yaw']); c = np.asarray(bx['c']); h = bx['h']
        poly = np.array([(c + u * h[0] * ax[0] + v * h[1] * ax[1])[:2] for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        for axis, value, sign in ((0, lo[0], 1), (0, hi[0], -1), (1, lo[1], 1), (1, hi[1], -1)):
            poly = A112.clip_polygon(poly, axis, value, sign)
            if len(poly) == 0:
                break
        if len(poly):
            x, y = poly.T; cov += abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) * 0.5
    return cov / (hi - lo).prod()


def quad4(g, pid, sub, b, n_pos=12, n_dir=32):
    sa, sb = sub % K, sub // K
    ua = (sa + (np.arange(n_pos) + 0.5) / n_pos) / K; ub = (sb + (np.arange(n_pos) + 0.5) / n_pos) / K
    A_, B_ = [x.ravel() for x in np.meshgrid(ua, ub, indexing='ij')]
    if g.PSID[pid] == 6:
        ip, iz = g.PSPH[pid]
        zs = (iz + B_) * 0.5 - 1; ph = (ip + A_) * (2 * math.pi / 8) - math.pi
        r = np.sqrt(np.maximum(1 - zs ** 2, 0)); nl = np.stack([r * np.cos(ph), r * np.sin(ph), zs], 1); p = g.SC + g.SR * nl
    else:
        p = g.PO[pid] + A_[:, None] * g.PE1[pid] + B_[:, None] * g.PE2[pid]; nl = np.broadcast_to(g.PN[pid], p.shape)
    iz, ipb = b // 16, b % 16
    z0, z1 = (iz / 4, (iz + 1) / 4) if not (g.PSID[pid] == 6 and iz == 0) else (-1.0, 0.25)
    zz = z0 + (z1 - z0) * (np.arange(n_dir) + 0.5) / n_dir; pp = (ipb + (np.arange(n_dir) + 0.5) / n_dir) * (2 * math.pi / 16) - math.pi
    Z, PH = [x.ravel() for x in np.meshgrid(zz, pp, indexing='ij')]
    rr = np.sqrt(np.maximum(1 - Z ** 2, 0)); loc = np.stack([rr * np.cos(PH), rr * np.sin(PH), Z], 1)
    w = loc[:, :1] * g.PT[pid] + loc[:, 1:2] * g.PB[pid] + loc[:, 2:3] * g.PN[pid]
    cos = nl @ w.T; front = cos > 1e-6
    L = 1 + 0.5 * p[:, :1] + 0.3 * p[:, 1:2] + 0.4 * cos + 0.2 * w[None, :, 0]
    return float((L * front).sum() / max(front.sum(), 1))


def qualification(root):
    out = {}
    ident = json.loads((root / 'stageA' / 'identity.json').read_text())
    out['Q1'] = {'pass': bool(all(v for k, v in ident.items() if k.startswith('A1_table')) and ident['A1_lane_maxabs'] == 0 and ident['deep_events1to6_rel_l1'] <= 1e-6),
                 'lane_maxabs': ident['A1_lane_maxabs'], 'deep_events1to6_rel_l1': ident['deep_events1to6_rel_l1']}
    out['Q2'] = {'pass': bool(ident['A2_paired'] >= 2000 and ident['A2_rel_l1'] <= 1e-5), 'paired': ident['A2_paired'], 'rel_l1': ident['A2_rel_l1']}
    # Q3 synthetic
    geom0 = C.fixture('F0')[0]; g = A112.geom112(geom0)
    acc = load_acc(root / 'stageA' / 'synthetic.npz')
    rs = np.random.default_rng(11305002); rows = []; ok = 0
    cand = [int(p) for p in np.flatnonzero(g.PSID != 5)]
    while len(rows) < 96:
        pid = int(rs.choice(cand)); sub = int(rs.integers(16)); b = int(rs.integers(16 if g.PSID[pid] == 6 and rs.random() < .5 else 64))
        i = (pid * NS + sub) * NB + b
        if acc['buried'][i] > 0 or acc['U_w'][i] < 32:
            continue
        ex = quad4(g, pid, sub, b)
        w, s1, sq = acc['U_w'][i], acc['U_f20'][i], acc['U_sq0'][i]
        m = s1 / w; se = math.sqrt(3.0 * max(sq / w - m * m, 0) / w)                 # x WL112 measured overdispersion
        z = (m - ex) / max(se, 1e-12); ok += abs(z) <= 4
        rows.append({'pid': pid, 'sub': sub, 'bin': b, 'mc': m, 'exact': ex, 'z': z})
    sys.path.insert(0, str(HERE)); import tracer113 as T
    Tc, T2, T4, nat = T.tables4(acc)
    w4 = acc['U_w'].reshape(N, NS, NB)
    rec = (T4 * w4[..., None]).sum(1) / np.maximum(w4.sum(1), 1e-30)[..., None]
    parent = float(np.abs(rec - Tc)[nat['c']].max() / np.abs(Tc).max())
    out['Q3'] = {'pass': bool(ok / 96 >= 0.95 and parent <= 1e-6), 'fraction_within_4se': ok / 96, 'parent_max_rel': parent, 'rows': rows}
    # Q4 support
    n = C.dv(1024, 32)
    q4 = {}; allok = True
    for fix, fi in FIXI.items():
        geom = C.fixture(fix)[0]; g = A112.geom112(geom); sc = np.asarray(geom['S']['c'])
        accs = [load_acc(root / 'stageA' / f'{fix}_r{r}.npz') for r in range(NREP)]
        burs = np.stack([a['buried'].reshape(N, NS, NB).sum(2) for a in accs])
        tests = []
        for pid in np.flatnonzero(g.PSID == 0):
            for sub in range(NS):
                f = footprint4(g, geom, pid, sub)
                if f <= 0 and burs[:, pid, sub].sum() == 0:
                    continue
                x = burs[:, pid, sub].astype(float); sd = x.std(ddof=1); tot = NB * n
                # dispersion: between-replica, floored by the overdispersed binomial (factor 3, WL112) and integer resolution
                se = math.sqrt(max(sd * sd, 3.0 * tot * f * (1 - f), 0.25) / NREP)
                t = (x.mean() - tot * f) / se
                tests.append(t)
        crit = float(stats.t.ppf(1 - 0.01 / 2 / max(len(tests), 1), NREP - 1))
        # singular expectation (contact geometries only)
        exp_s = 0.; reg = np.zeros(N * NS, bool)
        if abs(sc[2] - C.R_SPHERE) < 1e-12:
            rs2 = 4 * C.R_SPHERE * H_MIN - 4 * H_MIN ** 2
            asub_f = None
            for pid in np.flatnonzero(g.PSID == 0):
                for sub in range(NS):
                    lo, hi = sub_rect(g, pid, sub)
                    dx = max(lo[0] - sc[0], 0, sc[0] - hi[0]); dy = max(lo[1] - sc[1], 0, sc[1] - hi[1])
                    if math.hypot(dx, dy) <= math.sqrt(rs2) * 1.01:              # sub-cell meets the singular disk
                        reg[pid * NS + sub] = True; asub_f = (hi - lo).prod()
            exp_s += NB * n * math.pi * rs2 / asub_f                              # equal-area sub-cells: total expectation
            lower = [pid for pid in np.flatnonzero(g.PSID == 6) if g.PSPH[pid][1] == 0]
            asub = 4 * math.pi * C.R_SPHERE ** 2 / 32 / NS
            for pid in lower:
                for sub in range(4):                                          # sv = 0 row
                    exp_s += NB * n * (2 * math.pi * C.R_SPHERE * 2 * H_MIN / (len(lower) * 4)) / asub; reg[pid * NS + sub] = True
        sing = [a['singular'].reshape(N * NS, NB).sum(1) for a in accs]
        ps = []
        for s_ in sing:
            k = s_.sum()
            p = 2 * min(stats.poisson.cdf(k, exp_s), stats.poisson.sf(k - 1, exp_s)) if exp_s > 0 else (1.0 if k == 0 else 0.0)
            ps.append(min(p, 1.0))
        unreg = float(sum(s_[~reg].sum() for s_ in sing))
        mis = max(float(a['mismatch'].sum() / a['U_w'].sum()) for a in accs)
        okf = bool(max(abs(t) for t in tests) <= crit and min(ps) >= 1e-4 and unreg == 0 and mis <= 1e-5) if tests else bool(min(ps) >= 1e-4 and unreg == 0 and mis <= 1e-5)
        q4[fix] = {'pass': okf, 'footprint_max_abs_t': max((abs(t) for t in tests), default=0.), 'critical_t': crit,
                   'singular_observed': [float(s_.sum()) for s_ in sing], 'singular_expected': exp_s, 'singular_p_min': min(ps),
                   'singular_unregistered': unreg, 'mismatch_rate_max': mis}
        allok &= okf
    out['Q4'] = {'pass': bool(allok), 'fixtures': q4}
    return out


def numeric_bound16(root, fix):
    bs = None
    for r in range(NREP):
        a = load_acc(root / 'stageA' / f'{fix}_r{r}.npz')
        n = np.maximum(a['U_w'], 1)
        T_ = np.stack([a[f'U_f2{c}'] + a[f'U_fh{c}'] for c in range(3)], -1) / n[:, None]
        mis = np.stack([a[f'misL{c}'] for c in range(3)], -1)
        sd = np.sqrt(np.maximum(np.stack([a[f'U_sq{c}'] for c in range(3)], -1) / n[:, None] - T_ ** 2, 0))
        b = np.abs(mis - a['mismatch'][:, None] * T_) / n[:, None] + a['singular'][:, None] / (n[:, None] + a['singular'][:, None]) * (T_ + 3 * sd)
        b = b @ LUMW
        bs = b if bs is None else np.maximum(bs, b)
    return bs


def load_streams(d, tag):
    st = []
    for j in range(2):
        with np.load(d / f'stream_{tag}_{j}.npz') as z:
            st.append({k: z[k] for k in z.files})
    with np.load(d / f'gbuffer_{tag}.npz') as z:
        gb = {k: z[k] for k in z.files}
    return st, gb


def main(root):
    root = Path(root); an = root / 'analysis'; an.mkdir(parents=True, exist_ok=True)
    q = qualification(root)
    metrics = {}; q5 = {'C_construction': 0., 'C_references': 0., 'floor_band': 0., 'paths': 0.}; q6 = {}; q7 = {}; q8 = {}
    for fix in FIXI:
        for r in range(NREP):
            q5['C_construction'] += float(load_acc(root / 'stageA' / f'{fix}_r{r}.npz')['bad_origin'].sum())
    for fix, fi in FIXI.items():
        geom, lc = C.fixture(fix)
        native = dict(np.load(root / 'stageA' / f'native_{fix}.npz'))
        bs = numeric_bound16(root, fix)
        sup = None
        for vname in C.views(fix):
            tag = f'{fix}_{vname}'
            st, gb = load_streams(root / 'stageA', tag)
            g3, reg, bands = MX.gate_masks(gb, geom, lc[0]['pos'])
            cv = C.contact_valid(reg)
            h = [s['phys_h'].astype(np.float64) for s in st]
            kap = MX.pixel_kappa(h[0], h[1], reg['floor'])
            mu = MX.region_mean_lum(0.5 * (h[0] + h[1]), reg['near']) if reg['near'].any() else float('nan')
            vm = {'contact_valid': cv, 'kappa': kap, 'mu_ref_near_high': mu, 'band_near': BN * mu, 'band_contrast': BC * mu, 'arms': {}}
            for s in st:
                m = json.loads(str(s['meta'])); q5['C_references'] += m.get('bad_origin', 0.)
                q5['floor_band'] += m.get('bad_floor', 0.); q5['paths'] += m.get('paths', 0.)
            q7[tag] = float(max((np.abs(s['phys_e1'] - s['ro_e1']).sum() + np.abs(s['phys_e2'] - s['ro_e2']).sum()) / (np.abs(s['phys_e1']).sum() + np.abs(s['phys_e2']).sum()) for s in st))
            R6 = [s['phys_e1'].astype(np.float64) + s['phys_e2'] + s['phys_h'] for s in st]
            Rf = [R6[j] + st[j]['tail'] for j in range(2)]
            vm['tail_share_near'] = float(MX.region_mean_lum(0.5 * (st[0]['tail'] + st[1]['tail']), reg['near']) / MX.region_mean_lum(0.5 * (Rf[0] + Rf[1]), reg['near'])) if reg['near'].any() else float('nan')
            arms = ARMS + (['S4D'] if 'hl|S4D_r0' in st[0] else [])
            for arm in arms:
                res = [[st[j][f'hl|{arm}_r{r}'].astype(np.float64) - h[j] for j in range(2)] for r in range(NREP)]
                o = {}
                if cv:
                    o['endpoints'] = MX.endpoint_stats(res, reg['near'], reg['flank'], kappa=kap)
                o['bands'] = MX.band_profile(res, bands)
                o['rmse'] = {k: MX.rmse_debiased(res, h, reg[k])[0] for k in ('near', 'flank', 'sphere_boundary', 'sphere_interior', 'sphere', 'contact_shadow', 'floor', 'floor_outside', 'full')}
                o['coverage'] = {r_: float(np.mean([MX.coverage16(st[0][f'W|{r_}'].astype(np.float64) + st[1][f'W|{r_}'], native[f'{arm}_r{r}']) for r in range(NREP)]))
                                 for r_ in ('near', 'flank', 'contact_shadow', 'sphere_boundary', 'full')}
                M = [[st[j]['ro_e1'].astype(np.float64) + st[j]['ro_e2'] + st[j][f'hl|{arm}_r{r}'] for j in range(2)] for r in range(NREP)]
                Mbar = [np.mean([M[r][j] for r in range(NREP)], 0) for j in range(2)]
                o['gate3_depth_matched'] = MX.gate3(Mbar, R6, g3, kap)
                o['gate3_full_depth'] = MX.gate3(Mbar, Rf, g3, kap)
                vm['arms'][arm] = o
            # Q6 materiality
            rows = {}
            for r_ in ('near', 'flank', 'contact_shadow', 'sphere_boundary', 'sphere', 'floor', 'full'):
                Wl = 0.5 * (st[0][f'W|{r_}'].astype(np.float64) + st[1][f'W|{r_}'])
                rows[r_] = float((Wl * bs).sum() / max(reg[r_].sum(), 1))
            qq = {}
            if cv:
                qq['near'] = bool(rows['near'] <= 0.1 * vm['band_near']); qq['contrast'] = bool(rows['near'] + rows['flank'] <= 0.1 * vm['band_contrast'])
            for r_ in ('sphere_boundary', 'sphere', 'contact_shadow', 'floor', 'full'):
                rms = math.sqrt(max(np.mean(h[0][reg[r_]] * h[1][reg[r_]]), 1e-30)); qq[r_] = bool(rows[r_] / rms <= 2.5e-4)
            q6[tag] = {'bounds': rows, 'qualified': qq}
            # Q8 decorrelated RNG control
            if 'S4D' in vm['arms'] and cv:
                d = {}
                for k in ('near', 'contrast'):
                    a, b = vm['arms']['S4']['endpoints'][k], vm['arms']['S4D']['endpoints'][k]
                    d[k] = {'diff': a['estimate'] - b['estimate'], 'combined_hw': math.sqrt(a['half_width'] ** 2 + b['half_width'] ** 2)}
                    d[k]['ok'] = bool(abs(d[k]['diff']) <= d[k]['combined_hw'])
                q8[tag] = d
            metrics[tag] = vm
            maps = {'ref_h': (0.5 * (h[0] + h[1])).astype(np.float32), 'ref_low': (0.5 * (R6[0] + R6[1]) - 0.5 * (h[0] + h[1])).astype(np.float32),
                    'tail': (0.5 * (st[0]['tail'] + st[1]['tail'])).astype(np.float32)}
            for arm in ARMS:
                maps[f'hl_{arm}'] = np.mean([0.5 * (st[0][f'hl|{arm}_r{r}'] + st[1][f'hl|{arm}_r{r}']) for r in range(NREP)], 0).astype(np.float32)
            for k_, m_ in reg.items():
                maps[f'mask_{k_}'] = m_
            maps['coverage'] = gb['coverage']
            np.savez_compressed(an / f'maps_A_{tag}.npz', **maps)
            print('VIEW', tag, flush=True)
    q['Q5'] = {'pass': bool(q5['C_construction'] == 0 and q5['C_references'] == 0 and q5['floor_band'] / max(q5['paths'], 1) <= 1e-7), **q5}
    unq = [f'{t}:{k}' for t, v in q6.items() for k, ok in v['qualified'].items() if not ok]
    q['Q6'] = {'unqualified': unq, 'per_view': q6}
    q['Q7'] = {'pass': bool(all(v <= 1e-5 for v in q7.values())), 'low_rel_l1': q7}
    q['Q8'] = {'pass': bool(all(x['ok'] for v in q8.values() for x in v.values())), 'per_view': q8}
    core = all(q[k]['pass'] for k in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q7', 'Q8'))
    q['verdict'] = 'NOT QUALIFIED' if not core else ('QUALIFIED' if not unq else 'PARTIALLY QUALIFIED')
    C.jd(q, an / 'stageA_qualification.json'); C.jd(metrics, an / 'stageA_metrics.json')
    decide(root, q, metrics)


def decide(root, q, metrics):
    out = {'stageA': q['verdict']}
    if q['verdict'] == 'NOT QUALIFIED':
        out['selected'] = 'STOP'; out['reason'] = 'Stage A not qualified'
        C.jd(out, root / 'stageB' / 'decision.json'); return
    unq = set(q['Q6']['unqualified'])
    adequate = True; equiv = True; rows = {}
    for tag, vm in metrics.items():
        s4, l4, u = vm['arms']['S4'], vm['arms']['L4'], vm['arms']['U']
        r = {}
        if vm['contact_valid']:
            for k, bk in (('near', 'band_near'), ('contrast', 'band_contrast')):
                e = s4['endpoints'][k]; el = l4['endpoints'][k]; band = vm[bk]
                r[f'S4_{k}_inside'] = bool(max(abs(e['lo']), abs(e['hi'])) <= band)
                r[f'S4_{k}_precision'] = bool(e['half_width'] <= 0.75 * band)
                r[f'L4_{k}_inside'] = bool(max(abs(el['lo']), abs(el['hi'])) <= band)
                r[f'L4_minus_S4_{k}'] = el['estimate'] - e['estimate']; r[f'L4_{k}_equiv'] = bool(abs(el['estimate'] - e['estimate']) <= 0.25 * band)
                if f'{tag}:{k}' in unq:
                    r[f'{k}_unqualified'] = True
            bb = [abs(a - b) for a, b in zip(l4['bands'][:8], s4['bands'][:8]) if not (math.isnan(a) or math.isnan(b))]
            r['L4_bands_equiv'] = bool(all(x <= 0.25 * vm['band_near'] for x in bb))
            adequate &= all(r[f'S4_{k}_inside'] and r[f'S4_{k}_precision'] and not r.get(f'{k}_unqualified') for k in ('near', 'contrast'))
            equiv &= all(r[f'L4_{k}_inside'] and r[f'L4_{k}_equiv'] for k in ('near', 'contrast')) and r['L4_bands_equiv']
        for reg in ('sphere_boundary', 'sphere', 'contact_shadow', 'floor', 'full'):
            ok = s4['rmse'][reg] <= 1.10 * u['rmse'][reg] + 0.0025; r[f'S4_nonreg_{reg}'] = bool(ok); adequate &= ok
        for reg in ('sphere_boundary', 'sphere_interior', 'contact_shadow'):
            ok = l4['rmse'][reg] <= 1.10 * s4['rmse'][reg] + 0.0025; r[f'L4_vs_S4_{reg}'] = bool(ok); equiv &= ok
        for reg in ('floor_outside', 'full'):
            ok = l4['rmse'][reg] <= 1.10 * u['rmse'][reg] + 0.0025; r[f'L4_nonreg_{reg}'] = bool(ok); equiv &= ok
        covs = [v for v in list(s4['coverage'].values()) + list(l4['coverage'].values()) if not math.isnan(v)]   # empty regions carry no queries
        cov = min(covs) if covs else float('nan')
        r['coverage_min'] = cov; adequate &= (not covs) or cov >= 0.995
        rows[tag] = r
    sys.path.insert(0, str(HERE)); import tracer113 as T
    budget = {}
    for fix in FIXI:
        desc = np.load(root / 'stageA' / f'vhs_{fix}.npy'); g = A112.geom112(C.fixture(fix)[0])
        budget[fix] = float(T.support_from_desc(desc, g).mean())
    within = all(v <= 0.30 for v in budget.values())
    out.update({'S4_adequate': bool(adequate), 'L4_equivalent': bool(equiv), 'support_fraction': budget, 'budget_ok': bool(within), 'views': rows})
    out['selected'] = 'L4' if (adequate and equiv and within) else ('S4' if adequate else 'STOP')
    out['note'] = 'S4 selection is provisional on the dense learned budget probe (precheck)' if out['selected'] == 'S4' else ''
    if C.DEV and os.environ.get('WL113_DEV_FORCE'):                    # development code-path test only (never evidence)
        out['selected_dev_actual'] = out['selected']; out['selected'] = os.environ['WL113_DEV_FORCE']
    C.jd(out, root / 'stageB' / 'decision.json')
    print('DECISION', out['selected'], 'S4 adequate', adequate, 'L4 equivalent', equiv, 'support', budget, flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
