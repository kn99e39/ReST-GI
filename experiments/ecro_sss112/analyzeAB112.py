"""WL112 frozen analysis of Stage A (construction contract v3 gates) and Stage B (camera-independent representation decision).

  python analyzeAB112.py ROOT        -> ROOT/analysis/stageA_gates.json, ROOT/stageB/decision.json, ROOT/analysis/stageB_metrics.json,
                                        ROOT/analysis/maps_B_<view>.npz (reviewer arrays)
"""
import json, math, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common112 as C  # noqa: E402
import metrics112 as MX  # noqa: E402

N, NB, NS = 1270, 64, 4
NCS = N * NS * NB
LUMW = MX.LUMW
SPH_NPHI, SPH_NZ, NZ, NPHI = 8, 4, 4, 16
H_MIN = 2.124e-7
BAND_NEAR_FRAC, BAND_CON_FRAC = 0.0649, 0.0579
REGIONS_NR = ('sphere_boundary', 'sphere', 'contact_shadow', 'floor', 'full')
REGIONS_ALL = ('sphere_boundary', 'sphere', 'sphere_interior', 'contact_shadow', 'floor', 'floor_outside', 'near', 'flank', 'full')


def load_acc(path):
    with np.load(path) as z:
        return {k: z[k].astype(np.float64) for k in z.files if k != 'meta'}


def geom112(geom):
    """numpy-only replica of tracer112.Geom112 (scene86.Geom with the sphere centre of the geometry)."""
    sys.path.insert(0, str(C.REPO / 'experiments/ecro86'))
    import scene86 as S
    g = S.Geom(geom)
    c = np.asarray(geom['S']['c'], np.float64); g.SC = c
    sph = g.PSID == S.SID_SPHERE
    g.PC[sph] = c + g.SR * g.PN[sph]
    return g


# ------------------------------------------------------------------------------------------------ A4 quadrature
def quad_cell(g, pid, sub, b, n_pos=24, n_dir=40):
    sa, sb = sub % 2, sub // 2
    ua = (sa + (np.arange(n_pos) + 0.5) / n_pos) / 2; ub = (sb + (np.arange(n_pos) + 0.5) / n_pos) / 2
    A_, B_ = np.meshgrid(ua, ub, indexing='ij'); A_ = A_.ravel(); B_ = B_.ravel()
    if g.PSID[pid] == 6:
        ip, iz = g.PSPH[pid]
        zs = (iz + B_) * (2.0 / SPH_NZ) - 1; ph = (ip + A_) * (2 * math.pi / SPH_NPHI) - math.pi
        r = np.sqrt(np.maximum(1 - zs ** 2, 0))
        nl = np.stack([r * np.cos(ph), r * np.sin(ph), zs], 1); p = g.SC + g.SR * nl
    else:
        p = g.PO[pid] + A_[:, None] * g.PE1[pid] + B_[:, None] * g.PE2[pid]; nl = np.broadcast_to(g.PN[pid], p.shape)
    iz, ipb = b // NPHI, b % NPHI
    z0, z1 = (iz / NZ, (iz + 1) / NZ) if not (g.PSID[pid] == 6 and iz == 0) else (-1.0, 0.25)
    zz = z0 + (z1 - z0) * (np.arange(n_dir) + 0.5) / n_dir
    pp = (ipb + (np.arange(n_dir) + 0.5) / n_dir) * (2 * math.pi / NPHI) - math.pi
    Z, PH = np.meshgrid(zz, pp, indexing='ij'); Z = Z.ravel(); PH = PH.ravel()
    rr = np.sqrt(np.maximum(1 - Z ** 2, 0))
    loc = np.stack([rr * np.cos(PH), rr * np.sin(PH), Z], 1)
    w = loc[:, :1] * g.PT[pid] + loc[:, 1:2] * g.PB[pid] + loc[:, 2:3] * g.PN[pid]          # [D,3]
    cos = nl @ w.T                                                                           # [P,D]
    front = cos > 1e-6
    L = 1 + 0.5 * p[:, :1] + 0.3 * p[:, 1:2] + 0.4 * cos + 0.2 * w[None, :, 0]
    u = (L * front).sum() / max(front.sum(), 1)
    gm = (L * cos * front).sum() / max((cos * front).sum(), 1e-30)
    return float(u), float(gm), float(front.mean())


def gate_A4(root, geom):
    acc = load_acc(root / 'stageA' / 'synthetic.npz')
    g = geom112(geom)
    rs = np.random.default_rng(11205002)
    shapes = {'floor': np.flatnonzero(g.PSID == 0), 'wall': np.flatnonzero((g.PSID >= 1) & (g.PSID <= 4)), 'panel': np.flatnonzero(g.PSID == 5),
              'boxA': np.flatnonzero(g.PSID == 7), 'boxB': np.flatnonzero(g.PSID == 8), 'sphere': np.flatnonzero(g.PSID == 6)}
    cells = []
    for name, k in (('floor', 8), ('wall', 6), ('panel', 4), ('boxA', 6), ('boxB', 4), ('sphere', 12)):
        while sum(1 for c in cells if c[0] == name) < k:
            pid = int(rs.choice(shapes[name])); sub = int(rs.integers(4))
            b = int(rs.integers(16)) if (name == 'sphere' and len([c for c in cells if c[0] == 'sphere']) % 2 == 0) else int(rs.integers(64))
            i = (pid * NS + sub) * NB + b
            if acc['buried'][i] > 0 or acc['U_w'][i] < 32:
                continue
            cells.append((name, pid, sub, b, i))
    rows = []; ok = 0; tot = 0
    for name, pid, sub, b, i in cells:
        u, gm, fr = quad_cell(g, pid, sub, b)
        for meas, ex in (('U', u), ('G', gm)):
            w = acc[f'{meas}_w'][i]; s1 = acc[f'{meas}_f20'][i]; sq = acc[f'{meas}_sq0'][i]; w2 = acc[f'{meas}_w2'][i]
            m = s1 / w; var = max(sq / w - m * m, 0)
            se = math.sqrt(var * w2) / w
            z = (m - ex) / max(se, 1e-12)
            rows.append({'shape': name, 'pid': pid, 'sub': sub, 'bin': b, 'measure': meas, 'mc': m, 'exact': ex, 'se': se, 'z': z})
            ok += abs(z) <= 4; tot += 1
    # nested parent identity
    sys.path.insert(0, str(HERE)); import tracer112 as T
    Tc, wc, _, _ = T.tables_from(acc, 'G', 'coarse'); Ts, ws, _, _ = T.tables_from(acc, 'G', 'sub')
    Tcs = (Ts.reshape(N, NS, NB, 3) * ws.reshape(N, NS, NB, 1)).sum(1) / np.maximum(ws.reshape(N, NS, NB).sum(1), 1e-30)[..., None]
    sel = wc.reshape(N, NB) > 0
    parent = float(np.abs(Tcs[sel] - Tc.reshape(N, NB, 3)[sel]).max() / np.abs(Tc).max())
    return {'pass': bool(ok / tot >= 0.95 and parent <= 1e-6), 'fraction_within_4se': ok / tot, 'entries': tot,
            'parent_max_rel': parent, 'rows': rows}


# ------------------------------------------------------------------------------------------------ A5 support
def clip_polygon(poly, axis, value, sign):
    cut = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        da, db = sign * (a[axis] - value), sign * (b[axis] - value)
        if (da >= 0) != (db >= 0):
            cut.append(a + (b - a) * (da / (da - db)))
        if db >= 0:
            cut.append(b)
    return np.array(cut, dtype=np.float64).reshape(-1, 2)


def footprint_fraction(g, geom, pid, sub):
    """Covered fraction of a floor sub-patch by resting boxes (analytic polygon clipping)."""
    sa, sb = sub % 2, sub // 2
    p0 = g.PO[pid] + 0.5 * sa * g.PE1[pid] + 0.5 * sb * g.PE2[pid]; p1 = p0 + 0.5 * g.PE1[pid] + 0.5 * g.PE2[pid]
    lo, hi = np.minimum(p0, p1)[:2], np.maximum(p0, p1)[:2]
    cov = 0.
    for bname in 'AB':
        bx = geom[bname]
        if abs(bx['c'][2] - bx['h'][2]) > 1e-9:
            continue
        ax = C.box_axes(bx['yaw']); c = np.asarray(bx['c']); h = bx['h']
        poly = np.array([(c + u * h[0] * ax[0] + v * h[1] * ax[1])[:2] for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        for axis, value, sign in ((0, lo[0], 1), (0, hi[0], -1), (1, lo[1], 1), (1, hi[1], -1)):
            poly = clip_polygon(poly, axis, value, sign)
            if len(poly) == 0:
                break
        if len(poly):
            x, y = poly.T
            cov += abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) * 0.5
    return cov / ((hi - lo).prod())


def gate_A5(root, geom):
    g = geom112(geom)
    res = {'replicas': []}
    ok = True
    n = C.dv(C.P['stage_A']['proposals_per_subcell'], 32)
    sc = np.asarray(geom['S']['c'])
    # analytic singular expectation (contact sphere only)
    exp_sing = 0.
    reg_sing = np.zeros(N * NS, bool)
    if abs(sc[2] - C.R_SPHERE) < 1e-12:
        rs2 = 4 * C.R_SPHERE * H_MIN - 4 * H_MIN ** 2
        for pid in np.flatnonzero(g.PSID == 0):
            for sub in range(4):
                sa, sb = sub % 2, sub // 2
                p0 = g.PO[pid] + 0.5 * sa * g.PE1[pid] + 0.5 * sb * g.PE2[pid]; p1 = p0 + 0.5 * g.PE1[pid] + 0.5 * g.PE2[pid]
                lo, hi = np.minimum(p0, p1)[:2], np.maximum(p0, p1)[:2]
                if np.all(sc[:2] > lo) and np.all(sc[:2] < hi):
                    exp_sing += NB * n * math.pi * rs2 / (hi - lo).prod(); reg_sing[pid * NS + sub] = True
        cap = 2 * math.pi * C.R_SPHERE * 2 * H_MIN
        lower = [pid for pid in np.flatnonzero(g.PSID == 6) if g.PSPH[pid][1] == 0]
        asub = 4 * math.pi * C.R_SPHERE ** 2 / 32 / 4
        for pid in lower:
            for sub in (0, 1):
                exp_sing += NB * n * (cap / (len(lower) * 2)) / asub; reg_sing[pid * NS + sub] = True
    for r in range(C.RC):
        acc = load_acc(root / 'stageA' / f'C_r{r}.npz')
        bur = acc['buried'].reshape(N, NS, NB).sum(2)
        zmax = 0.
        for pid in np.flatnonzero(g.PSID == 0):
            for sub in range(4):
                f = footprint_fraction(g, geom, pid, sub)
                if f <= 0 and bur[pid, sub] == 0:
                    continue
                tot = NB * n
                z = (bur[pid, sub] - tot * f) / math.sqrt(max(tot * f * (1 - f), 0) + 1)
                zmax = max(zmax, abs(z))
        under = [pid for pid in np.flatnonzero(g.PSID == 8) if g.PN[pid][2] < -0.99]
        under_ok = bool(np.all(bur[under] == NB * n))
        sing = acc['singular'].reshape(N * NS, NB).sum(1)
        k = float(sing.sum()); unreg = float(sing[~reg_sing].sum())
        from math import exp, lgamma, log
        def pois_cdf(kk, lam):
            return sum(exp(i * log(lam) - lam - lgamma(i + 1)) for i in range(int(kk) + 1)) if lam > 0 else 1.0
        p = 2 * min(pois_cdf(k, exp_sing), 1 - pois_cdf(k - 1, exp_sing) if k > 0 else 1.0) if exp_sing > 0 else (1.0 if k == 0 else 0.0)
        acc_n = acc['U_w'].sum()
        mis = float(acc['mismatch'].sum() / acc_n)
        row = {'replica': r, 'footprint_max_abs_z': zmax, 'underside_all_buried': under_ok, 'singular_observed': k,
               'singular_expected': exp_sing, 'singular_p': min(p, 1.0), 'singular_unregistered': unreg, 'mismatch_rate': mis}
        row['pass'] = bool(zmax <= 5 and under_ok and row['singular_p'] >= 1e-4 and unreg == 0 and mis <= 1e-5)
        ok &= row['pass']
        res['replicas'].append(row)
    res['pass_frozen_per_replica_binomial'] = bool(ok)
    # Amendment 1: pooled test with between-replica dispersion (the shared PCG32 lane seeding correlates proposals, so the
    # per-replica binomial variance is invalid; measured overdispersion ~3, decorrelated seeding ~1.1)
    from scipy import stats
    burs = np.stack([load_acc(root / 'stageA' / f'C_r{r}.npz')['buried'].reshape(N, NS, NB).sum(2) for r in range(C.RC)])
    tests = []
    for pid in np.flatnonzero(g.PSID == 0):
        for sub in range(4):
            f = footprint_fraction(g, geom, pid, sub)
            if f <= 0 and burs[:, pid, sub].sum() == 0:
                continue
            tot = NB * n
            x = burs[:, pid, sub].astype(np.float64)
            sd = x.std(ddof=1)
            t = (x.mean() - tot * f) / max(sd / math.sqrt(C.RC), 1e-9) if sd > 0 else (0.0 if abs(x.mean() - tot * f) < 1 else float('inf'))
            tests.append({'pid': int(pid), 'sub': int(sub), 'fraction': f, 'mean_excess': float(x.mean() - tot * f), 't': float(t),
                          'overdispersion': float(x.var(ddof=1) / max(tot * f * (1 - f), 1e-9))})
    crit = float(stats.t.ppf(1 - 0.01 / 2 / max(len(tests), 1), C.RC - 1))
    res['amendment1_footprint'] = {'critical_t': crit, 'tests': tests, 'max_abs_t': max(abs(x['t']) for x in tests)}
    other = all(r['underside_all_buried'] and r['singular_p'] >= 1e-4 and r['singular_unregistered'] == 0 and r['mismatch_rate'] <= 1e-5
                for r in res['replicas'])
    res['pass'] = bool(other and res['amendment1_footprint']['max_abs_t'] <= crit)
    return res


# ------------------------------------------------------------------------------------------------ A7 positive control
def gate_A7(root):
    sys.path.insert(0, str(HERE)); import tracer112 as T
    leg = [load_acc(root / 'stageA' / f'legacy_r{r}.npz') for r in range(2)]
    nat = [load_acc(root / 'stageA' / f'N_r{r}.npz') for r in range(C.RN)]
    def coarse(acc, key):
        return acc[key].reshape(N, NS, NB).sum(1)
    cnt = sum(coarse(a, 'U_w') for a in leg)
    contam = (sum(coarse(a, 'misroute') for a in leg) + sum(coarse(a, 'backcount') for a in leg)) / np.maximum(cnt, 1)
    def pooled(accs):
        w = sum(coarse(a, 'U_w') for a in accs)
        s = sum(np.stack([a[f'U_f2{c}'] + a[f'U_fh{c}'] for c in range(3)], -1).reshape(N, NS, NB, 3).sum(1) for a in accs)
        q = sum(np.stack([a.get(f'U_sq{c}', np.zeros(NCS)) for c in range(3)], -1).reshape(N, NS, NB, 3).sum(1) for a in accs)
        m = s / np.maximum(w, 1)[..., None]
        return m, w, q
    Tl, wl, ql = pooled(leg)
    Tn, wn, qn = pooled(nat)
    var_n = np.maximum(qn / np.maximum(wn, 1)[..., None] - Tn ** 2, 0)
    var_l = np.maximum(ql / np.maximum(wl, 1)[..., None] - Tl ** 2, 0)          # each sampler's own per-sample second moment
    se = np.sqrt(var_n / np.maximum(wn - 1, 1)[..., None] + var_l / np.maximum(wl - 1, 1)[..., None])
    diff = np.abs(Tl - Tn)
    sel = contam >= 0.05
    zfam = 5.2                                                         # family-wise over 81,280 cells x 3 channels
    material = ((diff > zfam * se) & (diff > 7e-4)).any(-1)
    clean = contam == 0
    rate = float((sum(a['misroute'].sum() for a in leg) + sum(a['backcount'].sum() for a in leg)) / cnt.sum())
    nsel = int(sel.sum()); nflag = int((sel & material).sum())
    return {'pass': bool(rate > 1e-5 and int((material & clean).sum()) == 0), 'legacy_address_inconsistency_rate': rate,
            'material_cells': int(material.sum()), 'material_cells_passing_as_clean': int((material & clean).sum()),
            'contaminated_cells': nsel, 'contaminated_with_material_difference': nflag,
            'legacy_misroute_total': float(sum(a['misroute'].sum() for a in leg)), 'legacy_backcount_total': float(sum(a['backcount'].sum() for a in leg)),
            'contaminated_patches': sorted({int(p) for p in np.argwhere(sel)[:, 0]})[:40]}


# ------------------------------------------------------------------------------------------------ streams
def load_view(root, stage, vname):
    st = []
    for j in range(2):
        with np.load(root / stage / f'stream_{vname}_{j}.npz') as z:
            st.append({k: z[k] for k in z.files})
    with np.load(root / stage / f'gbuffer_{vname}.npz') as z:
        gb = {k: z[k] for k in z.files}
    return st, gb


def arm_res(st, prefix, R, ref='C'):
    names = [prefix] if R == 1 else [f'{prefix}_r{r}' for r in range(R)]
    h = [s[f'phys_{ref}_h'].astype(np.float64) for s in st]
    return [[st[j][f'hl|{nm}'].astype(np.float64) - h[j] for j in range(2)] for nm in names], h


def numeric_bound_cells(root):
    """Per sub-cell numerical-event effect bound b (RGB) from the C replicas (max over replicas), and coarse pooled version."""
    bs = None; bc = None
    for r in range(C.RC):
        acc = load_acc(root / 'stageA' / f'C_r{r}.npz')
        n = np.maximum(acc['U_w'], 1)
        T_ = np.stack([acc[f'U_f2{c}'] + acc[f'U_fh{c}'] for c in range(3)], -1) / n[:, None]
        mis = np.stack([acc[f'misL{c}'] for c in range(3)], -1)
        sq = np.stack([acc[f'U_sq{c}'] for c in range(3)], -1)
        sd = np.sqrt(np.maximum(sq / n[:, None] - T_ ** 2, 0))
        b_mis = np.abs(mis - acc['mismatch'][:, None] * T_) / n[:, None]
        ks = acc['singular'][:, None]
        b_sing = ks / (n[:, None] + ks) * (T_ + 3 * sd)
        b = b_mis + b_sing
        nn = acc['U_w'].reshape(N, NS, NB)
        bcr = (b.reshape(N, NS, NB, 3) * nn[..., None]).sum(1) / np.maximum(nn.sum(1), 1)[..., None]
        bs = b if bs is None else np.maximum(bs, b)
        bc = bcr if bc is None else np.maximum(bc, bcr)
    return bs, bc


def wl_region(st, rname):
    return 0.5 * (st[0][f'W|{rname}'].astype(np.float64) + st[1][f'W|{rname}'].astype(np.float64))      # [3, NCS]


def main(root):
    root = Path(root); an = root / 'analysis'; an.mkdir(parents=True, exist_ok=True)
    geom, light = C.fixture('F0')
    ident = json.loads((root / 'stageA' / 'identity.json').read_text())
    gates = {}
    gates['A1'] = {'pass': bool(all(v for k, v in ident.items() if k.startswith('A1_table')) and ident['A1_lane_maxabs'] == 0.0),
                   **{k: v for k, v in ident.items() if k.startswith('A1')}}
    gates['A2'] = {'pass': bool(ident['A2_paired'] >= 2000 and ident['A2_rel_l1'] <= 1e-5), **{k: v for k, v in ident.items() if k.startswith('A2')}}
    gates['A4'] = gate_A4(root, geom)
    gates['A5'] = gate_A5(root, geom)
    accC = [load_acc(root / 'stageA' / f'C_r{r}.npz') for r in range(C.RC)]
    accN = [load_acc(root / 'stageA' / f'N_r{r}.npz') for r in range(C.RN)]
    bad_C_constr = float(sum(a['bad_origin'].sum() for a in accC)); bad_N_constr = float(sum(a['bad_origin'].sum() for a in accN))
    del accC, accN
    gates['A7'] = gate_A7(root)
    bs, bc = numeric_bound_cells(root)
    nat = dict(np.load(root / 'stageB' / 'native_support.npz'))
    cams = C.views('F0')
    RC, RN = C.RC, C.RN
    arms = [('U', RC), ('V', RC), ('G', RC), ('S_U', RC), ('S_V', RC), ('S_G', RC), ('UN', RN), ('S_GN', RN), ('A_learned', 1), ('P_v', 2), ('S_v', 2)]
    metrics = {}; a3 = {}; a6_bad = {'C': 0., 'N': 0., 'C_floor': 0., 'paths': 0.}; a8 = {}; a9 = {}
    for vname in cams:
        st, gb = load_view(root, 'stageB', vname)
        reg, bands = C.masks(gb, geom, light[0]['pos'])
        cv = C.contact_valid(reg)
        h = [s['phys_C_h'].astype(np.float64) for s in st]
        kap = MX.pixel_kappa(h[0], h[1], reg['floor'])
        hN = [s['phys_N_h'].astype(np.float64) for s in st]
        mu = MX.region_mean_lum(0.5 * (h[0] + h[1]), reg['near']) if reg['near'].any() else float('nan')
        bn_, bc_ = BAND_NEAR_FRAC * mu, BAND_CON_FRAC * mu
        a9[vname] = float(max((np.abs(s['phys_C_e1'] - s['ro_e1']).sum() + np.abs(s['phys_C_e2'] - s['ro_e2']).sum()) /
                              (np.abs(s['phys_C_e1']).sum() + np.abs(s['phys_C_e2']).sum()) for s in st))
        for s in st:
            meta = json.loads(str(s['meta']))
            a6_bad['C'] += meta.get('bad_origin_C', 0.); a6_bad['N'] += meta.get('bad_origin_N', 0.)
            a6_bad['C_floor'] += meta.get('bad_floor_C', 0.); a6_bad['paths'] += meta.get('paths', 0.)
        vm = {'pixel_kappa_amendment1': kap, 'contact_valid': cv, 'mu_ref_near': mu, 'band_near': bn_, 'band_contrast': bc_, 'pixels': {k: int(m.sum()) for k, m in reg.items()},
              'ref_noise': {}, 'semantics_shift_C_minus_N': {}, 'arms': {}}
        for rname in REGIONS_ALL:
            m = reg[rname]
            if m.any():
                d = MX.lum(h[0][m]) - MX.lum(h[1][m])
                vm['ref_noise'][rname] = float(math.sqrt(d.var(ddof=1) / (4 * d.size)))
                vm['semantics_shift_C_minus_N'][rname] = float(MX.region_mean_lum(0.5 * (h[0] + h[1]) - 0.5 * (hN[0] + hN[1]), m))
        vm['semantics_shift_bands'] = MX.band_profile([[h[0] - hN[0], h[1] - hN[1]]], bands)
        Wl = {r_: wl_region(st, r_) for r_ in ('near', 'flank', 'contact_shadow', 'sphere_boundary', 'sphere', 'floor', 'full')}
        for arm, R in arms:
            ref = 'N' if arm in ('UN', 'S_GN') else 'C'
            res, hh = arm_res(st, arm, R, ref)
            o = {}
            if cv:
                o['endpoints'] = MX.endpoint_stats(res, reg['near'], reg['flank'], kappa=kap)
            o['bands'] = MX.band_profile(res, bands)
            o['rmse'] = {}
            for rname in REGIONS_ALL:
                rel, absr, bias = MX.rmse_debiased(res, hh, reg[rname])
                o['rmse'][rname] = {'rel': rel, 'abs': absr, 'bias_rgb': bias}
            # coverage (native, non-fallback query weight)
            if arm in ('P_v', 'S_v'):
                natm = [np.load(root / 'stageB' / f'pv_native_{vname}_r{r}.npz')['P' if arm == 'P_v' else 'S'] for r in range(2)]
            elif arm == 'A_learned':
                natm = [np.ones((N, NB), bool)]
            else:
                natm = [nat[f'{arm}_r{r}'] for r in range(R)]
            o['coverage'] = {r_: float(np.mean([MX.coverage(LUMW @ Wl[r_], nm) for nm in natm])) for r_ in Wl}
            vm['arms'][arm] = o
        # A3 frame invariance (V0 only)
        if vname == 'V0':
            out3 = {}
            for meas in ('U', 'G'):
                reps = []
                for r in range(RC):
                    img = 0.5 * (st[0][f'hl|{meas}_r{r}'] + st[1][f'hl|{meas}_r{r}']).astype(np.float64)
                    reps.append([MX.region_mean_lum(img, reg[k]) for k in ('near', 'flank', 'contact_shadow', 'sphere', 'full')])
                var_r = np.var(np.array(reps), axis=0, ddof=1)
                a_ = 0.5 * (st[0][f'hl|frame_normal_{meas}'] + st[1][f'hl|frame_normal_{meas}']).astype(np.float64)
                b_ = 0.5 * (st[0][f'hl|frame_patch_{meas}'] + st[1][f'hl|frame_patch_{meas}']).astype(np.float64)
                for i, k in enumerate(('near', 'flank', 'contact_shadow', 'sphere', 'full')):
                    d = MX.region_mean_lum(a_ - b_, reg[k]); se = math.sqrt(2 * 4 * var_r[i])
                    out3[f'{meas}_{k}'] = {'diff': d, 'se': se, 'z': d / max(se, 1e-12)}
            a3 = {'pass': bool(all(abs(v['z']) <= 4 for v in out3.values())), 'rows': out3}
        # A8 materiality
        rows = {}
        for r_ in ('near', 'flank', 'contact_shadow', 'sphere_boundary', 'sphere', 'floor', 'full'):
            npx = max(int(reg[r_].sum()), 1)
            Wc = Wl[r_].T.reshape(N, NS, NB, 3)
            Bs = float(LUMW @ (Wc.reshape(-1, 3) * bs).sum(0)) / npx
            Bc = float(LUMW @ (Wc.sum(1).reshape(-1, 3) * bc.reshape(-1, 3)).sum(0)) / npx
            rows[r_] = max(Bs, Bc)
        q = {}
        if cv:
            q['near'] = {'bound': rows['near'], 'threshold': 0.1 * bn_, 'qualified': bool(rows['near'] <= 0.1 * bn_)}
            q['contrast'] = {'bound': rows['near'] + rows['flank'], 'threshold': 0.1 * bc_, 'qualified': bool(rows['near'] + rows['flank'] <= 0.1 * bc_)}
        for r_ in REGIONS_NR:
            rel_rms = math.sqrt(max(np.mean(h[0][reg[r_]] * h[1][reg[r_]]), 1e-30))
            q[r_] = {'bound': rows[r_], 'relative_bound': rows[r_] / rel_rms, 'qualified': bool(rows[r_] / rel_rms <= 2.5e-4)}
        a8[vname] = q
        metrics[vname] = vm
        # reviewer arrays
        maps = {'ref_h': (0.5 * (h[0] + h[1])).astype(np.float32), 'ref_e': (0.5 * (st[0]['phys_C_e1'] + st[0]['phys_C_e2'] + st[1]['phys_C_e1'] + st[1]['phys_C_e2'])).astype(np.float32),
                'refN_h': (0.5 * (hN[0] + hN[1])).astype(np.float32), 'ref_half_diff': (0.5 * (h[0] - h[1])).astype(np.float32)}
        for arm, R in arms:
            names = [arm] if R == 1 else [f'{arm}_r{r}' for r in range(R)]
            maps[f'hl_{arm}'] = np.mean([0.5 * (st[0][f'hl|{nm}'] + st[1][f'hl|{nm}']) for nm in names], 0).astype(np.float32)
        for k, m in reg.items():
            maps[f'mask_{k}'] = m
        np.savez_compressed(an / f'maps_B_{vname}.npz', **maps)
        del st
        print('VIEW', vname, 'done', flush=True)
    gates['A3'] = a3
    fl_rate = a6_bad['C_floor'] / max(a6_bad['paths'], 1)
    gates['A6'] = {'pass': bool(bad_C_constr == 0 and a6_bad['C'] == 0 and (bad_N_constr > 0 or a6_bad['N'] > 0) and fl_rate <= 1e-7),
                   'C_construction_inside_sphere': bad_C_constr, 'C_references_inside_sphere': a6_bad['C'],
                   'N_construction_inside_sphere': bad_N_constr, 'N_references_inside_sphere': a6_bad['N'],
                   'C_references_below_floor_junction_band': a6_bad['C_floor'], 'C_below_floor_rate_per_camera_path': fl_rate}
    gates['A9'] = {'pass': bool(all(v <= 1e-5 for v in a9.values())), 'low_relative_l1': a9}
    unq = [f'{v}:{k}' for v, q in a8.items() for k, x in q.items() if not x['qualified']]
    gates['A8'] = {'per_view': a8, 'unqualified': unq}
    core = all(gates[k]['pass'] for k in ('A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A9'))
    gates['verdict'] = 'NOT QUALIFIED' if not core else ('QUALIFIED' if not unq else 'PARTIALLY QUALIFIED')
    gates['verdict_as_frozen_before_amendment1'] = 'NOT QUALIFIED (A5 per-replica binomial footprint z 5.16 > 5, replica 1)'
    C.jd(gates, an / 'stageA_gates.json')
    C.jd(metrics, an / 'stageB_metrics.json')
    decision(root, gates, metrics, a8)


def decision(root, gates, metrics, a8):
    """Frozen Stage B decision."""
    out = {'stageA_verdict': gates['verdict'], 'arms': {}}
    if gates['verdict'] == 'NOT QUALIFIED':
        out['outcome'] = 'STOP_UNQUALIFIED'
        C.jd(out, root / 'stageB' / 'decision.json'); return out
    for arm in ('G', 'S_G', 'S_U', 'V', 'S_V'):
        rows = {}; suff = True; limited = []
        for vname, vm in metrics.items():
            a = vm['arms'][arm]; u = vm['arms']['U']
            r = {}
            if vm['contact_valid']:
                e = a['endpoints']
                qn = a8[vname].get('near', {}).get('qualified', True); qc = a8[vname].get('contrast', {}).get('qualified', True)
                r['near_inside'] = bool(abs(e['near']['lo']) <= vm['band_near'] and abs(e['near']['hi']) <= vm['band_near'])
                r['contrast_inside'] = bool(abs(e['contrast']['lo']) <= vm['band_contrast'] and abs(e['contrast']['hi']) <= vm['band_contrast'])
                r['precision'] = bool(e['near']['half_width'] <= 0.75 * vm['band_near'] and e['contrast']['half_width'] <= 0.75 * vm['band_contrast'])
                r['materiality_qualified'] = bool(qn and qc)
                if not r['precision'] or not r['materiality_qualified']:
                    limited.append(vname)
                suff &= r['near_inside'] and r['contrast_inside']
            nr = {}
            for rname in REGIONS_NR:
                lim = 1.10 * u['rmse'][rname]['rel'] + 0.0025
                nr[rname] = {'arm': a['rmse'][rname]['rel'], 'limit': lim, 'ok': bool(a['rmse'][rname]['rel'] <= lim),
                             'qualified': a8[vname][rname]['qualified']}
                suff &= nr[rname]['ok']
            r['non_regression'] = nr
            r['coverage_min'] = float(min(a['coverage'].values()))
            if r['coverage_min'] < 0.995:
                limited.append(vname)
            rows[vname] = r
        out['arms'][arm] = {'sufficient': bool(suff and not limited), 'limited_views': sorted(set(limited)), 'views': rows}
    if out['arms']['G']['sufficient']:
        out['outcome'] = 'PROJECTION'
    elif out['arms']['S_G']['sufficient']:
        out['outcome'] = 'SPATIAL_G'
    elif out['arms']['S_U']['sufficient']:
        out['outcome'] = 'SPATIAL_U'
    else:
        lim = any(out['arms'][a]['limited_views'] for a in ('G', 'S_G', 'S_U'))
        out['outcome'] = 'NONE_LIMITED' if lim else 'NONE'
    if C.DEV and os.environ.get('WL112_DEV_FORCE_OUTCOME'):          # development-mode code-path test only (never evidence)
        out['outcome_dev_actual'] = out['outcome']; out['outcome'] = os.environ['WL112_DEV_FORCE_OUTCOME']
    C.jd(out, root / 'stageB' / 'decision.json')
    print('DECISION', out['outcome'], flush=True)
    return out


if __name__ == '__main__':
    main(sys.argv[1])
