"""WL115 Stage A analysis (SELECTION set only; written before the Stage A streams were read).

  python analyze115A.py ROOT   -> analysis/stageA_metrics.json, analysis/stageA_summary.json, analysis/mapsA_<tag>.npz,
                                  analysis/fieldA_<tag>.npz (within-patch field / query distribution on the contact neighbourhood)
Gate quantities: metrics113.gate3 (approved 3 % contract, full depth, unchanged); the R5 RMSE non-regression control is PC4 (the dense
4 x 4 oracle). References: the WL114 Stage B deep physical streams (FULL = e1 + e2 + h + c7_12 + c13_24) of the same seeds.
"""
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'ecro_fd114')); sys.path.insert(0, str(HERE.parent / 'ecro_sss113'))
import common115 as C  # noqa: E402
import metrics113 as MX  # noqa: E402
import stageA115 as SA  # noqa: E402

K_STAR = 24
LIFT = 0.03


def lumv(x):
    return x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722


def load_view(root, tag):
    st = []
    for j in range(2):
        with np.load(C.WL114_ROOT / 'stageB' / f'stream_{tag}_{j}.npz') as z:
            s = {k: z[k] for k in z.files if k.startswith('phys_') or k in ('ro_e1', 'ro_e2', 'hl|S4_F')}
        s['tail'] = sum(s[f'phys_{n}'].astype(np.float64) for n in ('c7_12', 'c13_24'))
        with np.load(root / 'stageA' / 'streams' / f'stream_{tag}_{j}.npz') as z:
            for k in z.files:
                if k.startswith('hl|'):
                    s['new|' + k[3:]] = z[k]
            s['ro_e1_n'] = z['ro_e1']; s['ro_e2_n'] = z['ro_e2']
        st.append(s)
    with np.load(C.WL114_ROOT / 'stageB' / f'gbuffer_{tag}.npz') as z:
        gb = {k: z[k] for k in z.files}
    return st, gb


def window_map(Ms, Rs, kappa, size=5):
    from numpy.lib.stride_tricks import sliding_window_view as swv
    h = size // 2
    yM = [lumv(m) for m in Ms]; yR = [lumv(r) for r in Rs]
    d0, d1 = yM[0] - yR[0], yM[1] - yR[1]
    pad = lambda a: np.pad(a, h, mode='edge')
    md = swv(pad(0.5 * (d0 + d1)), (size, size)).mean((-1, -2)); mr = swv(pad(0.5 * (yR[0] + yR[1])), (size, size)).mean((-1, -2))
    vd = swv(pad(d0 - d1), (size, size)).var((-1, -2), ddof=1)
    return md / np.maximum(mr, 1e-9), np.sqrt(kappa * vd / (4 * size * size)) / np.maximum(mr, 1e-9)


def r1l_location(e, se, cen, g3, r):
    from scipy import stats
    nw = int(cen.sum())
    z = float(stats.norm.ppf(1 - 0.05 / (2 * max(nw, 1))))
    fail = cen & ((np.abs(e) - z * se) > MX.TAU)
    out = {}
    for nm, m in (('floor', g3['win_floor'] & ~g3['silhouette']), ('sphere_low', g3['win_sphere_low'] & ~g3['silhouette']), ('silhouette', g3['silhouette'])):
        f = fail & m
        out[nm] = {'failing': int(f.sum()), 'r_min': float(r[f].min()) if f.any() else None, 'r_max': float(r[f].max()) if f.any() else None,
                   'median_e': float(np.median(e[f])) if f.any() else None,
                   'max_abs_e': float(np.abs(e[cen & m]).max()) if (cen & m).any() else None}
    return out, fail


def view_metrics(root, tag, key, g, l, save_maps=False):
    st, gb = load_view(root, tag)
    g3, reg, bands = MX.gate_masks(gb, g, l[0]['pos'])
    cv = C.contact_valid(reg)
    h = [s['phys_h'].astype(np.float64) for s in st]
    kap = MX.pixel_kappa(h[0], h[1], reg['floor'])
    Rf = [st[j]['phys_e1'].astype(np.float64) + st[j]['phys_e2'] + h[j] + st[j]['tail'] for j in range(2)]
    pair = max(float(np.abs(s['ro_e1_n'] - s['ro_e1']).max() + np.abs(s['ro_e2_n'] - s['ro_e2']).max()) for s in st)
    pair_rel = max(float((np.abs(s['ro_e1_n'] - s['ro_e1']).sum() + np.abs(s['ro_e2_n'] - s['ro_e2']).sum()) /
                         (np.abs(s['ro_e1']).sum() + np.abs(s['ro_e2']).sum())) for s in st)
    auth = max(float(np.abs(s['new|S4F_wl114'] - s['hl|S4_F']).sum() / max(np.abs(s['hl|S4_F']).sum(), 1e-30)) for s in st)
    reps = [k[4:] for k in st[0] if k.startswith('new|')]
    Mc = {a: [st[j]['ro_e1'].astype(np.float64) + st[j]['ro_e2'] + st[j][f'new|{a}'] for j in range(2)] for a in reps}
    gap = float(g['S']['c'][2] - C.R_SPHERE)
    r = np.linalg.norm(gb['p'][..., :2] - np.asarray(g['S']['c'][:2]), axis=-1)
    cen = g3['win_floor'] | g3['win_sphere_low'] | g3['silhouette']
    vm = {'tag': tag, 'key': key, 'gap_m': gap, 'lifted': bool(gap >= LIFT), 'contact_valid': cv, 'kappa': kap,
          'low_pairing_max_abs': pair, 'low_pairing_rel_l1': pair_rel, 'kernel_auth_S4F_rel_l1': auth, 'reps': {}}
    maps = {}
    for a in reps:
        o = MX.gate3(Mc[a], Rf, g3, kap, ctrl=Mc.get('PC4'))
        e, se = window_map(Mc[a], Rf, kap)
        loc, fail = r1l_location(e, se, cen, g3, r)
        o['R1L_location'] = loc
        o['b_near'] = o['R1_near']['b'] if not math.isnan(o['R1_near'].get('b', float('nan'))) else None
        p = dict(o['pass'])
        if not cv:
            p = {k: v for k, v in p.items() if k.startswith('R5')}
        o['all_pass'] = bool(all(p.values()))
        vm['reps'][a] = o
        if save_maps:
            maps[f'e_{a}'] = e.astype(np.float32); maps[f'fail_{a}'] = fail
            maps[f'hl_{a}'] = (0.5 * (st[0][f'new|{a}'] + st[1][f'new|{a}'])).astype(np.float32)
    # table-noise estimate (capture views): half-table window errors
    for base in ('PC8', 'R32'):
        if f'{base}_h0' in reps:
            e0, _ = window_map(Mc[f'{base}_h0'], Rf, kap); e1, _ = window_map(Mc[f'{base}_h1'], Rf, kap)
            dn = 0.5 * np.abs(e0 - e1)[cen]
            vm[f'table_noise_{base}'] = {'window_noise_p50': float(np.median(dn)), 'window_noise_p99': float(np.percentile(dn, 99)),
                                         'window_noise_max': float(dn.max())}
    if save_maps:
        maps['ref_full'] = (0.5 * (Rf[0] + Rf[1])).astype(np.float32)
        maps['ref_high'] = (0.5 * (h[0] + h[1] + st[0]['tail'] + st[1]['tail'])).astype(np.float32)
        maps['low'] = (0.5 * (st[0]['ro_e1'] + st[0]['ro_e2'] + st[1]['ro_e1'] + st[1]['ro_e2'])).astype(np.float32)
        maps['r'] = r.astype(np.float32); maps['centres'] = cen
        for k in ('near', 'flank', 'floor', 'sphere'):
            maps[f'mask_{k}'] = reg[k]
        maps['silhouette'] = g3['silhouette']
        np.savez_compressed(root / 'analysis' / f'mapsA_{tag}.npz', **maps)
    return vm


def field_analysis(root, tag, key, dg, g):
    """Within-patch HIGH field on the contact neighbourhood from the capture (camera-conditioned, query-bin weighted) vs the U-measure
    32 x 32 / PC8 / PC4 values weighted by the same query bins: per fine cell of every contact patch."""
    import tracer115 as T
    with np.load(root / 'stageA' / 'capture' / f'cap_{tag}.npz') as z:
        cap = {k: z[k] for k in z.files}
    O = SA.oracle_tables(root, dg)
    pids = cap['pids']
    ctx = T.Ctx115(g, 'C'); G = ctx.G
    tq = np.stack([cap[f'Q32_t{c}'] for c in range(3)], -1).astype(np.float64)          # [P, 1024, NB, 3]
    hq = np.stack([cap[f'Q32_h{c}'] for c in range(3)], -1).astype(np.float64)
    tl = lumv(tq)                                                                       # query weight per (cell, bin)
    wsum = tl.sum(2)
    Lq = lumv(hq.sum(2)) / np.maximum(lumv(tq.sum(2)), 1e-30)
    def qavg(Tfine):
        return (tl * lumv(Tfine)).sum(2) / np.maximum(wsum, 1e-30)
    F32 = O['F32']
    PC8f = T.upsample(O['PC8'][pids], 8, 32); PC4f = T.upsample(O['PC4'][pids], 4, 32)
    out = {'pids': pids, 'n': cap['Q32_n'].sum(2), 'w': wsum, 'Lq': Lq, 'U32': qavg(F32), 'U8': qavg(PC8f), 'U4': qavg(PC4f)}
    # geometry of the fine cell centres: sphere polar angle from the bottom, floor horizontal distance to the support point
    Kf = 32
    a = (np.arange(Kf * Kf) % Kf + 0.5) / Kf; b = (np.arange(Kf * Kf) // Kf + 0.5) / Kf
    pos = np.zeros((len(pids), Kf * Kf, 3))
    for i, p in enumerate(pids):
        if G.PSID[p] == T.S.SID_SPHERE:
            zz = (G.PSPH[p, 1] + b) * (2.0 / T.S.SPH_NZ) - 1.0; ph = (G.PSPH[p, 0] + a) * (2 * math.pi / T.S.SPH_NPHI) - math.pi
            rr = np.sqrt(np.maximum(1 - zz * zz, 0))
            pos[i] = np.asarray(g['S']['c']) + C.R_SPHERE * np.stack([rr * np.cos(ph), rr * np.sin(ph), zz], -1)
        else:
            pos[i] = G.PO[p] + a[:, None] * G.PE1[p] + b[:, None] * G.PE2[p]
    sc = np.asarray(g['S']['c'])
    out['is_sphere'] = (G.PSID[pids] == T.S.SID_SPHERE)
    out['theta_deg'] = np.degrees(np.arccos(np.clip(-(pos[..., 2] - sc[2]) / C.R_SPHERE, -1, 1)))
    out['r_floor'] = np.linalg.norm(pos[..., :2] - sc[:2], axis=-1)
    out['pos'] = pos
    np.savez_compressed(root / 'analysis' / f'fieldA_{tag}.npz', **out)
    del ctx


def main(root):
    root = Path(root); an = root / 'analysis'; an.mkdir(parents=True, exist_ok=True)
    V = SA.views()
    res = {}
    for tag, (key, dg, g, l, cam, seeds) in V.items():
        res[tag] = view_metrics(root, tag, key, g, l, save_maps=tag in SA.CAPTURE_VIEWS or key in ('F1', 'F2', 'F3'))
        if tag in SA.CAPTURE_VIEWS:
            field_analysis(root, tag, key, dg, g)
        v = res[tag]
        print('A', tag, v['contact_valid'], {a: (v['reps'][a]['R1L']['failing'], round(100 * (v['reps'][a]['b_near'] or 0), 2)) for a in ('PC4', 'PC8', 'BL4', 'R32')}, flush=True)
    C.jd(res, an / 'stageA_metrics.json')
    # ---------------- summary over contact-valid SELECTION views
    reps = sorted({a for v in res.values() for a in v['reps']})
    cvs = [v for v in res.values() if v['contact_valid']]
    summ = {'n_views': len(res), 'n_contact_valid': len(cvs), 'n_lifted_cv': sum(v['lifted'] for v in cvs),
            'low_pairing_rel_l1_max': max(v['low_pairing_rel_l1'] for v in res.values()),
            'kernel_auth_S4F_rel_l1_max': max(v['kernel_auth_S4F_rel_l1'] for v in res.values()), 'reps': {}}
    for a in reps:
        rows = [v for v in cvs if a in v['reps']]
        if not rows:
            continue
        P = lambda req: int(sum(v['reps'][a]['pass'][req] for v in rows))
        ring = lambda i: [v['reps'][a]['R4_rings'][i]['b'] for v in rows if not math.isnan(v['reps'][a]['R4_rings'][i]['b'])]
        lif = [v['reps'][a]['b_near'] for v in rows if v['lifted']]; con = [v['reps'][a]['b_near'] for v in rows if not v['lifted']]
        summ['reps'][a] = {
            'n': len(rows), 'all_pass_views': int(sum(v['reps'][a]['all_pass'] for v in rows)),
            'pass_counts': {k: P(k) for k in ('R1', 'R1L', 'R2', 'R3_bias', 'R4', 'R5a_bias', 'R5b_bias', 'R5a_rmse', 'R5b_rmse')},
            'R1L_failing_total': int(sum(v['reps'][a]['R1L']['failing'] for v in rows)),
            'R1L_failing_by_region': {nm: int(sum(v['reps'][a]['R1L_location'][nm]['failing'] for v in rows)) for nm in ('floor', 'sphere_low', 'silhouette')},
            'R1L_max_abs_e_median': float(np.median([v['reps'][a]['R1L']['max_abs_e'] for v in rows])),
            'R1L_max_abs_e_max': float(np.max([v['reps'][a]['R1L']['max_abs_e'] for v in rows])),
            'ring0_mean': float(np.mean(ring(0))) if ring(0) else None, 'ring0_max_abs': float(np.max(np.abs(ring(0)))) if ring(0) else None,
            'ring1_mean': float(np.mean(ring(1))), 'ring1_max_abs': float(np.max(np.abs(ring(1)))),
            'near_bias_contact_mean': float(np.mean(con)) if con else None, 'near_bias_lifted_mean': float(np.mean(lif)) if lif else None,
            'near_bias_lifted_max': float(np.max(lif)) if lif else None, 'near_bias_abs_max': float(np.max(np.abs(con + lif)))}
    for base in ('PC8', 'R32'):
        tn = [v[f'table_noise_{base}'] for v in res.values() if f'table_noise_{base}' in v]
        if tn:
            summ[f'table_noise_{base}'] = {k: float(np.max([x[k] for x in tn])) for k in tn[0]}
    C.jd(summ, an / 'stageA_summary.json')
    print(json.dumps({a: {k: summ['reps'][a][k] for k in ('all_pass_views', 'R1L_failing_total', 'R1L_max_abs_e_median', 'ring0_mean', 'near_bias_lifted_mean')}
                      for a in summ['reps']}, indent=1), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
    sys.stdout.flush()
    import os; os._exit(0)
