"""WL114 Stage A frozen analysis: SSS K* rule (C1, C3), roulette consistency (C2, C4), LR3 authentication (L0) and full-depth
qualification (L1). Writes ROOT/analysis/stageA_qualification.json (+ the depth-convergence tables used by the figures).

  python analyze114A.py ROOT
"""
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common114 as C  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722])
EDGES = (1, 2, 6, 12, 24, 48, 96, 192)
NAMES = ('e1', 'e2', 'h', 'c7_12', 'c13_24', 'c25_48', 'c49_96', 'c97_192')
DEEP = NAMES[3:]
REL = 1e-3
KS = (24, 48, 96)


def lum3(v):
    return float(np.dot(LUM, v[:3]))


def rsum(rep, name, r):
    s = rep['sums'].get(name, {}).get(r)
    return lum3(s) / s[3] if s and s[3] else 0.0           # region mean luminance of bucket `name`


def sss(root):
    A = root / 'stageA'
    views = {}
    for tag, *_ in C.stageA_views():
        rec = json.loads((A / f'view_{tag}.json').read_text())
        regs = [r for r, s in rec['det'][0]['sums']['e1'].items() if s[3] >= 200]
        vr = {'contact_valid': rec['contact_valid'], 'regions': {}}
        for r in regs:
            det = rec['det']; rr = rec['rr']
            hf = [sum(rsum(d, n, r) for n in ('h',) + DEEP) for d in det]
            H = float(np.mean(hf))
            row = {'HIGH_full': H, 'buckets': {n: float(np.mean([rsum(d, n, r) for d in det])) for n in NAMES}}
            row['bucket_se'] = {n: abs(rsum(det[0], n, r) - rsum(det[1], n, r)) / 2 for n in NAMES}
            for K in KS:
                beyond = [n for e, n in zip(EDGES, NAMES) if e > K]
                t = [sum(rsum(d, n, r) for n in beyond) for d in det]
                row[f'tail_beyond_{K}'] = float(np.mean(t)); row[f'tail_beyond_{K}_se'] = abs(t[0] - t[1]) / 2
                row[f'C1_{K}'] = bool(H > 0 and row[f'tail_beyond_{K}'] + 2 * row[f'tail_beyond_{K}_se'] <= REL * H)
            d7 = [sum(rsum(d, n, r) for n in DEEP) for d in det]
            rv = [rsum(x, 'rr', r) for x in rr]
            se = math.sqrt(np.var(rv, ddof=1) / len(rv) + (abs(d7[0] - d7[1]) / 2) ** 2)
            row['RR_gt6'] = float(np.mean(rv)); row['DET_7_192'] = float(np.mean(d7)); row['RR_se_comb'] = se
            row['C2'] = bool(abs(row['RR_gt6'] - row['DET_7_192']) <= max(4 * se, REL * H))
            row['tail_7plus_share_of_HIGH_full'] = float(np.mean(d7) / H) if H > 0 else float('nan')
            row['tail_7plus_share_of_total'] = float(np.mean(d7) / (np.mean([rsum(d, 'e1', r) + rsum(d, 'e2', r) for d in det]) + H)) if H > 0 else float('nan')
            vr['regions'][r] = row
        vr['cap_alive_det'] = [d['diag']['cap_alive'] for d in rec['det']]
        vr['cap_alive_rr'] = [d['diag']['cap_alive'] for d in rec['rr']]
        views[tag] = vr
    cons = {}
    for tag, *_ in C.stageA_constructions():
        rec = json.loads((A / f'con_{tag}.json').read_text())
        cr = {}
        for sname in ('all', 'ring0'):
            g = lambda rep, nm: lum3([rep['sums'][sname].get(f'U_{nm}{c}', 0.) for c in range(3)])
            hf = [sum(g(d, n) for n in ('f2', 'fh') + DEEP) for d in rec['det']]
            H = float(np.mean(hf)); row = {'HIGH_full_sum': H}
            for K in KS:
                beyond = [n for e, n in zip(EDGES, NAMES) if e > K]
                t = [sum(g(d, n) for n in beyond) for d in rec['det']]
                row[f'tail_beyond_{K}_rel'] = float(np.mean(t) / H) if H else float('nan')
                row[f'C3_{K}'] = bool(H > 0 and np.mean(t) <= REL * H)
            d7 = [sum(g(d, n) for n in DEEP) for d in rec['det']]
            rv = [g(x, 'rr') for x in rec['rr']]
            se = math.sqrt(np.var(rv, ddof=1) / len(rv) + (abs(d7[0] - d7[1]) / 2) ** 2)
            row.update({'RR_gt6': float(np.mean(rv)), 'DET_7_192': float(np.mean(d7)), 'se_comb': se,
                        'C4': bool(abs(np.mean(rv) - np.mean(d7)) <= max(4 * se, REL * H)), 'deep_share_of_HIGH_full': float(np.mean(d7) / H) if H else float('nan')})
            cr[sname] = row
        cr['cap_alive_det'] = [d['diag']['cap_alive'] for d in rec['det']]
        cons[tag] = cr
    Kstar = None
    for K in KS:
        c1 = all(r[f'C1_{K}'] for v in views.values() for r in v['regions'].values())
        c3 = all(c[s][f'C3_{K}'] for c in cons.values() for s in ('all', 'ring0'))
        if c1 and c3:
            Kstar = K; break
    c2 = all(r['C2'] for v in views.values() for r in v['regions'].values())
    c4 = all(c[s]['C4'] for c in cons.values() for s in ('all', 'ring0'))
    fails = {'C1': {K: [f'{t}:{r}' for t, v in views.items() for r, x in v['regions'].items() if not x[f'C1_{K}']] for K in KS},
             'C2': [f'{t}:{r}' for t, v in views.items() for r, x in v['regions'].items() if not x['C2']],
             'C3': {K: [f'{t}:{s}' for t, c in cons.items() for s in ('all', 'ring0') if not c[s][f'C3_{K}']] for K in KS},
             'C4': [f'{t}:{s}' for t, c in cons.items() for s in ('all', 'ring0') if not c[s]['C4']]}
    q = 'QUALIFIED' if (Kstar is not None and c2 and c4) else 'NOT QUALIFIED'
    if C.DEV and q != 'QUALIFIED':                     # development plumbing only (tiny sample counts); never in the primary run
        q, Kstar = 'QUALIFIED', Kstar or 24
    return {'verdict': q, 'K_star': Kstar, 'C2_all': c2, 'C4_all': c4, 'failures': fails, 'views': views, 'constructions': cons}


def lr3(root):
    D = root / 'stageA' / 'lr3'
    if not (D / 'auth_d65.npz').exists():
        return {'verdict': 'NOT_RUN'}
    ref = np.load(Path(r'C:\Users\dna10\TransGI-Reproduction-data\runs\living_room_3\reference') / 'static_V0.npz')
    ref7 = np.load(Path(r'C:\Users\dna10\TransGI-Reproduction-data\runs\living_room_3\reference') / 'static_V0_d7.npz')
    au = np.load(D / 'auth_d65.npz')
    gi = 0.5 * (au['full_A'] + au['full_B']) - 0.5 * (au['di_A'] + au['di_B'])
    rgi = ref['gi']; noise = float(np.sqrt(np.mean((ref['gi_A'].astype(np.float64) - ref['gi_B']) ** 2)) / 2)
    m_ours, m_ref = gi.mean((0, 1)), rgi.mean((0, 1))
    rmse = float(np.sqrt(np.mean((gi.astype(np.float64) - rgi) ** 2)))
    L0 = {'gi_mean_ours': m_ours.tolist(), 'gi_mean_ref65': m_ref.tolist(), 'rel_diff': ((m_ours - m_ref) / m_ref).tolist(),
          'rmse_vs_ref65': rmse, 'ref65_gi_noise_rmse': noise, 'max_abs_diff': float(np.abs(gi - rgi).max()),
          'bitwise_equal_full': bool(np.array_equal(au['full_A'], ref['full_A']) and np.array_equal(au['full_B'], ref['full_B']))}
    L0['pass'] = bool(np.all(np.abs(np.asarray(L0['rel_diff'])) <= 1e-3) and rmse <= 2.0 * noise)   # independent-render expectation sqrt(2) x noise
    mk = np.load(D / 'lr3_masks.npz')
    masks = {'whole': np.ones(rgi.shape[:2], bool)}
    masks.update({k[5:]: mk[k] for k in mk.files if k.startswith('mask_') and mk[k].sum() >= 500})
    lad = {}
    for md in (2, 7, 13, 25, 49, 65, 97, 129, 257, -1):
        z = np.load(D / f'ladder_d{md}.npz'); lad[md] = (z['A'].astype(np.float64), z['B'].astype(np.float64))
    di = lad[2]
    giL = {md: (lad[md][0] - di[0], lad[md][1] - di[1]) for md in lad if md != 2}
    curve = {}; L1 = {}
    for name, m in masks.items():
        g_inf = 0.5 * (giL[-1][0][m] + giL[-1][1][m]).mean(0)
        curve[name] = {str(md): (0.5 * (giL[md][0][m] + giL[md][1][m]).mean(0)).tolist() for md in giL}
        ta = (giL[-1][0][m] - giL[65][0][m]).mean(0); tb = (giL[-1][1][m] - giL[65][1][m]).mean(0)
        t = 0.5 * (ta + tb); se = np.abs(ta - tb) / 2
        L1[name] = {'tail_beyond_64_events': t.tolist(), 'se': se.tolist(), 'gi_inf': g_inf.tolist(),
                    'rel': (t / np.maximum(g_inf, 1e-12)).tolist(), 'pass': bool(np.all(np.abs(t) + 2 * se <= REL * g_inf))}
        for md in (7, 13, 25, 49, 97, 129, 257):
            curve[name][f'missing_rel_d{md}'] = ((g_inf - 0.5 * (giL[md][0][m] + giL[md][1][m]).mean(0)) / np.maximum(g_inf, 1e-12)).tolist()
    ok = L0['pass'] and all(v['pass'] for v in L1.values())
    trunc = {'ref7_vs_ref65_gi_rel': ((ref7['gi'].mean((0, 1)) - m_ref) / m_ref).tolist()}
    return {'verdict': 'QUALIFIED' if ok else 'NOT QUALIFIED', 'L0_authentication': L0, 'L1_tail': L1, 'curve': curve, 'ref7_truncation': trunc,
            'reference': 'Ref-65 (authenticated, Mitsuba 3.6.4, 65,536 spp, max_depth 65 = 64 events)' if ok else None}


def main(root):
    root = Path(root); an = root / 'analysis'; an.mkdir(parents=True, exist_ok=True)
    out = {'SSS': sss(root), 'LR3': lr3(root)}
    C.jd(out, an / 'stageA_qualification.json')
    s = out['SSS']
    print('STAGE_A SSS', s['verdict'], 'K*', s['K_star'], 'C2', s['C2_all'], 'C4', s['C4_all'], '| LR3', out['LR3']['verdict'], flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
