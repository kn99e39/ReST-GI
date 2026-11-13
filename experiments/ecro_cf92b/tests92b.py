"""WL92B focused tests (protocol 'tests'), after the counterfactual tables and before any render.
  python tests92b.py <R> <R86> [--dev]   -> results/tests92b.json ; prints TESTS PASS | TESTS FAIL
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'ecro_attr92'))
DEV = '--dev' in sys.argv
TAGS = ['E2_T1', 'E2_T0', 'E1_T1_LSPOT', 'E4_T1', 'E4_T0']


def main():
    R, R86 = sys.argv[1], sys.argv[2]
    out = {}
    metas = {t: json.load(open(os.path.join(R, 'results', 'tables', f'CF_{t}.json'))) for t in TAGS if os.path.exists(os.path.join(R, 'results', 'tables', f'CF_{t}.json'))}
    idm = max(v for m in metas.values() for v in m['cf_identity'].values())
    out['T_CF_IDENTITY'] = {'max': idm, 'tags': list(metas), 'PASS': bool(idm <= 1e-5 and len(metas) == len(TAGS))}
    lin = [v for m in metas.values() for v in m['linear'].values()]
    out['T_LINEAR'] = {'max_scale': max(x['X_M12_scale'] for x in lin), 'min_value': min(x['min'] for x in lin),
                       'PASS': bool(max(x['X_M12_scale'] for x in lin) <= 1e-4 and min(x['min'] for x in lin) >= 0)}
    # T_EVENT on the new split (E1_T1_LSPOT): parts >= 0 and sum = a fresh oracle field within noise (flux 1 %)
    D = np.load(os.path.join(R, 'results', 'fields', 'D_E1_T1_LSPOT.npz'))
    parts = ('e2_p', 'e2_o', 'e3_p', 'e3_o')
    mn = min(float(D[f'{p}_{h}'].min()) for p in parts for h in 'ab')
    sa = sum(D[f'{p}_a'].astype(np.float64) for p in parts); sb = sum(D[f'{p}_b'].astype(np.float64) for p in parts)
    F = np.load(os.path.join(R, 'results', 'fields', 'F_E1_T1_LSPOT.npz'))
    fa, fb = F['fa'].astype(np.float64), F['fb'].astype(np.float64)
    flux = float((sa + sb).sum() / (fa + fb).sum())
    rl = float(np.abs(0.5 * (sa + sb) - 0.5 * (fa + fb)).sum() / (0.5 * (fa + fb)).sum())
    nz = float(np.abs(sa - sb).sum() / (sa + sb).sum() + np.abs(fa - fb).sum() / (fa + fb).sum())
    out['T_EVENT'] = {'min_part': mn, 'flux_ratio': flux, 'rel_l1': rl, 'half_split_noise': nz, 'PASS': bool(mn >= 0 and (DEV or (abs(flux - 1) <= 0.01 and rl <= 1.5 * nz)))}
    # T_DISJOINT: recompute from the data files
    cj = os.path.join(R, 'results', 'e5_candidates.json')
    if os.path.exists(cj):
        cand = json.load(open(cj))['geometry_admissible']
        tr = {'A': [], 'B': []}
        for sp in ('train', 'val', 'trainhq'):
            for fn in sorted(glob.glob(os.path.join(R86, 'data', sp, 'g*.npz'))):
                g = json.loads(str(np.load(fn)['geom']))
                for b in 'AB':
                    tr[b].append(g[b]['c'])
        tr = {b: np.array(v) for b, v in tr.items()}
        dm = [min(float(np.linalg.norm(tr[b] - np.array(c[t][b]['c']), axis=1).min()) for t in ('T0', 'T1') for b in 'AB') for c in cand]
        out['T_DISJOINT'] = {'min_box_distance_m': dm, 'PASS': bool(DEV or all(d >= 0.15 for d in dm))}
    else:
        out['T_DISJOINT'] = {'PASS': False, 'note': 'no candidates'}
    rp = json.load(open(os.path.join(R, 'results', 'repro.json')))
    out['T_REPRO'] = {'PASS': rp['PASS']}
    out['ALL_PASS'] = bool(all(v['PASS'] for v in out.values() if isinstance(v, dict)))
    json.dump(out, open(os.path.join(R, 'results', 'tests92b.json'), 'w'), indent=1)
    for k, v in out.items():
        if isinstance(v, dict):
            print(k, 'PASS' if v['PASS'] else 'FAIL', json.dumps(v)[:250], flush=True)
    print('TESTS', 'PASS' if out['ALL_PASS'] else 'FAIL', flush=True)


if __name__ == '__main__':
    main()
