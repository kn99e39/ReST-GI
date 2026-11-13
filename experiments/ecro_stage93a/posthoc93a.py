"""WL93A post hoc descriptive diagnostic (labelled; verdict, rules and criteria unchanged).

  python posthoc93a.py <R> <R86>   -> results/posthoc93a.json

On validation (and train-representative / sharp for context): per-geometry M and relative L1 of T0 / TQ vs the B5 reference,
dM per geometry, share of the pooled dM numerator change per geometry, and pooled dM (M and relative L1) on the Task-A
predeclared high-confidence subgroup (per-geometry B5 noise <= 0.10; declared for Task A, applied post hoc to Task B).
"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qref93a as QR
import eval93a as EV


def main():
    R, R86 = sys.argv[1:3]
    C = json.load(open(os.path.join(R, 'results', 'configs.json')))['configs']
    TA = json.load(open(os.path.join(R, 'results', 'taskA.json')))
    out = {'label': 'POST HOC DESCRIPTIVE (verdict, rules and criteria unchanged)', 'groups': {}}
    for gname in ('val', 'train_rep', 'sharp'):
        ids = QR.group(C, gname)
        st = {}; rows = []
        for i in ids:
            Qr = QR.load_ref(R, i, 'B5'); X = EV.load_stage(R, i); pm = EV.panel_mask(R86, C[i])
            r = {'id': C[i].get('file', C[i].get('tag')), 'N_B5': TA['per_cfg'][str(i)]['N_B5']}
            for s in EV.SEEDS:
                for a in ('T0', 'TQ'):
                    n = f'{a}_s{s}'; x = QR.stat(X[n], Qr, pm); st.setdefault(n, []).append(x)
                    r[f'M_{n}'] = QR.M_of([x], [0]); r[f'l1_{n}'] = x['l1'] / x['sq']; r[f'num_{n}'] = x['num0']
            rows.append(r)
        g = {'per_geometry': rows}
        for s in EV.SEEDS:
            dl1 = np.array([r[f'l1_TQ_s{s}'] - r[f'l1_T0_s{s}'] for r in rows]); dM = np.array([r[f'M_TQ_s{s}'] - r[f'M_T0_s{s}'] for r in rows])
            dn = np.array([r[f'num_TQ_s{s}'] - r[f'num_T0_s{s}'] for r in rows])
            k = int(np.argmax(np.abs(dn)))
            g[f's{s}'] = {'n': len(rows), 'geometries_TQ_better_M': int((dM < 0).sum()), 'geometries_TQ_better_l1': int((dl1 < 0).sum()),
                          'median_dM': float(np.median(dM)), 'median_d_rel_l1': float(np.median(dl1)),
                          'largest_numerator_change': {'id': rows[k]['id'], 'share_of_total_abs_change': float(abs(dn[k]) / max(np.abs(dn).sum(), 1e-30)), 'sign': float(np.sign(dn[k]))},
                          'pooled_dM_full': QR.M_of(st[f'TQ_s{s}'], range(len(rows))) - QR.M_of(st[f'T0_s{s}'], range(len(rows)))}
            hc = [j for j, r in enumerate(rows) if r['N_B5'] <= 0.10]
            if hc:
                a, b = [st[f'TQ_s{s}'][j] for j in hc], [st[f'T0_s{s}'][j] for j in hc]
                pa, pb = QR.pooled(a), QR.pooled(b)
                g[f's{s}']['high_confidence'] = {'n': len(hc), 'excluded': [rows[j]['id'] for j in range(len(rows)) if j not in hc],
                                                 'M_T0': pb['M'], 'M_TQ': pa['M'], 'dM': pa['M'] - pb['M'], 'boot': QR.boot(a, other=b),
                                                 'rel_l1_T0': pb['rel_l1'], 'rel_l1_TQ': pa['rel_l1']}
        out['groups'][gname] = g
    json.dump(out, open(os.path.join(R, 'results', 'posthoc93a.json'), 'w'), indent=1)
    for gname, g in out['groups'].items():
        print('POSTHOC', gname, json.dumps({s: {k: v for k, v in g[f's{s}'].items() if k != 'high_confidence'} for s in EV.SEEDS}), flush=True)
        for s in EV.SEEDS:
            if 'high_confidence' in g[f's{s}']:
                h = g[f's{s}']['high_confidence']
                print('  HC', gname, s, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items() if k != 'excluded'}), 'excluded', h['excluded'], flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
