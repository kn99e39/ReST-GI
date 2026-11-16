"""WL97 audit, track C1: independent recomputation of the WL92 (= WL86 gate F2) region-C E_H from raw renders, with MC standard error,
L1 noise floor, disjoint-half values and the composition of region C. numpy only; reads ~/rest_ecro86 and ~/rest_x92 read-only.

  python c1_wl92_eh.py <R86> <R92> <out.json>
E_H = sum_m |hl - h| / sum_m tot, tot = e1 + e2 + h (reference replicate mean), sums over pixels of mask m and channels.
"""
import glob, json, os, sys
import numpy as np

R86, R92, OUTFN = sys.argv[1:4]


def st(root, pat, key):
    f = sorted(glob.glob(os.path.join(root, 'results', 'renders', pat + '_r*.npz')))
    return np.stack([np.load(x)[key].astype(np.float64) for x in f]) if f else None


M = np.load(os.path.join(R86, 'results', 'masks', 'E2.npz'))
out = {'region_C': {'pixels': int(M['C'].sum()), 'on_panel': int((M['C'] & M['obj_panel']).sum()), 'on_sphere': int((M['C'] & M['obj_sphere']).sum()),
                    'on_boxes_static': int((M['C'] & M['obj_boxes_static']).sum()), 'C_mirror': int(M['C_mirror'].sum()), 'C_footprint': int(M['C_footprint'].sum())}, 'E_H': {}}
ARMS = ['B_ecro_s8601', 'B_ecro_s8602', 'L_R16', 'L_ENCB', 'ORA2', 'P_R16', 'P_ENCB', 'SL_ecro_s8601', 'SL_ecro_s8602', 'LG_ecro_s8601', 'LG_ecro_s8602']
for tag in ('E2_T1', 'E2_T0'):
    h = st(R86, f'REF_ER_{tag}', 'h'); tot = (st(R86, f'REF_ER_{tag}', 'e1') + st(R86, f'REF_ER_{tag}', 'e2') + h).mean(0)
    n = h.shape[0]; hm = h.mean(0); vh = h.var(0, ddof=1) / n
    for arm in ARMS + ['WL86_EC_s8601', 'WL86_EC_s8602']:
        x = st(R86, f"EC_ecro_{arm[-5:]}_ecro_{tag}", 'hl') if arm.startswith('WL86') else st(R92, f'X92_{arm}_{tag}', 'hl')
        if x is None:
            continue
        k = x.shape[0]; xm = x.mean(0); vx = x.var(0, ddof=1) / k
        for rn, m in (('C', M['C']), ('C_panel', M['C'] & M['obj_panel']), ('C_sphere', M['C'] & M['obj_sphere'])):
            den = tot[m].sum()
            if den <= 0:
                continue
            e = float(np.abs(xm[m] - hm[m]).sum() / den)
            out['E_H'][f'{tag}|{arm}|{rn}'] = {
                'E_H': e, 'signed': float((xm[m] - hm[m]).sum() / den), 'H_share': float(hm[m].sum() / den), 'pixels': int(m.sum()),
                'nb_coded': float((0.5 * np.abs(x[:k // 2].mean(0)[m] - x[k // 2:].mean(0)[m]).sum() + 0.5 * np.abs(h[:n // 2].mean(0)[m] - h[n // 2:].mean(0)[m]).sum()) / den),
                'se_mc': float(np.sqrt((vx[m] + vh[m]).sum()) / den), 'floor': float(np.sqrt(2 / np.pi * (vx[m] + vh[m])).sum() / den),
                'half1': float(np.abs(x[0][m] - h[:n // 2].mean(0)[m]).sum() / den), 'half2': float(np.abs(x[1][m] - h[n // 2:].mean(0)[m]).sum() / den),
                'ref_reps': int(n), 'arm_reps': int(k)}
        o = out['E_H'][f'{tag}|{arm}|C']
        print(tag, arm, {a: round(b, 5) for a, b in o.items() if a in ('E_H', 'signed', 'nb_coded', 'se_mc', 'floor', 'half1', 'half2')},
              'panel', round(out['E_H'].get(f'{tag}|{arm}|C_panel', {}).get('E_H', float('nan')), 4), 'sphere', round(out['E_H'].get(f'{tag}|{arm}|C_sphere', {}).get('E_H', float('nan')), 4), flush=True)
json.dump(out, open(OUTFN, 'w'), indent=1)
print(json.dumps(out['region_C']))
