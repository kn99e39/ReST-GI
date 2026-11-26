"""WL115 Stage D / E frozen analysis (protocol stage_B_frozen.stage_D_E).

  python analyzeD115.py ROOT  -> analysis/stageD_metrics.json, analysis/stageE_metrics.json, analysis/verdicts.json, analysis/mapsD_<tag>.npz
Approved 3 % gate quantities at full depth (metrics113.gate3 unchanged). R5 RMSE control for CAR arms = the matched B arm.
"""
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
for d in (HERE, HERE.parent / 'ecro_fd114', HERE.parent / 'ecro_sss113'):
    sys.path.insert(0, str(d))
import common115 as C  # noqa: E402
import metrics113 as MX  # noqa: E402
import evalD115 as ED  # noqa: E402
import analyzeB115 as AB  # noqa: E402

PAIRS = (('CAR1', 'B1'), ('CAR2', 'B2'))
TAU = MX.TAU


def load(root, tag, v):
    vv = dict(v, cls='FRESH' if v['cls'] == 'FRESH' else 'HELD_SEEN')        # SELECTION references are the WL114 Stage B streams too
    st, gb = AB.load_refs(root, tag, vv)
    ex = []
    for j in range(2):
        with np.load(root / 'stageD' / f'stream_{tag}_{j}.npz') as z:
            e = {k: z[k] for k in z.files}
        src = {'HELD_SEEN': root / 'stageB' / 'streams', 'FRESH': root / 'stageB' / 'streams', 'SELECTION': root / 'stageA' / 'streams'}[v['cls']]
        p = src / f'stream_{tag}_{j}.npz'
        if p.exists():
            with np.load(p) as z:
                for k in z.files:
                    if k in ('hl|PC4_O', 'hl|CAR_O', 'hl|PC4', 'hl|R16'):
                        e[k.replace('hl|PC4', 'hl|PC4_O') if k == 'hl|PC4' else k] = z[k]
                assert np.abs(z['ro_e1'] - e['ro_e1']).max() <= 1e-4 * max(np.abs(e['ro_e1']).max(), 1e-9) + 1e-6, 'LOW pairing'
        ex.append(e)
    return st, gb, ex


def view(root, tag, v):
    st, gb, ex = load(root, tag, v)
    arms = [k[3:] for k in ex[0] if k.startswith('hl|')]
    out, g3, reg, Rf, Mc, kap = AB.view_gate(st, gb, v, arms, ex)
    h = [st[j]['phys_h'].astype(np.float64) + st[j]['phys_c7_12'] + st[j]['phys_c13_24'] for j in range(2)]
    if reg['near'].any():
        out['mu_near_high_full'] = MX.region_mean_lum(0.5 * (h[0] + h[1]), reg['near'])
        d = MX.lum(h[0][reg['near']]) - MX.lum(h[1][reg['near']])
        out['ref_near_high_se'] = float(math.sqrt(kap * d.var(ddof=1) / (4 * d.size)))
    for a in arms:
        ctrl = Mc.get({'CAR1': 'B1', 'CAR2': 'B2'}.get(a, ''), None)
        o = AB.arm_metrics(Mc[a], Rf, g3, kap, ctrl, out['contact_valid'])
        if reg['near'].any():
            res = [ex[j][f'hl|{a}'].astype(np.float64) - h[j] for j in range(2)]
            o['near_high_residual'] = float(MX.lum(0.5 * (res[0] + res[1])[reg['near']]).mean())
            o['rmse_far'] = MX.rmse_debiased([res], h, reg['far_unchanged'])[0] if 'far_unchanged' in reg else None
        out['arms'][a] = o
    out['cls'] = v['cls']; out['key'] = v['key']; out['seq'] = v['seq']
    return out, Rf, Mc, reg


def h2(rows, car, b):
    rows = [r for r in rows if r['contact_valid'] and car in r['arms'] and b in r['arms'] and 'CAR_O' in r['arms'] and 'PC4_O' in r['arms']]
    exc = [abs(r['arms'][car]['b_near'] - r['arms']['CAR_O']['b_near']) <= 1.25 * abs(r['arms'][b]['b_near'] - r['arms']['PC4_O']['b_near']) + 0.005 for r in rows]
    tot = lambda a: sum(r['arms'][a]['R1L']['failing'] for r in rows)
    r0 = lambda a: float(np.mean([abs(r['arms'][a]['R4_rings'][0]['b']) for r in rows if not math.isnan(r['arms'][a]['R4_rings'][0]['b'])]))
    lif = lambda a: float(np.mean([abs(r['arms'][a]['b_near']) for r in rows if r['lifted']])) if any(r['lifted'] for r in rows) else float('nan')
    red = lambda x, y: 1 - y / x if x > 0 else float('nan')
    rm = [r['arms'][car]['pass'].get('R5a_rmse', True) and r['arms'][car]['pass'].get('R5b_rmse', True) for r in rows]
    o = {'n': len(rows), 'a_excess_frac': float(np.mean(exc)), 'b_R1L_reduction': red(tot(b), tot(car)), 'R1L_total': {car: tot(car), b: tot(b)},
         'c_ring0_reduction': red(r0(b), r0(car)), 'ring0_abs_mean': {car: r0(car), b: r0(b)}, 'd_lifted_reduction': red(lif(b), lif(car)),
         'lifted_abs_mean': {car: lif(car), b: lif(b)}, 'e_rmse_ok_frac': float(np.mean(rm))}
    o['pass'] = {'a': o['a_excess_frac'] >= 0.9, 'b': o['b_R1L_reduction'] >= 0.5, 'c': o['c_ring0_reduction'] >= 0.25, 'd': o['d_lifted_reduction'] >= 0.25,
                 'e': o['e_rmse_ok_frac'] == 1.0}
    return o


def main(root):
    root = Path(root); an = root / 'analysis'; an.mkdir(parents=True, exist_ok=True)
    V = ED.all_views(root); D = {}
    for tag, v in V.items():
        out, Rf, Mc, reg = view(root, tag, v)
        D[tag] = out
        if v['cls'] != 'SELECTION' and (v['seq'] is None or v['key'].endswith(('_0', '_3', '_4', '_5'))):
            m = lambda x: (0.5 * (x[0] + x[1])).astype(np.float32)
            np.savez_compressed(an / f'mapsD_{tag}.npz', ref_full=m(Rf), **{f'full_{a}': m(Mc[a]) for a in Mc}, mask_near=reg['near'], mask_floor=reg['floor'])
        print('D', tag, v['cls'], {a: (o['R1L']['failing'], round(100 * (o['b_near'] or 0), 2), o['all_pass']) for a, o in out['arms'].items() if not a.startswith('stale')}, flush=True)
    C.jd(D, an / 'stageD_metrics.json')
    ind = [r for r in D.values() if r['cls'] in ('HELD_SEEN', 'FRESH')]
    H2 = {car: h2(ind, car, b) for car, b in PAIRS}
    H2['FRESH_only'] = {car: h2([r for r in ind if r['cls'] == 'FRESH'], car, b) for car, b in PAIRS}
    sup = all(all(H2[c]['pass'].values()) for c, _ in PAIRS)
    nsup = all(not H2[c]['pass']['b'] and not H2[c]['pass']['d'] for c, _ in PAIRS)
    H2['verdict'] = 'SUPPORTED' if sup else ('NOT SUPPORTED' if nsup else 'INCONCLUSIVE')
    # ---------------- static gate pass counts
    static = {}
    for cls in ('HELD_SEEN', 'FRESH', 'SELECTION'):
        rows = [r for r in D.values() if r['cls'] == cls and r['contact_valid']]
        static[cls] = {a: {'views': len(rows), 'all_pass': int(sum(r['arms'][a]['all_pass'] for r in rows if a in r['arms'])),
                           'R1L_failing': int(sum(r['arms'][a]['R1L']['failing'] for r in rows if a in r['arms'])),
                           'pass_counts': {k: int(sum(r['arms'][a]['pass'][k] for r in rows if a in r['arms'])) for k in ('R1', 'R1L', 'R2', 'R3_bias', 'R4', 'R5a_bias', 'R5b_bias')}}
                       for a in ('B1', 'B2', 'CAR1', 'CAR2', 'CAR_O', 'PC4_O')}
    # R6b
    R6b = {}
    for key in sorted({r['key'] for r in D.values() if r['seq'] is None}):
        rows = [r for r in D.values() if r['key'] == key and r['contact_valid']]
        if len(rows) < 2:
            continue
        R6b[key] = {}
        for a in ('B1', 'B2', 'CAR1', 'CAR2'):
            bs = [r['arms'][a]['R1_near']['b'] for r in rows]; cs = [r['arms'][a]['R2_contrast']['c'] for r in rows if r['arms'][a]['R2_contrast']]
            R6b[key][a] = {'b_spread': max(bs) - min(bs), 'c_spread': max(cs) - min(cs) if cs else float('nan'),
                           'pass': bool(max(bs) - min(bs) <= TAU and (not cs or max(cs) - min(cs) <= TAU))}
    # ---------------- dynamic
    E = {}
    rb = json.loads((root / 'stageD' / 'rebuild_check.json').read_text()) if (root / 'stageD' / 'rebuild_check.json').exists() else {}
    for sname, q in C.sequences().items():
        rows = []
        for t in range(len(q['states'])):
            key = f'{sname}_{t}'
            rows.append({vn: D[f'{key}_{vn}'] for vn in q['views'] if f'{key}_{vn}' in D})
        if not rows or not rows[0]:
            continue
        trans = []
        for t in range(1, len(rows)):
            for vn in rows[t]:
                a_, b_ = rows[t - 1][vn], rows[t][vn]
                if a_['contact_valid'] and b_['contact_valid']:
                    dmu = b_['mu_near_high_full'] - a_['mu_near_high_full']; se = math.sqrt(a_['ref_near_high_se'] ** 2 + b_['ref_near_high_se'] ** 2)
                    sw = {c: bool(abs(b_['arms'][f'stale_{c}']['b_near']) > abs(b_['arms'][c]['b_near']) + 0.005) for c in ('B1', 'CAR1') if f'stale_{c}' in b_['arms']}
                    trans.append({'from': t - 1, 'to': t, 'view': vn, 'delta_full_high': dmu, 'se': se, 'resolved': bool(abs(dmu) >= 5 * se), 'stale_worse': sw})
        rev = []
        for i, j in q['revisits']:
            zi = np.load(root / 'stageD' / f'table_{sname}_{i}_CAR1.npz'); zj = np.load(root / 'stageD' / f'table_{sname}_{j}_CAR1.npz')
            bi = np.load(root / 'stageD' / f'table_{sname}_{i}_B1.npy'); bj = np.load(root / 'stageD' / f'table_{sname}_{j}_B1.npy')
            rel = lambda a, b: float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)) if a.shape == b.shape else float('inf')
            rev.append({'states': [i, j], 'CAR1_T4': rel(zi['T4'], zj['T4']), 'CAR1_T16': rel(zi['T16'], zj['T16']), 'CAR1_w': float(np.abs(zi['w'] - zj['w']).max()),
                        'B1': rel(bi, bj)})
        cont = []
        for t in range(1, len(q['states'])):
            wa = np.load(root / 'stageD' / f'table_{sname}_{t - 1}_CAR1.npz')['w']; wb = np.load(root / 'stageD' / f'table_{sname}_{t}_CAR1.npz')['w']
            cont.append({'from': t - 1, 'to': t, 'max_abs_dw': float(np.abs(wa - wb).max()), 'cells_switched_on_or_off': int(((wa > 0) != (wb > 0)).sum())})
        track = {}
        idx = q.get('track_states') or list(range(len(q['states'])))
        if q.get('gaps') is not None or q.get('track_states'):
            for a in ('B1', 'B2', 'CAR1', 'CAR2', 'CAR_O', 'PC4_O'):
                tk = {}
                for vn in q['views']:
                    rr = [rows[t][vn] for t in idx if vn in rows[t] and rows[t][vn]['contact_valid'] and a in rows[t][vn]['arms']]
                    if len(rr) < 3:
                        continue
                    ref = np.array([r['mu_near_high_full'] for r in rr]); pred = np.array([r['mu_near_high_full'] + r['arms'][a]['near_high_residual'] for r in rr])
                    dr_, dp_ = ref - ref[0], pred - pred[0]
                    slope = float(dr_ @ dp_ / max(dr_ @ dr_, 1e-30)); corr = float(np.corrcoef(dr_, dp_)[0, 1])
                    tk[vn] = {'slope': slope, 'corr': corr, 'pass': bool(abs(slope - 1) <= 0.1 and corr >= 0.98)}
                track[a] = tk
        allpass = {a: bool(all(r[vn]['arms'][a]['all_pass'] for r in rows for vn in r if a in r[vn]['arms'])) for a in ('B1', 'B2', 'CAR1', 'CAR2')}
        E[sname] = {'transitions': trans, 'discriminating_CAR1': sum(1 for t in trans if t['resolved'] and t['stale_worse'].get('CAR1')),
                    'discriminating_B1': sum(1 for t in trans if t['resolved'] and t['stale_worse'].get('B1')), 'revisits': rev, 'activation_continuity': cont,
                    'tracking_R7': track, 'gate3_all_states': allpass,
                    'rebuild': {k: v for k, v in rb.items() if k.startswith(sname + '_')},
                    'state_rows': {f'{t}_{vn}': {a: (rows[t][vn]['arms'][a]['b_near'], rows[t][vn]['arms'][a]['R1L']['failing'], rows[t][vn]['arms'][a]['all_pass'])
                                                 for a in rows[t][vn]['arms']} for t in range(len(rows)) for vn in rows[t]}}
    C.jd(E, an / 'stageE_metrics.json'); C.jd(R6b, an / 'R6b.json')
    # ---------------- gate classification
    def every(a):
        return bool(all(r['arms'][a]['all_pass'] for r in D.values() if r['contact_valid'] or True) and all(x[a]['pass'] for x in R6b.values()))
    dyn = {c: bool(all(all(t['pass'] for t in E[s]['tracking_R7'].get(c, {}).values()) for s in E)) for c in ('CAR1', 'CAR2')}
    part = all(every(c) and dyn[c] for c in ('CAR1', 'CAR2'))
    gate = {'classification': 'PARTIALLY_RESOLVED' if part else 'NOT_RESOLVED', 'static_all_views': {a: every(a) for a in ('B1', 'B2', 'CAR1', 'CAR2')},
            'R7_all': dyn, 'not_run': ['R3 equal-time frontier comparator (no equal-time operating point of the experiment-local CAR runtime)',
                                       'R9 other radius / medium (runs only if the static and dynamic gates pass)'],
            'note': 'RESOLVED unreachable while R3 / R9 are NOT_RUN'}
    C.jd({'H2': H2, 'static': static, 'gate': gate}, an / 'verdicts.json')
    print(json.dumps({'H2': {c: {k: H2[c][k] for k in ('a_excess_frac', 'b_R1L_reduction', 'c_ring0_reduction', 'd_lifted_reduction', 'e_rmse_ok_frac', 'pass')} for c, _ in PAIRS},
                      'verdict': H2['verdict'], 'gate': gate['classification']}, indent=1), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
    import os; sys.stdout.flush(); os._exit(0)
