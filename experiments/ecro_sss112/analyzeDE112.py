"""WL112 frozen analysis of Stage D (static / cross-view) and Stage E (dynamic) and the Mandatory SSS Gate classification.

  python analyzeDE112.py ROOT     -> ROOT/analysis/{stageD_metrics.json, stageE_metrics.json, gate.json, maps_D_*.npz, maps_E_*.npz}
"""
import json, math, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common112 as C  # noqa: E402
import metrics112 as MX  # noqa: E402

BAND_NEAR_FRAC, BAND_CON_FRAC = 0.0649, 0.0579
REG = ('sphere_boundary', 'sphere_interior', 'sphere', 'contact_shadow', 'floor', 'floor_outside', 'near', 'flank', 'full', 'far_unchanged')
NONREG = ('sphere_boundary', 'sphere_interior', 'contact_shadow', 'floor_outside', 'full')


def load(dirp, tag):
    st = []
    for j in range(2):
        with np.load(dirp / f'stream_{tag}_{j}.npz') as z:
            st.append({k: z[k] for k in z.files if not k.startswith('W|')})
    with np.load(dirp / f'gbuffer_{tag}.npz') as z:
        gb = {k: z[k] for k in z.files}
    return st, gb


def view_metrics(st, gb, geom, light_pos, arms):
    reg, bands = C.masks(gb, geom, light_pos)
    cv = C.contact_valid(reg)
    h = [s['phys_C_h'].astype(np.float64) for s in st]
    mu = MX.region_mean_lum(0.5 * (h[0] + h[1]), reg['near']) if reg['near'].any() else float('nan')
    out = {'contact_valid': cv, 'mu_ref_near': mu, 'band_near': BAND_NEAR_FRAC * mu, 'band_contrast': BAND_CON_FRAC * mu,
           'pixels': {k: int(m.sum()) for k, m in reg.items()}, 'arms': {},
           'low_pairing_rel_l1': float(max((np.abs(s['phys_C_e1'] - s['ro_e1']).sum() + np.abs(s['phys_C_e2'] - s['ro_e2']).sum()) / (np.abs(s['phys_C_e1']).sum() + np.abs(s['phys_C_e2']).sum()) for s in st)),
           'bad_origin': float(sum(json.loads(str(s['meta'])).get('bad_origin_C', 0.) for s in st))}
    if reg['near'].any():
        d = MX.lum(h[0][reg['near']]) - MX.lum(h[1][reg['near']])
        out['ref_near_se'] = float(math.sqrt(d.var(ddof=1) / (4 * d.size)))
    for arm in arms:
        if f'hl|{arm}' not in st[0]:
            continue
        res = [[st[j][f'hl|{arm}'].astype(np.float64) - h[j] for j in range(2)]]
        o = {}
        if cv:
            o['endpoints'] = MX.endpoint_stats(res, reg['near'], reg['flank'])
        o['bands'] = MX.band_profile(res, bands)
        o['rmse'] = {}
        for r in REG:
            rel, absr, bias = MX.rmse_debiased(res, h, reg[r])
            o['rmse'][r] = {'rel': rel, 'abs': absr, 'bias_rgb': bias}
        out['arms'][arm] = o
    # learned approximation error: candidate vs its physical oracle through the same readout
    for arm in arms:
        if arm in ('oracle',) or f'hl|{arm}' not in st[0] or 'hl|oracle' not in st[0]:
            continue
        d = [st[j][f'hl|{arm}'].astype(np.float64) - st[j]['hl|oracle'].astype(np.float64) for j in range(2)]
        rr = {}
        for r in ('near', 'contact_shadow', 'sphere', 'full'):
            m = reg[r]
            if m.any():
                rr[r] = float(math.sqrt(np.mean(0.5 * (d[0][m] + d[1][m]) ** 2)) / math.sqrt(max(np.mean(h[0][m] * h[1][m]), 1e-30)))
        out['arms'][arm]['vs_oracle_rel'] = rr
    return out, reg, bands, h


def pass_rules(vm, cand, ctrl='O_U'):
    a = vm['arms'][cand]; u = vm['arms'][ctrl]
    r = {}
    if vm['contact_valid']:
        e = a['endpoints']
        r['near_inside'] = bool(max(abs(e['near']['lo']), abs(e['near']['hi'])) <= vm['band_near'])
        r['contrast_inside'] = bool(max(abs(e['contrast']['lo']), abs(e['contrast']['hi'])) <= vm['band_contrast'])
        bb = [x for x in a['bands'][:8] if not math.isnan(x)]
        r['bands_inside'] = bool(all(abs(x) <= vm['band_near'] for x in bb))
        r['near_rmse_ok'] = bool(a['rmse']['near']['rel'] <= 0.10)
    nr = {}
    for reg in NONREG:
        lim = 1.10 * u['rmse'][reg]['rel'] + 0.0025
        nr[reg] = {'cand': a['rmse'][reg]['rel'], 'limit': lim, 'ok': bool(a['rmse'][reg]['rel'] <= lim)}
    r['non_regression'] = nr
    r['pass'] = bool(all(v for k, v in r.items() if isinstance(v, bool)) and all(v['ok'] for v in nr.values()))
    return r


def save_maps(an, tag, st, h, reg, arms):
    maps = {'ref_h': (0.5 * (h[0] + h[1])).astype(np.float32),
            'ref_e': (0.5 * (st[0]['phys_C_e1'] + st[0]['phys_C_e2'] + st[1]['phys_C_e1'] + st[1]['phys_C_e2'])).astype(np.float32),
            'ref_half_diff': (0.5 * (h[0] - h[1])).astype(np.float32)}
    for arm in arms:
        if f'hl|{arm}' in st[0]:
            maps[f'hl_{arm}'] = (0.5 * (st[0][f'hl|{arm}'] + st[1][f'hl|{arm}'])).astype(np.float32)
    for k, m in reg.items():
        maps[f'mask_{k}'] = m
    np.savez_compressed(an / f'maps_{tag}.npz', **maps)


def main(root):
    root = Path(root); an = root / 'analysis'; an.mkdir(parents=True, exist_ok=True)
    gatesA = json.loads((an / 'stageA_gates.json').read_text())
    dec = json.loads((root / 'stageB' / 'decision.json').read_text())
    gate = {'stageA': gatesA['verdict'], 'stageB': dec['outcome']}
    if gatesA['verdict'] == 'NOT QUALIFIED' or dec['outcome'] in ('STOP_UNQUALIFIED', 'NONE_LIMITED'):
        gate['classification'] = 'UNQUALIFIED'; C.jd(gate, an / 'gate.json'); return
    if dec['outcome'] == 'NONE':
        gate['classification'] = 'NOT_RESOLVED'; C.jd(gate, an / 'gate.json'); return
    cand = 'S2' if dec['outcome'].startswith('SPATIAL') else 'O_G'
    arms = ['A_learned', 'O_U', 'O_G', 'S2', 'oracle', 'stale']
    # ---------------- Stage D
    D = {}
    for fix in ('F0', 'F1'):
        geom, light = C.fixture(fix)
        for vname in C.views(fix):
            tag = f'{fix}_{vname}'
            st, gb = load(root / 'stageD', tag)
            vm, reg, bands, h = view_metrics(st, gb, geom, light[0]['pos'], arms)
            vm['pass'] = pass_rules(vm, cand)
            D[tag] = vm
            save_maps(an, f'D_{tag}', st, h, reg, arms)
            print('D', tag, vm['pass']['pass'], flush=True)
    C.jd(D, an / 'stageD_metrics.json')
    # ---------------- Stage E
    E = {}
    seqs = C.sequences()
    rebuild = json.loads((root / 'stageE' / 'rebuild_check.json').read_text())
    for sname, seq in seqs.items():
        rows = []
        for stt, geom in enumerate(seq['states']):
            row = {'label': seq['label'][stt], 'views': {}}
            for vname in seq['views']:
                tag = f'{sname}_{stt}_{vname}'
                st, gb = load(root / 'stageE', tag)
                vm, reg, bands, h = view_metrics(st, gb, geom, seq['light'][0]['pos'], arms)
                vm['pass'] = pass_rules(vm, cand)
                if 'stale' in vm['arms'] and vm['contact_valid']:
                    s_ = vm['arms']['stale']; c_ = vm['arms'][cand]
                    vm['stale_worse'] = bool((abs(s_['endpoints']['near']['estimate']) - abs(c_['endpoints']['near']['estimate']) >
                                              s_['endpoints']['near']['half_width'] + c_['endpoints']['near']['half_width'])
                                             or s_['rmse']['near']['rel'] > c_['rmse']['near']['rel'] + 0.01)
                row['views'][vname] = vm
                save_maps(an, f'E_{tag}', st, h, reg, arms)
            row['rebuild_max_rel'] = rebuild[f'{sname}_{stt}']
            rows.append(row)
            print('E', sname, stt, {v: r['pass']['pass'] for v, r in row['views'].items()}, flush=True)
        # transitions: resolved physical change of the near HIGH mean
        trans = []
        for stt in range(1, len(rows)):
            for vname in seq['views']:
                a, b = rows[stt - 1]['views'][vname], rows[stt]['views'][vname]
                if not (a['contact_valid'] and b['contact_valid']):
                    continue
                dmu = b['mu_ref_near'] - a['mu_ref_near']; se = math.sqrt(a['ref_near_se'] ** 2 + b['ref_near_se'] ** 2)
                trans.append({'from': stt - 1, 'to': stt, 'view': vname, 'delta_near_high': dmu, 'se': se, 'resolved': bool(abs(dmu) >= 5 * se),
                              'stale_worse': b.get('stale_worse')})
        # revisits: identical candidate tables
        cand_name = cand
        rev = []
        for i, j in seq['revisits']:
            ti = np.load(root / 'stageE' / f'table_{sname}_{i}_{cand_name}.npy'); tj = np.load(root / 'stageE' / f'table_{sname}_{j}_{cand_name}.npy')
            rev.append({'states': [i, j], 'max_rel': float(np.abs(ti - tj).max() / max(np.abs(ti).max(), 1e-30))})
        # unchanged far region vs state 0
        far = []
        for vname in seq['views']:
            base = rows[0]['views'][vname]['arms'][cand]['rmse']['far_unchanged']['rel']
            for stt, row in enumerate(rows):
                v = row['views'][vname]['arms'][cand]['rmse']['far_unchanged']['rel']
                far.append({'state': stt, 'view': vname, 'rel': v, 'limit': 1.10 * base + 0.0025, 'ok': bool(v <= 1.10 * base + 0.0025)})
        allpass = all(r['views'][v]['pass']['pass'] for r in rows for v in r['views'])
        disc = [t for t in trans if t['resolved'] and t['stale_worse']]          # transitions that count as dynamic evidence
        ok = (allpass and all(x['max_rel'] <= 1e-6 for x in rev) and all(r['rebuild_max_rel'] <= 1e-6 for r in rows)
              and all(x['ok'] for x in far) and len(disc) >= 1)
        E[sname] = {'states': rows, 'transitions': trans, 'discriminating_transitions': len(disc), 'revisits': rev, 'far_unchanged': far,
                    'all_states_pass': bool(allpass), 'pass': bool(ok)}
    C.jd(E, an / 'stageE_metrics.json')
    # ---------------- gate
    d_ok = all(v['pass']['pass'] for v in D.values())
    e_ok = all(s['pass'] for s in E.values())
    f0 = [v for k, v in D.items() if k.startswith('F0') and v['contact_valid']]
    improve = all(abs(v['arms'][cand]['endpoints']['near']['estimate']) <= 0.5 * abs(v['arms']['O_U']['endpoints']['near']['estimate']) for v in f0)
    gate.update({'candidate': cand, 'stageD_pass': bool(d_ok), 'stageE_pass': bool(e_ok), 'material_improvement_F0': bool(improve)})
    gate['classification'] = 'RESOLVED' if (d_ok and e_ok and gatesA['verdict'] in ('QUALIFIED', 'PARTIALLY QUALIFIED')) else \
        ('PARTIALLY_RESOLVED' if improve else 'NOT_RESOLVED')
    C.jd(gate, an / 'gate.json')
    print('GATE', json.dumps(gate), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
