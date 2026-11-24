"""WL113 frozen analysis of Stages C-F and the gate.

  python analyze113DE.py ROOT  -> ROOT/analysis/{stageD_metrics.json, stageE_metrics.json, stageF_truncation.json, gate.json, maps_D_*, maps_E_*}
"""
import json, math, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common113 as C  # noqa: E402
import metrics113 as MX  # noqa: E402

BN, BC = 0.0649, 0.0579
NONREG = ('sphere_boundary', 'sphere_interior', 'contact_shadow', 'floor_outside', 'full')
LEARNED = ['A_learned', 'O_U', 'CAND']


def load(d, tag, phys_dir=None):
    st = []
    for j in range(2):
        with np.load(d / f'stream_{tag}_{j}.npz') as z:
            s = {k: z[k] for k in z.files if not k.startswith('W|')}
        if phys_dir is not None:
            with np.load(phys_dir / f'stream_{tag}_{j}.npz') as z:
                for k in ('phys_e1', 'phys_e2', 'phys_h', 'tail', 'meta'):
                    s[k if k != 'meta' else 'phys_meta'] = z[k]
        st.append(s)
    with np.load(d / f'gbuffer_{tag}.npz') as z:
        gb = {k: z[k] for k in z.files}
    return st, gb


def view_eval(st, gb, geom, light_pos, arms, ctrl='O_U'):
    g3, reg, bands = MX.gate_masks(gb, geom, light_pos)
    cv = C.contact_valid(reg)
    h = [s['phys_h'].astype(np.float64) for s in st]
    kap = MX.pixel_kappa(h[0], h[1], reg['floor'])
    mu = MX.region_mean_lum(0.5 * (h[0] + h[1]), reg['near']) if reg['near'].any() else float('nan')
    R6 = [s['phys_e1'].astype(np.float64) + s['phys_e2'] + s['phys_h'] for s in st]
    Rf = [R6[j] + st[j]['tail'] for j in range(2)]
    vm = {'contact_valid': cv, 'kappa': kap, 'mu_ref_near_high': mu, 'band_near': BN * mu, 'band_contrast': BC * mu, 'arms': {},
          'low_pairing_rel_l1': float(max((np.abs(s['phys_e1'] - s['ro_e1']).sum() + np.abs(s['phys_e2'] - s['ro_e2']).sum()) /
                                          (np.abs(s['phys_e1']).sum() + np.abs(s['phys_e2']).sum()) for s in st))}
    if reg['near'].any():
        d = MX.lum(h[0][reg['near']]) - MX.lum(h[1][reg['near']])
        vm['ref_near_se'] = float(math.sqrt(kap * d.var(ddof=1) / (4 * d.size)))
        vm['tail_share_near'] = float(MX.region_mean_lum(0.5 * (st[0]['tail'] + st[1]['tail']), reg['near']) / MX.region_mean_lum(0.5 * (Rf[0] + Rf[1]), reg['near']))
    Mc = {a: [st[j]['ro_e1'].astype(np.float64) + st[j]['ro_e2'] + st[j][f'hl|{a}'] for j in range(2)] for a in arms if f'hl|{a}' in st[0]}
    for a in Mc:
        res = [[st[j][f'hl|{a}'].astype(np.float64) - h[j] for j in range(2)]]
        o = {}
        if cv:
            o['endpoints'] = MX.endpoint_stats(res, reg['near'], reg['flank'], kappa=kap)
        o['bands'] = MX.band_profile(res, bands)
        o['rmse'] = {k: MX.rmse_debiased(res, h, reg[k])[0] for k in ('near', 'flank', 'sphere_boundary', 'sphere_interior', 'sphere', 'contact_shadow', 'floor', 'floor_outside', 'full', 'far_unchanged')}
        o['gate3_full_depth'] = MX.gate3(Mc[a], Rf, g3, kap, ctrl=Mc.get(ctrl))
        o['gate3_depth_matched'] = MX.gate3(Mc[a], R6, g3, kap, ctrl=Mc.get(ctrl))
        # Stage F split of the near bias: method - full = (method - D6) + (D6 - full)
        if reg['near'].any():
            mn = MX.region_mean_lum(0.5 * (Mc[a][0] + Mc[a][1]), reg['near']); r6 = MX.region_mean_lum(0.5 * (R6[0] + R6[1]), reg['near'])
            rf = MX.region_mean_lum(0.5 * (Rf[0] + Rf[1]), reg['near'])
            o['near_split'] = {'method_minus_full_rel': (mn - rf) / rf, 'spatial_learned_rel': (mn - r6) / rf, 'truncation_rel': (r6 - rf) / rf}
        vm['arms'][a] = o
    for a in Mc:
        if a in ('oracle', 'oracle_U', 'oracle_S4') or 'hl|oracle' not in st[0]:
            continue
        d = [st[j][f'hl|{a}'].astype(np.float64) - st[j]['hl|oracle'].astype(np.float64) for j in range(2)]
        vm['arms'][a]['vs_oracle_rel'] = {r: float(math.sqrt(np.mean((0.5 * (d[0][reg[r]] + d[1][reg[r]])) ** 2)) / math.sqrt(max(np.mean(h[0][reg[r]] * h[1][reg[r]]), 1e-30)))
                                          for r in ('near', 'contact_shadow', 'sphere', 'full') if reg[r].any()}
    return vm, reg, h, R6, Rf


def hist_pass(vm, cand='CAND', ctrl='O_U'):
    a, u = vm['arms'][cand], vm['arms'][ctrl]; r = {}
    if vm['contact_valid']:
        e = a['endpoints']
        r['near_inside'] = bool(max(abs(e['near']['lo']), abs(e['near']['hi'])) <= vm['band_near'])
        r['contrast_inside'] = bool(max(abs(e['contrast']['lo']), abs(e['contrast']['hi'])) <= vm['band_contrast'])
        r['bands_inside'] = bool(all(abs(x) <= vm['band_near'] for x in a['bands'][:8] if not math.isnan(x)))
        r['near_rmse_ok'] = bool(a['rmse']['near'] <= 0.10)
    for reg in NONREG:
        r[f'nonreg_{reg}'] = bool(a['rmse'][reg] <= 1.10 * u['rmse'][reg] + 0.0025)
    r['pass'] = bool(all(r.values()))
    return r


def gate3_view_pass(vm, key, cand='CAND'):
    g = vm['arms'][cand][key]['pass']
    if not vm['contact_valid']:
        g = {k: v for k, v in g.items() if k.startswith('R5')}
    return bool(all(g.values())), g


def save_maps(an, tag, st, h, R6, Rf, reg, arms):
    maps = {'ref_h': (0.5 * (h[0] + h[1])).astype(np.float32), 'ref_d6': (0.5 * (R6[0] + R6[1])).astype(np.float32),
            'ref_full': (0.5 * (Rf[0] + Rf[1])).astype(np.float32), 'ro_low': (0.5 * (st[0]['ro_e1'] + st[0]['ro_e2'] + st[1]['ro_e1'] + st[1]['ro_e2'])).astype(np.float32)}
    for a in arms:
        if f'hl|{a}' in st[0]:
            maps[f'hl_{a}'] = (0.5 * (st[0][f'hl|{a}'] + st[1][f'hl|{a}'])).astype(np.float32)
    for k, m in reg.items():
        maps[f'mask_{k}'] = m
    np.savez_compressed(an / f'maps_{tag}.npz', **maps)


def main(root):
    root = Path(root); an = root / 'analysis'
    qa = json.loads((an / 'stageA_qualification.json').read_text()); dec = json.loads((root / 'stageB' / 'decision.json').read_text())
    gate = {'stageA': qa['verdict'], 'selected': dec['selected']}
    pre = json.loads((root / 'stageC' / 'precheck.json').read_text()) if (root / 'stageC' / 'precheck.json').exists() else None
    if qa['verdict'] == 'NOT QUALIFIED':
        gate['classification'] = 'UNQUALIFIED'; C.jd(gate, an / 'gate.json'); return
    if dec['selected'] == 'STOP' or pre is None or not pre['pass']:
        gate['classification'] = 'NOT_RESOLVED'; gate['reason'] = 'no adequate / admissible representation or failed pre-training checks'
        C.jd(gate, an / 'gate.json'); return
    arms = ['A_learned', 'O_U', 'CAND', 'oracle', 'oracle_U', 'oracle_S4', 'stale']
    D = {}
    for fix in ('F0', 'F1', 'F2', 'F3', 'F4'):
        geom, lc = C.fixture(fix)
        for vname in C.views(fix):
            tag = f'{fix}_{vname}'
            st, gb = load(root / 'stageD', tag, phys_dir=(root / 'stageA') if fix != 'F4' else None)
            if fix != 'F4':                                                     # attach the Stage A physical oracle (replica mean) for the readout
                with np.load(root / 'stageA' / f'stream_{tag}_0.npz') as z0, np.load(root / 'stageA' / f'stream_{tag}_1.npz') as z1:
                    sel = 'L4' if dec['selected'] == 'L4' else 'S4'
                    for j, z in enumerate((z0, z1)):
                        st[j]['hl|oracle'] = np.mean([z[f'hl|{sel}_r{r}'] for r in range(C.dv(4, 2))], 0)
                        st[j]['hl|oracle_U'] = np.mean([z[f'hl|U_r{r}'] for r in range(C.dv(4, 2))], 0)
                        st[j]['hl|oracle_S4'] = np.mean([z[f'hl|S4_r{r}'] for r in range(C.dv(4, 2))], 0)
            vm, reg, h, R6, Rf = view_eval(st, gb, geom, lc[0]['pos'], arms)
            vm['historical_pass'] = hist_pass(vm)
            vm['gate3_full_pass'], vm['gate3_full_detail'] = gate3_view_pass(vm, 'gate3_full_depth')
            vm['gate3_dm_pass'], vm['gate3_dm_detail'] = gate3_view_pass(vm, 'gate3_depth_matched')
            opp = root / 'stageD' / f'oppoint_{tag}.npz'
            if opp.exists() and vm['contact_valid']:
                z = np.load(opp); g3, _, _ = MX.gate_masks(gb, geom, lc[0]['pos'])
                rf = MX.region_mean_lum(0.5 * (Rf[0] + Rf[1]), g3['near']); r6 = MX.region_mean_lum(0.5 * (R6[0] + R6[1]), g3['near'])
                vm['operating_point_R1'] = {}
                for a in z.files:
                    vals = np.array([MX.region_mean_lum(fr, g3['near']) for fr in z[a]])
                    se = vals.std(ddof=1) / math.sqrt(len(vals))
                    vm['operating_point_R1'][a] = {'full_depth_b': float((vals.mean() - rf) / rf), 'depth_matched_b': float((vals.mean() - r6) / r6),
                                                   'hw': float(2.365 * se / rf)}
            D[tag] = vm
            save_maps(an, f'D_{tag}', st, h, R6, Rf, reg, arms)
            print('D', tag, vm['historical_pass']['pass'], vm['gate3_dm_pass'], vm['gate3_full_pass'], flush=True)
    C.jd(D, an / 'stageD_metrics.json')
    cvD = [v for v in D.values() if v['contact_valid']]
    def spread(rows, key):
        vals = []
        for v in rows:
            x = v['arms']['CAND'][key]
            vals.append((x['R1_near']['b'], x['R2_contrast']['c'] if x['R2_contrast'] else float('nan')))
        bs = [a for a, _ in vals]; cs = [c for _, c in vals if not math.isnan(c)]
        return {'b_spread': max(bs) - min(bs), 'c_spread': max(cs) - min(cs), 'pass': bool(max(bs) - min(bs) <= MX.TAU and max(cs) - min(cs) <= MX.TAU)}
    R6b = {fix: {k: spread([v for t, v in D.items() if t.startswith(fix) and v['contact_valid']], k) for k in ('gate3_full_depth', 'gate3_depth_matched')}
           for fix in ('F0', 'F1', 'F2', 'F3', 'F4')}
    # ---------------- Stage E
    E = {}
    rb = json.loads((root / 'stageE' / 'rebuild_check.json').read_text())
    for sname, seq in C.sequences().items():
        rows = []
        for stt, geom in enumerate(seq['states']):
            row = {'label': seq['label'][stt], 'views': {}, 'rebuild': rb[f'{sname}_{stt}']}
            sp = root / 'stageE' / f'support_{sname}_{stt}.npy'
            if sp.exists():
                s_ = np.load(sp); row['support_patches'] = int(s_.sum())
                row['support_floor_patches'] = int(s_[:256].sum())
            for vname in seq['views']:
                tag = f'{sname}_{stt}_{vname}'
                st, gb = load(root / 'stageE', tag)
                vm, reg, h, R6, Rf = view_eval(st, gb, geom, seq['light'][0]['pos'], arms)
                vm['historical_pass'] = hist_pass(vm)
                vm['gate3_full_pass'], vm['gate3_full_detail'] = gate3_view_pass(vm, 'gate3_full_depth')
                vm['gate3_dm_pass'], vm['gate3_dm_detail'] = gate3_view_pass(vm, 'gate3_depth_matched')
                if 'stale' in vm['arms'] and vm['contact_valid']:
                    s_, c_ = vm['arms']['stale'], vm['arms']['CAND']
                    vm['stale_worse'] = bool(abs(s_['endpoints']['near']['estimate']) - abs(c_['endpoints']['near']['estimate']) >
                                             s_['endpoints']['near']['half_width'] + c_['endpoints']['near']['half_width'] or s_['rmse']['near'] > c_['rmse']['near'] + 0.01)
                row['views'][vname] = vm
                save_maps(an, f'E_{tag}', st, h, R6, Rf, reg, arms)
            rows.append(row)
            print('E', sname, stt, {v: (r['historical_pass']['pass'], r['gate3_dm_pass'], r['gate3_full_pass']) for v, r in row['views'].items()}, flush=True)
        trans = []
        for stt in range(1, len(rows)):
            for vname in seq['views']:
                a, b = rows[stt - 1]['views'][vname], rows[stt]['views'][vname]
                if a['contact_valid'] and b['contact_valid']:
                    dmu = b['mu_ref_near_high'] - a['mu_ref_near_high']; se = math.sqrt(a['ref_near_se'] ** 2 + b['ref_near_se'] ** 2)
                    trans.append({'from': stt - 1, 'to': stt, 'view': vname, 'delta': dmu, 'se': se, 'resolved': bool(abs(dmu) >= 5 * se), 'stale_worse': b.get('stale_worse')})
        rev = []
        for i, j in seq['revisits']:
            ti = np.load(root / 'stageE' / f'table_{sname}_{i}_CAND.npy'); tj = np.load(root / 'stageE' / f'table_{sname}_{j}_CAND.npy')
            r = {'states': [i, j], 'table_max_rel': float(np.abs(ti - tj).max() / max(np.abs(ti).max(), 1e-30))}
            si, sj = root / 'stageE' / f'support_{sname}_{i}.npy', root / 'stageE' / f'support_{sname}_{j}.npy'
            if si.exists():
                r['support_identical'] = bool(np.array_equal(np.load(si), np.load(sj)))
            rev.append(r)
        track = None
        if 'gaps' in seq:
            tk = {}
            for vname in seq['views']:
                ref = []; pred = []
                for i, row in enumerate(rows):
                    v = row['views'][vname]
                    if v['contact_valid']:
                        ref.append(v['mu_ref_near_high']); pred.append(v['mu_ref_near_high'] + v['arms']['CAND']['endpoints']['near']['estimate'])
                ref, pred = np.array(ref), np.array(pred)
                dr_, dp_ = ref - ref[0], pred - pred[0]
                slope = float((dr_ @ dp_) / max(dr_ @ dr_, 1e-30)); corr = float(np.corrcoef(dr_, dp_)[0, 1]) if len(ref) > 2 else float('nan')
                tk[vname] = {'slope': slope, 'corr': corr, 'pass': bool(abs(slope - 1) <= 0.1 and corr >= 0.98)}
            track = tk
        far = []
        for vname in seq['views']:
            base = rows[0]['views'][vname]['arms']['CAND']['rmse']['far_unchanged']
            for i, row in enumerate(rows):
                v = row['views'][vname]['arms']['CAND']['rmse']['far_unchanged']; far.append({'state': i, 'view': vname, 'ok': bool(v <= 1.10 * base + 0.0025)})
        disc = [t for t in trans if t['resolved'] and t['stale_worse']]
        hist_all = all(r['views'][v]['historical_pass']['pass'] for r in rows for v in r['views'])
        ok = (hist_all and all(x['table_max_rel'] <= 1e-6 and x.get('support_identical', True) for x in rev)
              and all(r['rebuild']['table_max_rel'] <= 1e-6 and r['rebuild'].get('support_identical', True) for r in rows)
              and all(f['ok'] for f in far) and len(disc) >= 1 and (track is None or all(t['pass'] for t in track.values())))
        E[sname] = {'states': rows, 'transitions': trans, 'discriminating_transitions': len(disc), 'revisits': rev, 'tracking_R7': track,
                    'far_unchanged': far, 'historical_all_states': bool(hist_all), 'pass_depth_matched': bool(ok),
                    'gate3_full_all_states': bool(all(r['views'][v]['gate3_full_pass'] for r in rows for v in r['views'])),
                    'gate3_dm_all_states': bool(all(r['views'][v]['gate3_dm_pass'] for r in rows for v in r['views']))}
    C.jd(E, an / 'stageE_metrics.json')
    # ---------------- Stage F summary
    F = {tag: {a: v['arms'][a].get('near_split') for a in ('CAND', 'O_U', 'oracle', 'oracle_U', 'oracle_S4') if a in v['arms']} for tag, v in D.items() if v['contact_valid']}
    C.jd(F, an / 'stageF_truncation.json')
    # ---------------- gate
    d_hist = all(v['historical_pass']['pass'] for v in D.values())
    d_dm = all(v['gate3_dm_pass'] for v in D.values()) and all(R6b[f]['gate3_depth_matched']['pass'] for f in R6b)
    d_full = all(v['gate3_full_pass'] for v in D.values()) and all(R6b[f]['gate3_full_depth']['pass'] for f in R6b)
    e_dm = all(s['pass_depth_matched'] for s in E.values())
    e_full = all(s['gate3_full_all_states'] for s in E.values())
    gate.update({'stageD_historical': bool(d_hist), 'stageD_3pct_depth_matched': bool(d_dm), 'stageD_3pct_full_depth': bool(d_full),
                 'stageE_depth_matched_contract': bool(e_dm), 'stageE_3pct_full_depth': bool(e_full), 'R6b': R6b,
                 'not_run': ['R3 equal-time frontier comparator', 'R9 other radius / medium']})
    if d_full and e_full:
        gate['classification'] = 'PARTIALLY_RESOLVED'; gate['note'] = 'full-depth 3 % contract met on every evaluated class, but R3-frontier and R9 radius / medium are NOT_RUN, so RESOLVED is not reachable in this batch'
    elif d_hist and e_dm:
        gate['classification'] = 'PARTIALLY_RESOLVED'; gate['note'] = 'spatial artifact resolved at depth 6 (historical contract, static and dynamic); approved full-depth 3 % contract not met'
    else:
        gate['classification'] = 'NOT_RESOLVED'
    C.jd(gate, an / 'gate.json')
    print('GATE', json.dumps({k: v for k, v in gate.items() if k != 'R6b'}), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
