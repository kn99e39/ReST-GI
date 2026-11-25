"""WL114 frozen analysis.

  python analyze114.py ROOT B   Stage B (oracle-only, Stage B streams + tables) -> analysis/stageB_metrics.json, stageB/decision.json
  python analyze114.py ROOT D   Stage D (learned arms) -> analysis/stageD_metrics.json, gate.json, verdicts.json, maps_*
Metric definitions are WL113's (metrics113 / analyze113DE); only the arm set, the full-depth reference (D6 + buckets <= K*), the
R5 RMSE control per reference (OB1 full depth, O_U depth-matched) and the WL114 decomposition / verdict rules are new.
"""
import json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'ecro_sss113'))
import common114 as C  # noqa: E402
import metrics113 as MX  # noqa: E402
import tracer114 as T  # noqa: E402
import eval114 as EV  # noqa: E402

BN, BC = 0.0649, 0.0579
NONREG = ('sphere_boundary', 'sphere_interior', 'contact_shadow', 'floor_outside', 'full')
D6_ARMS = ('A', 'A1r', 'A2', 'O_U', 'S4_D6', 'U_D6', 'stale_A1r')
F_ARMS = ('B1', 'B2', 'OB1', 'S4_F', 'U_F', 'stale_B1')
STATIC = EV.STATIC


def load(root, tag, K, learned=True):
    st = []
    for j in range(2):
        with np.load(root / 'stageB' / f'stream_{tag}_{j}.npz') as z:
            s = {k: z[k] for k in z.files}
        s['tail'] = sum(s[f'phys_{n}'].astype(np.float64) for n in T.deep_through(K) if f'phys_{n}' in s)
        dp = root / 'stageD' / f'stream_{tag}_{j}.npz'
        if learned and dp.exists():
            with np.load(dp) as z:
                for k in z.files:
                    if k.startswith('hl|'):
                        s[k] = z[k]
                s['ro_e1_D'] = z['ro_e1']; s['ro_e2_D'] = z['ro_e2']
        st.append(s)
    with np.load(root / 'stageB' / f'gbuffer_{tag}.npz') as z:
        gb = {k: z[k] for k in z.files}
    return st, gb


def view_eval(st, gb, geom, lp):
    g3, reg, bands = MX.gate_masks(gb, geom, lp)
    cv = C.contact_valid(reg)
    h = [s['phys_h'].astype(np.float64) for s in st]
    hf = [h[j] + st[j]['tail'] for j in range(2)]
    kap = MX.pixel_kappa(h[0], h[1], reg['floor'])
    R6 = [st[j]['phys_e1'].astype(np.float64) + st[j]['phys_e2'] + h[j] for j in range(2)]
    Rf = [R6[j] + st[j]['tail'] for j in range(2)]
    mu6 = MX.region_mean_lum(0.5 * (h[0] + h[1]), reg['near']) if reg['near'].any() else float('nan')
    muF = MX.region_mean_lum(0.5 * (hf[0] + hf[1]), reg['near']) if reg['near'].any() else float('nan')
    vm = {'contact_valid': cv, 'kappa': kap, 'mu_near_high_D6': mu6, 'mu_near_high_full': muF,
          'band_near_D6': BN * mu6, 'band_contrast_D6': BC * mu6, 'band_near_full': BN * muF, 'band_contrast_full': BC * muF, 'arms': {}}
    if reg['near'].any():
        vm['trunc_near_rel'] = float((MX.region_mean_lum(0.5 * (R6[0] + R6[1]), reg['near']) - MX.region_mean_lum(0.5 * (Rf[0] + Rf[1]), reg['near'])) /
                                     MX.region_mean_lum(0.5 * (Rf[0] + Rf[1]), reg['near']))
        d = MX.lum(hf[0][reg['near']]) - MX.lum(hf[1][reg['near']])
        vm['ref_near_full_se'] = float(math.sqrt(kap * d.var(ddof=1) / (4 * d.size)))
        vm['tail_share_near_high'] = float(MX.region_mean_lum(0.5 * (st[0]['tail'] + st[1]['tail']), reg['near']) / muF)
    if 'ro_e1_D' in st[0]:
        vm['low_pairing_B_vs_D'] = float(max(np.abs(s['ro_e1'] - s['ro_e1_D']).max() + np.abs(s['ro_e2'] - s['ro_e2_D']).max() for s in st))
    vm['low_pairing_phys_vs_ro'] = float(max((np.abs(s['phys_e1'] - s['ro_e1']).sum() + np.abs(s['phys_e2'] - s['ro_e2']).sum()) /
                                             (np.abs(s['phys_e1']).sum() + np.abs(s['phys_e2']).sum()) for s in st))
    arms = [k[3:] for k in st[0] if k.startswith('hl|')]
    Mc = {a: [st[j]['ro_e1'].astype(np.float64) + st[j]['ro_e2'] + st[j][f'hl|{a}'] for j in range(2)] for a in arms}
    for a in arms:
        o = {}
        res6 = [[st[j][f'hl|{a}'].astype(np.float64) - h[j] for j in range(2)]]
        resF = [[st[j][f'hl|{a}'].astype(np.float64) - hf[j] for j in range(2)]]
        if cv:
            o['endpoints_D6'] = MX.endpoint_stats(res6, reg['near'], reg['flank'], kappa=kap)
            o['endpoints_full'] = MX.endpoint_stats(resF, reg['near'], reg['flank'], kappa=kap)
        o['bands_D6'] = MX.band_profile(res6, bands); o['bands_full'] = MX.band_profile(resF, bands)
        o['rmse_D6'] = {k: MX.rmse_debiased(res6, h, reg[k])[0] for k in ('near', 'flank', 'sphere_boundary', 'sphere_interior', 'sphere', 'contact_shadow', 'floor', 'floor_outside', 'full', 'far_unchanged')}
        o['rmse_full'] = {k: MX.rmse_debiased(resF, hf, reg[k])[0] for k in ('near', 'flank', 'sphere_boundary', 'sphere_interior', 'sphere', 'contact_shadow', 'floor', 'floor_outside', 'full', 'far_unchanged')}
        o['gate3_full_depth'] = MX.gate3(Mc[a], Rf, g3, kap, ctrl=Mc.get('OB1'))
        o['gate3_depth_matched'] = MX.gate3(Mc[a], R6, g3, kap, ctrl=Mc.get('O_U'))
        if reg['near'].any():
            o['b_full'] = MX.rel_bias(Mc[a], Rf, reg['near'], kap)['b']; o['b_D6'] = MX.rel_bias(Mc[a], R6, reg['near'], kap)['b']
        vm['arms'][a] = o
    return vm, reg, h, hf, R6, Rf


def hist_pass(vm, cand, ctrl, depth):
    a, u = vm['arms'][cand], vm['arms'].get(ctrl); r = {}
    sfx = 'D6' if depth == 'D6' else 'full'
    if vm['contact_valid']:
        e = a[f'endpoints_{sfx}']
        r['near_inside'] = bool(max(abs(e['near']['lo']), abs(e['near']['hi'])) <= vm[f'band_near_{sfx}'])
        r['contrast_inside'] = bool(max(abs(e['contrast']['lo']), abs(e['contrast']['hi'])) <= vm[f'band_contrast_{sfx}'])
        r['bands_inside'] = bool(all(abs(x) <= vm[f'band_near_{sfx}'] for x in a[f'bands_{sfx}'][:8] if not math.isnan(x)))
        r['near_rmse_ok'] = bool(a[f'rmse_{sfx}']['near'] <= 0.10)
    if u is not None:
        for reg in NONREG:
            r[f'nonreg_{reg}'] = bool(a[f'rmse_{sfx}'][reg] <= 1.10 * u[f'rmse_{sfx}'][reg] + 0.0025)
    r['pass'] = bool(all(r.values()))
    return r


def gate3_pass(vm, arm, key):
    g = vm['arms'][arm][key]['pass']
    if not vm['contact_valid']:
        g = {k: v for k, v in g.items() if k.startswith('R5')}
    return bool(all(g.values())), g


def save_maps(an, tag, st, h, hf, R6, Rf, reg):
    m = lambda a, b: (0.5 * (a + b)).astype(np.float32)
    maps = {'ref_h': m(h[0], h[1]), 'ref_hf': m(hf[0], hf[1]), 'ref_d6': m(R6[0], R6[1]), 'ref_full': m(Rf[0], Rf[1]),
            'ro_low': m(st[0]['ro_e1'] + st[0]['ro_e2'], st[1]['ro_e1'] + st[1]['ro_e2'])}
    for k in st[0]:
        if k.startswith('hl|'):
            maps[f'hl_{k[3:]}'] = m(st[0][k], st[1][k])
    for k, v in reg.items():
        maps[f'mask_{k}'] = v
    np.savez_compressed(an / f'maps_{tag}.npz', **maps)


def all_views():
    for key, (g, l, vl, _) in EV.states().items():
        for vn, cam, seeds in vl:
            yield key, vn, g, l


# ---------------------------------------------------------------------------------------------------- Stage B
def table_stats(root, key):
    zs = [np.load(EV.table_path(root, key, r)) for r in range(len(EV.states()[key][3]))]
    G = T.Ctx114.__mro__  # noqa: F841 (documentation: patch classes from the geometry tables below)
    import tracer114 as TT
    g, l, _, _ = EV.states()[key]
    ctx = TT.Ctx114(g, 'C'); Gt = ctx.G
    r0 = TT.ring0(Gt, g['S']['c'][:2]); sph = Gt.PSID == TT.S.SID_SPHERE; flo = Gt.PSID == 0
    classes = {'ring0_sphere': r0 & sph, 'ring0_floor': r0 & flo, 'other_sphere': sph & ~r0, 'other_floor': flo & ~r0, 'walls_boxes': ~sph & ~flo}
    out = {}
    U6 = np.mean([z['U_D6'] for z in zs], 0); UF = np.mean([z['U_F'] for z in zs], 0)
    S6 = np.mean([z['S4_D6'] for z in zs], 0); SF = np.mean([z['S4_F'] for z in zs], 0)
    lum = lambda x: x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722
    out['ratio_F_over_D6'] = {c: float(lum(UF[m]).sum() / max(lum(U6[m]).sum(), 1e-30)) for c, m in classes.items() if m.any()}
    for tag, S4, U in (('D6', S6, U6), ('F', SF, UF)):
        L4 = lum(S4[r0]); Lc = lum(U[r0]); ok = Lc > 1e-6
        cv = np.std(L4, axis=1) / np.maximum(Lc, 1e-30)
        out[f'within_cell_cv_ring0_{tag}'] = {'median': float(np.median(cv[ok])), 'p90': float(np.percentile(cv[ok], 90))}
    a, b = lum(U6[r0]), lum(UF[r0])
    cors = [np.corrcoef(a[i], b[i])[0, 1] for i in range(len(a)) if a[i].std() > 0 and b[i].std() > 0]
    out['angular_profile_corr_ring0'] = {'median': float(np.median(cors)), 'min': float(np.min(cors))} if cors else None
    del ctx
    return out


def stageB(root):
    K = EV.kstar(root); an = root / 'analysis'; an.mkdir(parents=True, exist_ok=True)
    res = {'K_star': K, 'views': {}, 'tables': {}}
    for key in STATIC:
        res['tables'][key] = table_stats(root, key)
    for key, vn, g, l in all_views():
        tag = f'{key}_{vn}'
        st, gb = load(root, tag, K, learned=False)
        vm, reg, h, hf, R6, Rf = view_eval(st, gb, g, l[0]['pos'])
        vm['hist_S4_D6'] = hist_pass(vm, 'S4_D6', None, 'D6')
        vm['hist_S4_F_fullbands'] = hist_pass(vm, 'S4_F', None, 'F')
        for a in ('S4_F', 'U_F', 'S4_D6', 'U_D6'):
            vm[f'gate3_full_{a}'] = gate3_pass(vm, a, 'gate3_full_depth')
        vm['gate3_dm_S4_D6'] = gate3_pass(vm, 'S4_D6', 'gate3_depth_matched')
        if vm['contact_valid'] and key in STATIC:
            sF = abs(vm['arms']['S4_F']['b_full'])
            vm['attributable'] = bool(abs(vm['trunc_near_rel']) >= 2 * sF)
        res['views'][tag] = vm
        if key in STATIC or key.endswith('_0') or key.endswith('_3'):
            save_maps(an, f'B_{tag}', st, h, hf, R6, Rf, reg)
        print('B', tag, vm.get('trunc_near_rel'), vm['arms']['S4_F'].get('b_full'), vm.get('attributable'), flush=True)
    att = [v['attributable'] for t, v in res['views'].items() if 'attributable' in v]
    frac = float(np.mean(att)) if att else 0.
    dec = {'attributable_fraction_static': frac, 'n_static_contact_valid': len(att), 'proceed': bool(frac >= 0.75 or C.DEV), 'K_star': K}
    res['decision'] = dec
    C.jd(res, an / 'stageB_metrics.json'); C.jd(dec, root / 'stageB' / 'decision.json')
    print('STAGE_B', json.dumps(dec), flush=True)


# ---------------------------------------------------------------------------------------------------- Stage D
def spread(rows, arm, key):
    bs = [v['arms'][arm][key]['R1_near']['b'] for v in rows]
    cs = [v['arms'][arm][key]['R2_contrast']['c'] for v in rows if v['arms'][arm][key]['R2_contrast']]
    if not bs:
        return None
    return {'b_spread': max(bs) - min(bs), 'c_spread': (max(cs) - min(cs)) if cs else float('nan'),
            'pass': bool(max(bs) - min(bs) <= MX.TAU and (not cs or max(cs) - min(cs) <= MX.TAU))}


def decomposition(vm):
    a = vm['arms']
    if not vm['contact_valid'] or 'S4_F' not in a:
        return None
    o = {'truncation': vm['trunc_near_rel'], 'spatial_F': a['S4_F']['b_full'], 'spatial_D6': a['S4_D6']['b_D6']}
    for arm, orc, key in (('B1', 'S4_F', 'b_full'), ('B2', 'S4_F', 'b_full'), ('A1r', 'S4_D6', 'b_D6'), ('A2', 'S4_D6', 'b_D6'), ('A', 'S4_D6', 'b_D6'),
                          ('OB1', 'U_F', 'b_full'), ('O_U', 'U_D6', 'b_D6')):
        if arm in a:
            o[f'learned_{arm}'] = a[arm][key] - a[orc][key]
            o[f'total_full_{arm}'] = a[arm]['b_full']
    return o


def stageD(root):
    K = EV.kstar(root); an = root / 'analysis'
    rb = json.loads((root / 'stageD' / 'rebuild_check.json').read_text())
    D = {}; E = {}
    for key, vn, g, l in all_views():
        if any(key.startswith(s + '_') for s in EV.SEQS):
            continue
        tag = f'{key}_{vn}'
        st, gb = load(root, tag, K)
        vm, reg, h, hf, R6, Rf = view_eval(st, gb, g, l[0]['pos'])
        vm['hist'] = {a: hist_pass(vm, a, 'O_U', 'D6') for a in ('A', 'A1r', 'A2', 'S4_D6')}
        vm['hist_fullbands'] = {a: hist_pass(vm, a, 'OB1', 'F') for a in ('B1', 'B2', 'S4_F')}
        vm['gate3_full'] = {a: gate3_pass(vm, a, 'gate3_full_depth') for a in vm['arms']}
        vm['gate3_dm'] = {a: gate3_pass(vm, a, 'gate3_depth_matched') for a in D6_ARMS if a in vm['arms']}
        vm['decomposition'] = decomposition(vm)
        D[tag] = vm
        save_maps(an, f'D_{tag}', st, h, hf, R6, Rf, reg)
        print('D', tag, {a: vm['gate3_full'][a][0] for a in ('B1', 'B2', 'A1r', 'S4_F')}, flush=True)
    R6b = {}
    for f in STATIC:
        rows = [v for t, v in D.items() if t.startswith(f + '_') and v['contact_valid']]
        R6b[f] = {a: {'full': spread(rows, a, 'gate3_full_depth'), 'dm': spread(rows, a, 'gate3_depth_matched')} for a in ('B1', 'B2', 'A1r', 'S4_F', 'S4_D6')}
    for sname in EV.SEQS:
        q = C.sequences()[sname]; rows = []
        for t, g in enumerate(q['states']):
            row = {'label': q['label'][t], 'views': {}, 'rebuild': {a: rb.get(f'{sname}_{t}_{a}') for a in ('A1r', 'B1')}}
            for vn in q['views']:
                tag = f'{sname}_{t}_{vn}'
                st, gb = load(root, tag, K)
                vm, reg, h, hf, R6, Rf = view_eval(st, gb, g, q['light'][0]['pos'])
                vm['hist'] = {a: hist_pass(vm, a, 'O_U', 'D6') for a in ('A', 'A1r', 'S4_D6')}
                vm['hist_fullbands'] = {a: hist_pass(vm, a, 'OB1', 'F') for a in ('B1', 'B2', 'S4_F')}
                vm['gate3_full'] = {a: gate3_pass(vm, a, 'gate3_full_depth') for a in vm['arms']}
                vm['gate3_dm'] = {a: gate3_pass(vm, a, 'gate3_depth_matched') for a in D6_ARMS if a in vm['arms']}
                vm['decomposition'] = decomposition(vm)
                for cur, stale in (('B1', 'stale_B1'), ('A1r', 'stale_A1r')):
                    if stale in vm['arms'] and vm['contact_valid']:
                        s_, c_ = vm['arms'][stale], vm['arms'][cur]
                        key = 'endpoints_full' if cur == 'B1' else 'endpoints_D6'
                        rk = 'rmse_full' if cur == 'B1' else 'rmse_D6'
                        vm[f'stale_worse_{cur}'] = bool(abs(s_[key]['near']['estimate']) - abs(c_[key]['near']['estimate']) >
                                                        s_[key]['near']['half_width'] + c_[key]['near']['half_width'] or s_[rk]['near'] > c_[rk]['near'] + 0.01)
                row['views'][vn] = vm
                if t in (0, 1, 3, 4):
                    save_maps(an, f'E_{tag}', st, h, hf, R6, Rf, reg)
            rows.append(row)
            print('E', sname, t, {v: {a: r['gate3_full'][a][0] for a in ('B1', 'B2', 'A1r')} for v, r in row['views'].items()}, flush=True)
        trans = []
        for t in range(1, len(rows)):
            for vn in q['views']:
                a, b = rows[t - 1]['views'][vn], rows[t]['views'][vn]
                if a['contact_valid'] and b['contact_valid']:
                    dmu = b['mu_near_high_full'] - a['mu_near_high_full']; se = math.sqrt(a['ref_near_full_se'] ** 2 + b['ref_near_full_se'] ** 2)
                    trans.append({'from': t - 1, 'to': t, 'view': vn, 'delta_full_high': dmu, 'se': se, 'resolved': bool(abs(dmu) >= 5 * se),
                                  'stale_worse_B1': b.get('stale_worse_B1'), 'stale_worse_A1r': b.get('stale_worse_A1r')})
        rev = []
        for i, j in q['revisits']:
            for a in ('A1r', 'B1'):
                ti = np.load(root / 'stageD' / f'table_{sname}_{i}_{a}.npy'); tj = np.load(root / 'stageD' / f'table_{sname}_{j}_{a}.npy')
                rev.append({'states': [i, j], 'arm': a, 'table_max_rel': float(np.abs(ti - tj).max() / max(np.abs(ti).max(), 1e-30))})
        track = None
        if 'gaps' in q:
            track = {}
            for arm, key in (('B1', 'full'), ('B2', 'full'), ('A1r', 'D6'), ('A1r_vs_full', 'full')):
                tk = {}
                for vn in q['views']:
                    ref, pred = [], []
                    for row in rows:
                        v = row['views'][vn]
                        if v['contact_valid']:
                            a_ = 'A1r' if arm == 'A1r_vs_full' else arm
                            mu = v['mu_near_high_full'] if key == 'full' else v['mu_near_high_D6']
                            ref.append(mu); pred.append(mu + v['arms'][a_][f'endpoints_{key}']['near']['estimate'])
                    ref, pred = np.array(ref), np.array(pred)
                    d_r, d_p = ref - ref[0], pred - pred[0]
                    slope = float((d_r @ d_p) / max(d_r @ d_r, 1e-30)); corr = float(np.corrcoef(d_r, d_p)[0, 1]) if len(ref) > 2 else float('nan')
                    tk[vn] = {'slope': slope, 'corr': corr, 'pass': bool(abs(slope - 1) <= 0.1 and corr >= 0.98)}
                track[arm] = tk
        far = {}
        for arm, rk in (('B1', 'rmse_full'), ('A1r', 'rmse_D6')):
            far[arm] = []
            for vn in q['views']:
                base = rows[0]['views'][vn]['arms'][arm][rk]['far_unchanged']
                for t, row in enumerate(rows):
                    v = row['views'][vn]['arms'][arm][rk]['far_unchanged']; far[arm].append({'state': t, 'view': vn, 'ok': bool(v <= 1.10 * base + 0.0025)})
        def seq_ok(arm, gkey):
            allv = all(r['views'][v][gkey][arm][0] for r in rows for v in r['views'])
            dyn = (all(x['table_max_rel'] <= 1e-6 for x in rev if x['arm'] == (arm if arm in ('A1r', 'B1') else 'B1')) and
                   all((r['rebuild'].get(arm if arm in ('A1r', 'B1') else 'B1') or {'table_max_rel': 0})['table_max_rel'] <= 1e-6 for r in rows) and
                   (track is None or all(t['pass'] for t in track.get(arm, track['B1']).values())))
            return bool(allv), bool(dyn)
        disc_B1 = [t for t in trans if t['resolved'] and t['stale_worse_B1']]
        disc_A1r = [t for t in trans if t['resolved'] and t['stale_worse_A1r']]
        hist_A1r = all(r['views'][v]['hist']['A1r']['pass'] for r in rows for v in r['views'])
        E[sname] = {'states': rows, 'transitions': trans, 'discriminating_B1': len(disc_B1), 'discriminating_A1r': len(disc_A1r), 'revisits': rev,
                    'tracking_R7': track, 'far_unchanged': far, 'historical_A1r_all_states': bool(hist_A1r),
                    'gate3_full_all_states': {a: seq_ok(a, 'gate3_full')[0] for a in ('B1', 'B2', 'A1r', 'S4_F')},
                    'dynamic_checks': {a: seq_ok(a, 'gate3_full')[1] for a in ('B1', 'A1r')},
                    'far_ok': {a: bool(all(f['ok'] for f in far[a])) for a in far}}
    C.jd(D, an / 'stageD_metrics.json'); C.jd(E, an / 'stageE_metrics.json'); C.jd(R6b, an / 'R6b.json')
    # ---------------- verdicts
    rows = [v for v in D.values() if v['contact_valid']] + [r['views'][vn] for s in E.values() for r in s['states'] for vn in r['views'] if r['views'][vn]['contact_valid']]
    eligible = [v for v in rows if abs(v['trunc_near_rel']) >= 0.03]
    H1 = {}
    tn = {a: json.loads((root / 'stageC' / a / 'training.json').read_text()) for a in ('A1r', 'B1', 'A2', 'B2')}
    ratio = {a: tn[a]['validation']['refined_whole_rel_l1'] / tn[a]['target_noise']['refined_half_noise_rel_l1'] for a in tn}
    for b, a in (('B1', 'A1r'), ('B2', 'A2')):
        red = [abs(v['arms'][b]['b_full']) < abs(v['arms'][a]['b_full']) for v in eligible]
        lex = [abs(v['arms'][b]['b_full'] - v['arms']['S4_F']['b_full']) <= 1.25 * abs(v['arms'][a]['b_D6'] - v['arms']['S4_D6']['b_D6']) + 0.005 for v in eligible]
        lex2 = [abs(v['arms'][b]['b_full'] - v['arms']['S4_F']['b_full']) > 2 * abs(v['arms'][a]['b_D6'] - v['arms']['S4_D6']['b_D6']) for v in eligible]
        H1[b] = {'n_eligible': len(eligible), 'reduce_frac': float(np.mean(red)) if red else float('nan'), 'learned_excess_ok_frac': float(np.mean(lex)) if lex else float('nan'),
                 'learned_excess_2x_frac': float(np.mean(lex2)) if lex2 else float('nan'), 'val_noise_ratio': ratio[b], 'val_noise_ratio_matched_A': ratio[a],
                 'table_ok': bool(ratio[b] <= 1.25 * ratio[a])}
    sup = all(H1[b]['reduce_frac'] >= 0.9 and H1[b]['learned_excess_ok_frac'] >= 0.9 and H1[b]['table_ok'] for b in H1)
    nsup = any(H1[b]['reduce_frac'] < 0.5 or H1[b]['learned_excess_2x_frac'] >= 0.5 for b in H1)
    H1['verdict'] = 'SUPPORTED' if sup else ('NOT SUPPORTED' if nsup else 'INCONCLUSIVE')
    # remaining-error attribution (B1, full-depth gate)
    attrib = {}
    for v in rows:
        gB = v['arms']['B1']['gate3_full_depth']['pass']; gS = v['arms']['S4_F']['gate3_full_depth']['pass']
        for req, ok in gB.items():
            if not ok:
                cls = 'spatial' if not gS.get(req, True) else 'learned'
                attrib.setdefault(req, {'spatial': 0, 'learned': 0}); attrib[req][cls] += 1
    # gate
    def static_all(arm):
        return all(v['gate3_full'][arm][0] for v in D.values()) and all((R6b[f][arm]['full'] or {'pass': True})['pass'] for f in R6b)
    seq_all = lambda arm: all(s['gate3_full_all_states'][arm] for s in E.values())
    dyn_B1 = all(s['dynamic_checks']['B1'] and s['far_ok']['B1'] and s['discriminating_B1'] >= 1 for s in E.values())
    met = static_all('B1') and static_all('B2') and seq_all('B1') and seq_all('B2') and dyn_B1
    gate = {'classification': 'PARTIALLY_RESOLVED' if met else 'NOT_RESOLVED', 'K_star': K,
            'static_full_gate_all_views': {a: bool(static_all(a)) for a in ('B1', 'B2', 'A1r', 'S4_F')},
            'sequences_full_gate_all_states': {a: bool(seq_all(a)) for a in ('B1', 'B2', 'A1r', 'S4_F')},
            'dynamic_B1': bool(dyn_B1), 'not_run': ['R3 equal-time frontier comparator', 'R9 other radius / medium'],
            'note': 'RESOLVED is unreachable in this batch (R3 / R9 NOT_RUN)'}
    C.jd({'H1': H1, 'remaining_error_attribution_B1': attrib, 'gate': gate}, an / 'verdicts.json'); C.jd(gate, an / 'gate.json')
    print('VERDICT', json.dumps(H1['verdict']), 'GATE', gate['classification'], flush=True)


if __name__ == '__main__':
    {'B': stageB, 'D': stageD}[sys.argv[2]](Path(sys.argv[1]))
