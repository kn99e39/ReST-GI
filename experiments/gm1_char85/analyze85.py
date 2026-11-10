"""WL85 Stages 2-3 analysis: geometric regions, reference fidelity, signed change, depth and class accounting, workload qualification.

  python analyze85.py <root>
Reads results/renders/{ER,P}_{T0,T1}_r<k>.npz (+ .json), results/gbuffer.npz. Writes results/analysis.json and results/maps.npz.
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, 'protocol', 'char85.json')))
FX = P['stage1_fixture']
NR = P['stage2_references']['replicas']
CLS = 'SMB'


def seg_box(o, p, lo, hi):
    d = p - o
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    t0 = (lo - o) / d; t1 = (hi - o) / d
    tmin = np.minimum(t0, t1).max(-1); tmax = np.maximum(t0, t1).min(-1)
    return (tmax >= np.maximum(tmin, 0)) & (tmin <= 1)


def regions(gb):
    mid0, p0, mid1, p1 = gb['mid0'], gb['p0'], gb['mid1'], gb['p1']
    S = FX['shapes']
    c0 = np.array(S['baffle']['center_T0']); h = np.array(S['baffle']['half']); c1 = c0 + np.array(FX['edit']['translation'])
    b0, b1 = (c0 - h, c0 + h), (c1 - h, c1 + h)
    Lp = np.array(FX['light']['position'])
    stat = (mid0 == mid1) & (mid0 >= 0) & (mid0 != 4) & (np.abs(p0 - p1).max(-1) < 1e-4)
    toL0 = seg_box(p0, Lp, *b0); toL1 = seg_box(p0, Lp, *b1)
    px = S['panel']['center'][0]
    Lm = Lp.copy(); Lm[0] = 2 * px - Lp[0]
    den = p0[..., 0] - Lm[0]
    t = (px - Lm[0]) / np.where(np.abs(den) < 1e-9, 1e-9, den)
    q = Lm + t[..., None] * (p0 - Lm)
    mirror = (mid0 == 0) & (t > 0) & (t < 1) & (np.abs(q[..., 1] - S['panel']['center'][1]) <= S['panel']['half'][0]) & (np.abs(q[..., 2] - S['panel']['center'][2]) <= S['panel']['half'][1])
    Lq0 = seg_box(np.broadcast_to(Lp, q.shape), q, *b0); Lq1 = seg_box(np.broadcast_to(Lp, q.shape), q, *b1)
    qp0 = seg_box(q, p0, *b0); qp1 = seg_box(q, p0, *b1)
    R = {}
    R['enabled_reflection'] = stat & mirror & Lq0 & ~Lq1 & ~qp0 & ~qp1
    R['shadow_new'] = stat & toL1 & ~toL0 & ~R['enabled_reflection']
    R['shadow_gone'] = stat & toL0 & ~toL1 & ~R['enabled_reflection']
    used = R['enabled_reflection'] | R['shadow_new'] | R['shadow_gone']
    R['reflection_static'] = stat & mirror & ~Lq0 & ~Lq1 & ~qp0 & ~qp1 & ~used
    R['stationary_rest'] = stat & ~used & ~R['reflection_static']
    R['stationary_all'] = stat
    for name, k in [('floor', 0), ('panel', 1), ('sphere', 2), ('wall', 3)]:
        R[name] = stat & (mid0 == k)
    R['mover'] = (mid0 == 4) | (mid1 == 4)
    R['frame'] = np.ones_like(stat)
    return R


def load(root, ref, cfg):
    arrs, meta = {}, []
    for r in range(NR):
        z = np.load(os.path.join(root, 'results', 'renders', f'{ref}_{cfg}_r{r}.npz'))
        for k in z.files:
            arrs.setdefault(k, []).append(z[k].astype(np.float32))
        meta.append(json.load(open(os.path.join(root, 'results', 'renders', f'{ref}_{cfg}_r{r}.json'))))
    return {k: np.stack(v) for k, v in arrs.items()}, meta


def se(x):
    x = np.asarray(x, np.float64)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def rel_l1(a, b, m):
    return float(np.abs(a[m] - b[m]).sum() / max(b[m].sum(), 1e-30))


def main():
    root = sys.argv[1]
    Rr = os.path.join(root, 'results')
    gb = dict(np.load(os.path.join(Rr, 'gbuffer.npz')))
    R = regions(gb)
    D = {(ref, cfg): load(root, ref, cfg) for ref in ['ER', 'P'] for cfg in ['T0', 'T1']}
    A, B_ = slice(0, NR // 2), slice(NR // 2, NR)
    Y = lambda x: x.mean(-1)
    out = {'pixels': {k: int(v.sum()) for k, v in R.items()}, 'regions': {}, 'refs': {}}
    maps = {'mask_' + k: v for k, v in R.items()}
    pooled = {}
    for ref in ['ER', 'P']:
        a0, a1 = D[(ref, 'T0')][0], D[(ref, 'T1')][0]
        d = Y(a1['dirty']) - Y(a0['dirty'])                         # [NR, H, W] signed change (clean paths cancel exactly)
        pooled[ref] = {'T0': a0['tot'].mean(0), 'T1': a1['tot'].mean(0), 'T1A': a1['tot'][A].mean(0), 'T1B': a1['tot'][B_].mean(0), 'T0A': a0['tot'][A].mean(0), 'T0B': a0['tot'][B_].mean(0),
                       'd': d.mean(0), 'dA': d[A].mean(0), 'dB': d[B_].mean(0), 'd_naive': (Y(a1['tot']) - Y(a0['tot'])).mean(0),
                       'created': Y(a1['dirty']).mean(0), 'destroyed': Y(a0['dirty']).mean(0), 'dmov1': a1['dmov'].mean(0), 'dmov0': a0['dmov'].mean(0),
                       'depth1': a1['depth'].mean(0), 'depth0': a0['depth'].mean(0), 'ddepth': (a1['ddepth'] - a0['ddepth']).mean(0),
                       'ddepthA': (a1['ddepth'] - a0['ddepth'])[A].mean(0), 'ddepthB': (a1['ddepth'] - a0['ddepth'])[B_].mean(0)}
        if ref == 'ER':
            pooled[ref].update({'dir1': a1['dir'].mean(0), 'mat1': a1['mat'].mean(0), 'ddir': (a1['ddir'] - a0['ddir']).mean(0), 'dmat': (a1['dmat'] - a0['dmat']).mean(0),
                                'ddirA': (a1['ddir'] - a0['ddir'])[A].mean(0), 'ddirB': (a1['ddir'] - a0['ddir'])[B_].mean(0),
                                'dmatA': (a1['dmat'] - a0['dmat'])[A].mean(0), 'dmatB': (a1['dmat'] - a0['dmat'])[B_].mean(0)})
        for k in ['T0', 'T1', 'd', 'created', 'destroyed', 'depth1', 'ddepth']:
            maps[f'{ref}_{k}'] = pooled[ref][k].astype(np.float32)
        cn = {cfg: [m['counters'] for m in D[(ref, cfg)][1]] for cfg in ['T0', 'T1']}
        tm = {cfg: [m['timing'] for m in D[(ref, cfg)][1]] for cfg in ['T0', 'T1']}
        out['refs'][ref] = {
            'path_dirty_fraction': {cfg: se([c['paths_dirty'] / c['paths'] for c in cn[cfg]]) for cfg in cn},
            'segment_dirty_fraction': {cfg: se([c['seg_dirty'] / c['seg_tot'] for c in cn[cfg]]) for cfg in cn},
            'segments_per_path': {cfg: se([c['seg_tot'] / c['paths'] for c in cn[cfg]]) for cfg in cn},
            'render_timing_median': {k: float(np.median([t[k] for t in tm['T1']])) for k in ['trace_s', 'eval_s', 'readback_s', 'gpu_ms', 'wall_s']},
            'truncation_depth6_share_T1': float(pooled[ref]['depth1'][..., 5].sum() / pooled[ref]['depth1'].sum()),
            'clean_identity_max_rel': float(np.abs((Y(a1['tot']) - Y(a1['dirty'])).mean(0) - (Y(a0['tot']) - Y(a0['dirty'])).mean(0)).max() / Y(a0['tot']).mean())}
    maps['ERmat1'] = pooled['ER']['mat1'].astype(np.float32); maps['ERdir1'] = pooled['ER']['dir1'].astype(np.float32)
    maps['ERdmat'] = pooled['ER']['dmat'].astype(np.float32); maps['ERddir'] = pooled['ER']['ddir'].astype(np.float32)
    for name, m in R.items():
        if m.sum() < 30:
            out['regions'][name] = {'pixels': int(m.sum()), 'note': 'too few pixels'}
            continue
        r = {'pixels': int(m.sum())}
        for ref in ['ER', 'P']:
            a0, a1 = D[(ref, 'T0')][0], D[(ref, 'T1')][0]
            pr = pooled[ref]
            t0 = [Y(a0['tot'][k])[m].mean() for k in range(NR)]; t1 = [Y(a1['tot'][k])[m].mean() for k in range(NR)]
            dd = [(Y(a1['dirty'][k]) - Y(a0['dirty'][k]))[m].mean() for k in range(NR)]
            t0m = float(np.mean(t0))
            q = {'T0': se(t0), 'T1': se(t1), 'signed_change': se(dd), 'signed_change_rel_T0': float(np.mean(dd) / max(t0m, 1e-30)),
                 'signed_change_snr': float(abs(np.mean(dd)) / max(se(dd)[1], 1e-30)),
                 'abs_change_rel_T0': float(np.abs(pr['d'][m]).mean() / max(t0m, 1e-30)),
                 'abs_change_noise_floor_rel_T0': float(0.5 * np.abs(pr['dA'][m] - pr['dB'][m]).mean() / max(t0m, 1e-30)),
                 'created_rel_T0': float(pr['created'][m].mean() / max(t0m, 1e-30)), 'destroyed_rel_T0': float(pr['destroyed'][m].mean() / max(t0m, 1e-30)),
                 'via_mover_vertex_rel_T0': [float(pr['dmov0'][m].mean() / max(t0m, 1e-30)), float(pr['dmov1'][m].mean() / max(t0m, 1e-30))],
                 'dirty_share_of_T1': float(pr['created'][m].sum() / max(Y(pr['T1'])[m].sum(), 1e-30)),
                 'depth_share_T1': (pr['depth1'][m].sum(0) / max(pr['depth1'][m].sum(), 1e-30)).tolist(),
                 'depth_share_T1_se': [se([a1['depth'][k][m][:, j].sum() / max(a1['depth'][k][m].sum(), 1e-30) for k in range(NR)])[1] for j in range(6)],
                 'depth_signed_change_rel_T0': (pr['ddepth'][m].mean(0) / max(t0m, 1e-30)).tolist(),
                 'depth_abs_change_share': (np.abs(pr['ddepth'][m]).sum(0) / max(np.abs(pr['ddepth'][m]).sum(), 1e-30)).tolist(),
                 'depth_abs_change_noise_share': (0.5 * np.abs(pr['ddepthA'][m] - pr['ddepthB'][m]).sum(0) / max(np.abs(pr['ddepth'][m]).sum(), 1e-30)).tolist(),
                 'uncertainty_rel_l1_halves_T1': 0.5 * rel_l1(pr['T1A'], pr['T1B'], m), 'uncertainty_rel_l1_halves_T0': 0.5 * rel_l1(pr['T0A'], pr['T0B'], m)}
            if ref == 'ER':
                tot1 = max(pr['dir1'][m].sum() + pr['mat1'][m].sum(), 1e-30)
                cells = np.concatenate([np.abs(pr['ddir'][m]).sum(0), np.abs(pr['dmat'][m]).sum(0)])
                noise = np.concatenate([0.5 * np.abs(pr['ddirA'][m] - pr['ddirB'][m]).sum(0), 0.5 * np.abs(pr['dmatA'][m] - pr['dmatB'][m]).sum(0)])
                tc = max(cells.sum(), 1e-30)
                names = [f'k1:{a}' for a in CLS] + [f'{a}<-{b}' for a in CLS for b in CLS]         # receiver <- sender
                sharp = sum(c for n, c in zip(names, cells) if 'S' in n) / tc
                broad = sum(c for n, c in zip(names, cells) if set(n.replace('k1:', '').replace('<-', '')) <= {'B'}) / tc
                modr = sum(c for n, c in zip(names, cells) if 'M' in n) / tc
                q['class'] = {'cells': names,
                              'T1_share': (np.concatenate([pr['dir1'][m].sum(0), pr['mat1'][m].sum(0)]) / tot1).tolist(),
                              'signed_change_rel_T0': (np.concatenate([pr['ddir'][m].mean(0), pr['dmat'][m].mean(0)]) / max(t0m, 1e-30)).tolist(),
                              'abs_change_share': (cells / tc).tolist(), 'abs_change_noise_share': (noise / tc).tolist(),
                              'abs_change_share_any_S': float(sharp), 'abs_change_share_all_B': float(broad), 'abs_change_share_any_M': float(modr),
                              'label': 'BROAD-DOMINATED' if broad >= 0.75 else ('SHARP-INDISPENSABLE' if sharp >= 0.25 else 'MIXED')}
            r[ref] = q
        # ER vs P
        pe, pp = pooled['ER'], pooled['P']
        for cfg in ['T0', 'T1']:
            g = rel_l1(pe[cfg], pp[cfg], m)
            unc = max(0.5 * rel_l1(pp[cfg + 'A'], pp[cfg + 'B'], m), 0.5 * rel_l1(pe[cfg + 'A'], pe[cfg + 'B'], m))
            yp = Y(pp[cfg])
            hl = m & (yp >= 3 * np.median(yp[m]))
            r[f'gap_{cfg}'] = {'rel_l1': g, 'uncertainty': unc, 'status': 'QUALIFIED' if unc <= g / 4 else 'INDETERMINATE',
                               'signed_rel': float((pe[cfg][m].sum() - pp[cfg][m].sum()) / max(pp[cfg][m].sum(), 1e-30)),
                               'highlight_pixels': int(hl.sum()), 'highlight_rel_l1': rel_l1(pe[cfg], pp[cfg], hl) if hl.sum() >= 30 else None,
                               'highlight_uncertainty': 0.5 * rel_l1(pp[cfg + 'A'], pp[cfg + 'B'], hl) if hl.sum() >= 30 else None}
        x, xa, xb, y, ya, yb = pp['d'][m], pp['dA'][m], pp['dB'][m], pe['d'][m], pe['dA'][m], pe['dB'][m]
        sxx = float((xa * xb).sum()); syy = float((ya * yb).sum()); sxy = float((x * y).sum())
        sl = sxy / sxx if sxx > 0 else None
        co = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None
        snr_x = sxx / max(float((x * x).sum()) - sxx, 1e-30) if sxx > 0 else 0.0
        st = 'INDETERMINATE' if (sl is None or co is None or r['P']['signed_change_snr'] < 3 and r['P']['abs_change_rel_T0'] < 4 * r['P']['abs_change_noise_floor_rel_T0']) else \
            ('ADEQUATE' if 0.9 <= sl <= 1.1 and co >= 0.9 else 'LIMITED')
        r['change_ER_vs_P'] = {'slope': sl, 'correlation': co, 'status': st, 'mean_change_ratio_ER_over_P': float(np.mean(y) / np.mean(x)) if abs(np.mean(x)) > 0 else None}
        out['regions'][name] = r
    # ---- workload qualification (REF-P)
    g = lambda reg, ref='P': out['regions'][reg][ref]
    Q = {}
    try:
        e = g('enabled_reflection'); s = g('shadow_new'); st = g('stationary_all'); sp = g('sphere')
        Q['Q1'] = {'value': e['signed_change_rel_T0'], 'snr': e['signed_change_snr'], 'PASS': bool(e['signed_change_rel_T0'] >= 0.05 and e['signed_change_snr'] >= 5)}
        Q['Q2'] = {'value': s['signed_change_rel_T0'], 'snr': s['signed_change_snr'], 'PASS': bool(s['signed_change_rel_T0'] <= -0.05 and s['signed_change_snr'] >= 5)}
        k3 = float(sum(st['depth_share_T1'][2:])); k3se = float(math.sqrt(sum(x * x for x in st['depth_share_T1_se'][2:])))
        ind = float(sum(e['depth_signed_change_rel_T0'][1:]) / e['signed_change_rel_T0']) if e['signed_change_rel_T0'] != 0 else 0.0
        Q['Q3'] = {'k_ge_3_share_T1': k3, 'snr': k3 / max(k3se, 1e-30), 'k_ge_2_share_of_change_enabled': ind, 'PASS': bool(k3 >= 0.02 and k3 / max(k3se, 1e-30) >= 5 and ind >= 0.5)}
        k2 = float(sum(sp['depth_share_T1'][1:]))
        Q['Q4'] = {'k_ge_2_share_T1_sphere': k2, 'PASS': bool(k2 >= 0.05)}
        Q['Q5'] = {'uncertainty': st['uncertainty_rel_l1_halves_T1'], 'PASS': bool(st['uncertainty_rel_l1_halves_T1'] <= 0.02)}
        Q['QUALIFIED'] = bool(all(v['PASS'] for v in Q.values()))
    except KeyError as ex:
        Q['QUALIFIED'] = False; Q['error'] = f'missing region {ex}'
    out['workload_qualification'] = Q
    json.dump(out, open(os.path.join(Rr, 'analysis.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(Rr, 'maps.npz'), **maps)
    print('QUALIFIED' if Q['QUALIFIED'] else 'UNSUITABLE', json.dumps(Q), flush=True)


if __name__ == '__main__':
    main()
