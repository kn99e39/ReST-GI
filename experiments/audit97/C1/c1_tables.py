"""WL97 audit, track C1: independent recomputation of the WL94..WL94D CK2 table metrics from the stored raw tables / field references.

Own numpy implementation of the protocol definitions (exitance = sum_bins,channels X * cos * dOmega; log-response amplitude = least-squares
slope through the origin of model log-ratio on reference log-ratio; corr = Pearson; rel L1 on per-patch exitance). Nothing from the audited
an94 / b94 / s94 modules is imported. Every metric is produced twice:
  coded : the weight vector as coded in an94.W  -> ((bin // 8) + 0.5) / 8 * 2pi/64     (8 rings x 8 sectors; NOT the table layout)
  true  : the weight vector of the real layout  -> ((bin // 16) + 0.5) / 4 * 2pi/64    (4 rings x 16 sectors, bin = iz * 16 + iphi)
Usage: python c1_tables.py <out.json>
Reads (read-only): ~/rest_x94, ~/rest_x94r, ~/rest_x94b, ~/rest_x94c, ~/rest_x94d  results/{tables,fields,pre}
"""
import hashlib, json, os, sys
import numpy as np

NB = 64
DOM = 2 * np.pi / NB
W_CODED = ((np.arange(NB) // 8) + 0.5) / 8 * DOM
W_TRUE = ((np.arange(NB) // 16) + 0.5) / 4 * DOM
VARIANTS = ['none', 'U07', 'U03', 'bx75', 'bx38', 'bx38i']
TEX = VARIANTS[1:]
CHECK = ['bx75', 'bx38', 'bx38i']
HOME = os.path.expanduser('~')


def sha(fn):
    h = hashlib.sha256()
    with open(fn, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()[:16]


def exit_(X, w):
    return np.einsum('nbc,b->n', X.astype(np.float64), w)


def ls_slope(m, r):
    d = float((r * r).sum())
    return float((m * r).sum() / d) if d > 0 else None


def pearson(m, r):
    if len(m) < 3 or m.std() == 0 or r.std() == 0:
        return None
    mm, rr = m - m.mean(), r - r.mean()
    return float((mm * rr).sum() / np.sqrt((mm * mm).sum() * (rr * rr).sum()))


def load_fields(root):
    F = {}
    for v in VARIANTS:
        z = np.load(os.path.join(root, 'results', 'fields', f'CKF_{v}.npz'))
        d = {}
        for h in 'ab':
            p, o = z[f'e2_p_{h}'].astype(np.float64), z[f'e2_o_{h}'].astype(np.float64)
            g = (z[f'e3_p_{h}'] + z[f'e3_o_{h}']).astype(np.float64)        # float32 sum first, as the stored convention
            d['Sp_' + h], d['So_' + h], d['S_' + h], d['G_' + h] = p, o, p + o, g
            d['B_' + h] = p + o + g
        for k in ('Sp', 'So', 'S', 'G', 'B'):
            d[k] = 0.5 * (d[k + '_a'] + d[k + '_b'])
        F[v] = d
    return F


def response(tab, F, v, km, kr, P, w):
    mv, m0 = exit_(tab[f'{v}|{km}'], w)[P], exit_(tab[f'none|{km}'], w)[P]
    rv, r0 = exit_(F[v][kr], w)[P], exit_(F['none'][kr], w)[P]
    ok = (mv > 0) & (m0 > 0) & (rv > 0) & (r0 > 0)
    lm, lr = np.log(mv[ok] / m0[ok]), np.log(rv[ok] / r0[ok])
    la = np.log(exit_(F[v][kr + '_a'], w)[P][ok] / exit_(F['none'][kr + '_a'], w)[P][ok])
    lb = np.log(exit_(F[v][kr + '_b'], w)[P][ok] / exit_(F['none'][kr + '_b'], w)[P][ok])
    hn = 0.5 * np.abs(la - lb)
    return {'amplitude': ls_slope(lm, lr), 'corr': pearson(lm, lr), 'patches': int(ok.sum()),
            'ref_snr_median': float(np.median(np.abs(lr) / np.maximum(hn, 1e-12))) if ok.any() else None,
            'amplitude_vs_half_a': ls_slope(lm, la), 'amplitude_vs_half_b': ls_slope(lm, lb),
            'baseline_err': float(np.abs(m0[ok] - r0[ok]).sum() / r0[ok].sum()),
            'baseline_signed': float((m0[ok] - r0[ok]).sum() / r0[ok].sum()),
            'baseline_half_noise': float(0.5 * np.abs(exit_(F['none'][kr + '_a'], w)[P][ok] - exit_(F['none'][kr + '_b'], w)[P][ok]).sum() / r0[ok].sum()),
            'textured_err': float(np.abs(mv[ok] - rv[ok]).sum() / rv[ok].sum())}


def group_amp(tab, F, v, arm, P, bm, w):
    def eg(X):
        return np.einsum('nbc,nb->n', X.astype(np.float64), bm * w[None, :])[P]
    mv, m0, rv, r0 = eg(tab[f'{v}|B_{arm}']), eg(tab[f'none|B_{arm}']), eg(F[v]['B']), eg(F['none']['B'])
    ok = (mv > 0) & (m0 > 0) & (rv > 0) & (r0 > 0)
    lm, lr = np.log(mv[ok] / m0[ok]), np.log(rv[ok] / r0[ok])
    return {'amplitude': ls_slope(lm, lr), 'corr': pearson(lm, lr), 'patches': int(ok.sum())}


def table_metrics(tab, F, M, arms, w):
    T, Tp, To = M['T'], M['T_panel_face'], M['T_other_faces']
    pfb = M['panel_facing_bins']; oth = T[:, None] & ~pfb
    out = {}
    for arm in arms:
        if f'none|B_{arm}' not in tab:
            continue
        a = {}
        for v in TEX:
            for cn, km, kr in (('B', f'B_{arm}', 'B'), ('S', f'S_{arm}', 'S'), ('G', 'G', 'G')):
                for sn, P in (('T', T), ('T_panel_face', Tp), ('T_other_faces', To)):
                    a[f'{v}|{cn}|{sn}'] = response(tab, F, v, km, kr, P, w)
            a[f'{v}|B|panel_facing_bins'] = group_amp(tab, F, v, arm, T, pfb, w)
            a[f'{v}|B|other_bins'] = group_amp(tab, F, v, arm, T, oth, w)
        nonT = ~T
        for v in TEX:
            rv, r0 = exit_(F[v]['B'], w), exit_(F['none']['B'], w)
            la = np.log(np.maximum(exit_(F[v]['B_a'], w), 1e-30) / np.maximum(exit_(F['none']['B_a'], w), 1e-30))
            lb = np.log(np.maximum(exit_(F[v]['B_b'], w), 1e-30) / np.maximum(exit_(F['none']['B_b'], w), 1e-30))
            lr = np.log(np.maximum(rv, 1e-30) / np.maximum(r0, 1e-30))
            q = nonT & (r0 > 0) & (np.abs(lr) > 4 * 0.5 * np.abs(la - lb))
            mv, m0 = exit_(tab[f'{v}|B_{arm}'], w), exit_(tab[f'none|B_{arm}'], w)
            for nm, sel in (('Q_S1', q & M['S1']), ('Q_nonS1', q & ~M['S1'])):
                ok = sel & (mv > 0) & (m0 > 0)
                lm = np.log(mv[ok] / m0[ok])
                a[f'{v}|receivers|{nm}'] = {'amplitude': ls_slope(lm, lr[ok]), 'corr': pearson(lm, lr[ok]), 'patches': int(ok.sum())}
        out[arm] = a
    ss = {}
    for nm, km in (('C0_fromS1', 'S_C0_fromS1'), ('C0_fromOther', 'S_C0_fromOther'), ('C2_rcv_fromS1', 'S_C2_rcv_fromS1'), ('C2_rcv_fromOther', 'S_C2_rcv_fromOther')):
        if f'none|{km}' not in tab:
            continue
        kr = 'Sp' if nm.endswith('S1') else 'So'
        r = response(tab, F, 'bx38', km, kr, T, w)
        rp = response(tab, F, 'bx38', km, kr, Tp, w)
        ss[nm] = {'baseline_err_T': r['baseline_err'], 'baseline_signed_T': r['baseline_signed'], 'baseline_half_noise_T': r['baseline_half_noise'],
                  'baseline_err_Tpf': rp['baseline_err'], 'baseline_half_noise_Tpf': rp['baseline_half_noise'],
                  'ref_share_of_S_untextured_T': float(exit_(F['none'][kr], w)[T].sum() / exit_(F['none']['S'], w)[T].sum()),
                  'ref_share_of_S_untextured_Tpf': float(exit_(F['none'][kr], w)[Tp].sum() / exit_(F['none']['S'], w)[Tp].sum())}
        for v in TEX:
            rr = response(tab, F, v, km, kr, T, w)
            ss[nm][v] = {'amplitude': rr['amplitude'], 'corr': rr['corr'], 'textured_err': rr['textured_err']}
    out['sender_split'] = ss
    return out


def mr_ok(tm):
    c = []
    for v in TEX:
        r = tm[f'{v}|B|T']
        c.append(r['amplitude'] is not None and 0.85 <= r['amplitude'] <= 1.15)
        if v in CHECK:
            c.append(r['corr'] is not None and r['corr'] >= 0.9)
    return bool(all(c))


JOBS = [  # (label, root, table file stem, arms)
    ('WL94|own_s8601', 'rest_x94', 'FZ_own_s8601', ['C0', 'C2_src', 'C2_rcv', 'C2_both', 'C2_srcS1']),
    ('WL94|own_s8602', 'rest_x94', 'FZ_own_s8602', ['C0', 'C2_src', 'C2_rcv', 'C2_both', 'C2_srcS1']),
    ('WL94|A_s8601', 'rest_x94', 'CD_A_s8601', ['C0', 'C2_rcv', 'C2_srcS1']),
    ('WL94|B_s8601', 'rest_x94', 'CD_B_s8601', ['C0', 'C2_rcv', 'C2_srcS1']),
    ('WL94A|C_s8601', 'rest_x94r', 'CD_C_s8601', ['C0', 'C2_rcv', 'C2_srcS1']),
    ('WL94A|C_s8602', 'rest_x94r', 'CD_C_s8602', ['C0', 'C2_rcv', 'C2_srcS1']),
    ('WL94A|A_s8602', 'rest_x94r', 'CD_A_s8602', ['C0', 'C2_rcv', 'C2_srcS1']),
    ('WL94A|Ar_s8601', 'rest_x94r', 'CD_Ar_s8601', ['C0', 'C2_rcv', 'C2_srcS1']),
]


def discover(root, pref, label, arms, have):
    d = os.path.join(HOME, root, 'results', 'tables')
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith('.npz') and f.startswith(pref) and not os.path.islink(os.path.join(d, f)):
            st = f[:-4]
            if (root, st) not in have:
                out.append((f'{label}|{st}', root, st, arms))
    return out


def main(outfn):
    jobs = list(JOBS)
    have = {(r, s) for _, r, s, _ in jobs}
    for root, label in (('rest_x94b', 'WL94B'), ('rest_x94c', 'WL94C'), ('rest_x94d', 'WL94D')):
        jobs += discover(root, 'CD_', label, ['C0', 'C2_rcv', 'C2_srcS1'], have)
        jobs += discover(root, 'FZ_', label, ['C0', 'C2_rcv', 'C2_srcS1'], have)
    res = {'weights': {'coded': W_CODED.tolist(), 'true': W_TRUE.tolist()}, 'jobs': {}}
    Fc = {}; Mc = {}
    for label, root, stem, arms in jobs:
        fn = os.path.join(HOME, root, 'results', 'tables', stem + '.npz')
        if not os.path.exists(fn):
            res['jobs'][label] = {'missing': fn}
            continue
        froot = os.path.realpath(os.path.join(HOME, root, 'results', 'fields'))
        if froot not in Fc:
            Fc[froot] = load_fields(os.path.join(HOME, root))
        mfn = os.path.realpath(os.path.join(HOME, root, 'results', 'pre', 'masks94.npz'))
        if mfn not in Mc:
            Mc[mfn] = dict(np.load(mfn))
        tab = dict(np.load(fn))
        o = {'table': fn, 'table_sha256': sha(fn), 'fields': froot, 'masks': mfn, 'masks_sha256': sha(mfn)}
        for wn, w in (('coded', W_CODED), ('true', W_TRUE)):
            tm = table_metrics(tab, Fc[froot], Mc[mfn], arms, w)
            o[wn] = tm
            o[wn + '_MR_ok_C0'] = mr_ok(tm['C0']) if 'C0' in tm else None
        res['jobs'][label] = o
        print(label, 'MR coded/true', o['coded_MR_ok_C0'], o['true_MR_ok_C0'], flush=True)
    json.dump(res, open(outfn, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
