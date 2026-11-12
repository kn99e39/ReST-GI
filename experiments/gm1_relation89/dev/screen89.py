"""WL89 dev screening of candidate workloads (SCREENING PARTITION: dev seeds 89900+, never reused for qualification or evaluation).
Geometry-only masks + low-budget family-tagged REF-ER at T0 / T1 -> projected qualification statistics at the final reference budget.
No relation / method output is computed.   python screen89.py <root> <nrep> <spp> <name>...
"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import fx89  # noqa: E402
import masks88  # noqa: E402
import pt88  # noqa: E402
PT, S, FX = fx89.PT, fx89.S, fx89.FX
root, nrep, spp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
FINAL = 8 * 8192
os.makedirs(os.path.join(root, 'screen'), exist_ok=True)
for w in sys.argv[4:]:
    ex = FX[w]
    gok = bool(S.geom_ok(ex['T0']) and S.geom_ok(ex['T1']) and all(S.light_ok(l['pos'], ex[t]) for ls in (ex['light_set'], 'LMIX89', 'LSPOT89')
                                                                     for l in FX['light_sets'][ls] for t in ('T0', 'T1')))
    m, aux = masks88.compute(w)
    fx89.fx88.setup(w + '_T0')
    imgs = {}
    for t in ('T0', 'T1'):
        ctx = PT.Ctx(ex[t]); lt = PT.Lights(FX['light_sets'][ex['light_set']])
        imgs[t] = [pt88.render_fam(ctx, lt, FX['camera'], FX['camera']['res'], spp, 89900 + r)[0] for r in range(nrep)]
    Y = lambda o, k: o[k].mean(-1)
    sc = math.sqrt(nrep * spp / FINAL)            # floor at the final budget = floor here x sc
    t0 = np.mean([Y(o, 'tot') for o in imgs['T0']], 0)
    row = {'geometry_ok': gok}
    for k in ('A', 'Bset', 'C_ss', 'R_mirror', 'D'):
        mk = m[k]
        if mk.sum() < 30:
            row[k] = {'pixels': int(mk.sum())}
            continue
        b = float(t0[mk].mean())
        r = {'pixels': int(mk.sum())}
        for key in ('tot', 'h', 'H2'):
            d = [Y(a, key) - Y(c, key) for a, c in zip(imgs['T1'], imgs['T0'])]
            x = np.mean(d, 0)
            fl = 0.5 * np.abs(np.mean(d[:nrep // 2], 0) - np.mean(d[nrep // 2:], 0))[mk].mean() / b if nrep >= 2 else float('nan')
            r[f'abs_{key}'] = float(np.abs(x[mk]).mean() / b); r[f'snr_{key}_final'] = r[f'abs_{key}'] / max(fl * sc, 1e-12)
        dG = np.mean([Y(a, 'H1G') + Y(a, 'H2G') - Y(c, 'H1G') - Y(c, 'H2G') for a, c in zip(imgs['T1'], imgs['T0'])], 0)
        dh = np.mean([Y(a, 'h') - Y(c, 'h') for a, c in zip(imgs['T1'], imgs['T0'])], 0)
        dH2 = np.mean([Y(a, 'H2') - Y(c, 'H2') for a, c in zip(imgs['T1'], imgs['T0'])], 0)
        r['H2_share'] = float(np.abs(dH2[mk]).sum() / max(np.abs(dh[mk]).sum(), 1e-30)); r['G_share'] = float(np.abs(dG[mk]).sum() / max(np.abs(dh[mk]).sum(), 1e-30))
        row[k] = r
    print('SCREEN', w, json.dumps(row), flush=True)
    json.dump(row, open(os.path.join(root, 'screen', f'{w}.json'), 'w'), indent=1)
os._exit(0)
