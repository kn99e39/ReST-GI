"""WL97 audit, track E4: event-count experiment with the INSTALLED production field tracer (mat94-patched pt86a.field_src), read-only.
Runs field_src(split=True) on CK2 "none" with the CKF_none half-a seed (94301) and kmax = 2, 3, 5 and compares with the stored CKF_none.

Usage: python e4_kmax.py <CODE/experiments> <R (own dir)> <CKF_none.npz> <out.json> [spp=4096]
"""
import json, os, sys, time
import numpy as np

CODE, R, CKF, OUT = sys.argv[1:5]
SPP = int(sys.argv[5]) if len(sys.argv) > 5 else 4096
sys.path.insert(0, os.path.join(CODE, 'ecro_mat94'))
import mat94 as MT  # noqa: E402

FAMS = ('e2_p', 'e2_o', 'e3_p', 'e3_o')
PT, S, A, FX, ctx, lt = MT.ctx_ck2(R)
MT.set_pmat(PT, ctx, MT.variant_pmat(ctx.G, 'none'))
z = np.load(CKF)
ref = {k: z[f'{k}_a'].astype(np.float64) for k in FAMS}
res = {'spp': SPP, 'seed': 94301, 'stored_spp': int(z['spp'])}
for kmax in (2, 3, 5):
    t0 = time.time()
    t = A.field_src(ctx, A.Src(lt, 'true', ()), SPP, 94301, kmax=kmax, b=128, split=True)
    r = {}
    for k in FAMS:
        x = t[k].astype(np.float64)
        r[k] = {'sum': float(x.sum()), 'stored_sum': float(ref[k].sum()), 'max_abs': float(np.abs(x).max()),
                'rel_l1_vs_stored': float(np.abs(x - ref[k]).sum() / max(ref[k].sum(), 1e-30)),
                'max_abs_diff_vs_stored': float(np.abs(x - ref[k]).max()), 'min': float(x.min())}
    r['seconds'] = time.time() - t0
    res[f'kmax{kmax}'] = r
    if kmax == 3:
        np.savez_compressed(OUT.replace('.json', '_kmax3.npz'), **t)
    print('kmax', kmax, {k: (round(r[k]['sum'], 3), round(r[k]['rel_l1_vs_stored'], 6)) for k in FAMS}, round(r['seconds'], 1), flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
print('E4KMAX DONE', flush=True)
os._exit(0)
