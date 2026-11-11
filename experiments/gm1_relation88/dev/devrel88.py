"""WL88 dev smoke of rel88 on the dev geometry Wk (no evaluation configuration). python devrel88.py <root>"""
import json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..'))
import fx88, rel88 as R, pt88  # noqa
PT, Q, X = fx88.PT, fx88.Q, R.X
dr = PT.dr
E = fx88.fx87.P87['explicit']; RP = fx88.P88['relation']
root = sys.argv[1]; cfg = sys.argv[2] if len(sys.argv) > 2 else 'Wk_T1'
fx88.setup(cfg)
geom, ls = PT.config(cfg)
ctx = PT.Ctx(geom); lt = PT.Lights(PT.FX['light_sets'][ls])
print('peaked', R.peaked(), flush=True)
t0 = time.perf_counter(); quad = Q.Quad(ctx.u_state_np, E['M_ray'], E['seed_ray']); print('quad', time.perf_counter() - t0, flush=True)
t0 = time.perf_counter(); dense = R.Dense(RP['dense_sub'], 1024, RP['seed_dense']); print('dense16', time.perf_counter() - t0, flush=True)
B0, o0, q, st, T0 = X.build(ctx, lt, quad, E['P'], E['Mq'], E['seed_q'])
r0 = R.SAM(ctx, quad, dense, 8, 0, RP['seed_area'], st=st)
Bz = R.recurrence([r0], q)[0].cpu().numpy()
print('m=0 vs R0 max rel', float(np.abs(Bz - B0).max() / B0.max()), flush=True)
for kap, m in [(8, 4), (4, 4), (8, 16)]:
    T_ = {}
    r1 = R.SAM(ctx, quad, dense, kap, m, RP['seed_area'], st=st, timing=T_)
    R.sync(); t0 = time.perf_counter(); B1 = R.recurrence([r1], q)[0].cpu().numpy(); R.sync(); ta = time.perf_counter() - t0
    print(f'R1 k{kap} m{m}', 'pairs', len(r1.pi), 'valid', r1.n_area_valid, 'missing', r1.missing, 'rays', r1.rays(), 'MB', r1.nbytes() / 1e6,
          json.dumps({k: round(v, 4) for k, v in T_.items()}), 'apply', round(ta, 4), flush=True)
    np.save(os.path.join(root, f'dev_B1_k{kap}m{m}.npy'), B1)
np.save(os.path.join(root, 'dev_B0.npy'), B0)
f = pt88.field_orders(ctx, lt, 256, 88777)
np.save(os.path.join(root, 'dev_ord.npy'), f.astype(np.float32))
G = ctx.G
T = f[1:].sum(0)
w = (G.PAREA[:, None] * ((np.arange(64) // 16 + 0.5) / 4)[None])[..., None]
for name, Bx in [('R0', B0)] + [(f'R1 k{k}m{m}', np.load(os.path.join(root, f'dev_B1_k{k}m{m}.npy'))) for k, m in [(8, 4), (4, 4), (8, 16)]]:
    row = {}
    for c, sel in {'floor': G.PMAT == 0, 'panel': G.PMAT == 1, 'sphere': G.PMAT == 2, 'walls': G.PMAT == 3, 'G': G.PMAT == 4, 'boxB': G.PMAT == 5}.items():
        row[c] = (round(float((Bx[sel] * w[sel]).sum() / (T[sel] * w[sel]).sum()), 3), round(float(np.abs(Bx[sel] - T[sel]).sum() / T[sel].sum()), 3))
    print(name, 'flux / relL1 vs 256-spp truth', json.dumps(row), flush=True)
q1 = f[0]
print('q vs order-1 truth flux / relL1', float((q * w).sum() / (q1 * w).sum()), float(np.abs(q - q1).sum() / q1.sum()), flush=True)
os._exit(0)
