"""WL97 audit, Track A, check A2b: (i) block comparison mine vs stored M64 by (ring_in, ring_out); (ii) convergence of my quadrature for grazing
incident directions (n x n strata per outgoing bin). numpy only."""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
import er_np as E
OUT = os.path.join(ROOT, 'results', 'audit97', 'A')
ST = np.load(os.path.join(ROOT, 'experiments', 'ecro86', 'data', 'mats86.npz'))['states'].astype(np.float64)
mine = np.load(os.path.join(OUT, 'a2_m64_mine.npz'))['M']
A4 = np.load(os.path.join(ROOT, 'results', 'ecro_attr95a', 'ops95a.npz'))['A4'].astype(np.float64)
iz = np.arange(64) // 16
res = {'block_sum_ratio_mine_over_stored_green': [[float(mine[iz == a][:, iz == b, 1].sum() / A4[iz == a][:, iz == b, 1].sum()) for b in range(4)] for a in range(4)],
       'block_sum_stored_green': [[float(A4[iz == a][:, iz == b, 1].sum()) for b in range(4)] for a in range(4)],
       'row_sum_ratio_ring0_rows_green': [float(mine[i, :, 1].sum() / A4[i, :, 1].sum()) for i in range(16)]}
conv = {}
for ci in (0.015625, 0.046875, 0.109375, 0.234375, 0.4):
    w = np.array([math.sqrt(1 - ci * ci) * math.cos(0.3), math.sqrt(1 - ci * ci) * math.sin(0.3), ci])
    o = {}
    for n in (24, 48, 96, 192):
        og = E.bin_grid(n)
        f = E.fr(ST[1], w[None, None, :], og)                 # [NB, n*n, 3]
        o[n] = {'albedo_true': float((f[..., 1] * og[..., 2]).mean(1).sum() * E.DOMEGA), 'table_flux': float((f[..., 1].mean(1) * E.WT).sum()),
                'ring0_ring0_sum': float(f[iz == 0][..., 1].mean(1).sum())}
    conv[str(ci)] = o
res['convergence_single_direction_green'] = conv
json.dump(res, open(os.path.join(OUT, 'a2b_convergence.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
