"""WL97 audit track E3: independent float64 ER port (er97.ER) vs production DrClosure83 values stored by val97_dr.py.

  python cmp97.py <OUT>          reads OUT/val_pairs.npz, writes OUT/cmp_er.json
"""
import json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import er97  # noqa: E402

OUT = sys.argv[1]
z = np.load(os.path.join(OUT, 'val_pairs.npz'))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
res = {}
for nm in ('uniform', 'graze_mirror', 'graze_any'):
    wi = torch.tensor(z[f'{nm}_wi'], dtype=torch.float64, device=dev); wo = torch.tensor(z[f'{nm}_wo'], dtype=torch.float64, device=dev)
    for m in (0, 1, 2, 4):
        er = er97.ER(z['states'][m], dev)
        f = torch.stack(er.eval((wi[:, 0], wi[:, 1], wi[:, 2]), (wo[:, 0], wo[:, 1], wo[:, 2])), 1).cpu().numpy()
        p = z[f'{nm}_f{m}']
        ok = (z[f'{nm}_wi'][:, 2] > 1e-3) & (z[f'{nm}_wo'][:, 2] > 1e-3)
        d = np.abs(f - p)[ok] / np.maximum(np.abs(p[ok]), 1e-12)
        res[f'{nm}|mat{m}'] = {'n': int(ok.sum()), 'median_rel': float(np.median(d)), 'p999_rel': float(np.quantile(d, 0.999)), 'max_rel': float(d.max()),
                               'sum_ratio_port_over_production': [float(f[ok, c].sum() / p[ok, c].sum()) for c in range(3)],
                               'max_abs_over_max_value': float(np.abs(f - p)[ok].max() / p[ok].max()), 'production_max': float(p[ok].max())}
res['pass'] = bool(all(v['p999_rel'] < 1e-3 and abs(v['sum_ratio_port_over_production'][0] - 1) < 1e-5 for k, v in res.items()))
json.dump(res, open(os.path.join(OUT, 'cmp_er.json'), 'w'), indent=1)
for k, v in res.items():
    print(k, v)
