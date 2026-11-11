"""WL88 dev look renders (dev seeds 88990+, 256^2, low spp; reference estimator only, no candidate output). python look88.py <root> <W> <spp>"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import fx88  # noqa: E402
PT = fx88.PT
root, w, spp = sys.argv[1], sys.argv[2], int(sys.argv[3])
fx88.PT.dr.set_flag(fx88.PT.dr.JitFlag.KernelHistory, True)
out = {}
for t in ('T0', 'T1'):
    fx88.setup(f'{w}_{t}')
    geom, ls = PT.config(f'{w}_{t}')
    ctx = PT.Ctx(geom)
    acc, T_ = ctx.render(PT.Lights(PT.FX['light_sets'][ls]), 'ER', PT.FX['camera'], 256, spp, 88990, 'cam')
    o = PT.pack(acc, 256, spp)
    for k, v in o.items():
        out[f'{t}_{k}'] = v
os.makedirs(os.path.join(root, 'screen'), exist_ok=True)
np.savez_compressed(os.path.join(root, 'screen', f'look_{w}.npz'), **out)
print('look', w, flush=True)
os._exit(0)
