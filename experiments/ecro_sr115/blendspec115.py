"""WL115 (user request): export the exact scene of a WL115 view as a JSON spec for Blender (blend115_build.py).
python blendspec115.py STATE_KEY OUT.json   STATE_KEY e.g. SEQ_A_lift_3 (gap 50 mm) or F0"""
import json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'ecro_fd114'))
import common115 as C
import eval114 as EV
sys.path.insert(0, str(HERE.parent / 'ecro86'))
import scene86 as S
key, out = sys.argv[1], sys.argv[2]
g, l, vl, _ = EV.states()[key]
G = S.Geom(g)
rects = []
for r in S.stationary_rects():
    c, a1, a2, h1, h2, nu, nv, mat = r
    rects.append({'centre': list(map(float, c)), 'a1': list(map(float, a1)), 'a2': list(map(float, a2)), 'h': [float(h1), float(h2)], 'mat': int(mat)})
boxes = []
for b, mat in (('A', 4), ('B', 5)):
    M = G.box_matrix(b)
    boxes.append({'name': f'box{b}', 'matrix': M.tolist(), 'mat': mat})
mats = np.load(HERE.parent / 'ecro86' / 'data' / 'mats86.npz')
spec = {'state': key, 'geometry': g, 'sphere': {'centre': [float(v) for v in g['S']['c']], 'radius': float(S.FX['sphere']['radius'])},
        'sss': {'sigma_t': 12.0, 'albedo': [0.92, 0.80, 0.65]}, 'rects': rects, 'boxes': boxes,
        'lambert_albedo': {str(i): float(mats['albedo'][i]) for i in range(6) if mats['lambert'][i]},
        'layered_desc': {str(i): mats['desc'][i].tolist() for i in range(6)},
        'lights': [{'pos': [float(v) for v in x['pos']], 'rgb': [float(v) for v in x['rgb']]} for x in l],
        'cameras': {vn: dict(cam, res=512) for vn, cam, seeds in vl}}
json.dump(spec, open(out, 'w'), indent=1)
print(json.dumps({k: spec[k] for k in ('state', 'sphere', 'lights', 'cameras')}, indent=1))
