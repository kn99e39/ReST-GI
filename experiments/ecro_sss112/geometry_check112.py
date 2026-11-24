"""WL112 geometry-only pre-freeze check: validity of fixtures / sequence states and contact-valid views from centre-ray gbuffers.
No radiance is computed."""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common112 as C  # noqa: E402
import tracer112 as T  # noqa: E402

out = {}
cases = [('F0', C.fixture('F0')[0], C.fixture('F0')[1], C.views('F0')), ('F1', C.fixture('F1')[0], C.fixture('F1')[1], C.views('F1'))]
for sname, seq in C.sequences().items():
    cams = C.views(seq['fixture'])
    for i, g in enumerate(seq['states']):
        cases.append((f'{sname}_{i}', g, seq['light'], {v: cams[v] for v in seq['views']}))
for name, g, light, cams in cases:
    row = {'valid': C.geometry_valid(g), 'light_valid': C.light_valid(light[0]['pos'], g), 'views': {}}
    ctx = T.Ctx112(g, 'C')
    for v, cam in cams.items():
        gb = ctx.gbuffer(dict(cam, res=C.RES), C.RES)
        reg, bands = C.masks(gb, g, light[0]['pos'])
        row['views'][v] = {'contact_valid': C.contact_valid(reg) if C.P else bool(reg['near'].sum() >= 200 and reg['flank'].sum() >= 1000),
                           **{k: int(m.sum()) for k, m in reg.items() if k != 'full'}}
    out[name] = row
    print(name, row['valid'], row['light_valid'], {v: (r['contact_valid'], r['near'], r['flank'], r['sphere_boundary']) for v, r in row['views'].items()}, flush=True)
C.jd(out, C.REPO / 'results/ecro_sss112/dev/geometry_check112.json')
import os
os._exit(0)
