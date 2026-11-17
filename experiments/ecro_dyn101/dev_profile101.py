"""WL101 DEVELOPMENT profile (non-evidence): where does the frozen dynamic lifecycle spend time on this GPU?
Selection case only: SSS test0 <-> test1 (geometry + config light). Output: results/ecro_dyn101/dev/<tag>.json
  python dev_profile101.py <tag> [spp]
"""
import json, sys, time, cProfile, pstats, io
import numpy as np
import common101 as K

tag = sys.argv[1]; spp = int(sys.argv[2]) if len(sys.argv) > 2 else 1
case = 'sss'
rows = K.configs(case)['test']
model, ck = K.load_model(case)
A = K.FrozenECRO(case, rows[0]['geom'], model)
cam = K.cam_of(case, 512)
K.dr.set_flag(K.dr.JitFlag.KernelHistory, True)
seq = [1, 0] * 6
out = []
for i, j in enumerate(seq):
    T = K.Timer()
    t0 = time.perf_counter()
    d = A.edit(rows[j]['geom'], rows[j]['lights'][0], T)
    A.frame(d['lt'], d['Bd'], cam, 512, spp, 101000 + i, T)
    K.sync()
    out.append({'to': j, 'total_ms': (time.perf_counter() - t0) * 1e3, 'parts_ms': T.t, 'drjit_gpu_ms': T.g, 'dirty': d['dirty_rays']})
    print(json.dumps(out[-1]), flush=True)
# sub-step profile of one more edit
pr = cProfile.Profile()
T = K.Timer()
pr.enable()
d = A.edit(rows[1]['geom'], rows[1]['lights'][0], T)
A.frame(d['lt'], d['Bd'], cam, 512, spp, 101999, T)
K.sync()
pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(45)
p = K.ROOT100.parent / 'ecro_dyn101' / 'dev'; p.mkdir(parents=True, exist_ok=True)
json.dump({'events': out, 'cprofile': s.getvalue()}, open(p / f'{tag}.json', 'w'), indent=1)
print(s.getvalue()[:6000])
K.os._exit(0)
