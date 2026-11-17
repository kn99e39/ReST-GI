"""WL101 DEVELOPMENT timing of arm B on the selection case only (SSS test0 <-> test1). Non-evidence.
  FREEZE=1 python dev_time_b101.py <tag> [spp]
"""
import json, os, sys, time
import numpy as np
import rt101 as R

tag = sys.argv[1]; spp = int(sys.argv[2]) if len(sys.argv) > 2 else 1
case = 'sss'
rows = json.loads((R.ROOT100 / 'configs.json').read_text())[case]['test']
model = R.load_model(case)
cam = dict(R.C100.P['datasets'][case]['camera'], res=512)
rt = R.RT(rows[0]['geom'], case, model, rows[0]['lights'][0])
R.dr.set_flag(R.dr.JitFlag.KernelHistory, True)
torch = R.torch
ev = []


class T:
    def __init__(self):
        self.t = {}; self.g = {}

    def __call__(self, name, fn):
        R.sync(); R.dr.kernel_history()
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter(); a.record()
        r = fn()
        b.record(); R.sync()
        self.t[name] = (time.perf_counter() - t0) * 1e3
        self.g[name] = sum(float(h.get('execution_time', 0.0)) for h in R.dr.kernel_history())
        return r


for i, j in enumerate([1, 0] * 10):
    tm = T(); t0 = time.perf_counter()
    moved, g_old = tm('geometry_update', lambda: rt.set_geom(rows[j]['geom']))
    nd = tm('relation_update', lambda: rt.update_state(g_old, moved))
    tm('lights', lambda: rt.set_light(rows[j]['lights'][0]))
    s = tm('source', lambda: rt.source())
    B, _ = tm('generator_apply', lambda: rt.respond(s))
    Bd = tm('table_handoff', lambda: R.t2d(B.view(-1)))
    f = tm('frame', lambda: rt.frame(Bd, cam, 512, spp, 9000 + i))
    R.sync()
    ev.append({'to': j, 'total_ms': (time.perf_counter() - t0) * 1e3, 'parts_ms': tm.t, 'drjit_gpu_ms': tm.g, 'dirty': nd})
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in ev[-1].items() if k != 'parts_ms'}),
          {k: round(v, 2) for k, v in tm.t.items()}, flush=True)
tot = [e['total_ms'] for e in ev[4:]]
print('median total', np.median(tot), {k: round(float(np.median([e['parts_ms'][k] for e in ev[4:]])), 3) for k in ev[0]['parts_ms']})
p = R.REPO / 'results' / 'ecro_dyn101' / 'dev'
json.dump({'freeze': R.FREEZE, 'spp': spp, 'events': ev}, open(p / f'{tag}.json', 'w'), indent=1)
os._exit(0)
