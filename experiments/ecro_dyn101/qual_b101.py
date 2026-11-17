"""WL101 quality, optimized stack: arm-B predictions and the STALE control at the first-pass tour states.
  python qual_b101.py <V_SSS_TOUR|V_NEAR_TOUR> <out_dir>
B_sel: optimized lifecycle reached through the tour; B_stale: same LOW, but the table of the previous state (relation state,
factors and table not updated). 4096 spp, b = 32 launches, seed 101500 + 10 x state (identical to arm A).
"""
import hashlib, json, os, sys
import numpy as np

os.environ['FREEZE'] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seq101 as Q  # noqa: E402
import rt101 as R  # noqa: E402

wl, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
case, states = Q.tour_states(wl)
_, start, ev = Q.workload(wl)
model = R.load_model(case)
ck = hashlib.sha256(open(R.ROOT100 / case / 'model.pt', 'rb').read()).hexdigest()
cam = dict(R.C100.P['datasets'][case]['camera'], res=512)
rt = R.RT(start[0], case, model, start[1])
SPP = 4096
keys = ['e10', 'e11', 'e12', 'e20', 'e21', 'e22', 'hl0', 'hl1', 'hl2']
prevB = None


def pack(acc):
    o = {}
    for name in ('e1', 'e2', 'hl'):
        o[name] = np.stack([(acc[keys.index(f'{name}{c}')] / SPP).reshape(512, 512).astype(np.float32) for c in range(3)], -1)
    return o


for i, (g, L, sid) in enumerate(ev[: Q.WARM + 8]):
    moved, g_old = rt.set_geom(g)
    nd = rt.update_state(g_old, moved)
    rt.set_light(L)
    s = rt.source()
    B, _ = rt.respond(s)
    Bc = B.clone()
    if i >= Q.WARM:
        Bd = R.t2d(Bc.view(-1))
        o = pack(R.render_hq(rt, Bd, cam, 512, SPP, 101500 + 10 * sid))
        np.savez_compressed(os.path.join(out, f'PRED_B_{sid}.npz'), B=Bc.cpu().numpy(), **o)
        Bs = R.t2d(prevB.view(-1))
        o = pack(R.render_hq(rt, Bs, cam, 512, SPP, 101500 + 10 * sid))
        np.savez_compressed(os.path.join(out, f'PRED_STALE_{sid}.npz'), B=prevB.cpu().numpy(), **o)
        R.C100.jd({'state': sid, 'event': i, 'dirty_rays': nd, 'checkpoint_sha256': ck, 'seed': 101500 + 10 * sid, 'spp': SPP,
                   'stale_table_from_state': ev[i - 1][2], 'versions': {'mitsuba': R.mi.__version__, 'drjit': R.dr.__version__}},
                  os.path.join(out, f'PRED_B_{sid}.json'))
        print('PRED_B', wl, sid, nd, flush=True)
    prevB = Bc
os._exit(0)
