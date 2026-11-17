"""WL101 quality, frozen stack: physical references and arm-A predictions at the first-pass tour states.
  python qual_a101.py ref  <V_SSS_TOUR|V_NEAR_TOUR> <out_dir>
  python qual_a101.py pred <V_SSS_TOUR|V_NEAR_TOUR> <out_dir>
References: pt100 'cam' (6 events), 512^2, halves of 4096 spp (seeds 101100 / 101101 + 10 x state); state 0 reuses WL100 REF_0 halves
(identical geometry and light). Predictions: arm A lifecycle reached through the tour (warm + first pass), 4096 spp, b = 32,
seed 101500 + 10 x state. Every output carries its configuration and hashes.
"""
import json, os, shutil, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seq101 as Q  # noqa: E402
import common101 as K  # noqa: E402

mode, wl, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
case, states = Q.tour_states(wl)
cam = K.cam_of(case, 512)
SPP = 4096
if mode == 'ref':
    ctx = None
    for g, L, sid, prev in states:
        for h, sd in enumerate((101100, 101101)):
            fn = os.path.join(out, f'REF_{sid}_{h}.npz')
            if os.path.exists(fn):
                continue
            if sid == 0:
                src = K.ROOT100 / case / 'renders' / f'REF_0_{h}.npz'
                meta = json.loads((K.ROOT100 / case / 'renders' / f'REF_0_{h}.json').read_text())
                assert meta['config']['geom'] == g and meta['config']['lights'][0] == L and meta['spp'] == SPP
                shutil.copyfile(src, fn)
                K.C100.jd({'reused_from': str(src), 'sha256': K.sha(src), 'wl100_meta': meta}, fn.replace('.npz', '.json'))
                continue
            if ctx is None:
                ctx = K.pt100.Ctx(g, case)
            else:
                ctx.set_geom(g)
            t0 = time.perf_counter()
            acc, tim = ctx.render(K.PT.Lights(L), 'ER', cam, 512, SPP, sd + 10 * sid, 'cam', b=32)
            np.savez_compressed(fn, **K.PT.pack(acc, 512, SPP))
            K.C100.jd({'geom': g, 'light': L, 'seed': sd + 10 * sid, 'spp': SPP, 'timing': tim, 'frozen_sources': K.frozen_hashes()},
                      fn.replace('.npz', '.json'))
            print('REF', wl, sid, h, round(time.perf_counter() - t0, 1), 's', flush=True)
        gfn = os.path.join(out, f'MAP_{sid}.npz')
        if not os.path.exists(gfn):
            c2 = K.pt100.Ctx(g, case)
            np.savez_compressed(gfn, **c2.gbuffer(cam, 512))
elif mode == 'pred':
    model, ck = K.load_model(case)
    _, start, ev = Q.workload(wl)
    A = K.FrozenECRO(case, start[0], model)
    first = {s[2] for s in states}
    for i, (g, L, sid) in enumerate(ev[: Q.WARM + 8]):
        d = A.edit(g, L, K.Timer())
        if i < Q.WARM:
            continue
        acc, tim = A.ctx.render(d['lt'], 'ER', cam, 512, SPP, 101500 + 10 * sid, 'ecro', B=d['Bd'], b=32)
        o = K.PT.pack(acc, 512, SPP)
        np.savez_compressed(os.path.join(out, f'PRED_A_{sid}.npz'), B=d['B'], **o)
        K.C100.jd({'arm': 'A_sel', 'state': sid, 'event': i, 'checkpoint_sha256': ck, 'seed': 101500 + 10 * sid, 'spp': SPP, 'dirty_rays': d['dirty_rays'],
                   'versions': {'mitsuba': K.mi.__version__, 'drjit': K.dr.__version__}}, os.path.join(out, f'PRED_A_{sid}.json'))
        print('PRED_A', wl, sid, flush=True)
K.os._exit(0)
