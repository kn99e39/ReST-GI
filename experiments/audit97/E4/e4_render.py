"""WL97 audit, track E4: table-lookup renders of audit tables with NEW camera seeds, by a direct call of pt86.Ctx.render (mode "ecro").
The mat94 per-patch material patch is NOT installed (plain pt86, shape materials = the "none" variant).

Usage: python e4_render.py <CODE/experiments> <tables.npz> <outdir> <spp> <seed0> <nrep> <key1,key2,...>
Output: <outdir>/A_<key>_s<seed>.npz (hl [res, res, 3] float32).  Existing files are skipped.
"""
import json, os, sys, time
import numpy as np

CODE, TAB, OUTD, SPP, SEED0, NREP, KEYS = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7].split(',')
sys.path.insert(0, os.path.join(CODE, 'ecro86'))
import pt86 as PT  # noqa: E402

SC = json.load(open(os.path.join(CODE, 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')))
os.makedirs(OUTD, exist_ok=True)
ctx = PT.Ctx(SC['geom'])
lt = PT.Lights(SC['lights'])
cam = SC['camera']
T = np.load(TAB)
for key in KEYS:
    B = PT.fl(T[key].astype(np.float32))
    for r in range(NREP):
        seed = SEED0 + r
        fn = os.path.join(OUTD, f'A_{key}_s{seed}.npz')
        if os.path.exists(fn):
            continue
        t0 = time.time()
        acc, T_ = ctx.render(lt, 'ER', cam, cam['res'], SPP, seed, 'ecro', B=B, b=32)
        np.savez_compressed(fn, hl=PT.pack(acc, cam['res'], SPP)['hl'])
        print('RENDER', key, seed, round(time.time() - t0, 1), flush=True)
print('E4RENDER DONE', flush=True)
os._exit(0)
