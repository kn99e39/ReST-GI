"""WL97 audit, track E4: independent camera path-traced references split by event count (plain pt86 'cam' mode, new seeds).
For each seed two renders with IDENTICAL random streams: kmax = 3 (h = NEE at camera vertex 3 only = exactly 3 events) and kmax = 6
(h = 3..6 events).  h6 - h3 is therefore exactly the 4..6-event camera term = the table tail (3..5 events leaving vertex 2) as read.

Usage: python e4_ref.py <CODE/experiments> <outdir> <spp> <seed0> <nrep>
"""
import json, os, sys, time
import numpy as np

CODE, OUTD, SPP, SEED0, NREP = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
sys.path.insert(0, os.path.join(CODE, 'ecro86'))
import pt86 as PT  # noqa: E402

SC = json.load(open(os.path.join(CODE, 'ecro_local93d', 'addendum_checker', 'scene_ck2.json')))
os.makedirs(OUTD, exist_ok=True)
ctx = PT.Ctx(SC['geom'])
lt = PT.Lights(SC['lights'])
cam = SC['camera']
for r in range(NREP):
    seed = SEED0 + r
    for kmax in (3, 6):
        fn = os.path.join(OUTD, f'REFA_k{kmax}_s{seed}.npz')
        if os.path.exists(fn):
            continue
        t0 = time.time()
        acc, T_ = ctx.render(lt, 'ER', cam, cam['res'], SPP, seed, 'cam', b=32, kmax=kmax)
        p = PT.pack(acc, cam['res'], SPP)
        np.savez_compressed(fn, e1=p['e1'], e2=p['e2'], h=p['h'])
        print('REFA', kmax, seed, round(time.time() - t0, 1), flush=True)
print('E4REF DONE', flush=True)
os._exit(0)
