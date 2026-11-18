"""WL104 references for one WL84 image (all on the fixed mesh scene, WL81 camera, local RTX 5080).

  python refs104.py <image> <out_dir>
Mitsuba 'path' through the WL84 GPU renderer (render84.render_g), two independent halves each:
  LAY65  layered physical BSDF (lay81), max_depth 65       final physical reference
  LAY7   layered, max_depth 7 (= 6 scattering events, ECRO's D)
  LAY4   layered, max_depth 4, halves of 16384 spp        tessellation gate vs the WL84 analytic depth-4 reference
  ER7    learned-ER closure (cmp83), max_depth 7          ECRO-material 6-event reference (A3 gate partner)
  ER65   learned-ER closure, max_depth 65                 Learned-ER path tracing at full depth (the WL84 method without truncation)
  ER65_64  learned-ER, max_depth 65, 8 independent 64-spp renders (operating point of WL84's ER_L arm)
ECRO tracer components (pt102 'cam' mode, D = 6, learned-ER, two halves of 4096 spp): e1, e2, h (physical HIGH = events 3..6).
"""
import os
import sys
import time

import numpy as np

import common104 as C
import rt104 as RT

image, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
tag, light, _, _ = C.image_spec(image)
ii = C.IMAGES.index(image)
desc, state, h = RT.register(tag)
assert h == C.STATE_SHA[tag], 'STOP_STATE_HASH'
SPP = {'LAY65': 32768, 'LAY7': 32768, 'LAY4': 16384, 'ER7': 32768, 'ER65': 32768}
if C.SMOKE:
    SPP = {k: max(64, v // 256) for k, v in SPP.items()}
ARMS = {'LAY65': ('lay', 65), 'LAY7': ('lay', 7), 'LAY4': ('lay', 4), 'ER7': ('er', 7), 'ER65': ('er', 65)}
timing = {}
for ai, (arm, (kind, depth)) in enumerate(ARMS.items()):
    for half in (0, 1):
        fn = os.path.join(out, f'{arm}_h{half}.npz')
        if os.path.exists(fn):
            continue
        sc = RT.scene(tag, kind, light=light, with_sensor=False)
        seed = 104600 + 1000 * ii + 10 * ai + half
        m, acc, acc2, T = RT.render_ref(sc, SPP[arm], seed, depth, batch=C.BATCH['ref'])
        n = SPP[arm]
        var = (acc2 / n - (acc / n) ** 2) * n / (n - 1)
        np.savez(fn + '.tmp.npz', mean=m.astype(np.float32), var=var.reshape(C.RES, C.RES, 3).astype(np.float32), seed=seed, spp=n, depth=depth)
        os.replace(fn + '.tmp.npz', fn)
        timing[f'{arm}_h{half}'] = T
        print('REF', image, arm, half, round(T['wall_s'], 1), flush=True)
for q in range(8):
    fn = os.path.join(out, f'ER65_64_r{q}.npz')
    if os.path.exists(fn):
        continue
    sc = RT.scene(tag, 'er', light=light, with_sensor=False)
    m, acc, acc2, T = RT.render_ref(sc, 64, 104900 + 100 * ii + q, 65, batch=64)
    np.savez(fn + '.tmp.npz', mean=m.astype(np.float32)); os.replace(fn + '.tmp.npz', fn)
    timing[f'ER65_64_r{q}'] = T
ctx = RT.Ctx(RT.scene(tag, 'er'))
Lp = RT.point_lights(light)
for half in (0, 1):
    fn = os.path.join(out, f'PTC_h{half}.npz')
    if os.path.exists(fn):
        continue
    t0 = time.perf_counter()
    acc, T = ctx.render(Lp, 64 if C.SMOKE else 4096, 104800 + 10 * ii + half, 'cam', b=C.BATCH['tracer'])
    np.savez(fn + '.tmp.npz', **RT.P.pack(acc, C.RES, C.RES, 64 if C.SMOKE else 4096)); os.replace(fn + '.tmp.npz', fn)
    timing[f'PTC_h{half}'] = {**T, 'wall_total_s': time.perf_counter() - t0}
    print('REF', image, 'PTC', half, flush=True)
prev = os.path.join(out, 'refs_timing.json')
old = __import__('json').load(open(prev)) if os.path.exists(prev) else {}
old.update(timing)
C.write_json(prev, old)
print('REFS_COMPLETE', image, flush=True)
os._exit(0)
