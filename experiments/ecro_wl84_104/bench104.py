"""WL104 execution benchmark (not evidence): launch batch size vs wall / GPU time for the reference renderer and the ECRO tracer,
repeated calls to detect slow-down. Prints JSON lines.
  <venv-cu128 python> bench104.py
"""
import json
import time

import common104 as C
import rt104 as RT

tag = 'test83_iid_1'
RT.register(tag)
for kind, depth in (('lay', 65), ('er', 65)):
    for b in (64, 256, 1024):
        for rep in range(2):
            sc = RT.scene(tag, kind, light=2, with_sensor=False)
            t0 = time.perf_counter()
            m, acc, acc2, T = RT.render_ref(sc, 4096, 777 + rep, depth, batch=b)
            print(json.dumps({'arm': f'{kind}{depth}', 'batch': b, 'rep': rep, 'wall_s': round(time.perf_counter() - t0, 2),
                              'gpu_s': round(T['gpu_ms'] / 1e3, 2), 'trace_s': round(T['trace_s'], 2), 'eval_s': round(T['eval_s'], 2),
                              'launches': T['launches']}), flush=True)
ctx = RT.Ctx(RT.scene(tag, 'er'))
L = RT.point_lights(2)
for b in (8, 64, 256):
    t0 = time.perf_counter(); acc, T = ctx.render(L, 1024, 778, 'cam', b=b)
    print(json.dumps({'arm': 'PTC_cam', 'batch': b, 'wall_s': round(time.perf_counter() - t0, 2)}), flush=True)
import os
os._exit(0)
