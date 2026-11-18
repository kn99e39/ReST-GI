"""WL104A development probe (before the protocol freeze; computes NO HIGH comparison between arms).
Baseline preservation, paired-LOW identity, layered-tracer PASS_SEED handling vs Mitsuba LAY7 (WL104), timings, ROI pixel counts.
  <venv-cu128 python> dev104a.py <out_json>
"""
import json
import sys
import time

import numpy as np

import h104a as H

C, RT = H.C, H.RT
rep = {}
cfg = {'sphere_lower_nz_max': 0.0, 'floor_near': {'sphere_axis_m': 0.60, 'cube_footprint_m': 0.25}}
for image in ['test83_hard_1_L1']:
    tag, light, _, _ = C.image_spec(image)
    ii = C.IMAGES.index(image)
    ctx = H.ctx_for(tag, 'er')
    B, s, _, _ = H.learned_table(ctx, image)
    Bw = np.load(H.E104 / 'eval' / image / 'table.npz')['B']
    rep['table_bitwise_vs_wl104'] = bool(np.array_equal(B, Bw))
    r4 = H.render(ctx, image, 4, 104500 + 1000 * ii, 'ecro', B=B, b=4)
    w4 = np.load(H.E104 / 'eval' / image / 'ecro_spp4.npy')[0]
    rep['ecro4_replay_max_abs'] = float(max(np.abs(r4[k].astype(np.float32) - w4[j]).max() for j, k in enumerate(('e1', 'e2', 'hl'))))
    a = H.render(ctx, image, 64, 1099001, 'ecro', B=B)
    c = H.render(ctx, image, 64, 1099001, 'cam')
    rep['paired_low_max_abs'] = {k: float(np.abs(a[k] - c[k]).max()) for k in ('e1', 'e2')}
    rep['paired_low_rel_sum'] = float(abs((a['e1'] + a['e2']).sum() - (c['e1'] + c['e2']).sum()) / (c['e1'] + c['e2']).sum())
    for k, (mode, kw) in {'ecro': ('ecro', {'B': B}), 'cam_er': ('cam', {})}.items():
        t0 = time.perf_counter(); H.render(ctx, image, 4096, 1099100, mode, **kw); rep[f't4096_{k}_s'] = time.perf_counter() - t0
    t0 = time.perf_counter(); F, cnt = H.field_table(ctx, image, 1024, 1099200); rep['t_field1024_s'] = time.perf_counter() - t0
    rep['field_empty_slot_frac'] = float((cnt == 0).mean())
    ctxl = H.ctx_for(tag, 'lay')
    t0 = time.perf_counter(); lp = H.render(ctxl, image, 4096, 1099300, 'cam', pass_seed=True); rep['t4096_cam_lay_s'] = time.perf_counter() - t0
    ln = H.render(ctxl, image, 4096, 1099300, 'cam', pass_seed=False)
    m = H.masks(ctx, cfg)
    rep['roi_counts'] = {k: int(v.sum()) for k, v in m.items()}
    o = m['objects']
    lay7 = 0.5 * sum(np.load(H.E104 / 'refs' / image / f'LAY7_h{h}.npz')['mean'].astype(np.float64) for h in (0, 1))
    tot = lambda d: d['e1'] + d['e2'] + d['h']
    rep['lay_tracer_passseed_obj_mean_ratio_vs_LAY7'] = float(tot(lp)[o].mean() / lay7[o].mean())
    rep['lay_tracer_fixedseed_obj_mean_ratio_vs_LAY7'] = float(tot(ln)[o].mean() / lay7[o].mean())
    rep['lay_tracer_passseed_full_mean_ratio_vs_LAY7'] = float(tot(lp).mean() / lay7.mean())
print(json.dumps(rep, indent=1))
C.write_json(sys.argv[1], rep)
import os
os._exit(0)
