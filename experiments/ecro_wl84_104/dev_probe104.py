"""WL104 development probe before the protocol freeze (not evidence; no ECRO model is trained or evaluated here).
  <venv-cu128 python> dev_probe104.py <out_dir>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import common104 as C

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rep = {'sphere_mesh': C.uv_sphere_ply(C.SPHERE_PLY), 'sphere_ply_sha256': C.sha(C.SPHERE_PLY)}
C.quad_ply(C.QUAD_PLY); rep['quad_ply_sha256'] = C.sha(C.QUAD_PLY)
import rt104 as RT  # noqa: E402

mi, dr, P, R81 = RT.mi, RT.dr, RT.P, RT.R81
rep['versions'] = {'mitsuba': mi.__version__, 'drjit': dr.__version__}
rep['state_hash_match'] = {}
for im in C.IMAGES:
    tag = im.rsplit('_L', 1)[0]
    _, st, h = RT.register(tag)
    rep['state_hash_match'][tag] = h == C.STATE_SHA[tag]
print(json.dumps(rep), flush=True)

tag = 'test83_iid_1'
t0 = time.perf_counter()
ctx = RT.Ctx(RT.scene(tag, 'er'))
rep['ctx_build_s'] = time.perf_counter() - t0
rep['layout'] = ctx.G.summary()
rep['N'] = int(ctx.N)
# camera vs WL81 formula
ray, pix, _ = ctx.gen_rays(1, None)
r81 = R81.camera_rays(0, False)
dd = np.stack([np.asarray(ray.d[k]) - np.asarray(r81.d[k]) for k in range(3)], 1)
rep['camera_max_abs_dir_diff'] = float(np.abs(dd).max())
# state rays timing
t0 = time.perf_counter(); hit, dbin, _ = ctx.state_rays(); rep['state_rays_s'] = time.perf_counter() - t0
rep['state_rays_hit_frac'] = float((hit >= 0).mean())
# one field item light timing (64 spp, as a train field)
L = RT.PT.Lights([{'pos': [0.3, -1.0, 1.6], 'rgb': [1.0] * 3}])
t0 = time.perf_counter(); s = ctx.source(L); rep['source_s'] = time.perf_counter() - t0
t0 = time.perf_counter(); f2, fh, cnt = ctx.field_components(L, 64, 104001); rep['field_64spp_s'] = time.perf_counter() - t0
rep['field_finite'] = bool(np.isfinite(f2).all() and np.isfinite(fh).all())
rep['field_count_min_nonzero_frac'] = float((cnt > 0).mean())

# A3 smoke: ECRO physical tracer (cam mode, D = 6, ER) vs Mitsuba path max_depth 7 (ER), light 2, object mask
for light in (2,):
    Lp = RT.point_lights(light)
    t0 = time.perf_counter(); acc, _ = ctx.render(Lp, 256, 104100, 'cam', b=8); t_pt = time.perf_counter() - t0
    pk = P.pack(acc, C.RES, C.RES, 256)
    img_pt = pk['e1'] + pk['e2'] + pk['h']
    sc = RT.scene(tag, 'er', light=light, with_sensor=False)
    t0 = time.perf_counter(); img_mi, _, _, Tm = RT.render_ref(sc, 256, 104200, 7); t_mi = time.perf_counter() - t0
    obj, _ = R81.masks(sc)
    rep[f'A3_smoke_L{light}'] = {'obj_mean_ratio': float(img_pt[obj].mean() / img_mi[obj].mean()), 'full_mean_ratio': float(img_pt.mean() / img_mi.mean()),
                                 'obj_rel_l1_256spp': R81.rel_l1(img_pt, img_mi, obj), 't_pt_s': t_pt, 't_mi_s': t_mi}
    np.save(out / f'smoke_pt_L{light}.npy', img_pt.astype(np.float32)); np.save(out / f'smoke_mi_L{light}.npy', img_mi.astype(np.float32))
# tessellation smoke: layered depth 4, mesh vs analytic sphere, vs WL84 analytic reference
im = 'test83_iid_1_L2'
ref84 = np.load(C.W84REF / f'{im}_ref84.npy').astype(np.float64)
for sph in ('mesh', 'analytic'):
    sc = RT.scene(tag, 'lay', light=2, sphere=sph, with_sensor=False)
    t0 = time.perf_counter(); img, _, _, Tm = RT.render_ref(sc, 2048, 104300, 4); tt = time.perf_counter() - t0
    obj, _ = R81.masks(sc)
    rep[f'tess_smoke_{sph}'] = {'obj_rel_l1_vs_ref84': R81.rel_l1(img, ref84, obj), 'obj_mean_ratio': float(img[obj].mean() / ref84[obj].mean()),
                                't_2048spp_depth4_s': tt, 'gpu_ms': Tm.get('gpu_ms')}
for d_ in (7, 65):
    sc = RT.scene(tag, 'lay', light=2, with_sensor=False)
    t0 = time.perf_counter(); img, _, _, Tm = RT.render_ref(sc, 2048, 104400, d_); rep[f'lay_2048spp_depth{d_}_s'] = time.perf_counter() - t0
print(json.dumps(rep, indent=1, default=float))
C.write_json(out / 'dev_probe104.json', rep)
import os
os._exit(0)
