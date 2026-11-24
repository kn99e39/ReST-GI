"""WL113 development checks (throwaway seeds 1139xxxx; never evidence)."""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'ecro_sss112'))
import tracer113 as T  # noqa: E402
import common112 as C  # noqa: E402

PT = T.PT
out = {}
geom, lc = C.fixture('F0'); light = PT.Lights(lc)
ctx = T.Ctx113(geom, 'C')
cam = dict(C.V0_CAM, res=128)
# 1. deep pass events 1-6 == depth-6 camera pass; tail depth 24 vs 40
a, _, _, _ = ctx.cam_pass(light, cam, 128, 64, 11399001, 'cam', b=32)
t0 = time.perf_counter(); f, bad = ctx.full_pass(light, cam, 128, 64, 11399001, b=32); out['full24_128px_64spp_s'] = time.perf_counter() - t0
out['events1to6_rel_l1'] = float(sum(np.abs(a[k] - f[k]).sum() for k in ('e1', 'e2', 'h')) / sum(np.abs(a[k]).sum() for k in ('e1', 'e2', 'h')))
T.KMAX_FULL = 40
f40, _ = ctx.full_pass(light, cam, 128, 64, 11399002, b=32)
T.KMAX_FULL = 24
f24, _ = ctx.full_pass(light, cam, 128, 64, 11399002, b=32)
tot = T.lum(f24['e1'] + f24['e2'] + f24['h']).mean()
out['tail24_over_total'] = float(T.lum(f24['tail']).mean() / tot); out['tail40_minus_tail24_over_total'] = float((T.lum(f40['tail']) - T.lum(f24['tail'])).mean() / tot)
print(json.dumps(out), flush=True)
# 2. construction timing
t0 = time.perf_counter(); acc = ctx.construct4(light, 64, 11399010, b=8); out['construct4_n64_s'] = time.perf_counter() - t0
Tc, T2, T4, nat = T.tables4(acc)
rec = (T4.reshape(1270, 16, 64, 3) * acc['U_w'].reshape(1270, 16, 64, 1)).sum(1) / np.maximum(acc['U_w'].reshape(1270, 16, 64).sum(1), 1e-30)[..., None]
out['parent_rel'] = float(np.abs(rec - Tc).max() / np.abs(Tc).max())
print(json.dumps(out), flush=True)
# 3. VHS support on F0 and on dev training geometries
desc = ctx.vhs()
rng = desc[..., 0].max(1) - desc[..., 0].min(1)
g = ctx.G
for d in (0.05, 0.1, 0.15, 0.2, 0.3):
    sup = rng >= d
    out[f'support_frac_delta{d}'] = float(sup.mean())
    out[f'support_by_shape_delta{d}'] = {int(s): int(sup[g.PSID == s].sum()) for s in range(9)}
out['sphere_ring0_range'] = rng[(g.PSID == 6) & (g.PSPH[:, 1] == 0)].round(3).tolist()
out['floor155_range'] = float(rng[155])
print(json.dumps(out), flush=True)
(C.REPO / 'results/ecro_sss113/dev').mkdir(parents=True, exist_ok=True)
C.jd(out, C.REPO / 'results/ecro_sss113/dev/dev113.json')
np.save(C.REPO / 'results/ecro_sss113/dev/vhs_F0.npy', desc)
import os; sys.stdout.flush(); os._exit(0)
