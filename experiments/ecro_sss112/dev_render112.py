"""WL112 development probe (throwaway seeds 112992xx): render throughput, LOW pairing between physical and readout passes."""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tracer112 as T  # noqa: E402

PT, dr, mi = T.PT, T.dr, T.mi
REPO = T.EXP.parent
cfg = json.loads((REPO / 'results/ecro_material100/configs.json').read_text())['sss']['test'][0]
light = PT.Lights(cfg['lights'][0])
cam = dict(json.loads((T.EXP / 'ecro_material100/protocol.json').read_text())['datasets']['sss']['camera'])
ctx = T.Ctx112(cfg['geom'], 'C')
out = {}
acc = ctx.construct(light, 32, 11299200, b=32)
tabs = []
for meas in ('U', 'G'):
    for lev in ('coarse', 'sub'):
        Tt, *_ = T.tables_from(acc, meas, lev)
        tabs.append((f'{meas}_{lev}', PT.fl(Tt), lev))
tabs = tabs * 4
tabs = [(f'{n}{i}', a, k) for i, (n, a, k) in enumerate(tabs)]
res, spp = 512, 128
for b in (32, 64):
    t0 = time.perf_counter(); img, _, _, bad = ctx.cam_pass(light, cam, res, spp, 11299201, 'cam', b=b); dt = time.perf_counter() - t0
    out[f'cam_spp{spp}_b{b}_s'] = dt; print('cam', b, dt, bad, flush=True)
t0 = time.perf_counter(); img2, W, _, _ = ctx.cam_pass(light, cam, res, spp, 11299201, 'ecro', tables=tabs, b=32,
                                                       region_masks={'all': np.ones((res, res), bool)}); dt = time.perf_counter() - t0
out[f'ecro_{len(tabs)}tables_spp{spp}_s'] = dt; print('ecro', dt, flush=True)
for k in ('e1', 'e2'):
    out[f'pair_{k}_maxabs'] = float(np.abs(img[k] - img2[k]).max())
out['W_all_sum_vs_hl_U0'] = [float((W['all'][0] * np.repeat(T.tables_from(acc, 'U', 'sub')[0][:, 0], 1)).sum() / res / res * 0 + 0)]
hl = img2['U_coarse0']; h = img['h']
out['mean_h'] = float(T.lum(h).mean()); out['mean_hl_U'] = float(T.lum(hl).mean())
print(json.dumps(out, indent=1))
DEV = REPO / 'results/ecro_sss112/dev'
(DEV / 'dev_render112.json').write_text(json.dumps(out, indent=1))
import os
os._exit(0)
