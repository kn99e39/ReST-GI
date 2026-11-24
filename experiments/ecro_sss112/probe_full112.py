"""WL112 full-scale stage probe (development seeds +5,000,000 offset 7e6; never evidence): wall time per unit at real sizes."""
import json, os, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common112 as C  # noqa: E402
import tracer112 as T  # noqa: E402

PT = T.PT
out = {}
R = C.REPO / 'results/ecro_sss112/probe'; R.mkdir(parents=True, exist_ok=True)
geom, lc = C.fixture('F0'); light = PT.Lights(lc)
ctx = T.Ctx112(geom, 'C')
def mark(k, t0):
    out[k] = {'seconds': time.perf_counter() - t0, 'end': datetime.now().isoformat()}
    print(k, out[k], flush=True); C.jd(out, R / 'probe_full112.json')
t0 = time.perf_counter(); out['start_construct'] = datetime.now().isoformat()
acc = ctx.construct(light, 1024, 17201001, b=32); mark('construct_n1024', t0)
tabs = []
for r in range(11):
    for meas in ('U', 'G', 'V'):
        Tc, _, _, _ = T.tables_from(acc, meas, 'coarse'); Ts, _, _, _ = T.tables_from(acc, meas, 'sub')
        tabs += [(f'{meas}c{r}', PT.fl(Tc), 'coarse'), (f'{meas}s{r}', PT.fl(Ts), 'sub')]
tabs = tabs[:65]
import eval112 as E
gb = ctx.gbuffer(dict(C.V0_CAM, res=C.RES), C.RES); reg, _ = C.masks(gb, geom, lc[0]['pos'])
rm = {k: reg[k] for k in E.WREG}
t0 = time.perf_counter(); out['start_phys'] = datetime.now().isoformat()
img, _, _, _ = ctx.cam_pass(light, dict(C.V0_CAM, res=C.RES), C.RES, 4096, 17210000, 'cam', b=32); mark('phys_4096', t0)
t0 = time.perf_counter(); out['start_readout65'] = datetime.now().isoformat()
img, W, _, _ = ctx.cam_pass(light, dict(C.V0_CAM, res=C.RES), C.RES, 4096, 17210000, 'ecro', tables=tabs, b=32, region_masks=rm); mark('readout_65tables_4096', t0)
t0 = time.perf_counter(); out['start_readout6'] = datetime.now().isoformat()
img, W, _, _ = ctx.cam_pass(light, dict(C.V0_CAM, res=C.RES), C.RES, 4096, 17210000, 'ecro', tables=tabs[:6], b=32, region_masks=rm); mark('readout_6tables_4096', t0)
t0 = time.perf_counter(); out['start_pv'] = datetime.now().isoformat()
_, _, cells, _ = ctx.cam_pass(light, dict(C.V0_CAM, res=C.RES), C.RES, 1024, 17211000, 'cam', b=32, capture_cells=True); mark('pv_1024', t0)
t0 = time.perf_counter()
acc2 = ctx.construct(light, 32, 17220001, b=32); mark('construct_n32_training_field', t0)
os._exit(0)
