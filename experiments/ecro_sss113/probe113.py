"""WL113 full-scale stage probe (throwaway seeds 1179xxxx; never evidence): wall time per real-size unit; GPU utilisation is
logged externally by nvidia-smi and matched to the stage time stamps."""
import json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common113 as C  # noqa: E402
import tracer113 as T  # noqa: E402
import eval113 as E  # noqa: E402

PT = T.PT
R = C.REPO / 'results/ecro_sss113/probe'; R.mkdir(parents=True, exist_ok=True)
out = {}
geom, lc = C.fixture('F0'); light = PT.Lights(lc)
ctx = T.Ctx113(geom, 'C')
b = int(sys.argv[1]) if len(sys.argv) > 1 else 64


def stage(name, fn):
    s = datetime.now().isoformat(); t0 = time.perf_counter(); r = fn(); T.dr.sync_thread()
    out[name] = {'start': s, 'end': datetime.now().isoformat(), 'seconds': time.perf_counter() - t0}
    print(name, round(out[name]['seconds'], 1), flush=True); C.jd(out, R / f'probe113_b{b}.json'); return r


acc = stage('construct4_n1024_b32', lambda: ctx.construct4(light, 1024, 17901001, b=32))
stage('construct4_save_async', lambda: (E.write_npz(R / 'probe_acc.npz', **{k: v.astype(np.float32) for k, v in acc.items()}), None)[1])
Tc, T2, T4, _ = T.tables4(acc)
tabs = []
for r in range(4):
    tabs += [(f'U{r}', T.to_dense16(Tc, 'coarse')), (f'S2{r}', T.to_dense16(T2, 'sub2')), (f'S4{r}', T4), (f'L4{r}', T4), (f'R0{r}', T4)]
tabs += [(f'D{r}', T4) for r in range(4)]
cam = dict(C.V0_CAM, res=C.RES)
gb = ctx.gbuffer(cam, C.RES); reg, _ = C.masks(gb, geom, lc[0]['pos'])
stage('phys_d6_4096_compile+run', lambda: ctx.cam_pass(light, cam, C.RES, 4096, 17910000, 'cam', b=b))
stage('tail_1024', lambda: ctx.full_pass(light, cam, C.RES, 1024, 17911000, b=E.TLAUNCH))
tl = [(n_, PT.fl(np.ascontiguousarray(t_, np.float32).ravel()), 'sub') for n_, t_ in tabs]
rm = {k: reg[k] for k in E.WREG}
stage('readout_24tables_4096', lambda: ctx.cam_pass(light, cam, C.RES, 4096, 17910000, 'ecro', tables=tl, b=b, region_masks=rm))
stage('phys_d6_4096_warm', lambda: ctx.cam_pass(light, cam, C.RES, 4096, 17910001, 'cam', b=b))
stage('vhs', lambda: ctx.vhs())
E.flush_writes()
import os; sys.stdout.flush(); os._exit(0)
