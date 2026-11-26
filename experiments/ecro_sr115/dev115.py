"""WL115 development checks (throwaway seeds; never evidence): readout kernel identity against tracer112.cam_pass, BL on constant
fields, construct_k timing / address mismatch."""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common115 as C
import tracer115 as T
PT, mi, dr = T.PT, T.mi, T.dr
g, l = C.fixture('F0'); cam = dict(C.views('F0')['V0'], res=C.RES)
ctx = T.Ctx115(g, 'C'); light = PT.Lights(l)
z = np.load(C.WL114_ROOT / 'stageB' / 'tables' / 'oracle_F0_r0.npz')
S4 = z['S4_F']
seed = 7999001
t0 = time.perf_counter()
img, _, _, _ = ctx.cam_pass(light, cam, C.RES, 64, seed, 'ecro', tables=[('S4', PT.fl(S4), 'sub')], b=64)
t1 = time.perf_counter()
# constant-in-space field: per patch value depends on the bin only for rects; BL must equal PC exactly on rect-only queries
Tc = np.repeat(S4.mean(1, keepdims=True), 16, 1)
reps = {'S4': T.pc_uniform(S4, 4), 'BL4': T.bl_rep(S4, 4), 'C_pc': T.pc_uniform(Tc, 4), 'C_bl': T.bl_rep(Tc, 4)}
im2 = ctx.readout_multi(light, cam, C.RES, 64, seed, reps, b=64)
t2 = time.perf_counter()
r = {'cam_pass_s': t1 - t0, 'readout_multi_s': t2 - t1,
     'low_max_abs': float(max(np.abs(img['e1'] - im2['e1']).max(), np.abs(img['e2'] - im2['e2']).max())),
     'S4_max_abs': float(np.abs(img['S4'] - im2['S4']).max()), 'S4_rel': float(np.abs(img['S4'] - im2['S4']).sum() / np.abs(img['S4']).sum()),
     'BL4_vs_PC4_rel': float(np.abs(im2['BL4'] - im2['S4']).sum() / np.abs(im2['S4']).sum()),
     'const_bl_vs_pc_rel': float(np.abs(im2['C_bl'] - im2['C_pc']).sum() / np.abs(im2['C_pc']).sum())}
print(json.dumps(r, indent=1), flush=True)
# construct_k timing at K = 8 (one launch b = 2) and K = 32 on the contact patches
for Kk, pids, bb in ((8, None, 2), (8, None, 3), (32, T.contact_patches(ctx.G, g['S']['c'][:2]), 1)):
    t0 = time.perf_counter()
    acc, dg = ctx.construct_k(light, bb, 7999100, 23, Kk, pids=pids, b=bb)
    dr.sync_thread()
    print('construct_k', Kk, 'P', len(pids) if pids is not None else 1270, 'b', bb, 'sec', round(time.perf_counter() - t0, 2),
          'mismatch/counted', dg['mismatch'] / max(dg['counted'], 1), 'counted', dg['counted'], flush=True)
import os; os._exit(0)
