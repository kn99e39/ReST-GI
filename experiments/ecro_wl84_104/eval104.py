"""WL104 ECRO evaluation of one WL84 image with the trained model of its material.

  python eval104.py <image> <data_root> <model_dir> <out_dir>
Steps (timed, local RTX 5080 shared with the desktop): source encoding s for the WL84 point light (exact pt86 NEE source, the WL84
intensity), generator factors + application -> table B [N, 64, 3], ECRO frames (LOW e1 + e2 path-traced with the learned-ER closure +
hl lookup): 20 x 4 spp (operating point), 8 x 64 spp (WL84's ER_L spp), 2 x 1,024-spp halves (converged LOW; isolates table error).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import common104 as C
import rt104 as RT
import model86 as M
import train102 as TRN

image, data, mdir, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
tag, light, _, _ = C.image_spec(image)
ii = C.IMAGES.index(image)
_, state, h = RT.register(tag)
assert h == C.STATE_SHA[tag]
dev = torch.device('cuda')
mi, dr, P = RT.mi, RT.dr, RT.P


def sync():
    dr.sync_thread(); torch.cuda.synchronize()


ctx = RT.Ctx(RT.scene(tag, 'er'))
gs, rows = TRN.geostate(Path(data))
model = M.ECRO().to(dev); model.mat_states = rows
model.load_state_dict(torch.load(os.path.join(mdir, 'model.pt'), map_location=dev)); model.eval()
L = RT.point_lights(light)
T = {}
for rep in range(3):                                   # first call compiles; the last is reported
    sync(); t0 = time.perf_counter(); s = ctx.source(L); sync(); T['source_s'] = time.perf_counter() - t0
    with torch.no_grad():
        sync(); t0 = time.perf_counter(); f = model.factors(TRN.input_state(gs)); sync(); T['factors_s'] = time.perf_counter() - t0
        st = torch.as_tensor(s, device=dev)[None]
        sync(); t0 = time.perf_counter(); B = model.apply(f, gs, st)[0]; sync(); T['apply_s'] = time.perf_counter() - t0
Bn = B.detach().cpu().numpy().astype(np.float32)
np.savez_compressed(os.path.join(out, 'table.npz'), B=Bn, s=s)
Bf = P.fl(Bn)
W = H = C.RES


def frames(spp, seeds, name, b):
    imgs, walls = [], []
    for sd in seeds:
        acc, Tm = ctx.render(L, spp, sd, 'ecro', B=Bf, b=b)
        pk = P.pack(acc, W, H, spp)
        imgs.append(np.stack([pk['e1'], pk['e2'], pk['hl']]))
        walls.append(Tm['wall_s'])
    np.save(os.path.join(out, f'{name}.npy'), np.stack(imgs).astype(np.float32))       # [n, 3 (e1, e2, hl), H, W, 3]
    return walls


NF4, NF64, SC = (4, 2, 64) if C.SMOKE else (20, 8, 1024)
w4 = frames(4, [104500 + 1000 * ii + q for q in range(NF4)], 'ecro_spp4', 4)
w64 = frames(64, [104520 + 1000 * ii + q for q in range(NF64)], 'ecro_spp64', 8)
wc = frames(SC, [104540 + 1000 * ii + q for q in range(2)], 'ecro_spp1024_halves', 8)
T['frame_4spp_wall_s'] = {'median_warm': float(np.median(w4[2:])), 'all': w4}
T['frame_64spp_wall_s'] = {'median_warm': float(np.median(w64[1:])), 'all': w64}
T['frame_1024spp_wall_s'] = wc
test = [TRN.Item(f) for f in sorted((Path(data) / 'test').glob('g*.npz'))]
ctrl = TRN.evaluate(model, gs, test) if test else None
C.write_json(os.path.join(out, 'eval.json'), {'image': image, 'timing': T, 'table_level_eval_lights_L1_L2': ctrl, 'N': int(ctx.N),
                                              'table_bytes': int(Bn.nbytes), 'peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20})
print('EVAL_COMPLETE', image, json.dumps(T)[:300], ctrl, flush=True)
os._exit(0)
