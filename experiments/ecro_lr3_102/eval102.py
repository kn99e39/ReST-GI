"""WL102 ECRO evaluation on LR3 with the scene's own area light (a held-out source class: the model never saw it).

  python eval102.py <scene.xml> <data_root> <model_dir> <out_dir>
Steps (each timed): source encoding s for the area light (SceneLights, 64 emitter samples per source point); generator factors +
application -> table B [N, 64, 3] (2..5 events at the second hit); ECRO frames (LOW e1, e2 path-traced + hl lookup):
  4 spp: 20 frames with different seeds (operating point; warm median wall time per frame), the first saved;
  2 x 512 spp halves (converged LOW; isolates the table error from LOW noise).
Physical components for the higher-order check: the same tracer at D = 6, 2 x 1,024 spp, e2 and h stored separately (A2-validated).
Point-light controls: table-level errors on the held-out control lights (test split), as train102.evaluate.
"""
import json, os, sys, time
import numpy as np
import torch
import pt102 as P

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ecro86'))
import model86 as M  # noqa: E402
import train102 as TRN  # noqa: E402

PT, mi, dr = P.PT, P.mi, P.dr
xml, data, mdir, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
dev = torch.device('cuda')
SMOKE = bool(os.environ.get('WL102_SMOKE'))           # development smoke: tiny sample counts, never evidence
NF, S512, S1024 = (3, 8, 8) if SMOKE else (20, 512, 1024)


def sync():
    dr.sync_thread(); torch.cuda.synchronize()


ctx = P.Ctx(P.load_scene(xml))
from pathlib import Path  # noqa: E402
gs, rows = TRN.geostate(Path(data))
model = M.ECRO().to(dev); model.mat_states = rows
model.load_state_dict(torch.load(os.path.join(mdir, 'model.pt'), map_location=dev)); model.eval()
T = {}
L = P.SceneLights(64)
for rep in range(3):                                            # first call compiles; report the last
    sync(); t0 = time.perf_counter(); s = ctx.source(L); sync(); T['source_s'] = time.perf_counter() - t0
    with torch.no_grad():
        sync(); t0 = time.perf_counter(); f = model.factors(TRN.input_state(gs)); sync(); T['factors_s'] = time.perf_counter() - t0
        st = torch.as_tensor(s, device=dev)[None]
        sync(); t0 = time.perf_counter(); B = model.apply(f, gs, st)[0]; sync(); T['apply_s'] = time.perf_counter() - t0
Bn = B.detach().cpu().numpy().astype(np.float32)
np.savez_compressed(os.path.join(out, 'table_area_light.npz'), B=Bn, s=s)
Bf = P.fl(Bn)
W, H = ctx.scene.sensors()[0].film().crop_size()
frames = []
for i in range(NF):
    acc, Tm = ctx.render(L, 4, 4000 + i, 'ecro', B=Bf, b=4)
    frames.append(Tm['wall_s'])
    if i == 0:
        np.savez_compressed(os.path.join(out, 'ecro_spp4.npz'), **P.pack(acc, W, H, 4))
T['frame_4spp_wall_s'] = {'median_warm': float(np.median(frames[2:] if len(frames) > 2 else frames)), 'all': frames}
for h, seed in enumerate((5001, 5002)):
    acc, Tm = ctx.render(L, S512, seed, 'ecro', B=Bf, b=4)
    np.savez_compressed(os.path.join(out, f'ecro_spp512_h{h}.npz'), **P.pack(acc, W, H, S512))
for h, seed in enumerate((6001, 6002)):
    acc, Tm = ctx.render(L, S1024, seed, 'cam', b=8)
    np.savez_compressed(os.path.join(out, f'phys_d6_spp1024_h{h}.npz'), **P.pack(acc, W, H, S1024))
test = [TRN.Item(f) for f in sorted((Path(data) / 'test').glob('g*.npz'))]
ctrl = TRN.evaluate(model, gs, test) if test else None
val = TRN.evaluate(model, gs, [TRN.Item(f) for f in sorted((Path(data) / 'val').glob('g*.npz'))])
json.dump({'smoke': SMOKE, 'area_input_floor': TRN.AREA_FLOOR, 'timing': T, 'controls_point_lights_test': ctrl, 'validation': val, 'N': ctx.N, 'res': [W, H],
           'table_bytes': int(Bn.nbytes), 'peak_mib': torch.cuda.max_memory_allocated() / 2 ** 20}, open(os.path.join(out, 'eval.json'), 'w'), indent=1)
print('EVAL_COMPLETE', json.dumps(T)[:400], ctrl, flush=True)
