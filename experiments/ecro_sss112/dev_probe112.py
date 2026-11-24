"""WL112 development probe (throwaway seeds 112991xx): construction throughput and legacy-sampler identity with the WL100 target."""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tracer112 as T  # noqa: E402

PT = T.PT
REPO = T.EXP.parent
cfg = json.loads((REPO / 'results/ecro_material100/configs.json').read_text())['sss']['test'][0]
light = PT.Lights(cfg['lights'][0])
out = {}
ctxn = T.Ctx112(cfg['geom'], 'N')
for b in (8, 16, 32):
    t0 = time.perf_counter(); acc = ctxn.construct(light, 32, 11299100 + b, b=b); T.dr.sync_thread()
    out[f'v3N_n32_b{b}_s'] = time.perf_counter() - t0
    print('b', b, out[f'v3N_n32_b{b}_s'], flush=True)
t0 = time.perf_counter(); leg = ctxn.construct(light, 64, 11299150, b=16, legacy=True); out['legacy_n64_s'] = time.perf_counter() - t0
t0 = time.perf_counter(); v3 = ctxn.construct(light, 64, 11299151, b=16); out['v3N_n64_s'] = time.perf_counter() - t0
z = np.load(REPO / 'results/ecro_material100/sss/data/test/g00000.npz')
tgt = 0.5 * (z['f2'][0] + z['f2b'][0] + z['fh'][0] + z['fhb'][0])          # [1270,64,3], WL100 legacy sampler, 2 x 256 spp
Tl, wl, _, _ = T.tables_from(leg, 'U', 'coarse'); Tl = Tl.reshape(1270, 64, 3)
Tv, wv, _, _ = T.tables_from(v3, 'U', 'coarse'); Tv = Tv.reshape(1270, 64, 3)
rel = lambda a, b: float(np.abs(a - b).sum() / np.abs(b).sum())
out['legacy_vs_wl100_relL1'] = rel(Tl, tgt)
out['v3N_vs_wl100_relL1'] = rel(Tv, tgt)
half = 0.5 * (z['f2'][0] + z['fh'][0]) - 0.5 * (z['f2b'][0] + z['fhb'][0])
out['wl100_half_noise_relL1'] = float(np.abs(half).sum() / np.abs(tgt).sum())
out['legacy_misroute'] = float(leg['misroute'].sum()); out['legacy_backcount'] = float(leg['backcount'].sum())
d = np.abs(Tv - Tl).sum((1, 2)); top = np.argsort(d)[::-1][:12]
out['largest_patch_diffs_v3N_minus_legacy'] = [[int(p), float((Tv[p] - Tl[p]).mean()), float(Tl[p].mean())] for p in top]
print(json.dumps(out, indent=1), flush=True)
DEV = REPO / 'results/ecro_sss112/dev'; DEV.mkdir(parents=True, exist_ok=True)
(DEV / 'dev_probe112.json').write_text(json.dumps(out, indent=1))
import os
os._exit(0)
