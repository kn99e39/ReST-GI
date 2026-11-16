"""Read-only fixed-checkpoint MC variance probe, not training or model selection."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
UP = Path.home() / 'NeuralShaderDemo/external/8dna26'
sys.path.insert(0, str(UP))
os.chdir(UP)
import bridge as B
import numpy as np
import torch
import models.integrator
from PIL import Image
mi, dr = B.mi, B.dr
dr.set_flag(dr.JitFlag.LoopRecord, False)
dr.set_flag(dr.JitFlag.VCallRecord, False)
out = B.BASE / 'results/noise_epoch12'
out.mkdir(exist_ok=True)
if (out / 'metrics.json').exists():
    raise RuntimeError('Preserve existing diagnostic')
checkpoint = B.BASE / 'training_packed/CK2_bx38/epoch=11-step=98304.ckpt'
saved = torch.load(checkpoint, map_location='cpu')
assert saved['global_step'] == 98304
del saved
scene = mi.load_dict(B.make_scene(256, max_depth=7)[0])
integrator = mi.load_dict({'type':'neuralpath','max_depth':10,'rr_depth':5,
    'device':0,'model_type':'8dna','asset_instance0':str(checkpoint)})
blocks, partials = [], []
started = time.perf_counter()
seeds = [221000, 222000, 223000, 224000]
for block, seed in enumerate(seeds):
    total = np.zeros((256,256,3), np.float64)
    for offset in range(128):
        image = mi.render(scene, integrator=integrator, spp=1, seed=seed+offset)
        dr.eval(image)
        torch.cuda.synchronize()
        value = np.array(image, dtype=np.float64)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite render')
        total += value
        if offset == 31:
            partials.append(total.copy()/32)
        if (offset+1)%32 == 0:
            print('BLOCK', block+1, 'SPP', offset+1, 'SECONDS', time.perf_counter()-started, flush=True)
    blocks.append(total/128)
    np.save(out / f'block{block+1}_128spp.npy', blocks[-1].astype(np.float32))
blocks, partials = np.stack(blocks), np.stack(partials)
mean = blocks.mean(0)
v32 = partials.var(axis=0, ddof=1)
v128 = blocks.var(axis=0, ddof=1)
masks = np.load(Path.home()/'rest_x93d_ck2/results/pre/masks_ck2.npz')
mask_summary = {}
regions = {'frame':np.ones((256,256), dtype=bool)}
for key in masks.files:
    value = masks[key]
    if value.shape == (512,512) and value.dtype == np.bool_:
        # Strict interior mask: every child pixel belongs to the region.
        regions[key] = value.reshape(256,2,256,2).all(axis=(1,3))
for name, mask in regions.items():
    if not mask.any(): continue
    var32 = float(v32[mask].mean())
    var128 = float(v128[mask].mean())
    mask_summary[name] = {'pixels':int(mask.sum()), 'variance_32spp':var32,
        'variance_128spp':var128, 'variance_ratio_32_over_128':var32/max(var128,1e-30),
        'rms_noise_128spp_linear':float(np.sqrt(var128)),
        'rms_noise_512spp_estimated_linear':float(np.sqrt(var128/4))}
def save_png(name, hdr):
    p = np.maximum(hdr,0)
    ldr = np.clip(p/(1+p),0,1)**(1/2.2)
    Image.fromarray(np.rint(ldr*255).astype(np.uint8)).save(out/name)
save_png('epoch12_128spp.png', blocks[0])
save_png('epoch12_512spp.png', mean)
save_png('epoch12_128_vs_512.png', np.concatenate([blocks[0],mean],axis=1))
np.save(out/'mean_512spp.npy',mean.astype(np.float32))
metrics = {'checkpoint_step':98304, 'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'resolution':[256,256], 'independent_blocks':4,'spp_per_block':128,'seeds':seeds,
    'partial_spp':32, 'regions':mask_summary,'seconds':time.perf_counter()-started,
    'scope':'Fixed 12-epoch model, four independent seeds. Variance across independent blocks; no physical reference or bias/convergence verdict. 512spp RMS is an estimate from 128spp variance, not an independent 512spp replicate measurement.',
    'display':'exposure1, x/(1+x), gamma2.2. Comparison left128 right512; nested samples.'}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics),flush=True)
