"""User-requested single-frame 12/30 epoch preview; never changes training state."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

UP = Path.home() / 'NeuralShaderDemo' / 'external' / '8dna26'
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
checkpoint = B.BASE / 'training_packed/CK2_bx38/epoch=11-step=98304.ckpt'
saved = torch.load(checkpoint, map_location='cpu')
assert int(saved['epoch']) == 11 and int(saved['global_step']) == 98304
del saved
out = B.BASE / 'results'
stem = '8dna_ck2_epoch12_preview'
if (out / (stem + '.json')).exists():
    raise RuntimeError('Preview already exists; preserve its provenance')
scene = mi.load_dict(B.make_scene(512, max_depth=7)[0])
integrator = mi.load_dict({'type': 'neuralpath', 'max_depth': 10, 'rr_depth': 5,
                           'device': 0, 'model_type': '8dna',
                           'asset_instance0': str(checkpoint)})
total = np.zeros((512, 512, 3), dtype=np.float64)
start = time.perf_counter()
spp, seed = 128, 21201
for offset in range(spp):
    rendered = mi.render(scene, integrator=integrator, spp=1, seed=seed + offset)
    dr.eval(rendered)
    torch.cuda.synchronize()
    frame = np.array(rendered, dtype=np.float64)
    if not np.isfinite(frame).all():
        raise ValueError('Nonfinite render')
    total += frame
    if (offset + 1) % 16 == 0:
        print('RENDER', offset + 1, '/', spp, 'seconds', time.perf_counter() - start, flush=True)
hdr = (total / spp).astype(np.float32)
np.save(out / (stem + '.npy'), hdr)
mi.Bitmap(hdr).write(str(out / (stem + '.exr')))
positive = np.maximum(hdr, 0)
display = np.clip(positive / (1 + positive), 0, 1) ** (1 / 2.2)
png = out / (stem + '.png')
Image.fromarray(np.rint(display * 255).astype(np.uint8)).save(png)
metadata = {
    'stage': '12/30 epochs PREVIEW; not final qualification',
    'checkpoint': str(checkpoint), 'checkpoint_epoch_zero_based': 11,
    'checkpoint_global_step': 98304,
    'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'resolution': [512, 512], 'spp': spp, 'seed_base': seed,
    'sampling': '128 one-spp passes, seeds 21201..21328; same as first preview replicate',
    'display': 'exposure 1; Reinhard x/(1+x); gamma 2.2',
    'render_seconds': time.perf_counter() - start,
    'training': 'Independent read-only checkpoint render concurrent with ongoing training; transient resource contention possible.',
    'png_sha256': hashlib.sha256(png.read_bytes()).hexdigest(),
    'png': str(png),
}
(out / (stem + '.json')).write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata), flush=True)
