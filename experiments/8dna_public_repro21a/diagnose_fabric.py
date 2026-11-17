"""Bounded Fabric execution probe with phase logging and periodic Python stacks."""
import argparse
import faulthandler
import importlib
import json
import os
import sys
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--upstream', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--vcall', type=int, choices=[0, 1], required=True)
parser.add_argument('--res', type=int, default=32)
parser.add_argument('--spp', type=int, default=1)
parser.add_argument('--crop', type=int, nargs=4, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'))
args = parser.parse_args()
root = Path(args.upstream).resolve()
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
os.chdir(root / 'demo')
sys.path.insert(0, str(root))
started = time.perf_counter()
def event(phase, **values):
    record = {'phase': phase, 'seconds': time.perf_counter() - started, **values}
    print(json.dumps(record), flush=True)
    (out / 'phase.json').write_text(json.dumps(record, indent=2) + '\n')

faulthandler.enable()
faulthandler.dump_traceback_later(15, repeat=True)
event('import_start')
import numpy as np
import torch
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr
import models.integrator
torch.set_num_threads(4)
dr.set_flag(dr.JitFlag.LoopRecord, False)
dr.set_flag(dr.JitFlag.VCallRecord, bool(args.vcall))
event('scene_load_start', vcall=args.vcall, res=args.res, spp=args.spp)
definition = importlib.import_module('scenes.fabric').scene1(args.res)
if args.crop:
    x, y, width, height = args.crop
    definition['camera']['film'].update(crop_width=width, crop_height=height,
                                         crop_offset_x=x, crop_offset_y=y)
scene = mi.load_dict(definition)
event('asset_load_start')
integrator = mi.load_dict({'type': 'neuralpath', 'max_depth': 10, 'rr_depth': 5,
    'device': 0, 'model_type': '8dna', 'asset_instance0': str(root / 'checkpoints/fabric.ckpt')})
event('render_start')
image = mi.render(scene, integrator=integrator, spp=args.spp, seed=901000)
event('evaluate_start')
dr.eval(image)
torch.cuda.synchronize()
image = np.array(image, dtype=np.float32)
if not np.isfinite(image).all():
    raise RuntimeError('Nonfinite probe')
np.save(out / 'image.npy', image)
memory = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
    '--format=csv,noheader,nounits'], text=True)
gpu_mib = next((int(line.split(',')[1]) for line in memory.splitlines()
                if int(line.split(',')[0]) == os.getpid()), 0)
event('complete', shape=list(image.shape), mean=float(image.mean()),
      peak_torch_bytes=torch.cuda.max_memory_allocated(), gpu_process_mib=gpu_mib)
faulthandler.cancel_dump_traceback_later()
