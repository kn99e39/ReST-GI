"""Check tile projection, filter borders, serial/parallel output and resumption."""
import argparse
import importlib
import json
import os
import sys
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--upstream', required=True)
parser.add_argument('--base', required=True)
args = parser.parse_args()
root, base = Path(args.upstream).resolve(), Path(args.base).resolve()
os.chdir(root/'demo')
sys.path.insert(0, str(root))
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr
module = importlib.import_module('scenes.fabric')
camera = module.scene1(800)['camera']
full = mi.load_dict(camera)
ray_error = 0.0
for x, y, width, height in [(0,0,128,128), (336,336,128,128), (768,768,32,32)]:
    cropped = camera.copy()
    cropped['film'] = camera['film'].copy()
    cropped['film'].update(crop_width=width, crop_height=height, crop_offset_x=x, crop_offset_y=y)
    sensor = mi.load_dict(cropped)
    for u, v in [(0.1,0.2), (0.5,0.5), (0.8,0.9)]:
        ray_full, _ = full.sample_ray_differential(0,0.5,mi.Point2f((x+u*width)/800,(y+v*height)/800),mi.Point2f(0.5))
        ray_crop, _ = sensor.sample_ray_differential(0,0.5,mi.Point2f(u,v),mi.Point2f(0.5))
        for left, right in [(ray_full.o,ray_crop.o),(ray_full.d,ray_crop.d)]:
            ray_error = max(ray_error,float(np.abs(np.array(left)-np.array(right)).max()))
assert ray_error < 2e-6, ray_error

# Constant illumination exposes omitted/doubled pixels at tile boundaries.
camera = module.scene1(64)['camera']
scene = mi.load_dict({'type':'scene','camera':camera,'integrator':{'type':'path','max_depth':2},
    'environment':{'type':'constant','radiance':{'type':'rgb','value':[0.2,0.4,0.8]}}})
reference = np.array(mi.render(scene,spp=1,seed=9901))
stitched = np.zeros_like(reference)
for y in (0,32):
    for x in (0,32):
        left, top = max(0,x-1), max(0,y-1)
        right, bottom = min(64,x+33), min(64,y+33)
        cropped = camera.copy(); cropped['film'] = camera['film'].copy()
        cropped['film'].update(crop_width=right-left,crop_height=bottom-top,crop_offset_x=left,crop_offset_y=top)
        sensor = mi.load_dict(cropped)
        pixels = np.array(mi.render(scene,sensor=sensor,spp=1,seed=9902))
        stitched[y:y+32,x:x+32] = pixels[y-top:y-top+32,x-left:x-left+32]
seam_error = float(np.abs(stitched-reference).max())
assert seam_error < 2e-6, seam_error

name = 'fabric_neural_r0_256_s32.npy'
serial = np.load(base/'parallel1'/name)
parallel = np.load(base/'parallel2'/name)
difference = float(np.abs(serial-parallel).max())
assert np.isfinite(parallel).all() and np.allclose(serial,parallel,rtol=1e-5,atol=1e-5), difference
timings = [json.loads((base/f'parallel{workers}/fabric_neural_r0_256.json').read_text()) for workers in (1,2)]
result = {'camera_ray_max_abs':ray_error,'constant_scene_seam_max_abs':seam_error,
    'serial_parallel_max_abs':difference,'wall_seconds':[r['timed_wall_s'] for r in timings],
    'speedup':timings[0]['timed_wall_s']/timings[1]['timed_wall_s'],
    'average_cpu_cores':[r['average_cpu_cores'] for r in timings]}
resume = base/'resume/fabric_neural_r0_256_s32.npy'
if resume.exists():
    resumed = np.load(resume)
    result['resume_max_abs'] = float(np.abs(parallel-resumed).max())
    assert np.allclose(parallel,resumed,rtol=1e-5,atol=1e-5), result
(base/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
