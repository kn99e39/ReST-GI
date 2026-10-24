"""Minimal Mitsuba cuda_ad_rgb render that reports which OptiX user-space library the process mapped.

Purpose (worklog 43): reproduce and attribute the LabServer63 Mitsuba segfault. The kernel driver and libcuda are
595.71.05 there, but the system libnvoptix.so.1 is 570.195.03. Run it once without and once with
LD_LIBRARY_PATH=~/nvlibs/optix595 (see mi_env.sh). Small: 64x64, 4 spp, a cube over a plane under a constant
environment, so it needs a few hundred MiB of GPU memory.

    python -X faulthandler mi_optix_check.py --out <json>
"""
import argparse
import json
import os
import re
import sys
import time


def mapped_libs(pattern):
    """Paths of shared objects currently mapped into this process whose name matches pattern."""
    out = set()
    with open('/proc/self/maps') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6 and re.search(pattern, parts[-1]):
                out.add(parts[-1])
    return sorted(out)


def kernel_driver_version():
    try:
        with open('/proc/driver/nvidia/version') as f:
            m = re.search(r'Kernel Module(?: for \S+)?\s+([0-9.]+)', f.read())
            return m.group(1) if m else None
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--res', type=int, default=64)
    ap.add_argument('--spp', type=int, default=4)
    ap.add_argument('--teardown', default='default', choices=['default', 'cleanup', 'exit', 'quiet'])
    args = ap.parse_args()

    rec = {'kernel_driver': kernel_driver_version(), 'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH'),
           'python': sys.version.split()[0], 'stage': 'import'}

    def dump():
        with open(args.out, 'w') as f:
            json.dump(rec, f, indent=1)

    dump()  # a segfault leaves the last completed stage on disk
    import mitsuba as mi
    import drjit as dr
    mi.set_variant('cuda_ad_rgb')
    rec.update(mitsuba=mi.__version__, drjit=dr.__version__, stage='variant_set')
    dump()

    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {'type': 'path', 'max_depth': 4},
        'sensor': {'type': 'perspective', 'fov': 40,
                   'to_world': mi.ScalarTransform4f.look_at(origin=[3, 2.5, 3], target=[0, 0.3, 0], up=[0, 1, 0]),
                   'film': {'type': 'hdrfilm', 'width': args.res, 'height': args.res},
                   'sampler': {'type': 'independent', 'sample_count': args.spp}},
        'env': {'type': 'constant', 'radiance': {'type': 'rgb', 'value': 1.0}},
        'box': {'type': 'cube', 'to_world': mi.ScalarTransform4f.translate([0, 0.5, 0]).scale(0.5),
                'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': [0.7, 0.1, 0.1]}}},
        'floor': {'type': 'rectangle', 'to_world': mi.ScalarTransform4f.rotate([1, 0, 0], -90).scale(4),
                  'bsdf': {'type': 'diffuse'}},
    })
    rec['stage'] = 'scene_loaded'  # the OptiX acceleration structure is built here
    rec['optix_libs_after_load'] = mapped_libs(r'libnvoptix|libnvidia-rtcore|libnvidia-gpucomp')
    dump()

    t = time.time()
    img = mi.render(scene, spp=args.spp, seed=0)
    dr.sync_thread()
    arr = img.numpy()
    rec.update(stage='rendered', render_s=round(time.time() - t, 3), mean_rgb=[float(v) for v in arr.reshape(-1, arr.shape[-1]).mean(0)],
               finite=bool((arr == arr).all()))
    rec['optix_libs_after_render'] = mapped_libs(r'libnvoptix|libnvidia-rtcore|libnvidia-gpucomp')
    rec['libcuda'] = mapped_libs(r'libcuda\.so')
    rec['teardown'] = args.teardown
    dump()
    print(json.dumps(rec), flush=True)

    # Teardown study: the render above succeeds with the matching OptiX libraries, but the process can still
    # crash inside jit_shutdown() at interpreter exit.
    if args.teardown == 'quiet':
        dr.set_log_level(dr.LogLevel.Disable)
    if args.teardown in ('cleanup', 'exit'):
        import gc
        del scene, img
        gc.collect()
        dr.sync_thread()
        dr.flush_malloc_cache()
    if args.teardown == 'exit':
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == '__main__':
    main()
