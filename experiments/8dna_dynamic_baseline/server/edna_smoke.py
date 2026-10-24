"""Tiny official-8DNA smoke on LabServer63: released seal checkpoint, seal scene2 (demo notebook setup),
`neuralpath` vs the scene's own prbvolpath reference. Feasibility only, not evidence: it checks that torch, the
upstream CUDA extension and Mitsuba cuda_ad_rgb run together in one process with the matching OptiX libraries.

    source mi_env.sh; python edna_smoke.py --upstream ~/NeuralShaderDemo/external/8dna26 --out smoke.json

Memory is kept small (default 64x64) because the GPU is shared with another job; torch is capped with
set_per_process_memory_fraction. Exits with os._exit after writing its record (DrJit 0.4.6 can crash in
jit_shutdown's leak report at interpreter exit; see mi_optix_check.py).
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--upstream', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--res', type=int, default=64)
    ap.add_argument('--spp', type=int, default=16)
    ap.add_argument('--ref-spp', type=int, default=256)
    ap.add_argument('--mem-fraction', type=float, default=0.06)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    rec = {'stage': 'start', 'args': vars(args), 'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH')}

    def dump():
        with open(out, 'w') as f:
            json.dump(rec, f, indent=1)

    dump()
    import torch
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    os.chdir(args.upstream)
    sys.path.insert(0, args.upstream)
    import mitsuba as mi
    mi.set_variant('cuda_ad_rgb')
    import drjit as dr
    import models.integrator  # noqa: F401  registers neuralpath / neuralvolpath
    dr.set_flag(dr.JitFlag.LoopRecord, False)  # demo notebook
    dr.set_flag(dr.JitFlag.VCallRecord, True)  # project runs (worklog 21); notebook uses False
    import scenes.seal as seal
    rec.update(stage='imported', torch=torch.__version__, mitsuba=mi.__version__, drjit=dr.__version__,
               gpu=torch.cuda.get_device_name(0))
    dump()

    scene = mi.load_dict(seal.scene2(args.res))
    t = time.time()
    integ = mi.load_dict({'type': 'neuralpath', 'max_depth': 10, 'rr_depth': 5, 'device': 0, 'model_type': '8dna',
                          'asset_instance0': os.path.join(args.upstream, 'checkpoints', 'seal.ckpt')})
    rec.update(stage='loaded', integrator_load_s=round(time.time() - t, 2))
    dump()

    def render(integrator, spp, seed0, chunk=4):
        torch.cuda.synchronize()
        s = time.time()
        acc = None
        for k in range(spp // chunk):
            img = mi.render(scene, integrator=integrator, spp=chunk, seed=seed0 + k)
            acc = img if acc is None else acc + img
        arr = np.array(acc, dtype=np.float32) / (spp // chunk)
        torch.cuda.synchronize()
        return arr, time.time() - s

    torch.cuda.reset_peak_memory_stats()
    neural, t_n = render(integ, args.spp, 0)
    neural2, t_n2 = render(integ, args.spp, 0)  # second pass: compiled kernels warm
    rec.update(stage='neural_done', neural_first_s=round(t_n, 2), neural_warm_s=round(t_n2, 2),
               peak_torch_bytes=int(torch.cuda.max_memory_allocated()))
    dump()
    ref, t_r = render(scene.integrator(), args.ref_spp, 200000, chunk=64)

    def disp(x):
        return np.clip(np.clip(x, 0, None) ** (1 / 2.2), 0, 1)

    mse = float(np.mean((disp(neural) - disp(ref)) ** 2))
    rec.update(stage='done', ref_s=round(t_r, 2), finite=bool(np.isfinite(neural).all() and np.isfinite(ref).all()),
               neural_mean=[float(v) for v in neural.reshape(-1, 3).mean(0)],
               ref_mean=[float(v) for v in ref.reshape(-1, 3).mean(0)],
               display_psnr_vs_ref_db=round(-10 * np.log10(mse), 2) if mse > 0 else None,
               same_seed_repeat_max_abs=float(np.abs(neural - neural2).max()))
    np.savez_compressed(out.replace('.json', '.npz'), neural=neural, ref=ref)
    dump()
    print(json.dumps(rec), flush=True)
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
