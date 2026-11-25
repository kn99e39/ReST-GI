"""WL114 full-scale throughput / memory probe (throwaway dev seeds, never evidence)."""
import json, sys, time, threading, subprocess
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common114 as C  # noqa: E402
import tracer114 as T  # noqa: E402

PT, mi, dr = T.PT, T.mi, T.dr
SEED = 91141000
util = []
stop = False


def mon():
    while not stop:
        try:
            o = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
            u, m = [float(x) for x in o.strip().split(',')]
            util.append((time.perf_counter(), u, m))
        except Exception:
            pass
        time.sleep(0.5)


def phase(name, fn):
    global util
    util = []
    t0 = time.perf_counter(); r = fn(); dt = time.perf_counter() - t0
    u = [x[1] for x in util]; m = [x[2] for x in util]
    print(json.dumps({'phase': name, 'seconds': round(dt, 2), 'gpu_util_mean': float(np.mean(u)) if u else None,
                      'gpu_util_p10': float(np.percentile(u, 10)) if u else None, 'mem_peak_mib': max(m) if m else None}), flush=True)
    return r


def main(what):
    global stop
    th = threading.Thread(target=mon, daemon=True); th.start()
    geom, lc = C.fixture('F1'); light = PT.Lights(lc)
    ctx = T.Ctx114(geom, 'C')
    if 'con' in what:
        phase('construct_deep_b16_k23_n16_compile', lambda: ctx.construct_deep(light, 16, SEED, 23, b=16))
        phase('construct_deep_b16_k23_n32', lambda: ctx.construct_deep(light, 32, SEED + 1, 23, b=16))
        phase('construct4_b16_n32', lambda: ctx.construct4(light, 32, SEED + 2, b=16))
        phase('construct_deep_b16_k47_n32', lambda: ctx.construct_deep(light, 32, SEED + 3, 47, b=16))
        phase('construct_deep_b16_k191_n32', lambda: ctx.construct_deep(light, 32, SEED + 4, 191, b=16))
        phase('construct_deep_b16_rr6_n32', lambda: ctx.construct_deep(light, 32, SEED + 5, 1023, b=16, rr_from=6))
    if 'cam' in what:
        cam = dict(C.views('F1')['W0'], res=C.RES)
        for b in (32, 64):
            phase(f'deep_pass_k24_b{b}_256spp', lambda: ctx.deep_pass(light, cam, C.RES, 256, SEED + 10, 24, b=b))
        phase('full_pass113_k24_b64_256spp', lambda: ctx.full_pass(light, cam, C.RES, 256, SEED + 11, b=64))
        phase('deep_pass_k192_b64_256spp', lambda: ctx.deep_pass(light, cam, C.RES, 256, SEED + 12, 192, b=64))
        phase('deep_pass_rr6_b64_256spp', lambda: ctx.deep_pass(light, cam, C.RES, 256, SEED + 13, 1024, b=64, rr_from=6))
    stop = True


if __name__ == '__main__':
    main(sys.argv[1:] or ['con', 'cam'])
    import os; sys.stdout.flush(); os._exit(0)
