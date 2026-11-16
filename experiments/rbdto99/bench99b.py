"""WL99 dev: render throughput / GPU utilisation of the current pt99.render (execution check, not evidence)."""
import json, subprocess, sys, threading, time
import numpy as np
import pt99, wl99
w = {x['id']: x for x in wl99.WL}['A1_g90s1']
cfg = wl99.configs(w)['T0']
ctx = pt99.Ctx(cfg)
for b in (int(a) for a in sys.argv[1:]):
    ctx.render(cfg['camera'], 384, b, 1, b=b)          # warm-up / compile
    util = []
    stop = [False]
    def mon():
        while not stop[0]:
            o = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
            util.append(float(o.strip().split()[0])); time.sleep(0.5)
    th = threading.Thread(target=mon); th.start()
    acc, Tm = ctx.render(cfg['camera'], 384, 1024, 3, b=b)
    stop[0] = True; th.join()
    print(json.dumps({'b': b, 'wall_s_1024spp': round(Tm['wall_s'], 2), 'gpu_ms': round(Tm['gpu_ms'], 1), 'spp_per_s': round(1024 / Tm['wall_s'], 1),
                      'gpu_util_median': float(np.median(util)) if util else None}), flush=True)
