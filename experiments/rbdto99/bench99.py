"""WL99 dev: render timing breakdown (compile vs steady) for one candidate."""
import json, sys, time
import pt99
dr = pt99.dr
dr.set_flag(dr.JitFlag.KernelHistory, True)
c = json.load(open(sys.argv[1]))[int(sys.argv[2])]
b = int(sys.argv[3]) if len(sys.argv) > 3 else 16
ctx = pt99.Ctx(c)
for rep in range(3):
    t0 = time.perf_counter()
    acc, Tm = ctx.render(c['camera'], 384, 16, seed=7 + rep, b=b)
    print(rep, 'wall', round(time.perf_counter() - t0, 2), 'gpu_ms', round(Tm['gpu_ms'], 1), 'launches', Tm['launches'], flush=True)
