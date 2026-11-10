"""WL82 descriptive evaluation-throughput measurement (POST-CHAIN: declared in protocol evaluation.metrics but omitted from the frozen
chain code; run after the verdict, does not change any verdict). torch eager eval of 2^22 lanes per arm on the test_iid primary fits,
uncontended GPU, median of 20 timed repetitions after 3 warm-ups.

  python bench82_posthoc.py <fitdir> <out.json>
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import closure82 as CL  # noqa: E402

dev = torch.device('cuda')
fitdir, out = sys.argv[1], sys.argv[2]
n = 1 << 22
g = torch.Generator(device=dev).manual_seed(82200)
u = torch.rand(n, 4, generator=g, device=dev)


def dirs(a, b):
    st = torch.sqrt(1 - a * a)
    return torch.stack([st * torch.cos(6.2831853 * b), st * torch.sin(6.2831853 * b), a], 1)


wi, wo = dirs(u[:, 0], u[:, 1]), dirs(u[:, 2], u[:, 3])
res = {'lanes': n, 'note': 'torch eager, float32, eval only (no sampling); descriptive, not a runtime-utility claim'}
for tag in ['C0_cert_metric', 'E_cert_metric', 'Einc_unb_metric', 'ER_cert_metric']:
    arm, mode, _ = tag.split('_')
    z = torch.from_numpy(np.load(os.path.join(fitdir, f'fit_test_iid_{tag}_12000_generic.npz'))['z']).to(dev)
    p = CL.decode(arm, mode, z)
    m = torch.randint(0, z.shape[0], (n,), generator=g, device=dev)
    pl = CL.gather(p, m)
    ts = []
    with torch.no_grad():
        for r in range(23):
            torch.cuda.synchronize(); t0 = time.time()
            CL.eval_closure(arm, pl, wi, wo)
            torch.cuda.synchronize()
            if r >= 3:
                ts.append(time.time() - t0)
    res[tag] = {'median_s': float(np.median(ts)), 'ns_per_eval': float(np.median(ts)) / n * 1e9, 'state_floats': CL.compiled_state_floats(arm),
                'params': CL.n_params(arm)}
    print(tag, json.dumps(res[tag]), flush=True)
base = res['C0_cert_metric']['median_s']
for t in res:
    if isinstance(res[t], dict):
        res[t]['ratio_vs_C0'] = res[t]['median_s'] / base
res['gpu_procs_before'] = os.popen('nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader').read()
json.dump(res, open(out, 'w'), indent=1)
os._exit(0)
