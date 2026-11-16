"""WL97 audit track D: CPU-only probe of the WL96A 'generator' CPU-time anomaly (no GPU: CUDA_VISIBLE_DEVICES must be empty).

Hypothesis under test: model86.GeoState converts the int16 hit / uint8 dbin state arrays (N * K = 650,240 elements) to int64 with torch
on the CPU (torch.as_tensor(np_array, dtype=torch.long, device=cuda) converts the dtype on the source device), which opens an OpenMP
parallel region; the worker team then spin-waits, which (a) adds process CPU time far above wall and (b) slows the single-threaded
Python work that follows (the q table tracing).  The R0 arm converts with numpy (xport87.XOp: hit.astype(np.int64)) - no OpenMP region.

  CUDA_VISIBLE_DEVICES= python omp_cpu_probe.py out.json
"""
import json, os, sys, time
import numpy as np
import torch

assert not torch.cuda.is_available(), 'CPU-only probe'
N, K = 1270, 512
rng = np.random.default_rng(97004)
hit = rng.integers(-1, N, N * K).astype(np.int16)
dbin = rng.integers(0, 64, N * K).astype(np.uint8)


def work(n=160000):
    """Fixed single-threaded Python workload (stand-in for Dr.Jit tracing in Python)."""
    s = 0
    for i in range(n):
        s += i * i % 7
    return s


def conv_torch():
    a = torch.as_tensor(hit, dtype=torch.long); b = torch.as_tensor(dbin, dtype=torch.long)
    return a, b


def conv_numpy():
    a = torch.as_tensor(hit.astype(np.int64)); b = torch.as_tensor(dbin.astype(np.int64))
    return a, b


def event(conv, tail_s=0.12):
    """conversion, then the fixed workload, then idle-wait (sleep) to the end of a 120 ms window; returns walls and CPU times."""
    c0 = time.process_time(); t0 = time.perf_counter()
    conv()
    t1 = time.perf_counter(); c1 = time.process_time()
    work()
    t2 = time.perf_counter(); c2 = time.process_time()
    while time.perf_counter() - t0 < tail_s:
        time.sleep(0.002)
    t3 = time.perf_counter(); c3 = time.process_time()
    return {'conv_wall_ms': (t1 - t0) * 1e3, 'conv_cpu_ms': (c1 - c0) * 1e3, 'work_wall_ms': (t2 - t1) * 1e3, 'work_cpu_ms': (c2 - c1) * 1e3,
            'event_wall_ms': (t3 - t0) * 1e3, 'event_cpu_ms': (c3 - c0) * 1e3}


def med(rows):
    return {k: float(np.median([r[k] for r in rows])) for k in rows[0]}


out = {'torch': torch.__version__, 'num_threads': torch.get_num_threads(), 'num_interop': torch.get_num_interop_threads(),
       'parallel_info': torch.__config__.parallel_info(), 'cpu_count': os.cpu_count(),
       'env': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OMP_WAIT_POLICY', 'GOMP_SPINCOUNT', 'KMP_BLOCKTIME', 'MKL_NUM_THREADS')},
       'loadavg': os.getloadavg()}
for _ in range(3):
    event(conv_torch); event(conv_numpy)
rows = {'torch_conv': [], 'numpy_conv': []}
for i in range(12):                                   # interleaved
    time.sleep(0.3)                                   # let the worker team go to sleep between events
    rows['torch_conv'].append(event(conv_torch))
    time.sleep(0.3)
    rows['numpy_conv'].append(event(conv_numpy))
out['median'] = {k: med(v) for k, v in rows.items()}
out['rows'] = rows
m = out['median']
out['work_slowdown_ms'] = m['torch_conv']['work_wall_ms'] - m['numpy_conv']['work_wall_ms']
out['extra_event_cpu_ms'] = m['torch_conv']['event_cpu_ms'] - m['numpy_conv']['event_cpu_ms']
# same with a single torch thread (control: no worker team)
torch.set_num_threads(1)
r1 = []
for i in range(8):
    time.sleep(0.1); r1.append(event(conv_torch))
out['median_torch_conv_1thread'] = med(r1)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))
