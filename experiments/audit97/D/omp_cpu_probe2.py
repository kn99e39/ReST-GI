"""WL97 audit track D: CPU-only probe 2 (no GPU).  Does the OpenMP worker spin that follows the torch CPU dtype conversion slow a
single-threaded workload on a main thread that was ALREADY busy (as in the WL96A edit event, where the q table follows the generator)?
The main thread is kept busy for 0.4 s before every event (single-core turbo state), then: conversion, 5 consecutive fixed work chunks.

  CUDA_VISIBLE_DEVICES= python omp_cpu_probe2.py out.json
"""
import json, os, sys, time
import numpy as np
import torch

assert not torch.cuda.is_available(), 'CPU-only probe'
N, K = 1270, 512
rng = np.random.default_rng(97004)
hit = rng.integers(-1, N, N * K).astype(np.int16)
dbin = rng.integers(0, 64, N * K).astype(np.uint8)


def work(n):
    s = 0
    for i in range(n):
        s += i * i % 7
    return s


def busy(sec):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec:
        work(20000)


# calibrate a chunk to ~12 ms on a warm core
busy(0.5)
t0 = time.perf_counter(); work(100000); per = (time.perf_counter() - t0) / 100000
NCH = int(0.012 / per)


def event(kind):
    busy(0.4)
    c0 = time.process_time(); t0 = time.perf_counter()
    if kind == 'torch':
        a = torch.as_tensor(hit, dtype=torch.long); b = torch.as_tensor(dbin, dtype=torch.long)
    else:
        a = torch.as_tensor(hit.astype(np.int64)); b = torch.as_tensor(dbin.astype(np.int64))
    t1 = time.perf_counter()
    ch = []
    for _ in range(8):
        ta = time.perf_counter(); work(NCH); ch.append((time.perf_counter() - ta) * 1e3)
    c1 = time.process_time(); t2 = time.perf_counter()
    return {'conv_wall_ms': (t1 - t0) * 1e3, 'chunks_ms': ch, 'event_wall_ms': (t2 - t0) * 1e3, 'event_cpu_ms': (c1 - c0) * 1e3}


out = {'num_threads': torch.get_num_threads(), 'chunk_iters': NCH, 'loadavg': os.getloadavg()}
for _ in range(2):
    event('torch'); event('numpy')
rows = {'torch': [], 'numpy': []}
for i in range(15):
    rows['torch'].append(event('torch')); rows['numpy'].append(event('numpy'))
for k in rows:
    C = np.array([r['chunks_ms'] for r in rows[k]])
    out[k] = {'chunk_median_ms': np.median(C, 0).tolist(), 'first3_chunks_sum_median_ms': float(np.median(C[:, :3].sum(1))),
              'last3_chunks_sum_median_ms': float(np.median(C[:, 5:].sum(1))),
              'event_wall_ms': float(np.median([r['event_wall_ms'] for r in rows[k]])), 'event_cpu_ms': float(np.median([r['event_cpu_ms'] for r in rows[k]])),
              'conv_wall_ms': float(np.median([r['conv_wall_ms'] for r in rows[k]]))}
out['first3_slowdown_ratio'] = out['torch']['first3_chunks_sum_median_ms'] / out['numpy']['first3_chunks_sum_median_ms']
out['last3_slowdown_ratio'] = out['torch']['last3_chunks_sum_median_ms'] / out['numpy']['last3_chunks_sum_median_ms']
out['rows'] = rows
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, indent=1))
