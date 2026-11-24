"""Diagnostic (not evidence; owner request 2026-11-24): why the candidate CUDA graph takes more device time than its eager
components. Device time of every step of GraphCandidate._body('S') eagerly, and of partial graphs (network only, network
with cudnn.benchmark, passes only). RTX 3080 Ti (GPU1) only.

python dbg_graph_r111.py OUT_JSON
"""
import sys
import time

import numpy as np
import torch

import common_r111 as C
import recon_r111 as RC

W = C.RES
N = W * W


def gpu_ms(fn, rep=20):
    v = []
    for i in range(rep + 3):
        torch.cuda.synchronize()
        torch.cuda._sleep(4_000_000)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        if i >= 3:
            v.append(a.elapsed_time(b))
    return float(np.median(v))


def graph_of(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


@torch.no_grad()
def main(out):
    res = {"gpu": torch.cuda.get_device_name(0), "cudnn": torch.backends.cudnn.version(), "benchmark": torch.backends.cudnn.benchmark}
    net = RC.CompactNet(1.0).cuda().eval().to(memory_format=torch.channels_last)
    low = torch.rand(1, 3, W, W, device="cuda")
    high = torch.rand(1, 3, W, W, device="cuda")
    cover = torch.rand(9, W, W, device="cuda")
    normal = torch.rand(3, W, W, device="cuda")
    depth = torch.rand(W, W, device="cuda")
    ess = torch.full((1, 1, W, W), 4.0, device="cuda")
    st = {}
    st["features"] = gpu_ms(lambda: RC.features(low, high, cover[None], normal[None], depth[None, None], ess))
    x = RC.features(low, high, cover[None], normal[None], depth[None, None], ess)
    st["to_channels_last"] = gpu_ms(lambda: x.to(memory_format=torch.channels_last))
    xc = x.to(memory_format=torch.channels_last)

    def netf():
        with torch.autocast("cuda", dtype=torch.float16):
            return net(xc)
    st["network_autocast_cl"] = gpu_ms(netf)
    lg = netf()
    res["logits"] = {"dtype": str(lg.dtype), "channels_last": bool(lg.is_contiguous(memory_format=torch.channels_last)),
                     "contiguous": bool(lg.is_contiguous())}
    st["logits_contiguous"] = gpu_ms(lambda: lg.contiguous())
    st["spatial_fused"] = gpu_ms(lambda: RC.spatial_fused(low, high, lg))
    res["eager_steps_ms"] = st
    res["eager_sum_ms"] = sum(st.values())
    # partial graphs
    pg = {}
    g = graph_of(netf)
    pg["network_graph"] = gpu_ms(lambda: g.replay())
    g2 = graph_of(lambda: RC.spatial_fused(low, high, lg))
    pg["spatial_graph"] = gpu_ms(lambda: g2.replay())

    def body():
        xx = RC.features(low, high, cover[None], normal[None], depth[None, None], ess)
        with torch.autocast("cuda", dtype=torch.float16):
            l2 = net(xx.to(memory_format=torch.channels_last))
        lo, hi = RC.spatial_fused(low, high, l2)
        return lo + hi
    g3 = graph_of(body)
    pg["full_S_body_graph"] = gpu_ms(lambda: g3.replay())
    pg["full_S_body_eager"] = gpu_ms(body)
    torch.backends.cudnn.benchmark = True
    for _ in range(5):
        netf()
    pg["network_eager_benchmark"] = gpu_ms(netf)
    g4 = graph_of(netf)
    pg["network_graph_benchmark"] = gpu_ms(lambda: g4.replay())
    g5 = graph_of(body)
    pg["full_S_body_graph_benchmark"] = gpu_ms(lambda: g5.replay())
    res["graphs_ms"] = pg
    # per-layer device time (eager, autocast, channels_last)
    layers = {}
    hooks = []
    evs = {}

    def pre(name):
        def f(m, i):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            evs[name] = [e]
        return f

    def post(name):
        def f(m, i, o):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            evs[name].append(e)
        return f
    for n, m in net.named_children():
        hooks.append(m.register_forward_pre_hook(pre(n)))
        hooks.append(m.register_forward_hook(post(n)))
    torch.backends.cudnn.benchmark = False
    acc = {}
    for i in range(13):
        torch.cuda.synchronize()
        torch.cuda._sleep(4_000_000)
        netf()
        torch.cuda.synchronize()
        if i >= 3:
            for n, (a, b) in evs.items():
                acc.setdefault(n, []).append(a.elapsed_time(b))
    for h in hooks:
        h.remove()
    res["per_layer_ms"] = {n: float(np.median(v)) for n, v in acc.items()}
    C.write(out, res)
    for k in ("eager_steps_ms", "graphs_ms", "per_layer_ms", "logits"):
        print(k, res[k], flush=True)
    print("eager_sum", res["eager_sum_ms"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
