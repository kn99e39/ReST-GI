"""Diagnostic (not evidence; owner request 2026-11-24): where does the fixed per-call overhead of the guide pack / candidate
come from? Run on LabServer70's RTX 3080 Ti (GPU1) in the same container stack, never on the evidence GPU.

python dbg_overhead_r111.py OUT_JSON
"""
import json
import os
import sys
import time

import numpy as np

import engine_r111 as E
import recon_r111 as RC
import common_r111 as C

torch, dr, mi, R = E.torch, E.dr, E.mi, E.R
F = mi.Float
W = C.RES
N = W * W
REP = 30


def cpu_stat():
    for p in ("/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu,cpuacct/cpu.stat", "/sys/fs/cgroup/cpu/cpu.stat"):
        if os.path.exists(p):
            return {l.split()[0]: int(l.split()[1]) for l in open(p).read().splitlines() if len(l.split()) == 2}
    return {}


def med_ms(fn, rep=REP, warm=5):
    for _ in range(warm):
        fn()
    E.sync()
    v = []
    for _ in range(rep):
        E.sync()
        t0 = time.perf_counter()
        fn()
        E.sync()
        v.append(1e3 * (time.perf_counter() - t0))
    return float(np.median(v))


def main(out):
    res = {"cpus_env": os.environ.get("WL111_CPUS"), "nproc_visible": os.cpu_count(), "load1": E.load1(),
           "cgroup_before": cpu_stat(), "gpu": torch.cuda.get_device_name(0)}
    rt, Bd, B, audit = E.admit(lambda *a: None, replay=False)
    eng = E.Engine(rt, Bd, B)
    v0 = E.cam_dict(C.P["views"]["V0"])
    camb = dict(v0, origin=[v0["origin"][0] + 0.08] + v0["origin"][1:])
    # ---- A: pack call decomposition (wall, synchronized)
    A = {}
    A["cam_scalars_create_ms"] = med_ms(lambda: E.cam_array(v0, camb))
    for sub in (1, 2, 4, 8):
        key = ("pack", sub, True)
        eng.pack(v0, sub, camb, True)                      # record
        fz = eng._f[key]
        ca = E.cam_array(v0, camb)

        def call_only():
            out_ = fz(rt.scene, rt.tabs(), ca)
            dr.eval(out_)
        A[f"sub{sub}_frozen_call_ms"] = med_ms(call_only)
        A[f"sub{sub}_pack_total_ms"] = med_ms(lambda: eng.pack(v0, sub, camb, True))
        A[f"sub{sub}_tabs_ms"] = med_ms(lambda: rt.tabs())
    eng.frame_raw(v0, 1, 1)
    A["frame1_total_ms"] = med_ms(lambda: eng.frame_raw(v0, 1, 7))
    rt._frames = {}
    A["frame1_original_ms"] = med_ms(lambda: dr.eval(rt.frame(eng.Bd, v0, W, 1, 7)))
    res["A_pack_decomposition"] = A
    print("A", A, flush=True)
    # ---- B / C: pure GPU time vs CPU enqueue time per component (no CUPTI: it crashed with Dr.Jit / OptiX here).
    # GPU time: a 2 ms device sleep is queued first, so the CPU enqueues the start event, the work and the end event while
    # the GPU is still busy; the event pair then measures the work's device duration only.
    # CPU time: wall time to enqueue the work (no synchronization), i.e. the host cost per frame.
    def gpu_cpu(fn, rep=REP):
        gpu, cpu = [], []
        for i in range(rep + 3):
            E.sync()
            torch.cuda._sleep(4_000_000)
            s0, s1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s0.record()
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            s1.record()
            E.sync()
            if i >= 3:
                gpu.append(s0.elapsed_time(s1))
                cpu.append(1e3 * (t1 - t0))
        return {"gpu_ms": float(np.median(gpu)), "cpu_enqueue_ms": float(np.median(cpu))}
    Cc = {}
    x = torch.zeros(1, device="cuda")
    Cc["torch_tiny_op_cpu_us"] = 1e3 * gpu_cpu(lambda: [x.add_(1) for _ in range(200)], rep=10)["cpu_enqueue_ms"] / 200
    Cc["dr_opaque_scalar_cpu_us"] = 1e3 * gpu_cpu(lambda: [dr.opaque(F, 1.5) for _ in range(100)], rep=10)["cpu_enqueue_ms"] / 100
    Cc["pack_sub4"] = gpu_cpu(lambda: eng.pack(v0, 4, camb, True))
    Cc["pack_sub8_no_centre"] = gpu_cpu(lambda: eng.guides(v0, 8))
    Cc["cam_scalars"] = gpu_cpu(lambda: E.cam_array(v0, camb))
    Cc["frame_1spp"] = gpu_cpu(lambda: eng.frame_raw(v0, 1, 7))
    Cc["frame_4spp"] = gpu_cpu(lambda: eng.frame_raw(v0, 4, 7))
    lo = torch.rand(1, 3, W, W, device="cuda")
    lg = torch.zeros(1, 150, W, W, device="cuda", dtype=torch.float16)
    Cc["spatial_3pass_eager"] = gpu_cpu(lambda: RC.spatial_fused(lo, lo, lg))
    net = RC.CompactNet(1.0)
    cand = RC.Candidate(net, C.P["candidate"]["temporal_settings"])
    xin = torch.zeros(1, RC.CIN, W, W, device="cuda").to(memory_format=torch.channels_last)

    def netf():
        with torch.autocast("cuda", dtype=torch.float16):
            cand.net(xin)
    with torch.no_grad():
        Cc["network_fp16_eager"] = gpu_cpu(netf)
        p = eng.pack(v0, 4, camb, True)
        gp = RC.GraphCandidate(cand, N, True)
        gp.run(lo[0], lo[0], 4, p, v0, 1, R.PT.cam_basis)
        Cc["graph_run_with_copies"] = gpu_cpu(lambda: gp.run(lo[0], lo[0], 4, p, v0, 1, R.PT.cam_basis))
        for mode in ("static", "moving", "S"):
            g = gp.graphs[mode]
            Cc[f"graph_replay_only_{mode}"] = gpu_cpu(lambda: g.replay())
        import r3_r111
        import arms_r111 as AR
        r3 = r3_r111.load_r3()
        comp = eng.frame(v0, 4, 7)[None]
        aux = r3_r111.aux14(eng.guides(v0, 8), 4)
        r3g = AR.R3Graph(r3)
        Cc["r3_graph_replay_only"] = gpu_cpu(lambda: r3g.g.replay())
        Cc["r3_eager_forward"] = gpu_cpu(lambda: r3_r111.forward(r3, comp, aux))
    res["C_gpu_vs_cpu"] = Cc
    for k, v in Cc.items():
        print("C", k, v, flush=True)
    res["cgroup_after"] = cpu_stat()
    res["load1_end"] = E.load1()
    C.write(out, res)
    print("DBG_COMPLETE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
