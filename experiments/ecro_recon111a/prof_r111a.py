"""WL111A Stage C cost check (development profiling, not final timing evidence): reconstruction-subsystem latency of R3,
S111 (CompactNet) and the P4 / S111A architecture (FullResNet), each under WL111's execution contract (CUDA graph; guides
acquired online). Same device and process, arms interleaved per round, synchronized wall time.

  static form   reconstruction only (guides already resident)
  camera form   guide acquisition for the current camera (sub 4 for CT-AK, 8 x 8 for R3) + reconstruction

python prof_r111a.py OUT_JSON [--rounds N]
"""
import argparse
import random
import time

import numpy as np

import common_r111a as A
import common_r111 as C
import engine_r111 as E
import recon_r111 as RC
import recon_r111a as RA
import r3_r111
import arms_r111 as AR
import seq_r111 as SQ
import stageA_r111a as SA

torch, R = E.torch, E.R
W = C.RES


def main(out, rounds):
    eng, audit = SA.setup()
    cam = E.cam_dict(SQ.wl105_cameras()["Q00"])
    spp = 4
    comp = eng.frame(cam, spp, A.seed("smoke", 1, 0, 0))
    low, high = RC.raw_low_high(comp)
    r3g = AR.R3Graph(r3_r111.load_r3())
    cands = {"S111": AR.load_candidate(A.wl111_weights(0), 1.0),
             "P4": RA.load_candidate(A.OUT / "models" / "P4_r0.pt", "fullres")}
    graphs = {k: RC.GraphCandidate(c, W * W, False) for k, c in cands.items()}
    g4, g8 = eng.pack(cam, 4, None, False), eng.guides(cam, 8)

    def run(arm, camera_form):
        if arm == "R3":
            g = eng.guides(cam, 8) if camera_form else g8
            return r3g(comp[None], r3_r111.aux14(g, spp))
        g = eng.pack(cam, 4, None, False) if camera_form else g4
        return graphs[arm].run(low, high, spp, g, cam, 1, R.PT.cam_basis)

    keys = [(a, f) for a in ("R3", "S111", "P4") for f in (False, True)]
    for k in keys:
        for _ in range(5):
            run(*k)
    E.sync()
    times = {f"{a}_{'camera' if f else 'static'}": [] for a, f in keys}
    rs = random.Random(17990011)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(rounds):
        order = list(keys)
        rs.shuffle(order)
        for a, f in order:
            E.sync()
            t0 = time.perf_counter()
            run(a, f)
            E.sync()
            times[f"{a}_{'camera' if f else 'static'}"].append(1e3 * (time.perf_counter() - t0))
    res = {"device": torch.cuda.get_device_name(0), "spp": spp, "rounds": rounds,
           "median_ms": {k: float(np.median(v)) for k, v in times.items()},
           "p95_ms": {k: float(np.percentile(v, 95)) for k, v in times.items()},
           "params": {"S111": RC.n_params(cands["S111"].net), "P4": RC.n_params(cands["P4"].net)},
           "peak_mem_mb": torch.cuda.max_memory_allocated() / 2 ** 20, "load1": E.load1(),
           "nvidia_smi": E.versions().get("nvidia_smi")}
    m = res["median_ms"]
    res["P4_over_R3_camera"] = m["P4_camera"] / m["R3_camera"]
    res["P4_minus_S111_static"] = m["P4_static"] - m["S111_static"]
    C.write(out, res)
    print("PROF", res, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--rounds", type=int, default=60)
    a = ap.parse_args()
    main(a.out, a.rounds)
