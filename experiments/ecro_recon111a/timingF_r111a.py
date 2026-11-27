"""WL111A final timing evidence (protocol/recon111a.json "timing"): one process on the timing device under the guard, no
other WL111A process. Arms R0, R3 (CUDA graph, 8 x 8 guides), S111 (r0 network; S-mode graph, sub-4 guides) and S111A
(r0 network; same execution). One network per architecture: both realizations share the timing of their architecture.

  static      per spp: warm-up, then each of 30 rounds times one complete current frame of every arm in a shuffled order
              (ECRO frame + guide acquisition when the camera / transport revision changed (cached otherwise) +
              reconstruction to resident HDR output), synchronized wall time
  edit        per spp: each timed frame first applies a geometry update alternating cfg0 <-> cfg1 (renderer set_state,
              inside E2E); the revision change forces guide acquisition, then frame and reconstruction
  components  at 4 spp, interleaved per round, synchronized spans: geometry update, ECRO frame, guide acquisition (fresh),
              reconstruction (guides resident); torch-allocator peak per arm
  conditions  nvidia-smi clocks / power / temperature / throttle reasons and load1 before and after every block
  futility    Stage C: S111A camera-form subsystem (sub-4 guide acquisition + reconstruction) >= 0.70 x R3's (8 x 8 guides +
              reconstruction), or static reconstruction overhead (S111A - R0 static E2E at 4 spp) > 4 ms -> STOP (exit 3)

python timingF_r111a.py [--smoke]     -> OUT/final/timing/<block>.json, OUT/final/timing/futility.json
"""
import argparse
import random
import subprocess
import sys
import time

import numpy as np

import common_r111a as A
import common_r111 as C
import engine_r111 as E
import recon_r111a as RA
import r3_r111
import arms_r111 as AR
import seq_r111 as SQ
import stageA_r111a as SA

torch, R = E.torch, E.R
W = C.RES
P = C.read(A.HERE / "protocol" / "recon111a.json")
USE = ("R0", "R3", "S111", "S111A")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def conditions():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,clocks.sm,clocks.mem,clocks.max.sm,power.draw,power.limit,"
                        "temperature.gpu,utilization.gpu,memory.used,clocks_throttle_reasons.active",
                        "--format=csv,noheader"], capture_output=True, text=True, timeout=60).stdout.strip()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True, timeout=60).stdout.strip()
    return {"t": time.time(), "nvidia_smi": q, "compute_apps": apps, "load1": E.load1()}


def build_arms():
    dsg = C.read(A.WL111_EVID / "design.json")
    assert dsg["sub"] == 4
    arms = {"R0": AR.R0Arm(), "R3": AR.R3Arm(r3_r111.load_r3(), graph=True),
            "S111": AR.CandidateArm(AR.load_candidate(A.wl111_weights(0), dsg["m"]), R.PT.cam_basis, False, "S111"),
            "S111A": AR.CandidateArm(RA.load_candidate(A.OUT / "models" / "S111A_r0.pt", "fullres"), R.PT.cam_basis, False, "S111A")}
    return arms, dsg["sub"]


def main(smoke):
    out = A.OUT / ("final_smoke" if smoke else "final") / "timing"
    out.mkdir(parents=True, exist_ok=True)
    eng, audit = SA.setup()
    C.write(out / "admission_timing.json", audit)
    arms, sub = build_arms()
    T = P["timing"]
    warm, rounds = (1, 3) if smoke else (C.P["timing"]["warmup"], 30)
    grid = [1, 4] if smoke else P["quality_protocol"]["spp_grid"]
    rows = SQ.rows()
    s0 = (rows[0]["geom"], rows[0]["lights"][0])
    s1 = (rows[1]["geom"], rows[0]["lights"][0])
    v0 = E.cam_dict(C.P["views"]["V0"])
    fam = "smoke" if smoke else "timing"
    sd = lambda reg, slot, i: A.seed(fam, reg, slot, i % 1000)
    sync = E.sync

    def static_block():
        rs = random.Random(A.seed(fam, 9, 0, 0))
        res = {}
        eng.set_state(*s0)
        for si, spp in enumerate(grid):
            caches = {a: {} for a in USE}
            times = {a: [] for a in USE}
            for i in range(warm + rounds):
                order = list(USE)
                rs.shuffle(order)
                for a in order:
                    sync()
                    t0 = time.perf_counter()
                    comp = eng.frame(v0, spp, sd(0, si, i))
                    arms[a](AR.Frame(eng, comp, v0, spp, eng.revision, v0, False, sub), caches[a])
                    sync()
                    if i >= warm:
                        times[a].append(1e3 * (time.perf_counter() - t0))
            for a in USE:
                res[f"{a}_{spp}"] = times[a]
            res[f"load1_after_{spp}"] = E.load1()
            log("static", spp, {a: round(float(np.median(times[a])), 2) for a in USE})
        return res

    def edit_block():
        rs = random.Random(A.seed(fam, 9, 1, 0))
        res = {}
        eng.set_state(*s0)
        flip = [0]
        for si, spp in enumerate(grid):
            times = {a: [] for a in USE}
            upd = {a: [] for a in USE}
            for i in range(warm + rounds):
                order = list(USE)
                rs.shuffle(order)
                for a in order:
                    sync()
                    t0 = time.perf_counter()
                    flip[0] ^= 1
                    eng.set_state(*(s1 if flip[0] else s0))
                    sync()
                    t1 = time.perf_counter()
                    comp = eng.frame(v0, spp, sd(1, si, i))
                    arms[a](AR.Frame(eng, comp, v0, spp, eng.revision, v0, True, sub), {})
                    sync()
                    if i >= warm:
                        times[a].append(1e3 * (time.perf_counter() - t0))
                        upd[a].append(1e3 * (t1 - t0))
            for a in USE:
                res[f"{a}_{spp}"] = times[a]
                res[f"{a}_{spp}_update_part"] = upd[a]
            res[f"load1_after_{spp}"] = E.load1()
            log("edit", spp, {a: round(float(np.median(times[a])), 2) for a in USE})
        eng.set_state(*s0)
        return res

    def components_block():
        """4 spp, synchronized spans per arm and round (interleaved, shuffled): update (set_state flip), frame, guides
        (fresh acquisition of the arm's own guides), reconstruction (guides resident)."""
        rs = random.Random(A.seed(fam, 9, 2, 0))
        spp = 4
        spans = {a: {"update": [], "frame": [], "guides": [], "recon": []} for a in USE}
        flip = 0
        for i in range(warm + rounds):
            order = list(USE)
            rs.shuffle(order)
            for a in order:
                sync()
                t0 = time.perf_counter()
                flip ^= 1
                eng.set_state(*(s1 if flip else s0))
                sync()
                t1 = time.perf_counter()
                comp = eng.frame(v0, spp, sd(2, 0, i))
                sync()
                t2 = time.perf_counter()
                frm = AR.Frame(eng, comp, v0, spp, eng.revision, v0, True, sub)
                cache = {}
                if a == "R3":
                    frm.pack(8, False, cache)
                elif a != "R0":
                    frm.pack(sub, False, cache)
                sync()
                t3 = time.perf_counter()
                arms[a](frm, cache)
                sync()
                t4 = time.perf_counter()
                if i >= warm:
                    for k, x in (("update", t1 - t0), ("frame", t2 - t1), ("guides", t3 - t2), ("recon", t4 - t3)):
                        spans[a][k].append(1e3 * x)
        eng.set_state(*s0)
        peak = {}
        for a in USE:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            for i in range(3):
                comp = eng.frame(v0, spp, sd(2, 1, i))
                arms[a](AR.Frame(eng, comp, v0, spp, eng.revision, v0, False, sub), {})
            torch.cuda.synchronize()
            peak[a] = {"peak_allocated_mb": torch.cuda.max_memory_allocated() / 2 ** 20, "resident_mb": base / 2 ** 20}
        med = {a: {k: float(np.median(v)) for k, v in s.items()} for a, s in spans.items()}
        return {"spans_ms": spans, "median_ms": med, "torch_memory": peak,
                "params": {"R3": r3_r111.n_params(arms["R3"].model), "S111": RA.RC.n_params(arms["S111"].cand.net),
                           "S111A": RA.RC.n_params(arms["S111A"].cand.net)}}

    blocks = {"static": static_block, "edit": edit_block, "components": components_block}
    res = {}
    for name, fn in blocks.items():
        f = out / f"{name}.json"
        if f.exists():
            res[name] = C.read(f)
            continue
        pre = conditions()
        if smoke:
            r, ex = fn(), []
        else:
            r, ex = E.timed_block(fn, name, log)
        post = conditions()
        rec = {"times_ms" if name != "components" else "components": r, "exclusivity": ex, "conditions": {"pre": pre, "post": post},
               "environment": E.versions(), "device": torch.cuda.get_device_name(0)}
        C.write(f, rec)
        res[name] = rec
        log("timing", name, "done", pre["nvidia_smi"], "|", post["nvidia_smi"])
    # cost futility (Stage C declaration)
    med = res["components"]["components"]["median_ms"]
    cam = {a: med[a]["guides"] + med[a]["recon"] for a in ("R3", "S111", "S111A")}
    st = res["static"]["times_ms"]
    over = float(np.median(st["S111A_4"]) - np.median(st["R0_4"]))
    fut = {"camera_form_ms": cam, "S111A_over_R3_camera_form": cam["S111A"] / cam["R3"],
           "S111A_static_overhead_ms": over, "limits": {"camera_form_ratio": 0.70, "static_overhead_ms": 4.0}}
    fut["stop"] = bool(fut["S111A_over_R3_camera_form"] >= 0.70 or over > 4.0)
    C.write(out / "futility.json", fut)
    log("FUTILITY", fut)
    if fut["stop"] and not smoke:
        print("FUTILITY_STOP", flush=True)
        sys.exit(3)
    print("TIMINGF_COMPLETE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.smoke)
