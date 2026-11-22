"""WL109 Stage G: dynamic-state qualification (only after the static decision passed).

python dynamic109.py OUT RUN_DIR DEV_JSON METRICS_JSON [--smoke]

Frozen edit: WL100 SSS test configuration 0 -> configuration 2 geometry (both boxes translate and rotate), light 0 of
configuration 0 unchanged (the first transition of the WL101 V_SSS_TOUR validation workload), camera V0. The standard ECRO
lifecycle (set_geom, selective update_state, source, respond) regenerates the table; both estimators consume its bytes.
D1 table identity (selective update == full relation rebuild == fresh construction, bitwise) and shared bytes
D2 replay identity and LOW identity on the edited state; frozen == eager for both arms after the edit
D3 no persistent C1 state: C1 frame on the updated runtime == C1 frame on a freshly constructed runtime (same seed)
D4 paired expectation on the edited state (B1 / B2 rules of the static analysis)
D5 equal-time 4-spp primary MSE ratio on the edited state against a fresh frozen-ECRO reference and a fresh R0 time curve
D6 costs: update latency (shared), C1-specific state rebuild (none), first and warm frames after the edit
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from common109 import P, LUMA, apply_smoke, array_sha, read, rois, seed_dynamic, seed_dynamic_reference, sha, write, HERE
import runtime109 as RT


def main(out, run_dir, dev_json, metrics_json, smoke):
    import torch
    from gpu109 import Engine, timing_schedule
    from analyze109 import interp_mse, sse, luma_means, Z
    if smoke:
        apply_smoke()
    met = read(metrics_json)
    assert smoke or met["decision"]["static_pass"], "dynamic stage requires the static pass"
    cfg = P["dynamic"]
    M = int(read(dev_json)["selected_M"])
    out.mkdir(parents=True, exist_ok=False)
    logf = open(out / "dyn.log", "a")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(time.strftime("%H:%M:%S"), s, flush=True); logf.write(s + "\n"); logf.flush()
    R, rt, Bd0, audit, r105 = RT.admit()
    import seq101
    rows = seq101._rows("sss")
    geom2 = rows[cfg["edit_to_config"]]["geom"]
    light = rows[0]["lights"][0]
    cams = RT.cameras()
    cam = cams[cfg["view"]]
    res = {"protocol_sha256": sha(HERE / "protocol109.json"), "smoke": smoke, "selected_M": M, "admission": audit,
           "metrics_sha256": sha(metrics_json), "executed": False}
    tol = P["synthetic"]["structural_rel_l1"]

    def tsync(fn):
        R.sync(); t0 = time.perf_counter(); v = fn(); R.sync()
        return v, 1e3 * (time.perf_counter() - t0)
    # ---- warm the timing programs on the static state (a renderer that was running before the edit)
    eng = Engine(R, rt, Bd0)
    slots = [("R0", g, 0) for g in cfg["grid_spp"]] + [("C1", 4, M)]
    for si, (kind, spp, m) in enumerate(slots):
        for w in range(1 + P["final"]["timing_warmup"]):
            eng.run_frozen(kind, "dyn", cam, spp, seed_dynamic(39, si * 10 + w, smoke), m)
    # ---- edit through the original lifecycle
    (moved, old), t_geom = tsync(lambda: rt.set_geom(geom2))
    n_dirty, t_state = tsync(lambda: rt.update_state(old, moved))
    _, t_light = tsync(lambda: rt.set_light(light))
    s, t_src = tsync(lambda: rt.source())
    (B2, _), t_resp = tsync(lambda: rt.respond(s))
    B2 = B2.cpu().numpy().copy()
    res["update_ms"] = {"set_geom": t_geom, "update_state": t_state, "set_light": t_light, "source": t_src, "respond": t_resp,
                        "total": t_geom + t_state + t_light + t_src + t_resp, "dirty_rays": int(n_dirty), "moved": moved}
    # D1 table identity
    rt.rebuild_state()
    B2f = rt.respond(rt.source())[0].cpu().numpy().copy()
    saved_cl = dict(R._CL)                         # rt101 keeps the resident closure table in a module global
    fresh = R.RT(geom2, "sss", rt.model, light)
    B2n = fresh.respond(fresh.source())[0].cpu().numpy().copy()
    R._CL.clear(); R._CL.update(saved_cl)          # restore it for the original runtime (frozen recording needs it)
    d1 = {"selective_eq_full_bitwise": bool(np.array_equal(B2, B2f)), "selective_eq_fresh_bitwise": bool(np.array_equal(B2, B2n)),
          "selective_vs_fresh_rel_l1": float(np.abs(B2 - B2n).sum() / np.abs(B2n).sum()), "table_sha256": array_sha(B2),
          "table_changed_from_static": array_sha(B2) != audit["table_bytes_sha256"]}
    d1["pass"] = bool(d1["selective_eq_full_bitwise"] and d1["selective_vs_fresh_rel_l1"] <= cfg["table_fresh_rel_l1_max"]
                      and d1["table_changed_from_static"])
    Bd2 = R.PT.fl(B2)
    eng.Bd = Bd2                                   # both arms consume the regenerated table bytes from here on
    eng_f = Engine(R, fresh, R.PT.fl(B2))
    log("D1", d1)
    # D2 replay / LOW identity / frozen == eager on the edited state
    d2 = []
    for i in range(2):
        seed = seed_dynamic(0, i, smoke)
        r0 = eng.r0(cam, 4, seed)
        c1 = eng.c1_frame(cam, 4, seed, M, orig_hl=True)
        rp = eng.c1_frame(cam, 4, seed, 1, orig_hl=True, replay=True)
        fz0 = eng.run_frozen("R0", "dyncheck", cam, 4, seed); fz0 = eng.run_frozen("R0", "dyncheck", cam, 4, seed)
        fz1 = eng.run_frozen("C1", "dyncheck", cam, 4, seed, M); fz1 = eng.run_frozen("C1", "dyncheck", cam, 4, seed, M)
        fz0 = torch.stack([o.torch() for o in fz0]).view(-1, 512, 512) / 4
        fz1 = torch.stack([o.torch() for o in fz1]).view(-1, 512, 512) / 4
        e1 = eng.c1_frame(cam, 4, seed, M, orig_hl=False)
        rl = lambda a, b: float((a - b).abs().sum() / max(float(b.abs().sum()), 1e-30))
        row = {"low": rl(c1[0:6], r0[0:6]), "ho": rl(c1[9:12], r0[6:9]), "replay_high": rl(rp[6:9], r0[6:9]),
               "r0_frozen_eager": rl(fz0, r0), "c1_frozen_eager": rl(fz1, e1), "replay_bitwise": bool(torch.equal(rp[6:9], r0[6:9]))}
        row["pass"] = all(row[k] <= tol for k in ("low", "ho", "replay_high", "r0_frozen_eager", "c1_frozen_eager"))
        d2.append(row)
    # D3 fresh construction equality
    seed = seed_dynamic(1, 0, smoke)
    a = eng.c1_frame(cam, 4, seed, M, orig_hl=True)
    b = eng_f.c1_frame(cam, 4, seed, M, orig_hl=True)
    d3 = {"rel_l1": float((a - b).abs().sum() / b.abs().sum()), "bitwise": bool(torch.equal(a, b))}
    d3["pass"] = d3["rel_l1"] <= tol
    del eng_f, fresh
    torch.cuda.empty_cache()
    log("D2", d2, "D3", d3)
    # edited-state ROIs and reference
    guides = {**r105.aux_buffers(R, rt, cam), **r105.centre_gbuffer(R, rt, cam)}
    masks = rois(guides)
    blocks = []
    for k in range(cfg["reference_blocks"]):
        blocks.append(eng.r0_blocks(cam, cfg["reference_block_spp"], seed_dynamic_reference(k, smoke), b=min(32, cfg["reference_block_spp"])))
    comp = np.stack(blocks)
    refc = comp.mean(0)
    ref = refc[0:3] + refc[3:6] + refc[6:9]
    vref = (comp[:, 0:3] + comp[:, 3:6] + comp[:, 6:9]).var(0, ddof=1) / comp.shape[0]
    np.save(out / "reference_mean.npy", refc.astype(np.float32))
    res["reference"] = {"blocks": int(comp.shape[0]), "spp_per_block": cfg["reference_block_spp"], "sha256": sha(out / "reference_mean.npy")}
    del comp, blocks
    # D4 paired frames and D5 grid frames (raw kept)
    Rn = cfg["replicates"]
    raw = out / "raw"; raw.mkdir()
    L, HO, HC = [], [], []
    for r in range(Rn):
        seed = seed_dynamic(2, r, smoke)
        r0 = eng.r0(cam, 4, seed); c1 = eng.c1_frame(cam, 4, seed, M, orig_hl=True)
        np.save(raw / f"paired_r{r:02d}_R0.npy", r0.cpu().numpy()); np.save(raw / f"paired_r{r:02d}_C1H.npy", c1[6:9].cpu().numpy())
        r0 = r0.double().cpu().numpy(); c1 = c1.double().cpu().numpy()
        L.append(r0[0:3] + r0[3:6]); HO.append(r0[6:9]); HC.append(c1[6:9])
    L, HO, HC = np.stack(L), np.stack(HO), np.stack(HC)
    grid = {}
    for gi, g in enumerate(cfg["grid_spp"]):
        frames = []
        for r in range(Rn):
            f = eng.r0(cam, g, seed_dynamic(10 + gi, r, smoke)).cpu().numpy()
            np.save(raw / f"grid_spp{g:03d}_r{r:02d}.npy", f)
            frames.append((f[0:3] + f[3:6] + f[6:9]).astype(np.float64))
        grid[g] = np.stack(frames)
    # timing after the edit (first frame = first call of the pre-edit-warmed program after the scene change), exclusive GPU
    def block():
        tim = {"first_after_edit_ms": {}, "frame_ms": {}}
        for si, (kind, spp, m) in enumerate(slots):
            key = f"{kind}_{spp}"
            R.sync(); t0 = time.perf_counter()
            eng.run_frozen(kind, "dyn", cam, spp, seed_dynamic(40, si, smoke), m)
            R.sync(); tim["first_after_edit_ms"][key] = 1e3 * (time.perf_counter() - t0)
            for w in range(P["final"]["timing_warmup"]):
                eng.run_frozen(kind, "dyn", cam, spp, seed_dynamic(41, si * 10 + w, smoke), m)
            tim["frame_ms"][key] = []
        for rr, (kind, spp, m) in timing_schedule(slots, P["final"]["timing_rounds"], P["seeds"]["timing_order"] + 77):
            si = slots.index((kind, spp, m))
            tim["frame_ms"][f"{kind}_{spp}"].append(eng.timed(kind, "dyn", cam, spp, seed_dynamic(50 + si, rr, smoke), m))
        return tim
    tim, excl = RT.exclusive_block(block, log, require=not smoke)
    tim["exclusivity"] = excl
    res["timing"] = tim
    # D4 / D5 statistics
    A = P["analysis"]
    pm = masks["primary"]
    vr = float(vref[:, pm].mean())
    s1, s0 = sse(L + HC, ref, pm), sse(L + HO, ref, pm)
    mse1 = float(s1.mean() - vr)
    tg = np.array([np.median(tim["frame_ms"][f"R0_{g}"]) for g in cfg["grid_spp"]])
    mg = np.array([sse(grid[g], ref, pm).mean() - vr for g in cfg["grid_spp"]])
    tc = float(np.median(tim["frame_ms"]["C1_4"]))
    meq, flag = interp_mse(tc, tg, mg)
    rs = np.random.default_rng(A["bootstrap_seed"] + 900)
    Bn = A["bootstrap_draws"]
    sg = np.stack([sse(grid[g], ref, pm) for g in cfg["grid_spp"]])
    tgs = np.stack([np.asarray(tim["frame_ms"][f"R0_{g}"]) for g in cfg["grid_spp"]])
    tcs = np.asarray(tim["frame_ms"]["C1_4"])
    from analyze109 import interp_vec
    ip = rs.integers(0, Rn, (Bn, Rn)); ig = rs.integers(0, Rn, (Bn,) + sg.shape); it = rs.integers(0, tgs.shape[1], (Bn,) + tgs.shape)
    ic = rs.integers(0, len(tcs), (Bn, len(tcs)))
    b1 = s1[ip].mean(1) - vr
    bg = np.take_along_axis(np.broadcast_to(sg, (Bn,) + sg.shape), ig, 2).mean(2) - vr
    btg = np.median(np.take_along_axis(np.broadcast_to(tgs, (Bn,) + tgs.shape), it, 2), 2)
    beq = interp_vec(np.median(tcs[ic], 1), btg, bg)
    lr = np.log(b1 / beq)
    ratio = mse1 / meq if meq == meq else float("nan")
    se = float(np.nanstd(lr, ddof=1))
    d5 = {"mse_C1": mse1, "mse_R0_equal_time": meq, "interp_flag": flag, "t_C1_ms": tc, "t_R0_4_ms": float(tg[0]),
          "equal_time_ratio": ratio, "ci95": [ratio * math.exp(-Z * se), ratio * math.exp(Z * se)],
          "equal_spp_ratio": float(mse1 / (s0.mean() - vr))}
    d5["pass"] = bool(flag == "ok" and ratio <= cfg["benefit_ratio_max"] and d5["ci95"][1] < cfg["benefit_upper95_below"])
    # D4 bias rules (B1 / B2) on the edited state
    d4 = {}
    tau = A["bias_tolerance_relative"]
    for roi in P["rois"]["bias"]:
        m = masks[roi]
        if m.sum() == 0:
            continue
        d = luma_means(HC - HO, m)
        refl = float((LUMA @ ref[:, m]).mean())
        rel, rse = float(d.mean() / refl), float(d.std(ddof=1) / math.sqrt(Rn) / abs(refl))
        lo, hi = rel - Z * rse, rel + Z * rse
        state = "EQUIVALENT" if (lo >= -tau and hi <= tau) else ("BIASED" if (lo > tau or hi < -tau) else "UNRESOLVED")
        import stats109 as ST
        scale = float((L + HO)[:, :, m].var(0, ddof=1).mean())
        Qd = ST.spatial_bias_power(torch.as_tensor(HC - HO, device="cuda"), m, scale, A["Q_groups"])
        q, qup = Qd["point"], Qd["upper95"]
        qlim = A["spatial_bias_power_max"][roi]
        d4[roi] = {"relative": rel, "ci95": [lo, hi], "state": state, "Q": q, "Q_upper95": qup, "Q_pass": bool(qup <= qlim)}
    d4_pass = all(v["state"] == "EQUIVALENT" and v["Q_pass"] for v in d4.values())
    res.update({"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D4_pass": bool(d4_pass), "D5": d5})
    res["c1_persistent_state"] = "none: C1 holds no buffers across frames; its inner stream is seeded per frame and reads the current scene and table"
    res["correctness_pass"] = bool(d1["pass"] and all(r["pass"] for r in d2) and d3["pass"] and d4_pass)
    res["benefit_pass"] = d5["pass"]
    res["table_sha_after"] = RT.table_bytes_sha(Bd2)
    assert res["table_sha_after"] == d1["table_sha256"]
    res["executed"] = True
    write(out / "dyn.json", res)
    log("DYNAMIC correctness", res["correctness_pass"], "benefit", res["benefit_pass"], d5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("run"); ap.add_argument("dev_json"); ap.add_argument("metrics_json")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(Path(a.out).resolve(), Path(a.run).resolve(), Path(a.dev_json).resolve(), Path(a.metrics_json).resolve(), a.smoke)
