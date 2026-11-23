"""WL109A Stage F: dynamic closure on the frozen WL109 P7 edit (only after the WL109A static pass; run under guard109.py).

python dynamic109a.py OUT STATIC_JSON BUDGET_JSON [--smoke]

D1-D3, D5 and D6 repeat dynamic109.py (statements copied unchanged; WL109A seed family). D4 uses the qualified replicate
budget R (paired 4-spp frames) with the WL109 B1 / Q definitions and limits and the WL109A three-way Q state; the paired
frames are accumulated (sums of D and D^2 per 16-frame subgroup, per-frame mask statistics) instead of stored raw. D5 uses
the original 32-replicate R0 grid at 4-8 spp and the fresh 16 x 8192-spp reference; the C1 and nominal R0 squared errors
use all R paired frames.
"""
import argparse
import math
from pathlib import Path
import time

import numpy as np

from common109a import (P, P9, LUMA, QLIM, b1_state, q_from_group_sums, q_state, q_summary, read, sha, write, seed_dynamic,
                        seed_dynamic_reference, HERE, W109)
import common109 as C9
import runtime109 as RT


def main(out, static_json, budget_json, smoke):
    import torch
    from gpu109 import Engine, timing_schedule
    from analyze109 import interp_mse, interp_vec, Z
    from capture109a import Masks
    st = read(static_json)
    assert smoke or st["final_static_verdict"] == "STATIC_PASS", "dynamic stage requires the WL109A static pass"
    R = int(read(budget_json)["decision"]["R"]) if not smoke else 32
    cfg = P9["dynamic"]
    out.mkdir(parents=True, exist_ok=False)
    logf = open(out / "dyn.log", "a")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(time.strftime("%H:%M:%S"), s, flush=True); logf.write(s + "\n"); logf.flush()
    ev9 = Path.home() / P9["environment"]["remote_root"] / "results/ecro_estimator109/evidence"
    M = int(read(ev9 / "dev/dev.json")["selected_M"]); assert M == 1
    table0 = read(ev9 / "run/manifest.json")["table_sha_after"]
    Rr, rt, Bd0, audit, r105 = RT.admit()
    assert audit["table_bytes_sha256"] == table0
    import seq101
    rows = seq101._rows("sss")
    geom2 = rows[cfg["edit_to_config"]]["geom"]
    light = rows[0]["lights"][0]
    cams = RT.cameras()
    cam = cams[cfg["view"]]
    res = {"protocol109a_sha256": sha(HERE / "protocol109a.json"), "smoke": smoke, "M": M, "R": R, "admission": audit,
           "static_json_sha256": sha(static_json), "executed": False}
    tol = P9["synthetic"]["structural_rel_l1"]

    def tsync(fn):
        Rr.sync(); t0 = time.perf_counter(); v = fn(); Rr.sync()
        return v, 1e3 * (time.perf_counter() - t0)
    eng = Engine(Rr, rt, Bd0)
    slots = [("R0", g, 0) for g in cfg["grid_spp"]] + [("C1", 4, M)]
    for si, (kind, spp, m) in enumerate(slots):
        for w in range(1 + P9["final"]["timing_warmup"]):
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
    saved_cl = dict(Rr._CL)
    fresh = Rr.RT(geom2, "sss", rt.model, light)
    B2n = fresh.respond(fresh.source())[0].cpu().numpy().copy()
    Rr._CL.clear(); Rr._CL.update(saved_cl)
    d1 = {"selective_eq_full_bitwise": bool(np.array_equal(B2, B2f)), "selective_eq_fresh_bitwise": bool(np.array_equal(B2, B2n)),
          "selective_vs_fresh_rel_l1": float(np.abs(B2 - B2n).sum() / np.abs(B2n).sum()), "table_sha256": C9.array_sha(B2),
          "table_changed_from_static": C9.array_sha(B2) != audit["table_bytes_sha256"]}
    d1["pass"] = bool(d1["selective_eq_full_bitwise"] and d1["selective_vs_fresh_rel_l1"] <= cfg["table_fresh_rel_l1_max"]
                      and d1["table_changed_from_static"])
    Bd2 = Rr.PT.fl(B2)
    eng.Bd = Bd2
    eng_f = Engine(Rr, fresh, Rr.PT.fl(B2))
    log("D1", d1)
    # D2
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
    # D3
    seed = seed_dynamic(1, 0, smoke)
    a = eng.c1_frame(cam, 4, seed, M, orig_hl=True)
    b = eng_f.c1_frame(cam, 4, seed, M, orig_hl=True)
    d3 = {"rel_l1": float((a - b).abs().sum() / b.abs().sum()), "bitwise": bool(torch.equal(a, b))}
    d3["pass"] = d3["rel_l1"] <= tol
    del eng_f, fresh
    torch.cuda.empty_cache()
    log("D2", d2, "D3", d3)
    # edited-state ROIs and fresh reference
    guides = {**r105.aux_buffers(Rr, rt, cam), **r105.centre_gbuffer(Rr, rt, cam)}
    masks = C9.rois(guides)
    masks = {k: masks[k] for k in P9["rois"]["bias"] if masks[k].sum() > 0}
    MM = Masks(masks, torch)
    nb, bspp = (cfg["reference_blocks"], cfg["reference_block_spp"]) if not smoke else (2, 64)
    blocks = [eng.r0_blocks(cam, bspp, seed_dynamic_reference(k, smoke), b=min(32, bspp)) for k in range(nb)]
    comp = np.stack(blocks)
    refc = comp.mean(0)
    ref = refc[0:3] + refc[3:6] + refc[6:9]
    vref = (comp[:, 0:3] + comp[:, 3:6] + comp[:, 6:9]).var(0, ddof=1) / comp.shape[0]
    np.save(out / "reference_mean.npy", refc.astype(np.float32))
    res["reference"] = {"blocks": int(comp.shape[0]), "spp_per_block": bspp, "sha256": sha(out / "reference_mean.npy")}
    del comp, blocks
    refg = torch.as_tensor(ref.reshape(3, -1), device="cuda")
    # D4 / D5 paired frames (accumulated)
    sub = 16
    nsub = R // sub
    z = lambda: torch.zeros((3, 512 * 512), dtype=torch.float64, device="cuda")
    SG1 = np.empty((nsub, 3, 512 * 512)); SG2 = np.empty_like(SG1)
    sL, sH, sLL, sHH, sLH = z(), z(), z(), z(), z()
    lumD = np.empty((R, len(MM.names))); sse0 = np.empty(R); sse1 = np.empty(R)
    pm = masks["primary"]
    pmi = torch.as_tensor(np.flatnonzero(pm.reshape(-1)), device="cuda")
    ident_max = 0.0
    for r in range(R):
        if r % sub == 0:
            d1s, d2s = z(), z()
        seed = seed_dynamic(2, r, smoke)
        r0 = eng.r0(cam, 4, seed); c1 = eng.c1_frame(cam, 4, seed, M, orig_hl=True)
        ident_max = max(ident_max, float((c1[0:6] - r0[0:6]).abs().sum() / r0[0:6].abs().sum()))
        f = r0.double().view(9, -1); C = c1[6:9].double().view(3, -1)
        L = f[0:3] + f[3:6]; H = f[6:9]; D = C - H
        d1s += D; d2s += D * D
        sL += L; sH += H; sLL += L * L; sHH += H * H; sLH += L * H
        lumD[r] = MM.mean_luma(D).cpu().numpy()
        sse0[r] = float(((L + H - refg)[:, pmi] ** 2).mean()); sse1[r] = float(((L + C - refg)[:, pmi] ** 2).mean())
        if r % sub == sub - 1:
            SG1[r // sub] = d1s.cpu().numpy(); SG2[r // sub] = d2s.cpu().numpy()
    assert ident_max <= tol, ident_max
    vT = ((sLL - sL * sL / R) + (sHH - sH * sH / R) + 2 * (sLH - sL * sH / R)) / (R - 1)
    vT = vT.cpu().numpy()
    np.savez(out / "paired_accumulators.npz", SG1=SG1, SG2=SG2, varT0=vT, lumD=lumD, sse0=sse0, sse1=sse1, names=np.asarray(MM.names))
    gspp = cfg["grid_spp"]
    Rg = cfg["replicates"] if not smoke else 4
    grid = np.empty((len(gspp), Rg))
    for gi, g in enumerate(gspp):
        for r in range(Rg):
            f = eng.r0(cam, g, seed_dynamic(10 + gi, r, smoke)).double().view(9, -1)
            grid[gi, r] = float(((f[0:3] + f[3:6] + f[6:9] - refg)[:, pmi] ** 2).mean())
    np.save(out / "grid_sse.npy", grid)

    def block():
        tim = {"first_after_edit_ms": {}, "frame_ms": {}}
        for si, (kind, spp, m) in enumerate(slots):
            key = f"{kind}_{spp}"
            Rr.sync(); t0 = time.perf_counter()
            eng.run_frozen(kind, "dyn", cam, spp, seed_dynamic(40, si, smoke), m)
            Rr.sync(); tim["first_after_edit_ms"][key] = 1e3 * (time.perf_counter() - t0)
            for w in range(P9["final"]["timing_warmup"]):
                eng.run_frozen(kind, "dyn", cam, spp, seed_dynamic(41, si * 10 + w, smoke), m)
            tim["frame_ms"][key] = []
        for rr, (kind, spp, m) in timing_schedule(slots, P9["final"]["timing_rounds"] if not smoke else 2, P["seeds"]["timing_order"] + 77):
            si = slots.index((kind, spp, m))
            tim["frame_ms"][f"{kind}_{spp}"].append(eng.timed(kind, "dyn", cam, spp, seed_dynamic(50 + si, rr, smoke), m))
        return tim
    tim, excl = RT.exclusive_block(block, log, require=not smoke)
    tim["exclusivity"] = excl
    tim["watermark_bytes"] = {f"{k}_{s}": eng.watermark(k, "dyn", cam, s, seed_dynamic(60, i, smoke), m) for i, (k, s, m) in enumerate(slots)}
    res["timing"] = tim
    # D5
    A = P9["analysis"]
    vr = float(vref[:, pm].mean())
    mse1 = float(sse1.mean() - vr)
    tg = np.array([np.median(tim["frame_ms"][f"R0_{g}"]) for g in gspp])
    mg = grid.mean(1) - vr
    tc = float(np.median(tim["frame_ms"]["C1_4"]))
    meq, flag = interp_mse(tc, tg, mg)
    rs = np.random.default_rng(P["seeds"]["bootstrap"] + 900)
    Bn = A["bootstrap_draws"]
    tgs = np.stack([np.asarray(tim["frame_ms"][f"R0_{g}"]) for g in gspp]); tcs = np.asarray(tim["frame_ms"]["C1_4"])
    ip = rs.integers(0, R, (Bn, R)); ig = rs.integers(0, Rg, (Bn,) + grid.shape); it = rs.integers(0, tgs.shape[1], (Bn,) + tgs.shape)
    ic = rs.integers(0, len(tcs), (Bn, len(tcs)))
    b1 = sse1[ip].mean(1) - vr
    bg = np.take_along_axis(np.broadcast_to(grid, (Bn,) + grid.shape), ig, 2).mean(2) - vr
    btg = np.median(np.take_along_axis(np.broadcast_to(tgs, (Bn,) + tgs.shape), it, 2), 2)
    beq = interp_vec(np.median(tcs[ic], 1), btg, bg)
    lr = np.log(b1 / beq)
    ratio = mse1 / meq if meq == meq else float("nan")
    se = float(np.nanstd(lr, ddof=1))
    d5 = {"mse_C1": mse1, "mse_R0_equal_time": meq, "interp_flag": flag, "t_C1_ms": tc, "t_R0_4_ms": float(tg[0]),
          "equal_time_ratio": ratio, "ci95": [ratio * math.exp(-Z * se), ratio * math.exp(Z * se)],
          "equal_spp_ratio": float(mse1 / (sse0.mean() - vr)), "cost_ratio": tc / float(tg[0])}
    d5["pass"] = bool(flag == "ok" and ratio <= cfg["benefit_ratio_max"] and d5["ci95"][1] < cfg["benefit_upper95_below"])
    # D4
    G = 8
    k = nsub // G
    GS1 = SG1.reshape(G, k, 3, -1).sum(1); GS2 = SG2.reshape(G, k, 3, -1).sum(1)
    d4 = {}
    for roi, m in masks.items():
        idx = np.flatnonzero(m.reshape(-1))
        scale = float(vT[:, idx].mean())
        qsum = q_summary(q_from_group_sums(GS1[:, :, idx], GS2[:, :, idx], R // G, scale))
        refl = float((LUMA @ ref[:, m]).mean())
        b1 = b1_state({4: lumD[:, MM.names.index(roi)]}, refl)
        d4[roi] = {"Q": qsum, "Q_state": q_state({"4": qsum}, QLIM[roi]), "B1": b1, "pixels": int(m.sum())}
    states = [v["Q_state"] for v in d4.values()] + [v["B1"]["state"] for v in d4.values()]
    d4_state = "BIASED" if "BIASED" in states else ("UNRESOLVED" if "UNRESOLVED" in states else "EQUIVALENT")
    res.update({"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D4_state": d4_state, "D5": d5})
    res["c1_persistent_state"] = "none: C1 holds no buffers across frames; its inner stream is seeded per frame and reads the current scene and table"
    res["correctness_D1_D3"] = bool(d1["pass"] and all(r["pass"] for r in d2) and d3["pass"])
    res["benefit_pass"] = d5["pass"]
    res["table_sha_after"] = RT.table_bytes_sha(Bd2)
    assert res["table_sha_after"] == d1["table_sha256"]
    res["executed"] = True
    if not res["correctness_D1_D3"]:
        res["verdict"] = "NOT_SUPPORTED"
    elif d4_state == "BIASED":
        res["verdict"] = "BIAS_SUPPORTED"
    elif d4_state == "UNRESOLVED" or not d5["pass"]:
        res["verdict"] = "STATIC_ONLY_SUPPORTED"
    else:
        res["verdict"] = "QUALIFIED_STATIC_AND_DYNAMIC"
    write(out / "dyn109a.json", res)
    log("DYNAMIC", res["verdict"], "D4", d4_state, "D5", d5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("static_json"); ap.add_argument("budget_json")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(Path(a.out).resolve(), Path(a.static_json).resolve(), Path(a.budget_json).resolve(), a.smoke)
