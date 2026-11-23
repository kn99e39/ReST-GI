"""WL109A GPU captures (run only under guard109.py; outputs are deterministic functions of the seeds).

python capture109a.py OUT MODE [--R N] [--smoke]
  MODE null     Stage B: independent R0 null pool, 2048 frames per (view, bias spp); no C1 frame is rendered
  MODE paired   Stage C: paired R0 / C1 (M = 1) frames, R per (view, bias spp)
  MODE grid     Stage C: R0 quality-versus-time grid (18 WL109 levels x 64 replicates per view)
  MODE timing   Stage C: WL109 Stage 3 timing block per view (exclusive GPU)

Each unit (view, spp) or (view) is written to a temporary directory and renamed atomically; completed units are skipped,
so a guard restart resumes exactly. The renderer, C1, Engine and admission are imported unchanged from ecro_estimator109.
"""
import argparse
import os
from pathlib import Path
import shutil
import time

import numpy as np

from common109a import (P, P9, BIAS_SPP, DECISION_ROIS, EVAL_VIEWS, GRID, NOMINAL, LUMA, all_masks, raw_pixels, read, sha, write,
                        seed_grid, seed_null, seed_paired, seed_timing, HERE, W109)
import runtime109 as RT

RES = 512
SMOKE = {"null_frames": 64, "paired_R": 32, "grid_R": 2, "timing_rounds": 2, "timing_warmup": 1, "views": ("V0", "H4")}


class Masks:
    """Dense mask weights on the GPU for per-frame mask means; names in a fixed order."""

    def __init__(self, masks, torch):
        self.names = sorted(masks)
        W = np.stack([masks[k].reshape(-1).astype(np.float64) for k in self.names])
        self.cnt = torch.as_tensor(W.sum(1), device="cuda")
        self.W = torch.as_tensor(W, device="cuda")
        self.luma = torch.as_tensor(LUMA, device="cuda")

    def mean_luma(self, Y):                       # Y [3, HW] -> [K]
        return (self.luma @ Y) @ self.W.T / self.cnt

    def mean_rgb(self, Y):                        # Y [3, HW] -> [K] mean over channels and mask
        return (Y.sum(0) @ self.W.T) / (3 * self.cnt)


P_TABLE_SHA = None
WL109_M = None


def unit_done(d):
    return (d / "unit.json").exists()


def finish_unit(tmp, final, rec):
    rec["files"] = {p.name: sha(p) for p in sorted(tmp.iterdir()) if p.is_file()}
    write(tmp / "unit.json", rec)
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)


def views_for(smoke):
    return SMOKE["views"] if smoke else EVAL_VIEWS


# ------------------------------------------------------------------------------------------------ null pool
def run_null(out, eng, cams, audit, smoke, log, torch):
    cfg = P["calibration"]["null_pool"]
    N0 = SMOKE["null_frames"] if smoke else cfg["frames"]
    gp = (N0 // 2) // cfg["groups"]
    G = (N0 // 2) // gp
    for view in views_for(smoke):
        vi = EVAL_VIEWS.index(view)
        aux = RT.aux(view)
        masks = all_masks(aux)
        M = Masks(masks, torch)
        ridx = torch.as_tensor(raw_pixels(masks), device="cuda")
        for si, spp in enumerate(BIAS_SPP):
            final = out / "null" / f"{view}_spp{spp}"
            if unit_done(final):
                log("null", view, spp, "exists"); continue
            tmp = out / "null" / f".tmp_{view}_spp{spp}"
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            t0 = time.time()
            S1 = torch.zeros((G, 3, RES * RES), dtype=torch.float64, device="cuda"); S2 = torch.zeros_like(S1)
            sT = torch.zeros((3, RES * RES), dtype=torch.float64, device="cuda"); sT2 = torch.zeros_like(sT)
            lumH = np.empty((N0, len(M.names))); lumT = np.empty((N0, len(M.names)))
            raw = np.empty((N0, 6, ridx.numel()), np.float32)
            prevH = None
            for k in range(N0):
                f = eng.r0(cams[view], spp, seed_null(vi, si, k, smoke)).double().view(9, -1)
                L = f[0:3] + f[3:6]; H = f[6:9]; T = L + H
                sT += T; sT2 += T * T
                lumH[k] = M.mean_luma(H).cpu().numpy(); lumT[k] = M.mean_luma(T).cpu().numpy()
                raw[k] = torch.cat([L, H])[:, ridx].float().cpu().numpy()
                if k % 2 == 0:
                    prevH = H
                else:
                    D = prevH - H
                    g = (k // 2) // gp
                    S1[g] += D; S2[g] += D * D
                if k == 0:
                    np.save(tmp / "frame0.npy", f.float().view(9, RES, RES).cpu().numpy())
            np.save(tmp / "S1.npy", S1.view(G, 3, RES, RES).cpu().numpy()); np.save(tmp / "S2.npy", S2.view(G, 3, RES, RES).cpu().numpy())
            np.save(tmp / "sT.npy", sT.view(3, RES, RES).cpu().numpy()); np.save(tmp / "sT2.npy", sT2.view(3, RES, RES).cpu().numpy())
            np.save(tmp / "lumH.npy", lumH); np.save(tmp / "lumT.npy", lumT); np.save(tmp / "raw.npy", raw)
            np.save(tmp / "raw_idx.npy", ridx.cpu().numpy())
            del S1, S2
            torch.cuda.empty_cache()
            finish_unit(tmp, final, {"mode": "null", "view": view, "spp": spp, "frames": N0, "pairs": N0 // 2, "group_pairs": gp,
                                     "groups": G, "mask_names": M.names, "raw_channels": ["L0", "L1", "L2", "H0", "H1", "H2"],
                                     "seed_first": seed_null(vi, si, 0, smoke), "seed_last": seed_null(vi, si, N0 - 1, smoke),
                                     "smoke": smoke, "elapsed_s": time.time() - t0, "table_sha256": audit["table_bytes_sha256"]})
            log("null", view, spp, "done", round(time.time() - t0, 1))


# ------------------------------------------------------------------------------------------------ paired evidence
COMP_KEYS = ("var_L", "var_H", "two_cov_LH", "var_T_R0", "var_HC", "two_cov_L_HC", "var_T_C1", "cov_H_HC")


def components_from_sums(s, n, M):
    """pooled (over channels and each mask) unbiased components from per-pixel sums over n frames; s: dict of [3, HW]."""
    v = lambda a, aa: (aa - a * a / n) / (n - 1)
    c = lambda a, b, ab: (ab - a * b / n) / (n - 1)
    vl, vh, vc = v(s["L"], s["LL"]), v(s["H"], s["HH"]), v(s["C"], s["CC"])
    clh, clc, chc = c(s["L"], s["H"], s["LH"]), c(s["L"], s["C"], s["LC"]), c(s["H"], s["C"], s["HC"])
    maps = {"var_L": vl, "var_H": vh, "two_cov_LH": 2 * clh, "var_T_R0": vl + vh + 2 * clh, "var_HC": vc, "two_cov_L_HC": 2 * clc,
            "var_T_C1": vl + vc + 2 * clc, "cov_H_HC": chc}
    return {k: M.mean_rgb(maps[k]).cpu().numpy() for k in COMP_KEYS}


def run_paired(out, eng, cams, audit, R, smoke, log, torch):
    tol = P9["synthetic"]["structural_rel_l1"]
    sub = 16
    assert R % (sub * 8) == 0 or smoke
    Mstar = WL109_M
    for view in views_for(smoke):
        vi = EVAL_VIEWS.index(view)
        aux = RT.aux(view)
        masks = all_masks(aux)
        M = Masks(masks, torch)
        ridx = torch.as_tensor(raw_pixels(masks), device="cuda")
        refc, vref, refrec = RT.reference(view)
        ref = torch.as_tensor((refc[0:3] + refc[3:6] + refc[6:9]).reshape(3, -1), device="cuda")
        for si, spp in enumerate(BIAS_SPP):
            final = out / "paired" / f"{view}_spp{spp}"
            if unit_done(final):
                log("paired", view, spp, "exists"); continue
            tmp = out / "paired" / f".tmp_{view}_spp{spp}"
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            t0 = time.time()
            nsub = R // sub
            SG1 = np.empty((nsub, 3, RES, RES)); SG2 = np.empty((nsub, 3, RES, RES))
            z = lambda: torch.zeros((3, RES * RES), dtype=torch.float64, device="cuda")
            keys = ("L", "H", "C", "LL", "HH", "CC", "LH", "LC", "HC")
            glob = {k: z() for k in keys}
            subc = {k: np.empty((nsub, len(M.names))) for k in COMP_KEYS}
            per = {k: np.empty((R, len(M.names))) for k in ("lumD", "lumT0", "lumT1", "sse0", "sse1")}
            raw = np.empty((R, 9, ridx.numel()), np.float32)
            ident = {"low_bitwise": 0, "ho_bitwise": 0, "max_low_rel_l1": 0.0, "max_ho_rel_l1": 0.0}
            s = None
            for r in range(R):
                if r % sub == 0:
                    s = {k: z() for k in keys}; d1 = z(); d2 = z()
                seed = seed_paired(vi, si, r, smoke)
                r0 = eng.r0(cams[view], spp, seed)
                c1 = eng.c1_frame(cams[view], spp, seed, Mstar, orig_hl=True)       # sorted keys: e1 e2 hl ho
                lr = float((c1[0:6] - r0[0:6]).abs().sum() / r0[0:6].abs().sum())
                hr = float((c1[9:12] - r0[6:9]).abs().sum() / max(float(r0[6:9].abs().sum()), 1e-30))
                assert lr <= tol and hr <= tol, (view, spp, r, lr, hr)
                ident["low_bitwise"] += int(torch.equal(c1[0:6], r0[0:6])); ident["ho_bitwise"] += int(torch.equal(c1[9:12], r0[6:9]))
                ident["max_low_rel_l1"] = max(ident["max_low_rel_l1"], lr); ident["max_ho_rel_l1"] = max(ident["max_ho_rel_l1"], hr)
                f0 = r0.double().view(9, -1); hc = c1[6:9].double().view(3, -1)
                L = f0[0:3] + f0[3:6]; H = f0[6:9]; C = hc
                D = C - H; T0 = L + H; T1 = L + C
                d1 += D; d2 += D * D
                for acc in (s, glob):
                    acc["L"] += L; acc["H"] += H; acc["C"] += C; acc["LL"] += L * L; acc["HH"] += H * H; acc["CC"] += C * C
                    acc["LH"] += L * H; acc["LC"] += L * C; acc["HC"] += H * C
                per["lumD"][r] = M.mean_luma(D).cpu().numpy(); per["lumT0"][r] = M.mean_luma(T0).cpu().numpy()
                per["lumT1"][r] = M.mean_luma(T1).cpu().numpy()
                per["sse0"][r] = M.mean_rgb((T0 - ref) ** 2).cpu().numpy(); per["sse1"][r] = M.mean_rgb((T1 - ref) ** 2).cpu().numpy()
                raw[r] = torch.cat([L, H, C])[:, ridx].float().cpu().numpy()
                if r == 0:
                    np.save(tmp / "frame0_R0.npy", r0.cpu().numpy()); np.save(tmp / "frame0_C1H.npy", c1[6:9].cpu().numpy())
                if r % sub == sub - 1:
                    j = r // sub
                    SG1[j] = d1.view(3, RES, RES).cpu().numpy(); SG2[j] = d2.view(3, RES, RES).cpu().numpy()
                    cs = components_from_sums(s, sub, M)
                    for k in COMP_KEYS:
                        subc[k][j] = cs[k]
            np.save(tmp / "SG1.npy", SG1); np.save(tmp / "SG2.npy", SG2)
            np.save(tmp / "glob.npy", np.stack([glob[k].view(3, RES, RES).cpu().numpy() for k in keys]))
            np.savez(tmp / "subcomp.npz", **subc)
            np.savez(tmp / "perframe.npz", **per)
            np.save(tmp / "raw.npy", raw); np.save(tmp / "raw_idx.npy", ridx.cpu().numpy())
            del glob, s
            torch.cuda.empty_cache()
            finish_unit(tmp, final, {"mode": "paired", "view": view, "spp": spp, "R": R, "subgroup": sub, "subgroups": nsub,
                                     "M": Mstar, "glob_keys": list(keys), "mask_names": M.names, "identity": ident,
                                     "raw_channels": ["L0", "L1", "L2", "H0", "H1", "H2", "C0", "C1", "C2"],
                                     "seed_first": seed_paired(vi, si, 0, smoke), "seed_last": seed_paired(vi, si, R - 1, smoke),
                                     "reference": refrec, "smoke": smoke, "elapsed_s": time.time() - t0,
                                     "table_sha256": audit["table_bytes_sha256"]})
            log("paired", view, spp, "done", round(time.time() - t0, 1), ident)


# ------------------------------------------------------------------------------------------------ R0 grid
def run_grid(out, eng, cams, audit, smoke, log, torch):
    Rg = SMOKE["grid_R"] if smoke else P["evidence"]["grid"]["replicates"]
    for view in views_for(smoke):
        vi = EVAL_VIEWS.index(view)
        final = out / "grid" / view
        if unit_done(final):
            log("grid", view, "exists"); continue
        tmp = out / "grid" / f".tmp_{view}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        t0 = time.time()
        masks = all_masks(RT.aux(view))
        M = Masks(masks, torch)
        refc, vref, refrec = RT.reference(view)
        ref = torch.as_tensor((refc[0:3] + refc[3:6] + refc[6:9]).reshape(3, -1), device="cuda")
        sse = np.empty((len(GRID), Rg, len(M.names))); lum = np.empty_like(sse)
        for gi, spp in enumerate(GRID):
            for r in range(Rg):
                f = eng.r0(cams[view], spp, seed_grid(vi, gi, r, smoke)).double().view(9, -1)
                T = f[0:3] + f[3:6] + f[6:9]
                sse[gi, r] = M.mean_rgb((T - ref) ** 2).cpu().numpy(); lum[gi, r] = M.mean_luma(T).cpu().numpy()
        np.savez(tmp / "grid.npz", sse=sse, lum=lum, levels=np.asarray(GRID))
        finish_unit(tmp, final, {"mode": "grid", "view": view, "replicates": Rg, "levels": list(GRID), "mask_names": M.names,
                                 "seed_first": seed_grid(vi, 0, 0, smoke), "reference": refrec, "smoke": smoke,
                                 "elapsed_s": time.time() - t0, "table_sha256": audit["table_bytes_sha256"]})
        log("grid", view, "done", round(time.time() - t0, 1))


# ------------------------------------------------------------------------------------------------ timing
def run_timing(out, eng, cams, audit, R, smoke, log, torch):
    from gpu109 import timing_schedule
    warm = SMOKE["timing_warmup"] if smoke else P9["final"]["timing_warmup"]
    rounds = SMOKE["timing_rounds"] if smoke else P9["final"]["timing_rounds"]
    Mstar = WL109_M
    for view in views_for(smoke):
        vi = EVAL_VIEWS.index(view)
        final = out / "timing" / view
        if unit_done(final):
            log("timing", view, "exists"); continue
        tmp = out / "timing" / f".tmp_{view}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        slots = [("R0", s, 0) for s in GRID] + [("C1", s, Mstar) for s in NOMINAL]

        def block():
            rec = {"first_call_ms": {}, "frame_ms": {}, "e2e_ms": {}, "watermark_bytes": {}}
            eng._frozen.clear()
            for si, (kind, spp, m) in enumerate(slots):
                key = f"{kind}_{spp}"
                eng.sync(); t0 = time.perf_counter()
                eng.run_frozen(kind, view, cams[view], spp, seed_timing(vi, si, 0, smoke), m)
                eng.sync(); rec["first_call_ms"][key] = 1e3 * (time.perf_counter() - t0)
                for w in range(warm):
                    eng.run_frozen(kind, view, cams[view], spp, seed_timing(vi, si, 1 + w, smoke), m)
                rec["watermark_bytes"][key] = eng.watermark(kind, view, cams[view], spp, seed_timing(vi, si, 9, smoke), m)
                rec["frame_ms"][key] = []
                if spp in NOMINAL:
                    rec["e2e_ms"][key] = []
            for rr, (kind, spp, m) in timing_schedule(slots, rounds, P["seeds"]["timing_order"] + 10 + vi):
                si = slots.index((kind, spp, m))
                key = f"{kind}_{spp}"
                rec["frame_ms"][key].append(eng.timed(kind, view, cams[view], spp, seed_timing(vi, si, 10 + rr, smoke), m))
                if spp in NOMINAL:
                    rec["e2e_ms"][key].append(eng.timed(kind, view, cams[view], spp, seed_timing(vi, si, 50 + rr, smoke), m, compose=True))
            return rec
        rec, excl = RT.exclusive_block(block, log, require=not smoke)
        rec["exclusivity"] = excl
        rec["table_sha_after"] = RT.table_bytes_sha(eng.Bd)
        assert rec["table_sha_after"] == audit["table_bytes_sha256"]
        write(tmp / "timing.json", rec)
        finish_unit(tmp, final, {"mode": "timing", "view": view, "M": Mstar, "rounds": rounds, "warmup": warm, "smoke": smoke,
                                 "versions": RT.versions(eng.R), "table_sha256": audit["table_bytes_sha256"]})
        log("timing", view, {k: round(float(np.median(v)), 3) for k, v in rec["frame_ms"].items()})


def main(out, mode, R, smoke):
    global P_TABLE_SHA, WL109_M
    import torch
    from gpu109 import Engine
    out.mkdir(parents=True, exist_ok=True)
    ev9 = Path(os.path.expanduser("~")) / P9["environment"]["remote_root"] / "results/ecro_estimator109/evidence"
    man9 = read(ev9 / "run/manifest.json")
    P_TABLE_SHA = man9["table_sha_after"]
    WL109_M = int(read(ev9 / "dev/dev.json")["selected_M"])
    assert WL109_M == 1
    logf = open(out / f"capture_{mode}.log", "a")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(time.strftime("%H:%M:%S"), s, flush=True); logf.write(time.strftime("%H:%M:%S ") + s + "\n"); logf.flush()
    log("START", mode, "R", R, "smoke", smoke, "pid", os.getpid())
    Rr, rt, Bd, audit, r105 = RT.admit()
    assert audit["table_bytes_sha256"] == P_TABLE_SHA, ("table sha differs from WL109", audit["table_bytes_sha256"])
    log("admitted; table sha equals WL109", P_TABLE_SHA[:16])
    adm = out / f"admission_{mode}.json"
    if not adm.exists():
        write(adm, {"audit": audit, "versions": RT.versions(Rr), "code_sha256": {p.name: sha(p) for p in sorted(HERE.glob("*.py"))},
                    "wl109_code_sha256": {p.name: sha(p) for p in sorted(W109.glob("*.py"))}, "protocol109a_sha256": sha(HERE / "protocol109a.json")})
    eng = Engine(Rr, rt, Bd)
    cams = RT.cameras()
    if mode == "null":
        run_null(out, eng, cams, audit, smoke, log, torch)
    elif mode == "paired":
        run_paired(out, eng, cams, audit, R if not smoke else SMOKE["paired_R"], smoke, log, torch)
    elif mode == "grid":
        run_grid(out, eng, cams, audit, smoke, log, torch)
    elif mode == "timing":
        run_timing(out, eng, cams, audit, R, smoke, log, torch)
    else:
        raise KeyError(mode)
    assert RT.table_bytes_sha(Bd) == P_TABLE_SHA
    log("DONE", mode)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("mode")
    ap.add_argument("--R", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(Path(a.out).resolve(), a.mode, a.R, a.smoke)
