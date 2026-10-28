"""WL55 report tables (protocol fnto_k55): tractability rule, Stage E near-contact change (floor -> floor_c4) and dent / floor_c4 coupled
tables, oblique groups against the 0.125 mm references (A1.2 / A3), and the Stage F cost benchmark (GPU idle, CUDA-synchronised, median of 5).

    P53_MI_VARIANT=cuda_ad_rgb python k55_report.py tables <out json>
    P53_MI_VARIANT=cuda_ad_rgb python k55_report.py cost <out json>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import k55_analyze as KA
import k55_common as KCm
import k55_compose as KP
import k55_fixtures as KF
import k55_ops as KO
import k55_stageB as SBd
import c54_casp as CA
import p53_compose as CP

OUT = KA.ROOT / "out"


def tables(outp):
    BD = json.load(open(OUT / "analysis" / "stageBD.json")); res = {}
    # tractability: near-contact face-on groups of the design scenes, nee+KB, amended unit
    tr = {}
    for scene in ("gap8", "gap4", "floor", "bend90", "gap6s"):
        rules = [r for r in BD["full"][scene]["rules"] if r["near_contact"] and r["cam"] in ("gapdown", "gapup", "underup", "underdown", "bot")]
        red = [((abs(r["T_P10"]) - abs(r["T_CAK"])) / (abs(r["T_P10"]) - abs(r["T_P025"]))) for r in rules
               if None not in (r["T_P10"], r["T_CAK"], r["T_P025"]) and abs(r["T_P10"]) - abs(r["T_P025"]) > 0.005]
        cells = {a: json.load(open(KA.SB / scene / f"{a}_s1.json"))["cells"] for a in ("P10", "CAK", "P025")}
        tr[scene] = dict(n=len(red), recovery_median=float(np.median(red)) if red else None, recovery_min=float(np.min(red)) if red else None, cells=cells,
                         cells_ratio_CAK_P025=cells["CAK"] / cells["P025"], cells_ratio_CAK_P10=cells["CAK"] / cells["P10"])
    pooled = [x for s in ("gap8", "gap4", "floor", "bend90") for x in [tr[s]["recovery_median"]] if x is not None]
    tr["rule"] = dict(design_recovery_median_min=min(pooled), design_cells_ratio_max=max(tr[s]["cells_ratio_CAK_P025"] for s in ("gap8", "gap4", "floor", "bend90")),
                      passed=bool(min(pooled) >= 0.8 and max(tr[s]["cells_ratio_CAK_P025"] for s in ("gap8", "gap4", "floor", "bend90")) <= 0.5))
    res["tractability"] = tr
    # Stage E coupled tables for floor_c4 and dent (all arms), same machinery as stage B/D
    for scene in ("floor_c4", "dent"):
        sd, cyc = KA.cycles_of(scene); lab = KCm.labels_of(sd); S = KA.pair_stats(scene, sd, cyc, lab); summ = {}
        for cam, units in S.items():
            for key, g in units["amended"].items():
                arm, ro, ref = key.split("|")
                if ref != "cyc":
                    continue
                for gn, v in g.items():
                    for c in range(3):
                        if v["ruled"][c]:
                            e = summ.setdefault(f"{arm}|{ro}", dict(n=0, n_pass=0, worst=0.0, fails=[])); e["n"] += 1; ok = bool(v["passed_total"][c]); e["n_pass"] += ok
                            e["worst"] = max(e["worst"], abs(v["bias"][c]))
                            if not ok:
                                e["fails"].append([cam, gn, c, round(100 * v["bias"][c], 2)])
        res[f"coupled_{scene}"] = summ
        res[f"groups_{scene}"] = {cam: {k: {gn: [round(100 * b, 2) for b in v["bias"]] for gn, v in g.items() if any(v["ruled"])} for k, g in u["amended"].items()
                                        if k.split("|")[0] in ("P10", "CAK", "P025", "COR") and k.split("|")[2] == "cyc"} for cam, u in S.items()}
    # oblique groups against the 0.125 mm reference (amended unit, single seed), nee+KB
    ob = {}
    for scene, cams, s1 in (("gap8", ("obot", "obotx"), 5541), ("gap4", ("obot", "obotx"), 5541), ("gap6s", ("obot", "obotx"), 5543)):
        sd = KF.scene_def(scene); ss = KCm.classify_sceneset(sd); lab = KCm.labels_of(sd); cdir = OUT / "cycles_px0.125"; geo = json.load(open(cdir / f"ccam_{scene}.json"))
        for cam in cams:
            u = KCm.cycles_ports(ss, geo[cam], [np.load(cdir / f"cimg_{scene}_{cam}_s{s1}.npy")])["amended"]; key = KCm.dirkey(sd["cams"][cam]["dir"]); rec = {}
            for arm, members in (("P10_wl53", ("P10_wl53",)), ("P10", KA.PAIRS["P10"]), ("CAK", KA.PAIRS["CAK"]), ("P025", KA.PAIRS["P025"]), ("COR", KA.PAIRS["COR"])):
                ps = [KA.load_arm(scene, a, key) for a in members]
                if any(p is None for p in ps):
                    continue
                for ri, ro in enumerate(("nee", "ball")):
                    if any(p[ri] is None for p in ps):
                        continue
                    g = KCm.group_bias(np.mean([p[ri] for p in ps], 0), u["mean"], u["sel"], lab)
                    rec[f"{arm}|{ro}"] = {gn: [round(100 * b, 2) for b in v["bias"]] for gn, v in g.items() if any(v["ruled"])}
            ob[f"{scene}|{cam}"] = rec
    res["oblique_vs_0p125mm"] = ob
    KCm.dump(res, outp); print(json.dumps(res["tractability"], indent=1)); print(json.dumps({k: {a: (v["n_pass"], v["n"], round(100 * v["worst"], 2)) for a, v in d.items()} for k, d in res.items() if k.startswith("coupled")}, indent=1))
    for k, v in ob.items():
        for a, g in v.items():
            print(k, a, g)


def _t(fn, rep=5):
    ts = []
    for _ in range(rep):
        torch.cuda.synchronize(); t0 = time.time(); r = fn(); torch.cuda.synchronize(); ts.append(time.time() - t0)
    return float(np.median(ts)), r


def cost(outp):
    res = {"gpu_idle_check": __import__("subprocess").run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()}
    Kf = SBd.load_op("slab_L4_s1"); res["K_fine_2p5mm"] = dict(nnz=len(Kf["idx"]), bytes=int(len(Kf["idx"]) * 8 + Kf["val"].numel() * 4 + Kf["KC"].numel() * 4), tr_seconds=Kf["seconds"],
                                                                    photons=Kf["stats"]["photons"], KC_bytes=int(Kf["KC"].numel() * 4))
    for name in ("slab_L1d_s1", "slab_L2d_s1", "slab_L1d_din64"):
        r = SBd.load_op(name); res[name] = dict(nnz=len(r["idx"]), tr_seconds=r["seconds"], photons=r["stats"]["photons"])
    for scene in ("floor", "floor_c4", "gap8", "gap6s", "gap4"):
        sd = KF.scene_def(scene); sun = torch.load(KA.SB / scene / "sun.pt"); keys = sorted(set(SBd.cam_dirs_of(sd, SBd.SCENE_CAMS[scene]))); rec = {}
        for arm in ("P10", "P05", "P025", "CAK"):
            t_lev, levels = _t(lambda: SBd.cak_levels(sd) if arm == "CAK" else SBd.uniform_levels(sd, {"P10": 1, "P05": 2, "P025": 4}[arm]), 3)
            meds = [(ci, m, V) for ci, (n, k, m, V) in enumerate(sd["comps"]) if k == "medium"]
            def agg():
                return {ci: KP.arm_operator(Kf, m, V, levels[ci]) for ci, m, V in meds}
            t_agg, Kops = _t(agg, 3)
            ss, parent, _ = KP.build_ss(sd, levels); cnt = np.bincount(parent); mp = np.maximum(1, np.round(128 / cnt[parent])).astype(int)
            t_C, (C, esc, st) = _t(lambda: CA.cast_adaptive(ss, mp, 5581), 3)
            Tm = {ci: Kops[ci]["K"] for ci, m, V in meds}; phi = torch.rand(ss.Ptot * 64, 3, device="cuda")
            t_K, _ = _t(lambda: CP.apply_K(ss, Tm, CP.apply_C(C, phi)), 20); t_Conly, _ = _t(lambda: CP.apply_C(C, phi), 20)
            torch.cuda.reset_peak_memory_stats(); t_full, r = _t(lambda: KP.compose(sd, levels, Kops, sun, keys, rho=False, label=f"cost/{scene}/{arm}"), 2)
            rec[arm] = dict(cells=int(ss.Ptot), t_levels=t_lev, t_aggregate=t_agg, t_C=t_C, C_nnz=int(C._nnz()), t_iter_KC=t_K, t_iter_C=t_Conly, t_compose_total=t_full,
                            K_nnz={str(ci): Kops[ci]["nnz"] for ci, m, V in meds}, iters=r["meta"]["iters"], gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9)
            del Kops, C, Tm; torch.cuda.empty_cache()
        res[scene] = rec; print(scene, json.dumps(rec)[:800])
    KCm.dump(res, outp)


if __name__ == "__main__":
    {"tables": tables, "cost": cost}[sys.argv[1]](Path(sys.argv[2]))
