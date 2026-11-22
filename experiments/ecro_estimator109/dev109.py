"""WL109 development stage on WL105 training cameras (never the evaluation views): variance attribution, frozen M-selection
rule and futility gate.

python dev109.py OUT [--smoke]

For M in {1, 2, 4}: 16 paired instrumentation frames per dev view at 4 spp (LOW, original HIGH 'ho', C1 HIGH) and frozen
synchronized timing of R0 and lean C1. Predicted equal-time variance ratio
    r(M) = Var_C1(M) * t_C1(M) / (Var_R0 * t_R0)       (primary ROI, pooled; geometric mean over dev views)
This screen assumes R0 variance proportional to 1 / time and is used only to choose M and to stop on futility; the final
decision uses the measured R0 quality-versus-time curve on the evaluation views.
"""
import argparse
from pathlib import Path
import time

import numpy as np

from common109 import P, DEV_VIEWS, apply_smoke, gm, rois, seed_dev, seed_timing, sha, write, HERE
import runtime109 as RT


def main(out, smoke):
    import torch
    import stats109 as ST
    from gpu109 import Engine, timing_schedule
    if smoke:
        apply_smoke()
    cfg = P["development"]
    out.mkdir(parents=True, exist_ok=False)
    logf = open(out / "dev.log", "a")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    R, rt, Bd, audit, _ = RT.admit()
    eng = Engine(R, rt, Bd)
    cams = RT.cameras()
    res = {"protocol_sha256": sha(HERE / "protocol109.json"), "smoke": smoke, "admission": audit, "versions": RT.versions(R),
           "code_sha256": {p.name: sha(p) for p in sorted(HERE.glob("*.py"))}, "views": {}, "complete": False}
    Ms, Rn, spp = cfg["M_candidates"], cfg["replicates"], cfg["spp"]
    for di, view in enumerate(DEV_VIEWS):
        aux = RT.aux(view)
        mask = rois(aux)["primary"]
        rec = {"M": {}}
        boot = ST.Boot(Rn, cfg["bootstrap_draws"], P["seeds"]["bootstrap"] + 100 + di, "cuda")
        for mi_, M in enumerate(Ms):
            L, HO, HC = [], [], []
            for r in range(Rn):
                c = eng.c1_frame(cams[view], spp, seed_dev(di, mi_, r, smoke), M, orig_hl=True).double()
                L.append(c[0:3] + c[3:6]); HC.append(c[6:9]); HO.append(c[9:12])
            L, HO, HC = torch.stack(L), torch.stack(HO), torch.stack(HC)
            pv = lambda X, i=None: ST.pooled(ST.pix_var(X, i), mask)
            pc = lambda X, Y, i=None: ST.pooled(ST.pix_cov(X, Y, i), mask)
            m = {"var_L": pv(L), "var_H": pv(HO), "two_cov_LH": 2 * pc(L, HO), "var_T_R0": pv(L + HO),
                 "var_HC": pv(HC), "two_cov_L_HC": 2 * pc(L, HC), "var_T_C1": pv(L + HC),
                 "cov_HO_HC": pc(HO, HC)}
            m["var_T_ratio"] = ST.ratio_stat(lambda i: pv(L + HC, i), lambda i: pv(L + HO, i), boot)
            rec["M"][str(M)] = m
            log(view, "M", M, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()})
        # timing (frozen, synchronized, interleaved) on an exclusive GPU
        slots = [("R0", 0)] + [("C1", M) for M in Ms]

        def block():
            for kind, M in slots:                         # recording + warm-up
                for w in range(1 + cfg["timing_warmup"]):
                    eng.run_frozen(kind, view, cams[view], spp, seed_timing(10 + di, 0, w, smoke), M)
            times = {f"{k}{M}": [] for k, M in slots}
            for rr, (kind, M) in timing_schedule(slots, cfg["timing_rounds"], P["seeds"]["timing_order"] + di):
                times[f"{kind}{M}"].append(eng.timed(kind, view, cams[view], spp, seed_timing(10 + di, 1 + slots.index((kind, M)), rr, smoke), M))
            return times
        times, rec["timing_exclusivity"] = RT.exclusive_block(block, log, require=not smoke)
        rec["timing_ms"] = {k: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "n": len(v), "all": v}
                            for k, v in times.items()}
        rec["gpu_after"] = RT.gpu_snapshot()
        t0 = rec["timing_ms"]["R00"]["median"]
        for M in Ms:
            m = rec["M"][str(M)]
            tc = rec["timing_ms"][f"C1{M}"]["median"]
            m["cost_ratio"] = tc / t0
            m["predicted_equal_time_ratio"] = m["var_T_ratio"]["point"] * tc / t0
        res["views"][view] = rec
        write(out / "dev.json", res)
    # frozen selection rule
    pred = {str(M): gm([res["views"][v]["M"][str(M)]["predicted_equal_time_ratio"] for v in DEV_VIEWS]) for M in Ms}
    best = min(pred.values())
    chosen = min(int(M) for M, r in pred.items() if r <= best + cfg["tie_tolerance"])
    res["predicted_equal_time_ratio_gm"] = pred
    res["selected_M"] = chosen
    res["futile"] = bool(pred[str(chosen)] > cfg["futility_ratio"])
    res["complete"] = True
    write(out / "dev.json", res)
    log("DEV selected M", chosen, "predicted", pred, "futile", res["futile"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(Path(a.out).resolve(), a.smoke)
