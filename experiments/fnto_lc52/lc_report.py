"""WL52 report: predeclared criteria P1-P4 and the generalization tables from eval.json / bench.json / train.json.

    python lc_report.py <results dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

ARMS = ("ORA2", "C0", "C0p", "C1")


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed); m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main(d):
    d = Path(d); ev = json.loads((d / "eval/eval.json").read_text()); be = json.loads((d / "bench/bench.json").read_text()); tr = json.loads((d / "train/train.json").read_text())
    rep = {"train": {"params": tr["params"], "best": tr.get("best"), "last": tr["curve"][-1], "wall_s": tr.get("wall_s"),
                     "val_ce_last5_rel_range": float(np.ptp([c["val_ce"] for c in tr["curve"][-5:]]) / np.mean([c["val_ce"] for c in tr["curve"][-5:]]))},
           "single_flight": {}, "coupled": {}, "temporal": {}, "criteria": {}}
    for sn, R in ev["sets"].items():
        sf = {}
        for fn in ("smooth", "hot", "first", "noise"):
            sf[fn] = {a: {m: boot([r["single_flight"][fn][a][m] for r in R if r["single_flight"][fn][a].get(m) is not None]) for m in ("all", "near_contact_lt1cm", "strongly_occluded", "room", "box", "arm", "sheet", "abs_err_p99_over_meanE")
                          if any(r["single_flight"][fn][a].get(m) is not None for r in R)} for a in ARMS}
        sf["closure_uniform"] = {a: boot([r["single_flight"]["closure_uniform"][a] for r in R]) for a in ARMS}
        rep["single_flight"][sn] = sf
        C = [r["coupled"] for r in R if "coupled" in r]
        if C:
            cp = {}
            for a in ("ORA", "C0", "C0p", "C1"):
                cp[a] = {"converged": f"{sum(c[a]['status'] == 'converged' for c in C)}/{len(C)}", "iters": boot([c[a]["iters"] for c in C]),
                         "psnr_full_vs_REF": boot([np.mean(c[a]["psnr_full_vs_REF"]) for c in C if "psnr_full_vs_REF" in c[a]]) if any("psnr_full_vs_REF" in c[a] for c in C) else None,
                         "psnr_indirect_vs_REF": boot([np.mean(c[a]["psnr_indirect_vs_REF"]) for c in C if "psnr_indirect_vs_REF" in c[a]]) if any("psnr_indirect_vs_REF" in c[a] for c in C) else None,
                         "Eind_rel_l2_vs_REF": boot([c[a]["Eind_rel_l2_vs_REF"]["all"] for c in C if "Eind_rel_l2_vs_REF" in c[a]]) if any("Eind_rel_l2_vs_REF" in c[a] for c in C) else None}
                if a != "ORA":
                    cp[a]["rho_max"] = boot([max(c[a]["rho_RGB"]) for c in C if "rho_RGB" in c[a]])
                    for m in ("all", "near_contact_lt1cm", "strongly_occluded"):
                        v = [c[a]["port_rel_err_p99_vs_ORA"][m] for c in C if m in c[a].get("port_rel_err_p99_vs_ORA", {})]
                        if v:
                            cp[a][f"p99_port_err_vs_ORA_{m}"] = boot(v)
            dp = [np.mean(c["C1"]["psnr_full_vs_REF"]) - np.mean(c["C0"]["psnr_full_vs_REF"]) for c in C if "psnr_full_vs_REF" in c["C1"] and "psnr_full_vs_REF" in c["C0"]]
            cp["dPSNR_C1_minus_C0"] = boot(dp) if dp else None; cp["REF_split_rel_l2"] = boot([c["REF_split_rel_l2"] for c in C])
            rep["coupled"][sn] = cp
        T = [r["temporal"] for r in R if "temporal" in r]
        if T:
            rep["temporal"][sn] = {a: {k: boot([t[a][k] for t in T if a in t and t[a].get(k) is not None]) for k in ("delta_corr", "delta_rel_l2", "delta_rel_l2_transition")
                                       if any(a in t and t[a].get(k) is not None for t in T)} for a in ("C0", "C0p", "C1")}
            rep["temporal"][sn]["transition_ports_mean"] = float(np.mean([t["visibility_transition_ports"] for t in T]))
    ci = rep["coupled"].get("test_iid", {}); K = rep["criteria"]
    K["P1_coupled_fidelity"] = {"dPSNR": ci.get("dPSNR_C1_minus_C0"), "pass": bool(ci.get("dPSNR_C1_minus_C0") and ci["dPSNR_C1_minus_C0"][0] >= -0.5)}
    p2 = {}
    for sn in ("test_iid", "traj"):
        c = rep["coupled"].get(sn, {})
        for m in ("near_contact_lt1cm", "strongly_occluded"):
            k = f"p99_port_err_vs_ORA_{m}"
            if k in c.get("C1", {}) and k in c.get("C0", {}):
                p2[f"{sn}_{m}"] = {"C1": c["C1"][k][0], "C0": c["C0"][k][0], "pass": c["C1"][k][0] <= 2 * c["C0"][k][0]}
    t = rep["temporal"].get("traj", {})
    if "C1" in t and "delta_rel_l2_transition" in t["C1"]:
        p2["traj_transition_delta"] = {"C1": t["C1"]["delta_rel_l2_transition"][0], "C0": t["C0"]["delta_rel_l2_transition"][0], "pass": t["C1"]["delta_rel_l2_transition"][0] <= 2 * t["C0"]["delta_rel_l2_transition"][0]}
    K["P2_localized"] = {"items": p2, "pass": bool(p2) and all(v["pass"] for v in p2.values())}
    cl = rep["single_flight"]["test_iid"]["closure_uniform"]["C1"][0]; cv = ci.get("C1", {})
    K["P3_operator_validity"] = {"closure_C1_over_ORA": cl, "rho_max": cv.get("rho_max"), "converged": cv.get("converged"),
                                 "pass": abs(cl - 1) <= 0.05 and cv.get("rho_max") is not None and cv["rho_max"][0] < 1 and cv.get("converged", "0/1").split("/")[0] == cv.get("converged", "0/1").split("/")[1]}
    b = be["object_scale"]; base = b[min(b, key=lambda k: abs(int(k) - 4))]
    K["P4_speed"] = {"C_path_speedup_base": base["speedup_C_path"], "frame_speedup_base": base["speedup_frame"], "pass": base["speedup_C_path"] >= 2 and base["speedup_frame"] > 1}
    rep["bench_summary"] = {"object_scale": {k: {x: v[x] for x in ("N", "M", "C0_update_s", "C1_update_s", "C0_path_s", "C1_path_s", "C0_frame_s", "C1_frame_s", "speedup_C_path", "speedup_frame")} for k, v in b.items()},
                            "port_scale": {k: ({x: v[x] for x in ("N", "M", "C0_update_s", "C1_update_s", "C0_path_s", "C1_path_s", "speedup_C_path", "mem_bytes")} if "N" in v else v) for k, v in be["port_scale"].items()},
                            "lego_bvh_build_s": be["lego_bvh_build_s"]}
    (d / "report.json").write_text(json.dumps(rep, indent=1)); print(json.dumps(rep["criteria"], indent=1)); print(json.dumps(rep["train"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
