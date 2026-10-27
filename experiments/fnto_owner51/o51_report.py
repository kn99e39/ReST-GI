"""WL51 report: predeclared rules of protocol fnto_owner51 evaluated on the stage outputs (no new computation).

    python o51_report.py <results dir>
"""
import json
import sys
from pathlib import Path

import numpy as np


def ld(p):
    return json.loads(p.read_text()) if p.exists() else None


def main(d):
    d = Path(d); g1 = ld(d / "gate1_operator/g1.json"); vw = ld(d / "gate1_operator/views.json"); g3 = ld(d / "gate3_energy/g3_closure.json")
    g0 = ld(d / "gate0_provenance/g0_chord.json"); sl = ld(d / "gate2_renderer/slab.json"); rep = {"rules": {}, "tables": {}}
    R = rep["rules"]; Tb = rep["tables"]
    if g1:
        rho = {}
        for key in [k for k in g1 if k.endswith(("_5011", "_5012"))]:
            rho[key] = {mk: {"rho_RGB": v["rho_RGB"], "rho_max": v["rho_max"], "status": v["status"]} for mk, v in g1[key].items() if isinstance(v, dict) and "rho_RGB" in v}
        Tb["rho"] = rho
        repro = {}
        for key in ("H_5011", "L0C_5011", "L1C_5011"):
            p = g1.get(key, {}).get("none", {})
            if "init_seed1" in p:
                a, b = p["rho_max"], p["init_seed1"]["rho_max"]; repro[key] = {"seed0": a, "seed1": b, "rel": abs(a - b) / max(a, 1e-30), "pass": abs(a - b) / max(a, 1e-30) < 0.02}
        R["reproducible"] = repro
        nec, suf, near = {}, {}, {}
        for key in ("L0C_5011", "L1C_5011", "H_5011"):
            r = rho.get(key, {})
            if r.get("none", {}).get("rho_max", 0) >= 1:
                nec[key] = {mk: r[mk]["rho_max"] < 1 for mk in ("noYY", "noYO", "noOY", "noOO", "noYY_t3mm", "noYY_t3cm", "noCross") if mk in r}
                suf[key] = r.get("onlyYY", {}).get("rho_max", 0) >= 1
                near[key] = r.get("noYY_t3mm", {}).get("rho_max", 9) < 1
        R["necessary_groups"] = nec; R["YY_sufficient"] = suf; R["near_coincident_necessary"] = near
        sem_rho = {k: rho.get(f"{k}_5011", {}).get("SEM", {}).get("rho_max") for k in ("H", "L0C", "L1C")}
        Tb["SEM_rho_max"] = sem_rho
        Tb["modes"] = {key: {"prod": g1[key]["none"].get("mode"), "SEM": g1[key].get("SEM", {}).get("mode")} for key in ("H_5011", "L0C_5011", "L1C_5011") if key in g1}
    if vw:
        sol = vw.get("solves", {}); conv = {k: all(v.get("status") == "converged" for v in s.values()) for k, s in sol.items()}
        Tb["solve_converged_all_views"] = conv; Tb["pooled"] = vw.get("pooled"); Tb["energy"] = vw.get("energy")
        if g1:
            R["SEM_stabilises"] = {"rho_all_lt_1": all(v is not None and v < 1 for v in sem_rho.values()),
                                   "all_SEM_solves_converged": all(conv.get(f"{k}_SEM", False) for k in ("H", "L0C", "L1C"))}
            R["SEM_stabilises"]["pass"] = R["SEM_stabilises"]["rho_all_lt_1"] and R["SEM_stabilises"]["all_SEM_solves_converged"]
        m = {}
        for k in ("L0C", "L1C"):
            x = vw.get("pooled", {}).get(f"{k}_SEM_solve", {}).get("Yellow", {}).get("indirect_ratio_to_ref")
            m[k] = {"ratio": x, "pass": x is not None and 0.67 <= x <= 1.5}
        R["SEM_matches_reference_indirect"] = m
        ps = {}
        for k, s in vw.get("per_view", {}).items():
            vals = {cl: [v[cl]["psnr_vs_full"] for v in s.values() if cl in v] for cl in ("Yellow", "opaque", "glass", "box")}
            ps[k] = {cl: float(np.mean(x)) for cl, x in vals.items() if x}
            allv = [v["psnr_vs_full"] for v in s.values() if v.get("psnr_vs_full") is not None]
            ps[k]["all"] = float(np.mean(allv)) if allv else None
        Tb["psnr_vs_full_view_mean"] = ps
    if sl:
        def mean_ratio(cfg, kind):
            rs = [np.array(sl["runs"][cfg][s][kind]["ratio_RGB"]) for s in ("5101", "5102") if s in sl["runs"].get(cfg, {})]
            return np.mean(rs, 0) if rs else None
        ja, jb, jm, la = mean_ratio("J_above", "pdiff"), mean_ratio("J_below", "pdiff"), mean_ratio("J_below", "pmirr"), mean_ratio("L_above", "pdiff")
        chain = la is not None and bool(np.all((la >= 0.9) & (la <= 1.1)))
        R["gate2_chain_valid"] = {"L_above_pdiff_ratio_RGB": None if la is None else la.tolist(), "pass": chain}
        Tb["gate2_J_above_pdiff_ratio_RGB_descriptive"] = None if ja is None else ja.tolist()
        Tb["gate2_J_above_pmirr_ratio_RGB_descriptive"] = None if mean_ratio("J_above", "pmirr") is None else mean_ratio("J_above", "pmirr").tolist()
        R["gate2_diffuse_ancestor_rule"] = {"J_below_pdiff_ratio_RGB": None if jb is None else jb.tolist(),
                                            "confirmed": bool(chain and jb is not None and np.all(jb[:2] < 0.05)), "refuted": bool(jb is not None and np.any(jb[:2] >= 0.5))}
        R["gate2_glossy_ancestor_keeps_burley"] = {"J_below_pmirr_ratio_RGB": None if jm is None else jm.tolist(), "pass": bool(jm is not None and np.all((jm[:2] >= 0.9) & (jm[:2] <= 1.1)))}
        J = [np.array(sl["runs"]["J_below"][s]["L_top_center_RGB"]) for s in ("5101", "5102") if s in sl["runs"].get("J_below", {})]
        S = [np.array(sl["runs"]["S_below"][s]["L_top_center_RGB"]) for s in ("5101", "5102") if s in sl["runs"].get("S_below", {})]
        if len(J) == 2 and len(S) == 2:
            jm_, js_ = np.mean(J, 0), np.abs(J[0] - J[1]); sm_ = np.mean(S, 0)
            R["gate2_object_scope"] = {"J_below_Ltop_RGB": jm_.tolist(), "J_seed_diff_RGB": js_.tolist(), "S_below_Ltop_RGB": sm_.tolist(),
                                       "pass": bool(jm_[0] > 10 * js_[0] and sm_[0] < 0.05 * jm_[0])}
            G = [np.array(sl["runs"]["S_below"][s]["L_gap_center_RGB"]) for s in ("5101", "5102")]
            R["gate2_object_scope"]["S_below_Lgap_RGB"] = np.mean(G, 0).tolist()
        Tb["mesh_areas_m2"] = sl.get("mesh_areas_m2")
    if g3:
        Tb["C_closure"] = g3
        cap = g3["captured_fraction_by_sender"]; Q = np.array(g3["reciprocity_Q[r][s]"]); names = g3["classes"]
        R["C_closure_le_1.05"] = {n: cap[i] <= 1.05 for i, n in enumerate(names)}
        R["reciprocity_in_[0.9,1.1]"] = {f"{names[i]}<->{names[j]}": bool(0.9 <= Q[i, j] <= 1.1) for i in range(len(names)) for j in range(i + 1, len(names)) if np.isfinite(Q[i, j])}
    if g0:
        Tb["gate0_chord_proxy"] = g0
    (d / "report.json").write_text(json.dumps(rep, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))); print(json.dumps(rep["rules"], indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
