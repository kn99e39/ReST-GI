"""WL104K-C angular evaluation (protocol stage_D_angular, stage_C_synthetic) and the pre-test fit decision.

  python eval104kc.py design <out.json> <fit_dir> [<fit_dir> ...]   design use only: WL104K-B DEV set, each fit vs WL104K-B O0
  python eval104kc.py fitdecision <root>                             all fits + contracts present; training feasibility, truth
                                                                     feasibility (synthetic), DEV-set descriptive metrics; hashes of
                                                                     every fit / protocol / code / data -> fit_decision/fit_decision.json
                                                                     (+ .sha256). Reads no test_c array.
  python eval104kc.py test <root>                                    test lock (hashed fit decision, unchanged fits / protocol / code)
                                                                     -> independent test_c evaluation, synthetic truth recovery,
                                                                     contrasts, labels -> test_eval/test.json + arrays.npz

Arms. Physical (72 units = 9 materials x 8 starts unless noted): O0 = Stage A R8 (reproduced WL104K-B O0), R4 = Stage A R4 (4 starts,
WL104K batching), CR, O0C, E2W (continuations from O0), CR_R4 (CR from R4 for starts 1-4), E2_B = WL104K-B E2 (authenticated),
P_hist = archived WL104K P (4 starts). Synthetic (36 units): S_O0r, S_Dr (Stage A), CR_S, O0C_S, E2W_S (from S_O0r, train_original
lanes), CR_SD, O0C_SD (from S_Dr, WL104K-B train_coverage lanes).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

import common104kc as C

KB, CM, IO = C.KB, C.CM, C.IO
import eval104kb as EB  # noqa: E402
import data104kb as DK  # noqa: E402
import fit104kc as FC  # noqa: E402

N = C.N_MAT
Stop = C.Stop
PHYS = {"O0": ("stageA/fits/R8", 8), "R4": ("stageA/fits/R4", 4), "CR": ("fits/CR", 8), "O0C": ("fits/O0C", 8),
        "E2W": ("fits/E2W", 8), "CR_R4": ("fits/CR_R4", 8)}
SYN = {"S_O0r": ("stageA/fits/S_O0r", "original"), "S_Dr": ("stageA/fits/S_Dr", "coverage"), "CR_S": ("fits/CR_S", "original"),
       "O0C_S": ("fits/O0C_S", "original"), "E2W_S": ("fits/E2W_S", "original"), "CR_SD": ("fits/CR_SD", "coverage"),
       "O0C_SD": ("fits/O0C_SD", "coverage")}
KEYS = ("el1", "ealb", "eshape", "e_core", "e_shoulder", "core_l1", "shoulder_l1", "tail_l1", "el1_grazing", "e_prof", "e_h")
BROAD = {"el1": 0.05, "ealb": 0.10, "eshape": 0.10, "shoulder_l1": 0.10, "tail_l1": 0.10, "el1_grazing": 0.10}
CODE = ("experiments/ecro_wl84_104kc/eval104kc.py", "experiments/ecro_wl84_104kc/fit104kc.py",
        "experiments/ecro_wl84_104kc/common104kc.py", "experiments/ecro_wl84_104kc/decision104kc.py")


def cohorts():
    rows = CM.inputs()[0]
    dev = [i for i, r in enumerate(rows) if r["cohort"] == "development"]
    return {"all": list(range(N)), "development": dev, "held_out": [i for i in range(N) if i not in dev]}


def boot_index(coh):
    seed = int(C.P["seeds"]["bootstrap"])
    return {c: np.random.default_rng(seed).integers(0, len(ms), (EB.NB, len(ms))) for c, ms in coh.items()}


def fit_states(root, rel, R):
    d = root / rel
    fj = C.read(d / "fit.json")
    if C.sha(d / "states.npz") != fj["states_sha256"]:
        raise Stop(f"STATES_HASH: {rel}")
    if fj["parameters_per_material"] != 178:
        raise Stop(f"CAPACITY: {rel}")
    st = C.load_states(d / "states.npz", R)
    for s in st:
        if not (np.isfinite(s["y"]).all() and np.isfinite(s["g"]).all()):
            raise Stop(f"NONFINITE_STATE: {rel}")
    return st, fj


def physical_states(root):
    out, sha = {}, {}
    for a, (rel, R) in PHYS.items():
        if (root / rel / "states.npz").exists():
            out[a], _ = fit_states(root, rel, R)
            sha[a] = C.sha(root / rel / "states.npz")
    seen = C.authenticate_kb(["primary/fits/E2/states.npz", "primary/fits/E2/fit.json"])
    out["E2_B"] = C.load_states(C.KB_ROOT / "fits/E2/states.npz", 8)
    sha["E2_B"] = seen["results/ecro_wl84_104kb/primary/fits/E2/states.npz"]
    hist, _, seen_k = C.archived_P()
    out["P_hist"] = hist
    sha["P_hist"] = seen_k["results/ecro_wl84_104k/primary/fit_primary/states.npz"]
    return out, sha


def synthetic_states(root):
    out, sha = {}, {}
    for a, (rel, _) in SYN.items():
        if (root / rel / "states.npz").exists():
            out[a], _ = fit_states(root, rel, 4)
            sha[a] = C.sha(root / rel / "states.npz")
    return out, sha


def truth_state():
    raw = CM.inputs()[1]
    return KB.synthetic_truth(raw["R0"])


# ------------------------------------------------------------------ metrics
PROFILES = {}                                                       # ring residual profiles [R, n, P, ring] of the last evaluate_all


def evaluate_all(states, ds, pk):
    M = {}
    PROFILES.clear()
    for a, st in states.items():
        M[a], PROFILES[a] = EB.evaluate("P", st, ds, pk)
        print("evaluated", a, flush=True)
    return M


def truth_sets(ds, pk, truth):
    """Noise-free truth targets on a lane set / peak grid (synthetic evaluation)."""
    tgt = EB.predict_units("P", [truth], ds["wi"], ds["wo"])[0]
    tl = EB.predict_units("P", [truth], pk["peak_wi"], pk["peak_wo"])[0] @ C.LUMA
    dst = {**ds, "mean": tgt, "a": tgt, "b": tgt}
    pkt = {**pk, "ref": tl, "ref_a": tl, "ref_b": tl}
    return dst, pkt


def null_contrasts(M, coh, idx):
    """Batch-size numerical null: R4 vs O0 starts 1-4 and CR_R4 vs CR starts 1-4 (pooled median relative change per metric)."""
    out = {}
    for name, X, Y in (("R4_vs_O0", "R4", "O0"), ("CR_R4_vs_CR", "CR_R4", "CR")):
        if X not in M or Y not in M:
            continue
        out[name] = {}
        for k in KEYS:
            x = M[X][k][:4]
            y = M[Y][k][:4]
            out[name][k] = EB.pooled_contrast(x, y, coh["all"], idx["all"])["median"]
        out[name]["instability"] = float(EB.instability(M[X]["s_h"][:4]).mean() / EB.instability(M[Y]["s_h"][:4]).mean() - 1)
    if "CR_R4" in M and "R4" in M and "CR" in M:
        out["contrast_difference_CR_vs_O0"] = {}
        for k in KEYS:
            c4 = EB.pooled_contrast(M["CR_R4"][k][:4], M["R4"][k][:4], coh["all"], idx["all"])["median"]
            c8 = EB.pooled_contrast(M["CR"][k][:4], M["O0"][k][:4], coh["all"], idx["all"])["median"]
            out["contrast_difference_CR_vs_O0"][k] = c4 - c8
    return out


def null_level(nulls, k):
    v = [abs(nulls[s][k]) for s in ("R4_vs_O0", "CR_R4_vs_CR", "contrast_difference_CR_vs_O0") if s in nulls and k in nulls[s]]
    return max(v) if v else None


def labels(con_O0, con_O0C, MX, MO, MP, nulls):
    pe = con_O0["e_prof"]["all"]
    nl = null_level(nulls, "e_prof")
    num_ok = nl is not None and abs(pe["median"]) > 3 * nl
    pc = con_O0C["e_prof"]["all"] if con_O0C is not None else None
    peak = bool(pe["median"] <= -0.15 and pe["hi"] < 0 and pe["resolved"] and num_ok)
    compute_ok = bool(pc is not None and pc["median"] <= -0.15 and pc["hi"] < 0)
    st = con_O0["instability"]["all"]
    nli = null_level(nulls, "instability")
    stab = bool(st["median"] <= -0.25 and st["hi"] < 0 and st["resolved"] and nli is not None and abs(st["median"]) > 3 * nli)
    mat = np.median(MX["el1"].mean(-1), 0) / np.median(MO["el1"].mean(-1), 0) - 1
    vs_p = float(np.median(MX["el1"].sum((1, 2))) / np.median(MP["el1"].sum((1, 2))) - 1)
    rule = {f"{k}_le_+{int(100 * t)}pct": bool(con_O0[k]["all"]["median"] <= t) for k, t in BROAD.items()}
    rule["el1_vs_P_hist_le_+5pct"] = bool(vs_p <= 0.05)
    rule["no_material_el1_gt_+10pct"] = bool((mat <= 0.10).all())
    broad = all(rule.values())
    if peak and broad:
        lab = "QUALIFIED_IMPROVEMENT"
    elif peak:
        lab = "TRADE_OFF"
    elif broad:
        lab = "PRESERVED_NO_PEAK_GAIN"
    else:
        lab = "REGRESSION"
    if peak and con_O0C is not None and not compute_ok:            # compute control applies to CR only
        lab += "+COMPUTE_CONFOUNDED"
    return {"label": lab, "PEAK_GAIN": peak, "PEAK_GAIN_vs_O0C": compute_ok if con_O0C is not None else None, "e_prof_numerical_null": nl,
            "e_prof_numerically_resolved": bool(num_ok), "STABILITY_GAIN": stab, "instability_numerical_null": nli,
            "BROAD_PRESERVED": bool(broad), "broad_rule": rule, "el1_vs_P_hist": vs_p, "material_el1_change": mat}


def summaries(M, coh):
    return {a: EB.summary(m, coh) for a, m in M.items()}


# ------------------------------------------------------------------ design (dev only)
def run_design(out, dirs):
    coh = cohorts()
    idx = boot_index(coh)
    ds, pk = EB.load_sets(C.KB_ROOT, "dev", N)
    C.authenticate_kb(["primary/fits/O0/states.npz", "primary/data/dev.npz", "primary/data/dev_peak.npz"])
    states = {"O0": C.load_states(C.KB_ROOT / "fits/O0/states.npz", 8)}
    states["E2_B"] = C.load_states(C.KB_ROOT / "fits/E2/states.npz", 8)
    for d in dirs:
        states[Path(d).name], _ = fit_states(Path(d).parent, Path(d).name, 8)
    M = evaluate_all(states, ds, pk)
    res = {"set": "WL104K-B dev (design use only)", "contrasts_vs_O0": {}, "summary": {}}
    for a in M:
        if a == "O0":
            continue
        con = EB.contrast_table(M[a], M["O0"], coh, idx, keys=KEYS)
        res["contrasts_vs_O0"][a] = {k: {q: v for q, v in con[k]["all"].items() if q != "per_start"} for k in con}
    res["summary"] = {a: {k: float(np.median(M[a][k].sum((1, 2)))) for k in KEYS} for a in M}
    C.write(out, EB.clean(res))
    for a, c in res["contrasts_vs_O0"].items():
        print(a, {k: round(v["median"], 4) for k, v in c.items()}, flush=True)


# ------------------------------------------------------------------ fit decision (before the test is opened)
def truth_feasibility(root, name, data_kind):
    """For a synthetic CR arm: are the truth parameters feasible for the arm's own training bounds (noisy synthetic targets)?"""
    d = root / SYN[name][0]
    if not (d / "bounds.npz").exists():
        return None
    rows, raw, _, _, _, _ = CM.inputs()
    ds = KB.training_set(data_kind, N, C.KB_ROOT / "data")
    truth, target = FC.synthetic_targets(ds, raw)
    T = FC.Terms(ds, N, 4, target)
    x = np.concatenate([truth["y"], truth["g"].reshape(N, -1)], 1)
    xt = torch.as_tensor(np.tile(x, (4, 1)), device=C.DEV, dtype=torch.float32)
    with torch.no_grad():
        t = T.probe_terms(xt)
    g = torch.stack([t[k] for k in FC.CON], 1).cpu().numpy()
    with np.load(d / "bounds.npz") as z:
        b = z["bounds"]
    cv = (g / b - 1).max((1, 2))
    return {"units": int(len(cv)), "truth_feasible_units": int((cv <= 0).sum()), "truth_max_violation_per_unit": cv.tolist(),
            "truth_core_F_per_unit": (t["e_core"].mean(-1).cpu().numpy() / 1.0).tolist()}


def training_report(fj):
    ck = np.asarray(fj["returned_checkpoint"])
    h = fj["checkpoint_history"]
    rep = {"returned_checkpoint": ck.tolist(), "units_moved_from_warm_start": int((ck > 0).sum()), "units": int(len(ck))}
    if h and "F" in h[-1]:
        F = np.asarray([r["F"] for r in h])                         # [K, U]
        feas = np.asarray([r["feasible"] for r in h])
        mv = np.asarray([r["max_violation"] for r in h])
        Fret = np.where(ck > 0, F[np.maximum(ck - 1, 0), np.arange(len(ck))], 1.0)
        rep.update({"F_returned": Fret.tolist(), "F_returned_median": float(np.median(Fret)),
                    "units_F_le_0.9": int((Fret <= 0.9).sum()), "feasible_fraction_per_checkpoint": feas.mean(1).tolist(),
                    "max_violation_median_per_checkpoint": np.median(mv, 1).tolist(),
                    "F_last_two_rel_change_median": float(np.median(np.abs(F[-1] - F[-2]) / np.maximum(F[-2], 1e-12)))
                    if len(F) > 1 else None,
                    "final_rho_max": float(np.max(h[-1]["rho"])), "final_active_constraints_median": float(np.median(h[-1]["active_constraints"])),
                    "kkt_projected_gradient_median": float(np.median(fj["kkt"]["projected_lagrangian_gradient_norm"]))})
    else:
        crit = np.asarray([r["criterion"] for r in h])
        rep.update({"criterion_last_two_rel_change_median": float(np.median(np.abs(crit[-1] - crit[-2]) / np.abs(crit[-2])))
                    if len(crit) > 1 else None})
    ini, fin = fj["initial_probe_terms"], fj["final_probe_terms"]
    rep["training_terms_rel_change_median"] = {k: float(np.median(np.mean(fin[k], -1) / np.mean(ini[k], -1) - 1)) for k in ini}
    rep["seconds"] = fj["seconds"]
    rep["peak_allocated_bytes"] = fj["peak_allocated_bytes"]
    return rep


def run_fitdecision(root):
    t0 = time.monotonic()
    out = root / "fit_decision"
    if (out / "fit_decision.json").exists():
        raise Stop("FIT_DECISION_EXISTS")
    missing = [a for a, (rel, _) in {**PHYS, **{k: (v[0], 4) for k, v in SYN.items()}}.items() if not (root / rel / "states.npz").exists()]
    if missing:
        raise Stop(f"FITS_MISSING: {missing}")
    con = C.read(root / "contracts/fitted.json")
    if not con.get("passed"):
        raise Stop("CONTRACTS_FAILED: contracts/fitted.json")
    phys, psha = physical_states(root)
    syn, ssha = synthetic_states(root)
    rep = {a: training_report(C.read(root / PHYS[a][0] / "fit.json")) for a in ("CR", "O0C", "E2W", "CR_R4")}
    rep.update({a: training_report(C.read(root / SYN[a][0] / "fit.json")) for a in ("CR_S", "O0C_S", "E2W_S", "CR_SD", "O0C_SD")})
    tf = {a: truth_feasibility(root, a, SYN[a][1]) for a in ("CR_S", "CR_SD")}
    # DEV descriptive (WL104K-B dev; no decision depends on it)
    coh = cohorts()
    idx = boot_index(coh)
    ds, pk = EB.load_sets(C.KB_ROOT, "dev", N)
    Md = evaluate_all({a: phys[a] for a in ("O0", "CR", "O0C", "E2W", "E2_B")}, ds, pk)
    dev = {a: {k: {q: v for q, v in c["all"].items() if q != "per_start"} for k, c in
               EB.contrast_table(Md[a], Md["O0"], coh, idx, keys=KEYS).items()} for a in ("CR", "O0C", "E2W", "E2_B")}
    res = {"training": rep, "truth_feasibility": tf, "dev_descriptive_contrasts_vs_O0": dev,
           "fits_sha256": {**psha, **ssha}, "fit_json_sha256": {a: C.sha(root / rel / "fit.json") for a, (rel, _) in PHYS.items()},
           "contracts_sha256": C.sha(root / "contracts/fitted.json"), "protocol_sha256": C.sha(C.PROTOCOL),
           "stageA_sha256": C.sha(root / "stageA/stageA.json"), "code_sha256": {k: C.sha(C.REPO / k) for k in CODE},
           "data_sha256": {k: C.sha(root / "data" / k) for k in ("data.json",)}, "commit": C.commit(),
           "seconds": time.monotonic() - t0}
    C.write(out / "fit_decision.json", EB.clean(res))
    (out / "fit_decision.json.sha256").write_text(C.sha(out / "fit_decision.json") + "  fit_decision.json\n", encoding="utf-8")
    print("fit decision written", {a: (r.get("F_returned_median"), r["units_moved_from_warm_start"]) for a, r in rep.items()}, flush=True)


def test_lock(root):
    """Refuse unless the hashed fit decision exists and every recorded fit / protocol / code / data hash is unchanged."""
    fd, side = root / "fit_decision/fit_decision.json", root / "fit_decision/fit_decision.json.sha256"
    if not (fd.exists() and side.exists()):
        raise Stop("TEST_LOCKED: hashed fit decision missing")
    if C.sha(fd) != side.read_text(encoding="utf-8").split()[0]:
        raise Stop("TEST_LOCKED: fit decision hash mismatch")
    d = C.read(fd)
    rels = {**{a: rel for a, (rel, _) in PHYS.items()}, **{a: v[0] for a, v in SYN.items()}}
    for a, h in d["fits_sha256"].items():
        if a in rels and C.sha(root / rels[a] / "states.npz") != h:
            raise Stop(f"TEST_LOCKED: {a} changed after the fit decision")
    if d["protocol_sha256"] != C.sha(C.PROTOCOL):
        raise Stop("TEST_LOCKED: protocol changed")
    for k, h in d["code_sha256"].items():
        if C.sha(C.REPO / k) != h:
            raise Stop(f"TEST_LOCKED: code changed {k}")
    for k, h in d["data_sha256"].items():
        if C.sha(root / "data" / k) != h:
            raise Stop(f"TEST_LOCKED: data changed {k}")
    return d


def load_test_c(root, smoke=False):
    dj = C.read(root / "data/data.json")
    if (not dj["qualified"]["all"] or dj["smoke"]) and not (smoke and dj["smoke"]):
        raise Stop("DATA_UNQUALIFIED: test_c")
    for f, h in dj["outputs_sha256"].items():
        if C.sha(root / "data" / f) != h:
            raise Stop(f"DATA_HASH: {f}")
    lp, pp = root / "data/test_c.npz", root / "data/test_c_peak.npz"
    ds = KB.load_lane_set(lp, N)
    with np.load(lp) as z:
        polar = z["polar_deg"].reshape(N, -1)
    ds["rad"] = KB.mirror_radius(ds["wi"], ds["wo"])
    ds["graze"] = (polar >= EB.GRAZING_DEG).astype(float)
    ds["polar"] = polar
    pk = DK.load_peak(pp, N, guard=False)
    return ds, pk


def kb_test_sets():
    """WL104K-B test lanes + peak grid (opened in WL104K-B; regression evidence only), authenticated by the WL104K-B manifest and
    loaded without the WL104K-B test lock (that lock now refuses because decision104kb.py received the disclosed post-test
    execution fix c8bf685; the arrays themselves are unchanged and hash-verified here)."""
    C.authenticate_kb(["primary/data/test.npz", "primary/data/test_peak.npz"])
    lp, pp = C.KB_ROOT / "data/test.npz", C.KB_ROOT / "data/test_peak.npz"
    ds = KB.load_lane_set(lp, N)
    with np.load(lp) as z:
        polar = z["polar_deg"].reshape(N, -1)
    ds["rad"] = KB.mirror_radius(ds["wi"], ds["wo"])
    ds["graze"] = (polar >= EB.GRAZING_DEG).astype(float)
    ds["polar"] = polar
    return ds, DK.load_peak(pp, N, guard=False)


def run_test(root, smoke=False):
    fd = test_lock(root)                                            # before ANY test_c array is opened
    out = root / "test_eval"
    if (out / "test.json").exists():
        raise Stop("TEST_EXISTS")
    t0 = time.monotonic()
    coh = cohorts()
    idx = boot_index(coh)
    ds, pk = load_test_c(root, smoke)
    phys, psha = physical_states(root)
    if any(psha[a] != fd["fits_sha256"][a] for a in psha if a in fd["fits_sha256"]):
        raise Stop("TEST_LOCKED: physical states differ from the fit decision")
    M = evaluate_all(phys, ds, pk)
    prof = dict(PROFILES)
    con = {}
    for X in ("CR", "O0C", "E2W", "E2_B", "R4", "CR_R4"):
        if X in ("R4",):
            continue
        con[X] = EB.contrast_table(M[X], M["O0"], coh, idx, keys=KEYS)
    con["CR_vs_O0C"] = EB.contrast_table(M["CR"], M["O0C"], coh, idx, keys=KEYS)
    con["CR_vs_E2_B"] = EB.contrast_table(M["CR"], M["E2_B"], coh, idx, keys=KEYS)
    nulls = null_contrasts(M, coh, idx)
    lab = {X: labels(con[X], con["CR_vs_O0C"] if X == "CR" else None, M[X], M["O0"], M["P_hist"], nulls)
           for X in ("CR", "O0C", "E2W", "E2_B")}
    # synthetic truth recovery on test_c lanes / peak grid (noise-free truth)
    syn, ssha = synthetic_states(root)
    dst, pkt = truth_sets(ds, pk, truth_state())
    MS = evaluate_all(syn, dst, pkt)
    syn_pooled = {a: {k: float(np.median(m[k].sum((1, 2)))) for k in KEYS} for a, m in MS.items()}
    syn_inst = {a: float(EB.instability(m["s_h"]).mean()) for a, m in MS.items()}
    sc = {}
    for X, base in (("CR_S", "S_O0r"), ("O0C_S", "S_O0r"), ("E2W_S", "S_O0r"), ("CR_SD", "S_Dr"), ("O0C_SD", "S_Dr")):
        r = {k: syn_pooled[X][k] / syn_pooled[base][k] for k in KEYS}
        sc[X] = {"vs": base, "ratio": r, "per_start_e_prof_ratio": (MS[X]["e_prof"].sum((1, 2)) / MS[base]["e_prof"].sum((1, 2))).tolist()}
    sc["SYN_RECOVERY_CR_S"] = bool(sc["CR_S"]["ratio"]["e_prof"] <= 0.8 and sc["CR_S"]["ratio"]["el1"] <= 1.05)
    sc["SYN_RECOVERY_CR_S_vs_O0C_S"] = bool(syn_pooled["CR_S"]["e_prof"] <= 0.8 * syn_pooled["O0C_S"]["e_prof"])
    sc["SYN_RECOVERY_CR_SD"] = bool(sc["CR_SD"]["ratio"]["e_prof"] <= 0.8 and sc["CR_SD"]["ratio"]["el1"] <= 1.05)
    sc["data_CR_SD_vs_CR_S_e_prof"] = syn_pooled["CR_SD"]["e_prof"] / syn_pooled["CR_S"]["e_prof"]
    sc["data_S_Dr_vs_S_O0r_e_prof"] = syn_pooled["S_Dr"]["e_prof"] / syn_pooled["S_O0r"]["e_prof"]
    # WL104K-B test set (opened in WL104K-B: development / regression evidence only): descriptive contrasts vs O0
    dsb, pkb = kb_test_sets()
    MB = evaluate_all({a: phys[a] for a in ("O0", "CR", "O0C", "E2W", "E2_B")}, dsb, pkb)
    reg = {X: {k: {q: v for q, v in c["all"].items() if q != "per_start"} for k, c in
               EB.contrast_table(MB[X], MB["O0"], coh, idx, keys=KEYS).items()} for X in ("CR", "O0C", "E2W", "E2_B")}
    res = {"set": "WL104K-C test_c (independent)", "contrasts": {X: {k: {c: {q: v for q, v in d.items()} for c, d in t.items()}
                                                                    for k, t in con[X].items()} for X in con},
           "labels": lab, "numerical_null": nulls, "regression_wl104kb_test_vs_O0": reg, "summary": summaries(M, coh), "synthetic": {"pooled_median": syn_pooled,
                                                                                               "instability": syn_inst,
                                                                                               "controls": sc},
           "fits_sha256": {**psha, **ssha}, "fit_decision_sha256": C.sha(root / "fit_decision/fit_decision.json"),
           "protocol_sha256": C.sha(C.PROTOCOL), "commit": C.commit(), "seconds": time.monotonic() - t0}
    arrays = {f"{a}__{k}": M[a][k] for a in M for k in KEYS + ("s_h",)}
    arrays.update({f"{a}__ring_residual": v for a, v in prof.items()})
    arrays["ring_radius_deg"] = EB.rings(pk)[0]
    arrays["polar_deg"] = ds["polar"]
    arrays.update({f"synthetic__{a}__{k}": MS[a][k] for a in MS for k in KEYS + ("s_h",)})
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "arrays.npz", **arrays)
    C.write(out / "test.json", EB.clean(res))
    print("test labels", {X: v["label"] for X, v in lab.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("design", "fitdecision", "test"))
    ap.add_argument("path", type=Path)
    ap.add_argument("dirs", nargs="*", type=Path)
    ap.add_argument("--smoke", action="store_true", help="test mode: accept smoke test_c data (scratch roots only)")
    a = ap.parse_args()
    if a.mode == "design":
        run_design(a.path, a.dirs)
    elif a.mode == "fitdecision":
        run_fitdecision(a.path)
    else:
        run_test(a.path, a.smoke)


if __name__ == "__main__":
    main()
