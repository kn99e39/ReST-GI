"""WL104K-B decision table: mechanism by mechanism (O1 optimization, D coverage, E2 objective, E1 selection, F combination)
from the independent angular test, the synthetic controls, the dev stage-F decision and the scene replays, plus the overall
answers A-G with their evidence class. Nothing is re-measured here; missing inputs are reported as NOT_AVAILABLE.

  python decision104kb.py <root>

Reads <root>/test_eval/test.json (+ arrays.npz), dev_eval/dev.json, contracts/*.json ('passed'), scenes/<scene>/analysis104kb.json
(scenes_new/<scene>/analysis104kb.json when not under scenes/), scenes_new/manifest.json, logs/references.txt, data/data.json.
Writes <root>/decision.json.
Gating: the per-row angular label is the test evaluation's label (already gated there by data / isolation / O0 reproduction;
'angular_label_eval'), with the measured label kept as 'angular_label_measured'. The global status is FAILED only for a failed
contract (pre_fit, data, fitted_<arm>, leak) or a dev-decision hash mismatch, NOT_AVAILABLE when an expected contract is missing,
otherwise PASSED. Scene rule (protocol stage_G_scenes.scene_rule) is re-applied to each stored contrast {'point', 'lo', 'hi'}; a
stored label that disagrees is reported. A scene whose replayed states (analysis fits[arm].states_sha256) differ from the states
hashed by the test evaluation (test.json fits_sha256) or the dev decision is UNQUALIFIED.
"""
import argparse
from pathlib import Path

import numpy as np

import common104kb as KB
import eval104kb as E

NA = "NOT_AVAILABLE"
ROWS = {"O1": "Optimization", "D": "Coverage", "E2": "Objective", "E1": "Selection", "F": "Combination"}
CONTROL = {"O1": "SC_O1", "D": "SC_D", "E2": "SC_E2", "E1": "SC_E1"}
ARCHIVED = ("test83_iid_1_L2", "test83_hard_1_L1", "test83_iid_159_L2")
NEW = tuple(KB.P.get("seeds", {}).get("scene_seed_bases", {}).keys())
INDEPENDENT = ("test83_iid_159_L2",) + NEW
E1_KEYS = ("E1", "O0_E1", "E1_O0")
FITTED = ("O0", "O1", "D", "E2", "S_O0", "S_O1", "S_D", "S_E2")


def read(p):
    return KB.read(p) if p.exists() else None


UNRES = "UNRESOLVED_NOISE_FLOOR"


def scene_label(c):
    if not isinstance(c, dict) or c.get("lo") is None or c.get("hi") is None:
        return NA
    if c["hi"] < 0:
        return "IMPROVED"
    if c["lo"] > 0.02:
        return "REGRESSED"
    return "NEUTRAL"


def fits_mismatch(a, test, dev):
    """Replayed states vs the hashed states of the test evaluation and the dev decision -> list of mismatch notes."""
    notes = []
    fits = a.get("fits") or {}
    for src, ref in (("test.json", (test or {}).get("fits_sha256")), ("dev.json", (dev or {}).get("fits_sha256"))):
        if not isinstance(ref, dict):
            continue
        for arm, f in fits.items():
            h = f.get("states_sha256") if isinstance(f, dict) else None
            if h is None or arm not in ref:
                continue
            if h != ref[arm]:
                notes.append(f"fits/{arm} states_sha256 {h[:12]} != {src} {ref[arm][:12]}")
    return notes


def load_scenes(root, test=None, dev=None):
    out = {}
    for sub in ("scenes", "scenes_new"):
        for p in sorted((root / sub).glob("*/analysis104kb.json")):
            a = KB.read(p)
            name = a.get("scene", p.parent.name)
            if name not in out:
                out[name] = {**a, "_path": p.relative_to(root).as_posix(), "_sha256": KB.sha(p),
                             "_fits_mismatch": fits_mismatch(a, test, dev),
                             "_fits_not_replayed": [x for x in ((test or {}).get("fits_sha256") or {}) if x not in (a.get("fits") or {})],
                             "_fits_checked": bool(test is not None or dev is not None)}
    return out


def new_scenes_report(root, scenes):
    """Predeclared new scenes: attempted / qualified / failed_precision / errored (scenes_new/manifest.json, logs/references.txt)."""
    man = read(root / "scenes_new/manifest.json")
    done = (man or {}).get("scenes", {})
    errored = []
    ref_log = root / "logs/references.txt"
    if ref_log.exists():
        for line in ref_log.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[:2] == ["reference", "error"]:
                errored.append(parts[2])
    errored = [x for i, x in enumerate(errored) if x not in errored[:i]]
    attempted = [x for x in NEW if x in done or x in errored]
    qualified = [x for x in NEW if bool(done.get(x, {}).get("precision_passed"))]
    failed = [x for x in NEW if x in done and not done[x].get("precision_passed")]
    analysed = [x for x in qualified if x in scenes and scenes[x].get("identity_passed") and not scenes[x]["_fits_mismatch"]]
    return {"attempted": attempted, "qualified": qualified, "failed_precision": failed,
            "errored": [x for x in errored if x not in qualified], "qualified_and_analysed": analysed,
            "manifest_sha256": KB.sha(root / "scenes_new/manifest.json") if man is not None else None,
            "new_scene_evidence": "AVAILABLE" if analysed else "UNAVAILABLE"}


def scene_column(scenes, arm, run=True):
    """Per scene label for one row and the SCENE_SUPPORTED verdict."""
    labels, notes = {}, []
    for name in sorted(set(ARCHIVED) | set(scenes)):
        a = scenes.get(name)
        if not run:
            labels[name] = "NOT_RUN"
            continue
        if a is None:
            labels[name] = NA
            continue
        if not a.get("identity_passed", False):
            labels[name] = "UNQUALIFIED"
            continue
        if a.get("_fits_mismatch"):
            labels[name] = "UNQUALIFIED"
            notes.append(f"{name}: UNQUALIFIED, replayed states differ from the hashed states: {a['_fits_mismatch']}")
            continue
        con = a.get("contrasts", {})
        c = next((con[k] for k in (E1_KEYS if arm == "E1" else (arm,)) if k in con), None)
        tot = c.get("total") if isinstance(c, dict) else None
        lab = UNRES if isinstance(tot, dict) and tot.get("label") == UNRES else scene_label(tot)
        if tot is not None and tot.get("label") not in (None, lab):
            notes.append(f"{name}: stored label {tot.get('label')} != rule {lab}")
        if bool(a.get("independent")) != (name in INDEPENDENT):
            notes.append(f"{name}: analysis independent={a.get('independent')} differs from the protocol classification")
        labels[name] = lab
        if isinstance(c, dict) and isinstance(c.get("e1"), dict):
            labels[name + ":e1_first_bounce"] = UNRES if c["e1"].get("label") == UNRES else scene_label(c["e1"])
    if not run:
        return labels, "NOT_RUN", notes
    main = {k: v for k, v in labels.items() if ":" not in k}
    if any(v == "REGRESSED" for v in main.values()):
        verdict = "SCENE_NOT_SUPPORTED"
    elif any(v == "UNQUALIFIED" for v in main.values()):
        verdict = "UNQUALIFIED"
    elif any(v == NA for v in main.values()):
        verdict = NA
    elif any(v == UNRES for v in main.values()):
        # execution fix (disclosed): a scene whose O0 noise-debiased RMS is clipped at the noise floor cannot show "no regression";
        # conservatively INCONCLUSIVE, never SUPPORTED
        verdict = "INCONCLUSIVE_NOISE_FLOOR"
    elif any(v == "IMPROVED" for k, v in main.items() if k in INDEPENDENT):
        verdict = "SCENE_SUPPORTED"
    else:
        verdict = "SCENE_NOT_SUPPORTED"
    return labels, verdict, notes


def expected_contracts(dev):
    exp = ["pre_fit", "data"] + [f"fitted_{a}" for a in FITTED] + ["leak"]
    if dev is not None and dev.get("stage_F", {}).get("run"):
        exp.insert(-1, "fitted_F")
    return exp


def qualification(root, test, dev):
    """Global status: FAILED only for a failed contract / leak (or dev hash, set by main); NOT_AVAILABLE when an expected contract
    is missing; PASSED otherwise. Data / isolation / O0-reproduction qualification gates the test labels themselves (reported)."""
    files = sorted((root / "contracts").glob("*.json"))
    found = {p.stem: bool(KB.read(p).get("passed", False)) for p in files}
    exp = expected_contracts(dev)
    con = {c: found.get(c, NA) for c in exp}
    extra = {c: v for c, v in found.items() if c not in exp}
    dj = read(root / "data/data.json")
    q = (dj or {}).get("qualified", (dj or {}).get("qualification"))
    dq = {s: (q.get(s) if isinstance(q, dict) else q) for s in ("train_coverage", "dev", "dev_peak", "test", "test_peak", "isolation", "all")} if dj else {}
    dq = {s: (v.get("all", v.get("passed")) if isinstance(v, dict) else v) for s, v in dq.items()}
    failed = [c for c, v in {**con, **extra}.items() if v is False]
    missing = [c for c, v in con.items() if v == NA]
    status = "FAILED" if failed else (NA if missing else "PASSED")
    return {"contracts": con, "contracts_unexpected": extra or None, "contracts_failed": failed, "contracts_missing": missing,
            "data": dq or NA, "test_qualified": (test or {}).get("qualified", NA),
            "test_qualification": (test or {}).get("qualification", NA), "test_flags": (test or {}).get("flags", NA),
            "status": status}


def mechanism_row(X, test, dev, sc, qual, scenes):
    run = not (X == "F" and dev is not None and not dev["stage_F"]["run"])
    row = {"mechanism": ROWS[X], "stage_F_eligible_dev": (dev["stage_F"]["eligibility"].get(X, {}).get("eligible")
                                                         if dev is not None and X in ("O1", "D", "E2") else None)}
    if not run:
        row.update({k: "NOT_RUN" for k in ("angular_label", "peak_gain", "stability_gain", "broad_preserved", "control")})
    elif test is None or X not in test.get("labels", {}):
        row.update({k: NA for k in ("angular_label", "peak_gain", "stability_gain", "broad_preserved")})
        row["control"] = sc.get(CONTROL.get(X), {}).get("passed", NA) if X != "F" else NA
    else:
        lab = test["labels"][X]
        c = test["contrasts_vs_O0"][X]
        row.update({"angular_label_measured": lab["measured_label"], "angular_label_eval": lab["label"],
                    "flags": lab.get("flags", []), "peak_gain": lab["PEAK_GAIN"], "stability_gain": lab["STABILITY_GAIN"],
                    "broad_preserved": lab["BROAD_PRESERVED"], "control": lab["control_passed"] if lab["control_passed"] is not None else NA,
                    "positive_with_failed_control": lab["positive_with_failed_control"],
                    "e_prof_change": {k: c["e_prof"]["all"][k] for k in ("median", "lo", "hi", "resolved")},
                    "instability_change": {k: c["instability"]["all"][k] for k in ("median", "lo", "hi", "resolved")},
                    "el1_change": {k: c["el1"]["all"][k] for k in ("median", "lo", "hi")},
                    "cohort_e_prof_change": {coh: c["e_prof"][coh]["median"] for coh in c["e_prof"]}})
        row["angular_label"] = ("UNQUALIFIED" if qual["status"] == "FAILED" else NA if qual["status"] == NA else lab["label"])
    row["scenes"], row["scene_verdict"], row["scene_notes"] = scene_column(scenes, X, run)
    return row


def selection_row(test, sc, qual, scenes):
    row = {"mechanism": ROWS["E1"], "stage_F_eligible_dev": None}
    e = (test or {}).get("selection_E1", {}).get("primary_O0")
    ctl = sc.get("SC_E1", {}).get("passed")
    if e is None:
        row.update({k: NA for k in ("angular_label", "predictive", "selection_effective")})
    else:
        measured = "SELECTION_EFFECTIVE" if e["SELECTION_EFFECTIVE"] else "SELECTION_NOT_EFFECTIVE"
        lab = measured
        if not e["SELECTION_EFFECTIVE"] and ctl is False:
            lab += "+CONTROL_NOT_DEMONSTRATED"
        lab = "UNQUALIFIED" if e.get("qualified") is False else lab + "".join("+" + f for f in e.get("flags", []))
        row.update({"angular_label_measured": measured, "angular_label_eval": lab, "flags": e.get("flags", []),
                    "predictive": e["PREDICTIVE"], "selection_effective": e["SELECTION_EFFECTIVE"],
                    "n_positive_rho": e["n_positive_rho"], "median_rho": e["median_rho"], "rule": e["rule"],
                    "e_prof_change_vs_median_start": e["contrast_vs_median_start"]})
        row["angular_label"] = "UNQUALIFIED" if qual["status"] == "FAILED" else NA if qual["status"] == NA else lab
    row["control"] = ctl if ctl is not None else NA
    row["scenes"], row["scene_verdict"], row["scene_notes"] = scene_column(scenes, "E1")
    return row


def mech_answer(row, what):
    lab = row.get("angular_label", NA)
    ev = ["independent angular"]
    if row.get("control") not in (NA, None, "NOT_RUN"):
        ev.append("synthetic control")
    if lab in (NA, "UNQUALIFIED", "NOT_RUN"):
        return {"answer": f"UNRESOLVED ({lab})", "evidence_class": ["unresolved"]}
    gains = [g for g, k in (("peak", "peak_gain"), ("stability", "stability_gain")) if row.get(k) is True]
    if gains:
        ans = f"{what} of the O0 contract NOT adequate for {' and '.join(gains)} (intervention gain measured; label {lab})"
        if row.get("broad_preserved") is not True:
            ans += "; gain without broad preservation (trade-off, not a success)"
    elif row.get("control") is True:
        ans = f"no {what.lower()} deficit detected by this intervention (label {lab}; control demonstrated)"
    else:
        ans = f"UNRESOLVED: no gain and the synthetic control was not demonstrated (label {lab})"
        ev = ["unresolved"]
    return {"answer": ans, "evidence_class": ev}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--test", type=Path, default=None, help="alternative test json (Amendment 1 amended labels)")
    ap.add_argument("--out", type=Path, default=None, help="alternative output path (default <root>/decision.json)")
    args = ap.parse_args()
    root = args.root.resolve()
    test, dev = read(args.test or root / "test_eval/test.json"), read(root / "dev_eval/dev.json")
    if dev is not None and (root / "dev_eval/dev.json.sha256").exists():
        dev_hash_ok = KB.sha(root / "dev_eval/dev.json") == (root / "dev_eval/dev.json.sha256").read_text(encoding="utf-8").split()[0]
    else:
        dev_hash_ok = None
    if test is not None and dev is not None and test.get("dev_json_sha256") != KB.sha(root / "dev_eval/dev.json"):
        dev_hash_ok = False
    scenes = load_scenes(root, test, dev)
    new_scenes = new_scenes_report(root, scenes)
    sc = (test or {}).get("synthetic", {}).get("controls", {})
    qual = qualification(root, test, dev)
    if dev_hash_ok is False:
        qual["status"] = "FAILED"
        qual["dev_decision_hash"] = False
    rows = {X: mechanism_row(X, test, dev, sc, qual, scenes) for X in ("O1", "D", "E2")}
    rows["E1"] = selection_row(test, sc, qual, scenes)
    rows["F"] = mechanism_row("F", test, dev, sc, qual, scenes)
    if rows["F"]["angular_label"] != "NOT_RUN" and dev is not None:
        cs = [sc.get(CONTROL[m], {}).get("passed") for m in dev["stage_F"]["eligible"]]
        rows["F"]["control"] = False if False in cs else (NA if None in cs or not cs else True)
        rows["F"]["combined_mechanisms"] = dev["stage_F"]["eligible"]
    # ---------------- overall answers
    ans = {}
    peak_rows = [X for X in ("O1", "D", "E2", "F") if rows[X].get("peak_gain") is True and rows[X].get("angular_label") not in (NA, "UNQUALIFIED")]
    syn = (test or {}).get("synthetic", {}).get("pooled", {})
    ans["A_representation_adequacy"] = {
        "answer": ("peak error reducible within the unchanged 178-parameter ER-v1-SDM by "
                   + ", ".join(f"{X} ({rows[X]['angular_label']})" for X in peak_rows)) if peak_rows
        else ("NOT_DEMONSTRATED: no qualified fitting intervention produced PEAK_GAIN" if test is not None else f"UNRESOLVED ({NA})"),
        "evidence_class": ["independent angular", "synthetic control"] if peak_rows else (["synthetic control", "unresolved"] if syn else ["unresolved"]),
        "facts": {"synthetic_truth_pooled_e_prof_median": {S: (float(np.median(v["e_prof"])) if v else None) for S, v in syn.items()},
                  "note": "synthetic truth is exactly representable; a residual there is a fitting / data / objective limit, not capacity"}}
    ans["B_optimization_adequacy"] = mech_answer(rows["O1"], "Optimization")
    ans["C_data_adequacy"] = mech_answer(rows["D"], "Angular coverage")
    ans["D_objective_adequacy"] = mech_answer(rows["E2"], "Objective")
    e1 = rows["E1"]
    if e1.get("angular_label", NA) in (NA, "UNQUALIFIED"):
        ans["E_selection_adequacy"] = {"answer": f"UNRESOLVED ({e1.get('angular_label', NA)})", "evidence_class": ["unresolved"]}
    else:
        ans["E_selection_adequacy"] = {"answer": f"{e1['angular_label']} (PREDICTIVE={e1['predictive']}, median rho {e1['median_rho']:.3f})",
                                       "evidence_class": ["independent angular"] + (["synthetic control"] if e1["control"] not in (NA,) else [])}
    sv = {X: rows[X]["scene_verdict"] for X in rows}
    hard = {X: rows[X]["scenes"].get("test83_hard_1_L1", NA) for X in rows}
    indep = {X: {s: v for s, v in rows[X]["scenes"].items() if s in INDEPENDENT} for X in rows}
    ans["F_scene_utility"] = {"answer": sv, "independent_scene_labels": indep, "hard_1_labels": hard,
                              "new_scene_evidence": new_scenes["new_scene_evidence"],
                              "evidence_class": (["independent scene", "historical hard_1 descriptive"] if scenes else ["unresolved"]),
                              "scope": "HIGH held fixed from R0: not an end-to-end ECRO validation"}
    lim = {"unavailable": sorted({f"{X}:{k}" for X, r in rows.items() for k, v in r.items() if v == NA}),
           "failed_controls": sorted(k for k, v in sc.items() if v.get("passed") is False),
           "qualification_status": qual["status"]}
    arrays = root / "test_eval/arrays.npz"
    if test is not None and arrays.exists():
        with np.load(arrays) as z:
            prof = {a: z[f"profile__{a}"] for a in test["arms_evaluated"] if f"profile__{a}" in z.files}
            rad = z["profile_radius_deg"]
        lim["mirror_residual_median"] = {a: float(np.median(p[..., 0])) for a, p in prof.items()}
        lim["ring_residual_median"] = {a: dict(zip([f"{r:g}" for r in rad], np.median(p, (0, 1, 2)).tolist())) for a, p in prof.items()}
        s = test["summary"]
        lim["pooled_test_e_prof_median"] = {a: s[a]["e_prof"]["pooled_median"] for a in s}
        lim["pooled_test_el1_median"] = {a: s[a]["el1"]["pooled_median"] for a in s}
        lim["instability_I"] = {a: s[a].get("instability", {}).get("I") for a in s}
        lim["O0_reproduction_max_relative"] = test.get("O0_reproduction_max_relative")
        lim["O0_reproduction_status"] = (test.get("O0_reproduction") or {}).get("status")
    ans["G_remaining_limitations"] = {"facts": lim, "evidence_class": ["independent angular"] if test is not None else ["unresolved"]}
    res = {"rows": rows, "answers": ans, "qualification": qual, "dev_decision_hash_ok": dev_hash_ok, "new_scenes": new_scenes,
           "stage_F": (dev or {}).get("stage_F", {}).get("eligible", NA), "F_run": (dev or {}).get("stage_F", {}).get("run", NA),
           "scenes_found": {k: {"path": v["_path"], "sha256": v["_sha256"], "independent_protocol": k in INDEPENDENT,
                                "identity_passed": v.get("identity_passed"), "fits_checked": v["_fits_checked"],
                                "fits_mismatch": v["_fits_mismatch"], "test_arms_not_replayed": v["_fits_not_replayed"]}
                           for k, v in scenes.items()},
           "inputs_sha256": {rel: KB.sha(root / rel) for rel in ("test_eval/test.json", "dev_eval/dev.json", "data/data.json",
                                                                 "scenes_new/manifest.json", "logs/references.txt")
                             if (root / rel).exists()},
           "contracts_sha256": {p.name: KB.sha(p) for p in sorted((root / "contracts").glob("*.json"))},
           "commit": KB.commit(), "protocol_sha256": KB.sha(KB.PROTOCOL)}
    if args.test is not None:
        res["test_source"] = {"path": str(args.test), "sha256": KB.sha(args.test), "note": "Amendment 1 amended labels (see amendment1/result.json)"}
    E.write(args.out or root / "decision.json", res)
    print("decision", {X: (r["angular_label"], r["scene_verdict"]) for X, r in rows.items()}, flush=True)


if __name__ == "__main__":
    main()
