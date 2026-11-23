"""WL104K-B Amendment 1: O0 reproduction qualification by measured same-method nondeterminism (amendment1.json).

  python amend104kb.py evaluate <root>          reads <root>/amendment1/fits/{R4a,R4b,R8}, O0 and the archived WL104K P; writes
                                                 <root>/amendment1/result.json and <root>/amendment1/test_amended.json
  python amend104kb.py determinism <root>        compares <root>/scenes_interrupted/<scene>/fixed_r###.npz with the rerun replicates
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

import common104kb as KB

HERE = Path(__file__).resolve().parent
FLAG = "O0_REPRODUCTION_DISCREPANCY"


def final(fit_dir):
    fj = KB.read(fit_dir / "fit.json")
    return np.asarray(fj["final_terms"]["objective"], float), fj


def rel(x, y):
    return np.abs(x - y) / np.abs(y)


def stats(v):
    v = np.asarray(v, float)
    return {"n": int(v.size), "max": float(v.max()), "q95": float(np.quantile(v, .95)), "median": float(np.median(v)),
            "n_above_1e-4": int((v > 1e-4).sum()), "n_above_1e-2": int((v > 1e-2).sum())}


def evaluate(root):
    A = json.loads((HERE / "amendment1.json").read_text(encoding="utf-8"))
    n = 9
    o0, fo = final(root / "fits/O0")
    kfj = KB.read(KB.REPO / "results/ecro_wl84_104k/primary/fit_primary/fit.json")
    arch = np.asarray(kfj["P"]["final_objective"], float).reshape(-1)                 # 4 starts x 9, start-major
    r4a, fa = final(root / "amendment1/fits/R4a")
    r4b, fb = final(root / "amendment1/fits/R4b")
    r8, f8 = final(root / "amendment1/fits/R8")
    for f, R in ((fa, 4), (fb, 4), (f8, 8)):
        if f["start_keys"] != KB.START_KEYS[:R] or f["arm"] != {"method": "adam", "data": "original", "objective": "original",
                                                                 "synthetic": False} or f["budget"] != 12000:
            raise KB.Stop("AMENDMENT_RERUN_SPEC")
    comps = {"batch4_R4a_vs_archive": rel(r4a, arch), "batch4_R4b_vs_archive": rel(r4b, arch), "batch4_R4a_vs_R4b": rel(r4a, r4b),
             "batch8_R8_vs_O0": rel(r8, o0), "batchsize_R8_starts1to4_vs_R4a": rel(r8[:4 * n], r4a),
             "batchsize_R8_starts1to4_vs_R4b": rel(r8[:4 * n], r4b)}
    pooled = np.concatenate(list(comps.values()))
    target = rel(o0[:4 * n], arch)
    T, Pz = stats(target), stats(pooled)
    crit = {"max": T["max"] <= 2 * Pz["max"], "q95": T["q95"] <= 2 * Pz["q95"]}
    verdict = "NONDETERMINISM_CONSISTENT" if all(crit.values()) else "NOT_CONSISTENT"
    names = fo["materials"]
    res = {"amendment": A["id"], "criterion": A["criterion"], "verdict": verdict, "criterion_parts": crit,
           "target_O0_vs_archive": {**T, "per_unit": target.reshape(4, n).tolist()},
           "pooled_reruns": Pz, "components": {k: stats(v) for k, v in comps.items()},
           "materials": names, "rerun_seconds": {"R4a": fa["seconds"], "R4b": fb["seconds"], "R8": f8["seconds"]},
           "rerun_states_sha256": {k: KB.sha(root / f"amendment1/fits/{k}/states.npz") for k in ("R4a", "R4b", "R8")},
           "protocol_sha256": KB.sha(KB.PROTOCOL), "amendment_sha256": KB.sha(HERE / "amendment1.json"), "commit": KB.commit()}
    # amended labels (the frozen test.json is not modified)
    tp = root / "test_eval/test.json"
    test = KB.read(tp)
    t2 = copy.deepcopy(test)
    t2["amendment1"] = {"verdict": verdict, "source_test_sha256": KB.sha(tp), "original_qualified": test["qualified"],
                        "original_labels": {k: v["label"] for k, v in test["labels"].items()}}
    t2["qualification"]["O0_reproduction_amendment1"] = verdict == "NONDETERMINISM_CONSISTENT"
    others = all(v is not False for k, v in test["qualification"].items() if k != "O0_reproduction_within_1e-2")
    q = bool(others and verdict == "NONDETERMINISM_CONSISTENT")
    t2["qualified"] = q
    t2["flags"] = [FLAG] if q else test.get("flags", [])
    for X, lab in t2["labels"].items():
        L = lab["measured_label"]
        if L in ("INEFFECTIVE", "REGRESSION") and lab["control_passed"] is False:
            L += "+CONTROL_NOT_DEMONSTRATED"
        uses_cov = X == "D" or (X == "F" and (test.get("F") or {}).get("spec", {}).get("data") == "coverage")
        lab_q = q and not (uses_cov and test["data"]["qualified"]["train_coverage"] is False)
        lab["label"] = (L + "+" + FLAG) if lab_q else "UNQUALIFIED"
        lab["flags"] = [FLAG] if lab_q else lab.get("flags", [])
        lab["qualified"] = bool(lab_q)
    e = t2.get("selection_E1", {}).get("primary_O0")
    if e is not None:
        e["qualified"] = q
        e["flags"] = [FLAG] if q else e.get("flags", [])
    (root / "amendment1").mkdir(exist_ok=True)
    KB.write(root / "amendment1/test_amended.json", t2)
    res["test_amended_sha256"] = KB.sha(root / "amendment1/test_amended.json")
    KB.write(root / "amendment1/result.json", res)
    print("AMENDMENT1", verdict, {k: round(v, 5) for k, v in (("target_max", T["max"]), ("rerun_max", Pz["max"]),
                                                            ("target_q95", T["q95"]), ("rerun_q95", Pz["q95"]))}, flush=True)


def determinism(root):
    out = {}
    for d in sorted((root / "scenes_interrupted").glob("*")):
        rows = {}
        for f in sorted(d.glob("fixed_r*.npz")):
            g = root / "scenes" / d.name / f.name
            if not g.exists():
                continue
            with np.load(f) as a, np.load(g) as b:
                diff = max(float(np.abs(a[k] - b[k]).max()) for k in a.files)
                scale = max(float(np.abs(b[k]).max()) for k in b.files)
                rows[f.name] = {"max_abs_diff": diff, "max_abs_value": scale, "bitwise_equal": diff == 0.0, "same_keys": a.files == b.files}
        out[d.name] = rows
    KB.write(root / "amendment1/scene_determinism.json", out)
    print("determinism", {s: sum(r["bitwise_equal"] for r in v.values()) for s, v in out.items()}, flush=True)


if __name__ == "__main__":
    mode, root = sys.argv[1], Path(sys.argv[2]).resolve()
    {"evaluate": evaluate, "determinism": determinism}[mode](root)
