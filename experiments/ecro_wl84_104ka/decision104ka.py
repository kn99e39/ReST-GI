"""WL104K-A frozen rules (protocol.json) -> decision.json.

Usage: python decision104ka.py <run_root>
"""
import sys
from pathlib import Path

import numpy as np

import common104ka as KA

sys.path.insert(0, str(KA.HERE.parent / "ecro_wl84_104b"))
from stats104b import jk_ci  # noqa: E402

CM = KA.CM


def per_start_scene(an, R, event="total"):
    fx = an["fixed"]
    out = []
    for r in range(R):
        a, p = fx[f"A_{r}"]["objects"]["rms"][event], fx[f"P_{r}"]["objects"]["rms"][event]
        A, Pp = a["point"], p["point"]
        Al, Pl = np.asarray(a["loo"]), np.asarray(p["loo"])
        out.append(jk_ci((Pp - A) / A, (Pl - Al) / Al))
    return out


def label(ci):
    if all(c["hi"] < 0 for c in ci):
        return "STABLE_IMPROVEMENT"
    if all(c["lo"] > 0 for c in ci):
        return "STABLE_REGRESSION"
    return "MIXED"


def main():
    root = Path(sys.argv[1])
    xf = KA.read(root / "ext/fit.json")
    xe = KA.read(root / "ext_angular.json")
    tf = KA.read(root / "tests_fitted_ext.json")
    R = len(xf["start_keys"])
    scene = {}
    for tag, base in (("12k", KA.K_ROOT / "posthoc_scenes"), ("extended", root / "scenes_ext")):
        for s in KA.SCENES:
            an = KA.read(base / s / "analysis.json")
            row = {"identity_passed": an["identity_passed"]}
            for ev in ("e1", "e2", "total"):
                ci = per_start_scene(an, R, ev)
                row[ev] = {"per_start": [{k: float(c[k]) for k in ("point", "lo", "hi")} for c in ci], "label": label(ci)}
            scene.setdefault(tag, {})[s] = row
    attr = {s: KA.read(root / "lanes" / s / "attribution.json") for s in KA.SCENES}
    opt = xf["optimization_qualification"]
    ang = xe["angular_stability"]
    ext = scene["extended"]
    imp = sum(v["total"]["label"] == "STABLE_IMPROVEMENT" for v in ext.values())
    reg_total = any(v["total"]["label"] == "STABLE_REGRESSION" for v in ext.values())
    reg_e1 = any(v["e1"]["label"] == "STABLE_REGRESSION" for v in ext.values())
    ident_ok = all(v["identity_passed"] for t in scene.values() for v in t.values())
    if not ident_ok:
        integ = "UNRESOLVED"
    elif reg_total:
        integ = "NOT_JUSTIFIED"
    elif opt == "PASS" and ang == "STABLE" and imp >= 2 and not reg_e1:
        integ = "JUSTIFIED"
    else:
        integ = "UNRESOLVED"
    out = {"optimization_qualification": opt, "rule_a_fraction": {a: xf[a]["rule_a_fraction"] for a in CM.ARMS},
           "angular_stability": ang, "wl104k_gain_criteria_on_extended_states_descriptive": xe["wl104k_gain_criteria_on_extended_states_descriptive"],
           "extended_state_contracts_passed": tf["passed"], "scene_utility": scene,
           "lane_correspondence": {s: a["correspondence_passed"] for s, a in attr.items()},
           "attribution_labels": {s: {r: (v["label"] if v else None) for r, v in a["labels"].items()} for s, a in attr.items()},
           "integration_work_justified": integ,
           "adoption_readiness": "NOT_READY (gated-response sampler / pdf not designed; experiment-local evaluator only; owner decision)",
           "commit": CM.commit()}
    KA.write(root / "decision.json", out)
    print("decision", opt, ang, integ, {s: (v["total"]["label"], v["e1"]["label"]) for s, v in ext.items()}, flush=True)


if __name__ == "__main__":
    main()
