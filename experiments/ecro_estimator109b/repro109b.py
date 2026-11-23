"""WL109B Stage A: reproduce the WL109A calibration findings from authenticated artifacts (CPU only).

python repro109b.py WL109A_EVIDENCE_DIR EVAL109A_DIR WL109_EVIDENCE_DIR OUT

1. evaluation/109A files equal manifest109a.json, and the server-side WL109A calibration outputs equal them byte for byte.
2. The unchanged calib109a.pool reruns on the authenticated WL109A null pool; every numeric leaf of pool.json and the budget
   decision must equal the archived ones (the procedure is deterministic in its seeds).
3. The unchanged calib109a.hist reruns; every numeric leaf of hist.json must equal the archived one. While it runs, the
   per-draw upper bounds of each setting are recorded (wrapping calib109a.summaries, which hist calls once per setting over
   all draws) to compute the true median / 99th percentile that the archived `upper_median` / `upper_q99` fields mislabel.
4. Reproduces the WL109A observation that H3 / H4 material boundary fail at the WL109 design under the exact null.
STOP (exit 3) if any check fails.
"""
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_estimator109a"))
assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""
import common109a as CA  # noqa: E402
import calib109a as CB  # noqa: E402


def leaves_equal(a, b, path="", out=None, rel=1e-12, ab=1e-15):
    out = [] if out is None else out
    if isinstance(b, dict):
        if set(a) != set(b):
            out.append((path, "keys"))
        for k in b:
            if k in a:
                leaves_equal(a[k], b[k], f"{path}/{k}", out, rel, ab)
    elif isinstance(b, list):
        if len(a) != len(b):
            out.append((path, "len"))
        for i, (x, y) in enumerate(zip(a, b)):
            leaves_equal(x, y, f"{path}[{i}]", out, rel, ab)
    elif isinstance(b, bool) or isinstance(b, str) or b is None:
        if a != b:
            out.append((path, a, b))
    elif isinstance(b, (int, float)):
        if not (a == b or (isinstance(a, (int, float)) and (abs(a - b) <= ab or abs(a - b) <= rel * abs(b)))):
            if not (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
                out.append((path, a, b))
    return out


def main(ev_a, eval_a, ev9, out):
    out.mkdir(parents=True, exist_ok=True)
    rec = {"checks": {}}
    man = json.loads((eval_a / "manifest109a.json").read_text())
    bad = [f for f, v in man["files"].items() if CA.sha(eval_a / f) != v["sha256"]]
    bad += [f for f, v in man.get("qualitative", {}).get("files", {}).items() if CA.sha(eval_a / f) != v["sha256"]]
    srv = {"budget109a.json": "calib/budget109a.json", "pool.json": "calib/pool.json", "hist.json": "calib/hist.json",
           "synth.json": "calib/synth.json", "stageA.json": "stageA/stageA.json"}
    bad += [f"server:{s}" for f, s in srv.items() if CA.sha(ev_a / s) != man["files"][f]["sha256"]]
    rec["checks"]["artifact_hashes"] = {"bad": bad, "pass": not bad}
    print("hashes", rec["checks"]["artifact_hashes"], flush=True)

    # 2. pool rerun
    t0 = time.time()
    CB.pool(ev_a / "cap", out / "pool_rerun", ev_a / "calib/synth.json")
    a = json.loads((out / "pool_rerun/pool.json").read_text()); b = json.loads((ev_a / "calib/pool.json").read_text())
    a.pop("code_sha256", None); b.pop("code_sha256", None)
    d = leaves_equal(a, b)
    ba = json.loads((out / "pool_rerun/budget109a.json").read_text())["decision"]
    bb = json.loads((ev_a / "calib/budget109a.json").read_text())["decision"]
    rec["checks"]["pool_rerun"] = {"mismatches": len(d), "head": [str(x) for x in d[:10]], "decision_equal": ba == bb,
                                   "decision": bb, "elapsed_s": time.time() - t0, "pass": not d and ba == bb}
    print("pool", rec["checks"]["pool_rerun"], flush=True)

    # 3. hist rerun with recorded per-draw upper bounds
    recorded = []
    orig = CB.summaries

    def wrapped(Qsub):
        r = orig(Qsub)
        if np.ndim(Qsub) == 2 and Qsub.shape[0] > 1:
            recorded.append(np.asarray(r[2]))
        return r
    CB.summaries = wrapped
    t0 = time.time()
    CB.hist(ev9, out / "hist_rerun")
    CB.summaries = orig
    a = json.loads((out / "hist_rerun/hist.json").read_text()); b = json.loads((ev_a / "calib/hist.json").read_text())
    a.pop("code_sha256", None); b.pop("code_sha256", None)
    d = leaves_equal(a, b)
    keys = list(b["settings"])
    assert len(recorded) == len(keys), (len(recorded), len(keys))
    quant = {}
    for k, up in zip(keys, recorded):
        s = b["settings"][k]
        assert len(up) == s["draws"], (k, len(up), s["draws"])
        quant[k] = {"upper95_median": float(np.median(up)), "upper95_q99": float(np.quantile(up, 0.99)),
                    "upper95_mean": float(np.mean(up)), "archived_upper_median_field": s["q0"]["upper_median"],
                    "archived_field_equals_mean_of_per_draw_upper": bool(abs(s["q0"]["upper_median"] - np.mean(up)) <= 1e-12 * max(1, abs(np.mean(up)))),
                    "null_fail": s["q0"]["fail"], "limit": CA.QLIM[k.split("/")[1]], "draws": s["draws"]}
    rec["checks"]["hist_rerun"] = {"mismatches": len(d), "head": [str(x) for x in d[:10]], "elapsed_s": time.time() - t0,
                                   "label_defect_confirmed": all(v["archived_field_equals_mean_of_per_draw_upper"] for v in quant.values()),
                                   "pass": not d}
    (out / "hist_upper_quantiles.json").write_text(json.dumps(quant, indent=1))
    # 4. WL109A observation
    obs = {k: b["settings"][k]["q0"]["fail"] for k in keys if k.split("/")[1] == "material_boundary" and k.split("/")[0] in ("H3", "H4")}
    rec["checks"]["wl109_design_null_fail"] = {"H3_H4_material_boundary": obs,
                                                "others_max": max(b["settings"][k]["q0"]["fail"] for k in keys if k not in obs),
                                                "pass": min(obs.values()) > 0.3}
    rec["pass"] = all(c["pass"] for c in rec["checks"].values())
    rec["code_sha256"] = CA.sha(Path(__file__))
    (out / "stageA109b.json").write_text(json.dumps(rec, indent=1))
    print("STAGE A", "PASS" if rec["pass"] else "FAIL", flush=True)
    if not rec["pass"]:
        sys.exit(3)


if __name__ == "__main__":
    main(*[Path(x).resolve() for x in sys.argv[1:5]])
