"""WL104I native-proposal qualification: WL104F operator audit with proposals {A: A1 seed k, C: B1 seed k}; failing probes are
requalified once with the WL104G rule (fresh samples); the original failure is retained.

Usage: python native104i.py <angular_dir> <out_dir>
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

import common104i as CM

IO = CM.IO
sys.path.insert(0, str(CM.HERE.parent / "ecro_wl84_104f"))
sys.path.insert(0, str(CM.HERE.parent / "ecro_wl84_104g"))
import cross104f as XF  # noqa: E402
import qualify104g as QG  # noqa: E402

MATERIALS = ("test83_iid_1", "test83_hard_1", "test83_iid_159")


def requalify(state, theta, phi, seed):
    t, p = np.radians([theta, phi])
    wi = XF.mi.Vector3f(float(np.sin(t) * np.cos(p)), float(np.sin(t) * np.sin(p)), float(np.cos(t)))
    cl = XF.H.CD.DrClosure83("ER", state)
    known, empirical, parity = QG.native_blocks(cl, wi, np.random.default_rng(seed))
    numerical = QG.integrate(cl, wi)
    kr = QG.statistics(known, QG.OP["known_margin"], QG.EXACT)
    dr_ = QG.statistics(empirical, QG.OP["distribution_margin"], numerical.mean(0), numerical.std(0, ddof=1) / np.sqrt(len(numerical)))
    return {"known": kr, "distribution": dr_, "parity": parity, "passed": bool(kr["passed"] and dr_["passed"] and parity["passed"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("angular", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    rows = IO.read(IO.MATERIALS)
    with np.load(args.angular / "compiled_states.npz") as z:
        st = {k: z[k] for k in z.files}
    opt = XF.P["part_a"]["operator_audit"]
    res, req, n_fail = {}, {}, 0
    for k, seed in enumerate(CM.SEEDS):
        for j, tag in enumerate(MATERIALS):
            i = next(q for q, r in enumerate(rows) if r["tag"] == tag)
            pair = {"A": st[f"A1_{seed}"][i], "C": st[f"B1_{seed}"][i]}
            for name, v in pair.items():
                if not (np.isfinite(v).all() and v[45] > 0 and np.all(v[45:53] >= 0) and abs(v[45:53].sum() - 1) < 1e-6):
                    raise IO.Stop(f"INVALID_PROPOSAL {seed} {tag} {name}")
            recs = []
            for t, theta in enumerate(opt["theta_deg"]):
                for p, phi in enumerate(opt["phi_deg"]):
                    r = XF.operator_case(pair, theta, phi, 1049001 + 100 * k + 10 * j + 2 * t + p, opt)
                    recs.append(r)
                    if not r["passed"]:
                        n_fail += 1
                        idx = len(req)
                        # requalify only the failing proposal(s); a cross-proposal albedo failure involves both
                        bad = {a for g in ("pdf_mass", "known_cosine_moments") for a, x in r[g].items() if not x["passed"]}
                        if any(not x["passed"] for x in r["fixed_f_albedo"].values()):
                            bad |= {"A", "C"}
                        rq = {arm: requalify(pair[arm], theta, phi, 1049501 + 2 * idx + q) for q, arm in enumerate("AC") if arm in bad}
                        req[f"{seed}/{tag}/{theta}/{phi}"] = {"original": r, "requalification": rq,
                                                              "passed": all(x["passed"] for x in rq.values())}
            res[f"{seed}/{tag}"] = {"probes": recs, "lambert_probability": {a: float(v[45]) for a, v in pair.items()},
                                    "passed_direct": all(r["passed"] for r in recs)}
            print("audit", seed, tag, res[f"{seed}/{tag}"]["passed_direct"], round(time.monotonic() - t0, 1), flush=True)
    passed = all(v["passed"] for v in req.values())          # every direct failure was requalified once
    IO.write(args.out / "native_audit.json", {"label": "PASS" if passed else "FAIL", "direct_failures": n_fail,
                                              "requalifications": req, "audits": res, "settings": opt,
                                              "historical_exception": "WL104F: one of 288 known-integral checks failed (C, iid_1, 5 deg); WL104G requalified it as a fluctuation",
                                              "commit": IO.commit(), "seconds": time.monotonic() - t0})
    print("native audit", "PASS" if passed else "FAIL", n_fail, flush=True)


if __name__ == "__main__":
    main()
