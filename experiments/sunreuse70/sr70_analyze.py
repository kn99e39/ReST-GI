"""WL70 quality and verdict analysis (protocol sunreuse70_v1). Scores arms A (retraced sun) and B (reused sun) with the unchanged WL69
machinery (xbench69/x_analyze: refs, score with the K-seed SE, change), quality-matches them against the unchanged WL69 Cycles ladder
(analysis.json quality / CY render manifests) and applies the predeclared verdict map.

    cd <repo>/experiments/fnto_k55 && python ../sunreuse70/sr70_analyze.py <root> <out_json>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "xbench69"))
import x_analyze as XA  # noqa: E402

XB = Path.home() / "rest_xbench69"
MATCH = 0.10


def rg_score(R, files):
    """files: {state: [npz path seed1, seed2]} -> WL69 RG form (mean of K seeds, SE_oracle = half difference) + M4."""
    import k55_common as KCm
    P, A = {}, {}
    for st, fs in files.items():
        z = [np.load(f) for f in fs]
        seeds = [{cam: zz[f"{k}|nee"] + zz[f"{k}|ball"] for cam in XA.ALLC for k in [KCm.dirkey(R[st]['sd']['cams'][cam]['dir'])]} for zz in z]
        P[st] = {cam: 0.5 * (seeds[0][cam] + seeds[1][cam]) for cam in XA.ALLC}
        sc = [XA.score(R, st, s_) for s_ in seeds]
        so = {cam: {gn: [abs(a - b) / 2 for a, b in zip(sc[0]["cams"][cam][gn]["bias"], sc[1]["cams"][cam][gn]["bias"])] for gn in sc[0]["cams"][cam]} for cam in XA.ALLC}
        A[st] = XA.score(R, st, P[st], so)
    A["M4"] = XA.change(R, P["T0"], P["T1"]); return A


def q(a):
    return dict(M1=min(a["T0"]["M1"], a["T1"]["M1"]), M2=max(a["T0"]["M2"], a["T1"]["M2"]), M3=float(np.max([a["T0"]["M3"], a["T1"]["M3"]])), M4=float(np.max(a["M4"]["err"])))


def main():
    root = Path(sys.argv[1]); e2e = json.load(open(root / "e2e" / "e2e.json")); W = json.load(open(XB / "analysis.json"))
    R = XA.refs(); res = dict(protocol="sunreuse70_v1", arms={}, quality={})
    for arm in ("A", "B"):
        res["arms"][arm] = rg_score(R, {"T0": [root / "e2e" / f"floor__CAK_s{i}.npz" for i in (1, 2)], "T1": [root / "e2e" / f"floor_c4__CAK{arm}_s{i}.npz" for i in (1, 2)]})
        res["quality"][arm] = q(res["arms"][arm])
    res["quality"]["WL69_RG_CAK"] = W["quality"]["RG_CAK"]; nr = res["arms"]["A"]["T1"]["n_ruled"]
    L = W["quality"]["CY"]; man = [json.loads(x) for d in sorted((XB / "cy").glob("spp*")) for x in open(d / "render_manifest.jsonl")]

    def cy_time(spp):
        seeds = sorted({m["seed"] for m in man if m["scene"] == "floor_c4" and m["spp"] == spp})
        return [sum(m["seconds"] for m in man if m["scene"] == "floor_c4" and m["spp"] == spp and m["seed"] == sd and m["cam"] in XA.RULED) for sd in seeds]
    res["cy_ruled_T1_seconds"] = {s: cy_time(int(s)) for s in L}
    res["QM"] = dict(contract=next((int(s) for s in sorted(L, key=int) if len(L[s]) == 2 and all(x["M1"] == nr for x in L[s])), None))
    for arm in ("A", "B"):
        rq = res["quality"][arm]
        res["QM"][arm] = next((int(s) for s in sorted(L, key=int) if len(L[s]) == 2 and all(x["M1"] >= rq["M1"] and x["M2"] <= rq["M2"] and x["M3"] <= rq["M3"]
                                                                                             and x["M4"] <= rq["M4"] for x in L[s])), None)
    lat = {}
    for arm in ("A", "B"):
        per = [float(np.median([x["timing"]["total_s"] for x in e2e["runs"][t]["reps"] if x["arm"] == arm])) for t in ("s1", "s2")]
        lat[arm] = dict(median_per_K_seed=per, latency=max(per))
    res["latency"] = lat
    vb = res["arms"]["B"]; contract = vb["T0"]["M1"] == vb["T0"]["n_ruled"] and vb["T1"]["M1"] == vb["T1"]["n_ruled"]
    qm = res["QM"]["B"]; cy = max(res["cy_ruled_T1_seconds"][str(qm)]) if qm is not None else None
    b = lat["B"]["latency"]
    rel = None if cy is None else ("beaten" if b < (1 - MATCH) * cy else "matched" if b <= (1 + MATCH) * cy else "faster_full_recompute")
    res["comparison"] = dict(B_contract_met=bool(contract), QM_B_spp=qm, CY_at_QM_B_s=cy, B_latency_s=b, relation=rel,
                             CY_at_QM_contract_s=max(res["cy_ruled_T1_seconds"][str(res["QM"]["contract"])]) if res["QM"]["contract"] else None)
    XA.KCm.dump(res, sys.argv[2]) if hasattr(XA, "KCm") else json.dump(res, open(sys.argv[2], "w"), indent=1)
    print(json.dumps(dict(quality=res["quality"], QM=res["QM"], latency=res["latency"], comparison=res["comparison"]), indent=1)); sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
