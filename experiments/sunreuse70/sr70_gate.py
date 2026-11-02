"""WL70 Stage-4 gate (protocol sunreuse70_v1, stage4.gate): reuse is implemented and Stage 5 runs only if the predeclared validity rule
holds on identity.json. Exit 0 = proceed, 1 = stop (the chain records the reason; the identity evidence is reported either way).
    python sr70_gate.py <identity.json>
"""
import json
import sys


def gate(res):
    v, d, t = res["verdicts"], res["decisions"], res["tests"]
    reasons = []
    if not res["null_calibrated"]:
        reasons.append("noise model not calibrated (T0 null)")
    if not all(x["precise"] for x in v.values()):
        reasons.append("precision insufficient: " + ",".join(k for k, x in v.items() if not x["precise"]))
    if not (d["T1"]["accept"] and v["T1"]["identity"]):
        reasons.append(f"T1 not valid: accept={d['T1']['accept']} identity={v['T1']['identity']} checks={d['T1']['checks']}")
    for k, x in d.items():
        if x["expected"] == "reject" and x["accept"]:
            reasons.append(f"invalidation control {k} accepted (stale light)")
    rep = t.get("replica_equals_historical_T1_5591", {})
    if not rep or max(rep.values()) > 1e-5:
        reasons.append(f"replica differs from the historical run_sun: {rep}")
    return reasons


if __name__ == "__main__":
    r = gate(json.load(open(sys.argv[1])))
    print(json.dumps(dict(proceed=not r, reasons=r), indent=1)); sys.exit(1 if r else 0)
