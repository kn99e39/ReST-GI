"""WL109B static decision record (CPU).

python decide109b.py METRICS_JSON OUT_JSON [REPLICATION_METRICS_JSON]

Maps the analysis verdict to the final static verdict (protocol109b decision):
  BIASED_STAGE_E + replication: every flagged condition again non-EQUIVALENT -> BIAS_SUPPORTED; any flagged condition
                                EQUIVALENT in the replication -> INCONCLUSIVE (non-persistent)
  BIASED_STAGE_E without a replication record -> STAGE_E_PENDING
  INCONCLUSIVE / NOT_SUPPORTED / STATIC_PASS -> unchanged
"""
from pathlib import Path
import sys

from common109b import read, sha, write

met = Path(sys.argv[1]).resolve()
m = read(met)
rec = {"metrics_sha256": sha(met), "analysis_static_verdict": m["static_verdict"], "counts": m.get("counts"),
       "flagged": m.get("flagged")}
v = m["static_verdict"]
if v == "BIASED_STAGE_E":
    if len(sys.argv) > 3:
        rp = Path(sys.argv[3]).resolve()
        r = read(rp)
        rec["replication_sha256"] = sha(rp)
        persist = {}
        for key in m["flagged"]:
            b = r["bias"].get(key)
            persist[key] = None if b is None else (b["Q_state"] != "EQUIVALENT" or b["B1"]["state"] != "EQUIVALENT")
        rec["replication_persistent"] = persist
        v = "BIAS_SUPPORTED" if all(x for x in persist.values()) else "INCONCLUSIVE"
    else:
        v = "STAGE_E_PENDING"
rec["final_static_verdict"] = v
write(Path(sys.argv[2]), rec)
print("FINAL STATIC", v)
