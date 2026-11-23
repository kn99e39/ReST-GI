"""WL109A Stage E: final static decision record (CPU).

python decide109a.py METRICS_JSON OUT_JSON [STAGE_D_JSON]

Without Stage D the final static verdict is the analysis verdict. STAGE_D_REQUIRED is resolved only by a Stage D record
(protocol109a stage_D / decision.mapping): flagged pattern not replicated -> STATIC_PASS; replicated with a resolved
conditional-distribution mismatch -> NOT_SUPPORTED; replicated without a resolved mismatch -> INCONCLUSIVE.
"""
import sys
from pathlib import Path

from common109a import read, sha, write

met = Path(sys.argv[1]).resolve()
m = read(met)
rec = {"metrics_sha256": sha(met), "analysis_static_verdict": m["static_verdict"]}
v = m["static_verdict"]
if len(sys.argv) > 3:
    d = read(sys.argv[3])
    rec["stage_D_sha256"] = sha(sys.argv[3])
    if v == "STAGE_D_REQUIRED":
        v = {"not_replicated": "STATIC_PASS", "replicated_mismatch": "NOT_SUPPORTED",
             "replicated_no_mismatch": "INCONCLUSIVE"}[d["outcome"]]
rec["final_static_verdict"] = v
write(Path(sys.argv[2]), rec)
print("FINAL STATIC", v)
