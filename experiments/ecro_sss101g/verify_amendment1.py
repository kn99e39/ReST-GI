"""Amendment 1 check on the failing launch: original vs amended membership counts (geometry only)."""
from pathlib import Path
import json, sys
sys.argv = sys.argv[:2]
import run_generalization as RG
import gpu101g as GG
from oracle_field import ExposedOracle

reader, *_ = RG.F101.inputs(Path(sys.argv[1]).resolve())
dr = reader.dr
out = {}
for label, cls in (("original", ExposedOracle), ("amended", GG.amended_oracle(ExposedOracle, 8e-6))):
    oracle = cls(reader.R, reader.rt)
    *_, info = oracle.probes(32, 10203101, 57)
    out[label] = {k: int(dr.count(v)[0]) for k, v in info.items() if k in ("valid", "buried", "back", "tie", "invalid", "amended_tie")}
print(json.dumps(out))
assert out["amended"]["invalid"] == 0 and out["amended"]["amended_tie"] == 1
assert out["amended"]["tie"] == out["original"]["tie"] + 1
assert all(out["amended"][k] == out["original"][k] for k in ("valid", "buried", "back"))
