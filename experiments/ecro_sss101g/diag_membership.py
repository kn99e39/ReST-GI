"""Read-only replay of construction proposals (geometry only) to locate invalid membership samples."""
from pathlib import Path
import json, sys
sys.argv = sys.argv[:3]
import run_generalization as RG
from oracle_field import ExposedOracle

root, seed = Path(sys.argv[1]).resolve(), int(sys.argv[2])
reader, *_ = RG.F101.inputs(root)
oracle = ExposedOracle(reader.R, reader.rt)
found = []
for launch in range(128):
    diag = []
    oracle.probes(32, seed, launch, diagnostic=diag)
    for d in diag:
        found.append(d)
        print(json.dumps(d), flush=True)
print("launches scanned 128, invalid records", sum(len(d["slot"]) for d in found))
