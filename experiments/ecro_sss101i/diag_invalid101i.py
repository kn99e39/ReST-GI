"""Read-only replay of membership proposals (geometry only) under the v2 metric oracle: locate invalid samples.

python diag_invalid101i.py WL101_ROOT OUT SEED
"""
from pathlib import Path
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_sss101g"))
sys.path.insert(0, str(HERE.parent / "ecro_sss101h"))
argv = sys.argv[:]
sys.argv = sys.argv[:1]
import run_generalization as RG  # noqa: E402
from oracle_field import ExposedOracle  # noqa: E402
import construct101h as CO_H  # noqa: E402

root, out, seed = Path(argv[1]).resolve(), Path(argv[2]).resolve(), int(argv[3])
reader, *_ = RG.F101.inputs(root)
R, rt, dr = reader.R, reader.rt, reader.dr
geom = rt.inc.G
b_m = json.loads((HERE / "protocol/contract_v2.json").read_text())["b_m_m"]
l1, l2 = np.linalg.norm(geom.PE1, axis=1), np.linalg.norm(geom.PE2, axis=1)
oracle = CO_H.metric_oracle(ExposedOracle, b_m, (R.F(l1.astype(np.float32)), R.F(l2.astype(np.float32))))(R, rt)
found = []
for launch in range(128):
    diag = []
    _, _, _, _, _, info = oracle.probes(32, seed, launch, diagnostic=diag)
    if int(dr.count(info["invalid"])[0]) == 0:
        continue
    for d in diag:
        for i in range(len(d["slot"])):
            pid = d["expected_pid"][i]
            dist = min(min(d["u0"][i], 1 - d["u0"][i]) * l1[pid], min(d["u1"][i], 1 - d["u1"][i]) * l2[pid]) if geom.PSID[pid] != 6 else None
            rec = {k: v[i] for k, v in d.items() if isinstance(v, list)}
            rec.update(launch=launch, metric_edge_distance_m=dist,
                       still_invalid=not ((not rec["native_valid"]) and geom.PSID[pid] != 6 and dist is not None and dist <= b_m))
            found.append(rec)
out.mkdir(parents=True, exist_ok=True)
(out / f"invalid_{seed}.json").write_text(json.dumps(found, indent=1))
print(json.dumps([f for f in found if f["still_invalid"]], indent=1))
