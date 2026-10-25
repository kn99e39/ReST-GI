"""WL44 (K-side fidelity attribution) shared helpers. Reads r1 artefacts read-only; writes results/fnto_lego_k44/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_lego_r1"))
sys.path.insert(0, str(HERE.parent / "fnto_lego"))
import r1_common as C  # noqa: E402

OUT = C.ROOT / "results/fnto_lego_k44"


def save(name, rec):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec, indent=1, default=float)[:6000])
