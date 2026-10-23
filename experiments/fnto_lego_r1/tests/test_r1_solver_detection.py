"""WL42: the production r1 solve() loop must classify a known linear iteration correctly (audit #12 / F6).

The real State.solve() is run on a stub whose step() is L <- rho L + e (per-element, positive e, L0 = 0), so the
expected classes are known: rho 0.8 converged, 0.999 unconverged at MAX_IT (slow), 1.05 and 1.5 diverged.
Batch 6's branch `it > 5 and d > 1` could not fire for a non-negative Jacobi started at zero; this test fails if the
r1 detection does not separate these cases.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import r1_common as C  # noqa: E402
import r1_solve as SV  # noqa: E402


class Stub(SV.State):
    def __init__(self, rho):
        self.rho = rho; self.L = torch.zeros(1000, dtype=torch.float64); self.e = torch.rand(1000, dtype=torch.float64, generator=torch.Generator().manual_seed(0))

    def step(self, sun_on=True):
        Ln = self.rho * self.L + self.e; dn = float((Ln - self.L).norm()); self.L = Ln; return dn, float(Ln.norm())

    def aggregates(self):
        return None, None

    def outgoing(self, *a, **k):
        return None


def main():
    torch.cuda.synchronize = lambda: None
    res = {}
    for rho, want in ((0.8, "converged"), (0.999, "unconverged"), (1.05, "diverged"), (1.5, "diverged")):
        s = Stub(rho).solve()
        res[str(rho)] = {"status": s.status, "iters": s.iters, "expected": want, "ok": s.status == want}
    ok = all(v["ok"] for v in res.values())
    out = C.OUT1 / "tests"; out.mkdir(parents=True, exist_ok=True); (out / "solver_detection.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)); print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
