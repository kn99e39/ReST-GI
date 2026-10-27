"""WL49 focused tests (GPU). python tests/test_s49.py <out json>"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import s49_data as S  # noqa: E402
import r1_common as C  # noqa: E402

R = {}


def check(n, ok, **i):
    R[n] = {"pass": bool(ok), **i}; print(n, R[n], flush=True)


@torch.no_grad()
def main(outp):
    dev = torch.device("cuda"); L = S.Lego(dev); rf, rh = S.receivers(L); sp = S.splits(); d = S.dch()
    sha = hashlib.sha256((C.OUT1 / "K_k44P/kernels.pt").read_bytes()).hexdigest()
    check("frozen_k44_identity", sha.startswith("30d55da373a01ab8"), sha=sha[:16])
    check("train_test_separation", len(np.intersect1d(sp["fit"], sp["val"])) == 0 and len(np.intersect1d(rf, rh)) == 0 and len(sp["fit"]) == 480)
    check("object_scope_membership", L.nA == 852358 and len(np.unique(L.mat)) == 8 and set(np.unique(L.mat[rf]).tolist()) == {0}, materials=np.unique(L.mat).tolist())
    xi = torch.as_tensor(rf[:6], device=dev); rng = torch.Generator(device=dev).manual_seed(0)
    E = torch.rand(L.nA, 2, generator=rng, device=dev); Z = torch.zeros(L.nA, 1, device=dev)
    yS, yT = S.oracle(L, xi, E, d[1]); yz, _ = S.oracle(L, xi, Z, d[1]); y12, _ = S.oracle(L, xi, E.sum(1, keepdim=True), d[1])
    check("zero_field", float(yz.abs().max()) == 0.0)
    check("linear_superposition", float((y12[:, 0] - yS.sum(1)).abs().max() / y12.abs().max()) < 1e-5)
    check("finite_nonnegative", bool(torch.isfinite(yS).all() and (yS >= 0).all() and (yT >= 0).all()))
    # source-receiver index correctness + support boundary + self exclusion vs brute force
    r = torch.cdist(L.pos[xi], L.pos).clamp(min=2e-3); k = S.burley(r, d[1]) * (r < 16 * d[1]); k[torch.arange(6), xi] = 0
    bS = (k * (r < 0.30)) @ (E * L.area[:, None]); bT = (k * (r >= 0.30)) @ (E * L.area[:, None])
    check("index_support_self", float((bS.double() - yS).abs().max() / bS.abs().max()) < 1e-4 and float((bT.double() - yT).abs().max() / max(float(bT.abs().max()), 1e-30)) < 1e-4)
    # aggregation normalisation: constant field -> constant means; atom projection keeps aggregate means
    Bm, Cm = L.agg_means(torch.full((L.nA, 1), 3.0, device=dev)); P_ = L.atom_project(E); B1, C1 = L.agg_means(E); B2, C2 = L.agg_means(P_)
    bm, cm = L.barea_m > 0, L.carea_m > 0                                    # aggregates with Lego members (box aggregates excluded)
    vals = [float((Bm[bm] - 3).abs().max()), float((Cm[cm] - 3).abs().max()), float((B1 - B2)[bm].abs().max()), float((C1 - C2)[cm].abs().max())]
    check("aggregation_normalisation", max(vals) < 1e-4, max_abs=vals, lego_band_aggs=int(bm.sum()), lego_cells=int(cm.sum()), NB=L.NB, NC=L.NC)
    # probe field reproduces the training-time incident field semantics (sum of light contributions, nonnegative)
    Ef = S.probe_fields(L, "train", sp["fit"][:2]); check("probe_fields_finite", bool(torch.isfinite(Ef).all() and (Ef >= 0).all() and (Ef.sum(0) > 0).all()))
    R["ALL_PASS"] = all(v["pass"] for v in R.values() if isinstance(v, dict))
    Path(outp).parent.mkdir(parents=True, exist_ok=True); Path(outp).write_text(json.dumps(R, indent=1)); print("ALL_PASS", R["ALL_PASS"])


if __name__ == "__main__":
    main(sys.argv[1])
