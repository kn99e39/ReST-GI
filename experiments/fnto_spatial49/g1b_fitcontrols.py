"""WL49 Gate 1 amendment 2 - valid fitting controls on the SAME existing aggregation features (the predeclared A1 ridge
1e-6 fit over-fits: fit-field error 0.009-0.04 vs validation 1.2-15).

A1v   ridge LS (float64), lambda (relative to trace / D) chosen per channel on the VALIDATION fields from the fixed grid
      1e-6 ... 1e0 (7 values); never on test.
A1nn  nonnegative LS (physical kernel weights >= 0), projected FISTA, 1500 iterations, init = A0 ideal weights, float64.
A1*   per channel, the control with the lower validation shadow error; the Gate 1 decision uses A1* (rule unchanged).

    python g1b_fitcontrols.py <out root>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s49_data as S  # noqa: E402
from g1_capacity import feats, DB  # noqa: E402
import r1_common as C  # noqa: E402

GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def ridge(G, b, lam):
    D = G.shape[-1]; tr = torch.diagonal(G, dim1=1, dim2=2).sum(-1) / D
    return torch.linalg.solve(G + lam * tr[:, None, None] * torch.eye(D, device=G.device, dtype=G.dtype)[None], b[..., None])[..., 0]


def nnls(G, b, w0, iters=1500):
    Lc = torch.linalg.eigvalsh(G)[:, -1].clamp(min=1e-30)[:, None]
    w = w0.clone().clamp(min=0); z = w.clone(); t = 1.0
    for _ in range(iters):
        g = (G @ z[..., None])[..., 0] - b; wn = (z - g / Lc).clamp(min=0)
        tn = (1 + (1 + 4 * t * t) ** 0.5) / 2; z = wn + ((t - 1) / tn) * (wn - w); w, t = wn, tn
    rel_change = float(((G @ w[..., None])[..., 0] - b).norm() / b.norm())
    return w, rel_change


@torch.no_grad()
def main(root):
    root = Path(root); out = root / "gate1_existing_capacity"; dev = torch.device("cuda"); t00 = time.time()
    L = S.Lego(dev); sp = S.splits(); rf, rh = S.receivers(L); xf = torch.as_tensor(rf, device=dev); nf = len(rf)
    Z_ = np.load(root / "cache/oracle.npz")
    E = {"fit": S.probe_fields(L, "train", sp["fit"]), "val": S.probe_fields(L, "train", sp["val"]), "test_probe": S.probe_fields(L, "holdout", sp["test_probe"]),
         "view_T0": S.view_fields(L, "T0"), "view_T1": S.view_fields(L, "T1")}
    Zs = {}
    for k in E:
        Bm, Cm = L.agg_means(E[k]); Zs[k] = feats(L, xf, Bm, Cm).double()           # [R, 224, F]
    dist = {k: torch.as_tensor(Z_[f"dist_{k}"], device=dev) for k in ("test_probe", "view_T0")}
    rec = {"grid": GRID, "channels": {}}
    for ci, c in enumerate("RGB"):
        Y = {k: torch.as_tensor(Z_[f"{c}_{k}_S30"][:nf], device=dev) for k in E}
        A0 = torch.as_tensor(np.load(root / f"cache/gate1_coef_{c}.npz")["A0"], device=dev).double()
        Zf = Zs["fit"]; G = Zf @ Zf.transpose(1, 2); b = (Zf @ Y["fit"][..., None])[..., 0]
        shv = (E["val"][xf] == 0)
        val_err = {}; W = {}
        for lam in GRID:
            w = ridge(G, b, lam); p = torch.einsum("rd,rdf->rf", w, Zs["val"]); val_err[lam] = S.rel(p, Y["val"], shv); W[lam] = w
        lam_best = min(val_err, key=val_err.get)
        t = time.time(); wnn, rc = nnls(G, b, A0); t_nn = time.time() - t
        cand = {"A1v": W[lam_best], "A1nn": wnn}
        r = {"A1v_lambda": lam_best, "A1v_val_shadow_by_lambda": {str(k): v for k, v in val_err.items()}, "A1nn_fit_rel_residual": rc, "A1nn_s": t_nn, "sets": {}}
        for k in E:
            y = Y[k]; sh = E[k][xf] == 0; lit = ~sh; r["sets"][k] = {}
            for a, w in cand.items():
                p = torch.einsum("rd,rdf->rf", w, Zs[k])
                e = {"all": S.rel(p, y), "lit": S.rel(p, y, lit), "shadow": S.rel(p, y, sh), "shadow_ratio": float((p * sh).sum() / (y * sh).sum()),
                     "shadow_corr": float(np.corrcoef(p[sh].cpu().numpy(), y[sh].cpu().numpy())[0, 1])}
                if k in dist:
                    for lo, hi in zip(DB[:-2], DB[1:-1]):
                        m = sh & (dist[k] >= lo) & (dist[k] < hi)
                        if m.sum() > 100 and float((y * m).abs().sum()) > 0:
                            e[f"shadow_d{lo:.2f}-{hi:.2f}"] = S.rel(p, y, m)
                    e["per_field_shadow"] = [S.rel(p[:, f], y[:, f], sh[:, f]) if sh[:, f].any() else None for f in range(y.shape[1])]
                r["sets"][k][a] = e
        best = min(("A1v", "A1nn"), key=lambda a: r["sets"]["val"][a]["shadow"]); r["A1_star"] = best
        prim = {}
        for a in ("A1v", "A1nn"):
            p = torch.cat([torch.einsum("rd,rdf->rf", cand[a], Zs[k]) for k in ("test_probe", "view_T0")], 1)
            y = torch.cat([Y[k] for k in ("test_probe", "view_T0")], 1); sh = torch.cat([E[k][xf] == 0 for k in ("test_probe", "view_T0")], 1)
            prim[a] = {"shadow": S.rel(p, y, sh), "all": S.rel(p, y)}
        r["primary_test"] = prim; rec["channels"][c] = r
        np.savez(root / f"cache/gate1b_coef_{c}.npz", A1v=cand["A1v"].cpu().numpy(), A1nn=wnn.cpu().numpy())
        print(c, "lam", lam_best, "best", best, json.dumps(prim), flush=True)
    g1 = json.loads((out / "gate1.json").read_text())
    pr = {c: {"A1star_shadow": rec["channels"][c]["primary_test"][rec["channels"][c]["A1_star"]]["shadow"], "collision_shadow": g1["channels"][c]["primary_test"]["collision"]["shadow"]} for c in "RGB"}
    suff = all(pr[c]["A1star_shadow"] <= 0.05 for c in "RGB"); proceed = any(pr[c]["A1star_shadow"] >= 0.10 or pr[c]["collision_shadow"] >= 0.05 for c in "RGB")
    rec["decision_with_valid_controls"] = {"values": pr, "sufficient": suff, "proceed_gate2": (not suff) and proceed, "outcome4_stop": (not suff) and (not proceed)}
    rec["cost"] = {"wall_s": time.time() - t00, "peak_GB": torch.cuda.max_memory_allocated() / 1e9}
    (out / "gate1b.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec["decision_with_valid_controls"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
