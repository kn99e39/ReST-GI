"""WL49 Gate 2 - DASQ candidate vs the existing aggregation on the same oracle / fields / receivers (amendment 3).

alpha from counts only (B1: q95 elements <= 224, B2: <= 448); fit types: ideal-weight quadrature (primary), A1v (validation
ridge), A1nn (nonnegative LS init = ideal weights); B0 numbers from gate1 / gate1b. Also the B0 coverage diagnostic.

    python g2_candidate.py <out root>
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
import dasq as DQ  # noqa: E402
from g1_capacity import DB  # noqa: E402
from g1b_fitcontrols import GRID, ridge, nnls  # noqa: E402
import r1_common as C  # noqa: E402

ALPHAS = [0.25, 0.35, 0.5, 0.7, 1.0]
SETS = ("fit", "val", "test_probe", "view_T0", "view_T1")


@torch.no_grad()
def main(root):
    root = Path(root); out = root / "gate2_candidate"; out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda"); t00 = time.time()
    L = S.Lego(dev); sp = S.splits(); rf, rh = S.receivers(L); xf = torch.as_tensor(rf, device=dev); nf = len(rf)
    t = time.time(); T = DQ.Octree(L.pos, L.area); torch.cuda.synchronize(); t_tree = time.time() - t
    rec = {"t_octree_s": t_tree, "alpha_counts": {}}
    for a in ALPHAS:
        t = time.time(); Ss, Es, cnt = T.elements(L.pos[xf], xf, a, S.R_IN); torch.cuda.synchronize()
        c = cnt.float(); rec["alpha_counts"][str(a)] = {"median": float(c.median()), "q95": float(torch.quantile(c, 0.95)), "max": int(c.max()), "t_s": time.time() - t}
    q95 = {a: rec["alpha_counts"][str(a)]["q95"] for a in ALPHAS}
    aB1 = min([a for a in ALPHAS if q95[a] <= 224], default=None); aB2 = min([a for a in ALPHAS if q95[a] <= 448], default=None)
    rec["alpha_B1"], rec["alpha_B2"] = aB1, aB2; print(json.dumps(rec), flush=True)
    E = {"fit": S.probe_fields(L, "train", sp["fit"]), "val": S.probe_fields(L, "train", sp["val"]), "test_probe": S.probe_fields(L, "holdout", sp["test_probe"]),
         "view_T0": S.view_fields(L, "T0"), "view_T1": S.view_fields(L, "T1")}
    Z_ = np.load(root / "cache/oracle.npz"); dist = {k: torch.as_tensor(Z_[f"dist_{k}"], device=dev) for k in ("test_probe", "view_T0")}
    g1 = json.loads((root / "gate1_existing_capacity/gate1.json").read_text()); g1b = json.loads((root / "gate1_existing_capacity/gate1b.json").read_text())
    dch = S.dch(); rec["coverage_B0"] = {}; rec["conditions"] = {}
    # B0 coverage: share of the in-support response from sources inside the represented aggregates / cells
    nbr, nbrA, mid, midA = L.rep_lists(xf)
    for ci, d in enumerate(dch):
        num = torch.zeros(nf, 17, device=dev, dtype=torch.float64); den = torch.zeros_like(num)
        Fz = torch.cat([torch.ones(L.nA, 1, device=dev), E["view_T0"]], 1)
        for i0 in range(0, nf, 64):
            i1 = min(i0 + 64, nf); Br = torch.zeros(i1 - i0, L.NB, dtype=torch.bool, device=dev); Cr = torch.zeros(i1 - i0, L.NC, dtype=torch.bool, device=dev)
            rr = torch.arange(i1 - i0, device=dev)[:, None]
            Br[rr.expand_as(nbr[i0:i1]), nbr[i0:i1].clamp(min=0)] = nbr[i0:i1] >= 0; Cr[rr.expand_as(mid[i0:i1]), mid[i0:i1].clamp(min=0)] = mid[i0:i1] >= 0
            for s0 in range(0, L.nA, 1 << 15):
                s1 = min(s0 + (1 << 15), L.nA); r = torch.cdist(L.pos[xf[i0:i1]], L.pos[s0:s1]).clamp(min=2e-3)
                k = S.burley(r, d) * (r < 16 * d) * (r < S.R_IN) * (xf[i0:i1, None] != torch.arange(s0, s1, device=dev)[None])
                fl = torch.where(r < C.R_LOCAL, Br[:, L.f2b[s0:s1]], Cr[:, L.f2c[s0:s1]])
                AE = Fz[s0:s1] * L.area[s0:s1, None]
                num[i0:i1] += ((k * fl) @ AE).double(); den[i0:i1] += (k @ AE).double()
        sh = torch.cat([torch.ones(nf, 1, dtype=torch.bool, device=dev), E["view_T0"][xf] == 0], 1)
        rec["coverage_B0"]["RGB"[ci]] = {"uniform_field_share_median": float((num[:, 0] / den[:, 0].clamp(min=1e-60)).median()),
                                         "view_T0_shadow_share_pooled": float((num[:, 1:] * sh[:, 1:]).sum() / (den[:, 1:] * sh[:, 1:]).sum())}
    print("coverage", json.dumps(rec["coverage_B0"]), flush=True)
    cs = {k: T.cs_field(L.area, E[k]) for k in SETS}
    for tag, a in (("B2p", 1.0), ("B2d", 0.7)):                                     # amendment 4: B1 / B2 infeasible
        if a is None:
            continue
        Ss, Es, cnt = T.elements(L.pos[xf], xf, a, S.R_IN); t = time.time()
        Fm = {k: T.means(cs[k], Ss, Es) for k in SETS}; torch.cuda.synchronize(); t_feat = time.time() - t
        cond = {"alpha": a, "elements_median": float(cnt.float().median()), "elements_q95": float(torch.quantile(cnt.float(), 0.95)), "t_features_all_sets_s": t_feat, "channels": {}}
        for ci, d in enumerate(dch):
            c = "RGB"[ci]; t = time.time()
            W = T.ideal_weights(L.pos, L.area, L.pos[xf], xf, Ss, Es, lambda r: S.burley(r, d) * (r < 16 * d), S.R_IN); t_w = time.time() - t
            Y = {k: torch.as_tensor(Z_[f"{c}_{k}_S30"][:nf], device=dev) for k in SETS}
            Zf = Fm["fit"]; G = Zf @ Zf.transpose(1, 2); b = (Zf @ Y["fit"][..., None])[..., 0]
            shv = E["val"][xf] == 0; vbest, wv = None, None; verr = {}
            for lam in GRID:
                w = ridge(G, b, lam); e_ = S.rel(torch.einsum("rk,rkf->rf", w, Fm["val"]), Y["val"], shv); verr[str(lam)] = e_
                if vbest is None or e_ < vbest[1]:
                    vbest, wv = (lam, e_), w
            wnn, rc = nnls(G, b, W)
            arms = {"quad": W, "A1v": wv, "A1nn": wnn}; r = {"t_ideal_weights_s": t_w, "A1v_lambda": vbest[0], "A1v_val_by_lambda": verr, "A1nn_fit_rel_residual": rc, "sets": {}}
            for k in SETS:
                y = Y[k]; sh = E[k][xf] == 0; r["sets"][k] = {}
                for an, w in arms.items():
                    p = torch.einsum("rk,rkf->rf", w, Fm[k])
                    e = {"all": S.rel(p, y), "lit": S.rel(p, y, ~sh), "shadow": S.rel(p, y, sh), "shadow_ratio": float((p * sh).sum() / (y * sh).sum()),
                         "shadow_corr": float(np.corrcoef(p[sh].cpu().numpy(), y[sh].cpu().numpy())[0, 1])}
                    if k in dist:
                        for lo, hi in zip(DB[:-2], DB[1:-1]):
                            m = sh & (dist[k] >= lo) & (dist[k] < hi)
                            if m.sum() > 100 and float((y * m).abs().sum()) > 0:
                                e[f"shadow_d{lo:.2f}-{hi:.2f}"] = S.rel(p, y, m)
                        e["per_field_shadow"] = [S.rel(p[:, f], y[:, f], sh[:, f]) if sh[:, f].any() else None for f in range(y.shape[1])]
                    r["sets"][k][an] = e
            best = min(("A1v", "A1nn"), key=lambda an: r["sets"]["val"][an]["shadow"]); r["A1_star"] = best
            prim = {}
            for an, w in arms.items():
                p = torch.cat([torch.einsum("rk,rkf->rf", w, Fm[k]) for k in ("test_probe", "view_T0")], 1)
                y = torch.cat([Y[k] for k in ("test_probe", "view_T0")], 1); sh = torch.cat([E[k][xf] == 0 for k in ("test_probe", "view_T0")], 1)
                prim[an] = {"shadow": S.rel(p, y, sh), "all": S.rel(p, y)}
            r["primary_test"] = prim
            b0q = g1["channels"][c]["primary_test"]["A0"]["shadow"]; b0s = g1b["channels"][c]["primary_test"][g1b["channels"][c]["A1_star"]]["shadow"]
            r["B0_primary_shadow"] = {"quad": b0q, "A1star": b0s, "A1star_type": g1b["channels"][c]["A1_star"]}
            r["gain"] = {"quad": {"abs": b0q - prim["quad"]["shadow"], "rel": 1 - prim["quad"]["shadow"] / b0q},
                         "A1star": {"abs": b0s - prim[best]["shadow"], "rel": 1 - prim[best]["shadow"] / b0s}}
            cond["channels"][c] = r; print(tag, c, json.dumps({"prim": prim, "B0": r["B0_primary_shadow"], "gain": r["gain"]}), flush=True)
            np.savez(root / f"cache/gate2_{tag}_{c}.npz", quad=W.cpu().numpy(), A1v=wv.cpu().numpy(), A1nn=wnn.cpu().numpy())
        np.savez(root / f"cache/gate2_{tag}_elements_fit.npz", S=Ss.cpu().numpy(), E=Es.cpu().numpy())
        rec["conditions"][tag] = cond
        del Fm; torch.cuda.empty_cache()
    dec = {}
    if "B2p" in rec["conditions"]:
        for ft in ("quad", "A1star"):
            g = {c: rec["conditions"]["B2p"]["channels"][c]["gain"][ft] for c in "RGB"}
            dec[ft] = {"gains": g, "pass": bool(sum(g[c]["rel"] >= 0.25 for c in "RGB") >= 2 and any(g[c]["abs"] >= 0.05 for c in "RGB"))}
    rec["decision"] = {"by_fit_type": dec, "proceed_gate3": bool(dec.get("quad", {}).get("pass", False))}
    rec["cost"] = {"wall_s": time.time() - t00, "peak_GB": torch.cuda.max_memory_allocated() / 1e9}
    (out / "gate2.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec["decision"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
