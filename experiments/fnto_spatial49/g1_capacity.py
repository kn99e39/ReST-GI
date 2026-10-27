"""WL49 Gate 1 - capacity of the EXISTING K_band / K_mid aggregation on the Yellow in-support Burley response.

A0 ideal-weight quadrature (worklog 48 Q restricted to r < 30 cm), A1 relaxed per-receiver least squares on the same 224
aggregate features, A1g = A1 + 64 global field-PCA features, collision e_coll (atom projection keeps every aggregate
feature). Builds and caches the oracle responses for Gates 2-3 (results/fnto_spatial49/cache).

    python g1_capacity.py <out root>
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
import r1_common as C  # noqa: E402

DB = [0, 0.01, 0.03, 0.10, 0.30, 1e9]


def solve_ls(Z, y, lam=1e-6):
    """Batched ridge: Z [R, F, D], y [R, F] -> w [R, D]."""
    G = Z.transpose(1, 2) @ Z; D = G.shape[-1]; tr = torch.diagonal(G, dim1=1, dim2=2).sum(-1) / D
    G = G + lam * tr[:, None, None] * torch.eye(D, device=Z.device, dtype=Z.dtype)[None]
    return torch.linalg.solve(G, (Z.transpose(1, 2) @ y[..., None]))[..., 0]


def feats(L, xi, Bm, Cm):
    nbr, nbrA, mid, midA = L.rep_lists(xi)
    fb = Bm[nbr.clamp(min=0)] * (nbr >= 0)[..., None]; fc = Cm[mid.clamp(min=0)] * (mid >= 0)[..., None]
    return torch.cat([fb, fc], 1)                                                # [R, 224, F]


@torch.no_grad()
def main(root):
    root = Path(root); out = root / "gate1_existing_capacity"; cache = root / "cache"; out.mkdir(parents=True, exist_ok=True); cache.mkdir(exist_ok=True)
    dev = torch.device("cuda"); t00 = time.time(); L = S.Lego(dev); sp = S.splits(); rf, rh = S.receivers(L)
    xi_all = torch.as_tensor(np.concatenate([rf, rh]), device=dev); nf = len(rf)
    t = time.time()
    E = {"fit": S.probe_fields(L, "train", sp["fit"]), "val": S.probe_fields(L, "train", sp["val"]), "test_probe": S.probe_fields(L, "holdout", sp["test_probe"]),
         "view_T0": S.view_fields(L, "T0"), "view_T1": S.view_fields(L, "T1")}
    t_fields = time.time() - t
    PE = {k: L.atom_project(E[k]) for k in ("test_probe", "view_T0")}
    dist = {k: S.dist_to_lit(L, xi_all[:nf], E[k]) for k in ("test_probe", "view_T0")}
    rec = {"receivers": {"fit": int(nf), "heldout": int(len(rh))}, "fields": {k: int(v.shape[1]) for k, v in E.items()},
           "split_disjoint": bool(len(np.intersect1d(sp["fit"], sp["val"])) == 0), "t_fields_s": t_fields, "channels": {}, "atoms": int(len(L.atom_u))}
    bmem, bptr = S.csr(L.f2b, L.NB); cmem, cptr = S.csr(L.f2c, L.NC)
    # global PCA features of the fit fields (area-weighted irradiance)
    AEf = (E["fit"] * L.area[:, None]).double(); Gm = AEf.T @ AEf; ev, U = torch.linalg.eigh(Gm); U = U[:, -64:]
    V = (AEf @ U) / torch.linalg.norm(AEf @ U, dim=0, keepdim=True)              # [nA, 64] orthonormal source basis
    gfe = {k: (V.T @ (E[k] * L.area[:, None]).double()).float() for k in E}         # [64, F]
    store = {}
    for ci, d in enumerate(S.dch()):
        c = "RGB"[ci]; tc = time.time()
        Y = {k: S.oracle(L, xi_all, E[k], d) for k in E}; Yc = {k: S.oracle(L, xi_all[:nf], PE[k], d)[0] for k in PE}
        for k in E:
            store[f"{c}_{k}_S30"] = Y[k][0].cpu().numpy(); store[f"{c}_{k}_tail"] = Y[k][1].cpu().numpy()
        t_or = time.time() - tc
        # A0 ideal-weight quadrature
        xf = xi_all[:nf]; nbr, nbrA, mid, midA = L.rep_lists(xf)
        ab = S.cell_avg(L, xf, nbr, bmem, bptr, d, 0.0, C.R_LOCAL); ac = S.cell_avg(L, xf, mid, cmem, cptr, d, C.R_LOCAL, S.R_IN)
        cb = ab * nbrA * (nbr >= 0); cc = ac * midA * (mid >= 0)                          # x mean field of each element
        coef = torch.cat([cb, cc], 1)
        pred = {}; Zs = {}
        for k in E:
            Bm, Cm = L.agg_means(E[k]); Z = feats(L, xf, Bm, Cm); Zs[k] = Z
            pred[("A0", k)] = torch.einsum("rd,rdf->rf", coef, Z).double()
        # A1 / A1g relaxed least squares on the fit fields (fit receivers)
        tl = time.time(); yfit = Y["fit"][0][:nf].float()
        w1 = solve_ls(Zs["fit"].transpose(1, 2), yfit)
        Zg = {k: torch.cat([Zs[k], gfe[k][None].expand(nf, -1, -1)], 1) for k in E}
        w2 = solve_ls(Zg["fit"].transpose(1, 2), yfit); t_ls = time.time() - tl
        for k in E:
            pred[("A1", k)] = torch.einsum("rd,rdf->rf", w1, Zs[k]).double(); pred[("A1g", k)] = torch.einsum("rd,rdf->rf", w2, Zg[k]).double()
        res = {"t_oracle_s": t_or, "t_ls_s": t_ls, "A1_params_per_receiver": int(w1.shape[1]), "A1g_params_per_receiver": int(w2.shape[1]), "sets": {}}
        for k in E:
            y = Y[k][0][:nf]; lit = E[k][xf] > 0; sh = ~lit
            r = {}
            for a in ("A0", "A1", "A1g"):
                p = pred[(a, k)]
                r[a] = {"all": S.rel(p, y), "lit": S.rel(p, y, lit), "shadow": S.rel(p, y, sh),
                        "shadow_ratio": float((p * sh).sum() / (y * sh).sum()), "shadow_corr": float(np.corrcoef(p[sh].cpu().numpy(), y[sh].cpu().numpy())[0, 1])}
                if k in dist:
                    for lo, hi in zip(DB[:-1], DB[1:]):
                        m = sh & (dist[k] >= lo) & (dist[k] < hi)
                        if m.sum() > 100:
                            r[a][f"shadow_d{lo:.2f}-{min(hi, 9):.2f}"] = S.rel(p, y, m)
                if k in ("test_probe", "view_T0"):
                    r[a]["per_field_shadow"] = [S.rel(p[:, f], y[:, f], sh[:, f]) if sh[:, f].any() else None for f in range(y.shape[1])]
            if k in Yc:
                yc = Yc[k]; r["collision"] = {"all": S.rel(yc, y), "shadow": S.rel(yc, y, sh), "lit": S.rel(yc, y, lit),
                                              "per_field_shadow": [S.rel(yc[:, f], y[:, f], sh[:, f]) if sh[:, f].any() else None for f in range(y.shape[1])]}
                for lo, hi in zip(DB[:-1], DB[1:]):
                    m = sh & (dist[k] >= lo) & (dist[k] < hi)
                    if m.sum() > 100:
                        r["collision"][f"shadow_d{lo:.2f}-{min(hi, 9):.2f}"] = S.rel(yc, y, m)
            r["tail_share_shadow"] = float((Y[k][1][:nf] * sh).sum() / ((Y[k][1][:nf] + y) * sh).sum())
            res["sets"][k] = r
        # pooled primary test set (test_probe + view_T0)
        prim = {}
        for a in ("A0", "A1", "A1g"):
            p = torch.cat([pred[(a, "test_probe")], pred[(a, "view_T0")]], 1); y = torch.cat([Y["test_probe"][0][:nf], Y["view_T0"][0][:nf]], 1)
            sh = torch.cat([E["test_probe"][xf] == 0, E["view_T0"][xf] == 0], 1); prim[a] = {"shadow": S.rel(p, y, sh), "all": S.rel(p, y)}
        yc = torch.cat([Yc["test_probe"], Yc["view_T0"]], 1); prim["collision"] = {"shadow": S.rel(yc, y, sh), "all": S.rel(yc, y)}
        res["primary_test"] = prim; rec["channels"][c] = res
        np.savez(cache / f"gate1_coef_{c}.npz", A0=coef.cpu().numpy(), A1=w1.cpu().numpy(), A1g=w2.cpu().numpy())
        print(c, json.dumps(prim), f"{time.time() - tc:.0f}s", flush=True)
        del Zs, Zg, pred; torch.cuda.empty_cache()
    pr = {c: rec["channels"][c]["primary_test"] for c in "RGB"}
    suff = all(pr[c]["A1"]["shadow"] <= 0.05 for c in "RGB")
    proceed = any(pr[c]["A1"]["shadow"] >= 0.10 or pr[c]["collision"]["shadow"] >= 0.05 for c in "RGB")
    rec["decision"] = {"sufficient": suff, "proceed_gate2": (not suff) and proceed, "outcome4_stop": (not suff) and (not proceed)}
    np.savez(cache / "oracle.npz", rf=rf, rh=rh, fit_ids=sp["fit"], val_ids=sp["val"], **store,
             **{f"Erx_{k}": E[k][xi_all].cpu().numpy() for k in E}, **{f"dist_{k}": dist[k].cpu().numpy() for k in dist}, V64=V.float().cpu().numpy())
    rec["cost"] = {"wall_s": time.time() - t00, "peak_GB": torch.cuda.max_memory_allocated() / 1e9, "gpu": torch.cuda.get_device_name(0)}
    (out / "gate1.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec["decision"]))


if __name__ == "__main__":
    main(sys.argv[1])
