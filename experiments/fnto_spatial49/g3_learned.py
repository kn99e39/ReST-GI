"""WL49 Gate 3 - matched learned comparison L0 (existing K44 band / mid representative elements) vs L1 (DASQ alpha 1.0
elements). Identical network, descriptors, objective, budget, seeds and evaluation; only the element set differs.

Prediction (linear in the field): y_c(x) = sum_e exp(MLP(desc_e))_c * measure_e * mean field_e; desc_e (9) = element
centroid offset in the receiver frame / R_MID (3), distance / R_MID, element normal in the receiver frame (3),
log(measure / 1e-4) / 5, sqrt(measure) / max(distance, 1 mm) (clamped 10). measure = K44 represented area (L0) /
element area (L1). Target: oracle in-support response y_S30.

    python g3_learned.py <out root>
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s49_data as S  # noqa: E402
import dasq as DQ  # noqa: E402
from g1_capacity import DB  # noqa: E402
import fnto_model as M  # noqa: E402

PG = S.PROTO["gate3"]; STEPS = 20000; BR, BF = 64, 32; SEEDS = [4911, 4912]
SETS = ("fit", "val", "test_probe", "view_T0", "view_T1")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(9, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
        with torch.no_grad():
            self.f[-1].bias.fill_(math.log(10.0))

    def weights(self, desc, meas, valid):
        return torch.exp(self.f(desc).clamp(-30, 15)) * (meas * valid)[..., None]          # [R, K, 3]


def descriptors(x, nx, cen, nrm, meas, valid):
    t, b = M.det_frame(nx); Rm = torch.stack([t, b, nx], 1)                                   # [R, 3, 3]
    off = torch.einsum("rij,rkj->rki", Rm, cen - x[:, None]); d = off.norm(dim=-1, keepdim=True)
    ne = torch.einsum("rij,rkj->rki", Rm, nrm)
    la = torch.log(meas.clamp(min=1e-12) / 1e-4)[..., None] / 5; sz = (meas.clamp(min=0).sqrt()[..., None] / d.clamp(min=1e-3)).clamp(max=10)
    return torch.cat([off / S.R_IN, d / S.R_IN, ne, la, sz], -1) * valid[..., None]


@torch.no_grad()
def arm_tables(arm, L, T, ports, E):
    """Descriptors [R, K, 9], measure [R, K], valid [R, K] and per-set features {set: [R, K, F]} for receiver ports."""
    xi = torch.as_tensor(ports, device=L.dev); x = L.pos[xi]; nx = L.nrm[xi]
    if arm == "L0":
        nbr, nbrA, mid, midA = L.rep_lists(xi)
        cen = torch.cat([L.bpos[nbr.clamp(min=0)], L.cpos[mid.clamp(min=0)]], 1); nrm = torch.cat([L.bnrm[nbr.clamp(min=0)], L.cnrm[mid.clamp(min=0)]], 1)
        meas = torch.cat([nbrA, midA], 1).clamp(min=0); valid = torch.cat([nbr >= 0, mid >= 0], 1).float()
        feats = {}
        for k, Ek in E.items():
            Bm, Cm = L.agg_means(Ek); feats[k] = torch.cat([Bm[nbr.clamp(min=0)], Cm[mid.clamp(min=0)]], 1) * valid[..., None]
    else:
        parts = [T.elements(x[i:i + 4000], xi[i:i + 4000], 1.0, S.R_IN)[:2] for i in range(0, len(xi), 4000)]   # receiver chunks (traversal memory)
        K = max(p_[0].shape[1] for p_ in parts)
        pad = lambda a: torch.cat([a, torch.full((a.shape[0], K - a.shape[1]), -1, dtype=a.dtype, device=a.device)], 1)  # noqa: E731
        Ss = torch.cat([pad(p_[0]) for p_ in parts]); Es = torch.cat([pad(p_[1]) for p_ in parts])
        cen, nrm, meas = DQ.elem_geom(T, Ss, Es); valid = (Ss >= 0).float()
        feats = {}
        for k, Ek in E.items():
            cs = T.cs_field(L.area, Ek)
            feats[k] = torch.cat([T.means(cs, Ss[i:i + 200], Es[i:i + 200]).float() for i in range(0, Ss.shape[0], 200)]); del cs
    return descriptors(x, nx, cen, nrm, meas, valid), meas, valid, feats


def predict(net, D, meas, valid, F):
    w = net.weights(D, meas, valid); return torch.einsum("rkc,rkf->rcf", w, F)                 # [R, 3, F]


def evaluate(net, tab, Y, Erx, dist=None, chunk=200):
    D, meas, valid, feats = tab; out = {}
    with torch.no_grad():
        for k in feats:
            p = torch.cat([predict(net, D[i:i + chunk], meas[i:i + chunk], valid[i:i + chunk], feats[k][i:i + chunk]) for i in range(0, D.shape[0], chunk)])
            out[k] = p.double()
    return out


def metrics(P, Y, Erx, dist):
    res = {}
    for k in P:
        sh = Erx[k] == 0; r = {}
        for ci, c in enumerate("RGB"):
            p, y = P[k][:, ci], Y[k][c]
            e = {"all": S.rel(p, y), "lit": S.rel(p, y, ~sh), "shadow": S.rel(p, y, sh), "shadow_ratio": float((p * sh).sum() / (y * sh).sum()),
                 "shadow_corr": float(np.corrcoef(p[sh].cpu().numpy(), y[sh].cpu().numpy())[0, 1])}
            if dist is not None and k in dist:
                for lo, hi in zip(DB[:-2], DB[1:-1]):
                    m = sh & (dist[k] >= lo) & (dist[k] < hi)
                    if m.sum() > 100 and float((y * m).abs().sum()) > 0:
                        e[f"shadow_d{lo:.2f}-{hi:.2f}"] = S.rel(p, y, m)
                e["per_field_shadow"] = [S.rel(p[:, f], y[:, f], sh[:, f]) if sh[:, f].any() else None for f in range(y.shape[1])]
            r[c] = e
        res[k] = r
    return res


def main(root):
    root = Path(root); out = root / "gate3_learned_ab"; out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda"); t00 = time.time()
    L = S.Lego(dev); T = DQ.Octree(L.pos, L.area); DQ.set_normals(T, L.nrm); sp = S.splits(); rf, rh = S.receivers(L); nf = len(rf)
    with torch.no_grad():
        E = {"fit": S.probe_fields(L, "train", sp["fit"]), "val": S.probe_fields(L, "train", sp["val"]), "test_probe": S.probe_fields(L, "holdout", sp["test_probe"]),
             "view_T0": S.view_fields(L, "T0"), "view_T1": S.view_fields(L, "T1")}
        xh = torch.as_tensor(rh, device=dev); dist_h = {k: S.dist_to_lit(L, xh, E[k]) for k in ("test_probe", "view_T0")}
    Z_ = np.load(root / "cache/oracle.npz")
    Yf = {k: {c: torch.as_tensor(Z_[f"{c}_{k}_S30"][:nf], device=dev) for c in "RGB"} for k in SETS}
    Yh = {k: {c: torch.as_tensor(Z_[f"{c}_{k}_S30"][nf:], device=dev) for c in "RGB"} for k in SETS}
    Erf = {k: E[k][torch.as_tensor(rf, device=dev)] for k in SETS}; Erh = {k: E[k][xh] for k in SETS}
    rec = {"steps": STEPS, "batch": [BR, BF], "seeds": SEEDS, "arms": {}}
    for arm in ("L0", "L1"):
        t = time.time(); tab_f = arm_tables(arm, L, T, rf, E); tab_h = arm_tables(arm, L, T, rh, {k: E[k] for k in ("val", "test_probe", "view_T0", "view_T1")}); t_tab = time.time() - t
        D, meas, valid, feats = tab_f; Kel = valid.sum(1)
        ar = {"elements_median": float(Kel.median()), "elements_q95": float(torch.quantile(Kel, 0.95)), "t_tables_s": t_tab, "seeds": {}}
        Yfit = torch.stack([Yf["fit"][c] for c in "RGB"], 1).float()                        # [R, 3, F]
        for seed in SEEDS:
            torch.manual_seed(seed); g = torch.Generator(device=dev).manual_seed(seed); net = Net().to(dev)
            ar["params"] = sum(p.numel() for p in net.parameters())
            opt = torch.optim.Adam(net.parameters(), lr=1e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
            hist = []; vhist = []; t = time.time()
            Yval = torch.stack([Yf["val"][c] for c in "RGB"], 1).float()

            def val_loss():
                with torch.no_grad():
                    pv = torch.cat([predict(net, D[i:i + 250], meas[i:i + 250], valid[i:i + 250], feats["val"][i:i + 250]) for i in range(0, nf, 250)])
                    return float((((pv - Yval) ** 2).sum((0, 2)) / (Yval ** 2).sum((0, 2))).sum())
            for step in range(STEPS):
                if step % (STEPS // 20) == 0:
                    vhist.append(val_loss())
                ri = torch.randint(0, nf, (BR,), generator=g, device=dev); fi = torch.randint(0, Yfit.shape[2], (BF,), generator=g, device=dev)
                p = predict(net, D[ri], meas[ri], valid[ri], feats["fit"][ri][:, :, fi]); y = Yfit[ri][:, :, fi]
                loss = (((p - y) ** 2).sum((0, 2)) / (y ** 2).sum((0, 2)).clamp(min=1e-30)).sum()
                opt.zero_grad(); loss.backward(); opt.step(); sched.step(); hist.append(float(loss))
            torch.cuda.synchronize(); t_train = time.time() - t
            vhist.append(val_loss()); h = np.array(hist); seg = len(h) // 10
            plateau = float(abs(vhist[-1] - vhist[-3]) / max(vhist[-3], 1e-30))                # validation loss, last 10 % of steps (amendment 6)
            plateau_train = float(abs(h[-seg:].mean() - h[-2 * seg:-seg].mean()) / max(h[-2 * seg:-seg].mean(), 1e-30))
            Pf = evaluate(net, tab_f, Yf, Erf); Ph = evaluate(net, tab_h, Yh, Erh)
            r = {"t_train_s": t_train, "loss_last10pct": float(h[-seg:].mean()), "plateau_rel_change_val": plateau, "plateau_rel_change_train_minibatch": plateau_train, "interpretable_state": plateau < 0.02, "val_loss_curve": vhist,
                 "loss_curve_every100": h[::100].tolist(), "fit_receivers": metrics(Pf, Yf, Erf, None), "heldout_receivers": metrics(Ph, Yh, Erh, dist_h)}
            prim = {}
            for c in "RGB":
                ci = "RGB".index(c); p = torch.cat([Ph[k][:, ci] for k in ("test_probe", "view_T0")], 1); y = torch.cat([Yh[k][c] for k in ("test_probe", "view_T0")], 1)
                sh = torch.cat([Erh[k] == 0 for k in ("test_probe", "view_T0")], 1); prim[c] = {"shadow": S.rel(p, y, sh), "all": S.rel(p, y)}
            r["primary_heldout_test"] = prim; ar["seeds"][str(seed)] = r
            torch.save(net.state_dict(), out / f"ckpt_{arm}_seed{seed}.pt"); print(arm, seed, json.dumps(prim), f"plateau {plateau:.4f}", f"{t_train:.0f}s", flush=True)
        rec["arms"][arm] = ar
        del tab_f, tab_h, D, meas, valid, feats; torch.cuda.empty_cache()
    dec = {}
    for seed in map(str, SEEDS):
        g = {c: 1 - rec["arms"]["L1"]["seeds"][seed]["primary_heldout_test"][c]["shadow"] / rec["arms"]["L0"]["seeds"][seed]["primary_heldout_test"][c]["shadow"] for c in "RGB"}
        dec[seed] = {"rel_gain": g, "pass": sum(v >= 0.15 for v in g.values()) >= 2,
                     "both_interpretable": rec["arms"]["L0"]["seeds"][seed]["interpretable_state"] and rec["arms"]["L1"]["seeds"][seed]["interpretable_state"]}
    rec["decision"] = {"per_seed": dec, "L1_better": all(d["pass"] for d in dec.values()), "interpretable": all(d["both_interpretable"] for d in dec.values())}
    rec["cost"] = {"wall_s": time.time() - t00, "peak_GB": torch.cuda.max_memory_allocated() / 1e9}
    (out / "gate3.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec["decision"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
