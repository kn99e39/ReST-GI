"""WL50 Gate 0/1 - matched training of the Yellow nonlocal arms on the shared residual target (protocol fnto_comp50).

L0 (K44 representative elements) and L1 (DASQ alpha 1.0 elements), identical worklog-49 network, objective, schedule,
batch and seeds. Batch = 8 fit configurations x 64 (port, config) target rows; loss = per-channel relative MSE of the
port-mean residual; Adam lr 1e-3, cosine to 0 over STEPS; validation curve on a fixed 30,000-row val subset every STEPS/30 steps; final metrics on all val / test rows.
Fair state (predeclared): relative range (max - min) / mean of the last 5 validation evaluations < 2 %.

    python c50_train.py <targets dir> <out dir>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c50_common as CC  # noqa: E402

STEPS = 30000; NCFG = 8; NROW = 64; SEEDS = [5011, 5012]; NEVAL = 30


class Feats:
    """Element tables for a set of receiver ports and per-configuration element features."""
    def __init__(self, kind, geo, ports):
        L, T, dev = geo.L, geo.T, geo.dev; self.kind, self.geo = kind, geo
        self.ports = torch.as_tensor(ports, device=dev, dtype=torch.long); x = L.pos[self.ports]; nx = L.nrm[self.ports]
        Ds, Ms, Vs, Ss, Es, NB, MD = [], [], [], [], [], [], []
        for i in range(0, len(ports), 4000):
            xi = self.ports[i:i + 4000]
            if kind == "L1":
                s_, e_, _ = T.elements(L.pos[xi], xi, 1.0, CC.S49.R_IN); cen, nrm, meas = CC.DQ.elem_geom(T, s_, e_); valid = (s_ >= 0).float(); Ss.append(s_.int()); Es.append(e_.int())
            else:
                nbr, nbrA, mid, midA = L.rep_lists(xi)
                cen = torch.cat([L.bpos[nbr.clamp(min=0)], L.cpos[mid.clamp(min=0)]], 1); nrm = torch.cat([L.bnrm[nbr.clamp(min=0)], L.cnrm[mid.clamp(min=0)]], 1)
                meas = torch.cat([nbrA, midA], 1).clamp(min=0); valid = torch.cat([nbr >= 0, mid >= 0], 1).float(); NB.append(nbr); MD.append(mid)
            Ds.append(CC.G3.descriptors(L.pos[xi], L.nrm[xi], cen, nrm, meas, valid).half()); Ms.append(meas); Vs.append(valid)
        K = max(d.shape[1] for d in Ds)
        pad = lambda a, v: torch.cat([a, torch.full((a.shape[0], K - a.shape[1]) + a.shape[2:], v, dtype=a.dtype, device=a.device)], 1)  # noqa: E731
        self.D = torch.cat([pad(d, 0) for d in Ds]); self.meas = torch.cat([pad(m, 0) for m in Ms]); self.valid = torch.cat([pad(v, 0) for v in Vs])
        if kind == "L1":
            self.S = torch.cat([pad(s, -1) for s in Ss]); self.E = torch.cat([pad(e, -1) for e in Es])
        else:
            self.nbr = torch.cat(NB); self.mid = torch.cat(MD)
        self.K = K; self.pos_of = {int(p): i for i, p in enumerate(ports)}

    def features(self, E1, rows):
        """E1 [nA] field of one configuration; rows = indices into self.ports -> [len(rows), K] area-mean features."""
        L, T = self.geo.L, self.geo.T
        if self.kind == "L1":
            AE = (L.area.double() * E1.double())[T.order]; cs = torch.cat([torch.zeros(1, device=E1.device, dtype=torch.float64), torch.cumsum(AE, 0)])
            s = self.S[rows].long(); e = self.E[rows].long(); ok = s >= 0; s, e = s.clamp(min=0), e.clamp(min=0)
            return ((cs[e] - cs[s]) / (T.csA[e] - T.csA[s]).clamp(min=1e-30) * ok).float()
        Bm, Cm = L.agg_means(E1[:, None]); nb, md = self.nbr[rows], self.mid[rows]
        return torch.cat([Bm[nb.clamp(min=0), 0] * (nb >= 0), Cm[md.clamp(min=0), 0] * (md >= 0)], 1)

    def predict(self, net, E1, rows):
        w = net.weights(self.D[rows].float(), self.meas[rows], self.valid[rows]); return (w * self.features(E1, rows)[..., None]).sum(1)


def main(tdir, out):
    tdir, out = Path(tdir), Path(out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda"); t00 = time.time()
    geo = CC.Geo(dev); tg = np.load(tdir / "targets.npz"); fields = torch.as_tensor(np.load(tdir / "fields_f16.npy"), device=dev)   # [ncfg, nA] f16
    ports = np.unique(tg["port"]); rows_port = np.searchsorted(ports, tg["port"])
    R = torch.as_tensor(tg["res"], device=dev).float(); cfg = torch.as_tensor(tg["cfg"], device=dev); rp = torch.as_tensor(rows_port, device=dev); spl = tg["split"]
    idx = {s: np.flatnonzero(spl == k) for k, s in enumerate(("fit", "val", "test"))}
    idx["val_curve"] = np.sort(np.random.default_rng(5010).choice(idx["val"], min(30000, len(idx["val"])), replace=False))   # fixed curve subset
    fit_by_cfg = {}
    for r in idx["fit"]:
        fit_by_cfg.setdefault(int(tg["cfg"][r]), []).append(r)
    fit_cfgs = np.array(sorted(fit_by_cfg)); fit_rows = {c: torch.as_tensor(v, device=dev) for c, v in fit_by_cfg.items()}
    rec = {"steps": STEPS, "batch": [NCFG, NROW], "seeds": SEEDS, "rows": {k: int(len(v)) for k, v in idx.items()}, "ports": int(len(ports)), "arms": {}}

    def eval_rows(F_, net, rows_np):
        P_ = torch.zeros(len(rows_np), 3, device=dev); rr = torch.as_tensor(rows_np, device=dev)
        for c in np.unique(tg["cfg"][rows_np]):
            m = torch.as_tensor(np.flatnonzero(tg["cfg"][rows_np] == c), device=dev)
            P_[m] = F_.predict(net, fields[int(c)].float(), rp[rr[m]])
        return P_

    def relerr(P_, rows_np):
        y = R[torch.as_tensor(rows_np, device=dev)]
        return [float((P_[:, c] - y[:, c]).norm() / y[:, c].norm()) for c in range(3)]
    for arm in ("L0", "L1"):
        t = time.time(); F_ = Feats(arm, geo, ports); t_tab = time.time() - t
        ar = {"K": F_.K, "elements_median": float(F_.valid.sum(1).median()), "t_tables_s": t_tab, "seeds": {}}
        for seed in SEEDS:
            torch.manual_seed(seed); g = torch.Generator(device=dev).manual_seed(seed); rng = np.random.default_rng(seed)
            net = CC.G3.Net().to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
            vcurve, hist = [], []; t = time.time()
            for step in range(STEPS + 1):
                if step % (STEPS // NEVAL) == 0:
                    with torch.no_grad():
                        vcurve.append(float(sum(relerr(eval_rows(F_, net, idx["val_curve"]), idx["val_curve"]))))
                if step == STEPS:
                    break
                loss = 0.0; num = torch.zeros(3, device=dev); den = torch.zeros(3, device=dev)
                for c in rng.choice(fit_cfgs, NCFG, replace=False):
                    rws = fit_rows[int(c)]; pick = rws[torch.randint(0, len(rws), (min(NROW, len(rws)),), generator=g, device=dev)]
                    p = F_.predict(net, fields[int(c)].float(), rp[pick]); y = R[pick]
                    num = num + ((p - y) ** 2).sum(0); den = den + (y ** 2).sum(0)
                loss = (num / den.clamp(min=1e-30)).sum(); opt.zero_grad(); loss.backward(); opt.step(); sch.step(); hist.append(float(loss))
            torch.cuda.synchronize(); t_train = time.time() - t
            last = np.array(vcurve[-5:]); fair = float((last.max() - last.min()) / last.mean())
            with torch.no_grad():
                Pv, Pt = eval_rows(F_, net, idx["val"]), eval_rows(F_, net, idx["test"])
            r = {"t_train_s": t_train, "val_curve": vcurve, "train_loss_every500": hist[::500], "fair_state_last5_rel_range": fair, "fair_state": fair < 0.02,
                 "val_relL2_RGB": relerr(Pv, idx["val"]), "test_relL2_RGB": relerr(Pt, idx["test"]), "params": sum(p.numel() for p in net.parameters())}
            yt = R[torch.as_tensor(idx["test"], device=dev)].cpu().numpy(); bm = tg["k44_bm"][idx["test"]]
            r["test_K44_bandmid_relL2_RGB"] = [float(np.linalg.norm(bm[:, c] - yt[:, c]) / np.linalg.norm(yt[:, c])) for c in range(3)]
            ar["seeds"][str(seed)] = r; torch.save(net.state_dict(), out / f"ckpt_{arm}_seed{seed}.pt")
            print(arm, seed, "val", [round(x, 4) for x in r["val_relL2_RGB"]], "test", [round(x, 4) for x in r["test_relL2_RGB"]], "fair", round(fair, 4), f"{t_train:.0f}s", flush=True)
        rec["arms"][arm] = ar; del F_; torch.cuda.empty_cache()
    dec = {}
    for s in map(str, SEEDS):
        a0, a1 = rec["arms"]["L0"]["seeds"][s], rec["arms"]["L1"]["seeds"][s]
        dec[s] = {"rel_gain_test_RGB": [1 - x1 / x0 for x0, x1 in zip(a0["test_relL2_RGB"], a1["test_relL2_RGB"])], "both_fair": a0["fair_state"] and a1["fair_state"]}
    rec["decision"] = {"per_seed": dec, "learned_comparison_interpretable": all(d["both_fair"] for d in dec.values())}
    rec["cost"] = {"wall_s": time.time() - t00, "peak_GB": torch.cuda.max_memory_allocated() / 1e9}
    (out / "train.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec["decision"], indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
