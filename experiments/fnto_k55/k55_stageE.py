"""WL55 Stage E (protocol fnto_k55, stage_E): non-rigid K variation at 2.5 mm and coupled stale-vs-current evaluation.

    P53_MI_VARIANT=cuda_ad_rgb python k55_stageE.py <out json>
Operators (material coordinates, 2.5 mm): rest slab_L4_s1/s2, bend90_L4_s1/s2, dent_L4_s1/s2. Change D = K_cur(s1) - K_rest(s1), seed noise
N = K_rest(s2) - K_rest(s1) (same noise level as D), at three aggregations: 2.5 mm cell pairs, the CAK cells of the deformed scene, 1 cm ports.
Coupled: D_img = pred(stale K) - pred(current K) per group (bin readout primary as WL54; nee+KB reported), with K-seed SE; criteria
A detect (||D|| / ||N|| >= 3), B predict (temporal change floor -> deformed tracked by the current-K prediction: slope in [0.9, 1.1], corr >= 0.9),
C rank (|D| > 2 (|B_cur| + 3 SE) and |B_cur| < |B_stale| by > 3 SE_D).
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

import k55_analyze as KA
import k55_common as KCm
import k55_fixtures as KF
import k55_ops as KO
import k55_stageB as SBd
import p53_fixtures as FX

OPS = SBd.OPS; SB = KA.SB


def cellpair(r, cof, ncell):
    """(3, ncell_out, ncell_in) totals, input averaged over the cell's in-cells by area weight already in aggregate (here: plain sums of
    fine entries, in-cells averaged uniformly: fine cells of equal rest area within a port)."""
    nrow, ncol = r["shape"]; row = r["idx"] // ncol; col = r["idx"] % ncol
    co = torch.as_tensor(cof)[(row // 64)]; ci = torch.as_tensor(cof)[(col // 16)]
    M = torch.zeros(3, ncell * ncell, dtype=torch.float64).index_add_(1, co * ncell + ci, r["val"].double())
    cnt = torch.bincount(torch.as_tensor(cof), minlength=ncell).double() * 16
    return M.view(3, ncell, ncell) / cnt[None, None, :]


def ring_energy(D, N):
    """energy of D and N by out-bin ring (8 x 8 concentric bins -> 4 rings), from fine raw operators D/N given as dicts."""
    out = {}
    for name, r in (("D", D), ("N", N)):
        row = r["idx"] // r["shape"][1]; ob = row % 64; iu, iv = ob % 8, ob // 8
        ring = torch.maximum((iu.float() - 3.5).abs(), (iv.float() - 3.5).abs()).long() // 1
        e = torch.zeros(3, 4, dtype=torch.float64).index_add_(1, ring.clamp(max=3), r["val"].double() ** 2)
        out[name] = e.tolist()
    return out


def diff_op(a, b):
    """raw sparse difference a - b (same shape)."""
    idx = torch.cat([a["idx"], b["idx"]]); val = torch.cat([a["val"], -b["val"]], 1)
    u, inv = torch.unique(idx, return_inverse=True); v = torch.zeros(3, len(u), dtype=torch.float64).index_add_(1, inv, val.double())
    return dict(idx=u, val=v.float(), shape=a["shape"])


def op_metrics(geo, sd):
    m, V = FX.domain_geometry("slab"); P = len(m["port_area"])
    rest1, rest2 = SBd.load_op("slab_L4_s1"), SBd.load_op("slab_L4_s2"); cur1, cur2 = SBd.load_op(f"{geo}_L4_s1"), SBd.load_op(f"{geo}_L4_s2")
    res = {}
    cak = SBd.cak_levels(sd)[0]
    for tag, cof, nc in (("fine_2p5mm", np.arange(P * 16), P * 16), ("cak_cells", KO.fine_to_cells(cak), int((cak ** 2).sum())), ("port_1cm", np.repeat(np.arange(P), 16), P)):
        Rr1, Rr2, C1, C2 = (cellpair(r, cof, nc) for r in (rest1, rest2, cur1, cur2))
        D = C1 - Rr1; N = Rr2 - Rr1; Dm = 0.5 * (C1 + C2) - 0.5 * (Rr1 + Rr2)
        rec = dict(cells=nc, fro_vs_noise=(D.norm(dim=(1, 2)) / N.norm(dim=(1, 2))).tolist(), fro_rel=(Dm.norm(dim=(1, 2)) / Rr1.norm(dim=(1, 2))).tolist(),
                   colsum_change_mean=((Dm.sum(1)) / Rr1.sum(1).clamp(min=1e-30)).mean(1).tolist(),
                   colsum_change_absmax=((Dm.sum(1)).abs() / Rr1.sum(1).clamp(min=1e-30)).max(1).values.tolist())
        if tag == "port_1cm":
            pc = m["port_center"]; dist = torch.tensor(np.linalg.norm(pc[:, None] - pc[None], axis=2))
            tot = (D ** 2).sum((1, 2)); Nt = (N ** 2).sum((1, 2))
            bins = [(0, 1e-9), (1e-9, 0.0151), (0.0151, 0.0301), (0.0301, 0.0501), (0.0501, 1.0)]
            rec["energy_by_distance_debiased"] = {f"{lo:.3f}-{hi:.3f}": [float(((D[k] ** 2)[(dist >= lo) & (dist < hi)].sum() - (N[k] ** 2)[(dist >= lo) & (dist < hi)].sum()) / max(float(tot[k] - Nt[k]), 1e-30)) for k in range(3)] for lo, hi in bins}
            sv = [torch.linalg.svdvals(Dm[k]) for k in range(3)]; svn = [torch.linalg.svdvals(N[k] / math.sqrt(2)) for k in range(3)]
            rec["rank_above_noise"] = [int((sv[k] > svn[k][0]).sum()) for k in range(3)]
            if geo == "dent":
                r_ = np.hypot(pc[:, 0], pc[:, 1]); out = []
                for k, Ld in enumerate((0.0181, 0.0058, 0.0018)):
                    sup = torch.tensor(r_ < 0.02 + 3 * Ld); both = sup[:, None] & sup[None, :]
                    out.append(float((D[k] ** 2)[both].sum() / tot[k]))
                rec["energy_within_dent_support"] = out
        res[tag] = rec
        del Rr1, Rr2, C1, C2, D, N, Dm
    res["out_bin_rings"] = ring_energy(diff_op(cur1, rest1), diff_op(rest2, rest1))
    res["photons_per_in_cell"] = rest1["n_per"]
    return res


def coupled(scene, rest_scene="floor"):
    sd, cyc = KA.cycles_of(scene); lab = KCm.labels_of(sd); sdr, cycr = KA.cycles_of(rest_scene); out = {}
    for cam, units in cyc.items():
        key = KCm.dirkey(sd["cams"][cam]["dir"]); u = units["amended"]; rec = {}
        for arm in ("P10", "CAK", "P025"):
            for ro in ("bin", "ball"):
                def ld(a):
                    z = np.load(SB / scene / f"{a}.npz"); kk = f"{key}|{ro}" if ro != "ball" else f"{key}|nee"
                    v = z[kk] + (z[f"{key}|ball"] if ro == "ball" else 0); return v
                try:
                    cur = [ld(f"{arm}_s{i}") for i in (1, 2)]; st = [ld(f"{arm}st_s{i}") for i in (1, 2)]
                except FileNotFoundError:
                    continue
                Bc = KCm.group_bias(np.mean(cur, 0), u["mean"], u["sel"], lab, u["se"]); Bs = KCm.group_bias(np.mean(st, 0), u["mean"], u["sel"], lab, u["se"])
                g = {}
                for gn in Bc:
                    if not any(Bc[gn]["ruled"]):
                        continue
                    m = u["sel"] & (lab == gn); sig = u["mean"][m].sum(0)
                    Ds = [(st[i][m] - cur[i][m]).sum(0) / sig for i in (0, 1)]; D = 0.5 * (Ds[0] + Ds[1]); seD = np.abs(Ds[0] - Ds[1]) / 2
                    bc = np.array(Bc[gn]["bias"]); bs = np.array(Bs[gn]["bias"]); se = np.array(Bc[gn]["se"])
                    seK = np.array([abs(KCm.group_bias(cur[0], u["mean"], u["sel"], lab)[gn]["bias"][c] - KCm.group_bias(cur[1], u["mean"], u["sel"], lab)[gn]["bias"][c]) / 2 for c in range(3)])
                    se_tot = np.hypot(se, seK)
                    g[gn] = dict(D=D.tolist(), se_D=seD.tolist(), B_cur=bc.tolist(), B_stale=bs.tolist(), se_cur=se_tot.tolist(),
                                 detect_3se=[bool(abs(D[c]) > 3 * max(seD[c], 1e-12)) for c in range(3)],
                                 rank=[bool(abs(D[c]) > 2 * (abs(bc[c]) + 3 * se_tot[c]) and abs(bc[c]) < abs(bs[c]) - 3 * seD[c]) for c in range(3)],
                                 limit=["bias_floor" if abs(bc[c]) + 3 * se_tot[c] > abs(D[c]) / 2 and abs(bc[c]) > 3 * se_tot[c] else ("oracle_variance" if 3 * se_tot[c] > abs(D[c]) / 2 else "none") for c in range(3)])
                rec[f"{arm}|{ro}"] = g
        # temporal change tracking (rest floor scene -> deformed), per port, current-K and stale-K predictions
        if cam in cycr:
            ur = cycr[cam]["amended"]; both = u["sel"] & ur["sel"]; med = np.array([l.startswith("dom") for l in lab])
            for arm in ("P10", "CAK"):
                try:
                    pc = np.mean([np.load(SB / scene / f"{arm}_s{i}.npz")[f"{key}|nee"] + np.load(SB / scene / f"{arm}_s{i}.npz")[f"{key}|ball"] for i in (1, 2)], 0)
                    ps = np.mean([np.load(SB / scene / f"{arm}st_s{i}.npz")[f"{key}|nee"] + np.load(SB / scene / f"{arm}st_s{i}.npz")[f"{key}|ball"] for i in (1, 2)], 0)
                    pr = np.mean([np.load(SB / rest_scene / f"{arm}_s{i}.npz")[f"{key}|nee"] + np.load(SB / rest_scene / f"{arm}_s{i}.npz")[f"{key}|ball"] for i in (1, 2)], 0)
                except FileNotFoundError:
                    continue
                mm = both & med
                if mm.sum() < 5:
                    continue
                dc = (u["mean"] - ur["mean"])[mm].ravel()
                for tag, p in (("current", pc), ("stale", ps)):
                    dp = (p - pr)[mm].ravel(); slope = float((dp * dc).sum() / (dc * dc).sum()); corr = float(np.corrcoef(dp, dc)[0, 1])
                    rec[f"track|{arm}|{tag}"] = dict(n_ports=int(mm.sum()), slope=slope, corr=corr, B_predict=bool(0.9 <= slope <= 1.1 and corr >= 0.9),
                                                     true_change_rel=float(np.abs(dc).sum() / np.abs(ur["mean"][mm]).sum()))
        out[cam] = rec
    return out


def main():
    res = {}
    for geo in ("bend90", "dent"):
        sd = KF.scene_def(geo); res[geo] = dict(operator=op_metrics(geo, sd))
        res[geo]["A_detect"] = [x >= 3 for x in res[geo]["operator"]["port_1cm"]["fro_vs_noise"]]
        if (KA.CYC / f"ccam_{geo}.json").exists():
            res[geo]["coupled"] = coupled(geo)
        torch.cuda.empty_cache()
    KCm.dump(res, Path(sys.argv[1])); print(json.dumps({g: {k: v["operator"][k]["fro_vs_noise"] for k in ("fine_2p5mm", "cak_cells", "port_1cm")} for g, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
