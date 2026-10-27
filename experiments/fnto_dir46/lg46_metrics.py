"""WL46 LG metrics (numpy): full-suite conditions (R0 production, R4 incoming-aware K_loc sender) on the historical held-out
suite, regime r1 (full_r1 references), display PSNR as worklogs 37-45.

Per condition: T0 S_T0 PSNR (all / Yellow / opaque / glass / box), T1 E_upd and D_stale PSNR (all, A_changed), indirect
component pred / ref per class on T0 (pred = S_T0 - frozen K_only), edit signed change (total and indirect, matched Lego
pixels, A_changed and stationary) as worklog 45 b4_edit; condition vs R0: view-level paired difference with bootstrap
95 % CI (2,000 resamples, seed 46), number of improving views. R0 vs the frozen worklog-44 eval (consistency).

    python lg46_metrics.py <lg dir> <out json> [conds=R0,R4]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parent / "fnto_lego"))
import r1_common as C  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722]); CLS = {"Yellow": [0], "opaque": [1, 3, 5, 6, 7], "glass": [2, 4], "box": [8]}


def psnr(p, r):
    return float(10 * np.log10(1 / max(((C.ldr3(p) - C.ldr3(r)) ** 2).mean(), 1e-30)))


def edit_stats(p, r, m):
    p, r = p[m], r[m]
    return {"px": int(m.sum()), "corr": float(np.corrcoef(p, r)[0, 1]), "slope": float((p * r).sum() / max((r * r).sum(), 1e-30)),
            "explained": float(1 - ((p - r) ** 2).sum() / max((r * r).sum(), 1e-30)), "magnitude_ratio": float(np.abs(p).sum() / max(np.abs(r).sum(), 1e-30))}


def main(lg, out, conds=("R0", "R4")):
    lg = Path(lg); Vf = np.load(C.OUT1 / "eval_K_k44P/views.npz"); B = np.load(C.B4 / "eval/views.npz")
    px = {s: np.load(C.OUT1 / f"eval_px/heldout_{s}.npz") for s in ("T0", "T1")}
    R = {(s, m): np.load(C.OUT1 / f"refs/heldout_{s}_{m}.npz") for s in ("T0", "T1") for m in ("first", "full")}
    rec = {"regime": "r1 (full_r1 references, display PSNR ldr3); K_only from the frozen worklog-44 eval (condition-independent)", "per_cond": {}, "per_view": {}}
    for c in conds:
        V = np.load(lg / c / "views.npz"); pv = {}
        for v in range(int(px["T0"]["nv"])):
            val = px["T0"][f"v{v}_valid"].astype(bool); mat = np.where(px["T0"][f"v{v}_dom"][val] == 1, 8, px["T0"][f"v{v}_mat"][val])
            fu = R[("T0", "full")][f"v{v}_rgb"][val].astype(np.float64); fr = R[("T0", "first")][f"v{v}_rgb"][val].astype(np.float64)
            s0 = V[f"heldout_T0_v{v}_S_T0"][:, :3].astype(np.float64); ko = Vf[f"heldout_T0_v{v}_K_only"][:, :3].astype(np.float64)
            r_ = {"T0_all": psnr(s0, fu)}
            pind, rind = (s0 - ko) @ LUM, (fu - fr) @ LUM
            for cn, ids in CLS.items():
                m = np.isin(mat, ids)
                if m.sum() > 100:
                    r_[f"T0_{cn}"] = psnr(s0[m], fu[m]); r_[f"ind_ratio_{cn}"] = float(pind[m].sum() / max(rind[m].sum(), 1e-30))
            if c == "R0":
                r_["frozen_eval_T0_all"] = psnr(Vf[f"heldout_T0_v{v}_S_T0"][:, :3].astype(np.float64), fu)
            val1 = px["T1"][f"v{v}_valid"].astype(bool); mk1 = px["T1"][f"v{v}_mk"][val1]; fu1 = R[("T1", "full")][f"v{v}_rgb"][val1].astype(np.float64)
            chg = B[f"heldout_T1_v{v}_reg_A_changed"].ravel()[mk1].astype(bool)
            for arm in ("E_upd", "D_stale"):
                p1 = V[f"heldout_T1_v{v}_{arm}"][:, :3].astype(np.float64)
                r_[f"T1_{arm}_all"] = psnr(p1, fu1)
                if chg.sum() > 100:
                    r_[f"T1_{arm}_A_changed"] = psnr(p1[chg], fu1[chg])
            pv[v] = r_
        rec["per_view"][c] = pv
        keys = sorted({k for x in pv.values() for k in x})
        rec["per_cond"][c] = {"mean": {k: float(np.mean([x[k] for x in pv.values() if k in x])) for k in keys}}
        # edit (matched Lego pixels, as worklog 45 b4_edit)
        cols = {k: [] for k in ("tot_p", "tot_r", "ind_p", "ind_r", "chg", "unch")}
        for v in range(int(px["T1"]["nv"])):
            p0, p1 = px["T0"], px["T1"]
            va0, va1 = p0[f"v{v}_valid"].astype(bool), p1[f"v{v}_valid"].astype(bool)
            l0, l1 = (p0[f"v{v}_dom"] == 0) & va0, (p1[f"v{v}_dom"] == 0) & va1
            com, i0, i1 = np.intersect1d(p0[f"v{v}_mk"][l0], p1[f"v{v}_mk"][l1], return_indices=True)
            r0 = np.cumsum(va0)[l0][i0] - 1; r1 = np.cumsum(va1)[l1][i1] - 1
            g = lambda s, arm, rr: V[f"heldout_{s}_v{v}_{arm}"][rr, :3].astype(np.float64) @ LUM  # noqa: E731
            k0 = Vf[f"heldout_T0_v{v}_K_only"][r0, :3].astype(np.float64) @ LUM; k1 = Vf[f"heldout_T1_v{v}_K_only"][r1, :3].astype(np.float64) @ LUM
            rr_ = lambda s, m, l, i: R[(s, m)][f"v{v}_rgb"][l][i].astype(np.float64) @ LUM  # noqa: E731
            F0, F1, f0, f1 = rr_("T0", "full", l0, i0), rr_("T1", "full", l1, i1), rr_("T0", "first", l0, i0), rr_("T1", "first", l1, i1)
            cols["tot_p"].append(g("T1", "E_upd", r1) - g("T0", "S_T0", r0)); cols["tot_r"].append(F1 - F0)
            cols["ind_p"].append((g("T1", "E_upd", r1) - k1) - (g("T0", "S_T0", r0) - k0)); cols["ind_r"].append((F1 - f1) - (F0 - f0))
            cols["chg"].append(B[f"heldout_T1_v{v}_reg_A_changed"].ravel()[com].astype(bool)); cols["unch"].append(B[f"heldout_T1_v{v}_reg_A_unchanged"].ravel()[com].astype(bool))
        d = {k: np.concatenate(x) for k, x in cols.items()}
        rec["per_cond"][c]["edit"] = {f"{q}_{reg}": edit_stats(d[f"{q}_p"], d[f"{q}_r"], d[m]) for q in ("tot", "ind") for reg, m in (("A_changed", "chg"), ("stationary", "unch"))}
    rng = np.random.default_rng(46); rec["paired_vs_R0"] = {}
    for c in conds[1:]:
        rec["paired_vs_R0"][c] = {}
        for k in ("T0_all", "T0_Yellow", "T0_opaque", "T0_box", "T1_E_upd_all", "T1_E_upd_A_changed"):
            vs = [v for v in rec["per_view"]["R0"] if k in rec["per_view"]["R0"][v] and k in rec["per_view"][c][v]]
            dif = np.array([rec["per_view"][c][v][k] - rec["per_view"]["R0"][v][k] for v in vs])
            bs = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(2000)]
            rec["paired_vs_R0"][c][k] = {"mean_dB": float(dif.mean()), "ci95": np.quantile(bs, [0.025, 0.975]).tolist(), "n_pos": int((dif > 0).sum()), "n": int(len(dif))}
    Path(out).write_text(json.dumps(rec, indent=1)); print(json.dumps({"per_cond_mean": {c: rec["per_cond"][c]["mean"] for c in conds}, "paired": rec["paired_vs_R0"],
                                                                      "edit": {c: rec["per_cond"][c]["edit"] for c in conds}}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], tuple(a[2].split(",")) if len(a) > 2 else ("R0", "R4"))
