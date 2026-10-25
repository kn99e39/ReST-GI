"""WL45 Track B4 / B5 - the held-out edit split into direct / intrinsic and indirect components (numpy; frozen
K_k44P predictions from eval_K_k44P/views.npz, regime r1 references, matched Lego pixels present in both views).

  direct / intrinsic change  pred: K_only(T1) - K_only(T0)                  ref: first(T1) - first(T0)
  indirect change, updated   pred: (E_upd - K_only)(T1) - (S_T0 - K_only)(T0)   ref: (full - first)(T1) - (full - first)(T0)
  indirect change, stale     pred: (D_stale - K_only)(T1) - (S_T0 - K_only)(T0)
  total change               pred: E_upd(T1) - S_T0(T0)                      ref: full(T1) - full(T0)
Per region (A_changed / A_unchanged = stationary, worklog-37 masks) and material class: sign agreement where the
reference change exceeds 3 sigma (sigma = MAD of the reference change on stationary pixels, per quantity),
correlation, least-squares slope, magnitude ratio sum|pred| / sum|ref|, net energy ratio sum pred / sum ref,
explained fraction; stationary preservation = rms(pred) / rms(ref) on stationary pixels. Luminance, linear radiance.
Bootstrap 95 % CI over the 16 views for the slope and correlation of the updated indirect change on A_changed.

    python b4_edit.py <out dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parents[1] / "fnto_lego"))
import r1_common as C  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722])
CLS = {"Yellow": [0], "opaque": [1, 3, 5, 6, 7], "glass": [2, 4]}


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    V = np.load(C.OUT1 / "eval_K_k44P/views.npz"); B = np.load(C.B4 / "eval/views.npz")
    p0, p1 = np.load(C.OUT1 / "eval_px/heldout_T0.npz"), np.load(C.OUT1 / "eval_px/heldout_T1.npz")
    R = {(s, m): np.load(C.OUT1 / f"refs/heldout_{s}_{m}.npz") for s in ("T0", "T1") for m in ("first", "full")}
    cols = {k: [] for k in ("dir_p", "dir_r", "ind_u", "ind_s", "ind_r", "tot_p", "tot_r", "chg", "unch", "mat", "view")}
    for v in range(int(p1["nv"])):
        va0, va1 = p0[f"v{v}_valid"].astype(bool), p1[f"v{v}_valid"].astype(bool)
        l0, l1 = (p0[f"v{v}_dom"] == 0) & va0, (p1[f"v{v}_dom"] == 0) & va1
        com, i0, i1 = np.intersect1d(p0[f"v{v}_mk"][l0], p1[f"v{v}_mk"][l1], return_indices=True)
        r0 = np.cumsum(va0)[l0][i0] - 1; r1 = np.cumsum(va1)[l1][i1] - 1
        g0 = lambda arm: V[f"heldout_T0_v{v}_{arm}"][r0, :3].astype(np.float64) @ LUM  # noqa: E731
        g1 = lambda arm: V[f"heldout_T1_v{v}_{arm}"][r1, :3].astype(np.float64) @ LUM  # noqa: E731
        rr = lambda s, m, l, i: R[(s, m)][f"v{v}_rgb"][l][i].astype(np.float64) @ LUM  # noqa: E731
        f0, f1, F0, F1 = rr("T0", "first", l0, i0), rr("T1", "first", l1, i1), rr("T0", "full", l0, i0), rr("T1", "full", l1, i1)
        k0, k1, s0 = g0("K_only"), g1("K_only"), g0("S_T0")
        cols["dir_p"].append(k1 - k0); cols["dir_r"].append(f1 - f0)
        cols["ind_u"].append((g1("E_upd") - k1) - (s0 - k0)); cols["ind_s"].append((g1("D_stale") - k1) - (s0 - k0)); cols["ind_r"].append((F1 - f1) - (F0 - f0))
        cols["tot_p"].append(g1("E_upd") - s0); cols["tot_r"].append(F1 - F0)
        cols["chg"].append(B[f"heldout_T1_v{v}_reg_A_changed"].ravel()[com]); cols["unch"].append(B[f"heldout_T1_v{v}_reg_A_unchanged"].ravel()[com])
        cols["mat"].append(p1[f"v{v}_mat"][l1][i1]); cols["view"].append(np.full(len(com), v))
    d = {k: np.concatenate(x) for k, x in cols.items()}

    def m_stats(p, r, m, sig):
        p, r = p[m], r[m]; big = np.abs(r) > 3 * sig
        return {"px": int(m.sum()), "sign_agree_gt3sigma": float((np.sign(p[big]) == np.sign(r[big])).mean()) if big.any() else None,
                "px_gt3sigma": int(big.sum()), "corr": float(np.corrcoef(p, r)[0, 1]) if m.sum() > 2 else None, "slope": float((p * r).sum() / max((r * r).sum(), 1e-30)),
                "magnitude_ratio": float(np.abs(p).sum() / max(np.abs(r).sum(), 1e-30)), "net_energy_ratio": float(p.sum() / r.sum()) if abs(r.sum()) > 1e-9 else None,
                "explained_fraction": float(1 - ((p - r) ** 2).sum() / max((r * r).sum(), 1e-30)), "rms_pred": float(np.sqrt((p * p).mean())), "rms_ref": float(np.sqrt((r * r).mean()))}
    rec = {}
    for q, (pk, rk) in {"direct_intrinsic": ("dir_p", "dir_r"), "indirect_updated": ("ind_u", "ind_r"), "indirect_stale": ("ind_s", "ind_r"), "total": ("tot_p", "tot_r")}.items():
        sig = 1.4826 * np.median(np.abs(d[rk][d["unch"]] - np.median(d[rk][d["unch"]])))
        rec[q] = {"noise_sigma": float(sig)}
        for reg, m in (("A_changed", d["chg"]), ("stationary", d["unch"])):
            rec[q][reg] = m_stats(d[pk], d[rk], m, sig)
            for cn, ids in CLS.items():
                mm = m & np.isin(d["mat"], ids)
                if mm.sum() > 200:
                    rec[q][f"{reg}:{cn}"] = m_stats(d[pk], d[rk], mm, sig)
    bs = np.random.default_rng(3); sl, co = [], []
    for _ in range(400):
        pick = bs.choice(16, 16); mm = np.concatenate([np.flatnonzero(d["chg"] & (d["view"] == q)) for q in pick])
        p, r = d["ind_u"][mm], d["ind_r"][mm]; sl.append((p * r).sum() / (r * r).sum()); co.append(np.corrcoef(p, r)[0, 1])
    rec["indirect_updated_A_changed_ci95"] = {"slope": np.quantile(sl, [0.025, 0.975]).tolist(), "corr": np.quantile(co, [0.025, 0.975]).tolist()}
    (out / "b4_edit.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1)[:5000])


if __name__ == "__main__":
    main(sys.argv[1])
