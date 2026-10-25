"""WL44 C + A2 - error decomposition of the r1 composed and kernel-only predictions on the held-out views (numpy only;
frozen predictions from r1_eval views.npz, references from the r1 regime; nothing is re-solved or re-rendered).

Per held-out view, at the identity-traced alpha>0 pixels, with matched regime r1 (full = full-GI r1 reference, first =
intrinsic-all first-event r1 reference; same cameras, suns, pixels, film, no clamp):
  K-only budget   pred_K = K_only arm (loc_sun + glob + band/mid; gathered term 0) vs first; ablations that drop the
                  nonlocal parts; per material x lit / shadowed: share of the display squared error and signed bias.
  indirect part   pred_ind = S_T0 - K_only  (linear radiance)      ref_ind = full - first
                  signed error, magnitude ratio, correlation (linear and display), per material.
  oracle swaps    full-scene display error of S_T0 vs full, of (first + pred_ind) [K replaced by the reference: the
                  composition / indirect error alone] and of (pred_K + ref_ind) [indirect replaced by the reference:
                  the kernel error alone].
  edit split      held-out T1 vs T0 (same cameras / suns): direct / intrinsic change pred_K(T1) - pred_K(T0) vs
                  first(T1) - first(T0); coupling-update change (E_upd - K_only)(T1) - (S_T0 - K_only)(T0) vs
                  (full - first)(T1) - (full - first)(T0), and the stale counterpart with D_stale.
Display = ldr3 (x / (1 + x)) ^ (1 / 2.2), the PSNR space of worklogs 37-42.

    python k44_decomp.py <eval dir> <tag>       -> results/fnto_lego_k44/C_decomp_<tag>.json
"""
import sys
from pathlib import Path

import numpy as np

import k44_common as K

MATS = K.C.MATERIALS[:8] + ["box"]
LUM = np.array([0.2126, 0.7152, 0.0722])


def psnr(se, n):
    return float(10 * np.log10(n / max(se, 1e-30)))


def main(evdir, tag):
    evdir = Path(evdir); V = np.load(evdir / "views.npz")
    px = {s: np.load(K.C.OUT1 / f"eval_px/{s}.npz") for s in ("heldout_T0", "heldout_T1")}
    ref = {(s, m): np.load(K.C.OUT1 / f"refs/{s}_{m}.npz") for s in px for m in ("full", "first")}
    D = K.C.ldr3
    rec = {"eval": str(evdir), "regime": "r1 (unclamped, point-sampled) for every reference used here"}
    for s, arm_full in (("heldout_T0", "S_T0"), ("heldout_T1", "E_upd")):
        acc = {}; nv = int(px[s]["nv"])
        for v in range(nv):
            g = lambda k: px[s][f"v{v}_{k}"]  # noqa: E731
            val = g("valid").astype(bool)
            mat = np.where(g("dom")[val] == 1, 8, g("mat")[val]).astype(int); lit = g("vis")[val].astype(bool)
            first = ref[(s, "first")][f"v{v}_rgb"][val].astype(np.float64); full = ref[(s, "full")][f"v{v}_rgb"][val].astype(np.float64)
            kk = V[f"{s}_v{v}_K_only"].astype(np.float64); ff = V[f"{s}_v{v}_{arm_full}"].astype(np.float64)
            pk, pf = kk[:, :3], ff[:, :3]
            loc, glob, bm = kk[:, 3:6], kk[:, 9:12], kk[:, 12:15]
            cand = {"K": pk, "K_loc_only": loc, "K_loc+glob": loc + glob, "K_loc+band_mid": loc + bm}
            for name, p in cand.items():
                se = ((D(p) - D(first)) ** 2).mean(1)
                a = acc.setdefault(f"Konly:{name}", {"se": 0.0, "n": 0}); a["se"] += se.sum(); a["n"] += len(se)
                if name == "K":
                    for dn, dm in (("lego", mat < 8), ("box", mat == 8)):
                        a = acc.setdefault(f"Konly_dom:{dn}", {"se": 0.0, "n": 0}); a["se"] += se[dm].sum(); a["n"] += int(dm.sum())
                    op = np.isin(mat, [1, 3, 5, 6, 7]) & ~lit
                    a = acc.setdefault("Konly_opaque_shadow", {"se": 0.0, "n": 0}); a["se"] += se[op].sum(); a["n"] += int(op.sum())
                    a = acc.setdefault("opaque_shadow_mean_pred_lum", {"se": 0.0, "n": 0}); a["se"] += float((p[op] @ LUM).sum()); a["n"] += int(op.sum())
                if name == "K":
                    for mi in range(9):
                        for li, lname in ((True, "lit"), (False, "shadow")):
                            m = (mat == mi) & (lit == li)
                            b = acc.setdefault(f"Kmat:{MATS[mi]}:{lname}", {"se": 0.0, "n": 0, "bias_disp": 0.0, "ref_lin": 0.0, "pred_lin": 0.0,
                                                                           "loc": 0.0, "glob": 0.0, "bm": 0.0})
                            b["se"] += se[m].sum(); b["n"] += int(m.sum()); b["bias_disp"] += ((D(p) - D(first))[m] @ LUM).sum()
                            b["ref_lin"] += (first[m] @ LUM).sum(); b["pred_lin"] += (p[m] @ LUM).sum()
                            b["loc"] += (loc[m] @ LUM).sum(); b["glob"] += (glob[m] @ LUM).sum(); b["bm"] += (bm[m] @ LUM).sum()
            # indirect component
            pi, ri = pf - pk, full - first
            for mi in list(range(9)) + [None]:
                m = np.ones(len(pi), bool) if mi is None else (mat == mi)
                c = acc.setdefault(f"ind:{'all' if mi is None else MATS[mi]}", {"n": 0, "sp": 0.0, "sr": 0.0, "spr": 0.0, "spp": 0.0, "srr": 0.0, "sse": 0.0,
                                                                                 "sd_err": 0.0})
                a_, b_ = pi[m] @ LUM, ri[m] @ LUM
                c["n"] += int(m.sum()); c["sp"] += a_.sum(); c["sr"] += b_.sum(); c["spr"] += (a_ * b_).sum(); c["spp"] += (a_ * a_).sum()
                c["srr"] += (b_ * b_).sum(); c["sse"] += ((a_ - b_) ** 2).sum()
            # oracle swaps on the full scene
            for name, p in (("full_pred", pf), ("oracle_K(first+pred_ind)", first + pi), ("oracle_ind(pred_K+ref_ind)", pk + ri), ("first_ref_only", first)):
                se = ((D(np.clip(p, 0, None)) - D(full)) ** 2).mean(1)
                a = acc.setdefault(f"swap:{name}", {"se": 0.0, "n": 0}); a["se"] += se.sum(); a["n"] += len(se)
        out = {}
        for k, a in acc.items():
            if k == "opaque_shadow_mean_pred_lum":
                out[k] = {"value": a["se"] / max(a["n"], 1), "px": a["n"]}
            elif k.startswith(("Konly", "swap:")):
                out[k] = {"pooled_psnr": psnr(a["se"], a["n"]), "px": a["n"]}
            elif k.startswith("Kmat:"):
                out[k] = {"px": a["n"], "pooled_psnr": psnr(a["se"], a["n"]) if a["n"] else None, "share_of_K_se": None, "se": a["se"],
                          "mean_bias_disp_lum": a["bias_disp"] / max(a["n"], 1), "pred_over_ref_lin": a["pred_lin"] / max(a["ref_lin"], 1e-30),
                          "mean_ref_lin": a["ref_lin"] / max(a["n"], 1), "mean_loc": a["loc"] / max(a["n"], 1), "mean_glob": a["glob"] / max(a["n"], 1),
                          "mean_band_mid": a["bm"] / max(a["n"], 1)}
            else:
                n = a["n"]; mp, mr = a["sp"] / n, a["sr"] / n
                cov = a["spr"] / n - mp * mr; vp = a["spp"] / n - mp * mp; vr = a["srr"] / n - mr * mr
                out[k] = {"px": n, "mean_pred_lum": mp, "mean_ref_lum": mr, "pred_over_ref": mp / mr if mr else None,
                          "corr": cov / np.sqrt(max(vp * vr, 1e-30)), "rmse_lum": np.sqrt(a["sse"] / n), "rms_ref_lum": np.sqrt(a["srr"] / n),
                          "explained_fraction": 1 - a["sse"] / max(a["srr"], 1e-30)}
        tot = sum(v["se"] for k, v in out.items() if k.startswith("Kmat:"))
        for k, v in out.items():
            if k.startswith("Kmat:"):
                v["share_of_K_se"] = v.pop("se") / tot
        rec[s] = out
    # edit split (T1 vs T0 at pixels present in both views, Lego only)
    es = {}
    for v in range(int(px["heldout_T1"]["nv"])):
        d0, d1 = px["heldout_T0"], px["heldout_T1"]
        va0, va1 = d0[f"v{v}_valid"].astype(bool), d1[f"v{v}_valid"].astype(bool)
        l0, l1 = (d0[f"v{v}_dom"] == 0) & va0, (d1[f"v{v}_dom"] == 0) & va1
        common, i0, i1 = np.intersect1d(d0[f"v{v}_mk"][l0], d1[f"v{v}_mk"][l1], return_indices=True)
        r0 = np.cumsum(va0)[l0][i0] - 1; r1 = np.cumsum(va1)[l1][i1] - 1
        q0 = lambda arm: V[f"heldout_T0_v{v}_{arm}"][r0, :3].astype(np.float64) @ LUM  # noqa: E731
        q1 = lambda arm: V[f"heldout_T1_v{v}_{arm}"][r1, :3].astype(np.float64) @ LUM  # noqa: E731
        f0 = ref[("heldout_T0", "first")][f"v{v}_rgb"][l0][i0] @ LUM; f1 = ref[("heldout_T1", "first")][f"v{v}_rgb"][l1][i1] @ LUM
        F0 = ref[("heldout_T0", "full")][f"v{v}_rgb"][l0][i0] @ LUM; F1 = ref[("heldout_T1", "full")][f"v{v}_rgb"][l1][i1] @ LUM
        pairs = {"direct_intrinsic": (q1("K_only") - q0("K_only"), f1 - f0),
                 "coupling_update": ((q1("E_upd") - q1("K_only")) - (q0("S_T0") - q0("K_only")), (F1 - f1) - (F0 - f0)),
                 "coupling_stale": ((q1("D_stale") - q1("K_only")) - (q0("S_T0") - q0("K_only")), (F1 - f1) - (F0 - f0))}
        for k, (a_, b_) in pairs.items():
            e = es.setdefault(k, [[], []]); e[0].append(a_); e[1].append(b_)
    rec["edit_split_lum_linear"] = {}
    for k, (A_, B_) in es.items():
        a_, b_ = np.concatenate(A_), np.concatenate(B_)
        rec["edit_split_lum_linear"][k] = {"px": int(len(a_)), "corr": float(np.corrcoef(a_, b_)[0, 1]), "slope": float((a_ * b_).sum() / (b_ * b_).sum()),
                                           "explained_fraction": float(1 - ((a_ - b_) ** 2).sum() / (b_ * b_).sum()), "rms_ref": float(np.sqrt((b_ * b_).mean())),
                                           "rmse": float(np.sqrt(((a_ - b_) ** 2).mean()))}
    K.save(f"C_decomp_{tag}.json", rec)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
