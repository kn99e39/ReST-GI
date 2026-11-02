"""WL70 predeclared noise-aware identity statistics (protocol sunreuse70_v1, stage2.statistics). Pure numpy; inputs are summary arrays of
independent sun-term replicas (two of the tested state A, two of the cached state B).

  R test       per element set and channel: D = mean(A) - mean(B), v = ((A1-A2)^2 + (B1-B2)^2) / 4 (unbiased for Var D), elements v > 0;
               R = sum D^2 / sum v, n_eff = (sum v)^2 / sum v^2; pass R <= 1 + 4 sqrt(3 / n_eff)   (F(n, 2n)-type null, sd ~ sqrt(3/n))
  group test   per group (material face / camera x face) and channel with share >= 2 %: |Delta_g| <= 0.001 Phi_g + 3 SE_g, SE_g^2 = sum v
  precision    every tested group: SE_g / Phi_g <= 0.003 (one tenth of the 3 % contract), else the comparison is precision-limited
  calibration  the same R statistic on a T0-vs-T0 null must satisfy |R - 1| <= 4 sqrt(3 / n_eff)
"""
import numpy as np

TOL_REL, K_SE, PREC, SHARE = 0.001, 3.0, 0.003, 0.02


def r_test(A1, A2, B1, B2):
    """Arrays (..., 3). Returns per channel dict(R, n_eff, limit, pass)."""
    out = []
    for c in range(A1.shape[-1]):
        a1, a2, b1, b2 = (np.asarray(x[..., c], float).ravel() for x in (A1, A2, B1, B2))
        D = 0.5 * (a1 + a2) - 0.5 * (b1 + b2); v = ((a1 - a2) ** 2 + (b1 - b2) ** 2) / 4; m = v > 0
        if not m.any():
            out.append(dict(R=None, n_eff=0, limit=None, passed=bool(np.all(D == 0)))); continue
        R = float((D[m] ** 2).sum() / v[m].sum()); ne = float(v[m].sum() ** 2 / (v[m] ** 2).sum()); lim = 1 + 4 * np.sqrt(3 / ne)
        out.append(dict(R=R, n_eff=ne, limit=float(lim), passed=bool(R <= lim and np.all(D[~m] == 0))))
    return out


def null_ok(rt):
    return all(x["R"] is not None and abs(x["R"] - 1) <= 4 * np.sqrt(3 / x["n_eff"]) for x in rt)


def group_test(A1, A2, B1, B2, labels, mean=False, min_n=5):
    """Element arrays (n, 3) with group labels (n,). mean=False: group sums, groups with >= 2 % of the channel total (fluxes);
    mean=True: group means over the group's elements, groups with >= min_n elements (radiances). One record per group and channel."""
    A1, A2, B1, B2 = (np.asarray(x, float) for x in (A1, A2, B1, B2)); labels = np.asarray(labels)
    v = ((A1 - A2) ** 2 + (B1 - B2) ** 2) / 4; Bm = 0.5 * (B1 + B2); Am = 0.5 * (A1 + A2); recs = []
    for g in np.unique(labels):
        sel = labels == g; n = int(sel.sum())
        if mean and n < min_n:
            continue
        W = n if mean else 1
        a = Am[sel].sum(0) / W; b = Bm[sel].sum(0) / W; se = np.sqrt(v[sel].sum(0)) / W
        for c in range(3):
            sh = float(Bm[sel, c].sum() / max(Bm[:, c].sum(), 1e-300))
            if not mean and sh < SHARE:
                continue
            d = float(a[c] - b[c]); prec = float(se[c] / max(abs(b[c]), 1e-300))
            recs.append(dict(group=g.item() if hasattr(g, "item") else g, ch=c, n=n, A=float(a[c]), B=float(b[c]), rel=d / max(abs(b[c]), 1e-300),
                             se_rel=prec, share=sh, passed=bool(abs(d) <= TOL_REL * abs(b[c]) + K_SE * se[c]), precise=bool(prec <= PREC)))
    return recs


def verdict(sets):
    """sets: {name: dict(r=[...], groups=[...])}. identity = every R and group passes; precise = every group precise."""
    ident = all(all(x["passed"] for x in s["r"]) and all(g["passed"] for g in s.get("groups", [])) for s in sets.values())
    prec = all(all(g["precise"] for g in s.get("groups", [])) for s in sets.values())
    return dict(identity=bool(ident), precise=bool(prec))
