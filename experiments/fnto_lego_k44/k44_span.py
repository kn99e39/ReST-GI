"""WL44 D - per-probe excitation coverage vs kernel error (numpy only).

For every held-out Stage A probe: share of its direct incident field (r1 coarse 8 cm basis, the Gate 2c measure) outside
the full span of the 1,000 training fields; joined with the frozen kernel's per-image probe PSNR (k44_fit A_fit_<tag>,
Lego pixels). Spearman / Pearson correlation of out-of-span energy with PSNR, overall and per family.

    python k44_span.py <tag>      -> results/fnto_lego_k44/D_span_<tag>.json
"""
import json
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

import k44_common as K
import r1_relations as RR
import r1_gate2_cov as G


def main(tag):
    A = RR.load()
    basis = {"r1_coarse": A["f2c"].astype(np.int64)}; dims = {"r1_coarse": len(A["cobj"])}
    tr = G.r1_fields(A, "train", basis, dims)["r1_coarse"]; ho = G.r1_fields(A, "holdout", basis, dims)["r1_coarse"]
    X = tr / np.maximum(np.linalg.norm(tr, axis=1, keepdims=True), 1e-12)
    _, _, Vt = np.linalg.svd(X.astype(np.float64), full_matrices=False)
    v = ho / np.maximum(np.linalg.norm(ho, axis=1, keepdims=True), 1e-12)
    out = ((v - (v @ Vt.T) @ Vt) ** 2).sum(1)
    fit = json.loads((K.OUT / f"A_fit_{tag}.json").read_text())["holdout"]["_per_image"]
    fam = np.load(K.C.OUT1 / "stagea/prep_holdout/family.npy")
    rows = [(r["j"], out[r["j"]], r["psnr_lego"], str(fam[r["j"]])) for r in fit if r["psnr_lego"] is not None]
    o = np.array([x[1] for x in rows]); p = np.array([x[2] for x in rows])
    rec = {"tag": tag, "n": len(rows), "out_of_span_q10_q50_q90": np.quantile(o, [0.1, 0.5, 0.9]).tolist(),
           "spearman_outspan_vs_psnr_lego": list(map(float, spearmanr(o, p))), "pearson": list(map(float, pearsonr(o, p))),
           "per_family": {}, "per_image": [{"j": int(a), "out_of_span": float(b), "psnr_lego": float(c), "family": d} for a, b, c, d in rows]}
    for f in sorted(set(x[3] for x in rows)):
        m = np.array([x[3] == f for x in rows])
        if m.sum() > 4:
            rec["per_family"][f] = {"n": int(m.sum()), "out_of_span_median": float(np.median(o[m])), "psnr_lego_mean": float(p[m].mean()),
                                    "spearman": list(map(float, spearmanr(o[m], p[m])))}
    lo, hi = o <= np.median(o), o > np.median(o)
    rec["psnr_lego_low_vs_high_outspan_half"] = [float(p[lo].mean()), float(p[hi].mean())]
    K.save(f"D_span_{tag}.json", rec)


if __name__ == "__main__":
    main(sys.argv[1])
