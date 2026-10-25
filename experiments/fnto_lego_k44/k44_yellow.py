"""WL44 A4 - held-out kernel-only error by colour channel and the re-weighting ceiling of the existing components
(numpy only; frozen predictions in r1 eval views.npz vs the first-event r1 references).

  per channel     Yellow's Burley SSS radius is (1.0, 0.2, 0.1) m (R, G, B) on a 0.9 x 1.6 x 0.9 m object, beyond K_band
                  (5 cm) and K_mid (30 cm); a support-range limit predicts R much worse than G and B.
  re-weighting    per material and channel, least-squares weights for (loc, glob, band+mid) against the reference on
                  half of the views, scored on the other half: how far a mis-weighting of the existing basis explains.

    python k44_yellow.py <eval dir> <tag>       -> results/fnto_lego_k44/A4_channels_<tag>.json
"""
import sys
from pathlib import Path

import numpy as np

import k44_common as K

MATS = K.C.MATERIALS[:8] + ["box"]


def main(evdir, tag):
    V = np.load(Path(evdir) / "views.npz"); s = "heldout_T0"; px = np.load(K.C.OUT1 / f"eval_px/{s}.npz")
    first = np.load(K.C.OUT1 / f"refs/{s}_first.npz"); D = K.C.ldr3
    data = {}
    for v in range(int(px["nv"])):
        val = px[f"v{v}_valid"].astype(bool); mat = np.where(px[f"v{v}_dom"] == 1, 8, px[f"v{v}_mat"])[val]
        ref = first[f"v{v}_rgb"][val].astype(np.float64); kk = V[f"{s}_v{v}_K_only"].astype(np.float64)
        for i in range(9):
            m = mat == i
            if m.any():
                d = data.setdefault(i, {"ref": [], "loc": [], "glob": [], "bm": [], "pred": [], "view": []})
                d["ref"].append(ref[m]); d["pred"].append(kk[m, :3]); d["loc"].append(kk[m, 3:6]); d["glob"].append(kk[m, 9:12]); d["bm"].append(kk[m, 12:15])
                d["view"].append(np.full(int(m.sum()), v))
    rec = {}
    for i, d in data.items():
        d = {k: np.concatenate(v) for k, v in d.items()}
        r = {"px": int(len(d["ref"]))}
        r["psnr_per_channel"] = [float(10 * np.log10(1 / max(((D(d["pred"][:, c]) - D(d["ref"][:, c])) ** 2).mean(), 1e-30))) for c in range(3)]
        r["pred_over_ref_per_channel"] = [float(d["pred"][:, c].sum() / max(d["ref"][:, c].sum(), 1e-30)) for c in range(3)]
        tr, te = d["view"] % 2 == 0, d["view"] % 2 == 1
        fit = np.zeros_like(d["ref"]); w = []
        for c in range(3):
            X = np.stack([d["loc"][:, c], d["glob"][:, c], d["bm"][:, c]], 1)
            coef, *_ = np.linalg.lstsq(X[tr], d["ref"][tr, c], rcond=None); w.append(coef.tolist()); fit[:, c] = X @ coef
        r["reweight_coef_loc_glob_bm_per_channel"] = w
        r["psnr_test_half_original"] = float(10 * np.log10(1 / ((D(d["pred"][te]) - D(d["ref"][te])) ** 2).mean()))
        r["psnr_test_half_reweighted"] = float(10 * np.log10(1 / ((D(np.clip(fit[te], 0, None)) - D(d["ref"][te])) ** 2).mean()))
        rec[MATS[i]] = r
    K.save(f"A4_channels_{tag}.json", rec)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
