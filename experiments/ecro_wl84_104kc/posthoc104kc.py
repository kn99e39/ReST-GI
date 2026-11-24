"""WL104K-C POST HOC, DESCRIPTIVE ONLY (written after all labels and the decision were fixed; changes no rule, state or label).

Which preservation constraint excludes the exactly representable synthetic truth? For CR_S / CR_SD: the truth parameters' training
terms on the arm's own noisy targets vs its per-probe bounds (bounds.npz), per constraint kind (el1, ealb, e_shoulder, e_tail),
and the same terms for the warm start (Stage A S_O0r / S_Dr) to show whether the warm start fits the noisy targets better than
the truth does.

  python posthoc104kc.py <root>          writes <root>/posthoc/truth_constraint_attribution.json
  python posthoc104kc.py <root> scenes   writes <root>/posthoc/scene_signed_cancellation.json (CPU; scene maps)
"""
import sys
from pathlib import Path

import numpy as np
import torch

import common104kc as C

KB, CM = C.KB, C.CM
import fit104kc as FC  # noqa: E402

N = C.N_MAT
ARMS = {"CR_S": ("fits/CR_S", "stageA/fits/S_O0r/states.npz", "original"),
        "CR_SD": ("fits/CR_SD", "stageA/fits/S_Dr/states.npz", "coverage")}


def terms(T, x):
    with torch.no_grad():
        t = T.probe_terms(torch.as_tensor(x, device=C.DEV, dtype=torch.float32))
    return {k: t[k].cpu().numpy() for k in list(FC.CON) + ["e_core"]}


def scene_cancellation(root):
    """POST HOC: signed decomposition of the start-mean luminance error on the objects mask (raw, NOT noise-debiased):
    e1 error vs the remainder (TOTAL - e1 = fixed e2 + fixed R0 HIGH) per scene, for O0 and CR."""
    rep = {}
    for d in sorted((root / "scenes").glob("*/maps.npz")):
        with np.load(d) as z:
            m = z["objects_mask"]
            r1, rt = z["reference_e1_lum"][m], z["reference_total_lum"][m]
            row = {}
            for a in ("O0", "CR", "O0C", "E2_B"):
                if f"{a}_e1_lum" not in z.files:
                    continue
                d1 = z[f"{a}_e1_lum"][m] - r1
                dt = z[f"{a}_total_lum"][m] - rt
                dr = dt - d1
                row[a] = {"mean_signed_e1": float(d1.mean()), "mean_signed_remainder": float(dr.mean()),
                          "raw_rms_e1": float(np.sqrt((d1 ** 2).mean())), "raw_rms_remainder": float(np.sqrt((dr ** 2).mean())),
                          "raw_rms_total": float(np.sqrt((dt ** 2).mean())), "corr_e1_remainder": float(np.corrcoef(d1, dr)[0, 1])}
        rep[d.parent.name] = row
    return rep


def main():
    root = Path(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "scenes":
        C.write(root / "posthoc/scene_signed_cancellation.json",
                {"note": "POST HOC descriptive; raw (not noise-debiased) luminance on the objects mask; start-mean images",
                 "commit": C.commit(), "scenes": scene_cancellation(root)})
        return
    rows, raw, _, _, _, _ = CM.inputs()
    out = {"note": "POST HOC descriptive attribution; not part of any predeclared rule or label", "commit": C.commit()}
    for arm, (fd, warm, kind) in ARMS.items():
        ds = KB.training_set(kind, N, C.KB_ROOT / "data")
        truth, target = FC.synthetic_targets(ds, raw)
        T = FC.Terms(ds, N, 4, target)
        xt = np.tile(np.concatenate([truth["y"], truth["g"].reshape(N, -1)], 1), (4, 1))
        xw = C.flat_x(C.load_states(root / warm, 4))
        with np.load(root / fd / "bounds.npz") as z:
            b = z["bounds"]                                    # [U, K, P]
        tt, tw = terms(T, xt), terms(T, xw)
        rep = {}
        for i, k in enumerate(FC.CON):
            viol = (tt[k] / b[:, i] - 1).max(-1)               # [U]
            rep[k] = {"units_truth_violates": int((viol > 0).sum()), "truth_violation_median": float(np.median(viol)),
                      "truth_over_warm_median_probe_ratio": float(np.median(tt[k] / np.maximum(tw[k], 1e-12))),
                      "truth_mean": float(tt[k].mean()), "warm_mean": float(tw[k].mean())}
        rep["e_core"] = {"truth_over_warm_median_probe_ratio": float(np.median(tt["e_core"] / np.maximum(tw["e_core"], 1e-12))),
                         "truth_mean": float(tt["e_core"].mean()), "warm_mean": float(tw["e_core"].mean())}
        out[arm] = rep
        print(arm, {k: (v.get("units_truth_violates"), round(v["truth_over_warm_median_probe_ratio"], 3)) for k, v in rep.items()},
              flush=True)
    C.write(root / "posthoc/truth_constraint_attribution.json", out)


if __name__ == "__main__":
    main()
