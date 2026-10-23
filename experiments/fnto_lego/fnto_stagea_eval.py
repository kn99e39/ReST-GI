"""FNTO Stage A attribution (descriptive): K alone on held-out probes, and the superposition probes (8dna26 .venv-cu128).

    python fnto_stagea_eval.py <kernels.pt> <out dir>

holdout   : first-intrinsic-event fidelity of the frozen K under unseen spatial incident patterns (pooled, per domain,
            per port material, lit / shadowed, shadow-side yellow).
superpose : for each triple (same camera, masks, box; sun 1 alone, sun 2 alone, both suns):
            renderer linearity  Cycles(both) vs Cycles(1) + Cycles(2)   (includes the direct-light clamp of the renderer)
            operator linearity  FNTO(i1 + i2) vs FNTO(i1) + FNTO(i2)
            prediction          FNTO(i1 + i2) vs Cycles(both)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

import fnto_common as C
import fnto_eval as EV
import fnto_train as TR


def main(kp, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    KA, KB = EV.load_kernels(kp, dev); Ks = [KA, KB]
    P = TR.Ports(dev)
    rec = {"holdout": TR.evaluate(Ks, P, TR.Split("holdout"), dev)}
    S = TR.Split("superpose")
    tabs = [TR.port_tables(Ks[o], P, o) for o in (0, 1)]
    rows = []
    byid = {int(i): j for j, i in enumerate(S.ids)}
    with torch.no_grad():
        for t in range(len(S.ids) // 3):
            j0, j1, j2 = byid[3 * t], byid[3 * t + 1], byid[3 * t + 2]
            d0, d1, d2 = (S.pixels(j) for j in (j0, j1, j2))
            common = np.intersect1d(np.intersect1d(d0["pix"], d1["pix"]), d2["pix"])
            sel = [np.searchsorted(d["pix"], common) for d in (d0, d1, d2)]
            c0, c1, c2 = (d["col"][s_] for d, s_ in zip((d0, d1, d2), sel))
            ren_lin = float(np.abs(c2 - (c0 + c1)).sum() / np.abs(c2).sum())
            px = TR.to_t({k: v[sel[2]] for k, v in d2.items()}, dev); px["im"] = torch.zeros_like(px["port"])
            v0 = torch.as_tensor(d0["vis"][sel[0]].astype(np.float32), device=dev); v1 = torch.as_tensor(d1["vis"][sel[1]].astype(np.float32), device=dev)
            pred_sum, pred_1, pred_2 = torch.zeros_like(px["col"]), torch.zeros_like(px["col"]), torch.zeros_like(px["col"])
            for o in (0, 1):
                m = px["dom"] == o
                if not m.any():
                    continue
                q = {k: v[m] for k, v in px.items()}
                f1 = (*TR.image_sun(S, [j2], P, o, dev, 1), v0[m]); f2 = (*TR.image_sun(S, [j2], P, o, dev, 2), v1[m])
                pred_sum[m] = TR.predict(Ks[o], P, tabs[o], o, q, [f1, f2])
                pred_1[m] = TR.predict(Ks[o], P, tabs[o], o, q, [f1]); pred_2[m] = TR.predict(Ks[o], P, tabs[o], o, q, [f2])
            ps, p1, p2 = (x.cpu().numpy() for x in (pred_sum, pred_1, pred_2))
            op_lin = float(np.abs(ps - p1 - p2).max() / max(np.abs(ps).max(), 1e-12))
            pred_err = float(np.abs(ps - c2).sum() / np.abs(c2).sum())
            mse = float(((C.ldr3(ps) - C.ldr3(c2)) ** 2).mean())
            rows.append({"triple": t, "px": int(len(common)), "renderer_superposition_rel_L1": ren_lin, "operator_superposition_max_rel": op_lin,
                         "prediction_rel_L1_vs_two_sun_render": pred_err, "prediction_display_psnr": float(10 * np.log10(1 / mse))})
            print(json.dumps(rows[-1]), flush=True)
    rec["superpose"] = {"triples": rows, "renderer_superposition_rel_L1_mean": float(np.mean([r["renderer_superposition_rel_L1"] for r in rows])),
                        "operator_superposition_max": float(np.max([r["operator_superposition_max_rel"] for r in rows])),
                        "prediction_display_psnr_mean": float(np.mean([r["prediction_display_psnr"] for r in rows]))}
    (out / "stagea_eval.json").write_text(json.dumps(rec, indent=1)); print(json.dumps({k: v for k, v in rec["superpose"].items() if k != "triples"}))
    print(json.dumps(rec["holdout"]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
