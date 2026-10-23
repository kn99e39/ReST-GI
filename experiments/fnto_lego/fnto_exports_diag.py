"""FNTO Batch 6 reviewer arrays for the actual outcome (server; torch only). Descriptive.

    python fnto_exports_diag.py <kernels.pt> <out npz>

For held-out T0 / T1 views 6 and 15: the frozen K applied to the delta sun only (coupling switched off; the full
composition diverged on every view), the first-event and full references, per-pixel albedo of the pixel's port
(fnto_diag D1 sum, i.e. amplification under a uniform incident field) and the ports whose relation row changed T0 -> T1.
For three Stage A holdout probes: K's first-event prediction vs the probe render.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

import fnto_common as C
import fnto_eval as EV
import fnto_solve as SV
import fnto_train as TR


@torch.no_grad()
def main(kp, out):
    dev = torch.device("cuda")
    KA, KB = EV.load_kernels(kp, dev); T = SV.Tables(KA, KB, dev)
    alb = np.load(C.OUT / "diag/port_albedo_components.npy").astype(np.float32).sum(1)
    r0 = np.load(C.OUT / "rel_T0.npz")["hit"]; r1 = np.load(C.OUT / "rel_T1_inc.npz")["hit"]; chg = (r0 != r1).any(1)
    pv = np.load(C.OUT / "eval_vis.npz"); arr = {}
    for set_name, st, rel in (("heldout_T0", "T0", r0), ("heldout_T1", "T1", r1)):
        D = np.load(C.OUT / f"eval_px/{set_name}.npz"); first = np.load(C.OUT / f"firstevent/{set_name}_first.npz")
        for v in (6, 15):
            Vp = np.unpackbits(pv[f"{set_name}_v{v}_{st}"])[:T.N].astype(bool)
            S0 = SV.State(T, rel, Vp, D[f"v{v}_light"]); E0, h0 = S0.incident_aggregates(); S0.h, S0.Eind = h0, E0; _, S0.gamma = S0.outgoing_iso(E0, h0)
            px = {k: torch.as_tensor(D[f"v{v}_{k}"].astype(np.float32 if k in ("pos", "nrm", "cam", "vis", "w4") else np.int64), device=dev)
                  for k in ("pos", "nrm", "cam", "vis", "dom", "port", "port4", "w4")}
            p0, parts = SV.render_pixels(T, S0, px)
            port = D[f"v{v}_port"]
            arr.update({f"{set_name}_v{v}_mk": D[f"v{v}_mk"], f"{set_name}_v{v}_konly": p0.cpu().numpy().astype(np.float16),
                        f"{set_name}_v{v}_parts": parts.cpu().numpy().astype(np.float16), f"{set_name}_v{v}_first": first[f"v{v}_rgb"].astype(np.float16),
                        f"{set_name}_v{v}_full": D[f"v{v}_tgt"].astype(np.float16), f"{set_name}_v{v}_albedo": alb[port].astype(np.float16),
                        f"{set_name}_v{v}_relchg": chg[port].astype(np.uint8)})
            del S0
    Ks = [KA, KB]; P = TR.Ports(dev); H = TR.Split("holdout")
    _, kept = TR.evaluate(Ks, P, H, dev, keep={3, 17, 42})
    for j, (pred, col, pix) in kept.items():
        arr[f"stagea_{j}_pred"] = pred.astype(np.float16); arr[f"stagea_{j}_ref"] = col.astype(np.float16); arr[f"stagea_{j}_pix"] = pix
    np.savez_compressed(out, **arr); print("saved", len(arr))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
