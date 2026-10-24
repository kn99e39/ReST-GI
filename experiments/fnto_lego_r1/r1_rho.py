"""WL42 Gate 4c - spectral radius of the homogeneous composed map K o C for a kernel file (same call as r1_eval.run).

    python r1_rho.py <kernels.pt> <out.json>        # writes {"T0": rho, "T1_inc": rho, "detail": {...}}
"""
import json
import sys

import numpy as np
import torch

import r1_common as C


def main(kpath, out):
    import r1_solve as SV
    from r1_eval import load_kernels
    dev = torch.device("cuda"); KA, KB = load_kernels(kpath, dev); T = SV.Tables(KA, KB, dev)
    rec = {"kernels": kpath, "detail": {}}
    for k in ("T0", "T1_inc"):
        r = SV.spectral_radius(T, np.load(C.OUT1 / f"rel_{k}.npz")["hit"]); rec["detail"][k] = r; rec[k] = r["rho_estimate"]
        print("spectral radius", k, r["rho_estimate"], flush=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
