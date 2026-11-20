"""WL104H POST HOC (after evidence; descriptive only, no rule): mechanism of the parameter-group couplings reported by params104h.

1. Saturation: raw logits whose sigmoid derivative is ~0 (knots, F0, roughness) -> locally dead parameter directions.
2. Per-lobe principal cosines inside one lobe: amplitude (3 channel logits of its slot) vs its F0 / roughness / anisotropy column.
3. Exact structural equivalence: scaling every attenuation knot of channel c by (1 + e) changes F exactly like scaling lam_L,c and
   lam_I*,c by (1 + e)^2 with the absorption slot compensating (softmax). Compared with exact JVPs (torch.func.jvp).

4. Per-probe cosine between base-term responses (near-duplicate lobes).

Usage: python posthoc104h_coupling.py <out_json>
       python posthoc104h_coupling.py terms <primary>/response/lanes.npz <out_json>
"""
import sys
from pathlib import Path

import numpy as np
import torch

import io104h as IO
import comp104h as K
import params104h as PP

DEV = K.H.dev


def main():
    out = Path(sys.argv[1])
    rows, raw, _, _ = IO.inputs()
    ref = IO.reference()
    res = {}
    for m, row in enumerate(rows):
        sl = slice(m * 20, (m + 1) * 20)
        wi = torch.as_tensor(ref["wi"][sl].reshape(-1, 3), device=DEV, dtype=torch.float64)
        wo = torch.as_tensor(ref["wo"][sl].reshape(-1, 3), device=DEV, dtype=torch.float64)
        w = ref["use"][sl] / ref["q"][sl]
        A = (w[..., None] * 0.5 * (ref["a"][sl] + ref["b"][sl])).sum((1, 2))
        RW = torch.as_tensor(np.repeat((np.sqrt(w) / A[:, None]).reshape(-1), 3), device=DEV)
        z0 = torch.as_tensor(raw["R0"][m], device=DEV, dtype=torch.float64)
        J = PP.jacobian(z0, wi, wo) * RW[:, None]
        p = IO.decode64(raw["R0"][m])
        zz = raw["R0"][m]
        sat = lambda x: float(torch.sigmoid(torch.tensor(x)) * (1 - torch.sigmoid(torch.tensor(x))))
        rec = {"knots_s": p["s"][0].cpu().numpy().round(6).tolist(),
               "knot_sigmoid_slope": [sat(zz[48 + i]) for i in range(15)],
               "f0": p["f0"][0].cpu().numpy().round(6).tolist(), "f0_sigmoid_slope": [sat(zz[3 * k + 2]) for k in range(7)],
               "alpha_u": p["au"][0].cpu().numpy().round(6).tolist(), "alpha_v": p["av"][0].cpu().numpy().round(6).tolist(),
               "lam": p["lam"][0].cpu().numpy().round(6).tolist(), "eta": float(p["eta"][0])}
        groups, lobes = K.raw_groups()
        per_lobe = {}
        for name, sub in lobes.items():
            if "fresnel" not in sub:
                continue
            Ua = torch.linalg.svd(J @ torch.as_tensor(sub["amplitude"], device=DEV), full_matrices=False)[0]
            d = {}
            for pg in ("fresnel", "roughness", "anisotropy"):
                v = J @ torch.as_tensor(sub[pg], device=DEV)
                nv = float(torch.linalg.norm(v))
                d[pg] = {"cos_to_own_amplitude_span": float(torch.linalg.norm(Ua.T @ v) / max(nv, 1e-300)),
                         "column_norm_relative": nv / float(torch.linalg.norm(J, dim=0).max())}
            per_lobe[name] = d
        rec["within_lobe"] = per_lobe
        # exact structural equivalence (per channel)
        f = PP.response_fn(wi, wo)
        s = p["s"][0]
        lam = p["lam"][0]
        lam9 = torch.softmax(torch.as_tensor(zz[21:48], device=DEV, dtype=torch.float64).reshape(3, 9), -1)
        eq = []
        for c in range(3):
            v_att = torch.zeros(64, device=DEV, dtype=torch.float64)
            v_att[48 + 5 * c:53 + 5 * c] = 1.0 / (1 - s[c])                     # ds_j = e s_j  (ds/dz = s (1 - s))
            # target dlam_r = 2 e lam_r for r in {0, 4..7}, 0 for outer, absorption compensates: dz_r = dlam_r / lam_r + k
            R = [0, 4, 5, 6, 7]
            target = torch.zeros(9, device=DEV, dtype=torch.float64)
            target[R] = 2 * lam9[c, R]
            target[8] = -target[R].sum()
            dz = target / lam9[c]                                                # softmax: dlam = lam (dz - <lam, dz>); <lam, dz> = 0
            v_amp = torch.zeros(64, device=DEV, dtype=torch.float64)
            v_amp[21 + 9 * c:30 + 9 * c] = dz
            _, ja = torch.func.jvp(f, (z0,), (v_att,))
            _, jb = torch.func.jvp(f, (z0,), (v_amp,))
            ja, jb = ja * RW, jb * RW
            eq.append({"channel": c, "relative_difference": float(torch.linalg.norm(ja - jb) / torch.linalg.norm(ja)),
                       "cos": float((ja * jb).sum() / (torch.linalg.norm(ja) * torch.linalg.norm(jb)))})
        rec["attenuation_amplitude_equivalence"] = eq
        res[row["tag"]] = rec
        print(row["tag"], [round(e["relative_difference"], 12) for e in eq], flush=True)
    IO.write(out, {"post_hoc": True, "commit": IO.commit(), "results": res})


def term_collinearity(lanes_path, out):
    """POST HOC: per material and probe, cosine between base-term responses (lane-weighted, RGB) -> median / min over probes."""
    import json
    with np.load(lanes_path) as z:
        lanes = {k: z[k] for k in z.files}
    tags = sorted({k.split("|")[0] for k in lanes})
    names = ("O0", "O1", "O2", "SL", "SI0", "SI1", "SI2", "SI3")
    res = {}
    for tag in tags:
        w = lanes[f"{tag}|w"].astype(np.float64)
        T = {n: lanes[f"{tag}|{n}"].astype(np.float64) for n in names}
        C = np.zeros((20, len(names), len(names)))
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                num = (w[..., None] * T[a] * T[b]).sum((1, 2))
                den = np.sqrt((w[..., None] * T[a] ** 2).sum((1, 2)) * (w[..., None] * T[b] ** 2).sum((1, 2)))
                C[:, i, j] = num / np.maximum(den, 1e-300)
        res[tag] = {"names": names, "median_over_probes": np.median(C, 0).round(4).tolist(),
                    "min_over_probes": C.min(0).round(4).tolist(), "per_probe": C.round(4).tolist()}
    IO.write(out, {"post_hoc": True, "results": res})


if __name__ == "__main__":
    if sys.argv[1] == "terms":
        term_collinearity(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main()
