"""WL104I POST HOC (after the frozen decision; descriptive only, no rule changed): why the frozen exact-null and tangent rules
failed. For every R0 / A / C state: the 8 smallest normalised singular values of both Jacobians, the coordinate loadings of the
candidate's smallest direction, sigmoid slopes of the shared lobe-shape logits, and tangent-space principal cosines restricted to
well-conditioned directions (sigma / sigma_max >= 1e-8 in both).

Usage: python posthoc104i_rank.py <out_json>
"""
import sys
from pathlib import Path

import numpy as np
import torch

import common104i as CM
import ident104i as ID


def to_original_t(y):
    """Torch version of GF.to_original (section kappa = (1 + max s') / 2) for chain-rule checks."""
    B = y.shape[0]
    z0 = torch.zeros_like(y[:, :1])
    out = torch.softmax(torch.cat([y[:, 21:30].reshape(B, 3, 3), z0[:, None].expand(B, 3, 1)], -1), -1)
    blk = out[..., 3]
    split = torch.softmax(torch.cat([y[:, 30:42].reshape(B, 3, 4), z0[:, None].expand(B, 3, 1)], -1), -1)
    sp = torch.sigmoid(y[:, 42:57].reshape(B, 3, 5))
    kap = 0.5 * (1.0 + sp.max(-1).values)
    lam9 = torch.cat([(blk * split[..., 4] * kap ** 2)[..., None], out[..., :3], blk[..., None] * split[..., :4] * (kap ** 2)[..., None],
                      (blk * (1 - kap ** 2))[..., None]], -1)
    s = sp / kap[..., None]
    return torch.cat([y[:, :21], torch.log(lam9).reshape(B, 27), (torch.log(s) - torch.log1p(-s)).reshape(B, 15), y[:, 57:58]], 1)

GF, IO = CM.GF, CM.IO


def main():
    out = Path(sys.argv[1])
    rows, raw, _, _ = CM.inputs()
    ref = IO.reference()
    groups = GF.raw_groups()
    names = [g for g, M in groups.items() for _ in range(M.shape[1])]
    Bm = torch.as_tensor(np.concatenate(list(groups.values()), 1), device=CM.DEV)
    res = {}
    for arm in ("R0", "A", "C"):
        for m, r in enumerate(rows):
            x, y, RW = ID.weights(ref, m)
            z = raw[arm][m]
            zt = torch.as_tensor(z, device=CM.DEV, dtype=torch.float64)
            yt = torch.as_tensor(GF.from_original(z[None])[0], device=CM.DEV, dtype=torch.float64)
            Jo = torch.func.jacfwd(CM.f_original(x, y))(zt) * RW[:, None]
            Jg = torch.func.jacfwd(CM.f_candidate(x, y))(yt) * RW[:, None]
            so, sg = torch.linalg.svdvals(Jo), torch.linalg.svdvals(Jg)
            _, s, Vh = torch.linalg.svd(Jg @ Bm, full_matrices=False)
            v = Vh[-1].cpu().numpy()
            load = {}
            for i, g in enumerate(names):
                load[g] = load.get(g, 0.0) + float(v[i] ** 2)
            Dy = torch.func.jacfwd(lambda q: CM.from_original_t(q[None])[0])(zt)            # [58, 64]
            zs = to_original_t(yt[None])[0]
            Dz = torch.func.jacfwd(lambda q: to_original_t(q[None])[0])(yt)               # [64, 58]
            Jos = torch.func.jacfwd(CM.f_original(x, y))(zs) * RW[:, None]
            chain = {"Jo_vs_Jg_Dy": float(torch.linalg.norm(Jo - Jg @ Dy) / torch.linalg.norm(Jo)),
                     "Jg_vs_Jo_section_Dz": float(torch.linalg.norm(Jg - Jos @ Dz) / torch.linalg.norm(Jg))}
            Uo, Ug = ID.colspace(Jo, 1e-8), ID.colspace(Jg, 1e-8)
            k = min(Uo.shape[1], Ug.shape[1])
            c = torch.linalg.svdvals(Uo.T @ Ug)[:k]
            slope = lambda q: float(torch.sigmoid(torch.tensor(q)) * (1 - torch.sigmoid(torch.tensor(q))))
            res[f"{arm}/{r['tag']}"] = {
                "original_smallest8": (so / so.max()).cpu().numpy()[-8:].tolist(),
                "candidate_smallest8": (sg / sg.max()).cpu().numpy()[-8:].tolist(),
                "candidate_smallest_direction_group_loading": {g: round(val, 4) for g, val in load.items() if val > 0.01},
                "min_lobe_shape_sigmoid_slope": min(slope(float(q)) for q in z[:21]),
                "rank_1e-8": [int(Uo.shape[1]), int(Ug.shape[1])],
                "tangent_min_cos_1e-8": float(c.min()), "chain_rule": chain}
            print(arm, r["tag"], chain, flush=True)
    IO.write(out, {"post_hoc": True, "commit": IO.commit(), "results": res})


if __name__ == "__main__":
    main()
