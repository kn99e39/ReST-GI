"""WL44 - noise floor from k44_noise_render outputs: display PSNR between the original and the re-seeded render, per
image, Lego / box, lit / shadowed (any light bit, from r1 prep pixel order); single-render floor = pair PSNR + 3.01 dB.

    python k44_noise_cmp.py <noise dir>     -> results/fnto_lego_k44/A_noise_floor.json
"""
import json
import sys
from pathlib import Path

import numpy as np

import k44_common as K


def main(nd):
    nd = Path(nd); rec = {"per_image": [], "pooled": {}}
    pool = {}
    cfgs = {c["i"]: c for c in json.loads((K.C.OUT1 / "stagea/holdout/configs.json").read_text())}
    bV = np.load(K.C.BOX_T0)["V"]; blo, bhi = bV.min(0), bV.max(0)
    for f in sorted(nd.glob("noise_*.npz")):
        z = np.load(f); i = int(f.stem.split("_")[1]); o = np.load(K.C.OUT1 / f"stagea/holdout/img_{i:05d}.npz")
        a, b = o["col"].astype(np.float64), z["col"].astype(np.float64)
        p = o["pos"].astype(np.float64); bo = cfgs[i]["box_offset"]
        dom = np.zeros(len(p), int)
        if bo is not None:
            dom[np.all((p >= blo + np.asarray(bo) - 1e-3) & (p <= bhi + np.asarray(bo) + 1e-3), 1)] = 1
        se = ((K.C.ldr3(a) - K.C.ldr3(b)) ** 2).mean(1)
        r = {"i": i, "family": cfgs[i]["family"], "pair_psnr": float(10 * np.log10(len(se) / se.sum()))}
        for nm, m in (("lego", dom == 0), ("box", dom == 1)):
            if m.any():
                r[f"pair_psnr_{nm}"] = float(10 * np.log10(m.sum() / max(se[m].sum(), 1e-30)))
                q = pool.setdefault(nm, [0.0, 0]); q[0] += se[m].sum(); q[1] += int(m.sum())
        q = pool.setdefault("all", [0.0, 0]); q[0] += se.sum(); q[1] += len(se)
        rec["per_image"].append(r)
    for k, (s, n) in pool.items():
        pp = float(10 * np.log10(n / s)); rec["pooled"][k] = {"pair_psnr": pp, "single_render_floor_psnr": pp + 3.0103, "px": n}
    K.save("A_noise_floor.json", rec)


if __name__ == "__main__":
    main(sys.argv[1])
