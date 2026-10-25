"""WL44 A3 - kernel fidelity against LOW-NOISE probe targets (frozen kernel; held-out Stage A configurations re-rendered at
4096 spp in fresh processes by k44_noise_render.py ... clean).

For each re-rendered held-out probe the frozen kernel (r1_train.predict, parts as k44_fit) is scored against
  orig   the 512-spp Stage A target the kernel was evaluated on in worklog 42
  clean  the 4096-spp target of the same configuration (8x samples, independent seed)
and orig vs clean measures the 512-spp target noise. If K vs clean >> K vs orig, the probe metric is noise-limited; if
they agree, the kernel error is genuine. Prep rows are matched to image pixels in order (r1_prep keeps pixels in image
order and copies the colour unchanged), so the kernel inputs are the r1 inputs.

    python k44_clean_eval.py <kernels.pt> <tag> <clean dir>    -> results/fnto_lego_k44/A3_clean_<tag>.json
"""
import sys
from pathlib import Path

import numpy as np
import torch

import k44_common as K
import r1_train as TR
from k44_fit import predict_parts

MATS = K.C.MATERIALS[:8] + ["box"]


@torch.no_grad()
def main(kpath, tag, cdir):
    torch.set_num_threads(24); dev = torch.device("cpu"); cdir = Path(cdir)
    P = TR.Ports(dev); KA, KB = TR.make_kernels(dev); s = torch.load(kpath, map_location=dev); KA.load_state_dict(s["A"]); KB.load_state_dict(s["B"])
    Ks = [KA.eval(), KB.eval()]; S = TR.Split("holdout")
    acc = {}; per = []
    for f in sorted(cdir.glob("clean_*.npz")):
        i = int(f.stem.split("_")[1]); cz = np.load(f); oz = np.load(K.C.OUT1 / f"stagea/holdout/img_{i:05d}.npz")
        d = TR.to_t(S.pixels(i), dev); d["im"] = torch.zeros_like(d["port"])
        rows_col = d["col"].numpy(); img_col = oz["col"]
        keep = np.zeros(len(img_col), bool); ptr = 0
        for k in range(len(img_col)):                                    # r1_prep drops untraced pixels, order preserved
            if ptr < len(rows_col) and np.array_equal(img_col[k], rows_col[ptr]):
                keep[k] = True; ptr += 1
        assert ptr == len(rows_col), (i, ptr, len(rows_col))
        clean = cz["col"][keep].astype(np.float64); orig = rows_col.astype(np.float64)
        L, nl, inc = TR.batch_inputs(S, [i], P, dev)
        parts = torch.zeros(len(orig), 4, 3)
        for o in (0, 1):
            m = d["dom"] == o
            if not m.any():
                continue
            tab = TR.source_tables(Ks[o], P, o, *inc[o], exact=True); px = {k: v[m] for k, v in d.items()}; idx = torch.nonzero(m)[:, 0]
            for c0 in range(0, len(idx), 8192):
                parts[idx[c0:c0 + 8192]] = torch.stack(predict_parts(Ks[o], P, tab, o, {k: v[c0:c0 + 8192] for k, v in px.items()}, L, nl), 1)
        pr = parts.sum(1).numpy().astype(np.float64)
        dom = d["dom"].numpy(); mat = np.where(dom == 1, 8, P.mat[d["port"]].numpy()); lit = d["vbits"].numpy() > 0; fam = str(S.family[i])
        D = K.C.ldr3
        se = {"K_vs_orig": ((D(pr) - D(orig)) ** 2).mean(1), "K_vs_clean": ((D(pr) - D(clean)) ** 2).mean(1), "orig_vs_clean": ((D(orig) - D(clean)) ** 2).mean(1)}
        r = {"i": i, "family": fam}
        for k, v in se.items():
            r[k] = float(10 * np.log10(len(v) / v.sum()))
        per.append(r)
        for key, m in [("all", np.ones(len(pr), bool)), (f"family:{fam}", np.ones(len(pr), bool))] + \
                [(f"mat:{MATS[q]}:{'lit' if l_ else 'shadow'}", (mat == q) & (lit == l_)) for q in range(9) for l_ in (True, False)]:
            if not m.any():
                continue
            a = acc.setdefault(key, {"n": 0, "ref_clean": 0.0, "ref_orig": 0.0, "pred": 0.0, **{k: 0.0 for k in se}})
            a["n"] += int(m.sum()); a["ref_clean"] += clean[m].mean(1).sum(); a["ref_orig"] += orig[m].mean(1).sum(); a["pred"] += pr[m].mean(1).sum()
            for k, v in se.items():
                a[k] += v[m].sum()
        print(r, flush=True)
    rec = {"kernels": kpath, "clean_spp": 4096, "orig_spp": 512, "per_image": per,
           "pooled": {k: {"px": a["n"], **{kk: float(10 * np.log10(a["n"] / max(a[kk], 1e-30))) for kk in ("K_vs_orig", "K_vs_clean", "orig_vs_clean")},
                          "pred_over_clean": a["pred"] / max(a["ref_clean"], 1e-30), "orig_over_clean": a["ref_orig"] / max(a["ref_clean"], 1e-30)}
                      for k, a in acc.items()}}
    K.save(f"A3_clean_{tag}.json", rec)


if __name__ == "__main__":
    main(*sys.argv[1:4])
