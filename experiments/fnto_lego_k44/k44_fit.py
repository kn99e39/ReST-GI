"""WL44 A/D - training fit vs held-out generalisation of a frozen r1 kernel on Stage A probes, with per-component parts.

If the kernel reproduces the TRAINING probes no better than the held-out probes, the limit is not generalisation
(coverage) but fit: representation, data consistency or optimisation. Parts follow r1_train.predict exactly
(predict_parts below is that code with the four terms kept apart): K_loc (exact lights, pixel bits), K_glob, K_band,
K_mid. Display = ldr3; per split / domain / material / lit-or-shadowed (any light bit) / family.

    python k44_fit.py <kernels.pt> <tag> [n_images=120] [device=cpu]   -> results/fnto_lego_k44/A_fit_<tag>.json
"""
import math
import sys

import numpy as np
import torch

import k44_common as K
import r1_train as TR

MATS = K.C.MATERIALS[:8] + ["box"]


def predict_parts(Km, P, tab, o, px, L, nl):
    canon, n, wo, port, im = px["pos"], px["nrm"], px["cam"], px["port"], px["im"]
    mat = P.mat[port] if o == 0 else torch.zeros_like(port)
    h = Km.feat(canon, mat); rho, kap, alpha = Km.bsdf_params(h); p, _ = Km.p_q(h); hx = Km.proj(h)
    glob = (p * tab["gamma"][im]).sum(1)
    Jm = int(nl.max()); Li = L[im][:, :Jm]
    bits = ((px["vbits"][:, None] >> torch.arange(Jm, device=n.device)[None]) & 1).float()
    bits = bits * (torch.arange(Jm, device=n.device)[None] < nl[im][:, None]).float()
    typ = Li[..., 0]; v = Li[..., 1:4] - canon[:, None]; r2 = (v * v).sum(-1).clamp(min=1e-8)
    w = torch.where((typ == 0)[..., None], torch.nn.functional.normalize(Li[..., 1:4], dim=-1), v / r2.sqrt()[..., None])
    cos = (w * n[:, None]).sum(-1).clamp(min=0)
    E = torch.where(typ == 0, Li[..., 8] * cos, Li[..., 8] * cos / r2) * bits
    loc = Km.loc_response(n, wo, rho, kap, alpha, w, E[..., None].expand(-1, -1, 3))
    nb = P.nbr[port]; use = (nb >= 0).float(); nbc = nb.clamp(min=0)
    W = Km.band_w(hx, tab["hs"][nbc], n, P.bnrm[nbc], (P.bpos[nbc] - canon[:, None]) / K.C.R_LOCAL, wo)
    band = torch.einsum("nkbc,nkb->nc", W * (P.nbrA[port] * use)[..., None, None], tab["hB"][im[:, None], nbc])
    mc = P.mid[port]; usem = (mc >= 0).float(); mcc = mc.clamp(min=0)
    Wm = Km.mid_w(hx, tab["hc"][mcc], n, P.cnrm[mcc], (P.cpos[mcc] - canon[:, None]) / K.C.R_MID, wo)
    mid = (Wm * (P.midA[port] * usem)[..., None] * tab["Ec"][im[:, None], mcc][..., None]).sum(1)
    return loc, glob, band, mid


@torch.no_grad()
def main(kpath, tag, n_img=120, dev="cpu"):
    torch.set_num_threads(24); dev = torch.device(dev)
    P = TR.Ports(dev); KA, KB = TR.make_kernels(dev); s = torch.load(kpath, map_location=dev); KA.load_state_dict(s["A"]); KB.load_state_dict(s["B"])
    Ks = [KA.eval(), KB.eval()]
    rec = {"kernels": kpath}
    for split in ("train", "holdout"):
        S = TR.Split(split); rng = np.random.default_rng(44)
        js = np.sort(rng.choice(S.n, min(n_img, S.n), replace=False))
        acc = {}; per_img = []
        for j in js:
            d = TR.to_t(S.pixels(j), dev); d["im"] = torch.zeros_like(d["port"])
            L, nl, inc = TR.batch_inputs(S, [int(j)], P, dev)
            parts = torch.zeros(len(d["col"]), 4, 3)
            for o in (0, 1):
                m = d["dom"] == o
                if not m.any():
                    continue
                tab = TR.source_tables(Ks[o], P, o, *inc[o], exact=True)
                px = {k: v[m] for k, v in d.items()}
                idx = torch.nonzero(m)[:, 0]
                for c0 in range(0, len(idx), 8192):
                    sub = {k: v[c0:c0 + 8192] for k, v in px.items()}
                    parts[idx[c0:c0 + 8192]] = torch.stack(predict_parts(Ks[o], P, tab, o, sub, L, nl), 1).cpu()
            pr = parts.sum(1).numpy(); col = d["col"].cpu().numpy(); pt = parts.numpy()
            dom = d["dom"].cpu().numpy(); mat = np.where(dom == 1, 8, P.mat[d["port"]].cpu().numpy()); lit = d["vbits"].cpu().numpy() > 0
            se = ((K.C.ldr3(pr) - K.C.ldr3(col)) ** 2).mean(1); fam = str(S.family[j])
            lg = dom == 0
            per_img.append({"j": int(j), "family": fam, "psnr_all": float(10 * np.log10(len(se) / max(se.sum(), 1e-30))),
                            "psnr_lego": float(10 * np.log10(lg.sum() / max(se[lg].sum(), 1e-30))) if lg.any() else None,
                            "psnr_box": float(10 * np.log10((~lg).sum() / max(se[~lg].sum(), 1e-30))) if (~lg).any() else None})
            abl = {"loc_only": pt[:, 0], "no_glob": pt[:, [0, 2, 3]].sum(1), "no_band": pt[:, [0, 1, 3]].sum(1), "no_mid": pt[:, [0, 1, 2]].sum(1)}
            se_abl = {k: ((K.C.ldr3(v) - K.C.ldr3(col)) ** 2).mean(1) for k, v in abl.items()}
            keys = [("all", np.ones(len(se), bool)), (f"family:{fam}", np.ones(len(se), bool))]
            keys += [(f"mat:{MATS[i]}:{'lit' if l_ else 'shadow'}", (mat == i) & (lit == l_)) for i in range(9) for l_ in (True, False)]
            for key, m in keys:
                if not m.any():
                    continue
                a = acc.setdefault(key, {"se": 0.0, "n": 0, "ref": 0.0, "pred": 0.0, "parts": np.zeros(4), **{f"se_{k}": 0.0 for k in abl}})
                a["se"] += se[m].sum(); a["n"] += int(m.sum()); a["ref"] += col[m].mean(1).sum(); a["pred"] += pr[m].mean(1).sum()
                a["parts"] += pt[m].mean(2).sum(0)
                for k in abl:
                    a[f"se_{k}"] += se_abl[k][m].sum()
        tot = acc["all"]["se"]
        rec[split] = {k: {"px": a["n"], "psnr": float(10 * np.log10(a["n"] / max(a["se"], 1e-30))), "share_se": a["se"] / tot,
                          "pred_over_ref": a["pred"] / max(a["ref"], 1e-30), "mean_ref": a["ref"] / a["n"],
                          "mean_parts_loc_glob_band_mid": (a["parts"] / a["n"]).tolist(),
                          **{f"psnr_{k}": float(10 * np.log10(a["n"] / max(a[f"se_{k}"], 1e-30))) for k in ("loc_only", "no_glob", "no_band", "no_mid")}}
                      for k, a in acc.items()}
        rec[split]["_per_image"] = per_img
        print(split, "all", round(rec[split]["all"]["psnr"], 2), flush=True)
    K.save(f"A_fit_{tag}.json", rec)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]) if len(a) > 2 else 120, a[3] if len(a) > 3 else "cpu")
