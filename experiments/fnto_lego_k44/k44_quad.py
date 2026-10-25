"""WL44 B - source quadrature isolation with the FROZEN kernel on held-out Stage A probes (Lego domain receivers).

For a fixed random subset of held-out probe pixels (Lego domain) the kernel response is evaluated with:
  rep          the production estimator (r1_train.predict: K/2 nearest + orientation-stratified area picks, K 96 / 128)
  rep_bf16     the same under bf16 autocast of the band / mid MLPs (the GPU training / eval numerics; CPU runs fp32)
  dense_band   band term = sum over ALL 1.5 cm aggregates of the domain within R_LOCAL of the receiver port, own area
  dense_mid    mid term  = sum over ALL 8 cm coarse ports within R_MID, own area
  dense_both   both dense (no representative sampling at all)
  fine_band    band term = sum over all FINE ports within R_LOCAL (own 19-bin incident, own normal / material descriptor,
               own area) instead of aggregates; mid dense — isolates the aggregation resolution of the band sources
Everything else (K_loc with exact lights, K_glob exact, kernels, incident fields) is identical. Display PSNR vs the
probe target on the subset, overall / per material x lit-shadow; mean |change| of the band and mid terms.
The dense evaluations are diagnostics, not a proposed runtime (cost reported).

    python k44_quad.py <kernels.pt> <tag> [n_images=40] [px_per_image=400] [fine_px_per_image=80]
"""
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

import k44_common as K
import r1_train as TR
from k44_fit import predict_parts

MATS = K.C.MATERIALS[:8]


def pair_sum(fn, n_out, pix, src, args, chunk=200000):
    out = torch.zeros(n_out, 3)
    for c in range(0, len(pix), chunk):
        out.index_add_(0, pix[c:c + chunk], fn(pix[c:c + chunk], src[c:c + chunk], *args))
    return out


@torch.no_grad()
def main(kpath, tag, n_img=40, n_px=400, n_fine=80):
    torch.set_num_threads(24); dev = torch.device("cpu")
    P = TR.Ports(dev); KA, KB = TR.make_kernels(dev); s = torch.load(kpath, map_location=dev); KA.load_state_dict(s["A"]); KB.load_state_dict(s["B"])
    Km = KA.eval(); nA = P.nA
    bsel = torch.nonzero(P.bobj == 0)[:, 0].numpy(); csel = torch.nonzero(P.cobj == 0)[:, 0].numpy()
    btree, ctree, ftree = cKDTree(P.bpos[bsel].numpy()), cKDTree(P.cpos[csel].numpy()), cKDTree(P.pos[:nA].numpy())
    S = TR.Split("holdout"); rng = np.random.default_rng(4401); js = np.sort(rng.choice(S.n, min(n_img, S.n), replace=False))
    bm_lego = torch.where(P.bobj == 0, P.bmat, 0); cm_lego = torch.where(P.cobj == 0, P.cmat, 0)
    acc = {}; cost = {}
    for j in js:
        d = TR.to_t(S.pixels(int(j)), dev); d["im"] = torch.zeros_like(d["port"])
        lg = torch.nonzero(d["dom"] == 0)[:, 0].numpy()
        if len(lg) < n_px:
            continue
        pick = np.sort(rng.choice(lg, n_px, replace=False)); fine_pick = np.zeros(n_px, bool); fine_pick[rng.choice(n_px, n_fine, replace=False)] = True
        px = {k: v[torch.as_tensor(pick)] for k, v in d.items()}
        L, nl, inc = TR.batch_inputs(S, [int(j)], P, dev)
        tab = TR.source_tables(Km, P, 0, *inc[0], exact=True)
        loc, glob, band, mid = predict_parts(Km, P, tab, 0, px, L, nl)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            _, _, band16, mid16 = predict_parts(Km, P, tab, 0, px, L, nl)
        band16, mid16 = band16.float(), mid16.float()
        canon, n, wo, port = px["pos"], px["nrm"], px["cam"], px["port"]
        h = Km.feat(canon, P.mat[port]); hx = Km.proj(h)
        rp = P.pos[port].numpy()
        # dense band over all aggregates within R
        t0 = time.time()
        lists = btree.query_ball_point(rp, K.C.R_LOCAL)
        pix = torch.as_tensor(np.repeat(np.arange(n_px), [len(x) for x in lists])); src = torch.as_tensor(bsel[np.concatenate(lists).astype(np.int64)])
        def fb(pi, si):
            W = Km.band_w(hx[pi], tab["hs"][si][:, None], n[pi], P.bnrm[si][:, None], (P.bpos[si][:, None] - canon[pi][:, None]) / K.C.R_LOCAL, wo[pi])[:, 0]
            return torch.einsum("pbc,pb->pc", W * P.barea[si][:, None, None], tab["hB"][0, si])
        band_d = pair_sum(fb, n_px, pix, src, ()); cost.setdefault("dense_band_pairs", []).append(len(pix)); cost.setdefault("dense_band_s", []).append(time.time() - t0)
        # dense mid over all coarse ports within R
        t0 = time.time()
        lists = ctree.query_ball_point(rp, K.C.R_MID)
        pix = torch.as_tensor(np.repeat(np.arange(n_px), [len(x) for x in lists])); src = torch.as_tensor(csel[np.concatenate(lists).astype(np.int64)])
        def fm(pi, si):
            Wm = Km.mid_w(hx[pi], tab["hc"][si][:, None], n[pi], P.cnrm[si][:, None], (P.cpos[si][:, None] - canon[pi][:, None]) / K.C.R_MID, wo[pi])[:, 0]
            return Wm * (P.carea[si] * tab["Ec"][0, si])[:, None]
        mid_d = pair_sum(fm, n_px, pix, src, ()); cost.setdefault("dense_mid_pairs", []).append(len(pix)); cost.setdefault("dense_mid_s", []).append(time.time() - t0)
        # fine-port band (subset of the subset)
        t0 = time.time()
        fi = np.flatnonzero(fine_pick); lists = ftree.query_ball_point(rp[fi], K.C.R_LOCAL)
        pix = torch.as_tensor(fi[np.repeat(np.arange(len(fi)), [len(x) for x in lists])]); src = torch.as_tensor(np.concatenate(lists).astype(np.int64))
        Hp = inc[0][1][0]                                                     # [nA, 19] per-port binned incident (white lights)
        hs_fine = Km.src_desc(P.mat[:nA])
        def ff(pi, si):
            W = Km.band_w(hx[pi], hs_fine[si][:, None], n[pi], P.nrm[si][:, None], (P.pos[si][:, None] - canon[pi][:, None]) / K.C.R_LOCAL, wo[pi])[:, 0]
            return torch.einsum("pbc,pb->pc", W * P.area[si][:, None, None], Hp[si])
        band_f = pair_sum(ff, n_px, pix, src, ()); cost.setdefault("fine_band_pairs", []).append(len(pix)); cost.setdefault("fine_band_s", []).append(time.time() - t0)
        col = px["col"].numpy(); mat = P.mat[port].numpy(); lit = px["vbits"].numpy() > 0
        base = loc + glob
        variants = {"rep": base + band + mid, "rep_bf16": base + band16 + mid16, "dense_band": base + band_d + mid, "dense_mid": base + band + mid_d,
                    "dense_both": base + band_d + mid_d}
        variants_f = {"fine_rep": base + band + mid, "fine_dense_both": base + band_d + mid_d, "fine_band": base + band_f + mid_d}
        for vname, vv in list(variants.items()) + list(variants_f.items()):
            msk = fine_pick if vname.startswith("fine") else np.ones(n_px, bool)
            se = ((K.C.ldr3(vv.numpy()) - K.C.ldr3(col)) ** 2).mean(1)
            for key, m in [("all", msk)] + [(f"{MATS[i]}:{'lit' if l_ else 'shadow'}", msk & (mat == i) & (lit == l_)) for i in range(8) for l_ in (True, False)]:
                if m.any():
                    a = acc.setdefault(vname, {}).setdefault(key, [0.0, 0]); a[0] += se[m].sum(); a[1] += int(m.sum())
        for nm, a_, b_ in (("band_dense_vs_rep", band_d, band), ("mid_dense_vs_rep", mid_d, mid), ("band_fine_vs_dense", band_f[fine_pick], band_d[fine_pick]),
                           ("band_bf16_vs_fp32", band16, band), ("mid_bf16_vs_fp32", mid16, mid)):
            c = acc.setdefault("_terms", {}).setdefault(nm, [0.0, 0.0, 0.0])
            c[0] += float((a_ - b_).abs().sum()); c[1] += float(b_.abs().sum()); c[2] += float((a_ - b_).sum())
        print(int(j), {k: round(float(10 * np.log10(v["all"][1] / max(v["all"][0], 1e-30))), 2) for k, v in acc.items() if k != "_terms"}, flush=True)
    rec = {"kernels": kpath, "images": [int(x) for x in js], "px_per_image": n_px, "fine_px_per_image": n_fine,
           "psnr": {v: {k: {"psnr": float(10 * np.log10(a[1] / max(a[0], 1e-30))), "px": a[1]} for k, a in d_.items()} for v, d_ in acc.items() if v != "_terms"},
           "term_change": {k: {"mean_abs_rel_change": c[0] / max(c[1], 1e-30), "signed_rel_change": c[2] / max(c[1], 1e-30)} for k, c in acc["_terms"].items()},
           "cost": {k: float(np.mean(v)) for k, v in cost.items()}}
    K.save(f"B_quad_{tag}.json", rec)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *(int(x) for x in a[2:]))
