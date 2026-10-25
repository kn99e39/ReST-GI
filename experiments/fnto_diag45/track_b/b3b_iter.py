"""WL45 Track B3b / B3c - per-iteration accounting of the frozen K_k44P / C composition, and the opaque receivers'
gathered indirect field against an independently rendered one.

B3b: for held-out T0 views (default 6, 15), the production solve (r1_solve.State with rel_T0 and the eval_vis_k44 port
sun fractions) is stepped by hand; after every step the pixels are rendered and compared with the r1 references:
display PSNR of the full prediction vs full_r1, predicted indirect (prediction - K_only) vs full_r1 - first_r1
(luminance ratio, correlation) per material class, the mean gathered irradiance per class, and the solver residual.
B3c (if passes_v<v>.npz exist): at opaque pixels (Gray / Black / Light_brown / Sand / Brown / box) the converged
gathered irradiance at the pixel's port E_FNTO vs E_ref = pi x DiffInd (Cycles diffuse-indirect pass, before colour),
and the receiver-side counterfactual rho_K E_ref / pi vs the reference diffuse-indirect radiance DiffInd x DiffCol
(rho_K = K_k44P's diffuse albedo at the pixel).
Source split (amendment): at convergence, the response to the gathered field restricted to rays hitting Yellow ports
vs all other rays, per class, as a fraction of the reference indirect (full_r1 - first_r1).

    python b3b_iter.py <out dir> [views=6,15] [max_iter=12]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parents[1] / "fnto_lego"))
import r1_common as C  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722])
CLASSES = {"Yellow": [0], "opaque": [1, 3, 5, 6, 7], "glass": [2, 4], "box": [8]}


@torch.no_grad()
def main(out, views=(6, 15), max_iter=12):
    import r1_solve as SV
    from r1_eval import load_kernels
    out = Path(out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda")
    KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); T = SV.Tables(KA, KB, dev)
    H = np.load(C.OUT1 / "rel_T0.npz")["hit"]; vis = np.load(C.OUT1 / "eval_vis_k44.npz")
    px = np.load(C.OUT1 / "eval_px/heldout_T0.npz"); first = np.load(C.OUT1 / "refs/heldout_T0_first.npz"); full = np.load(C.OUT1 / "refs/heldout_T0_full.npz")
    Vk = np.load(C.OUT1 / "eval_K_k44P/views.npz"); D = C.ldr3; rec = {"views": list(views), "B3b": {}, "B3c": {}}
    An = np.load(C.OUT1 / "anchors.npz"); hc = np.clip(H, 0, None)
    Hy = torch.as_tensor((H >= 0) & (An["mat"][hc] == 0) & (An["obj"][hc] == 0), device=dev); del hc   # rays that hit a Yellow port
    for v in views:
        val = px[f"v{v}_valid"].astype(bool); mat = np.where(px[f"v{v}_dom"][val] == 1, 8, px[f"v{v}_mat"][val])
        pxd = {k: torch.as_tensor(px[f"v{v}_{k}"][val].astype(np.float32 if k in ("pos", "nrm", "cam", "vis") else np.int64), device=dev)
               for k in ("pos", "nrm", "cam", "vis", "dom", "port")}
        f1 = first[f"v{v}_rgb"][val].astype(np.float64); fu = full[f"v{v}_rgb"][val].astype(np.float64); ko = Vk[f"heldout_T0_v{v}_K_only"][:, :3].astype(np.float64)
        ref_ind = (fu - f1) @ LUM
        S = SV.State(T, H, vis[f"heldout_T0_v{v}_T0"].astype(np.float32), px[f"v{v}_light"])
        rows = []
        for it in range(max_iter):
            dn, nn = S.step(); Eind, h = S.aggregates(); S.Eind, S.h = Eind, h; S.outgoing(Eind, h)
            pr, parts = SV.render_pixels(T, S, pxd); pr = pr.cpu().numpy().astype(np.float64)
            pind = (pr - ko) @ LUM; r = {"iter": it + 1, "rel_change": dn / max(nn, 1e-30),
                                         "full_psnr_vs_full_r1": float(10 * np.log10(1 / ((D(pr) - D(fu)) ** 2).mean()))}
            Ep = Eind.mean(1).cpu().numpy()
            for cname, ids in CLASSES.items():
                m = np.isin(mat, ids)
                if m.any():
                    r[cname] = {"ind_pred_over_ref": float(pind[m].sum() / max(ref_ind[m].sum(), 1e-30)), "ind_corr": float(np.corrcoef(pind[m], ref_ind[m])[0, 1]),
                                "mean_gathered_E_at_pixel_ports": float(Ep[pxd["port"].cpu().numpy()[m]].mean())}
            rows.append(r)
            if it > 0 and dn / max(nn, 1e-30) < SV.TOL:
                break
        rec["B3b"][f"view{v}"] = rows; print(v, json.dumps(rows[-1]), flush=True)
        # source split of the converged gathered field (WL45 amendment): the response to the last gather restricted to
        # rays that hit Yellow ports vs all other rays (sun off; linear, so the two parts sum to prediction - K_only)
        Lc = S.L.clone(); split = {}
        for nm, msk in (("yellow_src", Hy), ("other_src", ~Hy)):
            S.L = Lc * msk[..., None]
            Eind, h = S.aggregates(); h[torch.arange(T.N, device=dev), S.sunbin] -= S.Esun[:, None]; S.outgoing(Eind, h, sun_on=False)
            pr_s, _ = SV.render_pixels(T, S, {**pxd, "vis": torch.zeros_like(pxd["vis"])}); split[nm] = pr_s.cpu().numpy().astype(np.float64) @ LUM
            split[nm + "_E"] = Eind[pxd["port"]].mean(1).cpu().numpy().astype(np.float64)
        S.L = Lc; Eind, h = S.aggregates(); S.Eind, S.h = Eind, h; S.outgoing(Eind, h)
        rec["B3b"][f"view{v}_source_split"] = {}
        for cname, ids in CLASSES.items():
            m = np.isin(mat, ids)
            if m.any():
                rs = max(ref_ind[m].sum(), 1e-30)
                rec["B3b"][f"view{v}_source_split"][cname] = {
                    "pred_ind_from_yellow_src_over_ref_ind": float(split["yellow_src"][m].sum() / rs), "pred_ind_from_other_src_over_ref_ind": float(split["other_src"][m].sum() / rs),
                    "closure_split_sum_over_pred_ind": float((split["yellow_src"][m] + split["other_src"][m]).sum() / max(pind[m].sum(), 1e-30)),
                    "gathered_E_yellow_src_share": float(split["yellow_src_E"][m].sum() / max((split["yellow_src_E"][m] + split["other_src_E"][m]).sum(), 1e-30))}
        print(v, "split", json.dumps(rec["B3b"][f"view{v}_source_split"]), flush=True)
        pf = out.parent / "passes" / f"passes_v{v}.npz"
        if pf.exists():
            P = np.load(pf); op = np.isin(mat, [1, 3, 5, 6, 7, 8])
            Eref = np.pi * P["DiffInd"][val].astype(np.float64); dcol = P["DiffCol"][val].astype(np.float64)
            Ef = S.Eind[pxd["port"]].cpu().numpy().astype(np.float64)
            canon, port = pxd["pos"], pxd["port"]
            rho = np.zeros((len(mat), 3))
            for o in (0, 1):
                mm = torch.nonzero(pxd["dom"] == o)[:, 0]
                if len(mm):
                    Kk = T.Ks[o]; mt = T.mat[port[mm]] if o == 0 else torch.zeros_like(port[mm])
                    rho[mm.cpu().numpy()] = Kk.bsdf_params(Kk.feat(canon[mm], mt))[0].cpu().numpy()
            res = {}
            for cname, ids in (("opaque", [1, 3, 5, 6, 7]), ("box", [8]), ("Gray", [1]), ("Sand", [6]), ("Black", [3])):
                m = np.isin(mat, ids) & op & (Eref.mean(1) > 1e-4)
                if m.sum() < 100:
                    continue
                a, b = Ef[m].mean(1), Eref[m].mean(1)
                cf, rf = (rho[m] * Eref[m] / np.pi) @ LUM, (P["DiffInd"][val][m] * dcol[m]) @ LUM
                res[cname] = {"px": int(m.sum()), "E_FNTO_over_E_ref": float(a.sum() / b.sum()), "E_corr": float(np.corrcoef(a, b)[0, 1]),
                              "E_median_abs_rel": float(np.median(np.abs(a - b) / b)),
                              "rho_K_over_DiffCol_lum": float((rho[m] @ LUM).sum() / max((dcol[m] @ LUM).sum(), 1e-30)),
                              "counterfactual_rhoK_Eref_over_ref_diffind": float(cf.sum() / max(rf.sum(), 1e-30)), "counterfactual_corr": float(np.corrcoef(cf, rf)[0, 1])}
            rec["B3c"][f"view{v}"] = res; print(v, "B3c", json.dumps(res), flush=True)
        del S
    (out / "b3b_b3c.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], tuple(int(x) for x in a[1].split(",")) if len(a) > 1 else (6, 15), int(a[2]) if len(a) > 2 else 12)
