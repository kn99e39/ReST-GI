"""WL45 Track B3d (amendment 2) - K -> C interface: is the outgoing radiance a port SENDS into C (r1_solve.outgoing,
port-level tables, isotropic) consistent with the radiance K_k44P itself predicts at that surface (render_pixels path,
the K-only evaluation) and with the first-event reference?

For held-out T0 views (default 6, 15): State(T, rel_T0, eval_vis_k44 fractions, view sun), L = 0, so Eind = 0 and
I = outgoing(0, h_sun) is the first-bounce emission of every port (sun-driven: rho / pi E_sun + glob + band + mid;
the per-ray specular term of the hit port is reported separately as its mean over rays hitting the class).
Compared at pixels, by class (Yellow lit / shadow, opaque, box):
  I[port(pixel)] vs K_only(pixel) (frozen eval) vs first_r1(pixel), luminance sums, and the components of I.
The per-pixel K_only includes the pixel's own GGX lobes toward the camera; I is the cosine-weighted isotropic value.

    python b3d_interface.py <out dir> [views=6,15]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parents[1] / "fnto_lego"))
import r1_common as C  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722])


@torch.no_grad()
def main(out, views=(6, 15)):
    import r1_solve as SV
    from r1_eval import load_kernels
    out = Path(out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda")
    KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); T = SV.Tables(KA, KB, dev)
    H = np.load(C.OUT1 / "rel_T0.npz")["hit"]; vis = np.load(C.OUT1 / "eval_vis_k44.npz"); An = np.load(C.OUT1 / "anchors.npz")
    px = np.load(C.OUT1 / "eval_px/heldout_T0.npz"); first = np.load(C.OUT1 / "refs/heldout_T0_first.npz"); Vk = np.load(C.OUT1 / "eval_K_k44P/views.npz")
    rec = {"views": list(views)}
    for v in views:
        S = SV.State(T, H, vis[f"heldout_T0_v{v}_T0"].astype(np.float32), px[f"v{v}_light"])
        Eind, h = S.aggregates()                                                    # L = 0 -> Eind = 0, h = sun bin only
        I = S.outgoing(Eind, h).cpu().numpy().astype(np.float64); del Eind, h; torch.cuda.empty_cache()
        Iloc = (T.rho / math.pi * S.Esun[:, None]).cpu().numpy().astype(np.float64)
        val = px[f"v{v}_valid"].astype(bool); dom = px[f"v{v}_dom"][val]; mat = np.where(dom == 1, 8, px[f"v{v}_mat"][val]); lit = px[f"v{v}_vis"][val] > 0.5
        port = px[f"v{v}_port"][val].astype(np.int64); ko = Vk[f"heldout_T0_v{v}_K_only"][:, :3].astype(np.float64); fr = first[f"v{v}_rgb"][val].astype(np.float64)
        r = {}
        for cname, m in (("Yellow_lit", (mat == 0) & lit), ("Yellow_shadow", (mat == 0) & ~lit), ("Yellow", mat == 0),
                         ("opaque", np.isin(mat, [1, 3, 5, 6, 7])), ("box", mat == 8)):
            if m.sum() < 100:
                continue
            ip, kp, rp = I[port[m]] @ LUM, ko[m] @ LUM, fr[m] @ LUM
            r[cname] = {"px": int(m.sum()), "I_port_over_K_only_pixel": float(ip.sum() / kp.sum()), "I_port_over_first_ref": float(ip.sum() / rp.sum()),
                        "K_only_pixel_over_first_ref": float(kp.sum() / rp.sum()), "corr_I_port_vs_ref": float(np.corrcoef(ip, rp)[0, 1]),
                        "I_local_share": float((Iloc[port[m]] @ LUM).sum() / ip.sum()),
                        "I_over_ref_RGB": (I[port[m]].sum(0) / fr[m].sum(0)).tolist(), "K_only_over_ref_RGB": (ko[m].sum(0) / fr[m].sum(0)).tolist()}
        # all Yellow ports (not only visible ones): port emission vs its own sun irradiance (effective albedo of the sent light)
        yp = (An["mat"] == 0) & (An["obj"] == 0); Es = S.Esun.cpu().numpy(); Ar = An["area"].astype(np.float64); inside = (H == -2).mean(1)
        pw = lambda m: (math.pi * I[m] * Ar[m, None]).sum(0)                       # sent power (W / m^2 x m^2 per channel)  # noqa: E731
        pin = max((Es[yp] * Ar[yp]).sum(), 1e-30)
        r["all_yellow_ports"] = {"n": int(yp.sum()), "area_weighted_sent_power_over_sun_power_RGB": (pw(yp) / pin).tolist(),
                                 "lit_port_area_fraction": float(Ar[yp & (Es > 0)].sum() / Ar[yp].sum()), "sent_lum_share": {}}
        for lab, m in (("lit", yp & (Es > 0)), ("shadow_inside0", yp & (Es == 0) & (inside == 0)), ("shadow_inside_0_to_0.5", yp & (Es == 0) & (inside > 0) & (inside <= 0.5)),
                       ("shadow_inside_gt0.5", yp & (Es == 0) & (inside > 0.5))):
            r["all_yellow_ports"]["sent_lum_share"][lab] = {"area_share": float(Ar[m].sum() / Ar[yp].sum()), "sent_power_share": float(pw(m) @ LUM / (pw(yp) @ LUM)),
                                                             "sent_power_over_sun_power_RGB": (pw(m) / pin).tolist()}
        hitY = torch.as_tensor(yp[np.clip(H, 0, None)] & (H >= 0), device=dev); lumt = torch.as_tensor(LUM, device=dev).float()
        r["mean_spec_term_on_rays_hitting_Yellow_lum"] = float((S.spec[hitY].float() @ lumt).mean())
        r["mean_I_on_rays_hitting_Yellow_lum"] = float(torch.as_tensor(I @ LUM, device=dev).float()[S.Hc[hitY]].mean()); del hitY
        rec[f"view{v}"] = r; print(v, json.dumps(r, indent=1), flush=True)
        del S; torch.cuda.empty_cache()
    (out / "b3d_interface.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], tuple(int(x) for x in a[1].split(",")) if len(a) > 1 else (6, 15))
