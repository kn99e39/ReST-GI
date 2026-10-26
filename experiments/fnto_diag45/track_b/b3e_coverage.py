"""WL45 Track B3e (amendment 3) - which Yellow ports send the light that C gathers back onto Yellow, and were they ever
observed by K's supervision?

Supervision coverage of a Lego port = number of training-probe pixels mapped to it (stagea/prep2_train px_port,
px_dom == 0: the K_k44P training set). For held-out T0 views (default 6, 15):
  I = first-bounce port emission (as b3d_interface: L = 0, sun on), mean luminance per unit area by Yellow port class
  (lit / shadow x observed / never observed), area shares and sent-power shares;
  for the gather rays of the ports of visible Yellow pixels that hit Yellow ports: the share of the gathered
  Yellow-sourced irradiance carried by never-observed ports, and the median hit distance per class.
Also the share of all Yellow port area never observed.

    python b3e_coverage.py <out dir> [views=6,15]
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
    An = np.load(C.OUT1 / "anchors.npz"); N = len(An["obj"]); nA = int(An["n_A"])
    pr = C.OUT1 / "stagea/prep2_train"; pp = np.load(pr / "px_port.npy", mmap_mode="r"); pd = np.load(pr / "px_dom.npy", mmap_mode="r")
    cnt = np.zeros(N, np.int64)
    for s in range(0, len(pp), 1 << 24):
        a, d = np.asarray(pp[s:s + (1 << 24)]), np.asarray(pd[s:s + (1 << 24)])
        cnt[:nA] += np.bincount(a[(d == 0) & (a >= 0) & (a < nA)], minlength=nA)
    yp = (An["mat"] == 0) & (An["obj"] == 0); Ar = An["area"].astype(np.float64); obs = cnt > 0
    rec = {"views": list(views), "yellow_ports": int(yp.sum()), "yellow_area_never_observed": float(Ar[yp & ~obs].sum() / Ar[yp].sum()),
           "yellow_ports_never_observed": float((yp & ~obs).sum() / yp.sum()), "other_lego_area_never_observed": float(Ar[(An["obj"] == 0) & ~yp & ~obs].sum() / Ar[(An["obj"] == 0) & ~yp].sum())}
    KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); T = SV.Tables(KA, KB, dev)
    rel = np.load(C.OUT1 / "rel_T0.npz"); H, Tt = rel["hit"], rel["t"]; vis = np.load(C.OUT1 / "eval_vis_k44.npz"); px = np.load(C.OUT1 / "eval_px/heldout_T0.npz")
    for v in views:
        S = SV.State(T, H, vis[f"heldout_T0_v{v}_T0"].astype(np.float32), px[f"v{v}_light"])
        Eind, h = S.aggregates(); I = S.outgoing(Eind, h).cpu().numpy().astype(np.float64); Es = S.Esun.cpu().numpy(); del S, Eind, h; torch.cuda.empty_cache()
        Il = I @ LUM; r = {"classes": {}}; tot = (math.pi * Il[yp] * Ar[yp]).sum()
        for lab, m in (("lit_observed", yp & (Es > 0) & obs), ("lit_never", yp & (Es > 0) & ~obs), ("shadow_observed", yp & (Es == 0) & obs), ("shadow_never", yp & (Es == 0) & ~obs)):
            r["classes"][lab] = {"ports": int(m.sum()), "area_share": float(Ar[m].sum() / Ar[yp].sum()), "sent_power_share": float((math.pi * Il[m] * Ar[m]).sum() / tot),
                                 "mean_I_lum_area_weighted": float((Il[m] * Ar[m]).sum() / max(Ar[m].sum(), 1e-30))}
        val = px[f"v{v}_valid"].astype(bool); ypx = val & (px[f"v{v}_dom"] == 0) & (px[f"v{v}_mat"] == 0)
        rp = np.unique(px[f"v{v}_port"][ypx].astype(np.int64)); hh = H[rp]; tt = Tt[rp]; hc = np.clip(hh, 0, None)
        hy = (hh >= 0) & yp[hc]; Lr = np.where(hy, Il[hc], 0.0); Ly = Lr.sum()
        r["visible_yellow_receivers"] = {"ports": int(len(rp)), "receiver_ports_observed": float(obs[rp].mean()),
                                         "gathered_yellow_E_share_from_never_observed": float(Lr[hy & ~obs[hc]].sum() / max(Ly, 1e-30)),
                                         "gathered_yellow_E_share_from_shadow_never": float(Lr[hy & ~obs[hc] & (Es[hc] == 0)].sum() / max(Ly, 1e-30)),
                                         "ray_share_hitting_never_observed_yellow": float((hy & ~obs[hc]).sum() / max(hy.sum(), 1)),
                                         "median_hit_distance_m_observed_vs_never": [float(np.median(tt[hy & obs[hc]])) if (hy & obs[hc]).any() else None,
                                                                                     float(np.median(tt[hy & ~obs[hc]])) if (hy & ~obs[hc]).any() else None]}
        rec[f"view{v}"] = r; print(v, json.dumps(r, indent=1), flush=True)
    print(json.dumps({k: rec[k] for k in rec if not k.startswith("view")}))
    (out / "b3e_coverage.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], tuple(int(x) for x in a[1].split(",")) if len(a) > 1 else (6, 15))
