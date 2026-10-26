"""WL45 Track B3g (amendment 5) - from WHICH ports' gathered irradiance does the receiver's nonlocal response build the
Yellow indirect over-prediction?

Production solve (as b3b_iter: rel_T0, eval_vis_k44, view sun) run to convergence on held-out T0 views 6 / 15; then the
converged gathered field (Eind, h without the sun bin) is split by the port that holds it:
  'visible'       ports of any valid pixel of this view (all materials);
  'observed'      other ports with >= 1 training-probe pixel (prep2_train), not visible in this view;
  'never'         ports never observed in training (and not visible in this view);
and for each class the response at this view's pixels is rendered with the sun off and the receiver-local term
removed (pixel port L = 0), i.e. the nonlocal (glob / band / mid) response to that class's Eind. Linear, so the classes
plus the receiver-local term sum to the indirect prediction (closure reported). Per material class: share of the
predicted indirect and its ratio to the reference indirect (full_r1 - first_r1). Also the area-weighted mean gathered
Eind per port class for Yellow ports.

    python b3g_eind_split.py <out dir> [views=6,15]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "fnto_lego_r1")); sys.path.insert(0, str(HERE.parents[1] / "fnto_lego"))
import r1_common as C  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722]); CLASSES = {"Yellow": [0], "opaque": [1, 3, 5, 6, 7], "glass": [2, 4], "box": [8]}


@torch.no_grad()
def main(out, views=(6, 15)):
    import r1_solve as SV
    from r1_eval import load_kernels
    out = Path(out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda")
    An = np.load(C.OUT1 / "anchors.npz"); N = len(An["obj"]); nA = int(An["n_A"]); Ar = An["area"].astype(np.float64)
    pr = C.OUT1 / "stagea/prep2_train"; pp = np.load(pr / "px_port.npy", mmap_mode="r"); pd = np.load(pr / "px_dom.npy", mmap_mode="r"); cnt = np.zeros(N, np.int64)
    for s in range(0, len(pp), 1 << 24):
        a, d = np.asarray(pp[s:s + (1 << 24)]), np.asarray(pd[s:s + (1 << 24)]); cnt[:nA] += np.bincount(a[(d == 0) & (a >= 0) & (a < nA)], minlength=nA)
    cnt[nA:] = 1                                                                    # box ports: analytic grid, always supervised
    KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); T = SV.Tables(KA, KB, dev)
    H = np.load(C.OUT1 / "rel_T0.npz")["hit"]; vis = np.load(C.OUT1 / "eval_vis_k44.npz"); px = np.load(C.OUT1 / "eval_px/heldout_T0.npz")
    first = np.load(C.OUT1 / "refs/heldout_T0_first.npz"); full = np.load(C.OUT1 / "refs/heldout_T0_full.npz"); Vk = np.load(C.OUT1 / "eval_K_k44P/views.npz")
    rec = {"views": list(views)}; yp = (An["mat"] == 0) & (An["obj"] == 0)
    for v in views:
        val = px[f"v{v}_valid"].astype(bool); mat = np.where(px[f"v{v}_dom"][val] == 1, 8, px[f"v{v}_mat"][val])
        pxd = {k: torch.as_tensor(px[f"v{v}_{k}"][val].astype(np.float32 if k in ("pos", "nrm", "cam", "vis") else np.int64), device=dev) for k in ("pos", "nrm", "cam", "vis", "dom", "port")}
        ko = Vk[f"heldout_T0_v{v}_K_only"][:, :3].astype(np.float64); ref = (full[f"v{v}_rgb"][val] - first[f"v{v}_rgb"][val]).astype(np.float64) @ LUM
        visp = np.zeros(N, bool); visp[px[f"v{v}_port"][val].astype(np.int64)] = True
        cls = {"visible": visp, "observed": ~visp & (cnt > 0), "never": ~visp & (cnt == 0)}
        S = SV.State(T, H, vis[f"heldout_T0_v{v}_T0"].astype(np.float32), px[f"v{v}_light"]).solve()
        prd, _ = SV.render_pixels(T, S, pxd); pind = (prd.cpu().numpy().astype(np.float64) - ko) @ LUM
        Eind, h = S.aggregates(); h[torch.arange(T.N, device=dev), S.sunbin] -= S.Esun[:, None]
        Ln = S.L; S.L = torch.zeros_like(Ln[:1]).expand_as(Ln)                       # receiver-local term off (pixel port L = 0)
        resp = {}; pvz = {**pxd, "vis": torch.zeros_like(pxd["vis"])}
        for nm, m in cls.items():
            mt = torch.as_tensor(m, device=dev)
            S.outgoing(Eind * mt[:, None], h * mt[:, None, None], sun_on=False); r_, _ = SV.render_pixels(T, S, pvz); resp[nm] = r_.cpu().numpy().astype(np.float64) @ LUM
        S.L = Ln; Z = torch.zeros_like(Eind); S.outgoing(Z, torch.zeros_like(h), sun_on=False); r_, _ = SV.render_pixels(T, S, pvz); resp["receiver_local"] = r_.cpu().numpy().astype(np.float64) @ LUM
        Ey = Eind.cpu().numpy() @ LUM; r = {"iters": S.iters, "status": S.status, "classes": {}, "yellow_port_mean_Eind_area_weighted": {}}
        for nm, m in cls.items():
            mm = m & yp; r["yellow_port_mean_Eind_area_weighted"][nm] = {"area_share": float(Ar[mm].sum() / Ar[yp].sum()), "mean_Eind_lum": float((Ey[mm] * Ar[mm]).sum() / max(Ar[mm].sum(), 1e-30))}
        for cname, ids in CLASSES.items():
            m = np.isin(mat, ids)
            if m.sum() < 100:
                continue
            rs = ref[m].sum(); tot = sum(resp[k][m].sum() for k in resp)
            r["classes"][cname] = {"px": int(m.sum()), "pred_ind_over_ref": float(pind[m].sum() / rs), "closure_parts_over_pred_ind": float(tot / pind[m].sum()),
                                   **{f"{k}_over_ref": float(resp[k][m].sum() / rs) for k in resp}}
        rec[f"view{v}"] = r; print(v, json.dumps(r, indent=1), flush=True)
        del S, Ln, Eind, h; torch.cuda.empty_cache()
    (out / "b3g_eind_split.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], tuple(int(x) for x in a[1].split(",")) if len(a) > 1 else (6, 15))
