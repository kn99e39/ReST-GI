"""WL46 Stage 1 - capability audit of the frozen K_k44P / r1 C interface (measurements, nothing changed).

On held-out T0 views (default 6, 15), production state (rel_T0, eval_vis_k44, view sun):
  1. K_loc parameters per material over all ports: rho (lum), sum kappa (lum), alpha of each lobe (min / q05 / median).
  2. Directional dependence of the outgoing radiance K actually produces (converged production state): for a sample of
     ports per class, the nonlocal part (band + mid; glob is isotropic by design) and the local part (incoming 'E' and
     'dir') at the 19 hemispherical bin centres + 64 cosine-distributed directions; per class: ratio of the
     cosine-weighted hemispherical mean to the normal-direction value (flux / (pi L(n))), angular coefficient of
     variation, max / min over the 19 centres.
  3. Decomposition of worklog 45's B3d finding (visible Yellow ports send 0.35-0.41x the reference) with the sun-only
     first-bounce state (L = 0): at visible pixels, (a) the K pixel path toward the camera (= K_only), (b) the pixel path
     with the camera direction replaced by the pixel normal, (c) the port query toward the camera, (d) the port query at
     the port normal (= production sender), each over the first_r1 reference; per class.
  4. Query cost: nonlocal / local query throughput (queries per second) on this GPU.

    python s1_audit.py <out dir> [views=6,15] [ports_per_class=3000]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dir46_core as D  # noqa: E402
import r1_common as C  # noqa: E402

CLS = {"Yellow": [0], "opaque": [1, 3, 5, 6, 7], "glass": [2, 4], "box": [8]}


def lum(x):
    return x @ torch.as_tensor(D.LUM, dtype=torch.float32, device=x.device)


@torch.no_grad()
def main(out, views=(6, 15), npc=3000):
    import r1_solve as SV
    from r1_eval import load_kernels
    out = Path(out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda")
    KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); T = SV.Tables(KA, KB, dev)
    An = np.load(C.OUT1 / "anchors.npz"); cls_port = np.where(An["obj"] == 1, 8, An["mat"]).astype(np.int64)
    H = np.load(C.OUT1 / "rel_T0.npz")["hit"]; vis = np.load(C.OUT1 / "eval_vis_k44.npz"); px = np.load(C.OUT1 / "eval_px/heldout_T0.npz")
    first = np.load(C.OUT1 / "refs/heldout_T0_first.npz")
    rec = {"views": list(views), "kloc_params": {}, "directional": {}, "b3d_decomposition": {}, "cost": {}}
    for cn, ids in CLS.items():                                                      # 1. K_loc parameters
        m = torch.as_tensor(np.isin(cls_port, ids), device=dev)
        rho, kap, al = lum(T.rho[m]), lum(T.kap[m].sum(1)), T.alpha[m]
        rec["kloc_params"][cn] = {"ports": int(m.sum()), "rho_lum_median": float(rho.median()), "kappa_sum_lum_median": float(kap.median()),
                                  "kappa_sum_lum_q95": float(torch.quantile(kap[:100000], 0.95)),
                                  **{f"alpha{l}_min_q05_median": [float(al[:, l].min()), float(torch.quantile(al[:100000, l], 0.05)), float(al[:, l].median())] for l in range(2)}}
    rng = np.random.default_rng(46)
    cen = D.bin_centres_local()
    for v in views:
        S = SV.State(T, H, vis[f"heldout_T0_v{v}_T0"].astype(np.float32), px[f"v{v}_light"]).solve()     # production, converged
        r = {"iters": S.iters, "status": S.status}
        for cn, ids in CLS.items():                                                  # 2. directional dependence
            cand = np.flatnonzero(np.isin(cls_port, ids)); q = torch.as_tensor(np.sort(rng.choice(cand, min(npc, len(cand)), replace=False)), device=dev)
            nq = T.nrm[q]; W19 = D.to_world(nq, cen); Wc = D.cosine_dirs(nq, 64)
            res = {}
            for lab, fn in (("nonlocal_band_mid", lambda w: sum(D.nonlocal_out(T, S, q, w)[1:])),
                            ("local_E", lambda w: D.local_out(T, S, q, w, "E")), ("local_dir", lambda w: D.local_out(T, S, q, w, "dir"))):
                Ln = lum(fn(nq))
                L19 = torch.stack([lum(fn(W19[:, j].contiguous())) for j in range(W19.shape[1])], 1)
                Lc = torch.stack([lum(fn(Wc[:, j].contiguous())) for j in range(Wc.shape[1])], 1)
                ok = Ln > 1e-6
                flux_ratio = Lc.mean(1)[ok].sum() / Ln[ok].sum()                   # (1/pi) int L cos dw / L(n), summed over ports
                cv = (L19.std(1) / L19.mean(1).clamp(min=1e-9))[ok]
                mm = (L19.max(1).values / L19.min(1).values.clamp(min=1e-9))[ok]
                res[lab] = {"ports": int(ok.sum()), "flux_over_pi_L_normal_pooled": float(flux_ratio), "angular_cv_median": float(cv.median()),
                            "max_over_min_19_median": float(mm.median()), "max_over_min_19_q90": float(torch.quantile(mm, 0.9)),
                            "L_normal_mean_lum": float(Ln.mean()), "L_cosmean_mean_lum": float(Lc.mean())}
            r[cn] = res
        rec["directional"][f"view{v}"] = r
        del S; torch.cuda.empty_cache()
        # 3. B3d decomposition, sun-only first bounce
        S1 = SV.State(T, H, vis[f"heldout_T0_v{v}_T0"].astype(np.float32), px[f"v{v}_light"])
        Eind, h = S1.aggregates(); S1.outgoing(Eind, h)
        val = px[f"v{v}_valid"].astype(bool); matp = np.where(px[f"v{v}_dom"][val] == 1, 8, px[f"v{v}_mat"][val])
        pxd = {k: torch.as_tensor(px[f"v{v}_{k}"][val].astype(np.float32 if k in ("pos", "nrm", "cam", "vis") else np.int64), device=dev) for k in ("pos", "nrm", "cam", "vis", "dom", "port")}
        ref = torch.as_tensor(first[f"v{v}_rgb"][val].astype(np.float32), device=dev)
        a_, _ = SV.render_pixels(T, S1, pxd); b_, _ = SV.render_pixels(T, S1, {**pxd, "cam": pxd["nrm"]})
        port = pxd["port"]; cam = pxd["cam"]; nport = T.nrm[port]
        cam_up = torch.where(((nport * cam).sum(-1) > 0)[:, None], cam, nport)        # port frame may differ from the pixel's
        c_ = D.query_out(T, S1, port, cam_up, cam_up, "E"); d_ = D.query_out(T, S1, port, nport, nport, "E")
        dd = {}
        for cn, ids in CLS.items():
            m = torch.as_tensor(np.isin(matp, ids), device=dev)
            if m.sum() < 100:
                continue
            rl = lum(ref[m]).sum()
            dd[cn] = {"px": int(m.sum()), "pixel_path_cam_over_ref": float(lum(a_[m]).sum() / rl), "pixel_path_normal_over_ref": float(lum(b_[m]).sum() / rl),
                      "port_query_cam_over_ref": float(lum(c_[m]).sum() / rl), "port_query_normal_over_ref": float(lum(d_[m]).sum() / rl)}
        rec["b3d_decomposition"][f"view{v}"] = dd; print(v, json.dumps(dd), flush=True)
        del S1; torch.cuda.empty_cache()
    # 4. cost
    S = SV.State(T, H, vis["heldout_T0_v6_T0"].astype(np.float32), px["v6_light"]); Eind, h = S.aggregates(); S.outgoing(Eind, h)
    q = torch.as_tensor(rng.choice(T.N, 200000, replace=False), device=dev); w = T.nrm[q]
    for lab, fn in (("nonlocal", lambda: D.nonlocal_out(T, S, q, w)), ("local_E", lambda: D.local_out(T, S, q, w, "E")), ("local_dir", lambda: D.local_out(T, S, q, w, "dir"))):
        fn(); torch.cuda.synchronize(); t = time.time(); fn(); torch.cuda.synchronize(); rec["cost"][f"{lab}_queries_per_s"] = float(len(q) / (time.time() - t))
    t = time.time(); S.outgoing(Eind, h); torch.cuda.synchronize(); rec["cost"]["production_outgoing_s"] = time.time() - t
    rec["cost"]["gpu"] = torch.cuda.get_device_name(0); rec["cost"]["peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
    (out / "s1_audit.json").write_text(json.dumps(rec, indent=1)); print(json.dumps({k: rec[k] for k in ("kloc_params", "cost")}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], tuple(int(x) for x in a[1].split(",")) if len(a) > 1 else (6, 15), int(a[2]) if len(a) > 2 else 3000)
