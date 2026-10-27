"""WL48 Probe A - directional validity of the frozen K_k44P sender (queries only; nothing trained or changed).

Per held-out T0 view (6, 15) and port group (Yellow / opaque x observed / never observed x lit / shadow, protocol
fnto_diag48 probe_A): nonlocal (glob + band + mid) and local (production 'E') outgoing luminance at the port normal,
training camera directions, C-requested directions (-d of rel_T0 gathers hitting the port), 19 bin centres and 64
cosine directions (converged production state). Energy accounting on the worklog-45 first-event field (sun only).
Physical reference: worklog 47's three-camera Yellow SSS exit profiles.

    python probe_a48.py <out dir> [views=6,15] [ports_per_cell=1500]
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fnto_dir46"))
import dir46_core as D  # noqa: E402
import dir46_solve as DS  # noqa: E402
import r1_common as C  # noqa: E402
import fnto_model as M  # noqa: E402

LUM = torch.tensor(D.LUM, dtype=torch.float32)
OPAQUE = [1, 3, 5, 6, 7]


def lum(x):
    return x @ LUM.to(x.device)


def angle_min(dirs, ref):
    """Smallest angle (deg) from each dirs[i, m] to any ref[i, r] (ref padded with 0 vectors)."""
    c = torch.einsum("imk,irk->imr", dirs, ref).amax(-1).clamp(-1, 1)
    return torch.rad2deg(torch.acos(c))


@torch.no_grad()
def query(T, S, q, W):
    """Nonlocal / local luminance [n, m] of ports q [n] toward W [n, m, 3]."""
    n, m = W.shape[:2]; qq = q[:, None].expand(n, m).reshape(-1); ww = DS.upper(T.nrm[qq], W.reshape(-1, 3))
    nl, lo = [], []
    for s in range(0, len(qq), 1 << 16):
        g, b, mm = D.nonlocal_out(T, S, qq[s:s + (1 << 16)], ww[s:s + (1 << 16)])
        nl.append(lum(g + b + mm)); lo.append(lum(D.local_out(T, S, qq[s:s + (1 << 16)], ww[s:s + (1 << 16)], "E")))
    return torch.cat(nl).view(n, m), torch.cat(lo).view(n, m)


def sss_reference():
    p = C.ROOT / "results/fnto_smoke47/sss/sss.npz"
    if not p.exists():
        return None
    d = np.load(p); out = {}
    for lo, hi in ((0.03, 0.06), (0.06, 0.12), (0.12, 0.30)):
        v = {}
        for cam in ("top", "yp45", "ym45"):
            x = d[f"x_{cam}"].mean(0); m = (x >= lo) & (x < hi)
            v[cam] = np.mean([d[f"A_rep{r}_{cam}"].mean(0)[m].mean(0) for r in (0, 1)], 0)
        arr = np.stack([v[c] for c in ("top", "yp45", "ym45")])               # [3 cams, RGB]
        out[f"{lo:.2f}-{hi:.2f}"] = {"max_over_min_RGB": (arr.max(0) / arr.min(0)).tolist(), "cams_RGB": arr.tolist()}
    return out


@torch.no_grad()
def main(out, views=(6, 15), npc=1500):
    import r1_solve as SV
    from r1_eval import load_kernels
    out = Path(out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda"); t00 = time.time()
    KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); dig0 = M.digest(KA, KB); T = SV.Tables(KA, KB, dev)
    An = np.load(C.OUT1 / "anchors.npz"); nA = int(An["n_A"]); mat = An["mat"].astype(np.int64); area = An["area"].astype(np.float64)
    pr = C.OUT1 / "stagea/prep2_train"; pp = np.load(pr / "px_port.npy", mmap_mode="r"); pdm = np.load(pr / "px_dom.npy", mmap_mode="r")
    lego_rows = np.flatnonzero(np.asarray(pdm) == 0); ppl = np.asarray(pp)[lego_rows]
    cov = np.bincount(ppl, minlength=T.N); obs = cov > 0
    H = np.load(C.OUT1 / "rel_T0.npz")["hit"]; vis = np.load(C.OUT1 / "eval_vis_k44.npz"); px = np.load(C.OUT1 / "eval_px/heldout_T0.npz")
    rng = np.random.default_rng(4801)
    yel = (np.arange(T.N) < nA) & (mat == 0); opq = (np.arange(T.N) < nA) & np.isin(mat, OPAQUE)
    rec = {"coverage": {cn: {"ports": int(m.sum()), "observed_port_share": float(obs[m].mean()), "observed_area_share": float(area[m & obs].sum() / area[m].sum())}
                        for cn, m in (("Yellow", yel), ("opaque", opq))},
           "train_pixels_lego": int(len(lego_rows)), "physical_reference_wl47_sss": sss_reference(), "views": {}}
    cen = D.bin_centres_local()
    for v in views:
        vf = vis[f"heldout_T0_v{v}_T0"].astype(np.float32); light = px[f"v{v}_light"]
        S = SV.State(T, H, vf, light).solve(); lit = (S.Esun > 0).cpu().numpy()
        r = {"solve": {"status": S.status, "iters": S.iters}, "groups": {}}
        cells = {}
        for cn, cm in (("Yellow", yel), ("opaque", opq)):
            for on, om in (("observed", obs), ("never", ~obs)):
                for ln, lm in (("lit", lit), ("shadow", ~lit)):
                    cand = np.flatnonzero(cm & om & lm)
                    if len(cand):
                        cells[f"{cn}_{on}_{ln}"] = np.sort(rng.choice(cand, min(npc, len(cand)), replace=False))
        allq = np.unique(np.concatenate(list(cells.values())))
        # training camera directions (up to 64 per port) and C-requested directions (up to 16 per port)
        sel_rows = lego_rows[np.isin(ppl, allq)]; sp = np.asarray(pp)[sel_rows]
        cam = np.load(pr / "px_cam.npy", mmap_mode="r")[np.sort(sel_rows)].astype(np.float32); sp = np.asarray(pp)[np.sort(sel_rows)]
        o = np.argsort(sp, kind="stable"); sp, cam = sp[o], cam[o]; st = np.searchsorted(sp, allq); en = np.searchsorted(sp, allq, "right")
        CAM = np.zeros((len(allq), 64, 3), np.float32)
        for i, (a, b) in enumerate(zip(st, en)):
            if b > a:
                k = np.sort(rng.choice(np.arange(a, b), min(64, b - a), replace=False)); CAM[i, :len(k)] = cam[k]
        hm = np.isin(H, allq); pr_, kr_ = np.nonzero(hm); hq = H[pr_, kr_]; o = np.argsort(hq, kind="stable"); pr_, kr_, hq = pr_[o], kr_[o], hq[o]
        st2 = np.searchsorted(hq, allq); en2 = np.searchsorted(hq, allq, "right")
        REQ = np.zeros((len(allq), 16, 3), np.float32); nreq = np.zeros(len(allq), np.int64)
        dirs_np = None
        for i, (a, b) in enumerate(zip(st2, en2)):
            if b > a:
                k = np.sort(rng.choice(np.arange(a, b), min(16, b - a), replace=False)); nreq[i] = len(k)
                REQ[i, :len(k)] = -T.dirs[torch.as_tensor(pr_[k], device=dev), torch.as_tensor(kr_[k], device=dev)].float().cpu().numpy()
        q = torch.as_tensor(allq, device=dev); nq = T.nrm[q]
        CAMt, REQt = torch.as_tensor(CAM, device=dev), torch.as_tensor(REQ, device=dev)
        ncam = (CAMt.norm(dim=-1) > 0.5).sum(1)
        W19 = D.to_world(nq, cen); Wc = D.cosine_dirs(nq, 64)
        NLn, LOn = query(T, S, q, nq[:, None]); NL19, _ = query(T, S, q, W19); NLc, LOc = query(T, S, q, Wc)
        NLcam, _ = query(T, S, q, torch.where(CAMt.norm(dim=-1, keepdim=True) > 0.5, CAMt, nq[:, None].expand_as(CAMt))[:, :8].contiguous())
        NLreq, LOreq = query(T, S, q, torch.where(REQt.norm(dim=-1, keepdim=True) > 0.5, REQt, nq[:, None].expand_as(REQt)))
        cosm = NLc.mean(1); AI = NL19.max(1).values / NL19.min(1).values.clamp(min=1e-12); F = cosm / NLn.clamp(min=1e-12)
        ang_req = angle_min(REQt, CAMt); reqv = REQt.norm(dim=-1) > 0.5
        dev_req = (torch.log(NLreq.clamp(min=1e-12) / cosm[:, None].clamp(min=1e-12))).abs()
        sup = reqv & (ang_req <= 10) & (ncam[:, None] > 0); uns = reqv & (ang_req > 20) & (ncam[:, None] > 0)
        camv = torch.arange(8, device=dev)[None] < ncam[:, None]
        dev_cam = (torch.log(NLcam.clamp(min=1e-12) / cosm[:, None].clamp(min=1e-12))).abs()
        idx = {int(p): i for i, p in enumerate(allq)}
        finite = bool(torch.isfinite(NL19).all() and torch.isfinite(NLc).all() and torch.isfinite(NLreq).all() and torch.isfinite(NLcam).all())
        r["finite_all_queries"] = finite
        for cname, ports in cells.items():
            ii = torch.as_tensor([idx[int(p)] for p in ports], device=dev); ok = cosm[ii] > 1e-6
            g = {"ports": int(len(ports)), "ports_nonlocal_gt_1e-6": int(ok.sum()),
                 "nonlocal_normal_mean": float(NLn[ii, 0].mean()), "nonlocal_cosmean_mean": float(cosm[ii].mean()), "local_normal_mean": float(LOn[ii, 0].mean()),
                 "nonlocal_share_of_cos_flux": float(cosm[ii].sum() / max(float((cosm[ii] + LOc[ii].mean(1)).sum()), 1e-30)),
                 "C_requests_per_port_mean": float(nreq[ii.cpu().numpy()].mean())}
            if ok.sum() > 20:
                jj = ii[ok]
                g.update({"AI_median": float(AI[jj].median()), "AI_q90": float(torch.quantile(AI[jj], 0.9)), "AI_max": float(AI[jj].max()),
                          "F_cosmean_over_normal_median": float(F[jj].median()), "F_pooled": float(cosm[jj].sum() / NLn[jj, 0].sum()),
                          "req_over_normal_median": float((NLreq[jj] / NLn[jj].clamp(min=1e-12))[reqv[jj]].median()) if reqv[jj].any() else None,
                          "req_over_normal_q99": float(torch.quantile((NLreq[jj] / NLn[jj].clamp(min=1e-12))[reqv[jj]], 0.99)) if reqv[jj].any() else None,
                          "req_over_cosmean_pooled": float((NLreq[jj] * reqv[jj]).sum() / (cosm[jj, None] * reqv[jj]).sum()) if reqv[jj].any() else None,
                          "dev_supervised_req_median": float(dev_req[jj][sup[jj]].median()) if sup[jj].sum() > 20 else None,
                          "dev_unsupervised_req_median": float(dev_req[jj][uns[jj]].median()) if uns[jj].sum() > 20 else None,
                          "n_req_supervised": int(sup[jj].sum()), "n_req_unsupervised": int(uns[jj].sum()),
                          "dev_camera_dirs_median": float(dev_cam[jj][camv[jj]].median()) if camv[jj].sum() > 20 else None,
                          "req_angle_to_nearest_train_cam_median_deg": float(ang_req[jj][reqv[jj] & (ncam[jj, None] > 0)].median()) if (reqv[jj] & (ncam[jj, None] > 0)).any() else None})
            r["groups"][cname] = g
        # energy accounting on the first-event field (amendment 1: the converged state is released first - GPU memory)
        del S, NL19, NLc, LOc, NLreq, LOreq, NLcam, W19, Wc; torch.cuda.empty_cache()
        S1 = SV.State(T, H, vf, light); Eind, h = S1.aggregates(); S1.outgoing(Eind, h)
        qs = torch.as_tensor(np.sort(rng.choice(nA, 200000, replace=False)), device=dev); wq = (nA / len(qs))
        Wf = D.cosine_dirs(T.nrm[qs], 64); fl_nl = torch.zeros(3, device=dev); fl_lo = torch.zeros(3, device=dev); fl_n = torch.zeros(3, device=dev)
        fl_y = torch.zeros(3, device=dev); yq = torch.as_tensor(mat[qs.cpu().numpy()] == 0, device=dev)
        for s in range(0, len(qs), 4096):
            qq = qs[s:s + 4096]; Wq = Wf[s:s + 4096]; n_, m_ = Wq.shape[:2]; q2 = qq[:, None].expand(n_, m_).reshape(-1); w2 = Wq.reshape(-1, 3)
            g, b, mm = D.nonlocal_out(T, S1, q2, w2); lo = D.local_out(T, S1, q2, w2, "E")
            A_ = T.area[qq][:, None]
            nlm = (g + b + mm).view(n_, m_, 3).mean(1); lom = lo.view(n_, m_, 3).mean(1)
            fl_nl += (math.pi * nlm * A_).sum(0); fl_lo += (math.pi * lom * A_).sum(0); fl_y += (math.pi * (nlm + lom) * A_)[yq[s:s + 4096]].sum(0)
            gn, bn, mn = D.nonlocal_out(T, S1, qq, T.nrm[qq]); ln_ = D.local_out(T, S1, qq, T.nrm[qq], "E")
            fl_n += (math.pi * (gn + bn + mn + ln_) * A_).sum(0)
        Pin = float((T.area[:nA] * S1.Esun[:nA]).sum())
        r["energy_first_event"] = {"sun_power_absorbed_lego": Pin, "out_hemispherical_over_in_RGB": ((fl_nl + fl_lo) * wq / Pin).tolist(),
                                   "out_nonlocal_hemispherical_over_in_RGB": (fl_nl * wq / Pin).tolist(), "out_local_hemispherical_over_in_RGB": (fl_lo * wq / Pin).tolist(),
                                   "out_normal_production_over_in_RGB": (fl_n * wq / Pin).tolist(), "out_from_yellow_ports_hemispherical_over_in_RGB": (fl_y * wq / Pin).tolist(),
                                   "subsample_ports": int(len(qs)), "note": "flux = sum A pi mean_64cos L (all K parts); sun power = sum A Esun over Lego ports"}
        rec["views"][f"view{v}"] = r; print(v, json.dumps(r["groups"].get("Yellow_never_shadow", {})), json.dumps(r["energy_first_event"]), flush=True)
        del S1; torch.cuda.empty_cache()
    crit = {}
    for v in views:
        gg = rec["views"][f"view{v}"]["groups"]; e = rec["views"][f"view{v}"]["energy_first_event"]
        yo = [gg[k]["AI_median"] for k in ("Yellow_observed_lit", "Yellow_observed_shadow") if k in gg and "AI_median" in gg[k]]
        yn = [gg[k]["AI_median"] for k in ("Yellow_never_lit", "Yellow_never_shadow") if k in gg and "AI_median" in gg[k]]
        ds = [gg[k].get("dev_supervised_req_median") for k in ("Yellow_observed_lit", "Yellow_observed_shadow") if k in gg]
        du = [gg[k].get("dev_unsupervised_req_median") for k in ("Yellow_observed_lit", "Yellow_observed_shadow") if k in gg]
        ds = [x for x in ds if x is not None]; du = [x for x in du if x is not None]
        crit[f"view{v}"] = {"yellow_AI_median_observed": float(np.median(yo)) if yo else None, "yellow_AI_median_never": float(np.median(yn)) if yn else None,
                            "output_invalid": bool((np.median(yo + yn) > 1.5 if yo + yn else False) or max(e["out_hemispherical_over_in_RGB"]) > 1.0),
                            "port_supervision_association": bool(yo and yn and np.median(yn) >= 1.5 * np.median(yo)),
                            "direction_supervision_association": bool(ds and du and np.median(du) >= 1.5 * np.median(ds)),
                            "dev_supervised": float(np.median(ds)) if ds else None, "dev_unsupervised": float(np.median(du)) if du else None}
    rec["criteria"] = crit; rec["frozen_digest_identical"] = M.digest(KA, KB) == dig0
    rec["cost"] = {"wall_s": time.time() - t00, "peak_GB": torch.cuda.max_memory_allocated() / 1e9, "gpu": torch.cuda.get_device_name(0)}
    (out / "probe_a.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(crit, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], tuple(int(x) for x in a[1].split(",")) if len(a) > 1 else (6, 15), int(a[2]) if len(a) > 2 else 1500)
