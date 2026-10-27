"""WL48 Probe C - directional C reads in the worklog-46 MS scene: 19-bin basis vs exact per-ray sender queries.

C0 = R6a (iso senders + plane IS; stored worklog-46 predictions, recomputed here as the baseline check), C1 = R6br
(19-bin basis reads into the plane + IS; stored), C2 = exact per-ray nonlocal sender queries for every read into the
plane (subclass of ms46_eval.MSState; worklog-46 code unchanged), C2b = basis reads in C2's converged state (diagnostic).
Per-ray basis / exact sender ratios and per-class contributions on the plane IS rays; edit metrics as ms46_eval.

    python probe_c48.py <ms46 geom dir> <ms46 refs dir> <ms46_pred.npz> <out dir>
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
import ms46_eval as ME  # noqa: E402
import r1_common as C  # noqa: E402
import fnto_model as M  # noqa: E402

LUM = ME.LUM; CLS = ("Yellow", "opaque", "glass", "box", "plane", "miss")


class MSStateX(ME.MSState):
    @torch.no_grad()
    def k_sender(self, q, d):
        if self.plane_read == "exact":
            nlv = sum(D.nonlocal_out(self.T, self, q, DS.upper(self.T.nrm[q], -d)))
            return nlv + self.loc_iso[q] + self.sun_lobe(q, -d)
        return super().k_sender(q, d)


@torch.no_grad()
def plane_is_split(S, px, idx, cls_port):
    """Plane-pixel radiance toward the camera (IS path of ms46_eval.plane_pixels) split by the class of the hit."""
    T = S.T; dev = T.dev; cam = torch.as_tensor(px["cam"][idx], device=dev)
    pidx = np.searchsorted(np.flatnonzero(px["dom"] == 2), idx)
    wi = torch.as_tensor(px["is_dir"][pidx], device=dev); ww = torch.as_tensor(px["is_w"][pidx], device=dev); hit = torch.as_tensor(px["is_hit"][pidx].astype(np.int64), device=dev)
    L = torch.zeros(*hit.shape, 3, device=dev); ok = hit >= 0; kq = ok & (hit < T.N); pq = ok & (hit >= T.N)
    if kq.any():
        L[kq] = torch.cat([S.k_sender(hit[kq][s:s + (1 << 15)], wi[kq][s:s + (1 << 15)]) for s in range(0, int(kq.sum()), 1 << 15)])
    if pq.any():
        L[pq] = S.plane_out(hit[pq] - T.N, -wi[pq])
    f = ME.ggx_plane(wi, cam[:, None].expand_as(wi)); c = ME.KAP * f[..., None] * L * ww[..., None]
    cl = torch.full(hit.shape, 5, device=dev, dtype=torch.long); cl[kq] = cls_port[hit[kq]]; cl[pq] = 4
    split = torch.stack([(c * (cl == j)[..., None]).sum(1) for j in range(len(CLS))], 1)          # [n, 6, 3]
    return split.sum(1).cpu().numpy(), split.cpu().numpy()


@torch.no_grad()
def ray_compare(S, px, cls_port, nmax=400000, seed=4803):
    """Exact vs basis sender values on plane IS rays that hit K ports (same state)."""
    T = S.T; dev = T.dev; pidx = np.arange(len(px["is_hit"])); hit = px["is_hit"]; wi = px["is_dir"]
    r_, k_ = np.nonzero((hit >= 0) & (hit < T.N)); sel = np.random.default_rng(seed).choice(len(r_), min(nmax, len(r_)), replace=False)
    q = torch.as_tensor(hit[r_[sel], k_[sel]].astype(np.int64), device=dev); d = torch.as_tensor(wi[r_[sel], k_[sel]], device=dev)
    w = DS.upper(T.nrm[q], -d); ex = sum(D.nonlocal_out(T, S, q, w))
    nq = T.nrm[q]; td, bd = T.detf[0][q], T.detf[1][q]; b = M.hemi_bin(torch.stack([(w * td).sum(-1), (w * bd).sum(-1), (w * nq).sum(-1)], -1))
    ba = S.O[q, b.long()]; ln = S.NL_n[q]; L_ = torch.as_tensor(LUM, dtype=torch.float32, device=dev)
    e, bb, nn = ex @ L_, ba @ L_, ln @ L_; cl = cls_port[q]; out = {}
    for j, cn in enumerate(CLS[:4]):
        m = cl == j
        if m.sum() > 100:
            out[cn] = {"rays": int(m.sum()), "basis_over_exact_pooled": float(bb[m].sum() / e[m].sum().clamp(min=1e-30)),
                       "normal_over_exact_pooled": float(nn[m].sum() / e[m].sum().clamp(min=1e-30)),
                       "rel_L2_basis_vs_exact": float((bb[m] - e[m]).norm() / e[m].norm().clamp(min=1e-30)),
                       "exact_q99_over_median": float(torch.quantile(e[m], 0.99) / e[m].median().clamp(min=1e-30)),
                       "exact_max": float(e[m].max()), "finite": bool(torch.isfinite(e[m]).all())}
    return out


def edit_metrics(pred, G0, G1, rdir, rep, conds, v):
    a0 = {int(k): i for i, k in enumerate(G0[f"v{v}_mk"])}; m1 = G1[f"v{v}_mk"]
    both = [(a0[int(k)], j) for j, k in enumerate(m1) if int(k) in a0 and G0[f"v{v}_dom"][a0[int(k)]] == 2 and G1[f"v{v}_dom"][j] == 2]
    i0, i1 = map(np.array, zip(*both))
    r0 = np.load(rdir / f"ref_T0_v{v}.npz")["rgb"][G0[f"v{v}_mk"]][i0] @ LUM; r1 = np.load(rdir / f"ref_T1_v{v}.npz")["rgb"][m1][i1] @ LUM; dr = r1 - r0
    e0 = rep["T0"][G0["v0_mk"]] @ LUM; e1 = rep["T1"][G1["v0_mk"]] @ LUM
    out = {"px": int(len(i0)), "ref_change_rms": float(np.sqrt((dr ** 2).mean()))}
    for c in conds:
        if (c, "T0", v) not in pred or (c, "T1", v) not in pred:
            continue
        dp = pred[(c, "T1", v)][i1] @ LUM - pred[(c, "T0", v)][i0] @ LUM
        out[c] = {"corr": float(np.corrcoef(dp, dr)[0, 1]), "slope": float((dp * dr).sum() / max((dr * dr).sum(), 1e-30)),
                  "explained": float(1 - ((dp - dr) ** 2).sum() / max((dr * dr).sum(), 1e-30))}
    return out, (i0, i1, dr)


@torch.no_grad()
def main(gdir, rdir, pred_npz, out):
    import r1_solve as SV
    from r1_eval import load_kernels
    gdir, rdir, out = Path(gdir), Path(rdir), Path(out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda"); t00 = time.time()
    KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); dig0 = M.digest(KA, KB); T = SV.Tables(KA, KB, dev)
    An = np.load(C.OUT1 / "anchors.npz"); mat = np.zeros(T.N, np.int64); mat[:len(An["mat"])] = An["mat"]; obj = np.zeros(T.N, np.int64); obj[:len(An["obj"])] = An["obj"]
    cls_np = np.select([obj == 1, mat == 0, np.isin(mat, [2, 4])], [3, 0, 2], 1); cls_port = torch.as_tensor(cls_np, device=dev)
    stored = np.load(pred_npz); pred = {}; split = {}; rec = {"solves": {}, "ray_compare": {}, "timing": {}}
    for st in ("T0", "T1"):
        for v in range(3):
            for c, key in (("C0", "R6a"), ("C1", "R6br")):
                pred[(c, st, v)] = stored[f"{key}_{st}_v{v}"]
    for st in ("T0", "T1"):
        G = dict(np.load(gdir / f"geom_{st}.npz"))
        for cname, sm in (("C0r", dict(plane_mode="iso", nl="n", loc_in="E")), ("C2", dict(plane_mode="dir", nl="n", loc_in="E", plane_read="exact"))):
            t = time.time(); S = MSStateX(T, G, **sm).solve(); torch.cuda.synchronize(); ts = time.time() - t
            rec["solves"][f"{st}_{cname}"] = {"status": S.status, "iters": S.iters, "seconds": ts, "timing": S.timing, "hist": [float(x) for x in S.hist]}
            print(st, cname, rec["solves"][f"{st}_{cname}"]["status"], S.iters, f"{ts:.1f}s", flush=True)
            if S.status != "converged":
                del S; torch.cuda.empty_cache(); continue
            for v in range(3):
                px = {k[len(f"v{v}_"):]: G[k] for k in G if k.startswith(f"v{v}_")}
                val = px["valid"].astype(bool); kp = np.flatnonzero(val & (px["dom"] < 2)); pp = np.flatnonzero(val & (px["dom"] == 2))
                pxd = {k: torch.as_tensor(px[k][kp].astype(np.float32 if k in ("pos", "nrm", "cam", "vis") else np.int64), device=dev) for k in ("pos", "nrm", "cam", "vis", "dom", "port")}
                kpix = SV.render_pixels(T, S, pxd)[0].cpu().numpy()
                t = time.time(); pv, sp = plane_is_split(S, px, pp, cls_port); torch.cuda.synchronize(); rec["timing"][f"{st}_v{v}_{cname}_pixel_reads_s"] = time.time() - t
                img = np.zeros((len(px["mk"]), 3), np.float32); img[kp] = kpix; img[pp] = pv; pred[(cname, st, v)] = img; split[(cname, st, v)] = sp
                if cname == "C2":
                    t = time.time()
                    if S.O is None:
                        S.O = S.basis_table()
                    rec["timing"][f"{st}_basis_table_s"] = time.time() - t
                    S.plane_read = "basis"; pvb, spb = plane_is_split(S, px, pp, cls_port); S.plane_read = "exact"
                    img2 = img.copy(); img2[pp] = pvb; pred[("C2b", st, v)] = img2; split[("C2b", st, v)] = spb
                    rec["ray_compare"][f"{st}_v{v}"] = ray_compare(S, px, cls_port)
            del S; torch.cuda.empty_cache()
    # metrics
    rep = {st: np.load(rdir / "rep" / f"ref_{st}_v0.npz")["rgb"].astype(np.float64) for st in ("T0", "T1")}
    G0 = np.load(gdir / "geom_T0.npz"); G1 = np.load(gdir / "geom_T1.npz"); conds = ("C0", "C0r", "C1", "C2", "C2b")
    rec["views"] = {}; rec["edit"] = {}
    for v in range(3):
        for st, G in (("T0", G0), ("T1", G1)):
            mk = G[f"v{v}_mk"]; ref = np.load(rdir / f"ref_{st}_v{v}.npz")["rgb"][mk].astype(np.float64)
            val = G[f"v{v}_valid"].astype(bool); plane = val & (G[f"v{v}_dom"] == 2); pidx = np.flatnonzero(plane)
            r = {}
            for c in conds:
                if (c, st, v) not in pred:
                    continue
                p = pred[(c, st, v)].astype(np.float64)
                r[c] = {"plane_psnr": ME.psnr(p[plane], ref[plane]), "plane_lum_ratio": float((p[plane] @ LUM).sum() / (ref[plane] @ LUM).sum()),
                        "lego_psnr": ME.psnr(p[val & (G[f"v{v}_dom"] == 0)], ref[val & (G[f"v{v}_dom"] == 0)])}
                if (c, st, v) in split:
                    sp = split[(c, st, v)] @ LUM                                         # [n plane px, 6]
                    r[c]["plane_class_share"] = {cn: float(sp[:, j].sum() / sp.sum()) for j, cn in enumerate(CLS)}
                    r[c]["plane_class_over_ref"] = {cn: float(sp[:, j].sum() / (ref[pidx] @ LUM).sum()) for j, cn in enumerate(CLS)}
            if ("C0r", st, v) in pred:
                a, b = pred[("C0r", st, v)][plane], pred[("C0", st, v)][plane]; r["baseline_C0r_vs_stored_R6a_max_rel"] = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))
            rec["views"][f"{st}_v{v}"] = r
        em, (i0, i1, dr) = edit_metrics(pred, G0, G1, rdir, rep, conds, v)
        sig = None
        if v == 0:
            e0 = rep["T0"][G0["v0_mk"]][i0] @ LUM - np.load(rdir / "ref_T0_v0.npz")["rgb"][G0["v0_mk"]][i0] @ LUM
            e1 = rep["T1"][G1["v0_mk"]][i1] @ LUM - np.load(rdir / "ref_T1_v0.npz")["rgb"][G1["v0_mk"]][i1] @ LUM
            sig = float(1.4826 * np.median(np.abs(np.concatenate([e0, e1]))))
            rec["sigma_ref_change_v0"] = sig
        s_ = rec.get("sigma_ref_change_v0")
        big = np.abs(dr) > 3 * s_
        for c in conds:
            if c in em:
                dp = pred[(c, "T1", v)][i1] @ LUM - pred[(c, "T0", v)][i0] @ LUM
                em[c]["changed_px_rel_err"] = float(np.sqrt(((dp - dr)[big] ** 2).sum() / max((dr[big] ** 2).sum(), 1e-30)))
                em[c]["changed_px"] = int(big.sum())
        rec["edit"][f"v{v}"] = em
    def imgs(c, key):
        return [rec["views"][f"{st}_v{v}"][c][key] for st in ("T0", "T1") for v in range(3) if c in rec["views"][f"{st}_v{v}"]]
    l1, l2 = imgs("C1", "plane_lum_ratio"), imgs("C2", "plane_lum_ratio"); p1, p2 = imgs("C1", "plane_psnr"), imgs("C2", "plane_psnr")
    crit = {}
    if len(l1) == 6 and len(l2) == 6:
        dl = [abs(a - 1) - abs(b - 1) for a, b in zip(l1, l2)]; dp = [b - a for a, b in zip(p1, p2)]
        crit["basis_major_overbright"] = bool(sum(x >= 0.05 for x in dl) >= 5); crit["lum_closeness_gain"] = dl
        crit["exact_improves_static"] = bool(np.mean(dp) >= 0.2 and sum(x > 0 for x in dp) >= 5); crit["psnr_C2_minus_C1"] = dp
        dc = [rec["edit"][f"v{v}"]["C2"]["corr"] - rec["edit"][f"v{v}"]["C1"]["corr"] for v in range(3)]
        crit["exact_improves_edit"] = bool(sum(x >= 0.05 for x in dc) >= 2); crit["corr_C2_minus_C1"] = dc
    rec["criteria"] = crit; rec["frozen_digest_identical"] = M.digest(KA, KB) == dig0
    rec["cost"] = {"wall_s": time.time() - t00, "peak_GB": torch.cuda.max_memory_allocated() / 1e9}
    (out / "probe_c.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(crit, indent=1))
    np.savez_compressed(out / "probe_c_pred.npz", **{f"{c}_{s}_v{v}": p for (c, s, v), p in pred.items() if c in ("C2", "C2b", "C0r")})


if __name__ == "__main__":
    main(*sys.argv[1:5])
