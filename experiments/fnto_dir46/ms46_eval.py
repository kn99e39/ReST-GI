"""WL46 MS composition and evaluation (torch, RNA venv): frozen K44 on the Lego / box + an ANALYTIC glossy-plane domain
(rho 0, one GGX lobe kappa 0.9, alpha 0.04: the Cycles Glossy BSDF of the reference; an analytic control domain, labelled).

The plane domain's state: Lm [M, 128, 3] incoming radiance along its 128 cosine gathers (dm); its sender value toward w:
  'iso'  production semantics: kappa / pi E_m (the isotropic collapse of an indirect glossy lobe) + its sun lobe (0 here)
  'dir'  sum_j kappa GGX(n, dm_j, w) Lm_j pi / 128 (directional, from its own 128 cosine samples)
One Jacobi iteration: K tables from the current state -> new K-gather values (DirState.k_rays; gathers that hit the plane
read the plane sender toward -d) and new plane incoming (K sender values of the hit ports toward -d + their sun lobe).
The sun is fixed (one sun for all views), so one solve per (state, sender mode) serves every view.

Plane pixels toward the camera:
  'cos'  production pixel semantics: the BSDF over the pixel port's 128 cosine gathers (Lm)
  'is'   128 GGX importance-sampled rays from the pixel point (ms46_prep), reading the hit port's sender value
Conditions (protocol MS): R0m = (K 'n' / 'E', plane 'iso', 'cos'); R3m = (K 'basis' / 'E', plane 'dir', 'cos');
R6a = R0m senders with 'is'; R6b = R3m senders with 'is'. Lego / box pixels: r1_solve.render_pixels (production).

    python ms46_eval.py <geom dir> <refs dir> <out dir>
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dir46_solve as DS  # noqa: E402
import r1_common as C  # noqa: E402
import fnto_model as M  # noqa: E402

KAP, ALPHA = 0.9, 0.04; NP_ = torch.tensor([-1.0, 0.0, 0.0]); SUN = np.array([0.2, -0.5, 0.84]) / np.linalg.norm([0.2, -0.5, 0.84])
LUM = np.array([0.2126, 0.7152, 0.0722])


def ggx_plane(wi, wo):
    n = NP_.to(wi.device).expand_as(wi)
    return M.ggx(n, wi, wo, torch.full(wi.shape[:-1] + (1,), ALPHA, device=wi.device))[..., 0]


class MSState(DS.DirState):
    def __init__(self, T, G, plane_mode, plane_read="n", **mode):
        H = G["H"]; self.Nk = T.N; Hk = np.where(H >= T.N, -1, H)
        super().__init__(T, Hk, G["sun"], SUN, **mode)
        dev = T.dev; self.plane_mode = plane_mode; self.plane_read = plane_read
        rows, cols = np.nonzero(H >= T.N)
        self.mp = torch.as_tensor(rows, device=dev); self.mk = torch.as_tensor(cols, device=dev); self.mm = torch.as_tensor(H[rows, cols] - T.N, device=dev, dtype=torch.long)
        self.Hm = torch.as_tensor(G["Hm"].astype(np.int64), device=dev); self.dm = torch.as_tensor(G["dm"], device=dev)
        self.Lm = torch.zeros(*self.Hm.shape, 3, device=dev)
        assert float((NP_.numpy() @ SUN)) <= 0                                   # the plane front is not sun-lit: no plane sun lobe

    @torch.no_grad()
    def plane_out(self, m, w):
        if self.plane_mode == "iso":
            E = self.Lm[m].sum(1) * (math.pi / C.K_GATHER); return KAP / math.pi * E
        out = torch.empty(len(m), 3, device=self.T.dev)
        for s in range(0, len(m), 1 << 13):
            mm, ww = m[s:s + (1 << 13)], w[s:s + (1 << 13)]
            f = ggx_plane(self.dm[mm], ww[:, None].expand_as(self.dm[mm]))           # [n, 128]
            out[s:s + (1 << 13)] = KAP * (f[..., None] * self.Lm[mm]).sum(1) * (math.pi / C.K_GATHER)
        return out

    @torch.no_grad()
    def prepare(self, sun_on=True):
        super().prepare(sun_on)
        if self.plane_read == "basis" and self.O is None:                        # amendment 1: directional reads into the plane only
            self.O = self.basis_table()

    @torch.no_grad()
    def k_sender(self, q, d):
        """K sender value incl. the sun lobe, for K ports q read along d by the PLANE (gathers / pixels). plane_read 'n' = the
        K solve's own semantics; 'basis' = directional (19-bin) nonlocal value toward -d with the solve's local part."""
        if self.plane_read == "basis" and self.nl == "n":
            nq = self.T.nrm[q]; w = DS.upper(nq, -d); td, bd = self.T.detf[0][q], self.T.detf[1][q]
            b = M.hemi_bin(torch.stack([(w * td).sum(-1), (w * bd).sum(-1), (w * nq).sum(-1)], -1))
            loc = self.loc_iso[q] if self.loc_in == "E" else self.ray_values(q, d) - self.NL_n[q]
            return self.O[q, b.long()] + loc + self.sun_lobe(q, -d)
        return self.ray_values(q, d) + self.sun_lobe(q, -d)

    @torch.no_grad()
    def step(self, sun_on=True):
        assert sun_on
        self.prepare(True)
        ok = self.Hm >= 0; Lm_new = torch.zeros_like(self.Lm)
        q = self.Hm[ok]; d = self.dm[ok]; Lm_new[ok] = torch.cat([self.k_sender(q[s:s + (1 << 15)], d[s:s + (1 << 15)]) for s in range(0, len(q), 1 << 15)])
        pv = torch.cat([self.plane_out(self.mm[s:s + (1 << 16)], -self.T.dirs[self.mp[s:s + (1 << 16)], self.mk[s:s + (1 << 16)]].float())
                        for s in range(0, len(self.mm), 1 << 16)]) if len(self.mm) else None
        Ln, dn2, nn2 = self.k_rays(True)
        if pv is not None:
            old = self.L[self.mp, self.mk].float(); Ln[self.mp, self.mk] = pv.half()
            # k_rays counted these rays with the placeholder value 0 (Hk = -1): replace (0 - old)^2 by (pv - old)^2
            dn2 += float((pv - old).square().sum()) - float(old.square().sum()); nn2 += float(pv.square().sum())
        dn2 += float((Lm_new - self.Lm).square().sum()); nn2 += float(Lm_new.square().sum())
        self.L = Ln; self.Lm = Lm_new
        return math.sqrt(dn2), math.sqrt(nn2)


@torch.no_grad()
def plane_pixels(S, px, idx, sampling):
    """Radiance of plane pixels idx toward the camera."""
    T = S.T; dev = T.dev; cam = torch.as_tensor(px["cam"][idx], device=dev)
    if sampling == "cos":
        m = torch.as_tensor(px["port"][idx].astype(np.int64) - T.N, device=dev)
        f = ggx_plane(S.dm[m], cam[:, None].expand_as(S.dm[m]))
        return (KAP * (f[..., None] * S.Lm[m]).sum(1) * (math.pi / C.K_GATHER)).cpu().numpy()
    pidx = np.searchsorted(np.flatnonzero(px["dom"] == 2), idx)               # row of each plane pixel in the IS arrays
    wi = torch.as_tensor(px["is_dir"][pidx], device=dev); ww = torch.as_tensor(px["is_w"][pidx], device=dev); hit = torch.as_tensor(px["is_hit"][pidx].astype(np.int64), device=dev)
    L = torch.zeros(*hit.shape, 3, device=dev); ok = hit >= 0
    kq = ok & (hit < T.N); pq = ok & (hit >= T.N)
    if kq.any():
        L[kq] = S.k_sender(hit[kq], wi[kq])
    if pq.any():
        L[pq] = S.plane_out(hit[pq] - T.N, -wi[pq])
    f = ggx_plane(wi, cam[:, None].expand_as(wi))
    return (KAP * (f[..., None] * L * ww[..., None]).sum(1)).cpu().numpy()


def psnr(p, r):
    return float(10 * np.log10(1 / max(((C.ldr3(p) - C.ldr3(r)) ** 2).mean(), 1e-30)))


@torch.no_grad()
def main(gdir, rdir, out):
    import r1_solve as SV
    from r1_eval import load_kernels
    gdir, rdir, out = Path(gdir), Path(rdir), Path(out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda")
    KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); dig0 = M.digest(KA, KB); T = SV.Tables(KA, KB, dev)
    senders = {"iso": dict(plane_mode="iso", nl="n", loc_in="E"), "dir": dict(plane_mode="dir", nl="basis", loc_in="E"),
               "dirread": dict(plane_mode="dir", nl="n", loc_in="E", plane_read="basis")}
    cond = {"R0m": ("iso", "cos"), "R3m": ("dir", "cos"), "R6a": ("iso", "is"), "R6b": ("dir", "is"), "R3mr": ("dirread", "cos"), "R6br": ("dirread", "is")}
    pred = {}; rec = {"solves": {}, "views": {}, "conditions": cond}
    for state in ("T0", "T1"):
        G = dict(np.load(gdir / f"geom_{state}.npz"))
        for sname, sm in senders.items():
            t = time.time(); S = MSState(T, G, **sm).solve(); torch.cuda.synchronize()
            rec["solves"][f"{state}_{sname}"] = {"status": S.status, "iters": S.iters, "seconds": time.time() - t, "timing": S.timing, "n_below": S.n_below,
                                                  "rays_to_plane": int(len(S.mm)), "peak_GB": torch.cuda.max_memory_allocated() / 1e9,
                                                  "hist": [float(x) for x in S.hist], "ratio": [float(x) for x in S.ratio]}
            print(state, sname, rec["solves"][f"{state}_{sname}"], flush=True)
            if S.status != "converged":
                del S; torch.cuda.empty_cache(); continue
            for v in range(3):
                px = {k[len(f"v{v}_"):]: G[k] for k in G if k.startswith(f"v{v}_")}
                val = px["valid"].astype(bool); kp = np.flatnonzero(val & (px["dom"] < 2)); pp = np.flatnonzero(val & (px["dom"] == 2))
                pxd = {k: torch.as_tensor(px[k][kp].astype(np.float32 if k in ("pos", "nrm", "cam", "vis") else np.int64), device=dev) for k in ("pos", "nrm", "cam", "vis", "dom", "port")}
                kpix = SV.render_pixels(T, S, pxd)[0].cpu().numpy()
                for cname, (sn, smp) in cond.items():
                    if sn != sname:
                        continue
                    img = np.zeros((len(px["mk"]), 3), np.float32); img[kp] = kpix; img[pp] = plane_pixels(S, px, pp, smp)
                    pred[(cname, state, v)] = img
            del S; torch.cuda.empty_cache()
    # metrics
    rep = {st: np.load(rdir / "rep" / f"ref_{st}_v0.npz")["rgb"] for st in ("T0", "T1")}
    G0 = np.load(gdir / "geom_T0.npz"); G1 = np.load(gdir / "geom_T1.npz")
    for v in range(3):
        for state, G in (("T0", G0), ("T1", G1)):
            mk = G[f"v{v}_mk"]; ref = np.load(rdir / f"ref_{state}_v{v}.npz")["rgb"][mk].astype(np.float64)
            val = G[f"v{v}_valid"].astype(bool); dom = G[f"v{v}_dom"]; mat = G[f"v{v}_mat"]
            roi = {"plane": val & (dom == 2), "lego": val & (dom == 0), "box": val & (dom == 1), "yellow": val & (mat == 0), "all": val}
            r = {}
            for cname in cond:
                if (cname, state, v) not in pred:
                    continue
                p = pred[(cname, state, v)].astype(np.float64)
                r[cname] = {k: {"px": int(m.sum()), "psnr": psnr(p[m], ref[m]), "lum_ratio": float((p[m] @ LUM).sum() / max((ref[m] @ LUM).sum(), 1e-30))} for k, m in roi.items() if m.sum() > 50}
            if v == 0:
                rr = rep[state][mk].astype(np.float64); r["reference_replicate_plane_psnr"] = psnr(rr[roi["plane"]], ref[roi["plane"]])
            rec["views"][f"{state}_v{v}"] = r
        # edit on plane pixels present in both states (same image pixel)
        a0 = {int(k): i for i, k in enumerate(G0[f"v{v}_mk"])}; m1 = G1[f"v{v}_mk"]
        both = [(a0[int(k)], j) for j, k in enumerate(m1) if int(k) in a0 and G0[f"v{v}_dom"][a0[int(k)]] == 2 and G1[f"v{v}_dom"][j] == 2]
        if both:
            i0, i1 = map(np.array, zip(*both))
            r0 = np.load(rdir / f"ref_T0_v{v}.npz")["rgb"][G0[f"v{v}_mk"]][i0] @ LUM; r1 = np.load(rdir / f"ref_T1_v{v}.npz")["rgb"][m1][i1] @ LUM; dr = r1 - r0
            sig = None
            if v == 0:
                e0 = (rep["T0"][G0["v0_mk"]][i0] @ LUM - r0); e1 = (rep["T1"][m1][i1] @ LUM - r1); sig = float(1.4826 * np.median(np.abs(np.concatenate([e0, e1]))) / math.sqrt(2) * math.sqrt(2))
            rec["views"][f"edit_v{v}"] = {"px": int(len(i0)), "ref_change_rms": float(np.sqrt((dr ** 2).mean())), "sigma_ref_change": sig}
            for cname in cond:
                if (cname, "T1", v) not in pred or (cname, "T0", v) not in pred:
                    continue
                dp = pred[(cname, "T1", v)][i1] @ LUM - pred[(cname, "T0", v)][i0] @ LUM
                big = np.abs(dr) > 3 * (sig if sig else 0)
                rec["views"][f"edit_v{v}"][cname] = {"corr": float(np.corrcoef(dp, dr)[0, 1]), "slope": float((dp * dr).sum() / max((dr * dr).sum(), 1e-30)),
                                                     "explained": float(1 - ((dp - dr) ** 2).sum() / max((dr * dr).sum(), 1e-30)), "sign_agree_gt3sigma": float((np.sign(dp[big]) == np.sign(dr[big])).mean()) if big.any() else None}
    # summary: plane ROI mean gains, 2 x 2 interaction
    def roi_mean(c, k="plane"):
        xs = [rec["views"][f"{st}_v{v}"][c][k]["psnr"] for st in ("T0", "T1") for v in range(3) if c in rec["views"][f"{st}_v{v}"] and k in rec["views"][f"{st}_v{v}"][c]]
        return (float(np.mean(xs)), xs) if len(xs) == 6 else (None, xs)
    summ = {}
    for k in ("plane", "lego", "box", "yellow", "all"):
        mm = {c: roi_mean(c, k) for c in cond}
        summ[k] = {c: mm[c][0] for c in cond}
        for tag, (a1b1, a1b0, a0b1, a0b0) in (("full", ("R6b", "R3m", "R6a", "R0m")), ("read", ("R6br", "R3mr", "R6a", "R0m"))):
            if all(mm[c][0] is not None for c in (a1b1, a1b0, a0b1, a0b0)):
                summ[k][f"interaction_sender_x_sampling_{tag}"] = mm[a1b1][0] - mm[a1b0][0] - mm[a0b1][0] + mm[a0b0][0]
        for a_, b_ in (("R3m", "R0m"), ("R6a", "R0m"), ("R6b", "R6a"), ("R6b", "R0m"), ("R3mr", "R0m"), ("R6br", "R6a"), ("R6br", "R0m")):
            if mm[a_][0] is None or mm[b_][0] is None:
                summ[k][f"{a_}-{b_}"] = None; continue
            dif = np.array(mm[a_][1]) - np.array(mm[b_][1])
            summ[k][f"{a_}-{b_}"] = {"mean": float(dif.mean()), "n_pos": int((dif > 0).sum()), "n": int(len(dif)),
                                     "improves(>0.2dB & >=5/6)": bool(dif.mean() > 0.2 and (dif > 0).sum() >= 5) if k == "plane" else None}
    rec["summary_psnr"] = summ; rec["frozen_digest_identical"] = M.digest(KA, KB) == dig0
    (out / "ms46_eval.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(summ, indent=1), flush=True)
    np.savez_compressed(out / "ms46_pred.npz", **{f"{c}_{s}_v{v}": p for (c, s, v), p in pred.items()})


if __name__ == "__main__":
    main(*sys.argv[1:4])
