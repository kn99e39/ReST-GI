"""WL48 Probe B - nonlocal spatial capacity of K44's structure on the Yellow Burley operator (no learned K fitted).

Reference: k_c(x, s) = Burley_c(|x - s|) (worklog 45 A1a lengths, truncation 16 d, r >= 2 mm), receivers = 2000 Yellow
ports, sources = all 852,358 Lego ports (object scope), self port excluded. Approximations (protocol probe_B):
S5 / S30 support-limited exact; Q = K44 band (96 reps, 1.5 cm aggregates) + mid (128 reps, 8 cm cells) with cell-average
ideal weights on the area-mean field; Q+G64 / S30+G64 = + best rank-64 of the residual (Eckart-Young, HS weighting
sqrt(A_x) k sqrt(A_s)); R_r = best rank-r of the full operator. Errors: HS and responses to 16 held-out view sun fields.
K44 comparison: first-event nonlocal (glob + band + mid at the normal) vs s_c * oracle on the same receivers.

    python probe_b48.py <a1a_fit.json> <a3_oracle.json> <out dir>
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
import r1_common as C  # noqa: E402
import fnto_model as M  # noqa: E402

NREC = 2000; CH = 1 << 15; RANKS = [8, 16, 32, 64, 128, 256]
DBINS = [0, 0.01, 0.03, 0.10, 0.30, 10.0]


def burley(r, d):
    return (torch.exp(-r / d) + torch.exp(-r / (3 * d))) / (8 * math.pi * d * r)


def kern(xp, sp, d, self_mask=None):
    r = torch.cdist(xp, sp).clamp(min=2e-3); k = burley(r, d) * (r < 16 * d)
    return k, r


def csr(f2x, n):
    o = torch.argsort(f2x, stable=True); cnt = torch.bincount(f2x, minlength=n); ptr = torch.zeros(n + 1, dtype=torch.long, device=f2x.device); ptr[1:] = torch.cumsum(cnt, 0)
    return o, ptr


@torch.no_grad()
def cell_avg(xp, xid, cells, members, ptr, pos, area, d, lo, hi):
    """Area-weighted mean of k * [lo <= r < hi] over the fine members of cell cells[i, j] for receiver i (self excluded).
    Returns avg [n, J] and member area sums [n, J]."""
    n, J = cells.shape; cc = cells.clamp(min=0); a, b = ptr[cc], ptr[cc + 1]; cnt = (b - a) * (cells >= 0)
    L = int(cnt.max()); avg = torch.zeros(n, J, device=xp.device); asum = torch.zeros(n, J, device=xp.device)
    for j0 in range(0, J, 16):
        sl = slice(j0, j0 + 16); ar = torch.arange(L, device=xp.device)
        idx = a[:, sl, None] + ar; ok = ar < cnt[:, sl, None]; fid = members[idx.clamp(max=len(members) - 1)]
        r = (pos[fid] - xp[:, None, None]).norm(dim=-1).clamp(min=2e-3)
        k = burley(r, d) * (r < 16 * d) * (r >= lo) * (r < hi) * ok * (fid != xid[:, None, None])
        A = area[fid] * ok
        avg[:, sl] = (k * A).sum(-1) / A.sum(-1).clamp(min=1e-20); asum[:, sl] = A.sum(-1)
    return avg, asum


@torch.no_grad()
def main(fit_json, a3_json, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda"); t00 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    fit = json.loads(Path(fit_json).read_text())["fit"]; dch = [fit[c]["d_fit_m"] for c in "RGB"]
    sc = json.loads(Path(a3_json).read_text())["source_sets"]["obj"]["scale_per_channel"]
    A = np.load(C.OUT1 / "anchors.npz"); nA = int(A["n_A"])
    pos = torch.as_tensor(A["pos"][:nA], device=dev).float(); nrm = torch.as_tensor(A["nrm"][:nA], device=dev).float()
    area = torch.as_tensor(A["area"][:nA], device=dev).float(); mat = A["mat"][:nA].astype(np.int64)
    f2b = torch.as_tensor(A["f2b"][:nA].astype(np.int64), device=dev); f2c = torch.as_tensor(A["f2c"][:nA].astype(np.int64), device=dev)
    NB, NC = int(A["bpos"].shape[0]), int(A["cpos"].shape[0])
    bmem, bptr = csr(f2b, NB); cmem, cptr = csr(f2c, NC)
    rec_i = np.sort(np.random.default_rng(4802).choice(np.flatnonzero(mat == 0), NREC, replace=False))
    xi = torch.as_tensor(rec_i, device=dev); xp = pos[xi]; Ax = area[xi]
    nbr = torch.as_tensor(A["nbr"][rec_i].astype(np.int64), device=dev); nbrA = torch.as_tensor(A["nbrA"][rec_i], device=dev).float()
    mid = torch.as_tensor(A["mid"][rec_i].astype(np.int64), device=dev); midA = torch.as_tensor(A["midA"][rec_i], device=dev).float()
    # fields: 16 held-out T0 view sun fields on the Lego ports
    px = np.load(C.OUT1 / "eval_px/heldout_T0.npz"); vis = np.load(C.OUT1 / "eval_vis_k44.npz"); nv = int(px["nv"])
    E = torch.stack([C.E_SUN * (nrm @ torch.as_tensor(px[f"v{v}_light"], device=dev).float()).clamp(min=0)
                     * torch.as_tensor(vis[f"heldout_T0_v{v}_T0"].astype(np.float32)[:nA], device=dev) for v in range(nv)], 1)   # [nA, V]
    AE = area[:, None] * E
    lit_rx = (E[xi] > 0)                                                                    # [NREC, V]
    dl = torch.zeros(NREC, nv, device=dev)
    for v in range(nv):
        lp = pos[E[:, v] > 0]
        dl[:, v] = torch.cat([torch.cdist(xp[s:s + 250], lp).min(1).values for s in range(0, NREC, 250)]) if len(lp) else 1e9
    rec = {"receivers": NREC, "sources": nA, "d_m": dch, "s_c_wl45": sc, "views": nv,
           "support_tables": {"nbrA_sum_median_m2": float(nbrA.clamp(min=0).sum(1).median()), "midA_sum_median_m2": float(midA.clamp(min=0).sum(1).median()),
                              "band_valid_mean": float((nbr >= 0).float().sum(1).mean()), "mid_valid_mean": float((mid >= 0).float().sum(1).mean())},
           "channels": {}}
    ybase = {}
    for ci, d in enumerate(dch):
        tc = time.time()
        # Q coefficients (ideal weights) on aggregates: band (r < 5 cm), mid (r >= 5 cm)
        bavg, basum = cell_avg(xp, xi, nbr, bmem, bptr, pos, area, d, 0.0, C.R_LOCAL)
        cavg, casum = cell_avg(xp, xi, mid, cmem, cptr, pos, area, d, C.R_LOCAL, 1e9)
        Bc = torch.zeros(NREC, NB, device=dev); Mc = torch.zeros(NREC, NC, device=dev)
        Bc.scatter_add_(1, nbr.clamp(min=0), bavg * nbrA * (nbr >= 0) / basum.clamp(min=1e-20))
        Mc.scatter_add_(1, mid.clamp(min=0), cavg * midA * (mid >= 0) / casum.clamp(min=1e-20))
        sx = Ax.sqrt()[:, None]
        G = {k: torch.zeros(NREC, NREC, device=dev, dtype=torch.float64) for k in ("ref", "resQ", "tail30")}
        Y = {k: torch.zeros(NREC, nv, device=dev, dtype=torch.float64) for k in ("ref", "S5", "S30", "Q", "resQ", "tail30")}
        hs = {k: 0.0 for k in ("ref", "S5", "S30", "Q")}
        for s0 in range(0, nA, CH):
            sl = slice(s0, min(s0 + CH, nA)); k, r = kern(xp, pos[sl], d)
            k = k * (xi[:, None] != torch.arange(sl.start, sl.stop, device=dev)[None])
            kU = Bc[:, f2b[sl]] + Mc[:, f2c[sl]]
            w = area[sl].sqrt()[None]
            kt = k * sx * w; Ut = kU * sx * w; t30 = kt * (r >= C.R_MID)
            rq = kt - Ut                                                          # chunk Grams in fp32 (no TF32), accumulated in fp64
            G["ref"] += (kt @ kt.T).double(); G["resQ"] += (rq @ rq.T).double(); G["tail30"] += (t30 @ t30.T).double()
            hs["ref"] += float((kt.double() ** 2).sum()); hs["S5"] += float(((kt * (r >= C.R_LOCAL)).double() ** 2).sum())
            hs["S30"] += float((t30.double() ** 2).sum()); hs["Q"] += float((rq.double() ** 2).sum())
            ae = AE[sl].double()
            Y["ref"] += k.double() @ ae; Y["S5"] += (k * (r < C.R_LOCAL)).double() @ ae; Y["S30"] += (k * (r < C.R_MID)).double() @ ae
            Y["Q"] += kU.double() @ ae; Y["resQ"] += ((k - kU) * sx).double() @ ae; Y["tail30"] += ((k * (r >= C.R_MID)) * sx).double() @ ae
        # rank projections (left singular subspaces of the HS-weighted operators; responses carry the sqrt(A_x) factor)
        ev = {}
        for kname in ("ref", "resQ", "tail30"):
            lam, U = torch.linalg.eigh(G[kname]); ev[kname] = (lam.flip(0).clamp(min=0), U.flip(1))
        def proj(kname, y_sx, r_):                                               # best rank-r response, back to radiance units
            U = ev[kname][1][:, :r_]; return (U @ (U.T @ y_sx)) / sx.double()
        ysx_ref = Y["ref"] * sx.double()
        approx = {"S5": Y["S5"], "S30": Y["S30"], "Q": Y["Q"],
                  "Q_G64": Y["Q"] + proj("resQ", Y["resQ"], 64), "S30_G64": Y["S30"] + proj("tail30", Y["tail30"], 64)}
        for r_ in RANKS:
            approx[f"R{r_}"] = proj("ref", ysx_ref, r_)
        lam_ref = ev["ref"][0]; tot = float(lam_ref.sum())
        hs_rel = {"S5": math.sqrt(hs["S5"] / hs["ref"]), "S30": math.sqrt(hs["S30"] / hs["ref"]), "Q": math.sqrt(hs["Q"] / hs["ref"]),
                  "Q_G64": math.sqrt(float(ev["resQ"][0][64:].sum()) / tot), "S30_G64": math.sqrt(float(ev["tail30"][0][64:].sum()) / tot),
                  **{f"R{r_}": math.sqrt(float(lam_ref[r_:].sum()) / tot) for r_ in RANKS}}
        yr = Y["ref"]; res = {"hs_rel_error": hs_rel, "hs_check_eig_vs_direct": float(tot / hs["ref"]), "field": {}}

        def err(a, m):
            m = m.double(); return float(torch.sqrt((((a - yr) ** 2) * m).sum() / ((yr ** 2) * m).sum().clamp(min=1e-60)))
        for name, a in approx.items():
            f = {"all": err(a, torch.ones_like(yr)), "lit": err(a, lit_rx), "shadow": err(a, ~lit_rx),
                 "shadow_signed_ratio": float((a * (~lit_rx)).sum() / (yr * (~lit_rx)).sum().clamp(min=1e-60))}
            for lo, hi in zip(DBINS[:-1], DBINS[1:]):
                m = (dl >= lo) & (dl < hi) & ~lit_rx
                if m.sum() > 50:
                    f[f"shadow_dist_{lo:.2f}-{hi:.2f}"] = err(a, m)
            res["field"][name] = f
        res["shadow_response_share_by_dist"] = {f"{lo:.2f}-{hi:.2f}": float((yr * ((dl >= lo) & (dl < hi) & ~lit_rx)).sum() / (yr * ~lit_rx).sum()) for lo, hi in zip(DBINS[:-1], DBINS[1:])}
        res["seconds"] = time.time() - tc; rec["channels"]["RGB"[ci]] = res; ybase["RGB"[ci]] = (yr, approx["Q_G64"])
        print("RGB"[ci], json.dumps({"hs": hs_rel, "shadow": {k: v["shadow"] for k, v in res["field"].items()}}), flush=True)
        del G, Bc, Mc; torch.cuda.empty_cache()
    # K44 comparison: first-event nonlocal at the normal on the same receivers / views, vs s_c * oracle
    import r1_solve as SV
    from r1_eval import load_kernels
    KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); dig0 = M.digest(KA, KB); T = SV.Tables(KA, KB, dev)
    H = np.load(C.OUT1 / "rel_T0.npz")["hit"]; K44 = torch.zeros(NREC, nv, 3, device=dev, dtype=torch.float64)
    for v in range(nv):
        S1 = SV.State(T, H, vis[f"heldout_T0_v{v}_T0"].astype(np.float32), px[f"v{v}_light"]); Eind, h = S1.aggregates(); S1.outgoing(Eind, h)
        g, b, m = D.nonlocal_out(T, S1, xi, T.nrm[xi]); K44[:, v] = (g + b + m).double(); del S1
    kc = {}
    for ci, c in enumerate("RGB"):
        yr, yq = ybase[c]; o = sc[ci] * yr; kk = K44[..., ci]; sh = ~lit_rx
        e = lambda a: float(torch.sqrt((((a - o) ** 2) * sh).sum() / ((o ** 2) * sh).sum()))  # noqa: E731
        kc[c] = {"K44_shadow_rel_err_vs_scaled_oracle": e(kk), "Q_G64_shadow_rel_err_scaled": e(sc[ci] * yq), "K44_over_oracle_shadow_sum": float((kk * sh).sum() / (o * sh).sum()),
                 "K44_corr_shadow": float(np.corrcoef(kk[sh].cpu().numpy(), o[sh].cpu().numpy())[0, 1])}
    rec["k44_comparison"] = kc
    R = rec["channels"]["R"]["field"]
    rec["criteria"] = {"support_loss": R["S30"]["shadow"] >= 0.10, "aggregation_loss": (R["Q"]["shadow"] - R["S30"]["shadow"]) >= 0.05,
                       "rank_binding": R["Q_G64"]["shadow"] >= 0.10, "fitting_accounts": kc["R"]["K44_shadow_rel_err_vs_scaled_oracle"] >= 2 * kc["R"]["Q_G64_shadow_rel_err_scaled"],
                       "values_R_shadow": {k: R[k]["shadow"] for k in ("S5", "S30", "Q", "Q_G64", "S30_G64", "R64")}}
    rec["cost"] = {"wall_s": time.time() - t00, "peak_GB": torch.cuda.max_memory_allocated() / 1e9,
                   "storage_per_receiver_floats": {"band": 96 * 19 * 3, "mid": 128 * 3, "glob_rank64": 64 * 3 * 2, "dense_object_scope": nA}}
    rec["frozen_digest_identical"] = M.digest(KA, KB) == dig0
    (out / "probe_b.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec["criteria"], indent=1), json.dumps(kc, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
