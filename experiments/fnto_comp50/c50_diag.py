"""WL50 Gate 3 divergence attribution (amendment 1; added after the L0C solves diverged; no criterion changed).

For H / L0C / L1C (seed 5011): homogeneous power iteration of the composed map (no sun; production step; 40 iterations,
renormalised), per-channel growth rates, and in the final (dominant) mode the share of outgoing radiance by receiver class
(Yellow / opaque / glass / box) and, at Yellow ports, by term (K_loc, K_glob, K44 band + mid, arm). Also the gathered
irradiance share that Yellow ports receive from Yellow senders (Yellow -> Yellow coupling through C).

    python c50_diag.py <out dir> <L0 ckpt> <L1 ckpt>
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
import c50_common as CC  # noqa: E402
import r1_common as C  # noqa: E402


@torch.no_grad()
def terms(S, Eind, h):
    T = S.T; Etot = Eind
    loc = T.rho / math.pi * Etot + T.kap.sum(1) / math.pi * Eind
    glob = torch.zeros_like(loc)
    for o in (0, 1):
        ii = T.dom[o]; g = (T.area[ii][:, None, None] * T.q[o] * Etot[ii][:, None, :]).sum(0); glob[ii] = (T.p[o] * g[None]).sum(1)
    Ec = torch.zeros(T.Nc, 3, device=T.dev).index_add_(0, T.f2c, Etot * T.area[:, None]) / T.carea[:, None].clamp(min=1e-12)
    hB = torch.zeros(T.Nb, C.N_BINS_H, 3, device=T.dev).index_add_(0, T.f2b, h * T.area[:, None, None]) / T.barea[:, None, None].clamp(min=1e-12)
    bm = torch.zeros_like(loc)
    for s in range(0, T.N, 1 << 15):
        sl = slice(s, s + (1 << 15)); bm[sl] = S.keep[sl, None] * (T.Wm[sl].float() * Ec[T.mid[sl].clamp(min=0)]).sum(1)
    bm += S.keep[:, None] * T.band_sender(hB)
    arm = torch.zeros_like(loc)
    if CC.ARM is not None:
        nA = int(T.dom[0].numel()); arm[CC.ARM.ports] = CC.ARM.apply(Etot[:nA])
    return {"K_loc": loc, "K_glob": glob, "K44_band_mid": bm, "arm": arm}


@torch.no_grad()
def main(out, ck0, ck1):
    from r1_eval import load_kernels
    out = Path(out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda"); t00 = time.time()
    KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); T = CC.SV.Tables(KA, KB, dev); geo = CC.Geo(dev); yp = CC.yellow_ports()
    H = np.load(C.OUT1 / "rel_T0.npz")["hit"]
    A = np.load(C.OUT1 / "anchors.npz"); mat = np.zeros(T.N, np.int64); mat[:len(A["mat"])] = A["mat"]; obj = np.zeros(T.N, np.int64); obj[:len(A["obj"])] = A["obj"]
    cls = np.select([obj == 1, mat == 0, np.isin(mat, [2, 4])], [3, 0, 2], 1); clsT = torch.as_tensor(cls, device=dev); names = ["Yellow", "opaque", "glass", "box"]
    rec = {}
    for cn, arm in (("H", None), ("L0C", CC.ArmOp("L0", CC.load_net(ck0, dev), geo, yp)), ("L1C", CC.ArmOp("L1", CC.load_net(ck1, dev), geo, yp))):
        CC.activate(arm)
        S = CC.SV.State(T, H, np.zeros(T.N, np.float32), np.array([0.0, 0.0, 1.0]))
        g = torch.Generator(device=dev).manual_seed(0); S.L = (torch.rand(S.L.shape, generator=g, device=dev, dtype=torch.float16) * S.valid[..., None]).half()
        rates, rch = [], []
        for _ in range(40):
            n0 = S.L.float().norm(dim=(0, 1)); S.step(sun_on=False); n1 = S.L.float().norm(dim=(0, 1))
            if not torch.isfinite(n1).all() or float(n1.sum()) == 0:
                break
            rch.append((n1 / n0).tolist()); rates.append(float(n1.norm() / n0.norm())); S.L = (S.L.float() * (1e2 / n1.max())).half()
        Eind, h = S.aggregates(); tm = terms(S, Eind, h); I = sum(tm.values())
        lumI = I @ torch.tensor([0.2126, 0.7152, 0.0722], device=dev)
        Ar = T.area; r = {"rho_median_last10": float(np.median(rates[-10:])), "rho_per_channel_last10_median": np.median(np.array(rch[-10:]), 0).tolist(),
                          "class_share_of_outgoing_power": {nm: float((lumI.abs() * Ar)[clsT == k].sum() / (lumI.abs() * Ar).sum()) for k, nm in enumerate(names)}}
        ym = clsT == 0
        r["yellow_term_share_RGB"] = {k: ((v[ym] * Ar[ym, None]).sum(0) / (I[ym] * Ar[ym, None]).sum(0).clamp(min=1e-30)).tolist() for k, v in tm.items()}
        hit = torch.as_tensor(H, device=dev); hc = hit.clamp(min=0); ok = hit >= 0
        Ly = S.L.float().sum(-1) * ok
        from_y = (Ly * (clsT[hc] == 0)).sum(1); tot = Ly.sum(1).clamp(min=1e-30)
        r["yellow_receivers_gathered_share_from_yellow"] = float((from_y[ym] * Ar[ym]).sum() / (tot[ym] * Ar[ym]).sum())
        rec[cn] = r; print(cn, json.dumps(r), flush=True); del S; torch.cuda.empty_cache()
    CC.deactivate(); rec["cost"] = {"wall_s": time.time() - t00, "peak_GB": torch.cuda.max_memory_allocated() / 1e9}
    (out / "diag.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
