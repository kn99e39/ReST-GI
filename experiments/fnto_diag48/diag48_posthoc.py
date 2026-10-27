"""WL48 post-hoc diagnostics (amendment 2; after Probe A / B results, labelled post hoc, no criterion changed).

P1  Reference-semantics energy: the worklog-45 object-scoped Burley oracle with its fitted per-channel scales s_c (which
    reproduce the first_r1 references, WL45 A3) implies a Yellow first-event exitance; its flux pi * s_c * y * A summed over
    Yellow ports (random 20,000-receiver subsample, WL45 semantics: all Lego sources, r >= 2 mm, truncation 16 d) is
    compared with the absorbed Lego sun power - the same ratio Probe A measured for K44 (views 6, 15).
P2  Probe B's K44 comparison repeated with K44's nonlocal output averaged over 64 cosine directions (hemispherical mean)
    instead of the port normal (Probe A found the normal value 5-9x below every other queried direction).

    python diag48_posthoc.py <a1a_fit.json> <a3_oracle.json> <out dir>
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "fnto_dir46"))
import dir46_core as D  # noqa: E402
import r1_common as C  # noqa: E402
import fnto_model as M  # noqa: E402
from probe_b48 import burley  # noqa: E402


@torch.no_grad()
def oracle_y(xp, pos, AE, d, self_idx=None):
    out = torch.zeros(len(xp), AE.shape[1], device=xp.device, dtype=torch.float64); ch = max(1024, int(4e7 // len(xp)))
    for s0 in range(0, len(pos), ch):
        r = torch.cdist(xp, pos[s0:s0 + ch]).clamp(min=2e-3); k = burley(r, d) * (r < 16 * d)
        if self_idx is not None:
            k = k * (self_idx[:, None] != torch.arange(s0, min(s0 + ch, len(pos)), device=xp.device)[None])
        out += k.double() @ AE[s0:s0 + ch].double()
    return out


@torch.no_grad()
def main(fit_json, a3_json, out):
    import r1_solve as SV
    from r1_eval import load_kernels
    out = Path(out); out.mkdir(parents=True, exist_ok=True); dev = torch.device("cuda"); t00 = time.time()
    dch = [json.loads(Path(fit_json).read_text())["fit"][c]["d_fit_m"] for c in "RGB"]
    sc = json.loads(Path(a3_json).read_text())["source_sets"]["obj"]["scale_per_channel"]
    A = np.load(C.OUT1 / "anchors.npz"); nA = int(A["n_A"]); mat = A["mat"][:nA].astype(np.int64)
    pos = torch.as_tensor(A["pos"][:nA], device=dev).float(); nrm = torch.as_tensor(A["nrm"][:nA], device=dev).float(); area = torch.as_tensor(A["area"][:nA], device=dev).float()
    px = np.load(C.OUT1 / "eval_px/heldout_T0.npz"); vis = np.load(C.OUT1 / "eval_vis_k44.npz")
    yel = np.flatnonzero(mat == 0); rec = {"P1_reference_energy": {}, "P2_k44_cosmean_vs_oracle": {}}
    KA, KB = load_kernels(str(C.OUT1 / "K_k44P/kernels.pt"), dev); dig0 = M.digest(KA, KB); T = SV.Tables(KA, KB, dev)
    H = np.load(C.OUT1 / "rel_T0.npz")["hit"]
    sub = torch.as_tensor(np.sort(np.random.default_rng(4805).choice(yel, 20000, replace=False)), device=dev)
    scale = float(area[torch.as_tensor(yel, device=dev)].sum() / area[sub].sum())             # area-ratio extrapolation to all Yellow ports
    for v in (6, 15):
        E = C.E_SUN * (nrm @ torch.as_tensor(px[f"v{v}_light"], device=dev).float()).clamp(min=0) * torch.as_tensor(vis[f"heldout_T0_v{v}_T0"].astype(np.float32)[:nA], device=dev)
        AE = (area * E)[:, None]; Pin = float(AE.sum())
        flux = []
        for ci, d in enumerate(dch):
            y = oracle_y(pos[sub], pos, AE, d)[:, 0]
            flux.append(float((math.pi * sc[ci] * y * area[sub].double()).sum()) * scale / Pin)
        rec["P1_reference_energy"][f"view{v}"] = {"oracle_yellow_exitance_flux_over_absorbed_RGB": flux, "absorbed_sun_power": Pin}
        print("P1", v, flux, flush=True)
    # P2: Probe B receivers (seed 4802), 16 views, shadow receivers
    rec_i = np.sort(np.random.default_rng(4802).choice(yel, 2000, replace=False)); xi = torch.as_tensor(rec_i, device=dev)
    Wc = D.cosine_dirs(T.nrm[xi], 64); nv = int(px["nv"])
    kc = torch.zeros(2000, nv, 3, device=dev, dtype=torch.float64); kn = torch.zeros_like(kc); yo = torch.zeros_like(kc); lit = torch.zeros(2000, nv, dtype=torch.bool, device=dev)
    for v in range(nv):
        vf = vis[f"heldout_T0_v{v}_T0"].astype(np.float32); light = px[f"v{v}_light"]
        S1 = SV.State(T, H, vf, light); Eind, h = S1.aggregates(); S1.outgoing(Eind, h)
        q2 = xi[:, None].expand(2000, 64).reshape(-1); g, b, m = D.nonlocal_out(T, S1, q2, Wc.reshape(-1, 3))
        kc[:, v] = (g + b + m).view(2000, 64, 3).mean(1).double(); g, b, m = D.nonlocal_out(T, S1, xi, T.nrm[xi]); kn[:, v] = (g + b + m).double()
        E = S1.Esun[:nA].clone(); AE = (area * E)[:, None]; lit[:, v] = E[xi] > 0
        del S1, g, b, m, q2; torch.cuda.empty_cache()
        for ci, d in enumerate(dch):
            yo[:, v, ci] = sc[ci] * oracle_y(pos[xi], pos, AE, d, self_idx=xi)[:, 0]
    sh = ~lit
    for ci, c in enumerate("RGB"):
        o = yo[..., ci]
        def e(a):
            return float(torch.sqrt((((a - o) ** 2) * sh).sum() / ((o ** 2) * sh).sum()))
        rec["P2_k44_cosmean_vs_oracle"][c] = {"cosmean_rel_err": e(kc[..., ci]), "cosmean_over_oracle_sum": float((kc[..., ci] * sh).sum() / (o * sh).sum()),
                                              "cosmean_corr": float(np.corrcoef(kc[..., ci][sh].cpu().numpy(), o[sh].cpu().numpy())[0, 1]),
                                              "normal_rel_err": e(kn[..., ci]), "normal_over_oracle_sum": float((kn[..., ci] * sh).sum() / (o * sh).sum()),
                                              "lit_cosmean_over_oracle_sum": float((kc[..., ci] * lit).sum() / (o * lit).sum())}
    rec["frozen_digest_identical"] = M.digest(KA, KB) == dig0; rec["cost"] = {"wall_s": time.time() - t00, "peak_GB": torch.cuda.max_memory_allocated() / 1e9}
    (out / "posthoc.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
