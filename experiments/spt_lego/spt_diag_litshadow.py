"""Worklog 39 bounded diagnostic (post-result, descriptive): where is SPT-1's static error? (Windows venv)

    python spt_diag_litshadow.py

On held-out T0 (16 views) split pixels by the dataset's direct visibility: lit (v = 1: direct + indirect) vs shadowed
(v = 0: indirect only). Display-space MSE for SPT-1 (S_T0, eval path) and for original RNA_T0 (worklog-37 stored
display predictions in eval/views.npz, same reference). Also SPT's direct/indirect split and the reference energy share
of shadowed pixels. Does not change the protocol or any arm.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import torch

import spt_common as C
import spt_eval as EV
import spt_model as SM


def main():
    dev = torch.device("cuda")
    m = EV.load_model(C.OUT / "SPT_T0/state.pt", dev)
    R = SM.Relations("T0", dev)
    z = np.load(C.B4 / "eval/views.npz")
    f1 = h5py.File(C.B4 / "data/heldout_T0_s1_val.h5", "r"); f2 = h5py.File(C.B4 / "data/heldout_T0_s2_val.h5", "r")
    px = {k: np.load(C.OUT / f"pixels/heldout_T0_s1_val.{k}.npy") for k in ("idx", "w", "obj", "start")}
    acc = {k: [0.0, 0] for k in ("spt_lit", "spt_sh", "rna_lit", "rna_sh")}
    eng = [0.0, 0.0]; parts = {"lit_direct": 0.0, "lit_indirect": 0.0, "sh_indirect": 0.0, "lit_ref": 0.0, "sh_ref": 0.0, "nl": 0, "ns": 0}
    with torch.no_grad():
        alb = m.anchor_albedo(R)
        for v in range(16):
            mk = np.nan_to_num(f1["alpha"][v, :, 0].astype(np.float32)) > 0; s, e = px["start"][v], px["start"][v + 1]
            g = lambda f, k: np.nan_to_num(f[k][v].astype(np.float32))[mk]  # noqa: E731
            P, N, Cd, L, vis = g(f1, "position"), g(f1, "normal"), g(f1, "camera_dir"), g(f1, "light_dir"), g(f1, "diffuse_direct")[:, 0] > 0
            N = np.where(((N * Cd).sum(1) < 0)[:, None], -N, N); tgt = 0.5 * (g(f1, "color") + g(f2, "color"))
            T = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt, device=dev)  # noqa: E731
            wl = T(L.mean(0) / np.linalg.norm(L.mean(0)))
            pr, dp, ip, _, _ = EV.render_view(m, R, alb, wl, T(P), T(px["obj"][s:e].astype(np.int64), torch.int64), T(N), T(Cd),
                                              T(vis.astype(np.float32)), T(px["idx"][s:e], torch.int64), T(px["w"][s:e].astype(np.float32)))
            rna = z[f"heldout_T0_v{v}_RNA_T0"].astype(np.float32).reshape(-1, 3)[mk]
            ref = EV.ldr3(tgt)
            for nm, pred_disp in (("spt", EV.ldr3(pr)), ("rna", rna)):
                sq = ((pred_disp - ref) ** 2).mean(1)
                acc[f"{nm}_lit"][0] += float(sq[vis].sum()); acc[f"{nm}_lit"][1] += int(vis.sum())
                acc[f"{nm}_sh"][0] += float(sq[~vis].sum()); acc[f"{nm}_sh"][1] += int((~vis).sum())
            eng[0] += float(tgt[vis].sum()); eng[1] += float(tgt[~vis].sum())
            parts["lit_direct"] += float(dp[vis].sum()); parts["lit_indirect"] += float(ip[vis].sum()); parts["sh_indirect"] += float(ip[~vis].sum())
            parts["lit_ref"] += float(tgt[vis].sum()); parts["sh_ref"] += float(tgt[~vis].sum()); parts["nl"] += int(vis.sum()); parts["ns"] += int((~vis).sum())
    out = {k: {"mse": a / n, "psnr": float(10 * np.log10(n / a)), "px": n} for k, (a, n) in acc.items()}
    out["reference_energy_share_shadowed"] = eng[1] / sum(eng)
    out["mean_radiance"] = {"lit_ref": parts["lit_ref"] / parts["nl"] / 3, "lit_spt_direct": parts["lit_direct"] / parts["nl"] / 3,
                            "lit_spt_indirect": parts["lit_indirect"] / parts["nl"] / 3, "shadow_ref": parts["sh_ref"] / parts["ns"] / 3,
                            "shadow_spt_indirect": parts["sh_indirect"] / parts["ns"] / 3}
    (C.OUT / "eval/diag_litshadow_T0.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
