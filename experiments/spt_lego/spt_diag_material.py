"""Worklog 39 bounded diagnostic (post-result, descriptive): SPT-1 error per material (WSL RNA venv; B4 owner).

    PYTHONPATH=<rna upstream> python spt_diag_material.py

Material per pixel from the worklog-37 owner (triangle-identified; materials: 0 Yellow (subsurface), 1 Gray,
2 transparent, 3 Black, 4 Red_Glass, 5 Light_brown, 6 Sand, 7 Brown, 8 box B). Display MSE per material on held-out
T0 (S_T0, RNA_T0) and held-out T1 (C_upd, D_stale, C_full, RNA_T1, Surface-Owned A_full-PU / frozen SO_T0-PU), and per
material inside the predeclared A_changed region. RNA / Surface-Owned predictions are worklog-37's stored display
images (eval/views.npz). Does not change the protocol or any arm.
"""

from __future__ import annotations

import json
import sys

import h5py
import numpy as np
import torch

import spt_common as C
import spt_eval as EV
import spt_model as SM

sys.path.insert(0, str(C.ROOT / "experiments/rna_selective_reconstruction_lego"))
import b4_model as E  # noqa: E402

NAMES = ["Yellow", "Gray", "transparent", "Black", "Red_Glass", "Light_brown", "Sand", "Brown", "B"]


def main():
    dev = torch.device("cuda")
    mesh_mat = np.concatenate([np.load(C.ROOT / "results/rna_surface_ownership_lego/mesh/lego_mesh.npz")["material"], np.full(12, 8)])
    m0 = EV.load_model(C.OUT / "SPT_T0/state.pt", dev); m1 = EV.load_model(C.OUT / "SPT_T1/state.pt", dev)
    R = SM.Relations("T0", dev); rel = {"T0": (R.hit.clone(), R.vis.clone())}; R.set_state("T1"); rel["T1"] = (R.hit.clone(), R.vis.clone())
    z = np.load(C.B4 / "eval/views.npz")
    out = {}
    for set_name, state, arms, stored in (("heldout_T0", "T0", {"S_T0": (m0, "T0")}, ["RNA_T0"]),
                                          ("heldout_T1", "T1", {"C_upd": (m0, "T1"), "D_stale": (m0, "T0"), "C_full": (m1, "T1")}, ["RNA_T1", "A_full-PU", "SO_T0-PU"])):
        own = E.make_owner(state)
        f1 = h5py.File(C.B4 / f"data/{set_name}_s1_val.h5", "r"); f2 = h5py.File(C.B4 / f"data/{set_name}_s2_val.h5", "r")
        px = {k: np.load(C.OUT / f"pixels/{set_name}_s1_val.{k}.npy") for k in ("idx", "w", "obj", "start")}
        offs = np.stack(C.offsets(state))
        names = list(arms) + stored
        acc = {n: np.zeros((9, 2)) for n in names}; accc = {n: np.zeros((9, 2)) for n in names}
        with torch.no_grad():
            albs = {n: md.anchor_albedo(R) for n, (md, _) in arms.items()}
            for v in range(16):
                mk = np.nan_to_num(f1["alpha"][v, :, 0].astype(np.float32)) > 0; s, e = px["start"][v], px["start"][v + 1]
                g = lambda f, k: np.nan_to_num(f[k][v].astype(np.float32))[mk]  # noqa: E731
                P, N, Cd, L, vis = g(f1, "position"), g(f1, "normal"), g(f1, "camera_dir"), g(f1, "light_dir"), g(f1, "diffuse_direct")[:, 0] > 0
                _, _, (tri, _, _) = own(P.astype(np.float64), N.astype(np.float64), return_tri=True)
                mat = mesh_mat[tri]
                Nf = np.where(((N * Cd).sum(1) < 0)[:, None], -N, N); tgt = 0.5 * (g(f1, "color") + g(f2, "color")); ref = EV.ldr3(tgt)
                chg = z[f"heldout_T1_v{v}_reg_A_changed"].ravel()[mk] if set_name == "heldout_T1" else np.zeros(len(P), bool)
                T = lambda x, dt=torch.float32: torch.tensor(x, dtype=dt, device=dev)  # noqa: E731
                obj = px["obj"][s:e].astype(np.int64); wl = T(L.mean(0) / np.linalg.norm(L.mean(0)))
                preds = {}
                for n, (md, rs) in arms.items():
                    R.hit, R.vis = rel[rs]
                    alb = md.anchor_albedo(R)
                    pr, _, _, _, _ = EV.render_view(md, R, alb, wl, T(P - offs[obj]), T(obj, torch.int64), T(Nf), T(Cd), T(vis.astype(np.float32)),
                                                     T(px["idx"][s:e], torch.int64), T(px["w"][s:e].astype(np.float32)))
                    preds[n] = EV.ldr3(pr)
                for n in stored:
                    preds[n] = z[f"{set_name}_v{v}_{n}"].astype(np.float32).reshape(-1, 3)[mk]
                for n, pd in preds.items():
                    sq = ((pd - ref) ** 2).mean(1)
                    np.add.at(acc[n][:, 0], mat, sq); np.add.at(acc[n][:, 1], mat, 1)
                    np.add.at(accc[n][:, 0], mat[chg], sq[chg]); np.add.at(accc[n][:, 1], mat[chg], 1)
        out[set_name] = {n: {NAMES[i]: {"mse": float(acc[n][i, 0] / acc[n][i, 1]) if acc[n][i, 1] else None, "px": int(acc[n][i, 1]),
                                        "A_changed_mse": float(accc[n][i, 0] / accc[n][i, 1]) if accc[n][i, 1] else None, "A_changed_px": int(accc[n][i, 1])}
                             for i in range(9)} for n in names}
    (C.OUT / "eval/diag_material.json").write_text(json.dumps(out, indent=1))
    for s_, d in out.items():
        print("==", s_)
        for n, r in d.items():
            print(n.ljust(10), " ".join(f"{k[:6]}:{(v['mse'] or 0):.5f}" for k, v in r.items()))


if __name__ == "__main__":
    main()
