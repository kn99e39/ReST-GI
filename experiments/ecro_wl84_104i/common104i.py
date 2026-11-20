"""WL104I shared inputs, training starts and helpers (reuses the WL104H authenticated inputs)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_wl84_104h"))
sys.path.insert(0, str(HERE.parent / "ecro_wl84_104d"))
import io104h as IO  # noqa: E402
import gf104i as GF  # noqa: E402
from common104d import historical_states  # noqa: E402

P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
REPO = IO.REPO
TRAIN = REPO / "results/ecro_wl84_104f/inputs/train_original.npz"
OUT = REPO / "results/ecro_wl84_104i"
K, CL, CD, H = GF.K, GF.CL, GF.CD, GF.H
DEV = H.dev
SEEDS = tuple(P["stage_D_training"]["primary_seed_set"])
LUMA = np.array([0.2126, 0.7152, 0.0722])


def inputs():
    rows, raw, compiled, seen = IO.inputs()
    raw["ER_L"] = historical_states(IO.WL83_FITS, rows)["ER_L"].astype(np.float64)
    prov = IO.read(REPO / "evaluation/104D/amendment1/provenance.json")
    want = prov["parent_output_sha256"]["train_original.npz"]
    got = IO.sha(TRAIN)
    if got != want:
        raise IO.Stop(f"FROZEN_HASH_MISMATCH: {TRAIN}")
    seen["results/ecro_wl84_104f/inputs/train_original.npz"] = got
    return rows, raw, compiled, seen


def generic(n):
    return CL.init_raw("ER", "cert", n, DEV).double().cpu().numpy()


def training_starts(n):
    """{'generic': [n,64]} + {seed: generic + 0.5 N(0,1)} in ORIGINAL raw space (float64)."""
    g = generic(n)
    out = {"generic": g}
    for s in SEEDS:
        out[str(s)] = g + 0.5 * np.random.default_rng(s).standard_normal(g.shape)
    return out


def from_original_t(z):
    """Torch (differentiable) version of GF.from_original for one or many states [B, 64]."""
    B = z.shape[0]
    lam9 = torch.softmax(z[:, 21:48].reshape(B, 3, 9), -1)
    s = torch.sigmoid(z[:, 48:63].reshape(B, 3, 5))
    outer = lam9[..., 1:4]
    blk = 1.0 - outer.sum(-1)
    T = lam9[..., 0] + lam9[..., 4:8].sum(-1)
    a = torch.log(outer) - torch.log(blk)[..., None]
    b = torch.log(lam9[..., 4:8]) - torch.log(lam9[..., 0:1])
    sp = s * torch.sqrt(T / blk)[..., None]
    k = torch.log(sp) - torch.log1p(-sp)
    return torch.cat([z[:, :21], a.reshape(B, 9), b.reshape(B, 12), k.reshape(B, 15), z[:, 63:64]], 1)


def f_original(wi, wo):
    def f(z):
        return K.native_torch(CL.decode("ER", "cert", z[None]), wi, wo).reshape(-1)
    return f


def f_candidate(wi, wo):
    def f(y):
        return K.native_torch(GF.decode(y[None]), wi, wo).reshape(-1)
    return f


def drjit_eval(state, wi, wo):
    return H.er_eval_dr(np.asarray(state, np.float32), wi, wo)
