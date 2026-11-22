"""WL104K-B shared helpers: authenticated WL104K / WL104K-A inputs, the unchanged ER-v1-SDM response (re-used from WL104K),
single-material objective and peak helpers. Nothing in WL104K / WL104K-A / closure82 / gf104i is modified.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_wl84_104ka"))
import common104ka as KA  # noqa: E402  (WL104K-A: authenticated WL104K inputs)

CM = KA.CM                      # common104k (ER-v1-SDM response, inputs, Dr.Jit evaluator)
FT = KA.FT                      # fit104k (FitData, train = original O0 method)
EV = KA.EV                      # eval104k
IO, GF, CL = CM.IO, CM.GF, CM.CL
DEV = CM.DEV
REPO = CM.REPO
OUT = REPO / "results/ecro_wl84_104kb"
PROTOCOL = HERE / "protocol.json"
P = json.loads(PROTOCOL.read_text(encoding="utf-8")) if PROTOCOL.exists() else {}
LUMA = CM.LUMA
COMP = CM.COMP
NAMES = {"test83_iid_1": "iid_1", "test83_iid_0": "iid_0", "test83_hard_1": "hard_1", "test83_iid_191": "iid_191",
         "test83_iid_159": "iid_159", "test83_block_161": "block_161", "test83_block_186": "block_186",
         "test83_hard_76": "hard_76", "test83_hard_53": "hard_53"}
Stop = IO.Stop
sys.path.insert(0, str(HERE.parent / "ecro_wl84_104d"))
from common104d import DA  # noqa: E402  (frozen layered reference walk, fnto_compiled81 data81.ref_eval)
from data104d import oriented, mirror  # noqa: E402

# Y-parameter groups of ER-v1-GF (gf104i.raw_groups, orthonormal) and the 8 per-component gate groups.
Y_GROUPS = GF.raw_groups()


def authenticate_ka():
    """WL104K-A raw files against evaluation/104K-A/manifest.json (raw_sha256)."""
    man = IO.read(REPO / "evaluation/104K-A/manifest.json")
    raw = man.get("raw_sha256", {})
    seen = {}
    for rel, h in raw.items():
        p = REPO / "results/ecro_wl84_104ka" / rel
        if p.exists() and p.suffix in (".npz", ".json") and "lanes/test83_hard_1_L1" in rel:
            if CM.sha(p) != h:
                raise Stop(f"FROZEN_HASH_MISMATCH: {rel}")
            seen["results/ecro_wl84_104ka/" + rel] = h
    return seen


def sph(theta_deg, phi_deg):
    t, p = np.radians(theta_deg), np.radians(phi_deg)
    return np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1)


def mirror_radius(wi, wo):
    return CM.mirror_radius(wi, wo)


def peak_grid(incidence, radii_deg, n_az=16):
    """Outgoing directions on rings around the mirror direction. incidence [P, 3] -> wo [P, K, 3], radius [K]
    (radius 0 once, then n_az azimuths per ring; the WL104D peak-grid layout when radii = 0.25 .. 4)."""
    rad = [0.0] + [r for r in radii_deg for _ in range(n_az)]
    az = [0.0] + [2 * np.pi * k / n_az for _ in radii_deg for k in range(n_az)]
    rad, az = np.radians(np.array(rad)), np.array(az)
    wo = oriented(mirror(incidence)[:, None, :], rad[None, :], az[None, :])
    return wo, np.degrees(rad)


def sdm_eval_t(y, g, wi, wo):
    """Float64 torch F_SDM for one material: y [58], g [8, 3, 5], wi / wo [L, 3] torch -> [L, 3]."""
    p = GF.decode(y.reshape(1, -1))
    q = {k: v.expand(wi.shape[0], *v.shape[1:]) for k, v in p.items()}
    gg = g.reshape(1, *g.shape[-3:]).expand(wi.shape[0], *g.shape[-3:])
    return CM.eval_sdm(q, gg, wi, wo)


def objective_t(y, g, fit_m):
    """Original training objective (WL82 group_terms metric el1 + ealb, mean over probes) for one material.
    fit_m: {'wi','wo','mean','w'} torch [P, L, ...]."""
    Pn, L = fit_m["w"].shape
    pred = sdm_eval_t(y, g, fit_m["wi"].reshape(-1, 3), fit_m["wo"].reshape(-1, 3))
    m = FT.F.group_terms(pred, fit_m["mean"].reshape(-1, 3), fit_m["w"].reshape(-1), L)["metric"]
    return m.mean()


def to_t(a, dtype=torch.float64):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=DEV)


def write(p, o):
    CM.write(p, o)


def read(p):
    return CM.read(p)


def sha(p):
    return CM.sha(p)


def commit():
    return CM.commit()


# ------------------------------------------------------------------ WL104K-B data, starts and synthetic truth
DATA_DIR = OUT / "primary" / "data"
START_SEEDS = (104911, 104912, 104913, 104914, 104915, 104916, 104917)
START_KEYS = ["generic"] + [str(s) for s in START_SEEDS]


def starts_y(n, keys=None):
    """Matched physical starts in ER-v1-GF raw [R * n, 58] (start-major): generic + 0.5 N(0,1) in ORIGINAL raw space
    (common104i.training_starts convention; seeds 104911-104913 reproduce WL104K starts 2-4)."""
    keys = START_KEYS if keys is None else keys
    g = CM.J.CI.generic(n)
    out = []
    for k in keys:
        z = g if k == "generic" else g + 0.5 * np.random.default_rng(int(k)).standard_normal(g.shape)
        out.append(GF.from_original(z))
    return np.concatenate(out)


def load_lane_set(path, n):
    """WL104J-format lane set (flat [n * P, L, ...]) -> {'wi','wo','w','a','b','mean'} [n, P, L, ...] (common104k.load_set layout)."""
    with np.load(path) as z:
        s = {k: z[k].reshape(n, -1, *z[k].shape[1:]) for k in ("wi", "wo", "use", "q", "a", "b")}
    return {"wi": s["wi"], "wo": s["wo"], "w": s["use"] / s["q"], "a": s["a"], "b": s["b"], "mean": 0.5 * (s["a"] + s["b"])}


def training_set(name, n, data_dir=None):
    if name == "original":
        return CM.load_set("fit_set", n)
    if name == "coverage":
        return load_lane_set(Path(data_dir or DATA_DIR) / "train_coverage.npz", n)
    raise ValueError(name)


def raw_of_alpha(alpha):
    lo, hi = np.log(CL.C.ALPHA_MIN), np.log(CL.C.ALPHA_MAX)
    t = (np.log(alpha) - lo) / (hi - lo)
    return float(np.log(t) - np.log1p(-t))


SP_GATES = {"O0": [0.4, 0.55, 0.8, 1, 1], "O1": [1, 1, 0.85, 0.6, 0.45], "O2": [1, 1, 1, 1, 1], "L": [1, 0.8, 0.6, 0.5, 0.5],
            "I0": [0.5, 0.7, 0.9, 1, 1], "I1": [1, 0.9, 0.7, 0.6, 0.5], "I2": [0.6, 1, 0.6, 1, 0.6], "I3": [1, 1, 1, 1, 1]}


def synthetic_truth(raw_R0, alpha=0.03):
    """Protocol synthetic truth: GF(R0) with outer lobe O0 (GGX index 0) isotropic alpha = 0.03 and the WL104K SP gates."""
    y = GF.from_original(np.asarray(raw_R0, np.float64)).copy()
    y[:, 0] = y[:, 1] = raw_of_alpha(alpha)
    g = np.stack([np.broadcast_to(np.array(SP_GATES[c], float), (3, 5)) for c in COMP])
    return {"y": y, "g": np.broadcast_to(g, (len(y),) + g.shape).copy()}
