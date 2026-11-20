"""ER-v1-GF: gauge-fixed 58-parameter parameterization of the certified ER family (WL104I candidate; not adopted).

Derivation and limitations: docs/plans/ER_V1_GF_GAUGE_FIXED_PARAMETERIZATION.md.
Raw layout y [B, 58]: y[0:21] lobe shape (as original z[0:21]); y[21:30] outer logits a[c, k] (reference slot B = 0);
y[30:42] Lambert/inner split logits b[c, m] (reference Lambert = 0); y[42:57] knot logits; y[57] eta logit.
decode() returns the closure82 decoded dict (lam [B,3,8], au, av, f0, s [B,3,5], eta [B]); evaluation, compilation, the runtime
and the sampler are the unchanged closure82 / closure83_dr code.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_wl84_104h"))
import comp104h as K  # noqa: E402

CL, CD, H = K.CL, K.CD, K.H
N_GF = 58
SL_OUT, SL_SPLIT, SL_KNOT, I_ETA = slice(21, 30), slice(30, 42), slice(42, 57), 57
ETA_MAX = CL.ETA_MAX


def _alr(logits):
    """Additive log-ratio with a fixed zero reference in the LAST position: [..., n] -> probabilities [..., n + 1]."""
    z = torch.cat([logits, torch.zeros_like(logits[..., :1])], -1)
    return torch.softmax(z, -1)


def decode(y):
    """Candidate raw [B, 58] -> closure82 decoded dict (cert family)."""
    p = CL.C.decode(torch.cat([y[:, :21], torch.zeros_like(y[:, :27])], 1), "passive")   # lobe shape only (lam overwritten)
    B = y.shape[0]
    out = _alr(y[:, SL_OUT].reshape(B, 3, 3))                    # [B,3,4]: O1, O2, O3, block
    blk = out[..., 3:4]
    split = _alr(y[:, SL_SPLIT].reshape(B, 3, 4))                # [B,3,5]: I0..I3, Lambert (reference last)
    lam = torch.cat([blk * split[..., 4:5], out[..., :3], blk * split[..., :4]], -1)   # slots 0 L, 1..3 outer, 4..7 inner
    return {"lam": lam, "au": p["au"], "av": p["av"], "f0": p["f0"],
            "s": torch.sigmoid(y[:, SL_KNOT]).reshape(B, 3, 5),
            "eta": 1.0 + (ETA_MAX - 1.0) * torch.sigmoid(y[:, I_ETA])}


def logit(x):
    return np.log(x) - np.log1p(-x)


def from_original(z):
    """Original raw [B, 64] (float64 numpy) -> candidate raw [B, 58], exact gauge-invariant canonical coordinates."""
    z = np.asarray(z, np.float64)
    B = z.shape[0]
    lg = z[:, 21:48].reshape(B, 3, 9)
    lg = lg - lg.max(-1, keepdims=True)
    e = np.exp(lg)
    lam9 = e / e.sum(-1, keepdims=True)
    s = 1.0 / (1.0 + np.exp(-z[:, 48:63].reshape(B, 3, 5)))
    outer = lam9[..., 1:4]
    blk = 1.0 - outer.sum(-1)                                       # B = 1 - sum outer  (> T because absorption > 0)
    T = lam9[..., 0] + lam9[..., 4:8].sum(-1)
    y = np.zeros((B, N_GF))
    y[:, :21] = z[:, :21]
    y[:, SL_OUT] = (np.log(outer) - np.log(blk)[..., None]).reshape(B, 9)
    y[:, SL_SPLIT] = (np.log(lam9[..., 4:8]) - np.log(lam9[..., 0:1])).reshape(B, 12)
    sp = s * np.sqrt(T / blk)[..., None]
    y[:, SL_KNOT] = logit(sp).reshape(B, 15)
    y[:, I_ETA] = z[:, 63]
    return y


def to_original(y):
    """Candidate raw [B, 58] -> one interior original raw state [B, 64] with the same response (section kappa = (1 + max s') / 2)."""
    y = np.asarray(y, np.float64)
    B = y.shape[0]
    a = np.concatenate([y[:, SL_OUT].reshape(B, 3, 3), np.zeros((B, 3, 1))], -1)
    ea = np.exp(a - a.max(-1, keepdims=True)); out = ea / ea.sum(-1, keepdims=True)
    blk = out[..., 3]
    b = np.concatenate([y[:, SL_SPLIT].reshape(B, 3, 4), np.zeros((B, 3, 1))], -1)
    eb = np.exp(b - b.max(-1, keepdims=True)); split = eb / eb.sum(-1, keepdims=True)
    sp = 1.0 / (1.0 + np.exp(-y[:, SL_KNOT].reshape(B, 3, 5)))
    kap = 0.5 * (1.0 + sp.max(-1))
    lam9 = np.zeros((B, 3, 9))
    lam9[..., 0] = blk * split[..., 4] * kap ** 2
    lam9[..., 1:4] = out[..., :3]
    lam9[..., 4:8] = blk[..., None] * split[..., :4] * (kap ** 2)[..., None]
    lam9[..., 8] = blk * (1.0 - kap ** 2)
    z = np.zeros((B, 64))
    z[:, :21] = y[:, :21]
    z[:, 21:48] = np.log(lam9).reshape(B, 27)
    z[:, 48:63] = logit(sp / kap[..., None]).reshape(B, 15)
    z[:, 63] = y[:, I_ETA]
    return z


def compile_states(y, dtype=torch.float32):
    with torch.no_grad():
        t = torch.as_tensor(np.asarray(y), device=H.dev, dtype=dtype)
        return CD.compiled_state("ER", decode(t))


def compile_original(z, dtype=torch.float32):
    with torch.no_grad():
        t = torch.as_tensor(np.asarray(z), device=H.dev, dtype=dtype)
        return CD.compiled_state("ER", CL.decode("ER", "cert", t))


def raw_groups():
    """Orthonormal candidate raw-space groups (58)."""
    E = np.eye(N_GF)
    rough = np.stack([(E[3 * k] + E[3 * k + 1]) / math.sqrt(2) for k in range(7)], 1)
    aniso = np.stack([(E[3 * k] - E[3 * k + 1]) / math.sqrt(2) for k in range(7)], 1)
    fres = np.stack([E[3 * k + 2] for k in range(7)], 1)
    groups = {"outer_amplitude": E[:, SL_OUT], "split": E[:, SL_SPLIT], "roughness": rough, "anisotropy": aniso,
              "fresnel": fres, "attenuation": E[:, SL_KNOT], "eta": E[:, I_ETA:I_ETA + 1]}
    Bm = np.concatenate(list(groups.values()), 1)
    assert Bm.shape == (N_GF, N_GF) and np.allclose(Bm.T @ Bm, np.eye(N_GF))
    return groups


def original_gauge_vectors(z):
    """The six historical exact gauge directions at original raw state z [64] (WL104H): 3 softmax shifts, 3 attenuation scales."""
    z = np.asarray(z, np.float64)
    lg = z[21:48].reshape(3, 9)
    lam9 = np.exp(lg - lg.max(-1, keepdims=True)); lam9 /= lam9.sum(-1, keepdims=True)
    s = 1.0 / (1.0 + np.exp(-z[48:63].reshape(3, 5)))
    vs = []
    for c in range(3):
        v = np.zeros(64); v[21 + 9 * c:30 + 9 * c] = 1.0; vs.append(v)
    for c in range(3):
        v = np.zeros(64)
        v[48 + 5 * c:53 + 5 * c] = 1.0 / (1.0 - s[c])                    # d s_j = s_j
        tgt = np.zeros(9); R = [0, 4, 5, 6, 7]
        tgt[R] = -2.0 * lam9[c, R]; tgt[8] = -tgt[R].sum()                 # d lam_r = -2 lam_r (block), absorption compensates
        v[21 + 9 * c:30 + 9 * c] = tgt / lam9[c]
        vs.append(v)
    return np.stack(vs)


def gauge_orbit_point(z, kind, c, t):
    """Finite move along a historical gauge orbit: 'shift' adds t to channel c's 9 logits; 'scale' scales knots by kappa = exp(t)
    (kept < 1) and the Lambert/inner block by kappa^-2 with the absorption slot compensating. Returns a new original raw state."""
    z = np.array(z, np.float64, copy=True)
    if kind == "shift":
        z[21 + 9 * c:30 + 9 * c] += t
        return z
    lg = z[21 + 9 * c:30 + 9 * c]
    lam9 = np.exp(lg - lg.max()); lam9 /= lam9.sum()
    s = 1.0 / (1.0 + np.exp(-z[48 + 5 * c:53 + 5 * c]))
    kap = math.exp(t)
    R = [0, 4, 5, 6, 7]
    new = lam9.copy(); new[R] = lam9[R] / kap ** 2
    new[8] = 1.0 - new[:8].sum()
    if new[8] <= 0 or np.any(s * kap >= 1):
        raise ValueError("gauge move leaves the constraint set")
    z[21 + 9 * c:30 + 9 * c] = np.log(new)
    z[48 + 5 * c:53 + 5 * c] = logit(s * kap)
    return z
