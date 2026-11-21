"""WL104J shared definitions: authenticated inputs, incident groups, the four arm parameterizations and float64 evaluation.

Arms (protocol.json "arms"):
  A      shared ER-v1-GF state per material (58), gf104i.decode.
  B      amplitude-only local oracle: per incident group 8 amplitudes per channel (ALR against an absorption slot, sum <= 1);
         lobe shapes, eta and the attenuation function frozen from A (knots at the maximal admissible scale).
  C      full local lobe oracle: B's amplitudes + the 21 raw lobe shape parameters per group; eta and attenuation frozen.
  C_noF  secondary: C with the 7 F0 logits frozen at A's values.
B, C and C_noF are DIAGNOSTIC_ONLY. Nothing in closure82 / closure83 / gf104i is modified.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_wl84_104i"))
sys.path.insert(0, str(HERE.parent / "ecro_wl84_104d"))
import common104i as CI  # noqa: E402  (WL104H / WL104G authenticated inputs, ER-v1-GF, starts)

GF, CL, IO, K, CD = CI.GF, CI.CL, CI.IO, CI.K, CI.CD
DEV = CI.DEV
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
REPO = IO.REPO
OUT = REPO / "results/ecro_wl84_104j"
ANGLES = np.array(P["incident_groups"]["polar_deg"], float)
AZ = np.array(P["incident_groups"]["azimuth_deg"], float)
G = len(ANGLES)
NP = G * len(AZ)
PROBE_GROUP = np.repeat(np.arange(G), len(AZ))
POLAR = np.repeat(ANGLES, len(AZ))
LUMA = CI.LUMA
LANES = 1024
DESC = REPO / "results/ecro_wl84_104/inputs63/rest_compiled83/results/data/materials83.npz"
I_RAW = REPO / "results/ecro_wl84_104i/primary/fit/fit_states.npz"
ARMS = ("A", "B", "C", "C_noF")
STARTS = ["generic"] + [str(s) for s in P["seeds"]["starts"]]
Stop = IO.Stop


def sha(path):
    return IO.sha(path)


def byte_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def inputs():
    """rows, raw {R0, A, C, ER_L}, compiled, descriptors [9, 10], evaluation reference, WL104I states, authenticated hashes."""
    rows, raw, compiled, seen = CI.inputs()
    prov = IO.read(REPO / "evaluation/104D/amendment1/provenance.json")
    want = prov["parent_source_sha256"]["data/materials83.npz"]
    if sha(DESC) != want:
        raise Stop("FROZEN_HASH_MISMATCH: materials83.npz")
    seen[DESC.relative_to(REPO).as_posix()] = want
    with np.load(DESC) as z:
        desc = np.stack([z[r["set"]][r["index"]] for r in rows])
    for r, d in zip(rows, desc):
        if byte_hash(d) != r["descriptor_sha256"]:
            raise Stop(f"DESCRIPTOR_HASH_MISMATCH: {r['tag']}")
    man = IO.read(REPO / "evaluation/104I/manifest.json")
    if sha(I_RAW) != man["raw_sha256"]["fit/fit_states.npz"]:
        raise Stop("FROZEN_HASH_MISMATCH: WL104I fit_states.npz")
    seen[I_RAW.relative_to(REPO).as_posix()] = man["raw_sha256"]["fit/fit_states.npz"]
    with np.load(I_RAW) as z:
        wl104i = {k: z[k] for k in z.files}
    ref = IO.reference()
    prec, ok = IO.reference_precision(ref, len(rows))
    if not ok:
        raise Stop("EVALUATION_REFERENCE_PRECISION")
    if not (np.allclose(ref["wi"].reshape(len(rows), NP, -1, 3)[:, :, 0], incident_directions()[None], atol=1e-12)):
        raise Stop("EVALUATION_INCIDENCE_LAYOUT")
    return rows, raw, compiled, desc, ref, wl104i, seen


def sph(theta_deg, phi_deg):
    t, p = np.radians(theta_deg), np.radians(phi_deg)
    return np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1)


def incident_directions(polar=ANGLES, az=AZ):
    th, ph = np.meshgrid(polar, az, indexing="ij")
    return sph(th.ravel(), ph.ravel())


def reshape_set(d, n):
    """Flat [n * P, L, ...] arrays -> [n, P, L, ...]."""
    return {k: v.reshape(n, -1, *v.shape[1:]) for k, v in d.items()}


# ------------------------------------------------------------------ parameterizations
def alr8(l):
    """[..., 3, 8] logits -> amplitudes [..., 3, 8] with an implicit absorption slot (sum < 1)."""
    z = torch.cat([l, torch.zeros_like(l[..., :1])], -1)
    return torch.softmax(z, -1)[..., :8]


def decode_local(shape_raw, amp, s_n, eta):
    """shape_raw [B, 21], amp [B, 3, 8] logits, s_n [B, 3, 5] knot values, eta [B] -> closure82 decoded dict (ER cert family)."""
    z = torch.cat([shape_raw, torch.zeros(shape_raw.shape[0], 27, dtype=shape_raw.dtype, device=shape_raw.device)], 1)
    p = CL.C.decode(z, "passive")
    return {"lam": alr8(amp), "au": p["au"], "av": p["av"], "f0": p["f0"], "s": s_n, "eta": eta}


def section(y):
    """ER-v1-GF raw [N, 58] -> frozen/local B coordinates (float64 numpy): shape_raw [N, 21], amp [N, 3, 8], s_n [N, 3, 5],
    eta [N]. Knots at kappa_c = max_j s'_cj (largest admissible scale), Lambert + inner amplitudes * kappa^2, absorption
    = (1 - sum outer)(1 - kappa^2), computed without cancellation."""
    y = np.asarray(y, np.float64)
    t = torch.as_tensor(y, dtype=torch.float64, device=DEV)
    with torch.no_grad():
        p = GF.decode(t)
        lam, s = p["lam"], p["s"]
        kl = t[:, GF.SL_KNOT].reshape(-1, 3, 5)
        kmax = kl.max(-1).values                               # logit of kappa
        kap = torch.sigmoid(kmax)
        one_minus = torch.sigmoid(-kmax)                       # 1 - kappa, exact
        blk = 1.0 - lam[..., 1:4].sum(-1)
        absorb = blk * one_minus * (1.0 + kap)
        lb = lam.clone()
        lb[..., 0] = lam[..., 0] * kap ** 2
        lb[..., 4:8] = lam[..., 4:8] * (kap ** 2)[..., None]
        amp = torch.log(lb) - torch.log(absorb)[..., None]
        s_n = s / kap[..., None]
    out = {"shape_raw": y[:, :21].copy(), "amp": amp.cpu().numpy(), "s_n": s_n.cpu().numpy(), "eta": p["eta"].cpu().numpy()}
    if not all(np.isfinite(v).all() for v in out.values()) or (absorb <= 0).any():
        raise Stop("SECTION_NOT_INTERIOR")
    return out


def expand_groups(sec, G_=G):
    """Shared section -> per-group local arrays (identical in every group)."""
    return {"shape_raw": np.repeat(sec["shape_raw"][:, None], G_, 1), "amp": np.repeat(sec["amp"][:, None], G_, 1),
            "s_n": sec["s_n"], "eta": sec["eta"]}


def lane_params(arm, st, dtype=torch.float64, groups=None):
    """Decoded params for a batch of materials. st: A -> {'y': [N, 58]}; local -> {'shape_raw': [N, G, 21], 'amp': [N, G, 3, 8],
    's_n': [N, 3, 5], 'eta': [N]} (numpy or torch). Returns dict with batch [N] (A) or [N * G] (local, material-major)."""
    tt = lambda v: torch.as_tensor(v, dtype=dtype, device=DEV) if not torch.is_tensor(v) else v
    if arm == "A":
        return GF.decode(tt(st["y"]))
    sr, amp = tt(st["shape_raw"]), tt(st["amp"])
    N, Gs = sr.shape[:2]
    s_n = tt(st["s_n"])[:, None].expand(N, Gs, 3, 5).reshape(-1, 3, 5)
    eta = tt(st["eta"])[:, None].expand(N, Gs).reshape(-1)
    return decode_local(sr.reshape(-1, 21), amp.reshape(-1, 3, 8), s_n, eta)


def per_probe(p, arm, n, probe_group):
    """Decoded params -> per (material, probe) params [n * P] given the probe -> group map."""
    P_ = len(probe_group)
    if arm == "A":
        idx = torch.arange(n, device=DEV).repeat_interleave(P_)
    else:
        Gs = p["lam"].shape[0] // n
        idx = (torch.arange(n, device=DEV)[:, None] * Gs + torch.as_tensor(probe_group, device=DEV)[None]).reshape(-1)
    return {k: v[idx] for k, v in p.items()}


def to_lanes(pp, L):
    return {k: v[:, None].expand(v.shape[0], L, *v.shape[1:]).reshape(-1, *v.shape[1:]) for k, v in pp.items()}


def predict(arm, st, wi, wo, probe_group, dtype=torch.float64):
    """F = f cos_o [n, P, L, 3] (numpy float64) at lanes wi/wo [n, P, L, 3]; probe p uses group probe_group[p] (local arms)."""
    n, P_, L = wi.shape[:3]
    out = np.empty(wi.shape[:3] + (3,))
    with torch.no_grad():
        p = lane_params(arm, st, dtype)
        pp = per_probe(p, arm, n, probe_group)
        for m in range(n):
            sl = slice(m * P_, (m + 1) * P_)
            q = to_lanes({k: v[sl] for k, v in pp.items()}, L)
            a = torch.as_tensor(wi[m].reshape(-1, 3), dtype=dtype, device=DEV)
            b = torch.as_tensor(wo[m].reshape(-1, 3), dtype=dtype, device=DEV)
            out[m] = CL.eval_closure("ER", q, a, b).cpu().numpy().reshape(P_, L, 3)
    return out


def terms(arm, st, wi, wo, probe_group, dtype=torch.float64):
    """Per-term responses {O0..O2, SL, SI0..SI3} [n, P, L, 3] (WL104H names, S = s(mu_i) s(mu_o))."""
    n, P_, L = wi.shape[:3]
    names = ("O0", "O1", "O2", "SL", "SI0", "SI1", "SI2", "SI3")
    out = {k: np.empty(wi.shape[:3] + (3,)) for k in names}
    with torch.no_grad():
        p = lane_params(arm, st, dtype)
        pp = per_probe(p, arm, n, probe_group)
        for m in range(n):
            for j in range(P_):
                q = {k: v[m * P_ + j:m * P_ + j + 1] for k, v in pp.items()}
                a = torch.as_tensor(wi[m, j], dtype=dtype, device=DEV)
                b = torch.as_tensor(wo[m, j], dtype=dtype, device=DEV)
                t = K.terms_torch(q, a, b)
                for k in ("O0", "O1", "O2"):
                    out[k][m, j] = t[k].cpu().numpy()
                out["SL"][m, j] = (t["S"] * t["L"]).cpu().numpy()
                for k in range(4):
                    out[f"SI{k}"][m, j] = (t["S"] * t[f"I{k}"]).cpu().numpy()
    return out


def wl82_terms(pred, target, w):
    """WL82 per-probe el1 / ealb / eshape: pred, target [..., L, 3], w [..., L] -> arrays [...]."""
    W = w[..., None]
    a = (pred * W).sum(-2)
    b = (target * W).sum(-2)
    tot = np.maximum(b.sum(-1), 1e-20)
    el1 = (np.abs(pred - target) * W).sum((-2, -1)) / tot
    ealb = np.abs(a - b).sum(-1) / tot
    npred = pred / np.maximum(a[..., None, :], 1e-20)
    ntgt = target / np.maximum(b[..., None, :], 1e-20)
    eshape = (b * (W * np.abs(npred - ntgt)).sum(-2)).sum(-1) / tot
    return {"el1": el1, "ealb": ealb, "eshape": eshape}


def mirror_radius(wi, wo):
    m = wi * np.array([-1.0, -1.0, 1.0])
    return np.degrees(np.arccos(np.clip((m * wo).sum(-1), -1, 1)))


def write(path, obj):
    IO.write(path, obj)


def read(path):
    return IO.read(path)


def commit():
    return IO.commit()
