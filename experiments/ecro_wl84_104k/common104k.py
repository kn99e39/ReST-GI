"""WL104K shared definitions: authenticated inputs (WL104J data + WL104D reference), the ER-v1-SDM candidate response (torch and
experiment-local Dr.Jit), arm parameter layouts and helpers.

F_SDM(wi, wo) = sum_j q_j(mu_i) q_j(mu_o) C_j(wi, wo), j in (O0, O1, O2, L, I0, I1, I2, I3); C_j are the certified ER components
(cosine-weighted; L and I_k include s(mu_i) s(mu_o)); q_j piecewise linear through 5 knots (mu = 0, .15, .4, .7, 1), values in [0, 1].
Arms: A (no gates, closure82.eval_closure), T (one gate shared by all components), P (one gate per component). Not adopted.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ecro_wl84_104j"))
import common104j as J  # noqa: E402  (WL104J / WL104I / WL104H authenticated inputs and helpers)

GF, CL, IO, K, CD = J.GF, J.CL, J.IO, J.K, J.CD
DEV = J.DEV
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
REPO = J.REPO
OUT = REPO / "results/ecro_wl84_104k"
J_DATA = REPO / "results/ecro_wl84_104j/primary/data"
J_STATES = REPO / "results/ecro_wl84_104j/primary/fit_primary/states.npz"
COMP = ("O0", "O1", "O2", "L", "I0", "I1", "I2", "I3")
NJ = {"A": 0, "T": 1, "P": 8}
KNOTS = (0.0, 0.15, 0.4, 0.7, 1.0)
ARMS = ("A", "T", "P")
LUMA = J.LUMA
NP, G, PROBE_GROUP, POLAR, ANGLES = J.NP, J.G, J.PROBE_GROUP, J.POLAR, J.ANGLES
STARTS = J.STARTS
Stop = IO.Stop
wl82_terms, mirror_radius, sph = J.wl82_terms, J.mirror_radius, J.sph


def sha(p):
    return IO.sha(p)


def inputs():
    """WL104J inputs() plus authentication of the WL104J data sets and A states against evaluation/104J/manifest.json."""
    rows, raw, compiled, desc, ref, wl104i, seen = J.inputs()
    man = IO.read(REPO / "evaluation/104J/manifest.json")["raw_sha256"]
    for rel in ("primary/data/fit_set.npz", "primary/data/polar_transfer_set.npz", "primary/data/azimuth_transfer_set.npz",
                "primary/data/data.json", "primary/fit_primary/states.npz"):
        p = REPO / "results/ecro_wl84_104j" / rel
        if sha(p) != man[rel]:
            raise Stop(f"FROZEN_HASH_MISMATCH: {rel}")
        seen["results/ecro_wl84_104j/" + rel] = man[rel]
    dj = IO.read(J_DATA / "data.json")
    if not dj["qualified"]["all"]:
        raise Stop("WL104J_DATA_UNQUALIFIED")
    return rows, raw, compiled, desc, ref, seen


def load_set(name, n):
    with np.load(J_DATA / f"{name}.npz") as z:
        s = {k: z[k].reshape(n, -1, *z[k].shape[1:]) for k in ("wi", "wo", "use", "q", "a", "b")}
    return {"wi": s["wi"], "wo": s["wo"], "w": s["use"] / s["q"], "a": s["a"], "b": s["b"], "mean": 0.5 * (s["a"] + s["b"])}


def eval_reference(ref, n):
    R4 = {k: ref[k].reshape(n, NP, *ref[k].shape[1:]) for k in ("wi", "wo", "use", "q", "a", "b")}
    ev = {"wi": R4["wi"], "wo": R4["wo"], "w": R4["use"] / R4["q"], "a": R4["a"], "b": R4["b"], "mean": 0.5 * (R4["a"] + R4["b"])}
    pk = {k: ref[k].reshape(n, NP, *ref[k].shape[1:]) for k in ("peak_wi", "peak_wo", "peak_a", "peak_b")}
    peak = {"wi": pk["peak_wi"], "wo": pk["peak_wo"], "ref": (0.5 * (pk["peak_a"] + pk["peak_b"])) @ LUMA}
    return ev, peak


# ------------------------------------------------------------------ torch response
def components(p, wi, wo):
    """Certified ER components (cosine-weighted) for lanes with gathered params p: [N, 8, 3], order COMP. Same operations as
    closure82.eval_closure('ER') (closure83 robust refraction active through the h104b import)."""
    ci, co = wi[:, 2], wo[:, 2]
    co_ = co.clamp_min(0)[:, None]
    lam = p["lam"]
    fo, valid = CL.lobe_f(p["au"][:, CL.OUTER], p["av"][:, CL.OUTER], p["f0"][:, CL.OUTER], wi, wo)
    eta = p["eta"]
    fi, _ = CL.lobe_f(p["au"][:, CL.INNER], p["av"][:, CL.INNER], p["f0"][:, CL.INNER], CL.refract_up(wi, eta), CL.refract_up(wo, eta))
    fi = fi / (eta * eta)[:, None]
    S = CL.s_interp(p["s"], ci) * CL.s_interp(p["s"], co)                       # [N, 3]
    out = [lam[:, :, 1 + k] * (fo[:, k:k + 1] * co_) for k in range(3)]
    out.append(S * lam[:, :, 0] * (co_ / np.pi))
    out += [S * lam[:, :, 4 + k] * (fi[:, k:k + 1] * co_) for k in range(4)]
    C = torch.stack(out, 1)
    return torch.where(valid[:, None, None], C, torch.zeros_like(C))


def gate_interp(g, mu):
    """g [N, Jg, 3, 5] knot values, mu [N] -> [N, Jg, 3] (piecewise linear, mu clamped to [0, 1]; WL82 s_interp convention)."""
    k = torch.as_tensor(KNOTS, dtype=mu.dtype, device=mu.device)
    m = mu.clamp(0, 1)
    j = torch.bucketize(m, k[1:-1], right=True)
    t = (m - k[j]) / (k[j + 1] - k[j])
    N, Jg = g.shape[:2]
    jj = j[:, None, None, None].expand(N, Jg, 3, 1)
    g0 = torch.gather(g, 3, jj)[..., 0]
    g1 = torch.gather(g, 3, jj + 1)[..., 0]
    return g0 + (g1 - g0) * t[:, None, None]


def eval_sdm(p, g, wi, wo):
    """F_SDM [N, 3]; g [N, Jg, 3, 5] with Jg = 1 (shared) or 8 (per component)."""
    C = components(p, wi, wo)
    q = gate_interp(g, wi[:, 2]) * gate_interp(g, wo[:, 2])                   # [N, Jg, 3]
    return (q * C).sum(1)


def decode(arm, st, dtype=torch.float64):
    """st: {'y': [N, 58], 'g': [N, Jg, 3, 5]} -> (closure82 params [N], gates [N, Jg, 3, 5] or None)."""
    tt = lambda v: v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v), dtype=dtype, device=DEV)
    p = GF.decode(tt(st["y"]))
    return p, (tt(st["g"]) if arm != "A" else None)


def lane_eval(arm, p, g, wi, wo):
    if arm == "A":
        return CL.eval_closure("ER", p, wi, wo)
    return eval_sdm(p, g, wi, wo)


def predict(arm, st, wi, wo, dtype=torch.float64):
    """F [n, P, L, 3] (numpy) at lanes wi / wo [n, P, L, 3]; st arrays with leading n."""
    n, P_, L = wi.shape[:3]
    out = np.empty(wi.shape[:3] + (3,))
    with torch.no_grad():
        p, g = decode(arm, st, dtype)
        for m in range(n):
            q = {k: v[m:m + 1].expand(P_ * L, *v.shape[1:]) for k, v in p.items()}
            gm = None if g is None else g[m:m + 1].expand(P_ * L, *g.shape[1:])
            a = torch.as_tensor(wi[m].reshape(-1, 3), dtype=dtype, device=DEV)
            b = torch.as_tensor(wo[m].reshape(-1, 3), dtype=dtype, device=DEV)
            out[m] = lane_eval(arm, q, gm, a, b).cpu().numpy().reshape(P_, L, 3)
    return out


def comp_responses(arm, st, wi, wo, dtype=torch.float64):
    """Gated component responses [n, P, L, 8, 3] (numpy)."""
    n, P_, L = wi.shape[:3]
    out = np.empty(wi.shape[:3] + (8, 3))
    with torch.no_grad():
        p, g = decode("P" if arm == "A" else arm, {**st, "g": st.get("g", np.ones((n, 1, 3, 5)))}, dtype)
        for m in range(n):
            q = {k: v[m:m + 1].expand(P_ * L, *v.shape[1:]) for k, v in p.items()}
            a = torch.as_tensor(wi[m].reshape(-1, 3), dtype=dtype, device=DEV)
            b = torch.as_tensor(wo[m].reshape(-1, 3), dtype=dtype, device=DEV)
            C = components(q, a, b)
            gm = g[m:m + 1].expand(P_ * L, *g.shape[1:])
            C = C * (gate_interp(gm, a[:, 2]) * gate_interp(gm, b[:, 2]))
            out[m] = C.cpu().numpy().reshape(P_, L, 8, 3)
    return out


def identity_gates(n, arm):
    return np.ones((n, NJ[arm], 3, 5))


# ------------------------------------------------------------------ experiment-local Dr.Jit evaluator (scenes / runtime)
class DrSDM:
    """cl.eval(wi, wo, active) -> Color3f F_SDM for one material: comp104h.DrTerms components x Dr.Jit gates (same knot scheme
    as closure83_dr._s). Experiment-local; the production cmp83 plugin is not modified."""

    def __init__(self, state69, gates):
        mi, dr = K.mi, K.dr
        self.t = K.DrTerms(np.asarray(state69, np.float32))
        gates = np.asarray(gates, np.float32)                                    # [Jg, 3, 5]
        self.Jg = gates.shape[0]
        self.g = []
        for j in range(self.Jg):
            ch = []
            for c in range(3):
                vals = []
                for k in range(5):
                    x = mi.Float(float(gates[j, c, k])); dr.make_opaque(x); vals.append(x)
                ch.append(vals)
            self.g.append(ch)

    def _q(self, j, mu):
        mi, dr = K.mi, K.dr
        out = []
        for c in range(3):
            v = mi.Float(self.g[j][c][0])
            for k in range(4):
                t = dr.clamp((mu - KNOTS[k]) * (1.0 / (KNOTS[k + 1] - KNOTS[k])), 0.0, 1.0)
                v = v + (self.g[j][c][k + 1] - self.g[j][c][k]) * t
            out.append(v)
        return mi.Color3f(out[0], out[1], out[2])

    def eval(self, wi, wo, active=True):
        mi, dr = K.mi, K.dr
        t = self.t.terms(wi, wo, active, unrefracted=False)
        qq = [self._q(j, wi.z) * self._q(j, wo.z) for j in range(self.Jg)]
        q = (lambda j: qq[0]) if self.Jg == 1 else (lambda j: qq[j])
        out = q(0) * t["O0"] + q(1) * t["O1"] + q(2) * t["O2"]
        inner = q(3) * t["L"] + q(4) * t["I0"] + q(5) * t["I1"] + q(6) * t["I2"] + q(7) * t["I3"]
        return dr.select(t["valid"], out + t["S"] * inner, 0.0)


def dr_eval_numpy(cl, wi, wo):
    mi, dr = K.mi, K.dr
    V = lambda a: mi.Vector3f(*[mi.Float(np.ascontiguousarray(a[:, k], np.float32)) for k in range(3)])
    f = cl.eval(V(wi), V(wo), mi.Bool(True))
    return np.stack([np.asarray(f[c]) for c in range(3)], 1).astype(np.float64)


def write(path, obj):
    IO.write(path, obj)


def read(path):
    return IO.read(path)


def commit():
    return IO.commit()
