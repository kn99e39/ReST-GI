"""WL47 minimal directional nonlocal transport smoke - core (numpy only, float64; no learned K, no production code).

Synthetic oracles S0 (angular-insensitive) / S1 (angular-sensitive internal mirror) on 4 strip ports x 12 angular bins,
the renderer-domain geometry conventions (shared with mir47_render.py), the field / equal-irradiance pair sampler and
the closed-form linear operator fit with its metrics (protocol fnto_smoke47).
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROTO = json.loads((HERE / "protocol" / "smoke47.json").read_text())
PA, PM, PC = PROTO["stage_a"], PROTO["stage_b_mir"], PROTO["stage_c"]

# ---- synthetic angular partition ------------------------------------------------------------------------------------
RING = np.radians(PA["ring_edges_deg"]); NS = PA["n_sectors"]


def _bins():
    th, ph, om = [], [], []
    for r in range(len(RING) - 1):
        a, b = RING[r], RING[r + 1]
        for s in range(NS):
            th.append(math.asin(math.sqrt(0.5 * (math.sin(a) ** 2 + math.sin(b) ** 2))))
            ph.append(s * 2 * math.pi / NS)
            om.append((2 * math.pi / NS) / 2 * (math.sin(b) ** 2 - math.sin(a) ** 2))   # int cos dw over the ring sector
    return np.array(th), np.array(ph), np.array(om)


TH, PH, OMP = _bins()                       # centroid polar / azimuth, projected solid angle (sum = pi)
J = len(TH)
XC = np.array(PA["ports_x"]); W_PORT = PA["port_width"]; H = PA["depth_h"]; P = len(XC)


def dirs(th, ph):
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)


def emap(P_, cvec):
    """E_p = sum_j L(p, j) cvec_j  ->  [P, P * J]."""
    return np.kron(np.eye(P_), np.asarray(cvec)[None, :])


EMAP_S = emap(P, OMP)


def overlap(lo, hi, c, w):
    return max(0.0, min(hi, c + w / 2) - max(lo, c - w / 2)) / w


def spec_bin(j):
    """Mirrored-azimuth bin of the same ring: the exit bin of a specular reflection at a horizontal mirror."""
    r, s = divmod(j, NS)
    return r * NS + (s + NS // 2) % NS


def oracle_s1():
    S = PA["S1"]; T = np.zeros((P * J, P * J))
    for p in range(P):
        for j in range(J):
            r, s = divmod(j, NS)
            dx = -2 * H * math.tan(TH[j]) * math.cos(PH[j])          # light travels along -w (horizontal part)
            dx = 0.0 if abs(dx) < 1e-12 else dx
            so = spec_bin(j) % NS
            for q in range(P):
                ov = overlap(XC[p] - W_PORT / 2 + dx, XC[p] + W_PORT / 2 + dx, XC[q], W_PORT)
                if ov == 0:
                    continue
                for ds, f in ((0, S["f_spec"]), (1, S["f_side"]), (-1, S["f_side"])):
                    jo = r * NS + (so + ds) % NS
                    T[q * J + jo, p * J + j] += S["mirror_reflectance"] * f * ov * OMP[j] / OMP[jo]
    return T


def oracle_s0():
    S = PA["S0"]
    g = np.exp(-np.abs(XC[:, None] - XC[None, :]) / S["ell"]); g *= S["max_column_albedo"] / g.sum(0).max()
    G = S["t_in"] * g / math.pi                                         # Lambertian exit: L = flux / (pi A)
    return np.repeat(G, J, axis=0) @ EMAP_S


def flux_albedo(T, om_in, om_out):
    """Output flux / input flux per unit input state (equal port areas)."""
    return (T * om_out[:, None]).sum(0) / om_in


# ---- renderer-domain conventions (mir47_render.py imports these) -----------------------------------------------------
MIR_T = np.array(PM["directions_signed_tan"]); MIR_H = 0.2; MIR_WIN = [-0.45, -0.15, 0.15, 0.45]; MIR_WW = 0.1


def mir_travel(t):
    v = np.array([t, 0.0, -1.0]); return v / np.linalg.norm(v)


def mir_out(t):
    v = np.array([t, 0.0, 1.0]); return v / np.linalg.norm(v)


def mir_cos(t):
    return 1.0 / math.sqrt(1 + t * t)


def mir_exit_x(x_in, t):
    return x_in + 2 * MIR_H * t


# ---- fields ---------------------------------------------------------------------------------------------------------
def sample_fields(n, P_, J_, cvec, rng, nonzero=False):
    """[n, P_ * J_] input states with E_p = sum_j L cvec ~ U(0.2, 1) on active ports (protocol field_sampler).
    nonzero: redraw all-zero configurations (amendment 1, Stage C)."""
    L = np.zeros((n, P_, J_)); cvec = np.asarray(cvec)
    for i in range(n):
        while not L[i].any():
          for p in range(P_):
            if rng.random() < 0.75:
                k = int(rng.integers(1, 4)); js = rng.choice(J_, k, replace=False); w = rng.dirichlet(np.ones(k))
                L[i, p, js] = w * rng.uniform(0.2, 1.0) / cvec[js]
          if not nonzero:
              break
    return L.reshape(n, P_ * J_)


def pair_fields(LA, P_, J_, cvec, rng):
    """Equal-irradiance partners: same E per port, new 1-3 bin angular distribution."""
    cvec = np.asarray(cvec); A = LA.reshape(len(LA), P_, J_); B = np.zeros_like(A)
    for i in range(len(A)):
        for p in range(P_):
            E = float((A[i, p] * cvec).sum())
            if E > 0:
                k = int(rng.integers(1, 4)); js = rng.choice(J_, k, replace=False); w = rng.dirichlet(np.ones(k))
                B[i, p, js] = w * E / cvec[js]
    return B.reshape(len(A), P_ * J_)


# ---- learned operator ----------------------------------------------------------------------------------------------
def lap_synthetic():
    Lp = np.zeros((J, J))
    for j in range(J):
        r, s = divmod(j, NS)
        nb = [r * NS + (s + 1) % NS, r * NS + (s - 1) % NS] + ([(r + 1) * NS + s] if r + 1 < len(RING) - 1 else []) + ([(r - 1) * NS + s] if r > 0 else [])
        for k in nb:
            Lp[j, k] -= 1; Lp[j, j] += 1
    return np.kron(np.eye(P), (Lp + Lp.T) / 2)


def lap_chain(n_ports, n_dirs):
    Lp = np.zeros((n_dirs, n_dirs))
    for k in range(n_dirs - 1):
        Lp[k, k] += 1; Lp[k + 1, k + 1] += 1; Lp[k, k + 1] -= 1; Lp[k + 1, k] -= 1
    return np.kron(np.eye(n_ports), Lp)


def fit(Z, Y, mask, lap, lam=1e-6, eps=1e-9):
    """argmin_W sum_{o visible} ||W_o Z - Y_o||^2 + lam' tr(W^T Lap W) + eps' ||W||^2 (closed form, row-major vec)."""
    I = Z.shape[0]; O = Y.shape[0]; C = Z @ Z.T; sc = np.trace(C) / I
    Hm = np.kron(np.diag(mask.astype(float)), C) + lam * sc * np.kron(lap, np.eye(I)) + eps * sc * np.eye(O * I)
    rhs = ((mask[:, None] * Y) @ Z.T).reshape(-1)
    return np.linalg.solve(Hm, rhs).reshape(O, I)


def rel(a, b):
    d = float(np.linalg.norm(b)); return float(np.linalg.norm(a - b)) / d if d > 0 else float("nan")


def dir_demean(Y, n_ports, n_dirs):
    Yr = Y.reshape(n_ports, n_dirs, -1); return (Yr - Yr.mean(1, keepdims=True)).reshape(Y.shape)


def evaluate(Wrad, T, Ltr, Lte, pairs, mask, n_ports, n_dirs, nl_mask, Ttr=None):
    """Wrad / T act on the radiance-bin input. Ltr / Lte [I, N]; pairs (LA, LB) [I, n]. Ttr = training-target operator
    (renderer: the same basis set) - defaults to T."""
    Ttr = T if Ttr is None else Ttr
    Yte, Yh = T @ Lte, Wrad @ Lte; hid = ~mask
    LA, LB = pairs; dY = T @ LA - T @ LB; dYh = Wrad @ LA - Wrad @ LB
    a, b = np.random.default_rng(0).random(Lte.shape[0]), np.random.default_rng(1).random(Lte.shape[0])
    lin = float(np.linalg.norm(Wrad @ (a + b) - Wrad @ a - Wrad @ b) / np.linalg.norm(Wrad @ (a + b)))
    return {
        "heldout_rel": rel(Yh, Yte), "hidden_rel": rel(Yh[hid], Yte[hid]) if hid.any() else None,
        "visible_rel": rel(Yh[mask], Yte[mask]),
        "directional_rel": rel(dir_demean(Yh, n_ports, n_dirs), dir_demean(Yte, n_ports, n_dirs)),
        "nonlocal_rel": rel((Wrad * nl_mask) @ Lte, (T * nl_mask) @ Lte),
        "pair_discrimination": float(np.linalg.norm(dYh) / np.linalg.norm(dY)),
        "pair_diff_err": rel(dYh, dY),
        "linearity_residual": lin,
        "neg_frac": float((Wrad < -1e-9 * np.abs(Wrad).max()).mean()),
        "min_over_max": float(Wrad.min() / np.abs(Wrad).max()),
    }


def eonly_floors(T, Emap, Lte, pairs):
    Y = T @ Lte; Z = Emap @ Lte
    Wst = np.linalg.lstsq(Z.T, Y.T, rcond=None)[0].T
    LA, LB = pairs; YA, YB = T @ LA, T @ LB
    pf = math.sqrt(float(((YA - YB) ** 2).sum() / 2 / ((YA ** 2).sum() + (YB ** 2).sum())))
    return {"eonly_linear_floor": rel(Wst @ Z, Y), "eonly_pair_floor": pf}
