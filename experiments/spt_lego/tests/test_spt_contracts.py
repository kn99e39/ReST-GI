"""SPT-1 synthetic transport contracts (worklog 39). Analytic geometry (planes and boxes), the production state code.

    python -m pytest experiments/spt_lego/tests -q          (any venv with torch + numpy; CPU)

Relation tables are built here by analytic ray casting in the same format as spt_relations.py (hit sender anchor,
per-bin sun visibility, SH9 gather weights); the dynamic state is computed by spt_model.anchor_state /
single-light Jacobi, i.e. the code the Lego evaluation uses. Persistent quantities (albedo, E_sun) are fixed stubs.
These tests establish contract semantics only; they say nothing about architecture viability on a real scene.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spt_common as C  # noqa: E402
import spt_model as SM  # noqa: E402

UP = np.array([0.0, 0.0, 1.0])


class Box:
    def __init__(self, lo, hi, obj):
        self.lo, self.hi, self.obj = np.asarray(lo, float), np.asarray(hi, float), obj

    def hit(self, o, d):
        inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
        t1, t2 = (self.lo - o) * inv, (self.hi - o) * inv
        tn, tf = np.minimum(t1, t2).max(1), np.maximum(t1, t2).min(1)
        ok = (tf >= tn) & (tf > 1e-6)
        return np.where(ok, np.where(tn > 1e-6, tn, np.inf), np.inf)


def anchors_on_box(b, h):
    P, N = [], []
    for ax in range(3):
        u, v = [i for i in range(3) if i != ax]
        gu = np.arange(b.lo[u] + h / 2, b.hi[u], h); gv = np.arange(b.lo[v] + h / 2, b.hi[v], h)
        U, V = np.meshgrid(gu, gv, indexing="ij")
        for side, val in ((-1, b.lo[ax]), (1, b.hi[ax])):
            q = np.zeros((U.size, 3)); q[:, u] = U.ravel(); q[:, v] = V.ravel(); q[:, ax] = val
            n = np.zeros((U.size, 3)); n[:, ax] = side
            P.append(q); N.append(n)
    return np.concatenate(P), np.concatenate(N)


class SynthRel:
    """Relation tables in the production format for a list of boxes (each box = one object with its own anchors)."""

    def __init__(self, boxes, h=0.1, k=C.K_GATHER):
        P, N, O = [], [], []
        for b in boxes:
            p, n = anchors_on_box(b, h); P.append(p); N.append(n); O.append(np.full(len(p), b.obj))
        self.P, self.N, self.O = np.concatenate(P), np.concatenate(N), np.concatenate(O)
        rot = np.random.default_rng(0).uniform(0, 2 * np.pi, len(self.P))
        t, bb, n = C.frames(self.N, rot); L = C.cosine_dirs_local(k)
        self.dirs = L[None, :, 0:1] * t[:, None] + L[None, :, 1:2] * bb[:, None] + L[None, :, 2:3] * n[:, None]
        self.k = k
        self.cast(boxes)

    def cast(self, boxes):
        Na, k = len(self.P), self.k
        o = np.repeat(self.P + C.EPS * self.N, k, 0); d = self.dirs.reshape(-1, 3)
        ts = np.stack([b.hit(o, d) for b in boxes], 1); first = ts.argmin(1); tmin = ts.min(1)
        hp = o + d * np.where(np.isfinite(tmin), tmin, 0)[:, None]
        hit = np.full(len(d), -1)
        for bi, b in enumerate(boxes):
            m = np.isfinite(tmin) & (first == bi)
            cand = np.flatnonzero(self.O == b.obj)
            dd = np.linalg.norm(hp[m][:, None] - self.P[cand][None], axis=2)
            facing = (self.N[cand][None] * -d[m][:, None]) .sum(-1) > 0
            dd = np.where(facing, dd, dd + 10.0)
            hit[m] = cand[dd.argmin(1)]
        self.hit = hit.reshape(Na, k)
        bc = C.bin_centres()
        oo = np.repeat(self.P + C.EPS * self.N, C.N_BINS, 0); dd = np.tile(bc, (Na, 1))
        blocked = np.isfinite(np.stack([b.hit(oo, dd) for b in boxes], 1).min(1))
        self.vis = (~blocked).reshape(Na, C.N_BINS)

    def torch(self):
        class R:  # duck-typed spt_model.Relations
            pass
        r = R(); r.pos = torch.tensor(self.P, dtype=torch.float32); r.nrm = torch.tensor(self.N, dtype=torch.float32)
        r.obj = torch.tensor(self.O); r.sh = torch.tensor(C.sh9(self.dirs) * np.pi / self.k, dtype=torch.float32)
        r.hit = torch.tensor(self.hit); r.vis = torch.tensor(self.vis.astype(np.uint8)); r.Na, r.K = self.hit.shape
        r.bins = torch.tensor(C.bin_centres(), dtype=torch.float32); r.visbit = lambda s, b: r.vis[s, b].float()
        return r


class Stub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_esun = torch.nn.Parameter(torch.tensor(0.0))


def first_bounce(R, alb, wl):
    stub = Stub()
    return SM.anchor_state(stub, R, alb, torch.arange(R.Na), wl[None].expand(R.Na, 3), torch.zeros(R.Na, 3))


SUN = torch.tensor([0.3, 0.2, 0.93]) / torch.tensor([0.3, 0.2, 0.93]).norm()


def scene(box_x, occluder=False, wall=False):
    floor = Box([-1, -1, -0.1], [1, 1, 0.0], 0)
    sender = Box([box_x - 0.15, -0.15, 0.0], [box_x + 0.15, 0.15, 0.3], 1)
    boxes = [floor, sender]
    if occluder:
        boxes.append(Box([-0.75, -0.5, 0.0], [-0.70, 0.5, 0.6], 2))
    if wall:          # wall (0.5 m) on the side away from the sun: floor anchors at x < -0.6 never see the box (0.3 m tall)
        boxes.append(Box([-0.55, -3.0, 0.0], [-0.45, 3.0, 0.5], 3))
    return boxes


def albedo_for(O, red=(0.8, 0.05, 0.05)):
    a = np.full((len(O), 3), 0.5); a[O == 1] = red
    return torch.tensor(a, dtype=torch.float32)


def receivers(S):
    """Top-face floor anchors (persistent identity)."""
    return np.flatnonzero((S.O == 0) & (S.N[:, 2] > 0.5) & (np.abs(S.P[:, 2]) < 1e-9))


def test_linear_response_to_incident_change():
    """Contract 1: a local receiver's persistent state is untouched; its incident state scales with sender radiance."""
    S = SynthRel(scene(0.4)); R = S.torch()
    a1 = albedo_for(S.O); a2 = a1.clone(); a2[torch.tensor(S.O) == 1] *= 2
    g1, g2 = first_bounce(R, a1, SUN), first_bounce(R, a2, SUN)
    rec = receivers(S)
    sees_box = (S.hit[rec] >= 0) & (S.O[np.maximum(S.hit[rec], 0)] == 1)
    r = rec[sees_box.any(1) & ~((S.hit[rec] >= 0) & (S.O[np.maximum(S.hit[rec], 0)] == 0)).any(1)]
    if len(r):
        assert torch.allclose(g2[r], 2 * g1[r], rtol=1e-5, atol=1e-7)
    assert (g2[rec] - g1[rec]).abs().sum() > 0


def test_stationary_receiver_changes_when_sender_moves_and_unchanged_relations_are_exact():
    """Contract 2: moving the box changes stationary floor receivers; anchors whose relation rows are unchanged keep a
    bit-identical first-bounce state (state locality), and the persistent albedo of the floor is never touched."""
    S0, S1 = SynthRel(scene(0.0, wall=True)), SynthRel(scene(0.5, wall=True))
    a = albedo_for(S0.O)
    g0, g1 = first_bounce(S0.torch(), a, SUN), first_bounce(S1.torch(), a, SUN)
    rec = receivers(S0)
    same_rows = (S0.hit == S1.hit).all(1) & (S0.vis == S1.vis).all(1)
    # a row is only comparable if its senders' visibility is also unchanged
    senders_same = np.array([same_rows[np.maximum(S0.hit[i], 0)].all() for i in range(len(S0.P))])
    exact = rec[same_rows[rec] & senders_same[rec]]
    changed = rec[~same_rows[rec]]
    assert len(exact) > 0 and len(changed) > 0
    assert torch.equal(g0[exact], g1[exact])
    red0 = g0[changed, 0, 0] - g0[changed, 0, 1]; red1 = g1[changed, 0, 0] - g1[changed, 0, 1]
    assert (red0 - red1).abs().max() > 1e-3          # colour bleeding on stationary receivers follows the sender


def test_occluder_reduces_incident_transport():
    """Contract 3: inserting an occluder (third-party edit) lowers red incident radiance at receivers behind it."""
    S0, S1 = SynthRel(scene(-0.4)), SynthRel(scene(-0.4, occluder=True))
    a0 = albedo_for(S0.O); a1 = albedo_for(S1.O)
    g0, g1 = first_bounce(S0.torch(), a0, SUN), first_bounce(S1.torch(), a1, SUN)
    r0 = receivers(S0); r1 = receivers(S1)
    key0 = {tuple(np.round(S0.P[i], 6)): i for i in r0}
    pairs = [(key0[tuple(np.round(S1.P[j], 6))], j) for j in r1 if tuple(np.round(S1.P[j], 6)) in key0]
    i0 = np.array([p[0] for p in pairs]); i1 = np.array([p[1] for p in pairs])
    behind = S0.P[i0, 0] < -0.75
    red0 = (g0[i0, 0, 0] - g0[i0, 0, 1])[torch.tensor(behind)]; red1 = (g1[i1, 0, 0] - g1[i1, 0, 1])[torch.tensor(behind)]
    assert red0.sum() > 0 and red1.sum() < 0.5 * red0.sum()


def test_sh2_directional_limit_is_measured():
    """Contract 4 (stress): SH order 2 represents diffuse transfer of a concentrated incident lobe well, but not a
    glossy lobe. Measured, not hidden: diffuse error < 5 %, glossy (Phong n = 200) relative error > 50 %."""
    rng = np.random.default_rng(0)
    w = rng.normal(size=(200_000, 3)); w /= np.linalg.norm(w, axis=1, keepdims=True); w = w[w[:, 2] > 0]
    src = np.array([0.5, 0.0, 0.866]); lobe = (w @ src > np.cos(np.deg2rad(4))).astype(float)            # small bright source
    dw = 2 * np.pi / len(w)
    c = (C.sh9(w) * (lobe * w[:, 2])[:, None]).sum(0) * dw                                                # cosine-weighted SH
    rec_ = C.sh9(w) @ c                                                                                   # reconstruction of L*cos
    true_diff = (lobe * w[:, 2]).sum() * dw / np.pi                                                       # Lambertian response
    sh_diff = c[0] / 0.282095 / np.pi                                                                     # from the DC coefficient
    refl = np.array([-0.5, 0.0, 0.866])
    phong = lambda wi: (201 / (2 * np.pi)) * np.clip((wi @ np.array([-refl[0], -refl[1], refl[2]])), 0, None) ** 200  # noqa: E731
    true_gl = (phong(w) * lobe * w[:, 2]).sum() * dw
    sh_gl = (phong(w) * np.clip(rec_, 0, None)).sum() * dw
    assert abs(sh_diff - true_diff) / true_diff < 0.05
    assert abs(sh_gl - true_gl) / true_gl > 0.5


def test_multibounce_jacobi_converges():
    """Contract 5: the multi-bounce cache converges geometrically for albedo < 1 (two facing plates)."""
    boxes = [Box([-1, -1, -0.1], [1, 1, 0.0], 0), Box([-1, -1, 0.5], [1, 1, 0.6], 1)]
    S = SynthRel(boxes, h=0.2); R = S.torch()
    alb = torch.full((R.Na, 3), 0.7); wl = torch.tensor([0.0, 0.0, 1.0])
    R.vis = torch.ones_like(R.vis)                                       # light both plates: isolate the iteration
    stub = Stub(); prev = None; deltas = []
    import spt_eval as EV
    for it in (1, 2, 3, 4, 6, 8):
        E = EV.single_light_cache(stub, R, alb, wl, iters=it)
        if prev is not None:
            deltas.append(float((E - prev).abs().max()))
        prev = E
    assert all(d1 <= d0 + 1e-6 for d0, d1 in zip(deltas, deltas[1:]))
    assert deltas[-1] < 0.25 * deltas[0]
