"""WL115 Stage B candidate representation CAR-16 (contact-adaptive hierarchical refinement), frozen before any HELD_SEEN / FRESH
WL115 evidence stream.

Representation of the full-depth HIGH table at the event-2 query x in patch p, 4 x 4 sub-cell c, 16 x 16 child k(x), bin b:
    T(x, b) = (1 - w_c) T4[p, c, b] + w_c T16[p, c, k(x), b]
  * T4: the dense 4 x 4 table (WL113 / WL114 contract, unchanged),
  * T16: 4 x 4 children of every 4 x 4 sub-cell (16 x 16 per patch) on the support,
  * w_c = clamp((m_c - M0) / (M1 - M0), 0, 1): continuous activation from the cell's geometric proximity to a DIFFERENT rigid
    object, m_c = sum cos exp(-d / ELL) [hit on another object] / sum cos over the 64 bin-centre directions from the cell centre
    (geomdesc115.prox_other; geometry only; no camera, light, radiance, reference, contact ROI or sphere-ring rule),
  * support = cells with w_c > 0; it is recomputed from the current geometry at every state (no stale refinement state).
Non-negative and linear in the tables (convex combination of non-negative values); camera independent; source enters only through
the tables (linear source application preserved for learned tables).
Calibration (SELECTION capture views only, Stage A + A2): M0 = 0.002 keeps >= 99 % and M1 = 0.01 gives full weight to >= 85 % of the
query-weighted 16 x 16-vs-4 x 4 refinement importance of the diagnostic contact neighbourhood; 16 x 16 because the camera-conditioned
16 x 16 and 32 x 32 tables are equivalent (Q16R ~ Q32R) while 4 x 4 / 8 x 8 are not.
"""
import numpy as np
import tracer115 as T
import geomdesc115 as GD

NB = T.NB
ELL, M0, M1 = 0.0625, 0.002, 0.01
KF = 16


def support(ctx):
    """(w [N, 16] activation per 4 x 4 cell, pids of the patches with any active cell, m [N, 16])."""
    m = GD.prox_other(ctx, 4, ells=(ELL,))[..., 0]
    w = np.clip((m - M0) / (M1 - M0), 0.0, 1.0).astype(np.float32)
    pids = np.flatnonzero((w > 0).any(1))
    return w, pids, m


def blend_patch(T4p, T16p, w4p):
    """[256, NB, 3] blended 16 x 16 table of one patch: (1 - w) up(T4) + w T16 with w broadcast over each cell's 4 x 4 children."""
    up = T.upsample(T4p[None], 4, 16)[0]
    w16 = T.upsample(np.asarray(w4p, np.float32)[None, :, None, None], 4, 16)[0]       # [256, 1, 1]
    return ((1.0 - w16) * up + w16 * T16p).astype(np.float32)


def rep(T4, T16, w, pids):
    """'pc' readout representation of CAR-16: 4 x 4 on untouched patches, blended 16 x 16 on the support patches.
    T4 [N,16,NB,3]; T16 [len(pids),256,NB,3] (rows aligned with pids); w [N,16]."""
    fine = np.stack([blend_patch(T4[p], T16[i], w[p]) for i, p in enumerate(pids)]) if len(pids) else np.zeros((0, 256, NB, 3), np.float32)
    return T.pc_hybrid(T4, 4, fine, 16, pids)


def child_index_maps():
    """children of 4 x 4 cell c (u + 4 v) inside the 16 x 16 grid: [16, 16] fine sub indices."""
    out = np.zeros((16, 16), np.int64)
    for c in range(16):
        cu, cv = c % 4, c // 4
        for k in range(16):
            ku, kv = k % 4, k // 4
            out[c, k] = (4 * cu + ku) + 16 * (4 * cv + kv)
    return out
