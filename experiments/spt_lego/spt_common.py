"""SPT-1 shared definitions (numpy only): paths, light-direction bins, gather directions, real SH (l <= 2).

Worklog 39, Architecture Batch 5 (docs/Architecture.md §9a). Imported by the relation builder (Windows, Mitsuba), the
pixel-anchor mapper, the model/trainer (torch) and the synthetic contract tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
B4 = ROOT / "results/rna_selective_reconstruction_lego"
OUT = ROOT / "results/spt_lego"
SCENE = json.loads((B4 / "scene/scene.json").read_text())
DELTA = np.asarray(SCENE["delta"], np.float64)
B_UNIT = 16                       # B4 owner label of the box
OBJ_A, OBJ_B = 0, 1
K_GATHER = 128                    # gather directions per anchor
N_RING, N_AZ = 16, 32             # light bins on the upper hemisphere: uniform in z, uniform in azimuth
N_BINS = N_RING * N_AZ
EPS = 2e-3                        # ray-origin offset along the anchor normal [m]


def offsets(state: str):
    """(offset A, offset B) of a state relative to the canonical T0 placement (rigid translation only)."""
    if state == "T0":
        return np.zeros(3), np.zeros(3)
    if state == "T1":
        return np.zeros(3), DELTA.copy()
    raise ValueError(state)


def light_bin(d):
    """Upper-hemisphere light direction(s) [..., 3] -> bin index; z <= 0 is clamped into the lowest ring."""
    d = np.asarray(d, np.float64)
    z = np.clip(d[..., 2], 0.0, 1.0 - 1e-9)
    ring = np.minimum((z * N_RING).astype(np.int64), N_RING - 1)
    az = np.mod(np.arctan2(d[..., 1], d[..., 0]), 2 * np.pi)
    a = np.minimum((az / (2 * np.pi) * N_AZ).astype(np.int64), N_AZ - 1)
    return ring * N_AZ + a


def bin_centres():
    r, a = np.divmod(np.arange(N_BINS), N_AZ)
    z = (r + 0.5) / N_RING
    phi = (a + 0.5) / N_AZ * 2 * np.pi
    s = np.sqrt(1 - z * z)
    return np.stack([s * np.cos(phi), s * np.sin(phi), z], -1)


def cosine_dirs_local(k=K_GATHER):
    """k stratified cosine-weighted directions in the local frame (z = normal): Fibonacci spiral on the unit disk."""
    i = np.arange(k) + 0.5
    r = np.sqrt(i / k)
    phi = i * np.pi * (3 - np.sqrt(5))
    x, y = r * np.cos(phi), r * np.sin(phi)
    return np.stack([x, y, np.sqrt(np.maximum(0, 1 - x * x - y * y))], -1)


def frames(n, rot):
    """Orthonormal tangent frames for normals n [N,3] with an extra per-anchor rotation angle rot [N]."""
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    a = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([[0.0, 0, 1]]), np.array([[1.0, 0, 0]]))
    t = np.cross(a, n); t /= np.linalg.norm(t, axis=1, keepdims=True)
    b = np.cross(n, t)
    c, s = np.cos(rot)[:, None], np.sin(rot)[:, None]
    return c * t + s * b, -s * t + c * b, n


def sh9(d):
    """Real spherical harmonics l <= 2 of unit directions [..., 3] -> [..., 9]."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    return np.stack([0.282095 * np.ones_like(x), 0.488603 * y, 0.488603 * z, 0.488603 * x,
                     1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z,
                     0.546274 * (x * x - y * y)], -1)


def sha16(p):
    import hashlib

    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
