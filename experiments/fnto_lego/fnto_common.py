"""FNTO Batch 6 shared definitions (numpy only; importable from bpy, the Mitsuba/torch venv and the tests).

Worklog 41, Architecture Batch 6: Factorized Neural Transport Operator v0.1 (docs/FNTO_ARCHITECTURE_CONTRACT_V0_1.md).
Scene: the worklog-37 official Lego (domain A) + red box (domain B); T0 / T1 (+0.44 m box translation) unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
B4 = ROOT / "results/rna_selective_reconstruction_lego"
SPT = ROOT / "results/spt_lego"
OUT = ROOT / "results/fnto_lego"
MESH_A = ROOT / "results/rna_surface_ownership_lego/mesh/lego_mesh.npz"
BOX_T0 = B4 / "scene/B_box_T0.npz"
SCENE = json.loads((B4 / "scene/scene.json").read_text())
DELTA = np.asarray(SCENE["delta"], np.float64)          # T0 -> T1 translation of the box
B_UNIT = 16                                             # B4 owner label of the box
OBJ_A, OBJ_B = 0, 1
MATERIALS = ["Yellow", "Gray", "transparent", "Black", "Red_Glass", "Light_brown", "Sand", "Brown", "B_Red"]
MAT_BOX = 8                                             # material id of the box (domain B)
E_SUN = 5.0                                             # the generator's sun: irradiance 5 W/m^2 normal to the sun, angle 0

# ---- ports (anchors) and gathers ----------------------------------------------------------------------------------------
H_VOX = 0.012           # anchor cluster size [m]
K_GATHER = 128          # free-space gather directions per anchor (cosine distributed, per-anchor random rotation)
EPS = 1e-3              # ray-origin offset along the anchor normal [m]
N_LOCAL = 32            # neighbours of the local intrinsic kernel
R_LOCAL = 0.05          # radius of the local intrinsic kernel [m]
N_BINS_H = 19           # hemispherical incident bins of the band kernel: cap theta < 20 deg, 6 sectors 20-50, 12 sectors 50-90
H_MID = 0.04            # coarse source ports of the mid-range intrinsic kernel [m]
N_MID = 64              # coarse neighbours of the mid-range kernel
R_MID = 0.30            # radius of the mid-range kernel [m]


def offsets(state: str, box_offset=None):
    """(offset A, offset B) of a geometry state relative to the canonical T0 placement (rigid translation only)."""
    if state == "T0":
        return np.zeros(3), np.zeros(3)
    if state == "T1":
        return np.zeros(3), DELTA.copy()
    if state == "custom":
        return np.zeros(3), np.asarray(box_offset, np.float64)
    raise ValueError(state)


def cosine_dirs_local(k=K_GATHER):
    """k stratified cosine-weighted directions in the local frame (z = normal): Fibonacci spiral on the unit disk."""
    i = np.arange(k) + 0.5
    r = np.sqrt(i / k)
    phi = i * np.pi * (3 - np.sqrt(5))
    x, y = r * np.cos(phi), r * np.sin(phi)
    return np.stack([x, y, np.sqrt(np.maximum(0, 1 - x * x - y * y))], -1)


def frames(n, rot):
    """Orthonormal tangent frames for normals n [N,3] with a per-anchor rotation angle rot [N] -> (t, b, n)."""
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    a = np.where(np.abs(n[:, 2:3]) < 0.9, np.array([[0.0, 0, 1]]), np.array([[1.0, 0, 0]]))
    t = np.cross(a, n); t /= np.linalg.norm(t, axis=1, keepdims=True)
    b = np.cross(n, t)
    c, s = np.cos(rot)[:, None], np.sin(rot)[:, None]
    return c * t + s * b, -s * t + c * b, n


def hemi_bin_local(d):
    """Local direction(s) [..., 3] (z = normal) -> hemispherical bin in [0, 19): 0 = cap theta < 20 deg, 1..6 = azimuth
    sectors of the 20-50 deg ring, 7..18 = azimuth sectors of the 50-90 deg ring (below the horizon: clamped into it)."""
    d = np.asarray(d, np.float64)
    z = d[..., 2]
    az = np.mod(np.arctan2(d[..., 1], d[..., 0]), 2 * np.pi)
    s6 = 1 + np.minimum((az / (2 * np.pi) * 6).astype(np.int64), 5)
    s12 = 7 + np.minimum((az / (2 * np.pi) * 12).astype(np.int64), 11)
    return np.where(z > np.cos(np.radians(20)), 0, np.where(z > np.cos(np.radians(50)), s6, s12))


def sh9(d):
    """Real spherical harmonics l <= 2 of unit directions [..., 3] -> [..., 9]."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    return np.stack([0.282095 * np.ones_like(x), 0.488603 * y, 0.488603 * z, 0.488603 * x,
                     1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z,
                     0.546274 * (x * x - y * y)], -1)


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def ldr3(x):
    """The worklog-37 display transform."""
    x = np.clip(x, 0, None)
    return (x / (1 + x)) ** (1 / 2.2)
