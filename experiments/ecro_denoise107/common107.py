"""Shared WL107 provenance and frozen WL105 mappings."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
WL105 = HERE.parent / "ecro_denoise105"
sys.path.insert(0, str(WL105))
PROTOCOL = HERE / "protocol/denoise107.json"
P = json.loads(PROTOCOL.read_text())
BASE = P["provenance"]["base"]
VIEWS = ("V0", "H1", "H2", "H3", "H4", "H5")
SPPS = (4, 16, 32)
BINS = (0, 1/64, 1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, 63/64, 1)
LUMA = np.array((0.2126, 0.7152, 0.0722))
LAMBDAS = (0.03125, 0.125, 0.5, 1, 2, 8, 32)


def sha(path):
    with path.open("rb") as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def low_high(comp):
    return np.stack((comp[..., :3, :, :] + comp[..., 3:6, :, :], comp[..., 6:9, :, :]), axis=-4)


def total(comp):
    return comp[..., :3, :, :] + comp[..., 3:6, :, :] + comp[..., 6:9, :, :]


def rois(aux):
    from analyze105 import rois as frozen_rois

    masks = frozen_rois(aux)
    contact = masks.pop("contact_excluded")
    return {name: value & ~contact for name, value in masks.items()}


def coverage_masks(cover, eligible):
    return [eligible & (cover >= lo) & (cover < hi) for lo, hi in zip(BINS, BINS[1:])]


def panel(aux):
    masks = coverage_masks(aux["cover"][6], rois(aux)["silhouette"])
    return np.concatenate([i[np.linspace(0, len(i) - 1, min(len(i), 32), dtype=int)]
                           for m in masks if len(i := np.flatnonzero(m))])


def stencil(idx, height, width):
    rows, cols = np.divmod(idx, width)
    def offsets(p, n):
        p = (p[:, None] + np.arange(-7, 8)) % (2 * n - 2)
        return np.minimum(p, 2 * n - 2 - p)
    return (offsets(rows, height)[:, :, None] * width + offsets(cols, width)[:, None, :]).reshape(-1, 225)


def take_stencil(values, indices):
    """(..., H, W) -> (..., pixels, taps), in frozen unfold order."""
    return values.reshape(*values.shape[:-2], -1)[..., indices]
