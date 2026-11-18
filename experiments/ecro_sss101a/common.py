"""Frozen WL101A inputs and array helpers."""
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol" / "sss101a.json"
P = json.loads(PROTOCOL.read_text())
EXECUTION = HERE / "protocol" / "execution63.json"
EXECUTION_CONFIG = json.loads(EXECUTION.read_text())
RES = P["camera"]["res"]
EVENTS = ("e1", "e2", "hl")


def sha(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sample_seed(spp, replicate):
    return 10160000 + 100 * spp + replicate


def image(path, high="hl"):
    with np.load(path) as archive:
        result = np.stack([archive[k] for k in ("e1", "e2", high)]).transpose(1, 2, 3, 0)
    assert result.shape == (RES, RES, 3, 3)
    assert np.isfinite(result).all() and np.min(result) >= 0
    return result


def components(events):
    low = events[..., 0] + events[..., 1]
    high = events[..., 2]
    return np.stack((low + high, low, high, events[..., 0], events[..., 1]), axis=-1)


def rel_l1(a, b):
    return float(np.abs(a.astype(np.float64) - b).sum() / np.abs(b.astype(np.float64)).sum())


def masks(path):
    with np.load(path) as gbuffer:
        valid = gbuffer["ok"]
        sid = gbuffer["sid"]
        pos = gbuffer["p"]
    assert valid.shape == sid.shape == (RES, RES) and pos.shape == (RES, RES, 3)
    assert np.isfinite(pos[valid]).all()
    geom = P["sphere"]
    sphere = valid & (sid == geom["sphere_sid"])
    floor = valid & (sid == geom["floor_sid"])
    radius = geom["radius"]
    center = np.asarray(geom["center"])
    assert np.max(np.abs(pos[floor, 2])) < 1e-5
    assert np.max(np.abs(np.linalg.norm(pos[sphere] - center, axis=-1) - radius)) < 1e-5

    pad = np.pad(sphere, 4, constant_values=False)
    interior = np.logical_and.reduce([pad[y:y + RES, x:x + RES]
                                      for y in range(9) for x in range(9)])
    boundary = sphere & ~interior
    light = P["light"][0]["pos"]
    to_light = light - pos
    to_center = center - pos
    t = np.sum(to_center * to_light, axis=-1) / np.sum(to_light ** 2, axis=-1)
    closest = to_center - t[..., None] * to_light
    shadow = (t > 0) & (t < 1) & (np.sum(closest ** 2, axis=-1) <= radius ** 2)
    contact = np.sum(to_center[..., :2] ** 2, axis=-1) <= radius ** 2
    result = {"full": np.ones_like(valid), "sphere": sphere, "sphere_interior": interior,
              "sphere_boundary": boundary, "floor": floor, "contact_shadow": floor & (shadow | contact)}
    assert not np.any(sphere & floor)
    assert not np.any(interior & boundary)
    assert np.array_equal(interior | boundary, sphere)
    assert not np.any(result["contact_shadow"] & ~floor)
    assert all(mask.dtype == bool and np.any(mask) for mask in result.values())
    return result
