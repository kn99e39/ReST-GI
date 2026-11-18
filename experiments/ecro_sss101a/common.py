"""Frozen WL101A inputs and array helpers."""
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol" / "sss101a.json"
P = json.loads(PROTOCOL.read_text())
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
