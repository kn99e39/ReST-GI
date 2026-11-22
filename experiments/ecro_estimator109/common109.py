"""WL109 shared paths, protocol, seeds and numerical helpers (numpy only; no GPU imports)."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for d in ("ecro_denoise105", "ecro_denoise107", "ecro_denoise108"):
    sys.path.insert(0, str(EXP / d))

PROTOCOL = HERE / "protocol109.json"
P = json.loads(PROTOCOL.read_text())
LUMA = np.array((0.2126, 0.7152, 0.0722))
EVAL_VIEWS = tuple(P["views"]["eval"])
DEV_VIEWS = tuple(P["views"]["dev"])
GRID = tuple(P["r0_grid_spp"])
NOMINAL = tuple(P["nominal_spp"])
DECISION_ROIS = tuple(P["rois"]["bias"])


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def _clean(x):
    """NaN / inf -> None (JSON null); numpy scalars -> Python."""
    import math
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_clean(data), indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def apply_smoke():
    """Code-path smoke sizes (non-evidence seed family); never used for evidence."""
    P["final"].update(replicates=4, escalated_replicates=8, timing_rounds=3, timing_warmup=1)
    P["development"].update(replicates=4, timing_rounds=3, timing_warmup=1, bootstrap_draws=10)
    s = P["synthetic"]
    s["statistical"].update(replicates=8, bootstrap_draws=20, Q_groups=4)
    s["independence"].update(replicates=8, bootstrap_draws=20)
    s["replicate_independence"].update(replicates=8)
    s["quadrature"].update(pixels=128, frames=4, grid=32)
    s["toy"].update(K=300)
    P["analysis"].update(bootstrap_draws=200, component_bootstrap_draws=5, Q_groups=4)
    P["dynamic"].update(reference_blocks=2, reference_block_spp=64, replicates=4)


MASK64 = (1 << 64) - 1


def splitmix64(x):
    z = (int(x) + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def launch_seed(seed, k=0):
    """64-bit main-stream seed value passed to the unchanged rt101 frame program for evidence frames. The historical
    seed * 1000003 + k convention gives consecutive seeds correlated pcg_046 streams (WL109 development diagnostic), so
    evidence frames use a splitmix64 hash; archived-frame replays keep the historical convention."""
    return splitmix64((int(seed) << 16) + int(k))


def smoke_offset(smoke):
    return P["seeds"]["smoke_offset"] if smoke else 0


def seed_paired(view_index, spp, r, smoke=False):
    s = P["seeds"]
    return s["paired"] + 10000 * view_index + 100 * NOMINAL.index(spp) + r + smoke_offset(smoke)


def seed_grid(view_index, spp, r, smoke=False):
    s = P["seeds"]
    return s["grid"] + 10000 * view_index + 100 * GRID.index(spp) + r + smoke_offset(smoke)


def seed_dev(dev_index, m_index, r, smoke=False):
    s = P["seeds"]
    return s["dev"] + 10000 * dev_index + 100 * m_index + r + smoke_offset(smoke)


def seed_synth(table_index, config_index, r, smoke=False):
    s = P["seeds"]
    assert config_index < 1000 and r < 100
    return s["synthetic"] + 100000 * table_index + 100 * config_index + r + smoke_offset(smoke)


def seed_timing(view_index, slot, i, smoke=False):
    s = P["seeds"]
    assert view_index < 20 and slot < 100 and i < 100
    return s["timing"] + 10000 * view_index + 100 * slot + i + smoke_offset(smoke)


def seed_dynamic(slot, r, smoke=False):
    s = P["seeds"]
    return s["dynamic"] + 100 * slot + r + smoke_offset(smoke)


def seed_dynamic_reference(block, smoke=False):
    return P["seeds"]["dynamic_reference"] + block + smoke_offset(smoke)


# ------------------------------------------------------------------------------------------------ ROIs
def rois(aux):
    """WL105 geometry ROIs with the frozen contact exclusion (common107.rois) + the contact disk (descriptive only)."""
    from analyze105 import rois as frozen_rois
    masks = frozen_rois(aux)
    contact = masks.pop("contact_excluded")
    out = {name: value & ~contact for name, value in masks.items()}
    out["contact"] = contact
    return out


def distance_bands(aux, width=7):
    """Signed pixel distance to the sphere silhouette band (WL107 profile): 0 = mixed coverage, +k inside, -k outside."""
    from scipy.ndimage import distance_transform_edt
    sph = aux["cover"][6]
    mixed = (sph > 0) & (sph < 1)
    d = distance_transform_edt(~mixed)
    bands = {}
    contact = rois(aux)["contact"]
    valid = (aux["valid_frac"] > 0) & ~contact
    bands["0"] = mixed & valid
    for k in range(1, width + 1):
        ring = (np.rint(d) == k) & valid
        bands[f"+{k}"] = ring & (sph >= 1)
        bands[f"-{k}"] = ring & (sph <= 0)
    return bands


# ------------------------------------------------------------------------------------------------ statistics
def luma(x):
    return np.tensordot(LUMA, x, 1)


def normal_ci(point, se, z=1.959963984540054):
    return [float(point - z * se), float(point + z * se)]


def gm(x):
    x = np.asarray(x, np.float64)
    return float(np.exp(np.log(x).mean()))
