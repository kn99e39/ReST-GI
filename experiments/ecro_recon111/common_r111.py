"""WL111 shared code: protocol, paths, seeds, ROIs, image metrics and statistics. numpy only (no GPU import).

Paths follow the LabServer70 layout (RECON111_ROOT, default ~/rest_sss101a/results/ecro_recon111):
  ROOT/inputs   read-only links to the frozen inputs already on LabServer70 (WL108 inputs tree, WL107 capture) and copies of
                the archived WL101B static frames and the WL110 full-depth physical references
  ROOT/<stage>  outputs of this batch
"""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PROTOCOL = HERE / "protocol" / "recon111.json"
P = json.loads(PROTOCOL.read_text(encoding="utf-8"))
LUMA = np.array((0.2126, 0.7152, 0.0722))
RES = 512

ROOT = Path(os.environ.get("RECON111_ROOT", str(Path.home() / "rest_sss101a" / "results" / "ecro_recon111")))
INPUTS = ROOT / "inputs"
WL101 = INPUTS / "wl101"
D105 = INPUTS / "results" / "ecro_denoise105" / "data"
M105 = INPUTS / "results" / "ecro_denoise105" / "models"
E105 = INPUTS / "results" / "ecro_denoise105" / "eval"
REF107 = INPUTS / "results" / "ecro_denoise107" / "reference"
CAP107 = INPUTS / "capture"
RAW105 = INPUTS / "evaluation" / "105" / "raw_manifest105.json"
PHYS110 = INPUTS / "phys110"
WL101B = INPUTS / "wl101b_primary"


# ------------------------------------------------------------------------------------------------ io
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _clean(x):
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
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    return x


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_clean(data), indent=1, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


# ------------------------------------------------------------------------------------------------ seeds
MASK64 = (1 << 64) - 1


def splitmix64(x):
    z = (int(x) + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def launch_seed(seed, k=0):
    """64-bit main-stream seed value for the unchanged frame program (WL109 rule: splitmix64((seed << 16) + launch);
    the historical seed * 1000003 + k convention correlates consecutive seeds and is used only for archived replays)."""
    return splitmix64((int(seed) << 16) + int(k))


SMOKE_OFFSET = P["seeds"]["smoke_offset"]


def seed(family, i0, i1, i2, smoke=False):
    """Structured evidence seed: base(family) + i0 * 100000 + i1 * 1000 + i2 (index_0 < 10, index_1 < 100,
    index_2 < 1000). Families are 1,000,000 apart, so seeds of different families never coincide (tests_r111)."""
    S = P["seeds"]
    s0, s1, s2 = S["strides"]
    assert 0 <= i0 < S["family_width"] // s0 and 0 <= i1 < s0 // s1 and 0 <= i2 < s1 // s2, (family, i0, i1, i2)
    return S["families"][family] + i0 * s0 + i1 * s1 + i2 * s2 + (SMOKE_OFFSET if smoke else 0)


# ------------------------------------------------------------------------------------------------ ROIs (WL105 frozen)
def erode(m, k):
    p = np.pad(m, k, constant_values=False)
    H, W = m.shape
    return np.logical_and.reduce([p[y:y + H, x:x + W] for y in range(2 * k + 1) for x in range(2 * k + 1)])


def rois(aux):
    """WL105 analyze105.rois (same statements) with the frozen contact disk removed from every ROI and returned as
    'contact' (descriptive only), as common107 / common109. aux: cover (9,H,W), sid, ok, p (H,W,3), valid_frac."""
    cover, sid, ok, p, vf = aux["cover"], aux["sid"], aux["ok"], aux["p"], aux["valid_frac"]
    r = np.hypot(p[..., 0] - 0.9, p[..., 1] - 0.4)
    contact = ok & (sid == 0) & (r < 0.25)
    sph = cover[6]
    m = {"primary": (vf > 0) & ~contact,
         "sss_interior": erode(sph >= 1.0, 2), "silhouette": (sph > 0) & (sph < 1),
         "material_boundary": (cover.max(0) < 1) & (sph == 0) & (vf >= 1) & ~contact,
         "control": erode(cover[2] >= 1.0, 4)}
    out = {k: v & ~contact for k, v in m.items()}
    out["contact"] = contact
    out["full"] = vf > 0
    return out


DECISION_ROIS = ("silhouette", "sss_interior", "material_boundary")
BIAS_ROIS = ("primary", "sss_interior", "silhouette", "material_boundary", "control")


# ------------------------------------------------------------------------------------------------ metrics
def luma(x):
    return np.tensordot(LUMA, x, 1)


def sse_parts(y, ref, var_ref, mask):
    """-> (raw MSE over ROI x RGB, mean reference variance over ROI x RGB, mean ref^2). y, ref: (3,H,W); var_ref (3,H,W)
    is the variance of the reference MEAN (0 if unknown)."""
    d = (np.asarray(y, np.float64) - ref)[:, mask]
    vr = 0.0 if var_ref is None else float(np.asarray(var_ref)[:, mask].mean())
    return float((d ** 2).mean()), vr, float((ref[:, mask] ** 2).mean())


def rel_rmse(y, ref, var_ref, mask):
    """Debiased relative HDR RMSE: sqrt(max(MSE - mean Var(ref), 0) / mean ref^2) (WL109 definition)."""
    mse, vr, r2 = sse_parts(y, ref, var_ref, mask)
    return math.sqrt(max(mse - vr, 0.0) / r2)


def signed_bias(y, ref, mask):
    """Signed luminance bias relative to the ROI reference mean luminance."""
    ly, lr = luma(np.asarray(y, np.float64)), luma(ref)
    return float((ly - lr)[mask].mean() / lr[mask].mean())


def grad_ratio(y, t, m):
    """WL105 gradient-magnitude ratio |grad y| / |grad t| over adjacent pairs touching the ROI (luminance)."""
    yl, tl = luma(np.asarray(y, np.float64)), luma(t)
    num = den = 0.0
    for ax in (0, 1):
        dy, dt = np.diff(yl, axis=ax), np.diff(tl, axis=ax)
        mm = (m[:-1, :] | m[1:, :]) if ax == 0 else (m[:, :-1] | m[:, 1:])
        num += float((dy[mm] ** 2).sum())
        den += float((dt[mm] ** 2).sum())
    return math.sqrt(num / den) if den > 0 else float("nan")


def grad_error(y, t, m):
    """Relative RMSE of the luminance gradient over pairs touching the ROI (boundary-gradient preservation)."""
    yl, tl = luma(np.asarray(y, np.float64)), luma(t)
    num = den = 0.0
    for ax in (0, 1):
        dy, dt = np.diff(yl, axis=ax), np.diff(tl, axis=ax)
        mm = (m[:-1, :] | m[1:, :]) if ax == 0 else (m[:, :-1] | m[:, 1:])
        num += float(((dy - dt)[mm] ** 2).sum())
        den += float((dt[mm] ** 2).sum())
    return math.sqrt(num / den) if den > 0 else float("nan")


def frame_metrics(y, ref, var_ref, masks):
    """Per-frame metric record for an output y (3,H,W) against a reference mean (3,H,W) with mean variance var_ref."""
    out = {}
    for name, m in masks.items():
        if not m.any():
            continue
        mse, vr, r2 = sse_parts(y, ref, var_ref, m)
        out[name] = {"mse": mse, "var_ref": vr, "ref2": r2, "rel_rmse": math.sqrt(max(mse - vr, 0.0) / r2),
                     "bias": signed_bias(y, ref, m), "n": int(m.sum())}
    for name in ("silhouette", "material_boundary"):
        if masks.get(name) is not None and masks[name].any():
            out[name]["grad_ratio"] = grad_ratio(y, ref, masks[name])
            out[name]["grad_error"] = grad_error(y, ref, masks[name])
    return out


# ------------------------------------------------------------------------------------------------ statistics
Z95 = 1.959963984540054


def gm(x):
    x = np.asarray(x, np.float64)
    return float(np.exp(np.log(x).mean()))


def view_rmse(mse, var_ref, ref2):
    """mse (R,) raw per replicate, var_ref, ref2 scalars -> debiased relative RMSE of the view (pooled replicates)."""
    return math.sqrt(max(float(np.mean(mse)) - var_ref, 1e-30) / ref2)


def two_level_boot(per_view, stat, draws, seed_):
    """per_view: list over views of tuples of per-replicate arrays (aligned replicates for paired arms). stat(list) -> float.
    Resample views, then replicates within each resampled view (paired: same replicate indices for every array of a view).
    Returns the bootstrap sample array."""
    rs = np.random.default_rng(seed_)
    V = len(per_view)
    out = np.empty(draws)
    for b in range(draws):
        vs = rs.integers(0, V, V)
        sample = []
        for v in vs:
            arrays = per_view[v]
            n = len(arrays[0])
            idx = rs.integers(0, n, n)
            sample.append(tuple(np.asarray(a)[idx] for a in arrays))
        out[b] = stat(sample)
    return out


def ci95(samples):
    s = np.asarray(samples, np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
