"""WL54 shared helpers (protocol fnto_c54). Copies of two WL53 helpers (p53_render.sceneset_of, blender_ortho_rays) so that the
oracle-side code does not import the renderer module (which asserts the llvm variant); WL53 files are not modified."""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phys_ref53"))
import p53_compose as CP  # noqa: E402
import p53_fixtures as FX  # noqa: E402


def jclean(o):
    """JSON-safe copy: nonfinite floats -> None (protocol: no NaN / Infinity in machine-readable files)."""
    if isinstance(o, dict):
        return {str(k): jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jclean(o.tolist())
    return o


def dump(obj, path):
    json.dump(jclean(obj), open(path, "w"), indent=1, allow_nan=False)


def sceneset_of(sd):
    comps = [CP.Comp(n, k, m, V, rho=FX.FLOOR_RHO if k == "lambert" else None) for n, k, m, V in sd["comps"]]
    return CP.SceneSet(comps, FX.DOUT, FX.DIN)


def blender_ortho_rays(geo, offsets=((0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9))):
    import torch
    res, w = geo["res"], geo["width"]; o0 = np.asarray(geo["origin"]); R = np.asarray(geo["right"]); U = np.asarray(geo["up"]); Fw = np.asarray(geo["fwd"])
    rr, cc = np.meshgrid(np.arange(res), np.arange(res), indexing="ij"); out = []
    for ox, oy in offsets:
        x = (cc.ravel() + ox) / res * w - w / 2; y = (rr.ravel() + oy) / res * w - w / 2
        o = o0[None] + x[:, None] * R[None] + y[:, None] * U[None]
        out.append((torch.tensor(o, dtype=torch.float32, device="cuda"), torch.tensor(np.broadcast_to(Fw, o.shape).copy(), dtype=torch.float32, device="cuda")))
    return out
