"""WL55 shared helpers (protocol fnto_k55): paths, JSON output, an INDEPENDENT group-statistics implementation (written from the
protocol definitions; WL54's c54_stageB.group_stats is deliberately not used, so the Stage A replay is an independent check), and
Cycles port extraction with the declared (WL53 interior) and amended (WL54 centre-ray + coverage) comparison units.

Definitions (protocol "definitions"):
  bias B = sum(pred - ref) / sum(ref) over the group's selected ports, per channel; SE_ref from Cycles seeds (std / sqrt(n) per port,
  summed in quadrature over the group, divided by the group signal); a group is ruled iff >= 5 ports and >= 2 % of the camera signal;
  rule_3pct: |B| <= 0.03 + 3 SE.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phys_ref53")); sys.path.insert(0, str(HERE.parent / "fnto_c54"))

import p53_fixtures as FX  # noqa: E402

S50, C50 = math.sin(math.radians(50)), math.cos(math.radians(50))
FX.DIRS.setdefault("obotx", (-S50, 0.0, -C50))           # WL55 extra oblique camera (-x side, from below); runtime only, file unchanged
CAM5 = ("top", "bot", "otop", "obot", "obotx")
FACE = {0: "-x", 1: "+x", 2: "-y", 3: "+y", 4: "bot", 5: "top"}
MINPIX, MINPORTS, MINSHARE, COV, TOL = 30, 5, 0.02, 0.9, 0.03
V0_OFF = ((0.5, 0.5), (0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))   # WL53 interior unit
V1_OFF = ((0.5, 0.5),)                                                # WL54 centre-ray unit


def jclean(o):
    if isinstance(o, dict):
        return {str(k): jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jclean(o.tolist())
    return o


def dump(obj, path):
    json.dump(jclean(obj), open(path, "w"), indent=1, allow_nan=False)


def dirkey(d):
    for k, v in FX.DIRS.items():
        if np.allclose(v, d, atol=1e-6):
            return k
    raise KeyError(d)


# ------------------------------------------------------------------ independent group statistics


def group_bias(pred, ref, sel, labels, se_port=None, share_mask=None):
    """pred (P,) or (P, 3); ref (P,) or (P, 3) reference mean; sel bool (P,) selected ports; labels (P,) str.
    se_port: per-port SE of the reference mean (same shape as ref) or None. share_mask: ports that define the camera signal
    total for the 2 % ruling (default sel). Returns {group: dict(n, share[c], bias[c], se[c], ruled, passed[c], p50, p90)}."""
    pred = np.asarray(pred, float); ref = np.asarray(ref, float)
    if pred.ndim == 1:
        pred, ref = pred[:, None], ref[:, None]
        se_port = None if se_port is None else np.asarray(se_port, float)[:, None]
    sm = sel if share_mask is None else (share_mask & sel)
    tot = ref[sm].sum(0)
    out = {}
    for g in np.unique(labels[sel]):
        m = sel & (labels == g); s = ref[m].sum(0)
        if np.all(s <= 0):
            continue
        b = (pred[m] - ref[m]).sum(0) / np.where(s > 0, s, np.nan)
        se = np.sqrt((se_port[m] ** 2).sum(0)) / np.where(s > 0, s, np.nan) if se_port is not None else np.zeros_like(b)
        share = s / np.where(tot > 0, tot, np.nan)
        inside = share_mask is None or bool(share_mask[m].all())
        ruled = [bool(inside and m.sum() >= MINPORTS and sh >= MINSHARE) for sh in share]          # per channel, as WL53 p53_report.compare
        rel = np.abs(pred[m] - ref[m]) / np.maximum(np.abs(ref[m]), 1e-30)
        out[str(g)] = dict(n=int(m.sum()), share=share.tolist(), bias=b.tolist(), se=se.tolist(), ruled=ruled,
                           passed=[bool(abs(bb) <= TOL + 3 * ss) if r else None for bb, ss, r in zip(b, se, ruled)],
                           p50=np.percentile(rel, 50, axis=0).tolist(), p90=np.percentile(rel, 90, axis=0).tolist())
    return out


def labels_of(sd):
    lab = []
    for n, k, m, V in sd["comps"]:
        lab += [f"{n}:{FACE[int(f)]}" if k == "medium" else n for f in m["port_face"]]
    return np.array(lab)


# ------------------------------------------------------------------ Cycles port extraction (GPU intersector)


def classify_sceneset(sd):
    """Coarse SceneSet for pixel classification (geometry only; emitters count as black)."""
    import p53_compose as CP
    comps = [CP.Comp(n, "black" if k in ("emitter", "black") else k, m, V, rho=FX.FLOOR_RHO if k == "lambert" else None) for n, k, m, V in sd["comps"]]
    return CP.SceneSet(comps, FX.DOUT, FX.DIN)


def expected_pixels(ss, geo):
    import torch
    c = -np.asarray(geo["fwd"], float); c /= np.linalg.norm(c); px = geo["width"] / geo["res"]
    ct = torch.tensor(c, dtype=torch.float32, device="cuda")
    proj = torch.zeros(ss.Ptot, device="cuda").index_add_(0, ss.tri_port, ss.area * (ss.n * ct).sum(1).clamp(min=0))
    return proj.cpu().numpy() / px ** 2


def cycles_ports(ss, geo, imgs):
    """imgs: list of (res, res, 3) arrays (seeds). Returns dict(unit -> dict(mean (P,3), se (P,3), cnt (P,), sel (P,))) for the
    'amended' (centre ray, >= 30 px and >= 90 % expected coverage) and 'declared' (interior, >= 30 px) units, plus cov."""
    import p53_compose as CP
    import c54_common as CC
    exp = expected_pixels(ss, geo); out = {}
    for unit, off in (("amended", V1_OFF), ("declared", V0_OFF)):
        pid = CP.classify(ss, CC.blender_ortho_rays(geo, off))
        ms = [CP.port_means(pid, np.asarray(im).reshape(-1, 3), ss.Ptot) for im in imgs]
        cnt = ms[0][1]; arr = np.stack([m[0] for m in ms])
        se = arr.std(0, ddof=1) / math.sqrt(len(arr)) if len(arr) > 1 else np.zeros_like(arr[0])
        cov = cnt / np.maximum(exp, 1e-9)
        sel = (cnt >= MINPIX) & ((cov >= COV) if unit == "amended" else True)
        out[unit] = dict(mean=arr.mean(0), se=se, cnt=cnt, sel=sel, cov=cov, seeds=arr)
    return out
